import os
import sys

repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

import numpy as np
import torch
from torch import nn, Tensor
import tqdm.auto as tqdm
from sklearn.neighbors import BallTree
from sentence_transformers import SentenceTransformer
import ot

from models.prototypes.kmeans import KMeansPlusPlus, KMeansClassifier
from utils.pythonic.numeric_utils import set_difference, set_intersection
from models.diversity.similarity import CosineDissimilarity, CDist
from models.influence.fisher import FisherScorer


class TemperatureScheduler:
    """Temperature scheduler.

    Args:
    + `start_temp`: Initial temperature.
    + `stop_temp`: Final temperature.
    + `strategy`: 'cosine', 'exponential', 'linear', or 'inv_sqrt'. Defaults to `'cosine'`.
    """

    def __init__(self, start_temp: float = 1, stop_temp: float = 1e-4, strategy="cosine"):
        if strategy not in ["cosine", "exponential", "linear", "inv_sqrt"]:
            raise ValueError(f"Strategy {strategy} not supported.")

        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.strategy = strategy

    def __call__(self, step: int, num_steps: int, progress: float | None = None) -> float:
        step = np.clip(a=step, a_min=0, a_max=num_steps)
        if progress is None:
            progress = step / num_steps

        if self.strategy == "cosine":
            # Starts slow, drops fast in middle, settles slow
            # Popular in Diffusion and Transformer learning rates
            p2p = self.start_temp - self.stop_temp
            temp = self.stop_temp + p2p * 0.5 * (1 + np.cos(np.pi * progress))
            return temp.item()

        elif self.strategy == "exponential":
            return self.start_temp * (self.stop_temp / self.start_temp) ** progress

        elif self.strategy == "linear":
            return self.start_temp - progress * (self.start_temp - self.stop_temp)

        elif self.strategy == "inv_sqrt":
            # Sharp initial drop with a very long tail for refinement
            # Inspired by Transformer schedules
            # T = start / sqrt(step + 1), scaled to attempt hitting stop_temp
            # Note: This is an approximation to fit the start/stop constraints
            return max(self.stop_temp, self.start_temp / np.sqrt(step + 1))

        else:
            raise ValueError(f"Strategy {self.strategy} not supported.")


class DiscreteOTDistiller:
    """DiscreteOTDistiller.

    Args:
    + `source`.
    + `pool`.
    + `source_weights`.
    + `init_strat`. Defaults to `"kmeans"`.
    + `mode`. Defaults to `"cosine"`.
    + `k`. Defaults to `0.01`.
    + `batch_size`. Defaults to `0.1`.
    + `reg`. Defaults to `0.05`.
    + `accept_strat`. Defaults to `"mh"`.
    + `temp_scheduler`. Defaults to `None`.
    + `device`. Defaults to `None`.
    """

    def __init__(
        self,
        source: Tensor,
        pool: Tensor,
        k: int = 100,
        source_weights: Tensor | None = None,
        init_strat: str = "kmeans",
        mode: str = "cosine",
        batch_size: int | float | str = "sqrt",
        reg: float = 0.05,
        accept_strat: str = "mh",
        temp_scheduler: TemperatureScheduler | None = None,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.source = source.to(device=self.device)
        self.pool = pool.to(device=self.device)
        self.k = k
        self.source_weights = self._validate_args("source_weights", source_weights)
        self.init_strat = self._validate_args("init_strat", init_strat)
        self.mode = self._validate_args("mode", mode)
        self.batch_size = self._validate_args("batch_size", batch_size)
        self.reg = reg
        self.accept_strat = self._validate_args("accept_strat", accept_strat)
        self.temp_scheduler = temp_scheduler

        self._post_init()

    def _validate_args(self, arg: str, value):
        if arg == "source_weights":
            if value is None:
                N = self.source.shape[0]
                return torch.ones(size=[N], device=self.device) / N
            if value is not None:
                assert isinstance(value, Tensor), "source_weights must be a Tensor."
                assert value.shape[0] == self.source.shape[0], (
                    "source_weights must match source size."
                )
                value = value.to(device=self.device)
                value = value / value.sum(dim=0)
                return value
            return value

        elif arg == "init_strat":
            assert value in ["random", "kmeans++", "kmeans"], f"Unsupported init_strat: {value}."
            return value

        elif arg == "mode":
            assert value in ["cosine", "euclidean"], f"Unsupported geometry mode: {value}."
            return value

        elif arg == "batch_size":
            if isinstance(value, float):
                assert 0 < value < 1, "If batch_size is a float, it must be in (0, 1)."
            elif isinstance(value, int):
                assert value > 0, "If batch_size is an int, it must be positive."
            elif isinstance(value, str):
                assert value == "sqrt", "If batch_size is a string, it must be 'sqrt'."
                value = np.round(self.k**0.5).astype(int).item()
            else:
                raise ValueError("batch_size must be either float, int, or 'sqrt'.")
            return value

        elif arg == "accept_strat":
            assert value in ["strict", "mh"], f"Unsupported accept_strat: {value}."
            return value

        else:
            raise ValueError(f"Unknown argument: {arg}")

    def _post_init(self):
        # Geometry-specific normalization
        if self.mode == "cosine":
            self.source = self.source / (self.source.norm(dim=1, keepdim=True) + 1e-9)
            self.pool = self.pool / (self.pool.norm(dim=1, keepdim=True) + 1e-9)

        # Assign pool points to closest centroid (Euclidean)
        if self.init_strat == "random":
            rand_indices = torch.randperm(self.pool.shape[0])[: self.k]
            self.cds_idx = rand_indices.to(device=self.device)
        elif self.init_strat == "kmeans++":
            kmeanspp = KMeansPlusPlus(K=self.k)
            idx = kmeanspp.fit(X_train=self.source).to(device=self.device)
            # Enforce uniqueness
            idx_unique = []
            used = set()
            for i in idx:
                if i.item() not in used:
                    idx_unique.append(i.item())
                    used.add(i.item())
                else:
                    # Resample or pick next nearest pool point
                    for j in range(self.pool.shape[0]):
                        if j not in used:
                            idx_unique.append(j)
                            used.add(j)
                            break
            self.cds_idx = torch.tensor(idx_unique, device=self.device)
        else:
            kmeans = KMeansClassifier(K=self.k)
            centroids, log = kmeans.fit(X_train=self.source)
            distance = centroids.unsqueeze(dim=0) - self.pool.unsqueeze(dim=1)
            idx = distance.norm(p=2, dim=2).argmin(dim=0)
            # Enforce uniqueness
            idx_unique = []
            used = set()
            for i in idx:
                if i.item() not in used:
                    idx_unique.append(i.item())
                    used.add(i.item())
                else:
                    # Pick next nearest pool point
                    sorted_dist = torch.argsort(distance.norm(p=2, dim=2)[:, idx.tolist().index(i.item())])
                    for j in sorted_dist:
                        if j.item() not in used:
                            idx_unique.append(j.item())
                            used.add(j.item())
                            break
            self.cds_idx = torch.tensor(idx_unique, device=self.device)

        # Construct cost function
        if self.mode == "cosine":
            self.cost_fn = CosineDissimilarity(norm=False)
        else:
            self.cost_fn = CDist(p=2, power=1)

    def temperature(self, step: int, num_steps: int) -> float:
        if self.temp_scheduler is None:
            return 1.0
        else:
            return self.temp_scheduler(step, num_steps)

    @staticmethod
    def compute_utility(pi: Tensor, C: Tensor) -> Tensor:
        """Args:
        + `pi`: Transport plan of shape `[N, K]`.
        + `C`: Cost matrix of shape `[N, K]`.

        Returns: `utility` of shape `[K]`.
        """
        mass = pi.sum(dim=0)
        cost_per_src_obj = (pi * C).sum(dim=0)
        utility = mass / (cost_per_src_obj + 1e-9)
        return utility

    def stochastic_swap(self, step: int, num_steps: int):
        N, M = self.source.shape[0], self.pool.shape[0]

        assert self.source_weights is not None, (
            "source_weights must be provided (normalized, sum=1)"
        )
        source_weights = self.source_weights
        target_weights = torch.ones(size=[self.k], device=self.device) / self.k

        condensed = self.pool[self.cds_idx]  # [K, D]
        cost_mat = self.cost_fn(self.source, condensed)  # [N, K]
        reg = self.reg * cost_mat.mean()
        pi = ot.sinkhorn(  # [N, K]
            a=source_weights,  # Fisher-weighted source
            b=target_weights,  # uniform target
            M=cost_mat,  # cost matrix
            reg=reg,  # regularization term
        )
        utility = self.compute_utility(pi, cost_mat)  # [K]

        # Metropolis-Hastings swap
        temperature = self.temperature(step, num_steps)
        p_swap = (-utility / temperature).softmax(dim=0)
        iidx_remove = torch.multinomial(
            input=p_swap, num_samples=self.batch_size, replacement=False
        )
        idx_chosen = []
        used_indices = set(self.cds_idx.tolist())
        for iir in iidx_remove:
            # Orphaned source points
            mask_orphaned = self.select(x=pi[:, iir], top_p=0.9, top_k=10)
            weights = pi[mask_orphaned, iir]  # [V], V < N
            weights = weights / weights.sum(dim=0)

            # Compute barycenter (information-weighted)
            bary = (self.source[mask_orphaned] * weights.unsqueeze(dim=1)).sum(dim=0)  # [D]
            if self.mode == "cosine":
                bary = bary / (bary.norm(p=2, dim=0) + 1e-9)

            # Pool recruitment (geometry-specific)
            cost_pool_bary = self.cost_fn(self.pool, bary.unsqueeze(dim=0)).squeeze(dim=1)  # [M]
            nearest = torch.topk(cost_pool_bary.cpu(), k=10, largest=False)
            # Exclude already-selected indices
            for idx in nearest.indices.numpy():
                if idx not in used_indices:
                    idx_chosen.append(idx)
                    used_indices.add(idx)
                    break

        # Propose new set
        cds_idx_new = self.cds_idx.clone().detach()
        # Ensure batch replacements are unique
        for i, iir in enumerate(iidx_remove):
            cds_idx_new[iir] = idx_chosen[i]

        # Acceptance
        condensed_new = self.pool[cds_idx_new]
        cost_mat_new = self.cost_fn(self.source, condensed_new)  # [N, K]
        reg = self.reg * cost_mat_new.mean()
        pi_new = ot.sinkhorn(  # [N, K]
            a=source_weights,  # Fisher-weighted source
            b=target_weights,  # uniform target
            M=cost_mat_new,  # cost matrix
            reg=reg,  # regularization term
        )

        cost_old = (pi * cost_mat).sum()
        cost_new = (pi_new * cost_mat_new).sum()
        # Validate uniqueness before accepting proposal
        accept = self.decide_acceptance(
            cost_new=cost_new, cost_old=cost_old, temperature=temperature
        )
        if accept and len(set(cds_idx_new.tolist())) == len(cds_idx_new):
            self.cds_idx = cds_idx_new
        return cost_old.cpu().item(), cost_new.cpu().item(), accept

    @staticmethod
    def select(
        x: Tensor,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        sort: bool = True,
        as_mask: bool = False,
    ) -> Tensor:
        if top_p is None and top_k is None and min_p is None:
            raise ValueError("At least one selection criterion must be specified.")

        x_norm = x / x.sum(dim=0)
        idx_select = torch.arange(x.shape[0], device=x.device)
        if top_p is not None:
            sorted_x, sorted_idx = torch.sort(x_norm, descending=True)
            cumsum_x = torch.cumsum(sorted_x, dim=0)
            mask = cumsum_x <= top_p
            idx_top_p = sorted_idx[mask]
            idx_select = set_intersection(idx_select, idx_top_p)
        if top_k is not None:
            idx_top_k = x_norm.topk(k=top_k).indices
            idx_select = set_intersection(idx_select, idx_top_k)
        if min_p is not None:
            idx_min_p = (x_norm >= min_p).nonzero().squeeze(dim=1)
            idx_select = set_intersection(idx_select, idx_min_p)

        if sort:
            idx_select = idx_select[torch.argsort(x[idx_select], descending=True)]

        if as_mask:
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[idx_select] = True
            return mask

        return idx_select

    def decide_acceptance(self, cost_new: Tensor, cost_old: Tensor, temperature: float) -> bool:
        if self.accept_strat == "strict":
            accept: bool = cost_new < cost_old.item()
            return accept
        else:
            accept = False
            if cost_new < cost_old:
                accept = True
            else:
                prob = ((cost_old - cost_new) / temperature).exp()
                if prob > torch.rand(size=[]):
                    accept = True
            return accept

    def fit(self, num_epochs: int = 200):
        logs = {
            "cost_old": [],
            "cost_new": [],
            "accept": [],
            "cost_diff": [],
            "temperature": [],
            "trajectory": [],  # Track prototype indices at each step
        }
        pbar = tqdm.tqdm(range(num_epochs), desc="DiscreteOT")
        # Cache best-so-far solution
        best_cost = float("inf")
        best_indices = None
        for step in pbar:
            cost_old, cost_new, accept = self.stochastic_swap(step=step, num_steps=num_epochs)
            logs["cost_old"].append(cost_old)
            logs["cost_new"].append(cost_new)
            logs["accept"].append(accept)
            if step == 0:
                logs["cost_diff"].append(cost_new - cost_old)
            else:
                logs["cost_diff"].append(cost_old - logs["cost_old"][-2])
            temperature = self.temperature(step, num_epochs)
            logs["temperature"].append(temperature)
            logs["trajectory"].append(self.cds_idx.clone().detach().tolist())
            # Update best-so-far solution
            if cost_new < best_cost:
                best_cost = cost_new
                best_indices = self.cds_idx.clone().detach().cpu()
            pbar.set_postfix(
                {
                    "cost_old": f"{cost_old:.4f}",
                    "cost_new": f"{cost_new:.4f}",
                    "cost_diff": f"{logs['cost_diff'][-1]:.4f}",
                    "temperature": f"{temperature:.4f}",
                    "accept": f"{accept}",
                    "best_cost": f"{best_cost:.4f}",
                }
            )
        # Return best-so-far indices and logs
        return best_indices, logs


class TemporalKernel:
    """Temporal kernel to prioritize early influence.

    Args:
    + `start`: Start value. Defaults to `1`.
    + `stop`: Stop value. Defaults to `1e-4`.
    + `strategy`: One of `"cosine"`, `"exponential"`, `"linear"`, `"first"`, `"constant"` (all 1s), \
        `"last"`. Defaults to `"exponential"`.
    """

    supported_kernels = ["cosine", "exponential", "linear", "constant", "first", "last"]

    def __init__(self, start: float = 1, stop: float = 1e-4, strategy: str = "exponential"):
        if strategy not in self.supported_kernels:
            raise ValueError(f"Strategy {strategy} not supported.")

        self.start = start
        self.stop = stop
        self.strategy = strategy

    def __call__(self, num_steps: int) -> Tensor:
        progress = torch.linspace(start=0, end=1, steps=num_steps)

        if self.strategy == "cosine":
            p2p = self.start - self.stop
            return self.stop + p2p * 0.5 * (1 + torch.cos(np.pi * progress))
        elif self.strategy == "exponential":
            return self.start * (self.stop / self.start) ** progress
        elif self.strategy == "linear":
            return self.start - progress * (self.start - self.stop)
        elif self.strategy == "constant":
            return torch.ones(size=[num_steps]) * self.start
        elif self.strategy == "first":
            kernel = torch.zeros(size=[num_steps])
            kernel[0] = 1.0
            return kernel
        elif self.strategy == "last":
            kernel = torch.zeros(size=[num_steps])
            kernel[-1] = 1.0
            return kernel
        else:
            raise ValueError(f"Strategy {self.strategy} not supported.")


class TrajectoryAwareKnowledgeEstimator:
    """Computes knowledge values for each sample, optionally applying temporal kernel smoothing.

    Args:
    + `model`: The model for which to compute knowledge values.
    + `params_inf`: Parameters to compute influence. Can be a string (attribute of the model) or \
        one of the model's layer explicitly.
    + `temporal_kernel`: Temporal kernel strategy or instance. Defaults to `"exponential"`.
    + `loss_fn`: Loss function. Defaults to `"ce"`.
    + `verbose`: Whether to display a progress bar. Defaults to `True`.
    + `device`: Device for computation. Defaults to `"auto"`, which uses the model's device.
    """

    def __init__(
        self,
        model: nn.Module,
        params_inf: str | nn.Module,
        temporal_kernel: str | TemporalKernel = "exponential",
        loss_fn: str = "ce",
        verbose: bool = True,
        device: str = "auto",
    ):
        self.model = model
        self.params_inf: nn.Module = self._validate_args("params_inf", params_inf)
        self.temporal_kernel = self._validate_args("temporal_kernel", temporal_kernel)
        self.loss_fn = self._validate_args("loss_fn", loss_fn)
        self.verbose = verbose
        self.device = self._validate_args("device", device)

    def _validate_args(self, arg: str, value):
        if arg == "params_inf":
            if isinstance(value, str):
                msg = f"params_inf as str must be an attribute of the model, got: '{value}'."
                assert hasattr(self.model, value), msg
                return getattr(self.model, value)
            else:
                assert value

        elif arg == "temporal_kernel":
            if isinstance(value, str):
                assert value in ["cosine", "exponential", "linear"], (
                    f"Unsupported temporal kernel strategy: {value}."
                )
                return TemporalKernel(strategy=value)
            elif isinstance(value, TemporalKernel):
                return value
            else:
                raise ValueError(
                    "temporal_kernel must be either a string or a TemporalKernel instance."
                )

        elif arg == "loss_fn":
            if value == "ce":
                return nn.CrossEntropyLoss()
            else:
                return value

        elif arg == "device":
            if value == "auto":
                return next(self.model.parameters()).device
            else:
                raise value

        else:
            raise ValueError(f"Unknown argument: {arg}")

    def __call__(
        self,
        inputs: Tensor,
        targets: Tensor,
        trajectory_dir: str | None = None,
        opt_kwargs: dict = {
            "Class": torch.optim.AdamW,
            "kwargs": {"lr": 0.003, "weight_decay": 5e-4},
        },
        batch_size: int = 128,
        num_updates_per_step: int = 2,
        num_steps: int = 50,
    ) -> Tensor:
        """Compute knowledge values.

        Args:
        + `inputs`: Input data tensor of shape [N, *].
        + `targets`: Target labels tensor of shape [N].
        + `batch_size`: Batch size. Defaults to `128`.
        + `num_updates_per_step`: Number of updates per step. Defaults to `2`.
        + `num_steps`: Number of total steps. Defaults to `50`.
        + `trajectory_dir`: Directory containing weights of model. If specified, will load weights \
            instead of updating model for the trajectory. Defaults to `None`.
        + `opt_kwargs`: Kwargs for optimizer. Defaults to \
            `{"Class": torch.optim.AdamW, "kwargs": {"lr": 0.003, "weight_decay": 5e-4}}`.

        Returns: Knowledge values with respect to `(inputs, targets)` of shape [N].
        """
        influence_matrix = self.compute_influence_matrix(
            inputs=inputs,
            targets=targets,
            trajectory_dir=trajectory_dir,
            opt_kwargs=opt_kwargs,
            batch_size=batch_size,
            num_updates_per_step=num_updates_per_step,
            num_steps=num_steps,
        )
        knowledge = self.compute_knowledge_values(influence_matrix=influence_matrix)
        return knowledge

    def compute_influence_matrix(
        self,
        inputs: Tensor,
        targets: Tensor,
        trajectory_dir: str | None = None,
        opt_kwargs: dict = {},
        batch_size: int = 128,
        num_updates_per_step: int = 2,
        num_steps: int = 50,
    ) -> Tensor:
        preload_traj = isinstance(trajectory_dir, str)
        if preload_traj:
            print("`trajectory_dir` provided, will load model weights to compute knowledge.")
            weights_traj = [f for f in sorted(os.listdir(trajectory_dir)) if f.endswith(".pt")]
            num_steps = len(weights_traj)  # Override
        else:
            print("`trajectory_dir` not provided, will train model to compute knowledge.")
            opt = opt_kwargs["Class"](params=self.model.parameters(), **opt_kwargs["kwargs"])
            assert isinstance(num_steps, int), "num_steps must be an integer"
        assert isinstance(batch_size, int), "batch_size must be an integer"
        assert isinstance(num_updates_per_step, int), "num_updates_per_step must be an integer"

        infl_matrix = torch.zeros(size=[inputs.shape[0], num_steps])
        if self.verbose:
            pbar_desc = "TAKE | " + ("preload" if preload_traj else "train")
            pbar = tqdm.tqdm(range(num_steps), desc=pbar_desc)
        else:
            pbar = range(num_steps)

        for i in pbar:
            if preload_traj:
                log = self.update_classifier_via_weights(
                    inputs=inputs,
                    targets=targets,
                    path_weight=weights_traj[i],
                    batch_size=batch_size,
                    num_updates_per_step=num_updates_per_step,
                )
            else:
                log = self.update_classifier_via_train(
                    inputs=inputs,
                    targets=targets,
                    optimizer=opt,
                    batch_size=batch_size,
                    num_updates_per_step=num_updates_per_step,
                )

            fishr = FisherScorer(model=self.model, params_inf=self.params_inf, loss_fn=self.loss_fn)
            infl_matrix[:, i] = fishr(inputs=inputs, targets=targets)
            if isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix(log)

        return infl_matrix

    def update_classifier_via_train(
        self,
        inputs: Tensor,
        targets: Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        num_updates_per_step: int,
    ) -> dict:
        acc = []
        losses = []

        for _ in range(num_updates_per_step):
            batch_idx = torch.randperm(inputs.shape[0])[:batch_size]
            x = inputs[batch_idx]
            y = targets[batch_idx]
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc.append((pred.argmax(dim=1) == y).float().mean().item())
            losses.append(loss.item())

        return {"accuracy": sum(acc) / len(acc), "loss": sum(losses) / len(losses)}

    def update_classifier_via_weights(
        self,
        inputs: Tensor,
        targets: Tensor,
        path_weight: str,
        batch_size: int,
        num_updates_per_step: int,
    ) -> dict:
        state_dict = torch.load(path_weight, map_location=self.device)
        self.model.load_state_dict(state_dict)

        acc = []
        losses = []
        for _ in range(num_updates_per_step):
            batch_idx = torch.randperm(inputs.shape[0])[:batch_size]
            x = inputs[batch_idx]
            y = targets[batch_idx]
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            acc.append((pred.argmax(dim=1) == y).float().mean().item())
            losses.append(loss.item())

        return {"accuracy": sum(acc) / len(acc), "loss": sum(losses) / len(losses)}

    def compute_knowledge_values(self, influence_matrix: Tensor) -> Tensor:
        knowledge = 1 / influence_matrix  # [N, T], high knowledge = low influence
        knowledge = knowledge / knowledge.sum(dim=0, keepdim=True)  # [N, T], normalize by samples
        temporal = self.temporal_kernel(num_steps=influence_matrix.shape[1])  # [T]
        knowledge = (knowledge * temporal.unsqueeze(dim=0)).sum(dim=1)  # [N], convolve by temporal
        knowledge = knowledge / knowledge.sum(dim=0)  # [N], normalize over samples
        return knowledge


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def expt_cluster_2D_fisher():
        from PIL import Image
        from utils.data.synthetic import get_gaussian_clusters_2D

        NUM_CLUSTERS = 4
        COLORS = ["red", "blue", "green", "magenta", "cyan"]
        PLOT_STEP = 0.01
        min_size, max_size = 5, 50

        X_train, y_train, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)
        X_test, y_test, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)

        if not os.path.exists("./logs/models_prototypes_discreteot_cluster2d_fisher/"):
            os.makedirs("./logs/models_prototypes_discreteot_cluster2d_fisher/", exist_ok=True)
        else:
            for f in os.listdir("./logs/models_prototypes_discreteot_cluster2d_fisher/"):
                os.remove(os.path.join("./logs/models_prototypes_discreteot_cluster2d_fisher/", f))

        model = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=NUM_CLUSTERS),
        )
        take = TrajectoryAwareKnowledgeEstimator(model=model, params_inf=model[-1])
        num_steps = 50
        infl_matrix = take.compute_influence_matrix(
            opt_kwargs={"Class": torch.optim.AdamW, "kwargs": {"lr": 0.01}},
            inputs=X_train,
            targets=y_train,
            num_steps=num_steps,
        )
        infl_matrix = infl_matrix / infl_matrix.sum(dim=0, keepdim=True)

        infl_matrix = infl_matrix.cpu().numpy()
        for i in range(num_steps):
            fig, ax = plt.subplots()
            ax.set_title(
                f"Fisher influence with {NUM_CLUSTERS} clusters, step {i + 1}/{num_steps}"
            )
            infl = infl_matrix[:, i]
            s = min_size + (infl - infl.min()) / (infl.max() - infl.min()) * (max_size - min_size)
            for k in range(NUM_CLUSTERS):
                idx = y_train == k
                if np.count_nonzero(idx) == 0:
                    continue
                ax.scatter(
                    X_train[idx, 0],
                    X_train[idx, 1],
                    color=COLORS[k],
                    alpha=0.3,
                    s=s[idx],
                    label=f"Cluster {k}",
                )
            fig.savefig(
                fname=f"./logs/models_prototypes_discreteot_cluster2d_fisher/{i + 1:02d}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Make GIF
        image_files = [
            os.path.join("./logs/models_prototypes_discreteot_cluster2d_fisher/", path)
            for path in sorted(os.listdir("./logs/models_prototypes_discreteot_cluster2d_fisher/"))
            if path.endswith(".png")
        ]
        frames = [Image.open(img) for img in image_files]
        frames[0].save(
            fp="logs/models_prototypes_discreteot_cluster2d_fisher.gif",
            save_all=True,
            append_images=frames[1:],
            duration=5000 // len(frames),
            loop=0,
        )
        for f in os.listdir("./logs/models_prototypes_discreteot_cluster2d_fisher/"):
            os.remove(os.path.join("./logs/models_prototypes_discreteot_cluster2d_fisher/", f))
        os.removedirs("./logs/models_prototypes_discreteot_cluster2d_fisher/")

    def expt_sphere_2D_fisher():
        from PIL import Image
        from utils.data.synthetic import get_gaussian_clusters_2D

        NUM_CLUSTERS = 4
        COLORS = ["red", "blue", "green", "magenta", "cyan"]
        PLOT_STEP = 0.01
        min_size, max_size = 5, 50

        X_train, y_train, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)
        X_test, y_test, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)
        X_train = X_train / (X_train.norm(dim=1, keepdim=True))
        X_test = X_test / (X_test.norm(dim=1, keepdim=True))

        if not os.path.exists("./logs/models_prototypes_discreteot_sphere2d_fisher/"):
            os.makedirs("./logs/models_prototypes_discreteot_sphere2d_fisher/", exist_ok=True)
        else:
            for f in os.listdir("./logs/models_prototypes_discreteot_sphere2d_fisher/"):
                os.remove(os.path.join("./logs/models_prototypes_discreteot_sphere2d_fisher/", f))

        model = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=NUM_CLUSTERS),
        )
        take = TrajectoryAwareKnowledgeEstimator(
            model=model, params_inf=model[-1], 
        )
        num_steps = 50
        infl_matrix = take.compute_influence_matrix(
            inputs=X_train,
            targets=y_train,
            opt_kwargs={"Class": torch.optim.Adam, "kwargs": {"lr": 0.01}},
        )
        infl_matrix = infl_matrix / infl_matrix.sum(dim=0, keepdim=True)

        infl_matrix = infl_matrix.cpu().numpy()
        for i in range(num_steps):
            # xy ratio = 1:1
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.set_title(
                f"Fisher influence with {NUM_CLUSTERS} clusters, step {i + 1}/{num_steps}"
            )
            infl = infl_matrix[:, i]
            s = min_size + (infl - infl.min()) / (infl.max() - infl.min()) * (max_size - min_size)
            for k in range(NUM_CLUSTERS):
                idx = y_train == k
                if np.count_nonzero(idx) == 0:
                    continue
                ax.scatter(
                    X_train[idx, 0],
                    X_train[idx, 1],
                    color=COLORS[k],
                    alpha=0.3,
                    s=s[idx],
                    label=f"Cluster {k}",
                )
            fig.savefig(
                fname=f"./logs/models_prototypes_discreteot_sphere2d_fisher/{i + 1:02d}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Make GIF
        image_files = [
            os.path.join("./logs/models_prototypes_discreteot_sphere2d_fisher/", path)
            for path in sorted(os.listdir("./logs/models_prototypes_discreteot_sphere2d_fisher/"))
            if path.endswith(".png")
        ]
        frames = [Image.open(img) for img in image_files]
        frames[0].save(
            fp="logs/models_prototypes_discreteot_sphere2d_fisher.gif",
            save_all=True,
            append_images=frames[1:],
            duration=5000 // len(frames),
            loop=0,
        )
        for f in os.listdir("./logs/models_prototypes_discreteot_sphere2d_fisher/"):
            os.remove(os.path.join("./logs/models_prototypes_discreteot_sphere2d_fisher/", f))
        os.removedirs("./logs/models_prototypes_discreteot_sphere2d_fisher/")

    def expt_cluster_2D_temporalkernel():
        from PIL import Image
        from utils.data.synthetic import get_gaussian_clusters_2D

        NUM_CLUSTERS = 4
        COLORS = ["red", "blue", "green", "magenta", "cyan"]
        PLOT_STEP = 0.01
        min_size, max_size = 5, 50

        X_train, y_train, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)
        X_test, y_test, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)

        if not os.path.exists("./logs/models_prototypes_discreteot_cluster2d_fisher/"):
            os.makedirs("./logs/models_prototypes_discreteot_cluster2d_fisher/", exist_ok=True)
        else:
            for f in os.listdir("./logs/models_prototypes_discreteot_cluster2d_fisher/"):
                os.remove(os.path.join("./logs/models_prototypes_discreteot_cluster2d_fisher/", f))

        model = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=NUM_CLUSTERS),
        )
        take = TrajectoryAwareKnowledgeEstimator(
            model=model,
            params_inf=model[-1]
        )
        infl_matrix = take.compute_influence_matrix(
            inputs=X_train,
            targets=y_train,
            opt_kwargs={"Class": torch.optim.Adam, "kwargs": {"lr": 0.01}},
        )

        # Prioritize early influence with temporal kernels
        knowledge = {}
        for strategy in take.temporal_kernel.supported_kernels:
            # Reset the temporal kernel
            take.temporal_kernel = TemporalKernel(strategy=strategy)
            knowledge[strategy] = take.compute_knowledge_values(influence_matrix=infl_matrix)

            fig, ax = plt.subplots()
            ax.set_title(f"Weights with {NUM_CLUSTERS} clusters, temporal={strategy}")

            klg = knowledge[strategy].cpu().numpy()
            s = min_size + (klg - klg.min()) / (klg.max() - klg.min()) * (max_size - min_size)
            for k in range(NUM_CLUSTERS):
                idx = y_train == k
                if np.count_nonzero(idx) == 0:
                    continue
                ax.scatter(
                    X_train[idx, 0],
                    X_train[idx, 1],
                    color=COLORS[k],
                    alpha=0.3,
                    s=s[idx],
                    label=f"Cluster {k}",
                )
            fig.savefig(
                fname=f"./logs/models_prototypes_discreteot_cluster2d_strat={strategy}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    def expt_cluster_2D():
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from utils.data.synthetic import get_gaussian_clusters_2D

        NUM_CLUSTERS = 4
        K = 5
        COLORS = ["red", "blue", "green", "magenta", "cyan"]
        PLOT_STEP = 0.01
        NUM_BATCHES, STEP = 100, 2
        min_size, max_size = 5, 50

        X_train, y_train, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)
        X_test, y_test, _ = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)

        model = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=NUM_CLUSTERS),
        )
        # Evaluate influence matrix: influence over time
        take = TrajectoryAwareKnowledgeEstimator(model=model,params_inf=model[-1])
        knowledge_values = take(
            inputs=X_train,
            targets=y_train,
            opt_kwargs={"Class": torch.optim.AdamW, "kwargs" : {"lr": 0.01}},
        )

        # Use all points as pool, select K prototypes
        distiller = DiscreteOTDistiller(
            source=X_train,
            pool=X_test,
            source_weights=knowledge_values,
            init_strat="kmeans",
            mode="euclidean",
            k=K,
            batch_size="sqrt",
            reg=0.02,
            accept_strat="mh",
            temp_scheduler=TemperatureScheduler(start_temp=1e-2, stop_temp=1e-4, strategy="cosine"),
        )
        final_indices, logs = distiller.fit(num_epochs=100)
        selected = X_test[final_indices]
        print("Start", {k: v[0] for k, v in logs.items()})
        print("End", {k: v[-1] for k, v in logs.items()})

        # Plot all selected prototypes and examples
        min_size, max_size = 5, 50
        fig, ax = plt.subplots()
        ax.set_title(f"Discrete OT (K={K}) with {NUM_CLUSTERS} clusters")
        w = knowledge_values.cpu().numpy()
        w_scaled = min_size + (w - w.min()) / (w.max() - w.min()) * (max_size - min_size)
        for k in range(NUM_CLUSTERS):
            idx = y_train == k
            if np.count_nonzero(idx) == 0:
                continue
            ax.scatter(
                X_train[idx, 0],
                X_train[idx, 1],
                color=COLORS[k],
                alpha=0.3,
                s=w_scaled[idx],
                label=f"Cluster {k}",
            )
        fig.savefig(
            fname=f"./logs/models_prototypes_discreteot_cluster2d.png",
            dpi=300,
            bbox_inches="tight",
        )

        # Plot prototype trajectories
        cds_idx_hist = np.stack(logs["trajectory"])  # [steps, K]
        for proto in range(K):
            traj = X_test[cds_idx_hist[:, proto]]  # [steps, 2]
            ax.plot(traj[:, 0], traj[:, 1], color="black", linestyle="dashed", alpha=0.7)
            ax.scatter(traj[:, 0], traj[:, 1], color="black", marker="x", s=30, alpha=0.5)
        # Plot final prototypes
        ax.scatter(
            selected[:, 0],
            selected[:, 1],
            color="black",
            marker="*",
            s=200,
            label="Selected Prototypes",
        )
        ax.legend()
        fig.savefig(
            fname="./logs/models_prototypes_discreteot_cluster2d.png",
            dpi=300,
            bbox_inches="tight",
        )

    # expt_cluster_2D_fisher()
    expt_sphere_2D_fisher()
    # expt_cluster_2D_temporalkernel()
    # expt_cluster_2D()
