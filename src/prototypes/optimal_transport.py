import os
import sys

repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tqdm.auto as tqdm
from models.prototypes.kmeans import KMeansClassifier


# Early stopping
class ConvergenceStopper:
    """Stops optimization if the metric does not change significantly for a number of steps."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-2):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.prev_metric = float("inf")

    def __call__(self, metrics: dict) -> bool:
        stop = False
        metric = metrics["loss"]
        if abs(self.prev_metric - metric) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                stop = True
        else:
            self.counter = 0
        self.prev_metric = metric
        return stop


class TopKStopper:
    """Early stop if the top-K candidates do not change for a number of steps."""

    def __init__(self, patience: int = 10, ordered: bool = True):
        self.patience = patience
        self.counter = 0
        self.prev_idx_select: Tensor | None = None
        self.ordered = ordered

    def __call__(self, metrics: dict) -> bool:
        stop = False

        idx_select: Tensor = metrics["idx_select"]
        if not self.ordered:
            idx_select = idx_select.sort().values

        if self.prev_idx_select is not None:
            if (self.prev_idx_select == idx_select).all():
                self.counter += 1
                if self.counter >= self.patience:
                    stop = True
            else:
                self.counter = 0

        self.prev_idx_select = idx_select
        return stop


class DiversityLoss(nn.Module):
    def __init__(self, sigma: str | float = "median"):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: Tensor, w_x: Tensor) -> Tensor:
        dist = torch.cdist(x, x, p=2)
        sigma = self.get_sigma(dist)
        K = torch.exp(-(dist**2) / (2 * sigma**2))
        loss = w_x.t().matmul(K).matmul(w_x)
        return loss

    def get_sigma(self, dist: Tensor) -> float:
        if not isinstance(self.sigma, str):
            return self.sigma

        if self.sigma == "median":
            return float(dist.median())

        raise ValueError(f"Unknown sigma: {self.sigma}")


class LassoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input.abs().mean()


class SinkhornDistance(nn.Module):
    """Computes the Sinkhorn distance (entropic-regularized Wasserstein distance) between two
    distributions.

    Args:
    + `epsilon`: Coefficient of entropic regularization. Larger values increase stability but bias \
        the solution. Defaults to `1e-2`.
    + `max_iter`: Maximum number of Sinkhorn iterations. Defaults to `100`.
    + `tol`: Tolerance for convergence check. Defaults to `1e-5`.
    + `device`: Device to run the algorithm on. Defaults to `cuda`.

    ---

    The Sinkhorn algorithm solves the entropic-regularized Optimal Transport problem:

    W(wx, wy) = min_{P in U(wx, wy)} <C, P> - epsilon * H(P)

    where:
    - wx is the source measure.
    - wy is the target measure.
    - C is the cost matrix.
    - U(wx, wy) is the set of transport plans with marginals wx, wy.
    - H(P) is the entropy of the transport plan.
    """

    def __init__(
        self,
        epsilon: float = 1e-2,
        max_iter: int = 100,
        tol: float = 1e-5,
        device: str = "cuda",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def forward(
        self, x: Tensor, y: Tensor, w_x: Tensor | None = None, w_y: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Computes the Sinkhorn squared distance between two sets of samples.

        Args:
        + `x`: Source samples [N, D].
        + `y`: Target samples [M, D].
        + `w_x`: Source weights [N]. Sum must be 1. If None, assumes uniform.
        + `w_y`: Target weights [M]. Sum must be 1. If None, assumes uniform.
        Returns:
        + Sinkhorn distance [1].
        + Optimal transport plan [N, M].
        """
        C = torch.cdist(x, y, p=2) ** 2

        log_mu = self.get_log_weights(w_x, x.shape[0])
        log_nu = self.get_log_weights(w_y, y.shape[0])

        g = torch.zeros_like(log_nu)
        f = torch.zeros_like(log_mu)
        for _ in range(self.max_iter):
            f_prev = f.clone()
            f = self.epsilon * (
                log_mu - torch.logsumexp((g.unsqueeze(0) - C) / self.epsilon, dim=1)
            )
            g = self.epsilon * (
                log_nu - torch.logsumexp((f.unsqueeze(1) - C) / self.epsilon, dim=0)
            )

            err = (f - f_prev).abs().mean()
            if err < self.tol:
                break

        log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / self.epsilon
        P = log_P.exp()
        dist = torch.sum(P * C)
        return dist, P

    def get_log_weights(self, weights: Tensor | None, size: int) -> Tensor:
        if weights is None:
            weights = torch.ones(size=[size], device=self.device) / size
        else:
            weights = weights.abs() + 1e-8
            weights = weights / weights.sum(dim=0)
        return weights.log()


class OptimalTransportCondenser(nn.Module):
    """
    Condenses a dataset by selecting a subset from a fixed candidate pool that minimizes
    the Optical Transport distance to the original dataset.

    Args:
    + `sinkhorn`: Sinkhorn distance module.
    + `num_steps`: Number of gradient descent steps to optimize weights. Defaults to `100`.
    + `batch_size`: Batch size for gradient descent. Defaults to `256`.
    + `tau_scheduler`: Temperature scheduler for annealing. Defaults to `None`.

    ---

    The algorithm optimizes a probability distribution (weights) over the candidate pool
    such that the Sinkhorn divergence between the original dataset (assumed uniform)
    and the weighted candidate pool is minimized.

    min_{w in M} W(mu, sum_{j} w_j delta_{c_j})

    After optimization, the top-K candidates with highest weights are selected.
    """

    def __init__(
        self,
        sinkhorn: SinkhornDistance,
        batch_size: int = 256,
        opt_kwargs: dict | None = None,
        num_steps: int = 100,
        stopper: ConvergenceStopper | TopKStopper | None = None,
        loss_fn_diversity: nn.Module | None = None,
        loss_fn_lasso: nn.Module | None = None,
        coeff_diversity: float = 1,
        coeff_lasso: float = 0.05,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.sinkhorn = sinkhorn
        self.batch_size = batch_size
        self.num_steps = num_steps

        if opt_kwargs is not None:
            self.opt_kwargs = opt_kwargs
        else:
            self.opt_kwargs = {
                "Class": torch.optim.Adam,
                "kwargs": {"lr": 0.05},
            }

        if loss_fn_diversity is None:
            self.loss_fn_diversity = DiversityLoss()
        if loss_fn_lasso is None:
            self.loss_fn_lasso = LassoLoss()

        self.coeff_diversity = coeff_diversity
        self.coeff_lasso = coeff_lasso

        self.stopper = stopper
        self.device = device

    def fit(
        self,
        y: Tensor,
        x: Tensor | None,
        num_select: int,
        w_y: Tensor | None = None,
        init: str = "kmeans",
        save_history: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Condense inputs by selecting the best representatives from a candidate pool.

        Args:
        + `y`: Original dataset (Target) [N, D].
        + `x`: Pool of candidates (Source) [M, D]. If None, `y` itself is used as the pool.
        + `num_select`: Number of candidates to select.
        + `w_y`: Weights for target samples [N].
        + `init`: Initialization strategy for candidates weights (logits). "random" or "kmeans".
        + `save_history`: Whether to save the history of selected candidates.
        + `device`: Device to run on.
        + `stopper`: Convergence stopper.

        Returns:
        + Selected candidates.
        + Logs (including history if requested).
        """
        if x is None:
            x = y.clone().detach()
        y = y.to(device=self.device)
        x = x.to(device=self.device)

        if isinstance(w_y, Tensor):
            w_y = w_y.to(device=self.device)

        # Initialize learnable log-weights for candidates (Source x)
        logits_x = torch.normal(mean=0, std=0.01, size=[x.shape[0]], device=self.device)
        logits_x = nn.Parameter(data=logits_x, requires_grad=True)

        if init == "kmeans":
            kmeans = KMeansClassifier(K=num_select)
            centroids, _ = kmeans.fit(X_train=y)
            # Upweights K closest candidates to centroids
            dist = torch.cdist(x, centroids, 2)
            idx_closest = dist.argmin(dim=0)
            logits_x.data[idx_closest] = 2
            print(f"idx_closest: {idx_closest}")

        opt: torch.optim.Optimizer = self.opt_kwargs["Class"](  # ty:ignore
            params=[logits_x],
            **self.opt_kwargs["kwargs"],  # ty:ignore
        )
        history = []
        pbar = tqdm.tqdm(range(self.num_steps), desc="OT", leave=False)
        for i in pbar:
            batch_y, w_y = self.get_batch(inputs=y, weights=w_y, batch_size=self.batch_size)

            opt.zero_grad()
            w_x = F.softmax(logits_x, dim=0)
            loss_ot, _ = self.sinkhorn(x=x, y=batch_y, w_x=w_x, w_y=w_y)
            loss_diversity = self.loss_fn_diversity(x=x, w_x=w_x)
            loss_lasso = self.loss_fn_lasso(logits_x)
            loss: Tensor = (
                loss_ot + self.coeff_diversity * loss_diversity + self.coeff_lasso * loss_lasso
            )

            loss.backward()
            opt.step()

            idx_select = w_x.clone().detach().topk(k=num_select).indices
            if save_history:
                history.append(x[idx_select].unsqueeze(0))

            metrics = {
                "loss": loss.item(),
                "loss_ot": loss_ot.item(),
                "loss_dv": loss_diversity.item(),
                "loss_ls": loss_lasso.item(),
                "idx_select": idx_select.tolist(),
            }
            pbar.set_postfix(metrics)

            if self.stopper is not None:
                stop_signal = self.stopper(metrics)
                if stop_signal:
                    break

        # Select top K candidates
        idx_select = w_x.clone().detach().topk(k=num_select).indices
        candidates_select = x[idx_select]
        logs = {}
        if save_history:
            logs["history"] = torch.cat(history, dim=0)
        return candidates_select.clone().detach(), logs

    @staticmethod
    def get_batch(
        inputs: Tensor, weights: Tensor | None, batch_size: int
    ) -> tuple[Tensor, Tensor | None]:
        num_samples = inputs.shape[0]

        if batch_size < num_samples:
            indices = torch.randperm(num_samples)[:batch_size]
            batch_inputs = inputs[indices]
            if weights is not None:
                batch_weights = weights[indices]
                batch_weights = batch_weights / batch_weights.sum(dim=0)
            else:
                batch_weights = None
        else:
            batch_inputs = inputs
            batch_weights = weights

        return batch_inputs, batch_weights


if __name__ == "__main__":

    def test_optimal_transport_condenser():
        print("Testing Optimal Transport Condenser")

        import matplotlib.pyplot as plt

        from utils.data.synthetic import get_gaussian_clusters_2D

        num_clusters = 8
        K = 10

        inputs, labels, metadata = get_gaussian_clusters_2D(
            num_clusters=num_clusters, num_examples=1000, radius=6, sigma_diag=0.5
        )
        candidates = torch.normal(mean=0, std=5, size=[500, 2])

        print(f"inputs shape: {inputs.shape}")
        print(f"Candidates shape: {candidates.shape}")

        # Initialize condenser
        condenser = OptimalTransportCondenser(
            sinkhorn=SinkhornDistance(epsilon=0.1),
            opt_kwargs={"Class": torch.optim.Adam, "kwargs": {"lr": 0.01}},
            num_steps=200,
            coeff_diversity=1,
            coeff_lasso=0.01,
        )
        condensed, logs = condenser.fit(
            y=inputs,
            x=candidates,
            num_select=K,
            init="kmeans",
            # init=None,
            save_history=True,
        )

        condensed = condensed.cpu()
        inputs = inputs.cpu()
        candidates = candidates.cpu()
        if "history" in logs:
            history = logs["history"].cpu()

        print(f"Condensed shape: {condensed.shape}")
        print(f"Selected candidates:\n{condensed}")

        # Validation: Check if selected points are close to true centers
        for k in range(num_clusters):
            dist = (condensed - metadata["mu"][k]).norm(dim=1).min()
            print(f"Distance to centroid {k}: {dist.item():.4f}")

        # Visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(
            inputs[:, 0].numpy(), inputs[:, 1].numpy(), alpha=0.3, s=10, label="Original", c="blue"
        )
        plt.scatter(
            candidates[:, 0].numpy(),
            candidates[:, 1].numpy(),
            alpha=0.5,
            s=30,
            label="Cands",
            c="gray",
            marker="x",
        )
        # Plot history
        if "history" in logs:
            num_steps, num_select, _ = history.shape
            sorted_history = torch.zeros_like(history)
            for t in range(num_steps):
                step_points = history[t]
                sort_idx = step_points[:, 0].argsort()
                sorted_history[t] = step_points[sort_idx]
            history = sorted_history

            for k in range(num_select):
                plt.plot(
                    history[:, k, 0].numpy(),
                    history[:, k, 1].numpy(),
                    color="red",
                    linestyle="dashed",
                    alpha=0.5,
                    label="History" if k == 0 else None,
                )
                plt.scatter(
                    history[:, k, 0].numpy(),
                    history[:, k, 1].numpy(),
                    color="red",
                    marker=".",
                    s=5,
                    alpha=0.5,
                )

        # Plot selected points
        plt.scatter(
            condensed[:, 0].numpy(),
            condensed[:, 1].numpy(),
            alpha=1.0,
            s=200,
            label="Selected Points",
            c="red",
            marker="*",
            edgecolors="black",
            zorder=100,
        )

        plt.title("Optimal Transport Condensation with History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("./logs/models_prototypes_optimal_transport.png")

    test_optimal_transport_condenser()
