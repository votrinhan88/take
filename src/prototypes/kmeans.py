import torch
from torch import nn, Tensor
import tqdm.auto as tqdm
import pytorch_lightning as pl


class KMeansPlusPlus:
    """Returns the indices of the initial centroids chosen via k-means++.

    Args:
    + `K`: Number of centroids.
    + `verbose`: Flag to show info. Defaults to `True`.

    Algorithm:
    1. Choose one centroid `c0` uniformly at random from among the data points.
    2. For each data point `x`, compute `D(x)^2`, the squared distance between `x` and the nearest \
        centroid: `D(x)^2 = min_{c \\in C} ||x - c||^2`
    3. Choose one new data point at random as a new centroid, using a weighted probability \
        distribution where a point `x` is chosen with probability proportional to `D(x)^2`.
    4. Repeat until `K` centroids have been chosen.
    """

    def __init__(self, K: int, verbose: bool = True):
        self.K = K
        self.verbose = verbose

    def fit(self, X_train: Tensor) -> Tensor:
        idx_centroids = torch.randint(low=0, high=X_train.shape[0], size=[1], device="cpu")
        centroid = X_train[idx_centroids]
        dist_sq = torch.sum((X_train - centroid) ** 2, dim=1)

        if self.verbose:
            pbar: tqdm.tqdm = tqdm.trange(self.K - 1, desc="K-means++", leave=False)
        else:
            pbar = range(self.K - 1)

        for _ in pbar:
            if dist_sq.sum() == 0:
                idx_next = torch.randint(low=0, high=X_train.shape[0], size=[], device="cpu")
            else:
                idx_next = torch.multinomial(input=dist_sq, num_samples=1).to(device="cpu")
            idx_centroids = torch.cat([idx_centroids, idx_next], dim=0)

            # Update distances for next iteration
            centroid_new = X_train[idx_next]
            new_dist_sq = torch.sum((X_train - centroid_new) ** 2, dim=1)
            dist_sq = torch.minimum(dist_sq, new_dist_sq)

        return idx_centroids

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"


class KMeansClassifier(pl.LightningModule):
    """K-Means.

    Args:
    + `K`: Number of centroids.
    + `verbose`: Flag to show info. Defaults to `True`.
    """

    def __init__(self, K: int, verbose: bool = True):
        super().__init__()
        self.K = K
        self.verbose = verbose

        self.centroids: Tensor

    def forward(self, inputs: Tensor) -> Tensor:
        """Fix centroids, update labels"""
        distance = torch.cdist(x1=inputs, x2=self.centroids, p=2)
        yhat = distance.argmin(dim=1)
        return yhat

    def _backward(self, inputs: Tensor, target: Tensor):
        """Fix labels, update centroids"""
        for k in torch.arange(self.K):
            idx_k = (target == k).nonzero().squeeze(dim=1)
            if idx_k.numel() > 0:
                self.centroids[k, :] = torch.mean(inputs[idx_k, :], dim=0)

    def fit(
        self,
        X_train: Tensor,
        num_epochs: int = 100,
        centroids_init: Tensor | str = "kmeans++",
        tolerance: float | None = None,
        save_history: bool = True,
    ) -> tuple[Tensor, dict]:
        if isinstance(centroids_init, Tensor):
            self.centroids = centroids_init
        elif centroids_init == "random":
            self.centroids = X_train[torch.randperm(X_train.shape[0])][0 : self.K]
        elif centroids_init == "kmeans++":
            self.centroids = X_train[KMeansPlusPlus(K=self.K, verbose=self.verbose).fit(X_train)]

        centroids_history = self.centroids.clone().unsqueeze(dim=0)

        if self.verbose:
            pbar: tqdm.tqdm = tqdm.trange(num_epochs, desc="K-means", leave=False)
        else:
            pbar = range(num_epochs)

        metrics = {"loss_l1": [], "loss_mse": []}
        for epoch in pbar:
            yhat = self(X_train)
            self._backward(X_train, yhat)

            stop_flag = self.check_early_stop(
                tolerance=tolerance, centroids_history=centroids_history
            )

            if save_history:
                centroids_history = torch.cat(
                    [centroids_history, self.centroids.unsqueeze(dim=0)], dim=0
                )
            else:
                centroids_history = self.centroids.clone().unsqueeze(dim=0)

            if stop_flag:
                if self.verbose:
                    print(f"Stopped at epoch {epoch}! Centroids stop moving.")
                break

            metrics_epoch = self.evaluate(X_train)
            for k in metrics_epoch.keys():
                metrics[k].append(metrics_epoch[k])

            if self.verbose:
                pbar.set_postfix(metrics_epoch)  # ty: ignore

        log = {"pred": yhat, **metrics}
        if save_history:
            log["centroids_history"] = centroids_history
        return self.centroids.clone().detach(), log

    def evaluate(self, X_test: Tensor) -> dict[str, float]:
        total: int = 0
        loss_l1: float = 0
        loss_mse: float = 0
        loss_fn_l1 = nn.L1Loss(reduction="sum")
        loss_fn_mse = nn.MSELoss(reduction="sum")

        with torch.inference_mode():
            pred: Tensor = self(X_test)
            total += X_test.shape[0]
            assigned_centroids = self.centroids[pred.squeeze(dim=0)]
            loss_l1 += loss_fn_l1(assigned_centroids, X_test)
            loss_mse += loss_fn_mse(assigned_centroids, X_test)

        loss_l1 = loss_l1 / total
        loss_mse = loss_mse / total
        results = {
            "loss_l1": loss_l1.item(),
            "loss_mse": loss_mse.item(),
        }
        return results

    def check_early_stop(self, tolerance: float | None, centroids_history: Tensor) -> bool:
        if tolerance is None:
            stop_flag = (self.centroids == centroids_history[-1, :, :]).all()
        else:
            stop_flag = torch.isclose(
                input=self.centroids,
                other=centroids_history[-1, :, :],
                atol=tolerance,
            ).all()
        stop_flag: bool = stop_flag.item()
        return stop_flag

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"


if __name__ == "__main__":
    def expt_cluster_2D():
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from src.utils.data.synthetic import get_gaussian_clusters_2D

        NUM_CLUSTERS = 4
        K = 5
        COLORS = ["red", "blue", "green", "magenta", "cyan"]
        PLOT_STEP = 0.01

        X_train = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)[0]
        X_test = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)[0]

        # Init
        kmeans = KMeansClassifier(K=K)
        centroids, logs = kmeans.fit(X_train=X_train, num_epochs=20)
        kmeans.evaluate(X_train)
        kmeans.evaluate(X_test)

        # Plot all centroids and examples
        fig, ax = plt.subplots()
        ax.set_title(f"K-means clustering (K={K}) with {NUM_CLUSTERS} given clusters")
        for k in torch.arange(kmeans.K):
            # Centroids' path
            ax.plot(
                logs["centroids_history"][:, k, 0],
                logs["centroids_history"][:, k, 1],
                color=COLORS[k],
                linestyle="dashed",
            )
            ax.scatter(
                logs["centroids_history"][:, k, 0],
                logs["centroids_history"][:, k, 1],
                color=COLORS[k],
                marker="x",
            )
            # Training data
            ax.scatter(
                X_train[logs["pred"].squeeze() == k, 0],
                X_train[logs["pred"].squeeze() == k, 1],
                color=COLORS[k],
                alpha=0.5,
                s=2,
                zorder=100,
            )

        # Voronoi diagram for centroids
        ptp_X = X_train.max(dim=0)[0] - X_train.min(dim=0)[0]
        plot_x1 = torch.arange(
            start=(X_train[:, 0].min() - 0.2 * ptp_X[0]).item(),
            end=(X_train[:, 0].max() + 0.2 * ptp_X[1]).item(),
            step=PLOT_STEP,
        )
        plot_x2 = torch.arange(
            start=(X_train[:, 1].min() - 0.2 * ptp_X[0]).item(),
            end=(X_train[:, 1].max() + 0.2 * ptp_X[1]).item(),
            step=PLOT_STEP,
        )
        x1, x2 = torch.meshgrid([plot_x1 - PLOT_STEP / 2, plot_x2 - PLOT_STEP / 2])
        x = torch.cat([x1.flatten().unsqueeze(dim=1), x2.flatten().unsqueeze(dim=1)], dim=1)
        plot_yhat = kmeans(x).reshape([plot_x1.size()[0], plot_x2.size()[0]])

        ax.pcolormesh(
            x1,
            x2,
            plot_yhat,
            cmap=ListedColormap(COLORS[0 : kmeans.K]),
            alpha=0.3,
            shading="auto",
        )
        fig.savefig("./logs/models_prototypes_kmeans_cluster2d.png", dpi=300, bbox_inches="tight")

    def expt_agnews():
        from datasets import DatasetDict

        dataset = DatasetDict.load_from_disk(
            dataset_dict_path="./datasets_preembed/agnews/emb-tfidf-agnews-train-3072d"
        ).with_format("torch")

        # Convert to 2D for visualization
        X_train: Tensor = dataset["train"].shuffle()["embedding"][:10000]
        X_test: Tensor = dataset["test"].shuffle()["embedding"][:10000]

        for n_p in [1000, 500, 200, 100, 50, 20, 10]:
            print(f"Running KMeans with {n_p} prototypes")
            kmeans = KMeansClassifier(K=n_p)
            centroids, logs = kmeans.fit(X_train=X_train, num_epochs=100)
            print(kmeans.evaluate(X_train))
            print(kmeans.evaluate(X_test))

    expt_cluster_2D()
    expt_agnews()
