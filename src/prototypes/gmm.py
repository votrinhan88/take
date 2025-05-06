import os
import sys

repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import lightning as L
import tqdm.auto as tqdm

from models.prototypes.kmeans import KMeansClassifier
from models.prototypes.kmeans import KMeansPlusPlus


class GaussianMixtureModel(L.LightningModule):
    """The Expectation-Maximization algorithm with Gaussian Mixture.

    Args:
    + `X_train`: Input data of shape [N * d].
    + `K`: Number of Gaussians components.
    + `mus`: Mean of Gaussians, must be of shape [K * d]. Defaults to `None`,   \
        leave to initialize from a standard normal distribution.
    + `Sigmas`: Covariance matrix of Gaussians, must be of shape [K * d * d].   \
        Defaults to `None`, leave to initialize as the covariance matrix of     \
        `X_train` for each Gaussian.
    + `pis`: Prior of Gaussians, must be of shape [K]. Defaults to `None`, leave\
        to initialize as `1/K` for each Gaussian.
    + `trainable_mus`: Flag to set `mus` trainable. Defaults to `True`.
    + `trainable_Sigmas`: Flag to set `Sigmas` trainable. Defaults to `True`.
    + `trainable_pis`: Flag to set `pis` trainable. Defaults to `True`.
    """

    def __init__(
        self,
        K: int,
        reg_covar: float = 1e-6,
        verbose: bool = True,
    ):
        super().__init__()
        self.K: int = K
        self.reg_covar: float = reg_covar
        self.verbose: bool = verbose

    def fit(
        self,
        X_train: Tensor,
        num_epochs: int = 100,
        tol: float = 1e-3,
        means: list[float] | str = "kmeans",
        covs: list[float] | str = "full",
        weights: list[float] | str = "uniform",
        trainable_means: bool = True,
        trainable_covs: bool = True,
        trainable_weights: bool = True,
    ) -> dict:
        self.prepare_params(X_train=X_train, means=means, covs=covs, weights=weights)

        if self.verbose:
            pbar = tqdm.trange(num_epochs, desc="GMM", leave=False)
        else:
            pbar = range(num_epochs)

        metrics = {"loss_l1": [], "loss_mse": [], "log_likelihood": []}
        logllh_prev = -float("inf")
        for epoch in pbar:
            self.gaussians = self.get_gaussians(K=self.K, means=self.means, covs=self.covs)

            # Expectation (E) step: compute responsibilities
            log_posterior: Tensor = self(X_train)
            resp = log_posterior.softmax(dim=1)

            # Maximization (M) step: update parameters
            for k in range(self.K):
                mean_k = (resp[:, [k]] * X_train).sum(dim=0) / resp[:, [k]].sum(dim=0)

                if trainable_means:
                    self.means[k] = mean_k
                if trainable_covs:
                    cov_k = (
                        (resp[:, [k]] * (X_train - mean_k)).t()
                        @ (X_train - mean_k)
                        / resp[:, [k]].sum(dim=0)
                    )
                    # Regularize covariance
                    cov_k = cov_k + torch.eye(cov_k.shape[0], device=cov_k.device) * self.reg_covar
                    self.covs[k] = cov_k
                if trainable_weights:
                    weight_k = resp[:, [k]].mean(dim=0)
                    self.weights[k] = weight_k

            # Log-likelihood for convergence
            logllh = torch.logsumexp(log_posterior, dim=1).mean(dim=0).item()
            if abs(logllh - logllh_prev) < tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}, ll={logllh:.4f}")
                break
            logllh_prev = logllh

            # Log metrics
            metrics_epoch = {"log_likelihood": logllh}
            metrics_epoch.update(self.evaluate(X_train))
            for k in metrics_epoch.keys():
                metrics[k].append(metrics_epoch[k])
            if self.verbose:
                pbar.set_postfix(metrics_epoch)  # type: ignore

        # Update Gaussians
        self.gaussians = self.get_gaussians(K=self.K, means=self.means, covs=self.covs)
        log = {
            "pred": log_posterior.argmax(dim=1),
            "means": self.means,
            "covs": self.covs,
            "weights": self.weights,
            **metrics,
        }
        return log

    def prepare_params(
        self,
        X_train: Tensor,
        means: list[float] | str = "kmeans",
        covs: list[float] | str = "full",
        weights: list[float] | str = "uniform",
    ):
        self.num_samples = X_train.shape[0]
        self.dim = X_train.shape[1]

        # Means initialization
        if isinstance(means, str):
            if means not in ["kmeans", "kmeans++", "random"]:
                raise ValueError(f"Means initialization '{means}' not supported.")

            if means == "kmeans++":
                idx = KMeansPlusPlus(K=self.K, verbose=False).fit(X_train)
                self.means = X_train[idx]
            elif means == "kmeans":
                kmeans = KMeansClassifier(K=self.K, verbose=False).fit(X_train)
                self.means = kmeans[0]
            else:
                self.means = X_train[torch.randperm(X_train.shape[0])][0 : self.K]
        else:
            self.means = torch.tensor(means)

        # Covariances initialization
        if isinstance(covs, str):
            if covs not in ["full", "diag", "spherical"]:
                raise ValueError(f"Covariances initialization '{covs}' not supported.")

            resp = torch.ones(size=[X_train.shape[0], self.K], device=X_train.device) / self.K
            nk = resp.sum(dim=0) + 10 * torch.finfo(X_train.dtype).eps
            means = (resp.T @ X_train) / nk[:, None]
            gauss_args = [resp, X_train, nk, means, self.reg_covar]
            if covs == "full":
                self.covs = self._estimate_gaussian_covariances_full(*gauss_args)
            elif covs == "diag":
                self.covs = self._estimate_gaussian_covariances_diag(*gauss_args)
            elif covs == "spherical":
                self.covs = self._estimate_gaussian_covariances_spherical(*gauss_args)
        else:
            covs: Tensor = torch.tensor(covs)
            assert covs.shape == (self.K, self.dim, self.dim), "Cov must be of shape (K, D, D)."
            self.covs = covs

        # Weights initialization
        if isinstance(weights, str):
            if weights not in ["uniform"]:
                raise ValueError(f"Weights initialization '{weights}' not supported.")

            if weights == "uniform":
                self.weights = (
                    torch.ones(size=[self.K], dtype=X_train.dtype, device=X_train.device) / self.K
                )
        else:
            self.weights = torch.tensor(weights)
            self.weights = self.weights / self.weights.sum(dim=0, keepdim=True)

    @staticmethod
    def get_gaussians(K: int, means: Tensor, covs: Tensor) -> list[MultivariateNormal]:
        gaussians = [MultivariateNormal(loc=means[k], covariance_matrix=covs[k]) for k in range(K)]
        return gaussians

    def forward(self, input: Tensor) -> Tensor:
        """The Expectation step in the EM algorithm. Return log-probabilities."""
        log_likelihood = torch.zeros(size=[input.shape[0], self.K])
        for k in range(self.K):
            log_likelihood[:, k] = self.gaussians[k].log_prob(input)
        log_prior = self.weights.log().unsqueeze(dim=0)
        log_posterior = log_prior + log_likelihood
        return log_posterior

    def evaluate(self, X_test: Tensor) -> dict[str, float]:
        total = 0
        loss_l1 = 0
        loss_mse = 0
        loss_fn_l1 = nn.L1Loss(reduction="sum")
        loss_fn_mse = nn.MSELoss(reduction="sum")

        with torch.inference_mode():
            log_posterior: Tensor = self(X_test)
            total += X_test.shape[0]

            pred = log_posterior.argmax(dim=1)
            assigned_gaussian = self.means[pred]

            loss_l1 += loss_fn_l1(assigned_gaussian, X_test).item()
            loss_mse += loss_fn_mse(assigned_gaussian, X_test).item()

        loss_l1 = loss_l1 / total
        loss_mse = loss_mse / total
        results = {
            "loss_l1": loss_l1,
            "loss_mse": loss_mse,
        }
        return results

    @staticmethod
    def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
        n_components, n_features = means.shape
        cov = torch.empty((n_components, n_features, n_features), dtype=X.dtype, device=X.device)
        for k in range(n_components):
            diff = X - means[k]
            cov[k] = (resp[:, k][:, None] * diff).T @ diff / nk[k]
            cov[k].view(-1)[:: n_features + 1] += reg_covar
        return cov

    @staticmethod
    def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
        avg_X2 = (resp.T @ (X * X)) / nk[:, None]
        avg_means2 = means**2
        cov = avg_X2 - avg_means2 + reg_covar
        return cov

    @staticmethod
    def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
        avg_X2 = (resp.T @ (X * X)) / nk[:, None]
        avg_means2 = means**2
        cov = avg_X2 - avg_means2 + reg_covar
        cov = cov.mean(1)
        return cov
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, reg_covar={self.reg_covar})"


if __name__ == "__main__":
    import os, sys

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from utils.data.synthetic import get_gaussian_clusters_2D

    NUM_CLUSTERS = 4
    K = 5
    COLORS = [
        "red",
        "blue",
        "green",
        "magenta",
        "cyan",
        "yellow",
        "black",
        "tab:blue",  # fmt: off
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",  # fmt: off
        "tab:gray",
        "tab:olive",
        "tab:cyan",  # fmt: off
    ]
    PLOT_STEP = 0.01

    X_train = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)[0]
    X_test = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS)[0]

    # Init
    gmm = GaussianMixtureModel(K=K)
    logs = gmm.fit(X_train=X_train, num_epochs=20)
    gmm.evaluate(X_train)
    gmm.evaluate(X_test)

    # Plot all centroids and examples
    fig, ax = plt.subplots()
    ax.set_title(f"GMM clustering (K = {K}) with {NUM_CLUSTERS} given clusters")
    for k in torch.arange(gmm.K):
        # Centroids' path
        ax.scatter(
            gmm.means[:, 0],
            gmm.means[:, 1],
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
    plot_x1 = torch.arange(  # ty: ignore
        start=X_train[:, 0].min() - 0.2 * ptp_X[0],
        end=X_train[:, 0].max() + 0.2 * ptp_X[1],
        step=PLOT_STEP,
    )
    plot_x2 = torch.arange(  # ty: ignore
        start=X_train[:, 1].min() - 0.2 * ptp_X[0],
        end=X_train[:, 1].max() + 0.2 * ptp_X[1],
        step=PLOT_STEP,
    )
    x1, x2 = torch.meshgrid([plot_x1 - PLOT_STEP / 2, plot_x2 - PLOT_STEP / 2])
    x = torch.cat([x1.flatten().unsqueeze(dim=1), x2.flatten().unsqueeze(dim=1)], dim=1)
    plot_yhat = gmm(x).argmax(dim=1).reshape([plot_x1.shape[0], plot_x2.shape[0]])
    # Plot covariance ellipses
    ax.contourf(
        x1,
        x2,
        plot_yhat,
        cmap=ListedColormap(COLORS[0 : gmm.K]),
        alpha=0.3,
        # shading="auto",
    )

    ax.pcolormesh(
        x1,
        x2,
        plot_yhat,
        cmap=ListedColormap(COLORS[0 : gmm.K]),
        alpha=0.3,
        shading="auto",
    )
    fig.savefig("./logs/models_prototypes_gmm_cluster2d.png", dpi=300, bbox_inches="tight")
