import torch
from torch import Tensor


def get_gaussian_clusters_2D(
    num_clusters: int, sigma_diag: float = 0.2, radius: float = 1, num_examples: int = 600
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Generate several 2D Gaussian clusters around (0, 0)..

    Args:
    + `num_clusters`: Number of Gaussian clusters.
    + `sigma_diag`: Diagonal of covariance matrix of each Gaussian cluster. Higher means noisier. \
        Defaults to `0.2`.
    + `radius`: Radius of each Gaussian cluster. Defaults to `1`.
    + `num_examples`: Total number of examples to generate. Defaults to `600`.

    Returns:
    + A tuple of `(inputs, labels, metadata)`.
    """
    # Mu and Sigma for Gaussian distributions
    pi = torch.acos(torch.zeros(1)).item() * 2
    angle = torch.arange(num_clusters) * 2 * pi / num_clusters
    mu = radius * torch.cat(
        tensors=[
            torch.cos(angle).unsqueeze(dim=1),
            torch.sin(angle).unsqueeze(dim=1),
        ],
        dim=1,
    )
    sigma = sigma_diag * torch.eye(2)
    # Sample from Gaussian distributions
    examples_per_cluster = round(num_examples / num_clusters)
    X = torch.zeros(size=[0, 2])
    y = torch.zeros(size=[0], dtype=torch.int64)
    for k in range(num_clusters):
        gaussian_k = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mu[k, :],
            covariance_matrix=sigma,
        )
        X = torch.cat(tensors=[X, gaussian_k.sample([examples_per_cluster])], dim=0)
        y = torch.cat(tensors=[y, torch.tensor([k] * examples_per_cluster)], dim=0)

    # Shuffle data
    shuffle_index = torch.randperm(X.shape[0])
    X, y = X[shuffle_index], y[shuffle_index]

    metadata = {"mu": mu, "sigma": sigma}
    return X, y, metadata
