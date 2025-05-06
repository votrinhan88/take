from typing import Sequence
import torch
from torch import nn, Tensor


class GumbelSoftmax(nn.Module):
    def __init__(
        self,
        tau: float = 1,
        hard: bool = False,
        eps: float = 1e-10,
        dim: int = -1,
    ):
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.hard = hard
        self.eps = eps
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        out = torch.nn.functional.gumbel_softmax(
            logits=inputs,
            tau=self.tau,
            hard=self.hard,
            eps=self.eps,
            dim=self.dim,
        )
        return out


class Prototypes(torch.nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        dim: int,
    ):
        super(Prototypes, self).__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim

        self.log_pi = nn.Parameter(
            data=torch.normal(mean=0, std=1, size=[num_prototypes]).log_softmax(dim=0),
            requires_grad=True,
        )
        self.mu = nn.Parameter(
            data=torch.normal(mean=0, std=1, size=[num_prototypes, dim]),
            requires_grad=True,
        )
        self.log_sigma_chol = nn.Parameter(
            data=torch.zeros(size=[num_prototypes, dim]),
            requires_grad=True,
        )

        self.gumbel_softmax = GumbelSoftmax(tau=1.0, hard=False, eps=1e-10, dim=1)

    def forward(self, batch_size: Tensor) -> Tensor:
        log_pi = self.log_pi.log_softmax(dim=0).repeat(repeats=[batch_size, 1])
        p_cluster: Tensor = self.gumbel_softmax(log_pi)
        cluster = p_cluster.argmax(dim=1)

        mu = self.mu[cluster]
        sigma_chol = self.log_sigma_chol[cluster].exp()
        epsilon = torch.randn_like(sigma_chol)
        out = mu + epsilon * sigma_chol
        return out

    def get_circumference(self, batch_size: Tensor) -> Tensor:
        cluster = torch.randint(low=0, high=self.num_prototypes, size=[batch_size])

        mu = self.mu[cluster]
        sigma_chol = self.log_sigma_chol[cluster].exp()
        # Normalize to unit length: exactly one sigma away from mean
        epsilon = torch.randn_like(sigma_chol)
        epsilon = epsilon / epsilon.norm(dim=1, keepdim=True)
        out = mu + epsilon * sigma_chol
        return out

    def regularize_pi(self) -> Tensor:
        self.log_pi.data = self.log_pi.log_softmax(dim=0)


class Scorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]):
        super(Scorer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = [
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.ReLU(),
        ]
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            )
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        score = self.layers(input)
        return score




if __name__ == "__main__":
    import os, sys

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert (
        os.path.basename(repo_path) == "textdd"
    ), "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    import matplotlib.pyplot as plt
    from utils.data.synthetic import get_gaussian_clusters_2D
    from utils.data import get_dataloader, TensorPool
    from models.prototypes.kmeans import KMeansClassifier
    from models.prototypes.gmm import GaussianMixtureModel

    # Example usage
    num_prototypes = 5
    dim = 2

    X_real = get_gaussian_clusters_2D(
        num_clusters=5,
        num_examples=5000,
        radius=2,
    )[0]
    dataloader = get_dataloader(
        dataset={"train": TensorPool(tensors=[X_real])},
        batch_size=200,
        shuffle=True,
    )

    # Init
    kmeans = KMeansClassifier(K=num_prototypes)
    logs = kmeans.fit(X_train=X_real, num_epochs=20)
    kmeans.evaluate(X_real)

    gmm = GaussianMixtureModel(K=num_prototypes)
    gmm.fit(X_train=X_real, means=kmeans.centroids.detach(), num_epochs=50)

    prototypes = Prototypes(num_prototypes=num_prototypes, dim=dim)
    # prototypes.mu.data = kmeans.centroids.detach()

    scorer = Scorer(input_dim=dim, hidden_dims=[10, 10])

    opt_G = torch.optim.SGD(
        prototypes.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
    )
    opt_S = torch.optim.SGD(
        scorer.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
    )

    loss_fn = nn.L1Loss()

    fig, ax = plt.subplots()
    ax.scatter(
        X_real[:, 0].numpy(),
        X_real[:, 1].numpy(),
        label="Real",
        alpha=0.5,
        color="gray",
        s=2,
    )
    with torch.inference_mode():
        x_fake: Tensor = prototypes(batch_size=1000)
        x_circ: Tensor = prototypes.get_circumference(batch_size=2000)
    ax.scatter(
        x_fake[:, 0].numpy(),
        x_fake[:, 1].numpy(),
        label="Fake",
        alpha=0.5,
        color="red",
        s=2,
    )
    ax.scatter(
        x_circ[:, 0].numpy(),
        x_circ[:, 1].numpy(),
        label="1-sigma",
        alpha=0.5,
        color="green",
        s=2,
    )
    ax.scatter(
        kmeans.centroids[:, 0].numpy(),
        kmeans.centroids[:, 1].numpy(),
        label="k-means",
        alpha=0.5,
        color="blue",
        s=5,
    )
    ax.scatter(
        gmm.means[:, 0].numpy(),
        gmm.means[:, 1].numpy(),
        label="gmm",
        alpha=0.5,
        color="yellow",
        s=5,
    )
    ax.legend()
    ax.set(xlim=[-8, 8], ylim=[-8, 8])
    fig.savefig("./logs/prototypes_example_before.png")

    # mu = prototypes.mu.clone().detach()
    # log_pi = prototypes.log_pi.clone().detach()
    # log_sigma_chol = prototypes.log_sigma_chol.clone().detach()

    # for epoch in range(50):
    #     for batch in dataloader['train']:
    #         batch_size = batch[0].shape[0]
    #         x_real = batch[0]
    #         x_fake = prototypes(batch_size=batch_size)

    #         score_real:Tensor = scorer(x_real)
    #         score_fake:Tensor = scorer(x_fake)
    #         loss_G:Tensor = -score_fake.mean()

    #         opt_G.zero_grad()
    #         loss_G.backward(retain_graph=True)
    #         opt_G.step()

    #         score_real:Tensor = scorer(x_real)
    #         score_fake:Tensor = scorer(x_fake)
    #         loss_S:Tensor = (score_fake - score_real).mean()

    #         opt_S.zero_grad()
    #         loss_S.backward()
    #         opt_S.step()
    #         prototypes.regularize_pi()

    #     print(f"Epoch {epoch}, sr: {score_real.mean().item()}, sf: {score_fake.mean().item()}, loss_S: {loss_S.item()}, loss_G: {loss_G.item()}")

    # fig, ax = plt.subplots()
    # ax.scatter(X_real[:, 0].numpy(), X_real[:, 1].numpy(), label='Real', alpha=0.5, color='gray', s=2)
    # with torch.inference_mode():
    #     x_fake:Tensor = prototypes(batch_size=1000)
    #     x_circ:Tensor = prototypes.get_circumference(batch_size=2000)
    # ax.scatter(x_fake[:, 0].numpy(), x_fake[:, 1].numpy(), label='Fake', alpha=0.5, color='red', s=2)
    # ax.scatter(x_circ[:, 0].numpy(), x_circ[:, 1].numpy(), label='1-sigma', alpha=0.5, color='green', s=2)
    # ax.legend()
    # ax.set(xlim=[-8, 8], ylim=[-8, 8])
    # fig.savefig("./logs/prototypes_example_after.png")

    # print("Before")
    # print("mu:", mu)
    # print("log_pi:", log_pi)
    # print("log_sigma_chol:", log_sigma_chol)

    # print("After")
    # print("prototypes.mu:", prototypes.mu)
    # print("prototypes.log_pi:", prototypes.log_pi)
    # print("prototypes.log_sigma_chol:", prototypes.log_sigma_chol)

    print()
