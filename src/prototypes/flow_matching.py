from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
import lightning as L
from torchdiffeq import odeint


class VelocityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers: list[nn.Module] = [
            nn.Linear(in_features=input_dim + 1, out_features=hidden_dims[0])
        ]
        layers.append(nn.LeakyReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.view(-1, 1)
        x_t = torch.cat([x, t], dim=1)
        out = self.layers(x_t)
        return out


class FlowMatching(L.LightningModule):
    """Flow Matching for Generative Modeling"""

    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        opt_kw={"Class": torch.optim.Adam, "kwargs": {"lr": 1e-3}},
        loss_fn=nn.MSELoss(),
    ):
        super().__init__()
        self.model = model
        self.input_dim = input_dim
        self.opt_kw = opt_kw
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        x1 = batch[0]
        batch_size = x1.shape[0]

        x0 = torch.randn_like(x1)
        t = torch.rand(size=[batch_size], device=self.device)
        x_t = (1 - t).unsqueeze(dim=1) * x0 + t.unsqueeze(dim=1) * x1
        v_gt = x1 - x0
        v_pred = self.model(x_t, t)

        loss = self.loss_fn(input=v_pred, target=v_gt)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = self.opt_kw["Class"](self.model.parameters(), **self.opt_kw["kwargs"])
        return opt

    @torch.no_grad()
    def sample(
        self,
        x0: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
        steps: int = 100,
    ) -> Tensor:
        if (x0 is None) and (num_samples is None):
            raise ValueError("Either x0 or num_samples must be provided.")

        if x0 is None:
            x = torch.normal(mean=0, std=1, size=[num_samples, self.input_dim], device=self.device)
        else:
            x = x0.clone()

        def ode_func(t: Tensor, x: Tensor) -> Tensor:
            t = t * torch.ones(size=[x.shape[0]], device=self.device)
            x_next = self.model(x, t)
            return x_next

        ts = torch.linspace(0, 1, steps, device=self.device)
        x_t: Tensor = odeint(ode_func, x, ts, method="rk4")
        return x_t[-1]


if __name__ == "__main__":
    import os, sys

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert os.path.basename(repo_path) == "textdd", "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from models.prototypes.kmeans import KMeansClassifier
    from utils.data.synthetic import get_gaussian_clusters_2D
    from utils.data import get_dataloader, TensorPool

    def expt_cluster_2D():
        K = 4

        X_train = get_gaussian_clusters_2D(
            num_clusters=K, num_examples=10000, radius=2, sigma_diag=0.1
        )[0]
        dataset = {"train": TensorPool(tensors=[X_train])}
        dataloader = get_dataloader(
            dataset=dataset,
            batch_size=32,
            shuffle={"train": True, "test": False},
        )

        model = VelocityMLP(input_dim=2, hidden_dims=[8, 8], output_dim=2)
        flow = FlowMatching(model=model, input_dim=2)
        trainer = L.Trainer(max_epochs=10, accelerator="auto", log_every_n_steps=10)
        trainer.fit(model=flow, train_dataloaders=dataloader["train"])

        with torch.inference_mode():
            x_rand = flow.sample(num_samples=1000)

            x0_circ = torch.normal(mean=0, std=1, size=[1000, 2], device=flow.device)
            x0_circ = x0_circ / x0_circ.norm(dim=1, keepdim=True)
            x_circ_0 = flow.sample(x0=x0_circ * 0.25)
            x_circ_1 = flow.sample(x0=x0_circ * 0.5)
            x_circ_2 = flow.sample(x0=x0_circ)
            x_circ_3 = flow.sample(x0=x0_circ * 2)
            x_circ_4 = flow.sample(x0=x0_circ * 4)

        # Plot all centroids and examples
        fig, ax = plt.subplots()
        ax.set_title(f"FLow Matching with {K} clusters")
        # fmt: off
        ax.scatter(
            X_train[:, 0], X_train[:, 1],
            color="black", alpha=0.1, s=2, label="train",
        )
        ax.scatter(
            x_rand[:, 0], x_rand[:, 1],
            color="tab:blue", s=2, label="flow",
        )
        ax.scatter(
            x_circ_0[:, 0], x_circ_0[:, 1],
            color="tab:red", s=2, label="flow-circ-0.25std",
        )
        ax.scatter(
            x_circ_1[:, 0], x_circ_1[:, 1],
            color="tab:red", s=2, label="flow-circ-0.5std",
        )
        ax.scatter(
            x_circ_2[:, 0], x_circ_2[:, 1],
            color="tab:red", s=2, label="flow-circ-std",
        )
        ax.scatter(
            x_circ_3[:, 0], x_circ_3[:, 1],
            color="tab:red", s=2, label="flow-circ-2std",
        )
        ax.scatter(
            x_circ_4[:, 0], x_circ_4[:, 1],
            color="tab:red", s=2, label="flow-circ-4std",
        )


        # fmt: on
        ax.legend()
        fig.savefig("./logs/flow.png", dpi=300, bbox_inches="tight")

    expt_cluster_2D()
