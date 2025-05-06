from typing import Optional, Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import lightning as L


class LinearScheduler(nn.Module):
    timesteps: Tensor

    def __init__(self, start: float = 1e-4, end: float = 0.02, num_steps=1000):
        super().__init__()
        self.start = start
        self.end = end
        self.num_steps = num_steps

        self.register_buffer(
            name="timesteps",
            tensor=torch.linspace(start=start, end=end, steps=num_steps),
        )

    def forward(self, t: Tensor) -> Tensor:
        t = self.timesteps[t]
        return t


class ForwardDiffusion(L.LightningModule):
    def __init__(self, scheduler: LinearScheduler, num_steps: Optional[int] = 1000):
        if num_steps is None:
            assert (
                scheduler.num_steps is not None
            ), "Either provide a scheduler with num_steps or specify num_steps directly."
            num_steps = scheduler.num_steps

        super().__init__()
        self.scheduler = scheduler
        self.num_steps = num_steps

        self.betas = self.scheduler.timesteps.clone()
        self.alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to("cuda")
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(
            "cuda"
        )

    def q_sample(self, x_start, t, noise):
        x = (
            self.sqrt_alphas_cumprod[t][:, None] * x_start
            + self.sqrt_one_minus_alphas_cumprod[t][:, None] * noise
        )
        return x


class DenoiseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = [nn.Linear(in_features=input_dim + 1, out_features=hidden_dims[0])]
        layers.append(nn.LeakyReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            )
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.float().unsqueeze(1) / 1000.0  # Normalize timestep
        x_t = torch.cat([x, t], dim=1)
        out = self.layers(x_t)
        return out


class DDPM(L.LightningModule):
    """Denoising Diffusion Probabilistic Models"""

    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        num_steps: int = 1000,
        opt_kw={"Class": torch.optim.Adam, "kwargs": {"lr": 1e-3}},
    ):
        super().__init__()
        self.model = model
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.opt_kw = opt_kw

        self.diffusion = ForwardDiffusion(
            scheduler=LinearScheduler(num_steps=num_steps),
            num_steps=num_steps,
        )

    def training_step(self, batch, batch_idx):
        x = batch[0]
        batch_size = x.shape[0]

        t = torch.randint(
            low=0, high=self.num_steps, size=[batch_size], device=self.device
        )
        noise = torch.randn_like(x)
        x_noisy = self.diffusion.q_sample(x, t, noise)
        noise_pred = self.model(x_noisy, t)

        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = self.opt_kw["Class"](self.model.parameters(), **self.opt_kw["kwargs"])
        return opt

    def sample_step(self, x: Tensor, t: int) -> Tensor:
        batch_size = x.shape[0]

        t_batch = t * torch.ones(
            size=[batch_size], dtype=torch.long, device=self.device
        )

        beta = self.diffusion.betas[t]
        alpha = self.diffusion.alphas[t]
        alpha_bar = self.diffusion.alphas_cumprod[t]

        noise_pred = self.model(x, t_batch)

        coef1 = 1 / torch.sqrt(alpha)
        coef2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        x = coef1 * (x - coef2 * noise_pred)

        if t > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x += sigma * z
        return x

    @torch.no_grad()
    def sample(
        self, x0: Optional[Tensor] = None, num_samples: Optional[int] = None
    ) -> Tensor:
        if (x0 is None) and (num_samples is None):
            raise ValueError("Either x0 or num_samples must be provided.")

        if x0 is None:
            x = torch.normal(
                mean=0, std=1, size=[num_samples, self.input_dim], device=self.device
            )
        else:
            x = x0.clone()

        for t in reversed(range(self.num_steps)):
            x = self.sample_step(x, t)
        return x


if __name__ == "__main__":
    import os, sys

    repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
    assert (
        os.path.basename(repo_path) == "textdd"
    ), "Wrong parent folder. Please change to 'textdd'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    import matplotlib.pyplot as plt

    from utils.data import get_dataloader, TensorPool
    from utils.data.synthetic import get_gaussian_clusters_2D

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

        model = DenoiseMLP(input_dim=2, hidden_dims=[8, 8], output_dim=2)
        ddpm = DDPM(model=model, input_dim=2)
        trainer = L.Trainer(max_epochs=10, accelerator="auto", log_every_n_steps=10)
        trainer.fit(model=ddpm, train_dataloaders=dataloader["train"])

        with torch.inference_mode():
            x_rand = ddpm.sample(num_samples=1000)

            x0_circ = torch.normal(mean=0, std=1, size=[1000, 2], device=ddpm.device)
            x0_circ = x0_circ / x0_circ.norm(dim=1, keepdim=True)
            x_circ = ddpm.sample(x0=x0_circ)

        # Plot all centroids and examples
        fig, ax = plt.subplots()
        ax.set_title(f"DDPM with {K} clusters")
        # fmt: off
        ax.scatter(
            X_train[:, 0], X_train[:, 1],
            color="black", alpha=0.1, s=2, label="train",
        )
        ax.scatter(
            x_rand[:, 0], x_rand[:, 1],
            color="tab:blue", s=2, label="ddpm",
        )
        ax.scatter(
            x_circ[:, 0], x_circ[:, 1],
            color="tab:red", s=2, label="ddpm-circ",
        )
        # fmt: on
        ax.legend()
        fig.savefig("./logs/ddpm.png", dpi=300, bbox_inches="tight")

    expt_cluster_2D()
