# Change path
import os, sys

repo_path = os.path.abspath(os.path.join(__file__, "../../.."))
assert (
    os.path.basename(repo_path) == "textdd"
), "Wrong parent folder. Please change to 'textdd'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)


from typing import Sequence
import torch
from torch import nn, Tensor
import lightning as L

from models.modules.losses import get_reduction_fn


class EncoderDense(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super(EncoderDense, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = [nn.Linear(in_features=input_dim, out_features=hidden_dims[0])]
        layers.append(nn.GELU())
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            )
            layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers[0:-1])

        self.head_mu = nn.Sequential(
            nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim),
        )
        self.head_sigma = nn.Sequential(
            nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim),
        )

    def forward(self, input: Tensor) -> dict[str, Tensor]:
        x = self.layers(input)
        mu = self.head_mu(x)
        log_sigma: Tensor = self.head_sigma(x)
        sigma = log_sigma.exp()
        return {"mu": mu, "sigma": sigma}


class DecoderDense(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super(DecoderDense, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        layers = [nn.Linear(in_features=latent_dim, out_features=hidden_dims[0])]
        layers.append(nn.LeakyReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            )
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        x = self.layers(input)
        return x


class KLDivGuassiansLoss(nn.modules.loss._Loss):
    def __init__(
        self, mu_target: Tensor, sigma_target: Tensor, reduction: str = "mean"
    ):
        super().__init__()
        self.mu_target = mu_target
        self.sigma_target = sigma_target
        self.reduction = reduction

        self.reduction_fn = get_reduction_fn(reduction=reduction)

    def forward(self, mu: Tensor, sigma: Tensor) -> Tensor:
        loss = (
            +torch.log(self.sigma_target / sigma)
            + (sigma**2 + (mu - self.mu_target) ** 2) / (2 * self.sigma_target**2)
            - 1 / 2
        ).sum(dim=1)
        loss = self.reduction_fn(loss)
        return loss


class KLDivGuassiansStandardLoss(nn.modules.loss._Loss):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

        self.reduction_fn = get_reduction_fn(reduction=reduction)

    def forward(self, mu: Tensor, sigma: Tensor) -> Tensor:
        loss = 0.5 * (sigma**2 + mu**2 - 2 * sigma.log() - 1).sum(dim=1)
        loss = self.reduction_fn(loss)
        return loss


class VAE(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_dim: int,
        latent_dim: int,
        opt_E_kw: dict,
        opt_D_kw: dict,
        loss_fn_reconstruct: nn.Module = nn.MSELoss(),
        loss_fn_latent: nn.Module = KLDivGuassiansStandardLoss(),
        coeff_rc: float = 1.0,
        coeff_lt: float = 1.0,
    ):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.opt_E_kw = opt_E_kw
        self.opt_D_kw = opt_D_kw
        self.loss_fn_reconstruct = loss_fn_reconstruct
        self.loss_fn_latent = loss_fn_latent
        self.coeff_rc = coeff_rc
        self.coeff_lt = coeff_lt
        
        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_E = self.opt_E_kw["classifier"]["Class"](
            self.encoder.parameters(), **self.opt_E_kw["classifier"]["kwargs"]
        )
        opt_D = self.opt_D_kw["classifier"]["Class"](
            self.encoder.parameters(), **self.opt_D_kw["classifier"]["kwargs"]
        )
        return opt_E, opt_D

    def training_step(self, batch, batch_idx) -> Tensor:
        # Unpack data
        x: Tensor = batch[0]
        batch_size = x.shape[0]
        
        opt_E, opt_D = self.optimizers()

        out_E = self.encoder(x)
        mu, sigma = out_E["mu"], out_E["sigma"]
        epsilon = torch.normal(
            mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device
        )
        z = mu + sigma * epsilon
        x_r = self.decoder(z)

        # loss_rc = self.loss_fn_reconstruct(input=x_r, target=x)
        loss_rc = ((x_r - x)**2).sum(dim=1).mean(dim=0)
        loss_lt = self.loss_fn_latent(mu=mu, sigma=sigma)
        loss: Tensor = self.coeff_rc * loss_rc + self.coeff_lt * loss_lt

        opt_E.zero_grad()
        opt_D.zero_grad()
        self.manual_backward(loss)
        opt_E.step()
        opt_D.step()

        self.log_dict({"loss_rc": loss_rc, "loss_lt": loss_lt}, prog_bar=True)
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        # Unpack data
        x: Tensor = batch[0]
        batch_size = x.shape[0]

        out_E = self.encoder(x)
        mu, sigma = out_E["mu"], out_E["sigma"]
        epsilon = torch.normal(
            mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device
        )
        z = mu + sigma * epsilon
        x_r = self.decoder(z)

        loss_rc = self.loss_fn_reconstruct(input=x_r, target=x)
        loss_lt = self.loss_fn_latent(mu=mu, sigma=sigma)

        self.log_dict({"loss_rc": loss_rc, "loss_lt": loss_lt}, prog_bar=True)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]

        mu, sigma = self.encoder(x)
        epsilon = torch.normal(
            mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device
        )

        z = mu + sigma * epsilon
        x_r = self.decoder(z)
        return x_r

    def get_circumference(self, batch_size: Tensor, sigma: float = 1) -> Tensor:
        z = torch.normal(
            mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device
        )
        # Normalize to unit length: exactly one sigma away from mean
        z = sigma * z / z.norm(dim=1, keepdim=True)
        x_r = self.decoder(z)
        return x_r

    def fit(self, fit_kw, **L_trainer_kw):
        L_trainer = L.Trainer(**L_trainer_kw)
        L_trainer.fit(model=self, **fit_kw)


if __name__ == "__main__":
    def expt_cluster_2D():
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from utils.data.synthetic import get_gaussian_clusters_2D
        from models.prototypes.kmeans import KMeansClassifier
        from utils.data import get_dataloader, TensorPool

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

        X_train = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS, num_examples=10000, radius=2, sigma_diag=0.1)[0]
        X_test = get_gaussian_clusters_2D(num_clusters=NUM_CLUSTERS, num_examples=2000, radius=2, sigma_diag=0.1)[0]
        dataset = {
            'train': TensorPool(tensors=[X_train]),
            'test': TensorPool(tensors=[X_test]),
        }
        dataloader = get_dataloader(
            dataset=dataset,
            batch_size=32,
            shuffle={'train': True, 'test': False},
        )
        
        # Init
        kmeans = KMeansClassifier(K=K)
        logs = kmeans.fit(X_train=X_train, num_epochs=50)
        kmeans.evaluate(X_train)
        kmeans.evaluate(X_test)

        encoder = EncoderDense(input_dim=2, hidden_dims=[8, 8], latent_dim=2)
        decoder = DecoderDense(latent_dim=2, hidden_dims=[8, 8], output_dim=2)
        vae = VAE(
            encoder=encoder,
            decoder=decoder,
            input_dim=2,
            latent_dim=2,
            loss_fn_reconstruct=nn.L1Loss(),
            opt_E_kw={
                "classifier": {
                    "Class": torch.optim.Adam,
                    "kwargs": {"lr": 1e-3},#, "momentum": 0.9, "weight_decay": 5e-4},
                }
            },
            opt_D_kw={
                "classifier": {
                    "Class": torch.optim.Adam,
                    "kwargs": {"lr": 1e-3},#, "momentum": 0.9, "weight_decay": 5e-4},
                }
            },
            coeff_lt=0.003,
        )
        vae.fit(
            **{'max_epochs': 20},
            fit_kw={
                "train_dataloaders": dataloader["train"],
                "val_dataloaders": dataloader["test"],
            },
        )


        # Plot all centroids and examples
        fig, ax = plt.subplots()
        ax.set_title(f"K-means clustering (K = {K}) with {NUM_CLUSTERS} given clusters")
        for k in torch.arange(kmeans.K):
            # Centroids' path
            # ax.plot(
            #     logs["centroids_history"][:, k, 0],
            #     logs["centroids_history"][:, k, 1],
            #     color=COLORS[k],
            #     linestyle="dashed",
            # )
            ax.scatter(
                logs["centroids_history"][-1, k, 0],
                logs["centroids_history"][-1, k, 1],
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
        def plot_voronoi(X_train: Tensor, classifier, plot_step=PLOT_STEP):
            ptp_X = X_train.max(dim=0)[0] - X_train.min(dim=0)[0]
            plot_x1 = torch.arange(
                X_train[:, 0].min() - 0.2 * ptp_X[0],
                X_train[:, 0].max() + 0.2 * ptp_X[1],
                plot_step,
            )
            plot_x2 = torch.arange(
                X_train[:, 1].min() - 0.2 * ptp_X[0],
                X_train[:, 1].max() + 0.2 * ptp_X[1],
                plot_step,
            )
            x1, x2 = torch.meshgrid([plot_x1 - plot_step / 2, plot_x2 - plot_step / 2])
            x = torch.cat(
                tensors=[x1.flatten().unsqueeze(dim=1), x2.flatten().unsqueeze(dim=1)],
                dim=1,
            )
            yhat = classifier(x).reshape([plot_x1.size()[0], plot_x2.size()[0]])
            return {
                "x1": x1,
                "x2": x2,
                "yhat": yhat,
            }
        out_voronoi = plot_voronoi(X_train=X_train, classifier=kmeans)
        ax.pcolormesh(
            out_voronoi["x1"],
            out_voronoi["x2"],
            out_voronoi["yhat"],
            cmap=ListedColormap(COLORS[0 : kmeans.K]),
            alpha=0.3,
            shading="auto",
        )
        
        with torch.inference_mode():
            x_rand = vae.decoder(torch.normal(mean=0, std=1, size=[1000, vae.latent_dim], device=vae.device))
            x_circ = vae.get_circumference(batch_size=1000, sigma=1.0)
        ax.scatter(
            x_rand[:, 0], x_rand[:, 1],
            color="black", alpha=0.1, s=2, label="VAE-rand",
        )
        ax.scatter(
            x_circ[:, 0], x_circ[:, 1],
            color="blue", alpha=0.1, s=0.5, label="VAE-circ",
        )

        ax.legend()
        fig.savefig("./logs/kmeans_and_vae.png", dpi=300, bbox_inches="tight")

    expt_cluster_2D()


########################################################################################
class VAE(nn.Module):
    def __init__(self, dim=128, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

class LitVAE(L.LightningModule):
    def __init__(self, dim=128, latent_dim=32, lr=1e-3):
        super().__init__()
        self.model = VAE(dim, latent_dim)
        self.lr = lr

    def training_step(self, batch, _):
        x = batch
        x_hat, mu, logvar = self.model(x)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = recon_loss + kl
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, num_samples=16):
        z = torch.randn(num_samples, self.model.mu.out_features, device=self.device)
        return self.model.decoder(z)
