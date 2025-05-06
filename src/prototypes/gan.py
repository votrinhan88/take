import torch

from torch import nn, Tensor

class Generator(nn.Module):
    def __init__(self, latent_dim=32, out_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class LitGAN(pl.LightningModule):
    def __init__(self, dim=128, latent_dim=32, lr=2e-4):
        super().__init__()
        self.generator = Generator(latent_dim, dim)
        self.discriminator = Discriminator(dim)
        self.latent_dim = latent_dim
        self.lr = lr

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch
        batch_size = real.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake = self.generator(z)

        if optimizer_idx == 0:  # Generator step
            pred_fake = self.discriminator(fake)
            g_loss = -pred_fake.mean()
            self.log("g_loss", g_loss)
            return g_loss

        if optimizer_idx == 1:  # Discriminator step
            pred_real = self.discriminator(real)
            pred_fake = self.discriminator(fake.detach())
            d_loss = -(pred_real.mean() - pred_fake.mean())
            self.log("d_loss", d_loss)
            return d_loss

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        return [g_opt, d_opt], []

    @torch.no_grad()
    def sample(self, num_samples=16):
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.generator(z)
