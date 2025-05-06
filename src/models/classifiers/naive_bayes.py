from sklearn.naive_bayes import MultinomialNB
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import lightning as L


class NaiveBayes(L.LightningModule):
    """Naive Bayes Classifier.

    Args:
      `input_dim`: _description_.
      `epsilon`: _description_. Defaults to `1e-8`.
    """

    log_prior: nn.Parameter
    log_likelihood: nn.Parameter

    def __init__(self, input_dim: int, epsilon: float = 1e-8):
        super(NaiveBayes, self).__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon

        self.save_hyperparameters(
            {
                "input_dim": self.input_dim,
                "epsilon": self.epsilon,
            }
        )

    def fit(self, train_loader: DataLoader, encoder: nn.Module | None = None):
        x_all = torch.zeros(
            size=[1, self.hparams["input_dim"]],
            dtype=torch.float32,
            device=self.device,
        )
        y_all = torch.zeros(size=[1], dtype=torch.long, device=self.device)
        count = 0
        for batch in train_loader:
            if encoder is not None:
                x = encoder(batch["text"]).to(device=self.device)
            else:
                x = batch["embedding"].to(device=self.device)
            y = batch["label"].to(device=self.device)
            batch_size = x.shape[0]

            while count + batch_size > x_all.shape[0]:
                x_all = torch.cat([x_all, torch.zeros_like(x_all)], dim=0)
                y_all = torch.cat([y_all, torch.zeros_like(y_all)], dim=0)
            x_all[count : count + batch_size] = x.to_dense()
            y_all[count : count + batch_size] = y
            count = count + batch_size

        x_all = x_all[0:count]
        y_all = y_all[0:count]
        num_samples, *input_dim = x_all.shape

        # Compute class priors (log P(y))
        num_classes = y_all.unique().numel()
        class_counts = torch.zeros(size=[num_classes], device=self.device, dtype=torch.int64)
        for k in range(num_classes):
            class_counts[k] = (y_all == k).sum(dim=0)
        log_prior = torch.log(class_counts / num_samples)

        # Compute feature likelihoods (log P(x|y))
        # Naive Bayes so we assume independence of features to compute
        # Likelihood ~ feature counts: P(x|y)
        feature_count = torch.zeros(size=[num_classes, self.input_dim], device=self.device)
        for k in range(num_classes):
            feature_count[k, :] = x_all[y_all == k, :].sum(dim=0)
        # Extract probabilities
        feature_count = (feature_count + self.epsilon) / feature_count.sum(dim=1, keepdim=True)
        # log-likelihood: log P(x|y)
        log_likelihood = feature_count.log()

        self.register_parameter("log_prior", nn.Parameter(log_prior).to(self.device))
        self.register_parameter(
            "log_likelihood",
            nn.Parameter(log_likelihood).to(self.device),
        )

    def forward(self, input: Tensor) -> Tensor:
        # Compute log P(y) + log P(x|y)
        log_probs = self.log_prior + input @ self.log_likelihood.T
        return log_probs

    def evaluate(
        self, val_loader: DataLoader, encoder: nn.Module | None = None
    ) -> dict[str, float]:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = 0
        correct = 0
        total = 0

        with torch.inference_mode():
            for batch in val_loader:
                if encoder is not None:
                    x = encoder(batch["text"]).to(device=self.device)
                else:
                    x = batch["embedding"].to(device=self.device)
                y = batch["label"].to(device=self.device)
                outputs = self(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                loss += loss_fn(outputs, y).sum().item()
        accuracy = correct / total
        loss = loss / total
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        return {
            "accuracy": accuracy,
            "loss": loss,
        }


class GaussianNaiveBayes(L.LightningModule):
    mu: nn.Parameter
    sigma: nn.Parameter
    log_prior: nn.Parameter

    def __init__(self, input_dim: int, epsilon: float = 1e-8, device: str = "auto"):
        super(GaussianNaiveBayes, self).__init__()
        self.save_hyperparameters(
            {
                "input_dim": input_dim,
                "epsilon": epsilon,
            }
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    def fit(self, train_loader: DataLoader):
        x_all = torch.zeros(
            size=[1, self.hparams["input_dim"]],
            dtype=torch.float32,
            device=self.device,
        )
        y_all = torch.zeros(size=[1], dtype=torch.long, device=self.device)
        count = 0
        for data in train_loader:
            x, y = data
            x: Tensor = x.to(self.device)
            y: Tensor = y.to(self.device)
            batch_size = x.shape[0]

            while count + batch_size > x_all.shape[0]:
                x_all = torch.cat([x_all, torch.zeros_like(x_all)], dim=0)
                y_all = torch.cat([y_all, torch.zeros_like(y_all)], dim=0)
            x_all[count : count + batch_size] = x
            y_all[count : count + batch_size] = y
            count = count + batch_size

        x_all = x_all[0:count]
        y_all = y_all[0:count]
        num_samples, *input_dim = x_all.shape

        # Compute class priors (log P(y))
        num_classes = y_all.unique().numel()
        class_counts = torch.zeros(size=[num_classes], device=self.device, dtype=torch.int64)
        for k in range(num_classes):
            class_counts[k] = (y_all == k).sum(dim=0)
        log_prior = torch.log(class_counts / num_samples)

        # Compute feature likelihoods (log P(x|y))
        # Compute mean and variance for each feature conditioned on each class
        mu = torch.zeros([num_classes, *input_dim])
        sigma = torch.zeros([num_classes, *input_dim])
        for k in range(num_classes):
            x_k = x_all[y_all == k]
            mu[k] = x_k.mean(dim=0)
            sigma[k] = x_k.std(dim=0) + self.hparams["epsilon"]

        self.register_parameter("mu", nn.Parameter(mu).to(self.device))
        self.register_parameter("sigma", nn.Parameter(sigma).to(self.device))
        self.register_parameter("log_prior", nn.Parameter(log_prior).to(self.device))

    def forward(self, input: Tensor) -> Tensor:
        """Computes the log probabilities for each class given the input.

        Args:
          `input`: Input tensor of shape [B, D].

        Returns:
          Log probabilities of shape [B, K].
        """
        # Density from Normal distribution w.r.t mu and sigma of each class
        log_density = -0.5 * (
            +torch.log(2 * torch.pi * self.sigma.unsqueeze(dim=0) ** 2)
            + ((input.unsqueeze(dim=1) - self.mu.unsqueeze(dim=0)) ** 2)
            / (self.sigma.unsqueeze(dim=0) ** 2)
        )
        # Naive Bayes so we assume independence of features to compute
        # log-likelihood log P(x|y) = sum(log P(x_i|y))
        log_likelihood = torch.sum(log_density, dim=2)
        # Posterior (log P(y) + log P(x|y))
        log_posterior = self.log_prior.unsqueeze(dim=0) + log_likelihood
        return log_posterior
