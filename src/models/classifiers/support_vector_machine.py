import torch
from torch import nn, Tensor

from .base_classifier import BaseClassifier
from ..modules.losses import get_reduction_fn


class SupportVectorMachine(BaseClassifier):
    def __init__(self, input_dim: int, num_classes: int = 2, loss_fn: str = "hinge"):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.loss_fn = self._validate_args("loss_fn", loss_fn)

        self.linear = nn.Linear(in_features=input_dim, out_features=num_classes)

    def _validate_args(self, arg: str, value):
        if arg == "loss_fn":
            if value == "hinge":
                return MultiClassHingeLoss()
            else:
                raise ValueError(f"Unsupported loss function: {value}")

    def forward(self, input: Tensor) -> Tensor:
        x = self.linear(input)
        return x

    def forward_with_loss(self, input: Tensor, target: Tensor):
        pred = self(input)
        loss = self.loss_fn(pred, target)
        return pred, loss

    @property
    def hparams(self):
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "loss_fn": self.loss_fn.__class__.__name__,
        }

    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.hparams.items() if v is not None])


class MultiClassHingeLoss(nn.Module):
    def __init__(self, margin: float = 1.0, slack: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.slack = slack
        self.reduction = reduction

        self.reduction_fn = get_reduction_fn(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size = input.shape[0]
        true_scores = input[torch.arange(batch_size), target].unsqueeze(1)

        margins = input - true_scores + self.margin
        margins[torch.arange(batch_size), target] = 0
        loss = margins.clamp(min=0).sum(dim=1)

        loss = self.reduction_fn(loss)
        return loss

    @property
    def hparams(self):
        return {"margin": self.margin, "slack": self.slack, "reduction": self.reduction}
