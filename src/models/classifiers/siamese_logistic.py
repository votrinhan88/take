from torch import nn, Tensor
import torch

from .base_classifier import BaseClassifier


class SiameseLogistic(BaseClassifier):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        loss_fn: str = "ce",
        return_logits: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.loss_fn: nn.Module = self._validate_args("loss_fn", loss_fn)
        self.return_logits = return_logits

        # Classifier on combined representation [u; v; |u-v|; u*v]
        self.classifier = nn.Linear(input_dim * 4, num_classes)

        if not self.return_logits:
            if self.num_classes == 1:
                self.act = nn.Sigmoid()
            elif self.num_classes > 1:
                self.act = nn.Softmax(dim=1)

    def _validate_args(self, arg: str, value):
        if arg == "loss_fn":
            if value == "ce":
                if self.num_classes == 1:
                    return nn.BCEWithLogitsLoss()
                else:
                    return nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unsupported loss function: {value}")

    def forward(self, input: Tensor, other: Tensor) -> Tensor:
        u, v = input, other
        combined = torch.cat(tensors=[u, v, torch.abs(u - v), u * v], dim=-1)
        x = self.classifier(combined)
        if not self.return_logits:
            x = self.act(x)
        return x

    def forward_with_loss(self, input: Tensor, other: Tensor, target: Tensor):
        pred = self(input, other)
        loss = self.loss_fn(pred, target)
        return pred, loss

    @property
    def hparams(self):
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "loss_fn": self.loss_fn.__class__.__name__,
            "return_logits": self.return_logits,
        }
