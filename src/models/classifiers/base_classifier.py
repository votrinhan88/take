from torch import nn, Tensor


class BaseClassifier(nn.Module):
    """
    Unified interface for classifiers. Provides abstract methods for logit computation and loss
    calculation. All classifiers must inherit from this base and implement required methods.
    """

    def forward(self, *args, **kwargs) -> Tensor:
        """Compute logits for input(s).
        Subclasses define their own signature:
        + Single: `forward(input)`
        + Paired: `forward(input, other)`
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward_with_loss(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        """
        Compute (logits, loss) for given inputs and target.
        Target should always be passed as keyword argument `target=`
        to avoid positional ambiguity across subclasses:
        + Single: `forward_with_loss(input, target=target)`
        + Paired: `forward_with_loss(input, other, target=target)`
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def device(self):
        return next(self.parameters()).device

    @property
    def hparams(self) -> dict:
        return {}

    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.hparams.items() if v is not None])
