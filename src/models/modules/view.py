from torch import nn, Tensor


class View(nn.Module):
    """View the input to the given shape.

    Args:
    + `shape`: Output shape.
    """

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return f"shape={self.shape}"
