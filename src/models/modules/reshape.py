from torch import nn, Tensor


class Reshape(nn.Module):
    """Reshape the input to the given shape.

    Args:
    + `shape`: Output shape.
    """

    def __init__(self, shape: list[int]):
        super().__init__()
        self.shape = shape

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        x = input.view([batch_size, *self.shape])
        return x

    def extra_repr(self) -> str:
        return f"shape={self.shape}"
