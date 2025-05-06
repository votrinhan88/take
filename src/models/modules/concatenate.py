import torch
from torch import nn, Tensor


class Concatenate(nn.Module):
    """Concatenate a list of tensors by the given dimension.

    Args:
    + `dim`: Concatenate dimension. Defaults to `1`.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, tensors: list[Tensor]) -> Tensor:
        return torch.cat(tensors, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
