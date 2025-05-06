import torch
from torch import nn, Tensor


class MiniLMWrapper(nn.Module):
    """
    all-MiniLM-L6-v2
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device

    def forward(self, *args, **kwargs) -> Tensor:
        embedding = self.model.encode(*args, **kwargs)
        embedding = torch.from_numpy(embedding).to(self.device)
        return embedding

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
