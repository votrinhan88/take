from torch import nn, Tensor

from ..modules import Concatenate
from .base_classifier import BaseClassifier


class TextCNN(BaseClassifier):
    # dataset-distillation/networks/networks.py#L260-303

    supported_mask_strategies = ["trim_zero", "none"]

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 100,
        num_channels: int = 100,
        kernel_sizes: list[int] = [3, 4, 5],
        p_dropout: float = 0.5,
        mask_strategy: str = "trim_zero",
        return_logits: bool = True,
        loss_fn="ce",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.p_dropout = p_dropout
        self.mask_strategy = self._validate_args("mask_strategy", mask_strategy)
        self.return_logits = return_logits
        self.loss_fn = self._validate_args("loss_fn", loss_fn)

        # CNN layer
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=self.num_channels,
                kernel_size=(ks, self.embed_dim),
            )
            self.convs.append(nn.Sequential(conv, nn.ReLU()))
        # Max-over-time pooling
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.cat = Concatenate(dim=1)
        self.dropout = nn.Dropout(p_dropout)
        self.linear = nn.Linear(num_channels * len(kernel_sizes), num_classes)

        if not self.return_logits:
            if self.num_classes == 1:
                self.act = nn.Sigmoid()
            elif self.num_classes > 1:
                self.act = nn.Softmax(dim=1)

    def _validate_args(self, arg: str, value):
        if arg == "loss_fn":
            if value == "ce":
                return nn.CrossEntropyLoss()
            else:
                return value

        elif arg == "mask_strategy":
            if value not in self.supported_mask_strategies:
                msg = f"Invalid mask_strategy: {value}. Supported: {self.supported_mask_strategies}"
                raise ValueError(msg)
            return value

    def infer_mask(self, input: Tensor) -> Tensor | None:
        mask = None
        if self.mask_strategy == "trim_zero":
            mask = (input.abs().sum(dim=-1) != 0).long()
        return mask

    def forward(self, input: Tensor, mask: Tensor | None = None) -> Tensor:
        """Args:
        + `input`: Tensor of shape [B, S, E]
        + `mask`: Tensor of shape [B, S], 1 for real tokens, 0 for pad. If None, masking is skipped.
        """
        # input: [B, S, E] (batch, sequence, embedding)
        mask = self.infer_mask(input) if mask is None else mask  # [B, S] / None

        x = input.unsqueeze(dim=1)  # [B, 1, S, E]
        x_conv = [conv(x).squeeze(dim=3) for conv in self.convs]  # list[B, C, S-ks+1]
        if mask is not None:
            for i, (xc, ks) in enumerate(zip(x_conv, self.kernel_sizes)):
                conv_mask = mask[:, ks - 1 :]  # [B, S-ks+1] (rightmost of mask)
                x_conv[i] = xc * conv_mask.unsqueeze(dim=1)  # [B, C, S-ks+1]
        x_pool = [self.pool(x).squeeze(dim=2) for x in x_conv]  # list[B, C], pool along sequence
        x = self.cat(x_pool)  # [B, C * num_kernels]
        x = self.dropout(x)  # [B, C * num_kernels]
        x = self.linear(x)  # [B, K]
        if not self.return_logits:
            x = self.act(x)  # [B, K]
        return x

    def forward_with_loss(
        self,
        input: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pred = self(input, mask)
        loss = self.loss_fn(pred, target)
        return pred, loss

    def hparams(self):
        return {
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "num_channels": self.num_channels,
            "kernel_sizes": self.kernel_sizes,
            "p_dropout": self.p_dropout,
            "mask_strategy": self.mask_strategy,
            "return_logits": self.return_logits,
            "loss_fn": self.loss_fn.__class__.__name__,
        }
