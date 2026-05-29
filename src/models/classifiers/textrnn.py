from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np
import torch
from torch import nn, Tensor


class TextRNN(nn.Module):
    # dataset-distillation/networks/networks.py#35-80

    supported_mask_strategies = ["trim_zero", "none"]

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 100,
        hidden_dim: int = 100,
        num_layers: int = 2,
        bidirectional: bool = False,
        p_dropout: float = 0.5,
        mask_strategy: str = "trim_zero",
        return_logits: bool = True,
        loss_fn="ce",
    ):
        super(TextRNN, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.p_dropout = p_dropout
        self.mask_strategy = self._validate_args("mask_strategy", mask_strategy)
        self.return_logits = return_logits
        self.loss_fn = self._validate_args("loss_fn", loss_fn)

        # RNN layer
        self.rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=self.p_dropout,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.linear = nn.Linear(
            in_features=2 * self.hidden_dim if self.bidirectional else self.hidden_dim,
            out_features=self.num_classes,
        )
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

    def forward(self, input: Tensor) -> Tensor:
        output, (hidden, cell) = self.rnn(input)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        x = self.linear(hidden)
        if not self.return_logits:
            x = self.act(x)
        return x

    def forward(self, input: Tensor, mask: Tensor | None = None) -> Tensor:
        """Args:
        + `input`: Tensor of shape [B, S, E]
        + `mask`: Tensor of shape [B, S], 1 for real tokens, 0 for pad. If None, masking is skipped.
        """
        # input: [B, S, E] (batch, sequence, embedding)
        x = input
        if input.shape[1] == 0:
            x_pad = torch.zeros(input.shape[0], 1, input.shape[2], device=input.device)
            x = torch.cat([x, x_pad], dim=1)
        
        mask = self.infer_mask(x) if mask is None else mask  # [B, S] or None

        if mask is not None:
            lengths = mask.sum(dim=1).cpu()  # [B], true lengths
            lengths[lengths == 0] = 1  # all-0 embedding for zero-length (OOV) sequences
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.rnn(packed)  # [n_layers * n_dir, B, H]
        else:
            _, (hidden, _) = self.rnn(x)  # [n_layers * n_dir, B, H]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))  # [B, 2H]
        else:
            hidden = self.dropout(hidden[-1])  # [B, H]

        x = self.linear(hidden)  # [B, K]
        if not self.return_logits:
            x = self.act(x)
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
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.num_classes,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "p_dropout": self.p_dropout,
            "mask_strategy": self.mask_strategy,
            "return_logits": self.return_logits,
            "loss_fn": self.loss_fn.__class__.__name__,
        }
