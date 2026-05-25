import types
from typing import Callable, Optional, Any, Sequence

import numpy as np
import pytorch_lightning as pl


class PrintCallback(pl.Callback):
    """Print a progress bar to screen. Included in trainers by default.

    Args:
    + `event_name`: _description_. Defaults to `"train_epoch_end"`.
    + `format_dict`: Dict of functions to format `(metric, value)` to `string`. Defaults to `None`.
    """

    BLOCKS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    def __init__(self, format_dict: dict | None = None, event_name: str = "train_epoch_end"):
        super().__init__()
        self.format_dict = {} if format_dict is None else format_dict
        self.event_name = event_name

        self._format = self.register_format(format_dict=self.format_dict)
        # Dynamically bind the event method
        setattr(self, f"on_{self.event_name}", self._on_event)

    def _on_event(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        metrics = trainer.callback_metrics
        if metrics is None:
            return
        
        metrics_dict = {
            "epoch": trainer.current_epoch,
            **{k: metrics[k] for k in sorted(metrics.keys())},
        }
        print(" | ".join(f"{k}: {self._format(k, v)}" for k, v in metrics_dict.items()))

    @staticmethod
    def format_default(k: str, v: float) -> str:
        return f"{v:.4g}"

    def format_multicounter(self, k: str, v: np.ndarray) -> str:
        max = v.max() + 1e-8
        norm_count = np.round(8 * v / max).astype(int).tolist()
        return "".join(["[", *[self.BLOCKS[i] for i in norm_count], "]"])

    def format_adaptive(self, k: str, v: float | Sequence[int | float]) -> str:
        return self.format_dict.get(k, self.format_default)(k=k, v=v)

    def register_format(self, format_dict: dict | None = None) -> Callable[[str, Any], str]:
        if len(self.format_dict) == 0:
            return self.format_default
        else:
            return self.format_adaptive
