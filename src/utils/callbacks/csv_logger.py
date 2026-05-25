import csv
import os

from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from transformers import TrainerCallback
from ..pythonic.dict_utils import flatten_dictlist


class CsvLoggerPL(pl.Callback):
    """Log all available metrics to a CSV file at a user-specified event."""

    def __init__(self, save_path: str, event_name: str = "train_epoch_end"):
        super().__init__()
        self.save_path = save_path
        self.event_name = event_name
        setattr(self, f"on_{self.event_name}", self._on_event)

    @rank_zero_only
    def _on_event(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        trainer.strategy.barrier()
        metrics = trainer.callback_metrics
        if metrics is None:
            return
        write_header = not os.path.isfile(self.save_path)
        metrics_dict = {"epoch": trainer.current_epoch, **flatten_dictlist(metrics)}
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(metrics_dict.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({k: self._format(k, v) for k, v in metrics_dict.items()})

    def _format(self, k: str, v: float) -> str:
        if isinstance(v, Tensor):
            return str(v.tolist())
        return str(v)


class CsvLoggerHF(TrainerCallback):
    """Log HuggingFace Trainer train and eval metrics to a CSV file.

    Writes one row per logging/evaluation event with sparse columns
    (empty string for metrics not produced by that event).
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.rows = []
        self.fieldnames = ["epoch", "step"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            train_keys = {"loss", "learning_rate", "grad_norm", "epoch"}
            train_logs = {k: v for k, v in logs.items() if k in train_keys}
            if not train_logs:
                return
            self._write({"epoch": state.epoch, "step": state.global_step, **train_logs})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._write({"epoch": state.epoch, "step": state.global_step, **metrics})

    def _write(self, row: dict):
        self.rows.append(row)
        new_keys = [k for k in row if k not in self.fieldnames]
        if new_keys:
            self.fieldnames.extend(new_keys)
            self._rewrite()
        else:
            self._append(row)

    def _append(self, row: dict):
        write_header = not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _rewrite(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self.rows)
