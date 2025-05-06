import csv
import os

from torch import Tensor
import lightning as L
from lightning.fabric.utilities.rank_zero import rank_zero_only
from ..pythonic.dict_utils import flatten_dictlist

class CsvLoggerCallback(L.Callback):
    """Log all available metrics to a CSV file at a user-specified event.

    Args:
    + `save_path`: Path to the CSV file for logging metrics.
    + `event_name`: Name of the event to log at. Defaults to `"train_epoch_end"`.
    """

    def __init__(self, save_path: str, event_name: str = "train_epoch_end"):
        super().__init__()
        self.save_path = save_path
        self.event_name = event_name
        # Dynamically bind the event method
        setattr(self, f"on_{self.event_name}", self._on_event)

    @rank_zero_only
    def _on_event(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs):
        # Synchronize metrics across processes
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
            writer.writerow({k: self.format(k, v) for k, v in metrics_dict.items()})
    
    def format(self, k: str, v: float) -> str:
        if isinstance(v, Tensor):
            return str(v.tolist())
        
        return str(v)
