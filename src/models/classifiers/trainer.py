import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

from .base_classifier import BaseClassifier
from src.utils.metrics import WallTime


class ClassifierTrainer(pl.LightningModule):
    """Classifier trainer.

    Args:
    + `classifier`: Trainer model.
    + `encoder`: Encoder model.
    + `loss_fn`: Loss function.
    + `optimizer_kw`: Dict of optimizer kwargs.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        encoder: nn.Module | None,
        optimizer_kw: dict,
        num_classes: int | None = None,
        preembed: bool = False,
        paired: bool = False,
    ):
        super().__init__()
        self.classifier = classifier
        self.encoder = encoder
        self.optimizer_kw = optimizer_kw
        self.num_classes = num_classes
        self.preembed = preembed
        self.paired = paired

        if self.num_classes is None:
            self.num_classes: int = self.classifier.num_classes

        hparams = {"optimizer_kw": self.optimizer_kw} 
        self.save_hyperparameters(hparams)

        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self.wall_time = WallTime(unit="hours")

    def configure_optimizers(self):
        opt_C = self.optimizer_kw["classifier"]["Class"](
            self.classifier.parameters(), **self.optimizer_kw["classifier"]["kwargs"]
        )
        return opt_C

    def step(self, batch):
        if not self.paired:
            return self._single_step(batch)
        else:
            return self._paired_step(batch)

    def _single_step(self, batch):
        if self.preembed:
            x = batch["embeddings"].to(device=self.device)
        else:
            x = batch["text"]
            if self.encoder is not None:
                x = self.encoder(x).to(device=self.device)
        y = batch["label"]
        logit, loss = self.classifier.forward_with_loss(x, target=y)
        return logit, y, loss

    def _paired_step(self, batch):
        y = batch["label"]
        if self.preembed:
            x_0 = batch["embeddings_0"].to(device=self.device)
            x_1 = batch["embeddings_1"].to(device=self.device)
        else:
            x_0 = batch["text_0"]
            x_1 = batch["text_1"]
            if self.encoder is not None:
                x_0 = self.encoder(x_0).to(device=self.device)
                x_1 = self.encoder(x_1).to(device=self.device)
        logit, loss = self.classifier.forward_with_loss(x_0, x_1, target=y)
        return logit, y, loss

    def training_step(self, batch, batch_idx) -> Tensor:
        logit, y, loss = self.step(batch)
        self.train_acc.update(logit, y.to(dtype=torch.int64))
        self.log(name="train_loss", value=loss, prog_bar=True)
        self.log(name="train_acc", value=self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(name="wall_time", value=self.wall_time, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        logit, y, loss = self.step(batch)
        self.val_acc.update(logit, y.to(dtype=torch.int64))
        self.log(name="val_loss", value=loss, prog_bar=True)
        self.log(name="val_acc", value=self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def fit(self, fit_kw, **L_trainer_kw) -> dict:
        L_trainer = pl.Trainer(**L_trainer_kw)
        L_trainer.fit(self, **fit_kw)
        return {k: v.item() for k, v in L_trainer.callback_metrics.items()}

    def evaluate(self, eval_kw, **L_trainer_kw) -> dict:
        L_trainer = pl.Trainer(**L_trainer_kw)
        L_trainer.validate(self, **eval_kw)
        return {k: v.item() for k, v in L_trainer.callback_metrics.items()}
