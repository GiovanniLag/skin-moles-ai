from __future__ import annotations
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)

from .dermanet import DermResNetSE


class DermaNetLightning(pl.LightningModule):
    """Thin Lightning wrapper for :class:`DermResNetSE`.
    
    Parameters:
    -----------
    num_classes : int
        Number of output classes.
    class_weights : torch.Tensor
        Class weights for handling class imbalance.
    model_cfg : Optional[Dict[str, Any]]
        Additional keyword arguments for :class:`DermResNetSE`.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 penalty) for the optimizer.
    epochs : int
        Total number of training epochs (used for LR scheduler).
    confusion_matrix_dir : Optional[str]
        If provided, saves the confusion matrix plot in this directory at the end of each validation epoch and test epoch.
    num_to_label : Optional[Dict[int, str]]
        Mapping from numerical labels to string labels for confusion matrix plotting.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor,
        model_cfg: Optional[Dict[str, Any]] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 40,
        confusion_matrix_dir: Optional[str] = None,
        num_to_label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DermResNetSE(num_classes=num_classes, **(model_cfg or {}))
        self.register_buffer("class_weights", class_weights)
        # Define loss function with class weights and label smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=0.05
        )

        # Validation metrics
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

        # Test metrics
        self.test_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

        self.confusion_matrix_dir = confusion_matrix_dir
        self.num_to_label = num_to_label

    def forward(self, x):
        logits, *_ = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["labels"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _shared_eval(self, batch, stage: str):
        x, y = batch["image"], batch["labels"]
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if stage == "val":
            self.val_auroc.update(probs, y)
            self.val_f1.update(probs, y)
            self.val_acc1.update(probs, y)
            self.val_acc5.update(probs, y)
            self.val_confmat.update(preds, y)
        else:
            self.test_auroc.update(probs, y)
            self.test_f1.update(probs, y)
            self.test_acc1.update(probs, y)
            self.test_acc5.update(probs, y)
            self.test_confmat.update(preds, y)

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, "test")

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        f1 = self.val_f1.compute()
        acc1 = self.val_acc1.compute()
        acc5 = self.val_acc5.compute()
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/f1", f1)
        self.log("val/acc1", acc1)
        self.log("val/acc5", acc5)

        if self.trainer.is_global_zero:
            cm = self.val_confmat.compute().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            if self.num_to_label:
                num_classes = self.hparams.num_classes
                ax.set_xticks(range(num_classes))
                ax.set_xticklabels([self.num_to_label[i] for i in range(num_classes)], rotation=45, ha='right')
                ax.set_yticks(range(num_classes))
                ax.set_yticklabels([self.num_to_label[i] for i in range(num_classes)])
            plt.tight_layout()
            if self.logger: # Log confusion matrix to the logger
                self.logger.experiment.add_figure("val/confusion_matrix", fig, self.current_epoch)
            if self.confusion_matrix_dir: # Save to file if path is provided
                os.makedirs(self.confusion_matrix_dir, exist_ok=True)

                conf_matrix_name = os.path.join(self.confusion_matrix_dir, f"val_confusion_matrix-epoch{self.current_epoch}.png")
                fig.savefig(conf_matrix_name)
            plt.close(fig)
        
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_acc1.reset()
        self.val_acc5.reset()
        self.val_confmat.reset()

    def on_test_epoch_end(self):
        auroc = self.test_auroc.compute()
        f1 = self.test_f1.compute()
        acc1 = self.test_acc1.compute()
        acc5 = self.test_acc5.compute()
        self.log("test/auroc", auroc)
        self.log("test/f1", f1)
        self.log("test/acc1", acc1)
        self.log("test/acc5", acc5)

        if self.trainer.is_global_zero:
            cm = self.test_confmat.compute().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            if self.num_to_label:
                num_classes = self.hparams.num_classes
                ax.set_xticks(range(num_classes))
                ax.set_xticklabels([self.num_to_label[i] for i in range(num_classes)], rotation=45, ha='right')
                ax.set_yticks(range(num_classes))
                ax.set_yticklabels([self.num_to_label[i] for i in range(num_classes)])
            plt.tight_layout()
            if self.logger:  # Log confusion matrix to the logger
                self.logger.experiment.add_figure("test/confusion_matrix", fig, self.current_epoch)
            if self.confusion_matrix_dir:  # Save to file if path is provided
                os.makedirs(self.confusion_matrix_dir, exist_ok=True)

                conf_matrix_name = os.path.join(self.confusion_matrix_dir, "test_confusion_matrix.png")
                fig.savefig(conf_matrix_name)
            plt.close(fig)
        
        self.test_auroc.reset()
        self.test_f1.reset()
        self.test_acc1.reset()
        self.test_acc5.reset()
        self.test_confmat.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.999)
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": sched}