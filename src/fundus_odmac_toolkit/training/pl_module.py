import math
from typing import Any, Sequence, Tuple

import torch
import torchmetrics
import torchseg
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import (
    LRScheduler,
    LRSchedulerConfig,
    Optimizer,
    OptimizerLRSchedulerConfig,
    ReduceLROnPlateau,
)

from fundus_odmac_toolkit.training.data_factory import get_datamodule_from_config


class ODMacLightningModule(LightningModule):
    def __init__(self, model, encoder, training_params, dataset_params):
        super().__init__()
        self.model = torchseg.create_model(model, encoder, encoder_weights=True, classes=3)
        self.dice_loss = torchseg.losses.DiceLoss("multiclass")
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.params = training_params
        self.dataset_params = dataset_params
        self.metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(task="multiclass", num_classes=3, average="macro"),  # = mean Dice over classes
                torchmetrics.JaccardIndex(task="multiclass", num_classes=3),
            ],
            prefix="Validation ",
        )

        self.test_metrics = self.metrics.clone(prefix="Test ")

    def loss_fn(self, y_hat, y):
        # Cross-entropy penalises scattered false positives that Dice alone tolerates
        return self.dice_loss(y_hat, y) + self.params.get("ce_weight", 1.0) * self.ce_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"].long()

        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"].long()

        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        y_proba = torch.nn.functional.softmax(y_hat, dim=1)

        self.log_dict(self.metrics(y_proba, y), on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        x = batch["image"]
        y = batch["mask"].long()
        y_hat = self.model(x)
        y_proba = torch.nn.functional.softmax(y_hat, dim=1)
        self.log_dict(self.test_metrics(y_proba, y), on_epoch=True, add_dataloader_idx=True, sync_dist=True)

    def configure_optimizers(
        self,
    ) -> (
        Optimizer
        | Sequence[Optimizer]
        | Tuple[Sequence[Optimizer], Sequence[LRScheduler | ReduceLROnPlateau | LRSchedulerConfig]]
        | OptimizerLRSchedulerConfig
        | Sequence[OptimizerLRSchedulerConfig]
        | None
    ):
        lr = self.params.get("lr", 3e-4)
        encoder_lr = self.params.get("encoder_lr", lr / 10)  # pretrained encoder: smaller steps
        weight_decay = self.params.get("weight_decay", 1e-2)
        encoder_params = list(self.model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.model.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW(
            [{"params": encoder_params, "lr": encoder_lr}, {"params": other_params, "lr": lr}],
            weight_decay=weight_decay,
        )

        # Linear warm-up followed by cosine decay to 0, stepped every optimizer step
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.params.get("warmup_ratio", 0.05) * total_steps)

        def schedule(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def prepare_data(self) -> None:
        self.datamodule = get_datamodule_from_config(self.dataset_params)

    def train_dataloader(self) -> Any:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> Any:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> Any:
        return self.datamodule.test_dataloader()
