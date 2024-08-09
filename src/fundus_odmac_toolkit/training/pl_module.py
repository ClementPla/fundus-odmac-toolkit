from typing import Sequence, Tuple

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
        self.loss_fn = torchseg.losses.DiceLoss("multiclass")
        self.params = training_params
        self.dataset_params = dataset_params
        self.metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Dice(num_classes=3),
                torchmetrics.JaccardIndex(task="multiclass", num_classes=3),
            ],
            prefix="Validation ",
        )

        self.test_metrics = self.metrics.clone(prefix="Test ")

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
        lr = self.params.get("lr", 1e-4)
        weight_decay = self.params.get("weight_decay", 1e-2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def prepare_data(self) -> None:
        self.datamodule = get_datamodule_from_config(self.dataset_params)

    def train_dataloader(self) -> torch.Any:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> torch.Any:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> torch.Any:
        return self.datamodule.test_dataloader()
