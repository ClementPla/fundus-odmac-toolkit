import os

from fundus_odmac_toolkit.training.pl_module import ODMacLightningModule
from nntools.utils import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

wandb.require("core")


def train():
    config = Config("config/config.yaml")

    project_name = config["logger"]["project"]

    wandb_logger = WandbLogger(project=project_name, config=config.tracked_params)

    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_RUN_NAME"] = wandb_logger.experiment.name

    checkpoint_callback = ModelCheckpoint(
        monitor="Validation MulticlassJaccardIndex",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", project_name, os.environ["WANDB_RUN_NAME"]),
    )

    pl_module = ODMacLightningModule(
        config["model"]["architecture"], config["model"]["encoder"], config["training"], config["data"]
    )
    pl_module.prepare_data()

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="Validation MulticlassJaccardIndex", patience=100, mode="max"),
            LearningRateMonitor(),
        ],
    )

    trainer.fit(pl_module)
    trainer.test(pl_module, ckpt_path="best")


if __name__ == "__main__":
    train()
