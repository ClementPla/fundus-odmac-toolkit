import argparse
import os

from fundus_odmac_toolkit.training.pl_module import ODMacLightningModule
from nntools.utils import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

wandb.require("core")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", help="Overrides model.architecture from the config")
    parser.add_argument("--encoder", help="Overrides model.encoder from the config")
    args = parser.parse_args()

    config = Config("config/config.yaml")
    if args.arch:
        config["model"]["architecture"] = args.arch
    if args.encoder:
        config["model"]["encoder"] = args.encoder

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
            LearningRateMonitor(),
        ],
    )

    trainer.fit(pl_module)
    trainer.test(pl_module, ckpt_path="best")


if __name__ == "__main__":
    train()
