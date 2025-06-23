#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a model on the test set.
Help on the usage:
    python scripts/5-model-test.py --help
"""

import argparse
from pydoc import locate

import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
import torch
import wandb

from deeprec.data import DeepRecDataModule
from deeprec.training import define_wandb_metrics
from deeprec.utils import ROOT_DIR, wandb_checkpoint_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on the test set.")
    parser.add_argument("project")
    parser.add_argument("run_id")
    parser.add_argument(
        "-a", "--alias", default="best", help="'best', 'latest', or int"
    )
    args = parser.parse_args()

    # Use TensorFloat32 datatype
    torch.set_float32_matmul_precision("high")
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    test(wandb_project=args.project, wandb_run_id=args.run_id, alias=args.alias)


def test(wandb_project: str, wandb_run_id: str, alias: str = "best") -> None:
    """Fit and test a DeepRec model"""
    # Set up W&B logger
    # Remove user name from project path
    wandb_project = wandb_project.split("/")[-1]
    wandb.init(project=wandb_project, id=wandb_run_id, resume="must")
    run = wandb.run
    if run is None:
        raise RuntimeError(
            "wandb.run is None. Failed to initialize W&B run. Check your project and run_id."
        )
    wandb_logger = WandbLogger()

    # Download checkpoint
    ckpt_file = wandb_checkpoint_download(
        project=wandb_project, run_id=wandb_run_id, alias=alias
    )

    # Get config
    config = run.config

    # Model creation
    model_class = locate(config["model"]["class_path"])
    if isinstance(model_class, L.LightningModule):
        model = model_class.load_from_checkpoint(ckpt_file, **config["model"])
    else:
        raise TypeError(
            f"Provided class_path '{model_class}' is not a LightningModule."
        )

    # Data creation
    dm = DeepRecDataModule(**config["data"])

    # Use W&B run name as directory
    model_dir = ROOT_DIR / f"models/trained/{run.name}"
    model_dir.mkdir(exist_ok=True)

    # Initialize trainer
    trainer = L.Trainer(
        default_root_dir=model_dir,
        accelerator="gpu",
        max_epochs=config["trainer"]["max_epochs"],
        logger=wandb_logger,
        callbacks=[RichProgressBar()],
    )

    # Define metrics
    define_wandb_metrics()
    # Write data summary
    logger.info("Training with the following split:")
    logger.info(dm.split_stats)
    logger.info("Training with the following variables:")
    logger.info(dm.variables)

    # Fit and test
    trainer.test(model, datamodule=dm, ckpt_path=ckpt_file)
    wandb.finish()


if __name__ == "__main__":
    main()
