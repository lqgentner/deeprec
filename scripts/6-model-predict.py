#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create predictions of a trained model for the full data set period.
Help on the usage:
    python scripts/6-model-predict.py --help
"""

import argparse
from pydoc import locate

import lightning as L
import torch
import wandb
import xarray as xr
from lightning.pytorch.callbacks import RichProgressBar

from deeprec.data import DeepRecDataModule
from deeprec.utils import ROOT_DIR, wandb_checkpoint_download


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get predictions of a trained model for the full data set period."
    )
    parser.add_argument("project")
    parser.add_argument("run_id")
    parser.add_argument(
        "-a", "--alias", default="best", help="'best', 'latest', or 'v<int>'"
    )
    parser.add_argument(
        "-s",
        "--store",
        default="models/predictions/preds.zarr",
        help="path of zarr store",
    )
    args = parser.parse_args()

    predict(
        wandb_project=args.project,
        wandb_run_id=args.run_id,
        alias=args.alias,
        zarr_store=args.store,
    )


def predict(
    wandb_project: str, wandb_run_id: str, zarr_store: str, alias: str = "best"
) -> None:
    # Use TensorFloat32 datatype
    torch.set_float32_matmul_precision("high")

    wandb.login()
    api = wandb.Api()
    run = api.run(f"{wandb_project}/{wandb_run_id}")

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
        logger=False,
        callbacks=[RichProgressBar()],
    )

    # Make prediction
    pred = trainer.predict(model, datamodule=dm)
    pred_name = f"{wandb_run_id}_{alias}"
    # Safety copy
    pred_xr = dm.predictions_to_xarray(pred, name=pred_name)
    if isinstance(pred_xr, xr.DataArray):
        pred_ds = pred_xr.to_dataset()
    else:
        pred_ds = pred_xr

    # Set attributes
    pred_ds.attrs = {
        "model_class": model_class.__name__,
        "scenario": config["scenario"],
        "target": config["data"]["target_var"],
    }

    # Save in Zarr storage
    store_path = ROOT_DIR / zarr_store
    print(f"Writing prediction '{pred_name}' to zarr store '{store_path}...")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    pred_ds = pred_ds.chunk({"lat": 120, "lon": 120}).to_zarr(store_path, mode="a")
    print("Completed successfully.")


if __name__ == "__main__":
    main()
