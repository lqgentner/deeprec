#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge the TWSA and uncertainty predictions of multiple model ensembles and save as zarr
Help on the usage:
    python scripts/7-mix-ensemble.py --help
"""

import argparse
from pathlib import Path

from dask.diagnostics import ProgressBar
from loguru import logger
import numpy as np
import xarray as xr

from deeprec.preprocessing import calculate_grace_anomaly


def main():
    parser = argparse.ArgumentParser(
        description="Create a Laplacian mixture distribution by merging TWSA (mean) and uncertainty (scale parameter) predictions of an ensemble and save as zarr."
    )
    parser.add_argument(
        "in_store",
        help="Input Zarr store path that contains ensemble member predictions.",
    )
    parser.add_argument("out_store", help="Output Zarr store path")

    args = parser.parse_args()

    # Open dataset containing predictions
    ds_all = xr.open_zarr(args.in_store)

    # Split dataset containing mu ("pred_") and scale parameters ("uncertainty_") into two
    ds_mean, ds_scale = split_pred_uncertainty(ds_all)

    # Check if uncertainty prediction available
    has_uncertainty = bool(ds_scale)

    # Log status
    member_names = list(ds_mean.data_vars)
    logger.info("Ensemble with {} members found:", len(member_names))
    logger.info(member_names)
    if has_uncertainty:
        logger.info("Predictions of mean and scale parameter found.")
    else:
        logger.info("Only predictions of mean found.")

    # Calculate mean of mixture
    da_mean = ds_mean.to_dataarray("member").astype(np.float64)
    logger.info("Calculating mean of mixture...")
    da_mean_mix = da_mean.mean("member").compute()

    if has_uncertainty:
        da_scale = ds_scale.to_dataarray("member").astype(np.float64)
        # Aleatoric variance of mixture
        logger.info("Calculating aleatoric uncertainty of mixture...")
        da_var_ale_mix = (2 * da_scale**2).mean("member").compute()
        # Epistemic variance of mixture
        logger.info("Calculating epistemic uncertainty of mixture...")
        da_var_epi_mix = ((da_mean**2).mean("member") - da_mean_mix**2).compute()
        # Total variance of mixture
        da_var_mix = da_var_ale_mix + da_var_epi_mix
        # Standard deviations of mixture
        da_sig_ale_mix: xr.DataArray = np.sqrt(da_var_ale_mix)
        da_sig_epi_mix: xr.DataArray = np.sqrt(da_var_epi_mix)
        da_sig_mix: xr.DataArray = np.sqrt(da_var_mix)

    # Subtract GRACE baseline from mean
    da_mean_mix = calculate_grace_anomaly(da_mean_mix)

    # Add data arrays to dataset
    ds_out = da_mean_mix.to_dataset(name="twsa")

    if has_uncertainty:
        ds_out["sigma"] = da_sig_mix
        ds_out["sigma_epi"] = da_sig_epi_mix
        ds_out["sigma_ale"] = da_sig_ale_mix

    # Save as Zarr
    logger.info("Writing to Zarr store...")
    out_store = Path(args.out_store)
    out_store.parent.mkdir(exist_ok=True)
    (
        ds_out.chunk(time=-1, lat=120, lon=120)
        .astype(np.float32)
        .to_zarr(out_store, mode="w")
    )

    logger.info("Completed successfully.")


def split_pred_uncertainty(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Split dataset containing predictions and uncertainties into two datasets"""

    mean_names = [name for name in ds.data_vars if str(name).startswith("pred_")]
    scale_names = [
        name for name in ds.data_vars if str(name).startswith("uncertainty_")
    ]

    # Split in prediction and uncertainty datasets
    ds_mean = ds[mean_names]
    ds_scale = ds[scale_names]

    # Remove pred / uncertainty suffix from data variable names
    ds_mean = remove_name_prefix(ds_mean)
    if ds_scale:
        ds_scale = remove_name_prefix(ds_scale)
        # Check that names (<wandb_id>_<alias>) match
        assert list(ds_mean.data_vars) == list(ds_scale.data_vars)

    return ds_mean, ds_scale


def remove_name_prefix(ds: xr.Dataset) -> xr.Dataset:
    """Removes the variable name prefix. Variables must be named
    <prefix>_<wandb_id>_<alias>.
    """
    renamer = {name: "_".join(str(name).split("_")[1:]) for name in ds.data_vars}
    return ds.rename_vars(renamer)


if __name__ == "__main__":
    # Register dask progress bar
    ProgressBar().register()

    main()
