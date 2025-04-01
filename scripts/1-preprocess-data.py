#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess input and target netCDFs and save as Zarr store, according to specified YAML config file.
Usage:
    python scripts/1-preprocess-data.py <config YAML path>
"""

import argparse
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import dask
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from omegaconf import OmegaConf
from tqdm.std import tqdm

from deeprec.preprocessing import preprocessors as pp
from deeprec.utils import ROOT_DIR

time_chunks = {"time": 100}
space_chunks = {"time": -1, "lat": 120, "lon": 120}
dims = ("time", "lat", "lon", ...)


def main() -> None:
    """Run the script"""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Preprocess input and target netCDFs and save as Zarr store."
    )
    parser.add_argument(
        "config_file",
        help="Location of the configuration YAML.",
    )
    args = parser.parse_args()

    # Load configs
    config_dict = dict(OmegaConf.load(args.config_file))

    out_dir = Path(config_dict["out_dir"])
    config_inps: list[dict] = config_dict["inputs"]
    config_tgts: list[dict] = config_dict["targets"]
    config_feng: list[dict] = config_dict["engineering"]

    # Process dataset
    print("Load and process datasets lazily:")
    inps = process_datasets(config_inps)
    tgts = process_datasets(config_tgts)

    # Merge datasets and chunk space instead of time
    inps_ds = xr.merge(inps).chunk(space_chunks)
    tgts_ds = xr.merge(tgts).chunk(space_chunks)

    # Engineer additional inputs
    print("Perform feature engineering...")
    inps_ds = engineer_features(inps_ds, config_feng)

    # Save as Zarr
    out_dir.mkdir(exist_ok=True)
    zarr_inps = out_dir / "inputs.zarr"
    zarr_tgts = out_dir / "targets.zarr"

    delayed_inps = inps_ds.to_zarr(zarr_inps, mode="w", compute=False)
    delayed_tgts = tgts_ds.to_zarr(zarr_tgts, mode="w", compute=False)

    with ProgressBar():
        print("Write inputs to Zarr store:")
        delayed_inps.compute()
        print("Writing targets to Zarr store:")
        delayed_tgts.compute()
    print("Processing completed.")

    # Inspect zarr stores
    print("Zarr stores structure:")
    zgroup = zarr.open(out_dir)
    print(zgroup.tree())


def process_datasets(config: list[dict]) -> list[xr.Dataset]:
    """Open datasets and apply functions to them and their data variables."""
    ds_list = []
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # Iterate over datasets to load
        for ds_config in tqdm(config):
            # Open dataset
            files = ds_config["files"]
            if isinstance(files, str):
                # Dataset is a single file
                ds = xr.open_dataset(ROOT_DIR / files, decode_times=False)
            elif isinstance(files, Iterable):
                # Dataset is comprised of multiple files
                ds = xr.merge(
                    [
                        xr.open_dataset(ROOT_DIR / file, decode_times=False)
                        for file in files
                    ]
                )
            else:
                raise ValueError(
                    f"`files` must be str, tuple, or list, not {type(files)}."
                )
            # Create dict containing import and new variable name
            var_names = {
                dvar["name"]: dvar["rename"] if "rename" in dvar else dvar["name"]
                for dvar in ds_config["variables"]
            }
            # Handle ERA5 valid_time dimension
            if "valid_time" in ds.dims:
                ds = ds.rename(valid_time="time")
            ds = ds.chunk(time_chunks)
            # convert to float32
            ds = ds.astype("float32")
            # Decode netCDF time
            ds = pp.decode_time(ds)
            # Clean dimensions
            if "lon" in ds.dims or "longitude" in ds.dims:
                ds = pp.clean_grid(ds).transpose(*dims)
            # Run dataset pipeline
            if "pipeline" in ds_config:
                for item in ds_config["pipeline"]:
                    func = get_func(item)
                    ds = func(ds)
            # Run data variable pipelines
            for var_config in ds_config["variables"]:
                var_name = var_config["name"]
                if "pipeline" in var_config:
                    for item in var_config["pipeline"]:
                        func = get_func(item)
                        ds[var_name] = func(ds[var_name])
            # Select and rename data variables
            ds = ds[var_names.keys()].rename(var_names)
            # Add consistent dimension attributes
            ds = pp.set_dim_attrs(ds)
            # Append dataset to list
            ds_list.append(ds)
    return ds_list


def get_func(item: str | tuple[str, dict[str, Any]]) -> Callable:
    if isinstance(item, str):
        func = getattr(pp, item)
    elif isinstance(item, dict):
        # Function contains keyword arguments
        func = getattr(pp, item["func"])
        if "init_args" in item:
            kwargs = item["init_args"]
            # Apply keyword arguments to function
            func = partial(func, **kwargs)
    else:
        raise ValueError(
            f"Passed function {item} could not be parsed, must be str or dict."
        )
    return func


def engineer_features(ds: xr.Dataset, config: list[dict]) -> xr.Dataset:
    """Add features to a dataset."""
    # Loop over all specified preprocessing functions
    for item in config:
        func = get_func(item)
        # Apply function and convert output to float32
        ds = func(ds).astype("float32")
    return ds.chunk("auto")


if __name__ == "__main__":
    # Run preprocessing
    main()
