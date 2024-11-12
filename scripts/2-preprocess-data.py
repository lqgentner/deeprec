"""Preprocess inputs and targets and save as Zarr store, according to specified YAML config files"""

from collections.abc import Iterable, Callable
from typing import Any
from os import PathLike
from pathlib import Path
from functools import partial

import dask
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from ruamel.yaml import YAML
from tqdm.std import tqdm

from deepwaters.utils import ROOT_DIR
from deepwaters.preprocessing import preprocessors as pp

time_chunks = {"time": 100}
dims = ("time", "lat", "lon", ...)


def main(
    config: str | PathLike,
    out_dir: str | PathLike,
) -> None:
    """Run the script"""
    # Load configs
    config_dict = YAML().load(config)
    config_inps = config_dict["inputs"]
    config_tgts = config_dict["targets"]
    config_feng = config_dict["engineering"]

    # Process dataset
    print("Load and process datasets lazily:")
    inps = process_datasets(config_inps)
    tgts = process_datasets(config_tgts)

    # Merge datasets
    inps_ds = xr.merge(inps)
    tgts_ds = xr.merge(tgts)

    # Engineer additional inputs
    print("Perform feature engineering...")
    inps_ds = engineer_features(inps_ds, config_feng)

    # Save as Zarr
    out_dir = Path(out_dir)
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


def process_datasets(config: str | PathLike) -> list[xr.Dataset]:
    """Open datasets and apply functions to them and their data variables,
    as specified in a config YAML.
    """
    ds_list = []
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # Iterate over datasets to load
        for ds_config in tqdm(config):
            # Open dataset
            files = ds_config["files"]
            if isinstance(files, str):
                # Dataset is a single file
                ds = xr.open_dataset(ROOT_DIR / files, decode_times=False).chunk(
                    time_chunks
                )

            elif isinstance(files, Iterable):
                # Dataset is comprised of multiple files
                ds = xr.merge(
                    [
                        xr.open_dataset(ROOT_DIR / file, decode_times=False)
                        for file in files
                    ]
                ).chunk(time_chunks)
            else:
                raise ValueError(
                    f"`files` must be str, tuple, or list, not {type(files)}."
                )
            # Create dict containing import and new variable name
            var_names = {
                dvar["name"]: dvar["rename"] if "rename" in dvar else dvar["name"]
                for dvar in ds_config["variables"]
            }
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
            # Chunk lat and lon dimensions instead of time to prepare merge
            ds = pp.chunk_dataset(ds)
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


def engineer_features(ds: xr.Dataset, config: str | PathLike) -> xr.Dataset:
    """Add features to a dataset according to the passed config YAML."""
    # Loop over all specified preprocessing functions
    for item in config["pipeline"]:
        func = get_func(item)
        # Apply function and convert output to float32
        ds = func(ds).astype("float32")
    return ds.chunk("auto")


if __name__ == "__main__":
    config_file = ROOT_DIR / "config/preprocessing_config.yaml"
    output_dir = ROOT_DIR / "data/processed"

    # Run preprocessing
    main(config_file, output_dir)
    # Inspect zarr stores
    print("Zarr stores structure:")
    zgroup = zarr.open(output_dir)
    print(zgroup.tree())
