#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write the individual model outputs as netCDF file
Help on the usage:
    python scripts/9-write-results-members.py --help
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from omegaconf import OmegaConf

from deeprec.preprocessing import calculate_grace_anomaly
from deeprec.utils import generate_acdd_metadata, month_center_range


def main():
    parser = argparse.ArgumentParser(
        description="Write the unmixed model outputs as netCDF file"
    )
    parser.add_argument(
        "config_file",
        help="File name of YAML configuration that stores the dataset attributes",
    )

    args = parser.parse_args()

    config_file = Path(args.config_file)

    # Load configs
    config_dict = OmegaConf.load(config_file)

    in_store = Path(config_dict.in_store)
    out_dir = Path(config_dict.out_dir)
    coords_attrs = config_dict.coordinates
    data_vars_attrs = config_dict.data_variables
    ds_attrs = config_dict.dataset
    encoding = config_dict.encoding

    # Open dataset containing predictions
    ds = xr.open_zarr(in_store)

    # Convert coordinates:
    #   lat: 90 to -90 → -90 to 90
    #   lon: -180 to 180 → 0 to 360
    ds = ds.assign_coords(lon=(ds.lon) % 360).sortby("lon")
    ds = ds.reindex(lat=-ds.lat)

    # Convert variables from mm to cm
    ds = ds * 0.1

    # Move time stamps to month centers
    times = ds.get_index("time")
    times_centered = month_center_range(times[0], times[-1])
    ds = ds.assign(time=times_centered)

    # Split dataset containing location parameter ("pred_") and scale parameter ("uncertainty_") into two
    ds_loc, ds_scale = split_pred_uncertainty(ds)

    # Convert both datasets to data arrays along new 'member_id' dimension
    da_loc = ds_loc.to_dataarray("member_id", name="laplace_loc")
    da_scale = ds_scale.to_dataarray("member_id", name="laplace_scale")

    # Subtract GRACE baseline from location param
    da_loc = calculate_grace_anomaly(da_loc)

    # Recombine into one dataset
    ds = xr.merge([da_loc, da_scale])
    # Create a integer member ID
    ds = ds.assign_coords(member_id=np.arange(len(ds.member_id)))

    # Combine static (set in config) and dynamic attributes (obtained from ds)
    ds_attrs = {**ds_attrs, **generate_acdd_metadata(ds)}

    # Define main attributes, the rest is sorted alphabetically
    MAIN_KEYS = ["title", "summary", "keywords", "Conventions", "product_version"]
    ds_attrs_main = {k: ds_attrs[k] for k in MAIN_KEYS if k in ds_attrs}
    remaining_keys = set(ds_attrs.keys()) - set(MAIN_KEYS)
    ds_attrs_sorted = {k: ds_attrs[k] for k in sorted(remaining_keys)}

    # Set data variable and coordinate attributes
    ds.attrs = {**ds_attrs_main, **ds_attrs_sorted}
    for var_name, var_attrs in data_vars_attrs.items():
        ds[var_name].attrs = var_attrs
    for coord_name, coord_attrs in coords_attrs.items():
        ds[coord_name].attrs = coord_attrs

    # Set encoding
    ds.dr.set_var_encoding("all", **encoding.general)
    ds.time.encoding = encoding.time

    # Write file
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / (ds_attrs["title"] + ".nc")
    print(f"Writing {file_path}...")
    ds.to_netcdf(file_path, mode="w", engine="h5netcdf")
    print("Successfully completed.")


def split_pred_uncertainty(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Split dataset containing predictions and uncertainties into two datasets"""

    mean_names = [name for name in ds.data_vars if name.startswith("pred_")]
    scale_names = [name for name in ds.data_vars if name.startswith("uncertainty_")]

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
    renamer = {name: "_".join(name.split("_")[1:]) for name in ds.data_vars}
    return ds.rename_vars(renamer)


if __name__ == "__main__":
    # Register dask progress bar
    ProgressBar().register()

    main()
