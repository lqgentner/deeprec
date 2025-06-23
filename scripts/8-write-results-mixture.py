#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write the mixed model outputs as netCDF file
Help on the usage:
    python scripts/8-write-results-mixture.py --help
"""

import argparse
from pathlib import Path
from typing import Any

from dask.diagnostics import ProgressBar
from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import xarray as xr

from deeprec.utils import generate_acdd_metadata, month_center_range


def main():
    parser = argparse.ArgumentParser(
        description="Write the mixed model outputs as netCDF file"
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
    logger.info("Writing {}...", file_path)
    ds.to_netcdf(file_path, mode="w", engine="h5netcdf")
    logger.info("Successfully completed.")


def create_dynamic_attrs(ds: xr.Dataset) -> dict[str, Any]:
    # Extract dynamic attributes
    TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
    timenow_str = pd.Timestamp.utcnow().strftime(TIME_FORMAT)

    times = ds.get_index("time")
    lats = ds.get_index("lat")
    lons = ds.get_index("lon")

    timestart_str = times[0].strftime(TIME_FORMAT)
    timeend_str = times[-1].strftime(TIME_FORMAT)

    latmin = float(lats[0])
    latmax = float(lats[-1])
    lonmin = float(lons[0])
    lonmax = float(lons[-1])

    # Create dict of attributes
    attrs = {}
    attrs["time_coverage_start"] = timestart_str
    attrs["time_coverage_end"] = timeend_str
    attrs["geospatial_lat_min"] = latmin
    attrs["geospatial_lat_max"] = latmax
    attrs["geospatial_lon_min"] = lonmin
    attrs["geospatial_lon_max"] = lonmax
    attrs["date_created"] = timenow_str

    return attrs


if __name__ == "__main__":
    # Register dask progress bar
    ProgressBar().register()

    main()
