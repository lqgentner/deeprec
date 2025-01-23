#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write the mixed model outputs as netCDF file
Help on the usage:
    python scripts/8-write-results-mixed.py --help
"""

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from deeprec.utils import month_center_range

PRODUCT_VERSION = "v1.0"

ATTRS_TIME = {
    "standard_name": "time",
    "long_name": "Time",
    "axis": "T",
    "coverage_content_type": "coordinate",
}
ATTRS_LAT = {
    "standard_name": "latitude",
    "long_name": "Latitude",
    "units": "degrees_north",
    "axis": "Y",
    "coverage_content_type": "coordinate",
}

ATTRS_LON = {
    "standard_name": "longitude",
    "long_name": "Longitude",
    "units": "degrees_east",
    "axis": "X",
    "coverage_content_type": "coordinate",
}
ATTRS_TWSA = {
    "standard_name": "lwe_thickness",
    "long_name": "TWS anomaly in liquid water equivalent thickness",
    "units": "cm",
    "coverage_content_type": "modelResult",
}
ATTRS_SIG = {
    "standard_name": "uncertainty_total",
    "long_name": "Total uncertainty",
    "units": "cm",
    "comment": "1-sigma predictive uncertainty, obtained using a deep ensemble with 5 members",
    "coverage_content_type": "modelResult",
}
ATTRS_SIG_ALE = {
    "standard_name": "uncertainty_aleatoric",
    "long_name": "Aleatoric uncertainty",
    "units": "cm",
    "comment": "1-sigma predictive uncertainty, obtained using a deep ensemble with 5 members",
    "coverage_content_type": "modelResult",
}
ATTRS_SIG_EPI = {
    "standard_name": "uncertainty_epistemic",
    "long_name": "Epistemic uncertainty",
    "units": "cm",
    "comment": "1-sigma predictive uncertainty, obtained using a deep ensemble with 5 members",
    "coverage_content_type": "modelResult",
}
ENCODING = {
    "dtype": np.dtype("float32"),
    "zlib": True,
    "complevel": 6,
}
ENCODING_TIME = {
    "dtype": np.dtype("float32"),
    "units": "hours since 1901-01-01T00:00:00Z",
    "calendar": "gregorian",
}


def main():
    parser = argparse.ArgumentParser(
        description="Write the mixed model outputs as netCDF file"
    )
    parser.add_argument(
        "in_store",
        help="Input Zarr store path that contains mixed predictions.",
    )
    parser.add_argument(
        "out_dir", help="Output directory where netCDF should be placed."
    )
    parser.add_argument(
        "-i",
        "--input_config",
        help="The input configuration used",
        choices=["era", "era-rdcd", "wghm-era"],
    )

    args = parser.parse_args()

    # Open dataset containing predictions
    ds = xr.open_zarr(args.in_store)

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

    # Set attributes
    attrs_ds = generate_attrs(ds, args.input_config)
    ds.attrs = attrs_ds
    ds.time.attrs = ATTRS_TIME
    ds.lat.attrs = ATTRS_LAT
    ds.lon.attrs = ATTRS_LON
    ds.twsa.attrs = ATTRS_TWSA
    ds.sigma.attrs = ATTRS_SIG
    ds.sigma_ale.attrs = ATTRS_SIG_ALE
    ds.sigma_epi.attrs = ATTRS_SIG_EPI

    # Set encoding
    ds.dr.set_var_encoding("all", **ENCODING)
    ds.time.encoding = ENCODING_TIME

    # Write file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / (attrs_ds["title"] + ".nc")
    print(f"Writing {file_path}...")
    ds.to_netcdf(file_path, mode="w", engine="h5netcdf")
    print("Successfully completed.")


def generate_attrs(
    ds: xr.Dataset, input_config: Literal["era", "era-rdcd", "wghm-era"]
) -> dict[str, Any]:
    timeformat = "%Y-%m-%dT%H:%M:%SZ"
    timenow_str = pd.Timestamp.utcnow().strftime(timeformat)

    times = ds.get_index("time")
    lats = ds.get_index("lat")
    lons = ds.get_index("lon")

    timestart_str = times[0].strftime(timeformat)
    timeend_str = times[-1].strftime(timeformat)

    latmin = lats[0]
    latmax = lats[-1]
    lonmin = lons[0]
    lonmax = lons[-1]

    # Create dicts of attributes that depend on input configuration
    era5_vars = [
        "d2m",
        "t2m",
        "e",
        "lai_hv",
        "lai_lv",
        "pev",
        "sro",
        "ssro",
        "sp",
        "tp",
        "swvl1",
        "swvl2",
        "swvl3",
        "swvl4",
    ]
    era5_rdcd_vars = [
        "t2m",
        "sp",
        "tp",
        "sd",
    ]
    era5_vars.sort()
    era5_rdcd_vars.sort()
    era5_str = ", ".join(era5_vars)
    era5_rdcd_str = ", ".join(era5_rdcd_vars)

    titles = {
        "era-rdcd": f"DeepRec_CSR_ERA5x{len(era5_rdcd_vars)}+HI_ensemble-mixture",
        "era": f"DeepRec_CSR_ERA5x{len(era5_vars)}+HI_ensemble-mixture",
        "wghm-era": f"DeepRec_CSR_WGHM+ERA5x{len(era5_vars)}+HI_ensemble-mixture",
    }

    inputs = {
        "era-rdcd": (
            f"ERA5 single levels monthly means ({era5_rdcd_str}), "
            + "ISIMIP land use (irrigated cropland, rainfed cropland, pastures, urban areas), "
            + "ISIMIP lake area fraction, "
            + "NOAA ERSSTv5 Oceanic Niño Index"
        ),
        "era": (
            f"ERA5 single levels monthly means ({era5_str}), "
            + "ISIMIP land use (irrigated cropland, rainfed cropland, pastures, urban areas), "
            + "ISIMIP lake area fraction, "
            + "NOAA ERSSTv5 Oceanic Niño Index"
        ),
        "wghm-era": (
            "WaterGAP v2.2e 20CRv3-ERA5 TWS, "
            + f"ERA5 single levels monthly means ({era5_str}), "
            + "ISIMIP land use (irrigated cropland, rainfed cropland, pastures, urban areas), "
            + "ISIMIP lake area fraction, "
            + "NOAA ERSSTv5 Oceanic Niño Index"
        ),
    }

    attrs = {
        "Conventions": "ACDD-1.3",
        "title": titles[input_config],
        "product_version": PRODUCT_VERSION,
        "summary": "Reconstructed GRACE TWS anomalies using CSR mascon solutions as target",
        "keywords": "GRACE, gravity, terrestrial water storage anomaly",
        "institution": "Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland",
        "creator_name": "Luis Q. Gentner",
        "creator_email": "luis.gentner@outlook.com",
        "creator_type": "person",
        "time_coverage_start": timestart_str,
        "time_coverage_end": timeend_str,
        "time_coverage_resolution": "P1M",
        "time_mean_removed": "2004.000 to 2009.999",
        "geospatial_lat_min": latmin,
        "geospatial_lat_max": latmax,
        "geospatial_lat_units": "degree_north",
        "geospatial_lat_resolution": "0.5 degree",
        "geospatial_lon_min": lonmin,
        "geospatial_lon_max": lonmax,
        "geospatial_lon_units": "degree_east",
        "geospatial_lon_resolution": "0.5 degree",
        "model_target": "CSR GRACE/GRACE-FO RL06.3 mascon solutions with all corrections",
        "model_input": inputs[input_config],
        "date_created": timenow_str,
        "comment": "The ground truth of this reconstruction, the CSR GRACE RL06.3 mascon product, "
        + "was downsampled to 0.5° before it was used. "
        + "Only grid cells were reconstructed that are considered as land "
        + "by both JPL and GSFC mascon land masks and are not part of Greenland or Antarctica.",
    }
    return attrs


if __name__ == "__main__":
    # Register dask progress bar
    ProgressBar().register()

    main()
