"""Script that cleans other reconstruction products by Humphrey, Li, and Yin"""

from pathlib import Path

import janitor  # noqa
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

from deepwaters.utils import ROOT_DIR
from deepwaters.preprocessing import calculate_grace_anomaly

CHUNKS = chunks = {"time": -1, "lat": 120, "lon": 120}


def main():
    # Data paths

    out_store = ROOT_DIR / "data/processed/reconstructions.zarr"

    hum_dir = (
        ROOT_DIR
        / "data/raw/reconstructions/humphrey"
        / "01_monthly_grids_ensemble_means_allmodels"
    )
    hum_gsfc = hum_dir / "GRACE_REC_v03_GSFC_ERA5_monthly_ensemble_mean.nc"
    hum_jpl = hum_dir / "GRACE_REC_v03_JPL_ERA5_monthly_ensemble_mean.nc"

    li_csr = ROOT_DIR / "data/raw/reconstructions/li/GRID_CSR_GRACE_REC.mat"

    yin_dir = ROOT_DIR / "data/raw/reconstructions/yin"
    yin_gsfc = yin_dir / "GSFC-based GTWS-MLrec TWS.nc"
    yin_csr = yin_dir / "CSR-based GTWS-MLrec TWS.nc"
    yin_jpl = yin_dir / "JPL-based GTWS-MLrec TWS.nc"

    # Perform cleaning
    print("Prepare reconstructions...")
    ds_list = [
        clean_humphrey(hum_gsfc, "gsfc"),
        clean_humphrey(hum_jpl, "jpl"),
        clean_li(li_csr, "csr"),
        clean_yin(yin_gsfc, "gsfc"),
        clean_yin(yin_csr, "csr"),
        clean_yin(yin_jpl, "jpl"),
    ]
    recs = xr.merge(ds_list).chunk(CHUNKS).astype("float32")

    # Save to Zarr store
    print("Write Reconstructions to Zarr store:")
    delayed = recs.to_zarr(out_store, mode="w", compute=False)
    with ProgressBar():
        delayed.compute()

    # Print Zarr structure
    print("Zarr store structure:")
    zgroup = zarr.open(out_store)
    print(zgroup.tree())


def clean_humphrey(file: Path, target_name: str) -> xr.Dataset:
    """Clean Humphrey's reconstructions"""
    ds = xr.open_dataset(file).chunk(CHUNKS)
    # Time: 15th to 1st of month
    ds["time"] = ds.get_index("time") - pd.offsets.MonthBegin()
    ds = (
        ds.assign(rec_detrend=ds.rec_ensemble_mean + ds.rec_seasonal_cycle)
        .drop_vars(["rec_ensemble_p05", "rec_ensemble_p95", "rec_seasonal_cycle"])
        .rename(
            rec_detrend=f"humphrey_{target_name}_detrend",
            rec_ensemble_mean=f"humphrey_{target_name}_deseason",
        )
    )
    # Calculate the GRACE anomaly (2004 - 2009)
    ds = calculate_grace_anomaly(ds)

    return ds


def clean_li(file: Path, target_name: str) -> xr.Dataset:
    """Clean Li's reconstruction"""
    time_idx = pd.date_range("1979-07-01", "2020-06-01", freq="MS")
    # Open Matlab file and fix coordinates
    ds = (
        xr.open_dataset(file, engine="h5netcdf", phony_dims="sort")
        .rename(
            {
                "phony_dim_0": "time",
                "phony_dim_1": "long",
                "phony_dim_2": "lat",
                "grace_rec_deseasonalised": f"li_{target_name}_deseason",
                "grace_rec_detrended": f"li_{target_name}_detrend",
                "grace_rec_full": f"li_{target_name}_full",
            }
        )
        .squeeze()
        .rename(long="lon")
        .chunk(CHUNKS)
        .assign_coords(time=time_idx)
        .set_index(lat="lat", lon="lon", time="time")
        .transpose("time", "lat", "lon")
        .drop_vars(["str_month", "str_year"])
        .drop_encoding()
    )
    # [0, 360] to [-180, 180]
    ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180).sortby("lon")
    # Centimeters to millimeters
    ds *= 10
    # Calculate the GRACE anomaly (2004 - 2009)
    ds = calculate_grace_anomaly(ds)

    return ds


def clean_yin(file: Path, target_name: str) -> xr.Dataset:
    """Clean Yin's reconstructions"""

    step = 0.25
    lat_idx = np.arange(90, -90 - step, -step)
    lon_idx = np.arange(0, 359.75 + step, step)
    time_idx = pd.date_range("1940-01-01", "2022-12-01", freq="MS")

    ds = (
        xr.open_dataset(file)
        .rename(x="lon", y="lat", z="time", TWSA="twsa")
        .drop_dims("zz")
        .chunk(CHUNKS)
        .assign_coords(time=time_idx)
        .assign_coords(lat=lat_idx)
        .assign_coords(lon=(lon_idx + 180) % 360 - 180)
        .sortby("lon")
    )

    # Downsample to 0.5°
    ds = ds.coarsen(lat=2, lon=2, boundary="trim").mean()
    # Workaround for different grid cell steps
    ds = ds.assign(lat=ds.lat - 0.125, lon=ds.lon + 0.125)

    # Fill missing land areas with zeros
    # (Land is what both JPL and GSFC define as land)
    targets_zarr = ROOT_DIR / "data/processed/targets.zarr"
    tgts = xr.open_zarr(targets_zarr)
    land_mask = tgts.land_mask_jpl * tgts.land_mask_gsfc

    ds = ds.assign(twsa_zerofill=ds.twsa.fillna(0).where(land_mask == 1))
    ds = ds.rename(
        twsa=f"yin_{target_name}_full",
        twsa_zerofill=f"yin_{target_name}_zerofill",
    )
    # Calculate the GRACE anomaly (2004 - 2009)
    ds = calculate_grace_anomaly(ds)

    return ds


if __name__ == "__main__":
    main()
