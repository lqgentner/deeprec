import xarray as xr
import numpy as np
from dask.diagnostics.progress import ProgressBar

from deepwaters.utils import ROOT_DIR
import deepwaters.preprocessing as pp


def main() -> None:
    TREND_YRS = 5
    period = 12
    # trend usually is 150 % of seasonal
    trend = 12 * TREND_YRS
    seasonal = round(trend / 1.5)

    dims = ("time", "lat", "lon", ...)

    in_file = (
        ROOT_DIR
        / "data/raw/inputs/watergap22e"
        / "watergap22e_gswp3-era5_tws_histsoc_monthly_1901_2022.nc"
    )
    out_file = (
        ROOT_DIR
        / "data/interim/inputs/watergap22e"
        / f"watergap22e_gswp3-era5_twsa-detrended-{TREND_YRS}yrs_monthly_1901_2022.nc"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(in_file, decode_times=False).chunk({"lat": 90, "lon": 90})
    ds = pp.decode_time(ds)
    # Clean dimensions
    ds = pp.clean_grid(ds).transpose(*dims)
    # Calculate anomaly for GRACE baseline
    ds["tws"] = pp.set_twsa_attrs(pp.calculate_grace_anomaly(ds.tws))
    # Decompose TWSA
    ds = pp.detrend_vars(
        ds, names=["tws"], period=period, seasonal=seasonal + 1, trend=trend + 1
    )
    # Rename resulting vars
    ds = ds.rename({"tws": "twsa", "tws_detrended": f"twsa_detrended_{TREND_YRS}yrs"})
    # Save modified dataset
    delayed = (
        ds.drop_encoding()
        .dw.set_var_encoding(
            "all", compression="gzip", compression_opts=4, dtype=np.dtype("float32")
        )
        .to_netcdf(out_file, engine="h5netcdf", compute=False)
    )
    # Perform computation
    with ProgressBar():
        delayed.compute()


if __name__ == "__main__":
    main()
