"""This file contains data preprocessing functions"""

import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import MonthBegin
from regionmask import mask_geopandas
from shapely.geometry import box

from ..regions import countries
from ..utils import XrObj, month_center_range


def clean_grid(ds: XrObj) -> XrObj:
    """Convert lat and lon indices to the following format:
    - `lat = [+ 90, ..., - 90]`
    - `lon = [-180, ..., +180]`
    """
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if np.ceil(ds.lon[-1]) == 360:
        # Convert lon from [0, 360] to [-180, 180]
        ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180).sortby("lon")
    if ds.lat[0] < 0:
        # Convert lat from [-90, 90] to [90, -90]
        ds = ds.reindex(lat=-ds.lat)
    return ds


def coarsen_grid(ds: XrObj) -> XrObj:
    """Downsample grid with coarsen.
    Works for 0.25° grid without poles
    [-89.875, -89.625, ...]"""
    return ds.coarsen(lat=2, lon=2).mean()


def reindex_grid(ds: XrObj) -> XrObj:
    """Downsample grid with reindex.
    Works for 0.25° grid with poles
    [-90., 89.75, ...]
    This function does not take the mean,
    but removes every other value."""
    step = 0.5
    lat_05 = np.arange(89.75, -89.75 - step, -step)
    lon_05 = np.arange(-179.75, 179.75 + step, step)
    return ds.reindex(lat=lat_05, lon=lon_05)


def decode_time(ds: xr.Dataset) -> xr.Dataset:
    """Manual implementation of decode time, as
    the Xarray functionality does not handle time dimensions
    that are encoded as 'years since'..."""

    # Locate netCDF time encoding
    keys_to_check = [
        ("encoding", "units"),
        ("encoding", "Units"),
        ("attrs", "units"),
        ("attrs", "Units"),
    ]
    units = None
    for dict_name, key in keys_to_check:
        container = getattr(ds.time, dict_name)
        if key in container:
            units = container[key]
            break
    if units is None:
        raise ValueError("Could not find time encoding.")

    # Unpack netCDF time units
    # CF datetime units follow the format: "UNIT since DATE"
    # this parses out the unit and date allowing for extraneous
    # whitespace.
    matches = re.match(r"(.+) since (.+)", units)
    if not matches:
        raise ValueError(f"invalid time units: {units}")

    delta_units, ref_date = (s.strip() for s in matches.groups())
    ref_date = pd.Timestamp(ref_date)
    # Remove time zone information
    if ref_date.tz is not None:
        ref_date = ref_date.tz_convert(None)
    # Create time index
    if delta_units == "years":
        times = [
            ref_date + pd.DateOffset(years=year)
            for year in ds.time.values.astype("int")
        ]
        time_idx = pd.DatetimeIndex(times, freq="infer")
    elif delta_units == "months":
        times = [
            ref_date + pd.DateOffset(months=month)
            for month in ds.time.values.astype("int")
        ]
        time_idx = pd.DatetimeIndex(times, freq="infer")
    else:
        # Frequencies up to "days" are natively supported by Pandas
        time_idx = pd.DatetimeIndex(
            ref_date + pd.to_timedelta(ds.time, unit=delta_units), freq="infer"
        )
    return ds.assign(time=time_idx)


def yearly2monthly(obj: XrObj) -> XrObj:
    """Reindexes a Dataset or DataArray from a yearly to a monthly time dimension.
    The values are repeated in the forward direction,not interpolated.
    """
    time_yearly = obj.time.values
    if not pd.infer_freq(time_yearly) == "YS-JAN":
        raise ValueError("Object must have time dimension with freq='YS-JAN'.")
    # First date: XXXX-01-01
    time_start = time_yearly[0]
    # Last date: YYYY-12-01
    time_end = time_yearly[-1] + pd.DateOffset(months=11)
    time_monthly = pd.date_range(time_start, time_end, freq="MS")
    return obj.reindex(time=time_monthly, method="ffill")


def extend_time_const(obj: XrObj, end_time: str | pd.Timestamp) -> XrObj:
    """Extend DataArray by repeating the last time step's values until specified end time."""

    # Get the existing times
    times = obj.time.values

    # Detect frequency from the time series
    freq = pd.infer_freq(times)
    if freq is None:
        raise ValueError("Could not infer frequency from time series")

    # Detect if provided end time is too small
    if len(pd.date_range(start=times[-1], end=end_time, freq=freq)) <= 1:
        raise ValueError(
            f"Invalid `end_time` {end_time}. Must be >= last time step + period."
        )

    # Create extended time range
    ext_times = pd.date_range(start=times[0], end=end_time, freq=freq)

    # Return obj with extended time
    return obj.reindex(time=ext_times, method="ffill")


def align_time(
    ds: xr.Dataset, tolerance: pd.Timedelta = pd.Timedelta(days=5)
) -> xr.Dataset:
    """Reindex a dataset with an unevenly spaced time dimension
    (e.g., GRACE products) to a monthly spaced time dimension.
    The values are not interpolated, but taken from the nearest
    timestamp, if there is one closer than or equal to the specified
    tolerance. Otherwise, the values are dropped.

    ### Note
    GRACE timestamps are at the center of their integration period. For example,
    a GRACE product from 2002-04-01 to 2002-04-31 has the timestamp 2002-04-16.
    Contrary to that, all other input datasets use a timestamp at the first day of every month,
    even though their values describe the average of a full calendar month.
    To retain compatibility, the month-centered timestamps are set to the first day of the month *after* the
    reindexing.
    """
    time_orig = pd.Series(ds.time)
    time_cntr = pd.Series(month_center_range(time_orig.min(), time_orig.max()))

    ds = ds.reindex(
        time=time_cntr,
        method="nearest",
        tolerance=tolerance,
        # Drop all timesteps that could not be matched
    ).dropna("time", how="all")

    # Floor all dates to the first day of month
    time_month_start = (
        pd.to_datetime(ds.time) + MonthBegin() - MonthBegin(normalize=True)
    )
    ds = ds.assign(time=time_month_start)
    return ds


def align_time_interp(
    ds: xr.Dataset,
    interp_tolerance: pd.Timedelta = pd.Timedelta(days=31),
    subst_tolerance: pd.Timedelta = pd.Timedelta(days=5),
) -> xr.Dataset:
    """Reindex a dataset with an unevenly spaced time dimension
    (e.g., GRACE products) to a monthly spaced time dimension.
    The values are interpolated, if the previous and next index value is closer
    than `interp_tolerance`. Otherwise, the values are directly substituted, if there is a nearby index
    value (in any direction) closer than `subst_tolerance`.
    timestamp, if there is one closer than or equal to the specified
    tolerance. If a timestamp is not interpolatable or substitutable, the values are dropped.

    ### Note

    GRACE timestamps are at the center of their integration period. For example,
    a GRACE product from 2002-04-01 to 2002-04-31 has the timestamp 2002-04-16.
    Contrary to that, all other input datasets use a timestamp at the first day of every month,
    even though their values describe the average of a full calendar month.
    To retain compatibility, the month-centered timestamps are set to the first day of the month *after* the
    reindexing.
    """

    time_orig = pd.Series(ds.time)
    time_cntr = pd.Series(month_center_range(time_orig.min(), time_orig.max()))

    # Match every month center with next smaller timestamp, if available
    time_df = pd.merge_asof(
        time_cntr.rename("time_cntr"),
        time_orig.rename("time_orig_prev"),
        left_on="time_cntr",
        right_on="time_orig_prev",
        tolerance=interp_tolerance,
        direction="backward",
    )
    # Match every month center with next larger timestamp, if available
    time_df = pd.merge_asof(
        time_df,
        time_orig.rename("time_orig_next"),
        left_on="time_cntr",
        right_on="time_orig_next",
        tolerance=interp_tolerance,
        direction="forward",
    )
    # Match every month center with nearby timestamp, if available
    time_df = pd.merge_asof(
        time_df,
        time_orig.rename("time_orig_near"),
        left_on="time_cntr",
        right_on="time_orig_near",
        tolerance=subst_tolerance,
        direction="nearest",
    )
    # Timesteps which can be interpolated
    can_interp = time_df[["time_orig_prev", "time_orig_next"]].notna().all(axis=1)
    time_interp = time_df.time_cntr[can_interp]

    # Timesteps which can only be substituted
    can_subst = time_df.time_orig_near.notna()
    can_subst_only = can_subst & ~can_interp
    time_subst_only = time_df.time_cntr[can_subst_only]

    # Perform interpolation of timestamps
    ds_interp = ds.interp(time=time_interp)

    # Perform substitution of remaining timestamps
    orig_subst_only = time_df.time_orig_near[can_subst_only].values
    ds_subst = ds.sel(time=orig_subst_only).assign(time=time_subst_only)

    # Combine datasets
    ds = xr.concat([ds_interp, ds_subst], dim="time").sortby("time")

    # Floor all dates to the first day of month
    time_month_start = (
        pd.to_datetime(ds.time) + MonthBegin() - MonthBegin(normalize=True)
    )
    ds = ds.assign(time=time_month_start)

    return ds


def sel_time(ds: XrObj, start: Any = None, end: Any = None) -> XrObj:
    return ds.sel(time=slice(start, end))


def set_dim_attrs(ds: XrObj) -> XrObj:
    dim_attrs = {
        "time": {
            "standard_name": "time",
            "long_name": "Time",
            "axis": "T",
        },
        "lat": {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
        "lon": {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    }
    for dim, attrs in dim_attrs.items():
        if dim in ds.dims:
            ds[dim].attrs = attrs
    return ds


def chunk_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Chunk a dataset dependent on its size. Only the space dimensions are chunked,
    to facilitate later merging and aggregating across the time dimension.
    """

    # Perform chunking
    for var_name in list(ds.data_vars):
        dims = set(ds[var_name].dims)
        # Only chunk data variable if it has 3 dimensions (to prevent very small chunks)
        if dims == {"time", "lat", "lon"}:
            chunks = {"time": -1, "lat": 120, "lon": 120}
        elif dims == {"time"}:
            chunks = {"time": -1}
        elif dims == {"lat", "lon"}:
            chunks = {"lat": -1, "lon": -1}
        else:
            raise ValueError(f"Chunking of dataset dimensions {dims} not implemented.")
        ds[var_name] = ds[var_name].chunk(chunks)
    return ds


def add_nvector(ds: xr.Dataset) -> xr.Dataset:
    """Adds the three components of the normal vector to the Earth's surface
        as data variables to the Dataset.
    For advantages of the n-vector in comparison to the latitude/longitude, refer to
        [Gade, 2010](https://doi.org/10.1017/S0373463309990415)."""

    lat_rad = np.deg2rad(ds.lat)
    lon_rad = np.deg2rad(ds.lon)
    ds = ds.assign(
        nvec_x=np.sin(lat_rad),
        nvec_y=np.sin(lon_rad) * np.cos(lat_rad),
        nvec_z=-np.cos(lon_rad) * np.cos(lat_rad),
    )
    for axis in ["x", "y", "z"]:
        ds[f"nvec_{axis}"].attrs = {
            "standard_name": f"n-vector_{axis}",
            "long_name": f"{axis} Component of n-vector",
            "units": "radians",
        }
    return ds


def add_periodic_time(ds: xr.Dataset, freq: int = 1) -> xr.Dataset:
    """Adds the sine and cosine of the day of year as data variables to the Dataset"""

    # Create data variable names
    varname_cos = "year_cos" if freq == 1 else f"year{freq}_cos"
    varname_sin = "year_sin" if freq == 1 else f"year{freq}_sin"
    # Assign new data variables
    doy = ds.time.dt.dayofyear
    ds[varname_cos] = np.cos(freq * doy * 2 * np.pi / 365.25)
    ds[varname_sin] = np.sin(freq * doy * 2 * np.pi / 365.25)
    # Create attributes
    ds[varname_cos].attrs = {
        "standard_name": f"cosine_of_time_{freq}_per_year",
        "long_name": f"Cosine of Time, {freq}x per year",
        "units": "radians",
    }
    ds[varname_sin].attrs = {
        "standard_name": f"sine_of_time_with_{freq}_per_year",
        "long_name": f"Sine of Time, {freq}x per year",
        "units": "radians",
    }
    return ds


def add_epoch_time(ds: xr.Dataset) -> xr.Dataset:
    """Adds the epoch, the time measured in leap seconds since 1901-01-01.
    Similar to Unix time, only with a different starting date.
    """
    epoch = (ds.get_index("time") - pd.Timestamp("1901-01-01")) / pd.Timedelta("1s")
    return ds.assign(time_epoch=epoch)


def _guess_bounds(points: xr.DataArray) -> xr.DataArray:
    """Guess bounds of grid cells."""
    if not len(points.dims) == 1:
        raise ValueError("Points must only contain one dimension.")
    dim_name = points.dims[0]
    step = points[1] - points[0]

    min_bounds = points - step / 2
    max_bounds = points + step / 2

    bounds = np.array([min_bounds, max_bounds]).transpose()
    da = xr.DataArray(bounds, dims=(dim_name, "bounds"), coords=points.coords)
    return da


def add_cell_area(ds: xr.Dataset) -> xr.Dataset:
    """
    Add spherical segment areas in m².
    Area weights are calculated for each lat/lon cell as:

    $$
        r^2 (lon_1 - lon_0) (sin(lat_1) - sin(lat_0))
    $$

    Taken from SciTools iris library.
    """
    EARTH_RADIUS = 6371000.0  # m

    # fill in a new array of areas
    radius_sqr = EARTH_RADIUS**2
    lat_bounds = np.deg2rad(_guess_bounds(ds.lat))
    lon_bounds = np.deg2rad(_guess_bounds(ds.lon))

    ylen = np.sin(lat_bounds[:, 1]) - np.sin(lat_bounds[:, 0])
    xlen = lon_bounds[:, 1] - lon_bounds[:, 0]
    # Absolute because backwards bounds (min > max) give negative areas
    areas = np.abs(radius_sqr * ylen * xlen)

    return ds.assign(cell_area=areas)


def add_country_exclusion_mask(
    ds: xr.Dataset,
    country_names: list,
    var_name: str = "exclusion_mask",
    buffer_distance: int | float | None = None,
) -> xr.Dataset:
    """Add a mask where to exclude certain countries. All provided countries
    are 0/False, while all remaining grid cells are 1/True."""

    gdf = countries(country_names)

    assert gdf.crs == "EPSG:4326"

    if buffer_distance:
        # Clip the buffered geometries to avoid exceeding geographic limits
        # Define a clipping box (in the same CRS as the buffered data)
        clipping_box = box(-180, -90, 180, 90)
        clipping_box = gpd.GeoSeries([clipping_box], crs=4326).to_crs(crs=3395).iloc[0]
        gdf = (
            # Project to World Mercator which uses meters
            gdf.to_crs(crs=3395)
            # Buffer the geometries
            .buffer(buffer_distance, single_sided=True)
            # Clip the buffered geometries using the intersection
            .intersection(clipping_box)
            # Transform back to original CRS
            .to_crs(crs=4326)
        )
    # Create DataArray filled with zeros
    masked = xr.DataArray(
        np.zeros((ds.sizes["lat"], ds.sizes["lon"])),
        coords={"lat": ds.coords["lat"], "lon": ds.coords["lon"]},
    )
    # Create mask
    mask = mask_geopandas(gdf, masked)
    # Fill all areas which are not part of the specified countries with ones
    masked = masked.where(mask.notnull()).fillna(1)

    return ds.assign({var_name: masked})


def calculate_grace_anomaly(da: XrObj) -> XrObj:
    """Calculate the 'GRACE-style' anomaly with the base 2004.000 - 2009.999."""
    # GRACE baseline for anomaly generation
    baseline = slice("2004", "2009")
    mean = da.sel(time=baseline).mean("time")
    return da - mean


def calculate_anomaly(da: XrObj) -> XrObj:
    """Calculate the anomaly over all available time steps."""
    mean = da.mean("time")
    return da - mean


def set_twsa_attrs(da: xr.DataArray) -> xr.DataArray:
    da.attrs.update(
        {
            "standard_name": "twsa",
            "long_name": "Terrestrial Water Storage Anomaly",
            "units": "mm",
        }
    )
    return da


def cm2mm(da: XrObj) -> XrObj:
    return da * 10


def m2mm(da: XrObj) -> XrObj:
    return da * 1000


def clean_era5_coords(obj: XrObj) -> XrObj:
    """Drop `expver` and `number` coordinates"""
    if "expver" in obj.coords:
        obj = obj.drop_vars("expver")
    if "number" in obj.coords:
        obj = obj.drop_vars("number")
    return obj


def clean_era5_attrs(obj: XrObj) -> XrObj:
    """Removes 'GRIB' keys and unknown values from the attribute list."""

    def clean_attrs(o: XrObj) -> XrObj:
        o.attrs = {
            key: value
            for key, value in obj.attrs.items()
            if not (key.startswith("GRIB_") or value == "unknown")
        }
        return o

    # Clean outer attrs (for both DataArray and Dataset)
    obj = clean_attrs(obj)

    # Clean sub-attrs, if Dataset
    if isinstance(obj, xr.Dataset):
        for da in obj.data_vars.values():
            da = clean_attrs(da)

    return obj


def normalize_time(obj: XrObj) -> XrObj:
    """Convert times to midnight."""
    obj["time"] = pd.to_datetime(obj.time.values).normalize()
    return obj


def calc_oni_from_sst(sst: xr.DataArray) -> xr.DataArray:
    """Calculate the Oceanic Niño Index (ONI) from a gridded sea surface
    temperature (SST) data array."""
    # Calculate El Nino 3.4 (5°N-5°S, 120°W - 170 °W)
    nino34 = (
        sst.sel(lon=slice(190, 240), lat=slice(6, -6))
        .dr.weight_lat()
        .mean(["lat", "lon"])
        .compute()
    )
    # Subtract monthly climatology
    baseline = nino34.sel(time=slice("1991", "2020"))
    climatology = baseline.groupby("time.month").mean("time")
    anomalies = (nino34.groupby("time.month") - climatology).drop_vars("month")
    # ONI is the 3 month rolling mean (centered) of the anomalies
    oni = anomalies.rolling(time=3, center=True).mean().dropna("time")
    return oni
