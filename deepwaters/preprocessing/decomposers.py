"""Trend-seasonal decomposition with LOESS for Xarray DataArrays"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.seasonal import MSTL, STL


def _stl(series: np.ndarray, **stl_kwargs) -> np.ndarray:
    """Perform a seasonal trend decomposition using LOESS (STL) for a single grid cell."""
    # Filter out grid cells which are NaN
    if np.all(np.isnan(series)):
        decomposed = np.full((len(series), 4), np.nan)
    else:
        # result = seasonal_decompose(series, period=12)
        result = STL(series, **stl_kwargs).fit()
        # Stack the decomposed components together for easy handling
        decomposed = np.stack(
            [result.observed, result.trend, result.seasonal, result.resid], axis=-1
        )
    return decomposed


def _mstl(series: np.ndarray, **mstl_kwargs) -> np.ndarray:
    """Perform a multiple seasonal trend decomposition using LOESS (MSTL) for a single
    grid cell."""

    # Filter out grid cells which are NaN
    if np.all(np.isnan(series)):
        ncomps = len(mstl_kwargs["periods"]) + 3
        decomposed = np.full((len(series), ncomps), np.nan)
    else:
        # result = seasonal_decompose(series, period=12)
        result = MSTL(series, **mstl_kwargs).fit()
        # Stack the decomposed components together for easy handling
        decomposed = np.stack(
            [result.observed, result.trend, *result.seasonal.T, result.resid], axis=-1
        )
    return decomposed


def apply_stl(
    da: xr.DataArray,
    period: int = 12,
    seasonal: int = 25,
    trend: int = 37,
    **stl_kwargs,
) -> xr.DataArray:
    """Perform a seasonal-trend decomposition with LOESS (STL)
    for a DataArray with the dimensions (time, lat, lon).
    The returned DataArray has an additional dimension 'component'
    with 'observed', 'trend', 'seasonal', and 'resid' components.

    Parameters
    ----------

    da: DataArray
        The time series data to be decomposed.
    period: int, optional
        Periodicity of the sequence
    seasonal: int, optional
        The length of the seasonal smoother. Must be odd.
    trend: int, optional
        The length of the trend smoother. Must be odd.
        Typically 150 % of seasonal
    stl_kwargs:
        Optional arguments to pass to the STL function.

    Returns
    -------

    DataArray with the dimensions (time, lat, lon, component)
    """
    # Set larger interpolation steps to speed up computation
    seasonal_jump = low_pass_jump = int(0.15 * (period + 1))
    trend_jump = int(0.15 * 1.5 * (period + 1))
    stl_kwargs.setdefault("trend_jump", trend_jump)
    stl_kwargs.setdefault("seasonal_jump", seasonal_jump)
    stl_kwargs.setdefault("low_pass_jump", low_pass_jump)

    decomposed = xr.apply_ufunc(
        _stl,
        da,
        kwargs={"period": period, "seasonal": seasonal, "trend": trend, **stl_kwargs},
        input_core_dims=[["time"]],
        output_core_dims=[["time", "component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"component": 4}},
    )
    # Create component index
    component_idx = pd.CategoricalIndex(["observed", "trend", "seasonal", "resid"])
    decomposed = decomposed.assign_coords(component=component_idx)
    return decomposed


def apply_mstl(
    da: xr.DataArray,
    periods: list[int] | None = None,
    windows: int | list[int] | None = None,
    trend: int | None = None,
    **mstl_kwargs,
) -> xr.DataArray:
    """Perform a multiple seasonal-trend decomposition with LOESS (MSTL)
    for a DataArray with the dimensions (time, lat, lon).
    The returned DataArray has an additional dimension 'component'
    with 'observed', 'trend', 'seasonal', and 'resid' components.

    Parameters
    ----------

    da: DataArray
        The time series data to be decomposed.
    periods: list of ints, optional
        Periodicity of the seasonal components.
        If not provided, uses [6, 12] as default.
    windows: int or list of ints, optional
        Length of the seasonal smoothers for each corresponding period.
        Must be odd. If not provided, uses [61, 61] as default.
    trend: int, optional
        Length of the trend smoothers. Must be odd.
        if not provided, uses maximum of windows as default.
    mstl_kwargs:
        Optional arguments to pass to the STL function.

    Returns
    -------

    DataArray with the dimensions (time, lat, lon, component)
    """

    # Set default periods and windows
    if periods is None:
        periods = [6, 12]
    if windows is None:
        windows = [61] * len(periods)
    elif isinstance(windows, int):
        windows = [windows] * len(periods)

    if len(periods) != len(windows):
        raise ValueError("Length of windows must match length of periods.")

    # Set default trend and add it to stl_kwargs dict
    if trend is None:
        trend = max(windows)
    if "stl_kwargs" in mstl_kwargs:
        mstl_kwargs["stl_kwargs"]["trend"] = trend
    else:
        mstl_kwargs["stl_kwargs"] = {"trend": trend}

    decomposed = xr.apply_ufunc(
        _mstl,
        da,
        kwargs={"periods": periods, "windows": windows, **mstl_kwargs},
        input_core_dims=[["time"]],
        output_core_dims=[["time", "component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"component": 3 + len(periods)}},
    )
    # Create component index
    seasonal_names = [f"seasonal_{period}" for period in periods]
    component_idx = pd.CategoricalIndex(["observed", "trend", *seasonal_names, "resid"])
    decomposed = decomposed.assign_coords(component=component_idx)
    return decomposed


def detrend_vars(
    ds: xr.Dataset,
    names: str | Iterable,
    add_stl_comps: bool = False,
    period: int = 12,
    seasonal: int = 25,
    trend: int = 37,
    **stl_kwargs,
) -> xr.DataArray:
    """Detrend variables from a Dataset using a seasonal-trend
    decomposition with LOESS (STL)

    Parameters
    ----------

    ds: Dataset
        The Dataset containing the data variables to decompose
    names: str or Iterable
        The data variables to decompose
    add_stl_comps: bool, optional
        If True, adds trend, seasonal, and resid as additional
        data variables with the names '{name}_{component}'.
    period: int, optional
        Periodicity of the sequence
    seasonal: int, optional
        The length of the seasonal smoother. Must be odd.
    trend: int, optional
        The length of the trend smoother. Must be odd.
        Typically 150 % of seasonal
    stl_kwargs:
        Optional arguments to pass to the STL function.

    Returns
    -------

    Dataset with additional decomposition data variables


    """

    if isinstance(names, str):
        names = [names]

    for name in names:
        # Perform decomposition
        da = apply_stl(
            ds[name], period=period, seasonal=seasonal, trend=trend, **stl_kwargs
        )
        # Extract components
        da_trend = da.sel(component="trend").drop_vars("component")
        da_seasonal = da.sel(component="seasonal").drop_vars("component")
        da_resid = da.sel(component="resid").drop_vars("component")

        # Add detrended to dataset
        ds[f"{name}_detrended"] = da_seasonal + da_resid

        if add_stl_comps:
            # Add decomposition components
            ds[f"{name}_trend"] = da_trend
            ds[f"{name}_seasonal"] = da_seasonal
            ds[f"{name}_resid"] = da_resid

    return ds
