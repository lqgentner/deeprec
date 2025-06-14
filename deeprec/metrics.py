"""Containes evaluation metrics to apply on Datasets and Data Arrays"""

from typing import Literal

import numpy as np
from numpy import ndarray
from torch import Tensor
import xarray as xr
from xskillscore import (
    mae,
    mape,
    me,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_eff_p_value,
    pearson_r_p_value,
    r2,
    rmse,
    smape,
    spearman_r,
    spearman_r_eff_p_value,
    spearman_r_p_value,
)

from .utils import XrObj


def kge(
    o: XrObj,
    m: xr.DataArray,
    dim: str | list[str] | None = None,
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Original Kling-Gupta Efficiency (KGE) as per
    [Gupta et al., 2009](https://doi.org/10.1016/j.jhydrol.2009.08.003).

    Parameters
    ----------
    o : Dataset or DataArray
        Observed value array(s) over which to apply the function.
    m : DataArray
        Modelled/predicted value array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : {"default", True, False}
        Whether to keep attributes on xarray Datasets/dataarrays after operations. Can be
        - True : to always keep attrs
        - False : to always discard attrs
        - default : to use original logic that attrs should only be kept in unambiguous circumstances

    Returns
    -------
    DataArray or Dataset
        Kling-Gupta Efficiency.

    """
    # GET the PCC
    r = pearson_r(o, m, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    with xr.set_options(keep_attrs=keep_attrs):
        # Calculate error in spread of flow alpha
        alpha = m.std(dim) / o.std(dim)

        # Calculate error in volume beta (bias of mean discharge)
        beta = m.mean(dim) / o.mean(dim)

        # Calculate the Kling-Gupta Efficiency KGE
        kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_


def kgeprime(
    o: XrObj,
    m: xr.DataArray,
    dim: str | list[str] | None = None,
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Modified Kling-Gupta Efficiency (KGE') as per
    [Gupta et al., 2009](https://doi.org/10.1016/j.jhydrol.2009.08.003).

    Parameters
    ----------
    o : Dataset or DataArray
        Observed value array(s) over which to apply the function.
    m : DataArray
        Modelled/predicted value array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : {"default", True, False}
        Whether to keep attributes on xarray Datasets/dataarrays after operations. Can be
        - True : to always keep attrs
        - False : to always discard attrs
        - default : to use original logic that attrs should only be kept in unambiguous circumstances



    Returns
    -------
    DataArray or Dataset
        Modified Kling-Gupta Efficiency.

    """
    # GET the PCC
    r = pearson_r(o, m, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    with xr.set_options(keep_attrs=keep_attrs):
        m_mean = m.mean(dim)
        o_mean = o.mean(dim)
        # calculate error in spread of flow gamma
        # (avoiding cross correlation with bias by dividing by the mean)
        gamma = (m.std(dim) / m_mean) / (o.std(dim) / o_mean)

        # Calculate error in volume beta (bias of mean discharge)
        beta = m_mean / o_mean

        # Calculate the Kling-Gupta Efficiency KGE
        kgeprime_ = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)

    return kgeprime_


def nse(
    o: XrObj,
    m: xr.DataArray,
    dim: str | list[str] | None = "time",
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Nash-Sutcliffe Efficiency (NSE) as per
    [Nash and Sutcliffe, 1970](https://doi.org/10.1016/0022-1694(70)90255-6).
    Values higher than 0 indicate that the model predicts better than the long-term mean.

    Parameters
    ----------
    o : Dataset or DataArray
        Observed value array(s) over which to apply the function.
    m : DataArray
        Modelled/predicted value array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : {"default", True, False}
        Whether to keep attributes on xarray Datasets/dataarrays after operations. Can be
        - True : to always keep attrs
        - False : to always discard attrs
        - default : to use original logic that attrs should only be kept in unambiguous circumstances

    Returns
    -------
    DataArray or Dataset
        Nash-Sutcliffe Efficiency.

    """

    if keep_attrs is False:
        keep_attrs = "default"

    with xr.set_options(keep_attrs=keep_attrs):
        mse_ = mse(o, m, dim=dim, skipna=skipna)
        o_var = o.var(dim=dim, skipna=skipna)
        nse_ = 1 - mse_ / o_var
    return nse_


def nsec(
    o: XrObj,
    m: xr.DataArray,
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Cyclostationary Nash-Sutcliffe Efficiency (NSE). Values higher than 0 indicate that
    the model predicts better than the monthly climatology.
    Only supports averaging over time dimension.

    Parameters
    ----------
    o : Dataset or DataArray
        Observed value array(s) over which to apply the function.
    m : DataArray
        Modelled/predicted value array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : {"default", True, False}
        Whether to keep attributes on xarray Datasets/dataarrays after operations. Can be
        - True : to always keep attrs
        - False : to always discard attrs
        - default : to use original logic that attrs should only be kept in unambiguous circumstances

    Returns
    -------
    DataArray or Dataset
        Cyclostationary Nash-Sutcliffe Efficiency.

    """

    # Calculate the monthly climatology
    o_clim = o.groupby("time.month").mean()
    # Substract the monthly mean from each time step
    o_anom = (o.groupby("time.month") - o_clim).drop_vars("month")
    # Calculate the variance of the anomalies
    o_var = (o_anom**2).mean("time", skipna=skipna)

    with xr.set_options(keep_attrs=keep_attrs):
        mse_ = mse(o, m, dim="time", skipna=skipna)
        nsec_ = 1 - mse_ / o_var
    return nsec_


ALLOWED_METRICS = {
    # xskillscore metrics
    "mae": mae,
    "mape": mape,
    "me": me,
    "median_absolute_error": median_absolute_error,
    "mse": mse,
    "pearson_r": pearson_r,
    "pearson_r_eff_p_value": pearson_r_eff_p_value,
    "pearson_r_p_value": pearson_r_p_value,
    "r2": r2,
    "rmse": rmse,
    "smape": smape,
    "spearman_r": spearman_r,
    "spearman_r_eff_p_value": spearman_r_eff_p_value,
    "spearman_r_p_value": spearman_r_p_value,
    # My custom metrics
    "kge": kge,
    "kgeprime": kgeprime,
    "nse": nse,
    "nsec": nsec,
}


def pred_metric(
    metric: str,
    y_true: xr.DataArray,
    y_pred: xr.DataArray | ndarray,
    skipna: bool = False,
) -> float:
    """Calculate a metric on a ML prediction which has stacked time, lat, and lon dimensions."""

    # Get corresponding metric
    if metric not in ALLOWED_METRICS:
        available_metrics = ", ".join(sorted(ALLOWED_METRICS.keys()))
        raise ValueError(
            f"Metric '{metric}' not allowed. Available metrics: {available_metrics}"
        )
    metric_func = ALLOWED_METRICS[metric]

    # Ensure prediction is xarray DataArray
    if isinstance(y_pred, xr.DataArray):
        pass
    elif isinstance(y_pred, ndarray):
        y_pred = y_true.copy(data=y_pred)
    elif isinstance(y_pred, Tensor):
        y_pred = y_true.copy(data=y_pred.cpu().numpy())
    else:
        raise TypeError(f"`y_pred` type {type(y_pred).__name__} not supported")

    # Unstack ()
    def preprocess(da):
        return (
            da
            # Unstack: ("sample",) -> ("time", "lat", "lon")
            .unstack()
            # Stack: ("time", "lat", "lon") -> ("time", "space")
            .stack(space=["lat", "lon"])
        )

    out = metric_func(preprocess(y_true), preprocess(y_pred), dim="time", skipna=skipna)

    return float(out.mean().values)
