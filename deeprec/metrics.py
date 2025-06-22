"""Containes evaluation metrics to apply on Datasets and Data Arrays"""

from typing import Literal

import xarray as xr

from deeprec.utils import XrObj


def mae(
    o: XrObj,
    m: xr.DataArray,
    dim: str | list[str] | None = None,
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Mean Absolute Error.

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
        Mean Absolute Error

    """
    with xr.set_options(keep_attrs=keep_attrs):
        ae = xr.ufuncs.abs(o - m)
        mae_ = ae.mean(dim, skipna=skipna)

    return mae_


def mse(
    o: XrObj,
    m: xr.DataArray,
    dim: str | list[str] | None = None,
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Mean Squared Error.

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
        Mean Squared Error

    """
    with xr.set_options(keep_attrs=keep_attrs):
        se = (o - m) ** 2
        mse_ = se.mean(dim, skipna=skipna)

    return mse_


def rmse(
    o: XrObj,
    m: xr.DataArray,
    dim: str | list[str] | None = None,
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Root Mean Squared Error.

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
        Root Mean Squared Error

    """
    mse_ = mse(o, m, dim=dim, skipna=skipna, keep_attrs=keep_attrs)
    with xr.set_options(keep_attrs=keep_attrs):
        rmse_ = xr.ufuncs.sqrt(mse_)

    return rmse_


def my_pearson_r(
    o: XrObj,
    m: xr.DataArray,
    dim: str | list[str] | None = None,
    skipna: bool = False,
    keep_attrs: bool | Literal["default"] = "default",
) -> XrObj:
    """
    Pearson's correlation coefficient.

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
        Pearson's correlation coefficient

    """
    with xr.set_options(keep_attrs=keep_attrs):
        if isinstance(m, xr.Dataset):
            # xr.cov() only accepts DataArrays, convert first
            da_m = m.to_dataarray("model")
            da_cov = xr.cov(o, da_m, dim=dim, ddof=0)
            cov = da_cov.to_dataset("model")
        else:
            cov = xr.cov(o, m, dim=dim, ddof=0)

        r = cov / (o.std(dim, skipna=skipna) * m.std(dim, skipna=skipna))

    return r


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
    r = pearson_r(o, m, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    with xr.set_options(keep_attrs=keep_attrs):
        alpha = m.std(dim, skipna=skipna) / o.std(dim, skipna=skipna)
        beta = m.mean(dim, skipna=skipna) / o.mean(dim, skipna=skipna)
        kge_ = 1 - xr.ufuncs.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

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
    [Gupta et al., 2012](https://doi.org/10.1016%2Fj.jhydrol.2012.01.011).

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
    r = pearson_r(o, m, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    with xr.set_options(keep_attrs=keep_attrs):
        m_mean = m.mean(dim, skipna=skipna)
        o_mean = o.mean(dim, skipna=skipna)
        gamma = (m.std(dim, skipna=skipna) / m_mean) / (
            o.std(dim, skipna=skipna) / o_mean
        )
        beta = m_mean / o_mean
        kgeprime_ = 1 - xr.ufuncs.sqrt(
            (r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2
        )

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
