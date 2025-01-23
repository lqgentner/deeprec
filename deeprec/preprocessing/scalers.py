"""Implementation of scalers for Xarray, inspired by Scikit-Learn"""

import numpy as np
import xarray as xr


class AbstractScaler:
    def __init__(self) -> None:
        self.center = None
        self.scale = None
        # Save common dtype if all dtypes in dataset are identical
        self._common_dtype = None

    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        """Fit the scaler on a Xarray Dataset"""
        if self.center is None:
            raise AttributeError(
                "This scaler instance is not fitted yet. Call 'fit' with appropriate arguments before using 'transform'."
            )

        ds = (ds - self.center) / self.scale

        if self._common_dtype:
            ds = ds.astype(self._common_dtype)

        return ds


class StandardScaler(AbstractScaler):
    """Scale a dataset by removing the mean and scaling to the standard deviation,
    according to Scikit-Learn's `StandardScaler`."""

    def fit(self, ds: xr.Dataset) -> None:
        self._common_dtype = _get_common_dtype(ds)
        self.center = ds.mean()
        scale = ds.std()
        self.scale = _handle_zeros_in_scale(scale)


class RobustScaler(AbstractScaler):
    """Scale a dataset by removing the median and scaling to the interquartile range (IQR),
    according to Scikit-Learn's `RobustScaler`."""

    def fit(self, ds: xr.Dataset) -> None:
        self._common_dtype = _get_common_dtype(ds)
        self.center = ds.median()
        scale = ds.quantile(0.75) - ds.quantile(0.25)
        self.scale = _handle_zeros_in_scale(scale)


def _get_common_dtype(ds: xr.Dataset) -> np.dtype | None:
    """Returns the dtype of the Dataset, if it is identical across all data variables. Returns None if not."""

    common_dtype = None
    dvars = list(ds.data_vars)
    if all(ds[dvar].dtype == ds[dvars[0]].dtype for dvar in dvars):
        common_dtype = ds[dvars[0]].dtype
    return common_dtype


def _handle_zeros_in_scale(scale: xr.Dataset) -> xr.Dataset:
    """Set scales of near constant features to 1 to avoid division by very small values."""
    EPS = 10 * np.finfo("float32").eps
    for dvar in scale:
        if scale[dvar] < EPS:
            scale[dvar] = 1.0
    return scale
