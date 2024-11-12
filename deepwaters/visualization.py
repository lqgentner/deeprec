import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from cartopy.mpl.geocollection import GeoQuadMesh
from pandas import Timestamp
from xarray.plot.facetgrid import FacetGrid

from deepwaters.regions import basins


def plot_grace_gap(ax: plt.Axes) -> plt.Axes:
    """Add the area of the GRACE/GRACE-FO gap to the axis of a plot.
    time must be on the x-axis.
    """
    return ax.axvspan(
        Timestamp("2017-06-11"),
        Timestamp("2018-06-16"),
        alpha=0.3,
        color="gray",
        label="GRACE/GRACE-FO gap",
    )


def plot_basinwise_map(
    basinwise_obj: xr.DataArray, spatial_obj: xr.DataArray, **plot_kwargs
) -> GeoQuadMesh | FacetGrid:
    """Plot a basin-wise averaged values on a world map.

    Parameters
    ----------

    basinwise_obj: DataArray
        Must have a 'region' dimension.
        Can have additional dimension, if they should be plotted on a FacetGrid
        (by specifying 'col=<dim>' or 'row=<dim>' as keyword argument).
    spatial_obj: DataArray
        Must contain 'lat' and 'lon' dimensions.
        The values are indifferent, only its coordinates are used as reference
        for the spatial extend of the resulting plot.
    kwargs:
        Additional keyword arguments for the plot routine.

    Returns
    -------
    GeoAxes or FacetGrid
        depending of plotting on a single axis or on multiple axis
        (by specifying 'col=<dim>' or 'row=<dim>' as keyword argument)
    """
    # Create spatial array ((lat, lon) / (space,) dimension for plotting on world map)
    # Dims: ("model", "space")
    if "time" in spatial_obj.coords:
        spatial_obj = spatial_obj.isel(time=-1).drop_vars("time")

    basinwise_spatial = xr.full_like(
        spatial_obj.stack(space=("lat", "lon")),
        fill_value=np.nan,
    ).compute()

    # Create region mask
    basin_shapes = basins(top=72)
    regions = regionmask.from_geopandas(
        basin_shapes,
        names="river",
        numbers="mrbid",
        abbrevs="_from_name",
        name="rivername",
    )
    # Dims: ("space",)
    basin_mask = regions.mask(basinwise_spatial, flag="names")

    # Assign RMSE means to spatial array
    for basin in basinwise_obj.region.values:
        value = basinwise_obj.sel(region=basin)
        basinwise_spatial.loc[{"space": basin_mask.cf == basin}] = value
    # Unstack space dimension
    basinwise_spatial = basinwise_spatial.unstack()

    # Assign attributes from basinwise array
    basinwise_spatial.attrs = basinwise_obj.attrs

    # Create plot
    p = basinwise_spatial.dw.projplot(**plot_kwargs)

    return p


def plot_missing_timesteps(
    obj: xr.DataArray,
    other: xr.DataArray | xr.Dataset,
    ax: plt.Axes,
    **plot_kwargs,
) -> None:
    """Plot the time steps which are missing from a data array but are present
    in the other data array as vertical span across an axis."""
    is_time_missing = ~other.time.isin(obj.dropna("time", how="all").time)
    time_missing = obj.time[is_time_missing]

    plot_kwargs.setdefault("facecolor", "gainsboro")
    plot_kwargs.setdefault("edgecolor", "none")
    plot_kwargs.setdefault("zorder", 0)

    for i, time in enumerate(time_missing):
        time = pd.Timestamp(time.values)
        start = time - pd.DateOffset(months=1)
        end = time + pd.DateOffset(months=1)
        if i == 0:
            ax.axvspan(start, end, **plot_kwargs, label="Target gaps")
        else:
            ax.axvspan(start, end, **plot_kwargs)
