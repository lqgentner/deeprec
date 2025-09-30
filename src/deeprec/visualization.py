from typing import Hashable

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.geocollection import GeoQuadMesh
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Timestamp
import regionmask
import xarray as xr
from xarray.plot.facetgrid import FacetGrid

from .regions import basins
from .utils import verify_dim_ispresent


def plot_grace_gap(ax: plt.Axes) -> Rectangle:
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
    basinwise_obj: xr.DataArray, spatial_obj: xr.DataArray, **kwargs
) -> GeoQuadMesh | FacetGrid:
    """Plot a basin-wise averaged values on a world map.
    Passing a spatial dummy DataArray with "lat" and "lon" dimension is
    required to specify the grid where the basin-wise averages should be
    plotted on. Recommended use via custom accessor:
    xr.DataArray.dr.projplot_basins()

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
        Additional keyword arguments passed to the `xarray.DataArray.plot` call

    Returns
    -------
    GeoAxes or FacetGrid
        depending of plotting on a single axis or on multiple axis
        (by specifying 'col=<dim>' or 'row=<dim>' as keyword argument)
    """
    # Verify dimensions
    verify_dim_ispresent(basinwise_obj, "region")
    verify_dim_ispresent(spatial_obj, "lat")
    verify_dim_ispresent(spatial_obj, "lon")

    # Create spatial array ((lat, lon) / (space,) dimension for plotting on world map)
    # Dims: ("model", "space")

    if "time" in spatial_obj.dims:
        spatial_obj = spatial_obj.isel(time=-1).drop_vars("time")

    basinwise_spatial = xr.full_like(
        spatial_obj.stack(space=("lat", "lon")),
        fill_value=np.nan,
    ).compute()

    # Extract basin names from array and convert to list
    basin_names = basinwise_obj.region.values.tolist()
    # Create region mask with same basins as basinwise_obj
    basin_shapes = basins(names=basin_names)
    regions = regionmask.from_geopandas(
        basin_shapes, names="riverbasin", name="riverbasin", overlap=False
    )
    # Dims: ("space",)
    basin_mask = regions.mask(basinwise_spatial, flag="names")

    # Assign mean values to spatial array
    for basin in basin_names:
        value = basinwise_obj.sel(region=basin)
        basinwise_spatial.loc[{"space": basin_mask.cf == basin}] = value
    # Unstack space dimension
    basinwise_spatial = basinwise_spatial.unstack()

    # Assign attributes from basinwise array
    basinwise_spatial.attrs = basinwise_obj.attrs

    # Create plot
    p = basinwise_spatial.dr.projplot(**kwargs)

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


def projplot_facet(
    da: xr.DataArray,
    crs: ccrs.CRS = ccrs.PlateCarree(),
    projection: ccrs.CRS = ccrs.EqualEarth(),
    row: Hashable | None = None,
    col: Hashable | None = None,
    col_wrap: int | None = None,
    global_extent: bool = True,
    coastlines: bool = False,
    gridlines: bool = False,
    coastlines_kwargs: dict | None = None,
    gridlines_kwargs: dict | None = None,
    **kwargs,
) -> FacetGrid:
    """Create a multi-axes projected plot using a FacetGrid.
    Used via custom accessor: xr.DataArray.dr.projplot()"""

    p = da.plot(
        transform=crs,
        row=row,
        col=col,
        col_wrap=col_wrap,
        subplot_kws={"projection": projection},
        **kwargs,
    )
    if global_extent:
        for ax in p.axs.flat:
            ax.set_global()
    if coastlines:
        for ax in p.axs.flat:
            ax.coastlines(**coastlines_kwargs)
    if gridlines:
        for ax in p.axs.flat:
            ax.gridlines(**gridlines_kwargs)

    return p


def projplot_single(
    da: xr.DataArray,
    crs: ccrs.CRS = ccrs.PlateCarree(),
    projection: ccrs.CRS = ccrs.EqualEarth(),
    global_extent: bool = True,
    coastlines: bool = False,
    gridlines: bool = False,
    coastlines_kwargs: dict | None = None,
    gridlines_kwargs: dict | None = None,
    ax: GeoAxes = None,
    **kwargs,
) -> GeoQuadMesh:
    """Create a single-axis projected plot.
    Used via custom accessor: xr.DataArray.dr.projplot()"""

    if ax:
        p = da.plot(transform=crs, ax=ax, **kwargs)
    else:
        p = da.plot(
            transform=crs,
            subplot_kws={"projection": projection},
            **kwargs,
        )

    if global_extent:
        p.axes.set_global()
    if coastlines:
        p.axes.coastlines(**coastlines_kwargs)
    if gridlines:
        p.axes.gridlines(**gridlines_kwargs)

    return p
