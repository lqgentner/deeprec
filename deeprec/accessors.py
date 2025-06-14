from collections.abc import Iterable
from typing import Any, Generic, Hashable, Literal

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.geocollection import GeoQuadMesh
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from xarray.computation.weighted import DataArrayWeighted, DatasetWeighted
from xarray.core.formatting import dim_summary
from xarray.plot.facetgrid import FacetGrid

from . import regions
from .utils import XrObj
from .visualization import plot_basinwise_map, projplot_facet, projplot_single


# (Geo-)Pandas accessor
@pd.api.extensions.register_dataframe_accessor("dr")
class PandasAccessor:
    """The DeepRec accessor for GeoPandas dataframes."""

    def __init__(self, geopandas_obj: gpd.GeoDataFrame):
        self._validate(geopandas_obj)
        self._obj = geopandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, gpd.GeoDataFrame):
            raise TypeError("Passed object must be a GeoDataFrame.")

    def projplot(
        self,
        crs: ccrs.CRS = ccrs.PlateCarree(),
        projection: ccrs.CRS = ccrs.EqualEarth(),
        global_extent: bool = True,
        coastlines: bool = False,
        gridlines: bool = False,
        coastlines_kwargs: dict[str, Any] | None = None,
        gridlines_kwargs: dict[str, Any] | None = None,
        ax: GeoAxes | None = None,
        **kwargs: dict[str, Any],
    ) -> GeoAxes:
        """
        Create a projected plot of shapley geometries.

        Parameters
        ----------

        crs: cartopy.CRS, default: PlateCarree
            Coordinate reference system of the data.

        projection: cartopy.CRS, default: EqualEarth
            Coordinate reference system of the plot

        global_extent: bool, default: True
            Whether to expand the plot to span the whole globe

        coastlines: bool, default: False
            Whether to add coastlines outlines.

        gridlines: bool, default: False
            Whether to add gridlines.

        coastlines_kwargs: dict, optional
            Dict with keywords passed to the
            `cartopy.mpl.geoaxes.GeoAxes.coastlines` call.

        gridlines_kwargs: dict, optional
            Dict with keywords passed to the
            `cartopy.mpl.geoaxes.GeoAxes.coastlines` call.

        **kwargs: optional
            Additional keyword arguments passed to the
            `cartopy.mpl.geoaxes.GeoAxes.add_geometries` call.

        Returns
        -------

        A GeoAxis object

        """

        if ax is None:
            ax = GeoAxes(projection=projection)

        ax.add_geometries(self._obj.geometry, crs=crs, **kwargs)

        if global_extent:
            ax.set_global()

        if coastlines:
            ax.coastlines(**coastlines_kwargs)

        if gridlines:
            ax.gridlines(**gridlines_kwargs)
        return ax


# Xarray accessor class for methods share between the DataArray and Dataset accessors
class XrBaseAccessor(Generic[XrObj]):
    """The DeepRec Xarray accessor base class"""

    def __init__(self, xarray_obj: XrObj):
        self._obj: XrObj = xarray_obj

    def crop_notnull(self) -> XrObj:
        """
        Crops dataset/data array by cutting-off NaNs that surround the data variables
        in lat and lon dimension. NaNs between not-NaN-values are not removed.
        Cropped lat/lon dimensions are convenient for plotting.
        """

        # Remove all NaNs
        obj_notna = self._obj.dropna("lat", how="all").dropna("lon", how="all")
        # Find lowest and highest not-NaN value
        lon_bounds = (obj_notna.lon.min(), obj_notna.lon.max())
        # lat dimension is descending
        lat_bounds = (obj_notna.lat.max(), obj_notna.lat.min())
        # Return cropped dataset
        return self._obj.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

    def select_basins(
        self,
        names: str | list[str] | None = None,
        top: int | None = None,
        return_region: bool = True,
        drop: bool = False,
    ) -> XrObj:
        """
        Select a Dataset / DataArray by basins.

        Parameters
        ----------

        names: string or list of strings, optional
            The names of river basins to select.
            If None, top must be provided.

        top: int, optional
            The number biggest basins to select.
            If None, names must be provided.

        return_region: bool, default: True
            If True, the returned dataframe has a `region` dimension.

            With a region dimesnion calculating weighted basin averages
            is more convenient, without one plotting is easier.

        drop: bool, default: False
            If True, coordinate labels outside of the river basins
            of the condition are dropped from the result. Only works
            if a single basin is selected


            Returns
            -------

            Dataset or DataArray

        """

        return regions.select_basins(self._obj, names, top, return_region, drop)

    def select_countries(
        self,
        names: str | list[str] | None = None,
        return_region: bool = True,
        drop: bool = False,
    ) -> XrObj:
        """
        Select a Dataset / DataArray by countries.

        Parameters
        ----------

        names: string or list of strings, optional
            The countries names to select.
            If None, all countries are selected.

        return_region: bool, default: True
            If True, the returned dataframe has a `region` dimension.

            With a region dimension calculating weighted basin averages
            is more convenient, without one plotting is easier.

        drop: bool, default: False
            If True, coordinate labels outside of the country
            are dropped from the result.


        Returns
        -------

        Dataset or DataArray

        """

        return regions.select_countries(self._obj, names, return_region, drop)

    def select_continents(
        self,
        names: str | list[str] | None = None,
        return_region: bool = True,
        drop: bool = False,
    ) -> XrObj:
        """
        Select a Dataset / DataArray by continents.

        Parameters
        ----------

        names: string or list of strings, optional
            The continents names to select.
            If None, all continents are selected.

        return_region: bool, default: True
            If True, the returned dataframe has a `region` dimension.

            With a region dimension calculating weighted basin averages
            is more convenient, without one plotting is easier.

        drop: bool, default: False
            If True, coordinate labels outside of the continent
            are dropped from the result.


        Returns
        -------

        Dataset or DataArray

        """

        return regions.select_continents(self._obj, names, return_region, drop)

    def insert_grace_gap_nans(self) -> XrObj:
        """
        Inserts a timestamp filled with NaNs during the GRACE/GRACE-FO gap (on 2018-01-01).
        This enables two independent lines for GRACE and GRACE-FO when creating a time series
        line plot.
        """

        extended_time = (
            self._obj.get_index("time")
            .append(pd.DatetimeIndex(["2018-01-01"]))
            .sort_values()
        )
        return self._obj.reindex(time=extended_time)

    def stack_spacetime(self, how_dropna: Literal["any", "all"] = "any") -> XrObj:
        """Creates a 1D dataset or data array from a 3D dataset or data array by stacking the dimensions ('time', 'lat', 'lon')
        to one 'sample' dimension."""
        return self._obj.stack(sample=("time", "lat", "lon")).dropna(
            "sample", how=how_dropna
        )

    def unstack_spacetime(self) -> XrObj:
        """Unstacks and sorts the ("time", "lat", "lon") dimensions of the
        provided dataset or data array."""
        return self._obj.unstack().sortby("lat", ascending=False).sortby("lon")

    @property
    def gbytes(self) -> float:
        """Total gigabytes consumed by the dataset / data array"""
        return self._obj.nbytes / 1e9

    @property
    def dim_summary(self) -> str:
        """Returns the dimensions and their lengths of a dataset / data array"""
        return dim_summary(self._obj)


# Xarray DataArray accessor
@xr.register_dataarray_accessor("dr")
class XrDataArrayAccessor(XrBaseAccessor):
    """The DeepRec accessor for xarray data arrays."""

    def projplot(
        self,
        crs: ccrs.CRS = ccrs.PlateCarree(),
        projection: ccrs.CRS = ccrs.EqualEarth(),
        row: Hashable | None = None,
        col: Hashable | None = None,
        col_wrap: int | None = None,
        global_extent: bool = True,
        coastlines: bool = False,
        gridlines: bool = False,
        coastlines_kwargs: dict[str, Any] | None = None,
        gridlines_kwargs: dict[str, Any] | None = None,
        ax: GeoAxes = None,
        **kwargs,
    ) -> GeoQuadMesh | FacetGrid:
        """
        Create a projected plot of a DataArray.

        Parameters
        ----------

        crs: cartopy.CRS, default: PlateCarree
            Coordinate reference system of the data.

        projection: cartopy.CRS, default: EqualEarth
            Coordinate reference system of the plot.

        row (Hashable or None, optional)
            If passed, make row faceted plots on this dimension name.

        col (Hashable or None, optional)
            If passed, make column faceted plots on this dimension name.

        col_wrap (int or None, optional)
            Use together with col to wrap faceted plots.

        global_extent: bool, default: True
            Whether to expand the plot to span the whole globe.

        coastlines: bool, default: False
            Whether to add coastlines outlines.

        gridlines: bool, default: False
            Whether to add gridlines.

        coastlines_kwargs: dict, optional
            Dict with keywords passed to the
            `cartopy.mpl.geoaxes.GeoAxes.coastlines` call.

        gridlines_kwargs: dict, optional
            Dict with keywords passed to the
            `cartopy.mpl.geoaxes.GeoAxes.coastlines` call.

        ax: GeoAxis, optional
            Plot on an existing axis (must be a Cartopy GeoAxis).
            Not compatible if creating a FacetGrid

        **kwargs: optional
            Additional keyword arguments passed to the `xarray.DataArray.plot` call.

        Returns
        -------

        A GeoAxis object

        """

        # Set plot keyword arguments, if not specified
        cbar_kwargs = dict(location="bottom", aspect=50, shrink=0.8, pad=0.1)
        kwargs.setdefault("cbar_kwargs", cbar_kwargs)
        kwargs.setdefault("cmap", "RdYlBu")
        kwargs.setdefault("zorder", 2.5)
        if coastlines:
            if coastlines_kwargs is None:
                coastlines_kwargs = {}
            coastlines_kwargs.setdefault("zorder", 4.0)
        if gridlines:
            if gridlines_kwargs is None:
                gridlines_kwargs = {}
            gridlines_kwargs.setdefault("zorder", 0.0)

        # Check if FacetGrid plot routine is required
        if col or row:
            if ax:
                raise TypeError("Can't create a FacetGrid on an existing Axis.")

            p = projplot_facet(
                self._obj,
                crs=crs,
                projection=projection,
                row=row,
                col=col,
                col_wrap=col_wrap,
                global_extent=global_extent,
                coastlines=coastlines,
                gridlines=gridlines,
                coastlines_kwargs=coastlines_kwargs,
                gridlines_kwargs=gridlines_kwargs,
                **kwargs,
            )
        else:
            p = projplot_single(
                self._obj,
                crs=crs,
                projection=projection,
                ax=ax,
                global_extent=global_extent,
                coastlines=coastlines,
                gridlines=gridlines,
                coastlines_kwargs=coastlines_kwargs,
                gridlines_kwargs=gridlines_kwargs,
                **kwargs,
            )
        return p

    def projplot_basins(
        self, spatial_obj: xr.DataArray, **kwargs
    ) -> GeoQuadMesh | FacetGrid:
        """Plot a basin-wise averaged values on a world map.
        The DataArray must have a "region" dimension corresponding to
        the river basins. Passing a spatial dummy DataArray with "lat" and "lon"
        dimension is required to specify the grid where the basin-wise
        averages should be plotted on.

        Parameters
        ----------

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

        return plot_basinwise_map(self._obj, spatial_obj, **kwargs)

    def weight_lat(
        self,
    ) -> DataArrayWeighted:
        """
        Returns a DataArray which is weighted to take into account that the grid cell
        area changes with latitude.

        Parameters
        ----------
        None

        Returns
        -------
        DataArrayWeighted
        """

        weights = np.cos(np.deg2rad(self._obj.lat))
        return self._obj.weighted(weights)

    def detrend(self, dim: int, deg: int = 1, keep_attrs: bool = True):
        """
        Detrend a DataArray.

        Parameters
        ----------

        dim: str
            Dimensions along which to apply detrend.

        deg: int, default: 1
            Degree of fitting polynomial which is used for detrending.

        keep_attrs: bool, default: True
            Whether to keep attributes after detrending.


        """
        obj = self._obj
        p = obj.polyfit(dim=dim, deg=deg)
        fit = xr.polyval(obj[dim], p.polyfit_coefficients)

        with xr.set_options(keep_attrs=keep_attrs):
            obj = obj - fit

        return obj


# Xarray Dataset accessor
@xr.register_dataset_accessor("dr")
class XrDatasetAccessor(XrBaseAccessor):
    """The DeepRec accessor for xarray datasets."""

    def weight_lat(
        self,
    ) -> DatasetWeighted:
        """
        Returns a Dataset which is weighted to take into account that the grid cell
        area changes with latitude.

        Parameters
        ----------
        None

        Returns
        -------
        DatasetWeighted
        """

        weights = np.cos(np.deg2rad(self._obj.lat))
        return self._obj.weighted(weights)

    def set_var_encoding(
        self, variables: list[str] | Literal["all"] = "all", **kwargs
    ) -> xr.Dataset:
        """
        Update the encoding settings for all variables of a Dataset.
        The encoding are format-specific settings for how the Dataset
        should be serialized.

        Parameters
        ----------
        variables: list or "all"
            The variables to set the encoding for.

        kwargs:
            Key-value pairs that are passed to the variable encoding.

            Example arguments for chunk-based compression:
            `compression="gzip", compression_opts=4`

        Returns
        -------
        Dataset


        """
        obj = self._obj

        match variables:
            case "all":
                variables = list(obj.data_vars)
            case str():
                variables = [variables]
            case Iterable():
                pass
            case _:
                raise ValueError(f"Invalid variables input type `{type(variables)}`.")

        for var in variables:
            obj[var].encoding.update(**kwargs)
        return obj

    def get_attrs(
        self, variables: list[str], keys: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Extract attributes of multiple Dataset variables.

        Parameters
        ----------

        variables: list of strings
            The Dataset variables from which attributes should be extracted.

        keys: list of strings
            The keys of the attributes that should be extracted.

        Returns
        -------
        dict:
            A nested dictionary with the variables on level 0 and the attributes
            on level 1.
        """

        ds = self._obj
        # Make sure inputs are of type list
        variables, keys = list(variables), list(keys)
        all_attrs = dict()
        for var in list(ds.dims):
            # Write attribute to nested dict if key exists in variable attributes
            all_attrs[var] = {
                key: ds[var].attrs[key]
                for key in keys
                if key in set(keys).intersection(ds[var].attrs)
            }

        return all_attrs

    def set_attrs(self, attributes: dict[str, dict[str, Any]]) -> xr.Dataset:
        """
        Write attributes of multiple Dataset variables.

        Parameters
        ----------

        attributes:
            A nested dictionary with the variables on level 0 and the attributes
            on level 1.

        Returns
        -------
        Dataset
        """

        ds = self._obj
        for var, attrs in attributes.items():
            for attr, value in attrs.items():
                ds[var].attrs[attr] = value
        return ds

    def time_notnull(self, dims: str | list[str] | None = None) -> xr.DataArray:
        """Test each time step in a Dataset whether it is NA for all values of
        the specified dimension(s) for at least one data variable."""
        # Check over spatial dimensions if dims are not specified
        if dims is None:
            dims = ["lat", "lon"]

        time_na = (
            self._obj.to_dataarray()
            .isnull()
            .all(dim=dims)
            .any(dim="variable")
            .compute()
        )
        return ~time_na
