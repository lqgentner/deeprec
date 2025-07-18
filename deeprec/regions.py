import cf_xarray  # noqa
import geopandas as gpd
import regionmask

from .utils import ROOT_DIR, XrObj


def basins(
    names: str | list[str] | None = None,
    top: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Returns a geopandas dataframe containing the selected river
    basins.

    Parameters
    ---------

    names: string or list of strings, optional
        The names of river basins to return.
        Leave empty to select basins after `top` or return all.

    top: int, optional
        The number of basins to select, sorted after basin area.
        Leave empty to select basins after `names` or return all.

    Returns
    -------

    GeoDataFrame
        A dataframe with basin information and geometry

    """

    file = ROOT_DIR / "data/processed/shapefiles/mrb/mrb_basins.shp"

    gdf = gpd.read_file(file, engine="pyogrio")

    if top and names:
        # Catch providing arguments for both top and names
        raise TypeError("Provide either `names` pr `top`, not both.")

    if top:
        # Select top X basins
        gdf = gdf[:top]

    if names:
        # Ensure `names` is a list if a single string is provided
        if isinstance(names, str):
            names = [names]
        # Select basins by names
        gdf = gdf.loc[gdf.riverbasin.isin(names)]

    # Return the full DataFrame if neither `top` nor `names` is provided

    return gdf


def countries(names: str | list[str] | None = None) -> gpd.GeoDataFrame:
    """
    Returns a geopandas dataframe containing the countries.

    Parameters
    ---------

    names: string or list of strings, optional
        The names of countries to return.
        Leave empty to return all.

    Returns
    -------

    GeoDataFrame
        A dataframe with basin information and geometry

    """

    file = (
        ROOT_DIR / "data/processed/shapefiles/naturalearth/ne_50m_admin_0_countries.shp"
    )

    gdf = gpd.read_file(file, engine="pyogrio")

    # Put single name string into list
    if isinstance(names, str):
        names = [names]

    if names:
        gdf = gdf.loc[gdf.admin.isin(names)]

    # Return the full DataFrame if `names` is not provided

    return gdf


def _select_region(
    obj: XrObj,
    regions: regionmask.Regions,
    return_region: bool = True,
    drop: bool = False,
) -> XrObj:
    """Select regions of an Xarray Dataset or DataArray by a Regionmask Regions object."""
    if return_region:
        # 3D case
        mask = regions.mask_3D(obj)
        out = obj.where(mask)
        if drop:
            # drop during where() gives error due to added coordinates
            out = out.dropna("lat", how="all").dropna("lon", how="all")
        # Use basin names instead of integers and drop unhelpful features
        out = (
            out.assign_coords(region=out.names.astype("object"))
            .drop_vars(["abbrevs", "names"])
            .transpose("region", ...)
        )
    else:
        # 2D case
        mask = regions.mask(obj, flag="names")
        out = obj.where(mask.cf.isin(regions.names), drop=drop)

    return out


def select_basins(
    obj: XrObj,
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

    regions = regionmask.from_geopandas(
        basins(names, top), names="riverbasin", name="riverbasin", overlap=False
    )

    return _select_region(obj, regions, return_region=return_region, drop=drop)


def select_countries(
    obj: XrObj,
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

    regions = regionmask.from_geopandas(
        countries(names),
        names="admin",
        abbrevs="adm0_a3",
        name="country",
    )

    return _select_region(obj, regions, return_region=return_region, drop=drop)
