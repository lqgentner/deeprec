"""Dataset download routines"""

from datetime import datetime
import io
from math import floor
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar
import zipfile

from numpy import datetime64
import pandas as pd
import requests
import wandb
import xarray as xr

ROOT_DIR = Path(__file__).resolve().parents[1]
"""Absolute base path of project. All paths are defined relative to this path."""

XrObj = TypeVar("XrObj", xr.Dataset, xr.DataArray)


def download_file(url: str, path: str | PathLike, timeout: float = 5.0, **get_kwargs):
    """Downloads a file from a provided URL"""
    filename = url.rsplit("/")[-1]
    # Ensure path is pathlib object
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    filepath = Path(path) / filename
    response = requests.get(url, timeout=timeout, **get_kwargs)
    with open(filepath, mode="wb") as file:
        file.write(response.content)


def download_zip(url: str, path: str, timeout: float = 5.0, **get_kwargs):
    """Downloads and unzips an archive from a provided URL"""
    request = requests.get(url=url, timeout=timeout, **get_kwargs)
    zip_file = zipfile.ZipFile(io.BytesIO(request.content))
    zip_file.extractall(path=path)


def month_center_range(
    start: str | pd.Timestamp | datetime | datetime64,
    end: str | pd.Timestamp | datetime | datetime64,
) -> pd.DatetimeIndex:
    """Return a monthly-spaced DatetimeIndex with timestamps at the
    center of their respective months. Note that the returned Index can extend
    outside of the provided start and end dates.
    """

    # Start and end of month beginnings range
    first_begin = pd.to_datetime(start).floor("d").replace(day=1)
    final_begin = pd.to_datetime(end).floor("d").replace(day=1)

    month_begins = pd.date_range(first_begin, final_begin, freq="MS")
    month_ends = month_begins + pd.DateOffset(months=1)
    month_centers = month_begins + (month_ends - month_begins) / 2

    return month_centers


def conv2d_out_size(
    in_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    return floor(
        (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def wandb_checkpoint_download(
    artifact_path: str | None = None,
    project: str | None = None,
    run_id: str | None = None,
    alias: str | int = "best",
) -> Path:
    """Download a model checkpoint from Weights & Biases.

    Parameters
    ----------

    artifact_path: str, optional
        The artifact name, prefixed by the entity and project.
        E.g., 'my_name/my_project/model-012345:v10'
        Either this or project and run_id must be provided.
    project: str, optional
        The W&B project name including the entity, e.g. 'my_name/my_project'.
    run_id: str, optional
        The W&B run ID.
    alias: 'best', 'latest', 'v<int>', or int, default: 'best'
        The artifact alias which specifies which checkpoint version to download.


    """
    if artifact_path is None:
        if project is None or run_id is None:
            raise ValueError(
                "Either artifact_path or project and run_id must be provided."
            )
        if isinstance(alias, int):
            alias = f"v{alias}"
        artifact_path = f"{project}/model-{run_id}:{alias}"

    # Download checkpoint
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    artifact_dir = artifact.download()

    return Path(artifact_dir) / "model.ckpt"


def verify_dim_ispresent(obj: XrObj, dim: str) -> None:
    """Verify if a dimension is present in an xarray object"""
    if dim not in obj.dims:
        raise ValueError(f"Dimension '{dim}' not present in {type(obj).__name__}.")


def repeat_by_weight(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """Repeat rows in a DataFrame according to the integer values of a weight column"""
    df = df.reindex(df.index.repeat(df[weight_col])).reset_index(drop=True)
    return df


def generate_acdd_metadata(ds: xr.Dataset | xr.DataArray) -> dict[str, Any]:
    """Extract metadata according to the Attribute Convention for Data Discovery (ACDD) standard.
    The Dataset/DataArray must have time, lat, and lon dimensions"""

    # Verify required dimensions are present
    required_dims = ["time", "lat", "lon"]
    for dim in required_dims:
        if dim not in ds.dims:
            raise ValueError(f"Dimension '{dim}' not present in the dataset.")

    # Extract dynamic attributes
    TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
    timenow_str = pd.Timestamp.utcnow().strftime(TIME_FORMAT)

    times = ds.get_index("time")
    lats = ds.get_index("lat")
    lons = ds.get_index("lon")

    timestart_str = times[0].strftime(TIME_FORMAT)
    timeend_str = times[-1].strftime(TIME_FORMAT)

    latmin = float(lats[0])
    latmax = float(lats[-1])
    lonmin = float(lons[0])
    lonmax = float(lons[-1])

    # Create dict of attributes
    attrs = {}
    attrs["time_coverage_start"] = timestart_str
    attrs["time_coverage_end"] = timeend_str
    attrs["geospatial_lat_min"] = latmin
    attrs["geospatial_lat_max"] = latmax
    attrs["geospatial_lon_min"] = lonmin
    attrs["geospatial_lon_max"] = lonmax
    attrs["date_created"] = timenow_str

    return attrs
