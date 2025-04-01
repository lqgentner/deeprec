"""Dataset download routines"""

import io
import sys
import zipfile
from datetime import datetime
from importlib import reload
from math import floor
from pathlib import Path
from typing import Literal, TypeVar

import pandas as pd
import requests
import wandb
import xarray as xr
from numpy import datetime64

ROOT_DIR = Path(__file__).resolve().parents[1]
"""Absolute base path of project. All paths are defined relative to this path."""

XrObj = TypeVar("XrObj", xr.Dataset, xr.DataArray)


def download_file(url, path):
    """Downloads a file from a provided URL"""
    filename = url.rsplit("/")[-1]
    # Ensure path is pathlib object
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    filepath = Path(path) / filename
    response = requests.get(url, timeout=10)
    with open(filepath, mode="wb") as file:
        file.write(response.content)


def download_zip(url, path):
    """Downloads and unzips an archive from a provided URL"""
    request = requests.get(url=url, timeout=10)
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


def reload_submodule(name: str) -> None:
    """Reload a submodule from a package"""
    ls = []
    # making copy to avoid regeneration of sys.modules
    for i, j in sys.modules.items():
        r, v = i, j
        ls.append((r, v))

    for i in ls:
        if i[0] == name:
            reload(i[1])
            break


def wandb_checkpoint_download(
    artifact_path: str = None,
    project: str = None,
    run_id: str = None,
    alias: Literal["best", "latest"] | int = "best",
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
    alias: 'best', 'latest', or int, default: 'best'
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
