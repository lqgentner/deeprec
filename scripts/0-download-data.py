#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Downloads Script

This script handles the download of all datasets used in the DeepRec study.

Notes:

  - Downloading JPL Tellus mascons requires a .netrc file in the user directory:
    machine urs.earthdata.nasa.gov
    login <your-user>
    password <your-pw>
  - Downloading the ERA5 variables requires a .cdsapirc file in the user directory:
    url: https://cds.climate.copernicus.eu/api
    key: <your-api-key>

Usage:
    python scripts/0-download-data.py --help
"""

import argparse
from pathlib import Path
import re
from urllib.parse import unquote
from zipfile import ZipFile

import cdsapi
import earthaccess
from loguru import logger
import requests
from tqdm import tqdm

from deeprec.utils import ROOT_DIR

# Available datasets for download
DATASETS = [
    "jpl_mascons",
    "csr_mascons",
    "gsfc_mascons",
    "watergap",
    "era5",
    "noaa_sst",
    "isimip_landuse",
    "isimip_lakes",
    "shapefiles",
    "reconstructions",
    "evaluation",
]


class DatasetDownloader:
    """Main class for handling dataset downloads."""

    def __init__(self, dl_path: str | Path):
        self.dl_path = Path(dl_path)
        self.dl_path.mkdir(parents=True, exist_ok=True)
        logger.info("Download path: {}", self.dl_path)

    def parse_filename_from_headers(self, headers: dict, url: str) -> str:
        """Extract filename from HTTP headers with comprehensive fallback logic."""
        filename = None

        # Try Content-Disposition header first
        content_disposition = headers.get("content-disposition", "")

        if content_disposition:
            # Handle different Content-Disposition formats
            filename_star_match = re.search(
                r"filename\*=([^']*)'([^']*)'(.+)", content_disposition
            )
            if filename_star_match:
                encoding = filename_star_match.group(1) or "utf-8"
                filename = unquote(filename_star_match.group(3), encoding=encoding)
            else:
                filename_match = re.search(r"filename=([^;]+)", content_disposition)
                if filename_match:
                    filename = filename_match.group(1).strip("\"'")
                    filename = unquote(filename)

        # Fallback to URL path
        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename:
                filename = "downloaded_file"

        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

        # Ensure we have an extension if possible
        if "." not in filename:
            content_type = headers.get("content-type", "").split(";")[0].lower()
            extension_map = {
                "text/plain": ".txt",
                "text/html": ".html",
                "application/pdf": ".pdf",
                "application/json": ".json",
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "application/zip": ".zip",
                "application/octet-stream": ".bin",
            }
            if content_type in extension_map:
                filename += extension_map[content_type]

        return filename

    def download_file(self, url: str, download_dir: Path, **requests_kwargs) -> Path:
        """Download file with progress tracking and robust filename parsing."""
        try:
            response = requests.get(url, stream=True, **requests_kwargs)
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Parse filename
            filename = self.parse_filename_from_headers(response.headers, url)

            # Create file path
            download_dir.mkdir(parents=True, exist_ok=True)
            file_path = download_dir / filename

            # Skip if file already exists
            if file_path.exists():
                logger.info("File already exists: {}", file_path)
                return file_path

            logger.info("Downloading {} to {}", filename, download_dir)

            with (
                file_path.open("wb") as f,
                tqdm(
                    desc=filename,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            logger.info("Successfully downloaded: {}", file_path)
            return file_path

        except Exception as e:
            logger.error("Failed to download {}: {}", url, e)
            raise

    def download_jpl_mascons(self):
        """Download JPL Mascons data."""

        try:
            logger.info("Downloading JPL Mascons...")
            earthaccess.login(strategy="netrc")
            results = earthaccess.search_data(
                short_name="TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"
            )
            download_dir = self.dl_path / "targets/jpl-mascons"
            download_dir.mkdir(parents=True, exist_ok=True)
            earthaccess.download(results, download_dir)
            logger.info("Downloaded JPL Mascons: {len(files)} files")
        except Exception as e:
            logger.error("Failed to download JPL Mascons: {}", e)

    def download_csr_mascons(self):
        """Download CSR Mascons data."""
        try:
            logger.info("Downloading CSR Mascons...")
            url = "https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
            download_dir = self.dl_path / "targets/csr-mascons"
            self.download_file(url, download_dir, verify=False)
        except Exception as e:
            logger.error("Failed to download CSR Mascons: {}", e)

    def download_gsfc_mascons(self):
        """Download GSFC Mascons data."""
        try:
            logger.info("Downloading GSFC Mascons...")
            url = "https://earth.gsfc.nasa.gov/sites/default/files/geo/gsfc.glb_.200204_202406_rl06v2.0_obp-ice6gd_halfdegree.nc"
            download_dir = self.dl_path / "targets/gsfc-mascons"
            self.download_file(url, download_dir)
        except Exception as e:
            logger.error("Failed to download GSFC Mascons: {}", e)

    def download_watergap(self):
        """Download WaterGAP Global Hydrology Model data."""
        try:
            logger.info("Downloading WaterGAP 2.2e...")
            url = "https://api.gude.uni-frankfurt.de/api/core/bitstreams/879ce7c3-4d21-4ee1-a83c-e830b13b9d2e/content"
            download_dir = self.dl_path / "inputs/watergap22e"
            self.download_file(url, download_dir)
        except Exception as e:
            logger.error("Failed to download WaterGAP: {}", e)

    def download_era5(self):
        """Download ERA5 monthly data."""
        try:
            logger.info("Downloading ERA5 monthly data...")

            START_YEAR = 1940
            END_YEAR = 2023
            VARIABLES = [
                "total_precipitation",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "high_vegetation_cover",
                "low_vegetation_cover",
                "evaporation",
                "potential_evaporation",
                "runoff",
                "snowfall",
                "snowmelt",
                "snow_depth",
                "snow_evaporation",
                "surface_pressure",
                "leaf_area_index_high_vegetation",
                "leaf_area_index_low_vegetation",
                "sub_surface_runoff",
                "surface_runoff",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
            ]

            dataset_path = self.dl_path / "inputs/era5-monthly"
            dataset_path.mkdir(exist_ok=True)

            c = cdsapi.Client()
            dataset = "reanalysis-era5-single-levels-monthly-means"

            for variable in VARIABLES:
                logger.info("Downloading ERA5 variable: {}", variable)

                dataset_file = (
                    dataset_path / f"era5-monthly_{variable}_{START_YEAR}-{END_YEAR}.nc"
                )
                if dataset_file.exists():
                    logger.info("File already exists: {}", dataset_file)
                    continue

                request = {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": [variable],
                    "year": [f"{year}" for year in range(START_YEAR, END_YEAR + 1)],
                    "month": [f"{month:02}" for month in range(1, 12 + 1)],
                    "time": ["00:00"],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                }

                c.retrieve(dataset, request, dataset_file)
                logger.info("Downloaded: {}", dataset_file)

            logger.info("ERA5 download completed.")
        except Exception as e:
            logger.error("Failed to download ERA5: {}", e)

    def download_noaa_sst(self):
        """Download NOAA Reconstructed Sea Surface Temperature."""
        try:
            logger.info("Downloading NOAA ERSST v5...")
            url = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
            download_dir = self.dl_path / "inputs/noaa-ersst-v5"
            self.download_file(url, download_dir)
        except Exception as e:
            logger.error("Failed to download NOAA SST: {}", e)

    def download_isimip_landuse(self):
        """Download ISIMIP Land Use data."""
        try:
            logger.info("Downloading ISIMIP Land Use...")
            download_dir = self.dl_path / "inputs/landuse"
            urls = {
                "totals": "https://files.isimip.org/ISIMIP3a/InputData/socioeconomic/landuse/histsoc/landuse-totals_histsoc_annual_1901_2021.nc",
                "urbanareas": "https://files.isimip.org/ISIMIP3a/InputData/socioeconomic/landuse/histsoc/landuse-urbanareas_histsoc_annual_1901_2021.nc",
            }

            for name, url in tqdm(urls.items(), desc="Downloading landuse"):
                self.download_file(url, download_dir)
        except Exception as e:
            logger.error("Failed to download ISIMIP Land Use: {}", e)

    def download_isimip_lakes(self):
        """Download ISIMIP Lake area fraction."""
        try:
            logger.info("Downloading ISIMIP Lake area fraction...")
            download_dir = self.dl_path / "inputs/pctlake"
            url = "https://files.isimip.org/ISIMIP3a/InputData/socioeconomic/lakes/histsoc/pctlake_histsoc_1901_2021.nc"
            self.download_file(url, download_dir)
        except Exception as e:
            logger.error("Failed to download ISIMIP Lakes: {}", e)

    def download_shapefiles(self):
        """Download shapefiles."""
        try:
            logger.info("Downloading shapefiles...")

            # GRDC Major River Basins
            url = "https://grdc.bafg.de/downloads/GRDC_Major_River_Basins_shp.zip"
            download_dir = self.dl_path / "shapefiles/mrb"
            self.download_file(url, download_dir)

            # NaturalEarth countries
            url = "https://naturalearth.s3.amazonaws.com/5.1.1/50m_cultural/ne_50m_admin_0_countries.zip"
            download_dir = self.dl_path / "shapefiles/naturalearth"
            self.download_file(url, download_dir)
        except Exception as e:
            logger.error("Failed to download shapefiles: {}", e)

    def download_reconstructions(self):
        """Download previous TWS reconstructions."""
        try:
            logger.info("Downloading TWS reconstructions...")

            # Humphrey, 2019
            url = "https://figshare.com/ndownloader/files/17990285"
            download_dir = self.dl_path / "reconstructions/humphrey"
            file_path = self.download_file(url, download_dir)
            # Extract ZIP
            with ZipFile(file_path) as zip:
                zip.extractall(path=file_path)

            # Yin, 2023
            url = "https://zenodo.org/records/10040927/files/CSR-based%20GTWS-MLrec%20TWS.nc"
            download_dir = self.dl_path / "reconstructions/yin"
            self.download_file(url, download_dir)

            # Palazzoli, 2025
            url = "https://zenodo.org/records/10953658/files/GRAiCE_BiLSTM.nc"
            download_dir = self.dl_path / "reconstructions/palazzoli"
            self.download_file(url, download_dir)

            # Li, 2021
            url = "https://datadryad.org/api/v2/files/665199/download"
            download_dir = self.dl_path / "reconstructions/li"
            self.download_file(url, download_dir)

        except Exception as e:
            logger.error("Failed to download reconstructions: {}", e)

    def download_evaluation_data(self):
        """Download evaluation datasets."""
        try:
            logger.info("Downloading evaluation datasets...")

            # Sea Level Rise Contributors
            url = (
                "https://zenodo.org/records/3862995/files/global_basin_timeseries.xlsx"
            )
            download_dir = self.dl_path / "eval/sea-level/frederikse"
            self.download_file(url, download_dir)

            # Extreme Event Intensity
            url = "https://zenodo.org/records/7599831/files/Figure2_data.xlsx"
            download_dir = self.dl_path / "eval/intensity/rodell"
            self.download_file(url, download_dir)

        except Exception as e:
            logger.error("Failed to download evaluation data: {}", e)

    def download_all(self, datasets=None):
        """Download all datasets or specific ones."""
        available_datasets = {
            "jpl_mascons": self.download_jpl_mascons,
            "csr_mascons": self.download_csr_mascons,
            "gsfc_mascons": self.download_gsfc_mascons,
            "watergap": self.download_watergap,
            "era5": self.download_era5,
            "noaa_sst": self.download_noaa_sst,
            "isimip_landuse": self.download_isimip_landuse,
            "isimip_lakes": self.download_isimip_lakes,
            "shapefiles": self.download_shapefiles,
            "reconstructions": self.download_reconstructions,
            "evaluation": self.download_evaluation_data,
        }

        if datasets is None:
            datasets = available_datasets.keys()

        for dataset in datasets:
            if dataset in available_datasets:
                logger.info("Starting download: {}", dataset)
                try:
                    available_datasets[dataset]()
                    logger.info("Completed download: {}", dataset)
                except Exception as e:
                    logger.error("Failed to download {}: {}", dataset, e)
            else:
                logger.warning(f"Unknown dataset: {dataset}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Download datasets for hydrology/climate study"
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        choices=DATASETS,
        metavar="",
        help=(
            "Specific datasets to download (default: all).  Allowed values are "
            + ", ".join(DATASETS)
        ),
    )
    args = parser.parse_args()

    dl_path = ROOT_DIR / "data/raw"
    downloader = DatasetDownloader(dl_path)
    downloader.download_all(args.datasets)


if __name__ == "__main__":
    main()
