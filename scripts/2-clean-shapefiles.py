#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process the basins, countries and continents shapefiles.
Usage:
    python scripts/2-clean-shapefiles.py
"""

import logging
from pathlib import Path

import geopandas as gpd
import janitor  # noqa: F401
import numpy as np

from deeprec.utils import ROOT_DIR


def main():
    # Set up logger
    log_fmt = "%(name)-12s: %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # Perform cleaning
    mrb_in = ROOT_DIR / "data/raw/shapefiles/mrb"
    mrb_out = ROOT_DIR / "data/processed/shapefiles/mrb/"
    mrb_file = "mrb_basins.shp"
    ne_in = ROOT_DIR / "data/raw/shapefiles/naturalearth"
    ne_out = ROOT_DIR / "data/processed/shapefiles/naturalearth"
    ne_file = "ne_50m_admin_0_countries.shp"
    cnt_in = ROOT_DIR / "data/raw/shapefiles/continents"
    cnt_out = ROOT_DIR / "data/processed/shapefiles/continents"
    cnt_file = "World_Continents.shp"
    mrb_out.mkdir(parents=True, exist_ok=True)
    ne_out.mkdir(parents=True, exist_ok=True)
    cnt_out.mkdir(parents=True, exist_ok=True)

    logger.info("Cleaning Major River Basins...")
    clean_mrb(mrb_in / mrb_file, mrb_out / mrb_file)

    logger.info("Cleaning Natural Earth countries...")
    clean_naturalearth(ne_in / ne_file, ne_out / ne_file)

    logger.info("Cleaning continents...")
    clean_continents(cnt_in / cnt_file, cnt_out / cnt_file)

    logger.info("Cleaning completed.")


def clean_mrb(file_in: Path, file_out: Path):
    """Clean Major River Basins shapefile"""

    # More than one layer found in 'mrb': 'mrb_basins' (default), 'mrb_named_rivers', 'mrb_rivernames', 'mrb_rivers'
    gdf = gpd.read_file(file_in, engine="pyogrio", layer="mrb_basins")
    # Clean all caps column headers
    gdf = gdf.clean_names()
    # # Removing parenthesis, UPPERCASE to Capitalized
    gdf.riverbasin = gdf.riverbasin.replace(" (.*)", "", regex=True).str.capitalize()
    # # # Sort after riverbasin size
    gdf = gdf.sort_values(by="sum_sub_ar", ascending=False, ignore_index=True)
    # # Remove duplicated rivers: [Uniq, Dup, Dup] -> [Uniq, Dup_0, Dup_1]
    gdf.riverbasin = gdf.riverbasin.where(
        ~gdf.riverbasin.duplicated(),
        gdf.riverbasin + "_" + gdf.groupby("riverbasin").cumcount().astype(str),
    )
    # Export cleaned shapefile
    gdf.to_file(file_out)


def clean_naturalearth(file_in: Path, file_out: Path):
    """Clean Natural Earth countries shapefile"""
    gdf = gpd.read_file(file_in, engine="pyogrio")
    # Remove unnecessary columns, leave geometry in last col
    gdf = gdf.iloc[:, np.r_[:10, -1]]
    # Clean all caps column headers
    gdf = gdf.clean_names()
    # Remove spaces in admin
    gdf.admin = gdf.admin.str.replace(" ", "")
    gdf.to_file(file_out)


def clean_continents(file_in: Path, file_out: Path):
    """Clean World continents shapefile"""
    gdf = gpd.read_file(file_in, engine="pyogrio")
    # Clean all caps column headers, drop cols
    gdf = gdf.clean_names().drop(["objectid_1", "sqmi"], axis=1)
    # Remove spaces in names
    gdf.continent = gdf.continent.str.replace(" ", "")
    # To WGS 84
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(file_out)


if __name__ == "__main__":
    main()
