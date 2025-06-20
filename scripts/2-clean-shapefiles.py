#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process the basins and countries shapefiles.
Usage:
    python scripts/2-clean-shapefiles.py
"""

from pathlib import Path

import geopandas as gpd
import janitor  # noqa: F401
from loguru import logger
import numpy as np

from deeprec.utils import ROOT_DIR


def main():
    # Perform cleaning
    mrb_in = ROOT_DIR / "data/raw/shapefiles/mrb/GRDC_Major_River_Basins_shp.zip"
    mrb_out = ROOT_DIR / "data/processed/shapefiles/mrb/mrb_basins.shp"

    ne_in = ROOT_DIR / "data/raw/shapefiles/naturalearth/ne_50m_admin_0_countries.zip"
    ne_out = (
        ROOT_DIR / "data/processed/shapefiles/naturalearth/ne_50m_admin_0_countries.shp"
    )

    logger.info("Cleaning Major River Basins...")
    clean_mrb(mrb_in, mrb_out)

    logger.info("Cleaning Natural Earth countries...")
    clean_naturalearth(ne_in, ne_out)

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
    # Make sure folder exists
    file_out.parent.mkdir(parents=True, exist_ok=True)
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
    # Make sure folder exists
    file_out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(file_out)


if __name__ == "__main__":
    main()
