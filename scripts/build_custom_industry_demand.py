# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import pandas as pd
import geopandas as gpd
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore")
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger


logger = create_logger(__name__)


def fuzzy_match(ethanol_plants_clean, uscities):
    """
    Fuzzy match ethanol plants with US cities based on name similarity.
    """
    # Get non-exact matches that have NaN in x and y columns
    non_matched = ethanol_plants_clean[
        ethanol_plants_clean["x"].isna() & ethanol_plants_clean["y"].isna()
    ]

    # Loop through each row in non_matched DataFrame and find the best match
    for index, row in non_matched.iterrows():
        city = row["City"]
        state = row["State"]

        # Filter uscities DataFrame for the same state to find candidates
        candidates = uscities[uscities["State"] == state]

        # Find the best match using fuzzy matching
        matches = get_close_matches(city, candidates['City'], n=1, cutoff=0.9)
        print(f"Fuzzy matching for {city}, {state}: {matches}")
        # If a match is found, update the coordinates
        if matches:
            logger.info(f"Fuzzy matching for {city}, {state}: {matches}")
            ethanol_plants_clean.loc[index, ["x", "y"]] = candidates.loc[candidates.City == matches[0], ["x", "y"]].values[0]
        else:
            # Dictionary for manual matching
            manual_matches = {
                "big stone": "big stone city",  # South Dakota
                "cedar rapids dry mill": "cedar rapids",  # Iowa,
                "ft dodge": "fort dodge",  # Iowa
                "mt vernon": "mount vernon",  # Indiana
                "saint joseph": "st. joseph",  # Missouri
                "sioux river": "sioux falls",  # South Dakota
            }
            # Match names manually
            if city in manual_matches:
                ethanol_plants_clean.loc[index, ["x", "y"]] = candidates.loc[candidates.City == manual_matches[city], ["x", "y"]].values[0]
                logger.info(f"Manual match for {city}, {state}: {manual_matches[city]}")
            else:
                # If no manual matches are found, coordinates are taken from internet
                coordinates = {"clymers": (40.71851246253904, -86.43850867600496),
                               "highwater": (44.2309628735522, -95.310245211268),
                               "jewell": (41.586835, -93.625000),
                               "pine lake": (42.45940778347515, -93.05686779545323),
                               "red trail": (46.87883259142844, -102.29853016314213),
                               "quad": (42.4778916168018, -95.4122032647622)
                }
                if city in coordinates:
                    ethanol_plants_clean.loc[index, ["y", "x"]] = coordinates[city]
                    logger.info(f"Coordinates for {city}, {state}: {coordinates[city]}")

    return ethanol_plants_clean


def prepare_ethanol_plants(ethanol_plants_raw, uscities):
    """
    Clean and prepare ethanol plants data.
    """
    # Drop rows with no city names
    ethanol_plants_clean = ethanol_plants_raw[ethanol_plants_raw.City.notna()]

    # Fill state names by forward filling
    ethanol_plants_clean["State"] = (
        ethanol_plants_clean["State"].fillna(method="ffill")
    )

    # Groupby state and city names, and sum the production capacity
    ethanol_plants_clean = (
        ethanol_plants_clean.groupby(["City", "State"])
        .agg({"MMgal/yr": "sum"})
        .reset_index()
    )

    # Make City and State names lowercase
    ethanol_plants_clean["City"] = ethanol_plants_clean["City"].str.lower()
    ethanol_plants_clean["State"] = ethanol_plants_clean["State"].str.lower()

    # Rename columns of uscities to match ethanol plants data
    uscities.rename(
        columns={
            "state_name":"State",
            "city":"City",
            "lat":"y",
            "lng":"x"
        }, inplace=True
    )

    # Groupby state and city names, and take mean for x and y coordinates
    uscities = (
        uscities.groupby(["City", "State"])
        .agg({"y": "mean", "x": "mean"})
        .reset_index()
    )

    # Make City and State names lowercase
    uscities["City"] = uscities["City"].str.lower()
    uscities["State"] = uscities["State"].str.lower()

    # Merge longitude and latitude columns from uscities based on State and City names
    ethanol_plants_clean = pd.merge(
        ethanol_plants_clean,
        uscities[["City", "State", "y", "x"]],
        how="left",
        left_on=["City", "State"],
        right_on=["City", "State"],
    )

    # Merge longitude and latitude columns for non-exact matches
    ethanol_plants_clean = fuzzy_match(ethanol_plants_clean, uscities)

    # Ethanol production capacity in MMgal/yr to MWh/a 
    # 1 gallon ethanol = 80.2 MJ https://indico.ictp.it/event/8008/session/3/contribution/23/material/slides/2.pdf
    # 1 MWh = 3600 MJ
    ethanol_plants_clean["MWh/a"] = ethanol_plants_clean["MMgal/yr"] * 1e6 * 80.2 / 3600
    logger.info("Prepared ethanol plants data")
    return ethanol_plants_clean


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_custom_industry_demand",
            simpl="",
            clusters="10",
            planning_horizons="2020",
            demand="AB",
            configfile="configs/calibration/config.base_AC.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load US cities locational information
    uscities = pd.read_csv(snakemake.input.uscity_map)

    if snakemake.params.add_ethanol:
        # load ethanol plants
        ethanol_plants_raw = pd.read_excel(snakemake.input.ethanol_plants, skiprows=2)

        # clean ethanol plants data
        ethanol_plants_clean = prepare_ethanol_plants(ethanol_plants_raw, uscities)
