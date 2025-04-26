# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../submodules/pypsa-earth/scripts/")))
import pandas as pd
import geopandas as gpd
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore")
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger
from build_industrial_distribution_key import map_industry_to_buses


logger = create_logger(__name__)


def process_uscities(uscities):
    """
    Process US cities data to extract relevant information.
    """
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
    uscities_grouped = (
        uscities.groupby(["City", "State"])
        .agg({"y": "mean", "x": "mean"})
        .reset_index()
    )

    # Get State name to state code mapping
    state_id_mapping = uscities[["State", "state_id"]].drop_duplicates().set_index("State")

    # Add state_id to uscities_grouped
    uscities_grouped["State_id"] = uscities_grouped["State"].map(state_id_mapping["state_id"])

    # Make City and State names lowercase
    uscities_grouped["City"] = uscities_grouped["City"].str.lower()
    uscities_grouped["State"] = uscities_grouped["State"].str.lower()

    return uscities_grouped


def fuzzy_match_ethanol(ethanol_plants_clean, uscities_clean):
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
        candidates = uscities_clean[uscities_clean["State"] == state]

        # Find the best match using fuzzy matching
        matches = get_close_matches(city, candidates['City'], n=1, cutoff=0.9)
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


def prepare_ethanol_plants(ethanol_plants_raw, uscities_clean):
    """
    Clean and prepare ethanol plants data.
    """
    # Drop rows with no city names (intermediate empty spaces)
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

    # Merge longitude and latitude columns from uscities based on State and City names
    ethanol_plants_clean = pd.merge(
        ethanol_plants_clean,
        uscities_clean[["City", "State", "y", "x"]],
        how="left",
        left_on=["City", "State"],
        right_on=["City", "State"],
    )

    # Merge longitude and latitude columns for non-exact matches
    ethanol_plants_clean = fuzzy_match_ethanol(ethanol_plants_clean, uscities_clean)

    # Ethanol production capacity in MMgal/yr to MWh/a 
    # 1 gallon ethanol = 80.2 MJ https://indico.ictp.it/event/8008/session/3/contribution/23/material/slides/2.pdf
    # 1 MWh = 3600 MJ
    ethanol_plants_clean["capacity"] = ethanol_plants_clean["MMgal/yr"] * 1e6 * 80.2 / 3600

    # Add country and industry column
    ethanol_plants_clean.loc[:, ["country", "industry"]] = ["US", "ethanol"]

    # Select relevant columns
    ethanol_plants_clean = ethanol_plants_clean[["country", "x", "y", "capacity", "industry"]]
    logger.info("Prepared ethanol plants data")
    return ethanol_plants_clean


def fuzzy_match_ammonia(ammonia_plants_clean, uscities_clean):
    """
    Fuzzy match ammonia plants with US cities based on name similarity.
    """
    # Get non-exact matches that have NaN in x and y columns
    non_matched = ammonia_plants_clean[
        ammonia_plants_clean["x"].isna() & ammonia_plants_clean["y"].isna()
    ]

    # Loop through each row in non_matched DataFrame and find the best match
    for index, row in non_matched.iterrows():
        city = row["City"]
        state = row["State_id"]

        # Filter uscities DataFrame for the same state to find candidates
        candidates = uscities_clean[uscities_clean["State_id"] == state]

        # Find the best match using fuzzy matching
        matches = get_close_matches(city, candidates['City'], n=1, cutoff=0.9)
        # If a match is found, update the coordinates
        if matches:
            logger.info(f"Fuzzy matching for {city}, {state}: {matches}")
            ammonia_plants_clean.loc[index, ["x", "y"]] = candidates.loc[candidates.City == matches[0], ["x", "y"]].values[0]
        else:
            # Take coordinates from internet
            coordinates = {"port neal": (42.33408214805535, -96.37646295212545),
                           "geismar": (30.224529130849646, -91.0565952614892)
            }
            if city in coordinates:
                ammonia_plants_clean.loc[index, ["y", "x"]] = coordinates[city]
                logger.info(f"Coordinates for {city}, {state}: {coordinates[city]}")

    return ammonia_plants_clean


def prepare_ammonia_plants(ammonia_plants_raw, uscities_clean):
    """
    Clean and prepare ammonia plants data.
    """
    # Groupby state and city names, and sum the production capacity
    ammonia_plants_clean = (
        ammonia_plants_raw.groupby(["City", "State_id"])
        .agg({"Production (thousand metric tons)": "sum"})
        .reset_index()
    )

    # Make City names lowercase
    ammonia_plants_clean["City"] = ammonia_plants_clean["City"].str.lower()

    # Merge longitude and latitude columns from uscities based on State and City names
    ammonia_plants_clean = pd.merge(
        ammonia_plants_clean,
        uscities_clean[["City", "State", "State_id", "y", "x"]],
        how="left",
        left_on=["City", "State_id"],
        right_on=["City", "State_id"],
    )

    # Merge longitude and latitude columns for non-exact matches
    ammonia_plants_clean = fuzzy_match_ammonia(ammonia_plants_clean, uscities_clean)

    # Convert production capacity from thousand metric tons to MWh/a
    # 1 metric ton ammonia = 5.17 MWh https://ammoniaenergy.org/articles/round-trip-efficiency-of-ammonia-as-a-renewable-energy-transportation-media/
    ammonia_plants_clean["capacity"] = ammonia_plants_clean["Production (thousand metric tons)"] * 1e3 * 5.17

    # Add country column
    ammonia_plants_clean.loc[:, ["country", "industry"]] = ["US", "ammonia"]

    # Select relevant columns
    ammonia_plants_clean = ammonia_plants_clean[["country", "x", "y", "capacity", "industry"]]
    logger.info("Prepared ammonia plants data")
    return ammonia_plants_clean


def read_pypsa_earth_industrial_database():
    """
    Read the industrial database from a pypsa-earth/data/industrial_database.csv
    """
    # Read the industrial database
    industrial_database = pd.read_csv(snakemake.input.pypsa_earth_industrial_database)

    # Select the country
    industrial_database = industrial_database[
        industrial_database["country"].isin(snakemake.params.countries)
    ]

    # Select iron and steel and cement industries
    industrial_database = industrial_database[
        industrial_database["technology"].isin([
            "Electric arc",
            "Integrated steelworks",
            "DRI + Electric arc",
            "Cement",
            ])
    ].rename(
        columns={
            "technology": "industry",
        }
    )
    # Select relevant columns
    industrial_database = industrial_database[["country", "x", "y", "industry", "capacity"]]

    return industrial_database


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_custom_industry_demand",
            simpl="",
            clusters="10",
            planning_horizons="2020",
            demand="AB",
            configfile="configs/calibration/config.base.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # snakemake params
    countries = snakemake.params.countries
    gadm_layer_id = snakemake.params.gadm_layer_id
    gadm_clustering = snakemake.params.alternative_clustering
    shapes_path = snakemake.input.shapes_path

    # load US cities locational information
    uscities = pd.read_csv(snakemake.input.uscity_map)

    # process uscities data
    uscities_clean = process_uscities(uscities)

    # load pypsa-earth industrial database
    industrial_database = read_pypsa_earth_industrial_database()

    if snakemake.params.add_ethanol:
        # load ethanol plants
        ethanol_plants_raw = pd.read_excel(snakemake.input.ethanol_plants, skiprows=2)

        # clean ethanol plants data
        ethanol_plants_clean = prepare_ethanol_plants(ethanol_plants_raw, uscities_clean)
    else:
        ethanol_plants_clean = pd.DataFrame(columns=["country", "y", "x", "capacity", "industry"])


    if snakemake.params.add_ammonia:
        # load ammonia plants
        ammonia_plants_raw = pd.read_excel(snakemake.input.ammonia_plants, sheet_name="Statista", index_col=0)

        # clean ammonia plants data
        ammonia_plants_clean = prepare_ammonia_plants(ammonia_plants_raw, uscities_clean)
    else:
        ammonia_plants_clean = pd.DataFrame(columns=["country", "y", "x", "capacity", "industry"])


    if snakemake.params.add_steel:
        # select steel plants from pypsa-earth industrial database
        steel_plants = industrial_database[industrial_database["industry"].isin([
            "Electric arc",
            "Integrated steelworks",
            "DRI + Electric arc",
        ])]
    else:
        steel_plants = pd.DataFrame(columns=["country", "x", "y", "industry", "capacity"])


    if snakemake.params.add_cement:
        # select cement plants from pypsa-earth industrial database
        cement_plants = industrial_database[industrial_database["industry"] == "Cement"]
    else:
        cement_plants = pd.DataFrame(columns=["country", "x", "y", "industry", "capacity"])

    # combine ethanol, ammonia, steel and cement plants data
    combined_plants = pd.concat([ethanol_plants_clean, ammonia_plants_clean, steel_plants, cement_plants], ignore_index=True, axis=0)

    # Map industry to buses
    custom_industrial_database = map_industry_to_buses(combined_plants, countries, gadm_layer_id, shapes_path, gadm_clustering)

    # Reset index and rename columns
    custom_industrial_database = custom_industrial_database.reset_index().rename(columns={"gadm_1": "bus"})

    # Groupby buses and industry, and sum the production capacity
    industrial_demand = custom_industrial_database.groupby(["bus", "industry"])["capacity"].sum().reset_index()

    # Format the industry demand by pivoting
    industrial_demand = industrial_demand.pivot(index="bus", columns="industry", values="capacity").rename_axis("MWh/a (kton/a)").reset_index().fillna(0)

    # Save the industrial database to CSV
    industrial_demand.to_csv(snakemake.output.industrial_energy_demand_per_node, index=False)
    logger.info("Custom industry demands were saved to CSV")
    