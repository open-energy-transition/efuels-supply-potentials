# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import pandas as pd
import numpy as np
import pypsa
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import warnings
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, \
                            configure_logging, load_network

warnings.simplefilter(action='ignore', category=FutureWarning)
logger = create_logger(__name__)


def nearest_shape(n, path_shapes, distance_crs):
    """
    The function nearest_shape reallocates buses to the closest "country" shape based on their geographical coordinates,
    using the provided shapefile and distance CRS.

    Parameters:
    - n: PyPSA network object containing buses with geographical coordinates.
    - path_shapes: Path to the shapefile containing country shapes.
    - distance_crs: Coordinate Reference System (CRS) used for distance calculations.

    Returns:
    - n: Updated PyPSA network object with buses assigned to the nearest country shape.
    """
    
    # Load the shapefile as a GeoDataFrame and set the index to the "NAME" column
    # The "geometry" column contains the shapes of the countries
    shapes = gpd.read_file(path_shapes, crs=distance_crs).set_index("NAME")["geometry"]

    # Iterate over each bus in the network
    for i in n.buses.index:
        # Create a Point object for the bus's geographical coordinates
        point = Point(n.buses.loc[i, "x"], n.buses.loc[i, "y"])
        
        # Calculate the distance from the bus to each country shape and sort by distance
        distance = shapes.distance(point).sort_values()
        
        # If the closest shape is within 1 km, assign the bus to that country
        if distance.iloc[0] < 1:
            n.buses.loc[i, "country"] = distance.index[0]
        else:
            # Print a warning if the bus is far from the nearest country shape
            print(
                f"The bus {i} is {distance.iloc[0]} km away from {distance.index[0]} "
            )

    # Return the updated network object
    return n


def preprocess_capacities(n_subregion):
    # Manually assign Honolulu to the US0 20 bus due to an issue with its distance calculation
    n_subregion.buses.loc[n_subregion.buses["country"] == "US", "country"] = "Honolulu"

    # Group generators by carrier and bus, then sum their nominal power
    series_gen_to_use = n_subregion.generators.groupby(["carrier", "bus"]).p_nom.sum()
    
    # Group storage units by carrier and bus, then sum their nominal power
    series_sto_to_use = n_subregion.storage_units.groupby(["carrier", "bus"]).p_nom.sum()

    # Combine the generator and storage unit data into a single series
    series_to_use = series_gen_to_use._append(series_sto_to_use)
    
    # Convert the series into a DataFrame, filling missing values with 0
    df = series_to_use.unstack(level=0, fill_value=0)
    
    # Map the bus indices to their corresponding countries
    df.index = df.index.map(lambda x: n_subregion.buses.loc[x, "country"])
    
    # Group the data by country and sum the values
    df = df.groupby(df.index).sum()

    # Identify columns related to wind generation
    wind_cols = [x for x in df.columns.unique() if 'wind' in x]
    
    # Aggregate all wind-related columns into a single "wind" column
    wind_df = df[wind_cols].agg(["sum"], axis=1)
    wind_df.rename(columns={"sum": "wind"}, inplace=True)

    # Identify columns related to gas generation (CCGT and OCGT)
    gas_cols = ["CCGT", "OCGT"]
    
    # Aggregate all gas-related columns into a single "gas" column
    gas_df = df[gas_cols].agg(["sum"], axis=1)
    gas_df.rename(columns={"sum": "gas"}, inplace=True)

    # Merge the original DataFrame with the aggregated wind and gas data
    merged_df = pd.concat([df, wind_df, gas_df], axis=1)
    
    # Drop the original wind and gas columns as they are now aggregated
    merged_df.drop(wind_cols + gas_cols, axis=1, inplace=True)
    
    # Return the processed DataFrame
    return merged_df


def preprocess_generation(n_subregion):
    """
    Processes the generation data for a given subregion by aggregating generator
    and storage unit outputs, and grouping them by country and carrier.

    Parameters:
    - n_subregion: PyPSA network object for the subregion.

    Returns:
    - generation_pypsa: DataFrame containing aggregated generation data by country and carrier.
    """

    # Calculate total generation for each generator carrier and bus, weighted by snapshot objectives
    gen_capacities = (n_subregion.generators_t
                        .p.multiply(n_subregion.snapshot_weightings.objective, axis=0).T
                        .groupby([n_subregion.generators.carrier, n_subregion.generators.bus])
                        .sum()).sum(axis=1).unstack(level=0, fill_value=0)

    # Map bus indices to their corresponding countries and group by country
    gen_capacities.index = gen_capacities.index.map(lambda x: n_subregion.buses.loc[x, "country"])
    gen_capacities = gen_capacities.groupby(gen_capacities.index).sum()

    # Calculate total generation for each storage unit carrier and bus, weighted by snapshot objectives
    storage_capacities = (n_subregion.storage_units_t
                            .p.multiply(n_subregion.snapshot_weightings.objective, axis=0).T
                            .groupby([n_subregion.storage_units.carrier, n_subregion.storage_units.bus])
                            .sum()).sum(axis=1).unstack(level=0, fill_value=0)

    # Map bus indices to their corresponding countries and group by country
    storage_capacities.index = storage_capacities.index.map(lambda x: n_subregion.buses.loc[x, "country"])
    storage_capacities = storage_capacities.groupby(storage_capacities.index).sum()

    # Combine generator and storage unit data into a single DataFrame, convert to TWh, and round values
    generation_pypsa = ((pd.concat([gen_capacities, storage_capacities], axis=1)) / 1e6).fillna(0).round(2)

    # Aggregate fossil fuel generation (CCGT, OCGT, coal) into a single column
    generation_pypsa["fossil fuels"] = generation_pypsa[[
        "CCGT", "OCGT", "coal"]].sum(axis=1)

    # Aggregate hydro generation (hydro, run-of-river, pumped hydro storage) into a single column
    generation_pypsa["hydro"] = generation_pypsa[["hydro", "ror", "PHS"]].sum(axis=1)

    # Aggregate wind generation (onshore and offshore) into a single column
    generation_pypsa["wind"] = generation_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum(axis=1)

    # Drop the original columns that have been aggregated
    generation_pypsa.drop(
        columns=["CCGT", "OCGT", "coal", "ror", "PHS", "offwind-ac", "offwind-dc", "onwind"], inplace=True)

    # Return the processed generation DataFrame
    return generation_pypsa


def preprocess_demand(n_subregion):
    """
    Processes the demand data for a given subregion by aggregating load data
    and grouping it by country.

    Parameters:
    - n_subregion: PyPSA network object for the subregion.

    Returns:
    - demand_df: DataFrame containing aggregated demand data by country.
    """

    # Sum the load data across all snapshots for each bus
    demand_df = n_subregion.loads_t.p.T.sum(axis=1)

    # Map the bus indices to their corresponding countries
    demand_df.index = demand_df.index.map(lambda x: n_subregion.buses.loc[x, "country"])

    # Group the demand data by country and sum the values
    demand_df = demand_df.groupby(demand_df.index).sum()

    # Set the index name to "country" for clarity
    demand_df.index.name = "country"

    # Convert the Series to a DataFrame and rename the column to "demand"
    demand_df = demand_df.to_frame().rename(columns={0: "demand"})

    return demand_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "subregion_analysis",
            configfile="configs/calibration/config.base.yaml",
            simpl="",
            ll="copt",
            clusters="10",
            opts="Co2L-24H",
            sopts="24H",
            planning_horizons="2020",
            discountrate="0.071",
            demand="AB",
        )

    configure_logging(snakemake)
    
    # Load the PyPSA network from the input file
    network = pypsa.Network(snakemake.input.p_network)
    
    # Path to the shapefile and the CRS for distance calculations
    path_shapes = snakemake.input.path_shapes
    distance_crs = "EPSG:3857"

    # Assign buses to the nearest country shape
    n = nearest_shape(network, path_shapes, distance_crs)
    
    # Export the updated network to a NetCDF file
    n.export_to_netcdf(snakemake.output.network)

    # Preprocess installed capacities, generation, and demand data
    capacities_df = preprocess_capacities(n)
    generation_df = preprocess_generation(n)
    demand_df = preprocess_demand(n)

    # Create and save a bar plot for installed capacities
    fig1 = px.bar(capacities_df, barmode='stack', text_auto='.1f', orientation='h')
    fig1.update_layout(width=1000, yaxis_title='Installed capacity PyPSA (GW)')
    fig1.write_image(snakemake.output.installed_capacity_plot)

    # Create and save a bar plot for generation capacities
    fig2 = px.bar(generation_df, barmode='stack', text_auto='.1f', orientation='h')
    fig2.update_layout(width=1000, yaxis_title='Generation capacity PyPSA (TWh)')
    fig2.write_image(snakemake.output.generation_capacity_plot)

    # Create and save a bar plot for demand
    fig3 = px.bar(demand_df, barmode='stack', text_auto='.1f', orientation='v')
    fig3.update_layout(width=1000, yaxis_title='Demand PyPSA (TWh)')
    fig3.write_image(snakemake.output.demand_plot)