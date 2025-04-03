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
    """
    
    shapes = gpd.read_file(path_shapes, crs=distance_crs).set_index("NAME")["geometry"]

    for i in n.buses.index:
        point = Point(n.buses.loc[i, "x"], n.buses.loc[i, "y"])
        distance = shapes.distance(point).sort_values()
        if distance.iloc[0] < 1:
            n.buses.loc[i, "country"] = distance.index[0]
        else:
            print(
                f"The bus {i} is {distance.iloc[0]} km away from {distance.index[0]} "
            )

    return n

def preprocess_capacities(n_subregion):
    
    # since there is an issue with honolulu being further away from the US0 20 bus, we can manually assign it to the US0 20 bus
    n_subregion.buses.loc[n_subregion.buses["country"] == "US", "country"] = "Honolulu"

    series_gen_to_use = n_subregion.generators.groupby(["carrier", "bus"]).p_nom.sum()
    series_sto_to_use = n_subregion.storage_units.groupby(["carrier","bus"]).p_nom.sum()

    series_to_use = series_gen_to_use._append(series_sto_to_use)
    df = series_to_use.unstack(level=0, fill_value=0)
    df.index = df.index.map(lambda x: n_subregion.buses.loc[x, "country"])
    df = df.groupby(df.index).sum()

    wind_cols = [x for x in df.columns.unique() if 'wind' in x]
    wind_df = df[wind_cols].agg(["sum"], axis=1)
    wind_df.rename(columns={"sum": "wind"}, inplace=True)

    gas_cols = ["CCGT", "OCGT"]
    gas_df = df[gas_cols].agg(["sum"], axis=1)
    gas_df.rename(columns={"sum": "gas"}, inplace=True)

    merged_df = pd.concat([df, wind_df, gas_df], axis=1)
    merged_df.drop(wind_cols + gas_cols, axis=1, inplace=True)
    
    return merged_df

def preprocess_generation(n_subregion):
    gen_capacities = (n_subregion.generators_t
                        .p.multiply(n_subregion.snapshot_weightings.objective, axis=0).T
                        .groupby([n_subregion.generators.carrier, n_subregion.generators.bus])
                        .sum()).sum(axis=1).unstack(level=0, fill_value=0)

    gen_capacities.index = gen_capacities.index.map(lambda x: n_subregion.buses.loc[x, "country"])
    gen_capacities = gen_capacities.groupby(gen_capacities.index).sum()

    storage_capacities = (n_subregion.storage_units_t
                            .p.multiply(n_subregion.snapshot_weightings.objective, axis=0).T
                            .groupby([n_subregion.storage_units.carrier, n_subregion.storage_units.bus])
                            .sum()).sum(axis=1).unstack(level=0, fill_value=0)

    storage_capacities.index = storage_capacities.index.map(lambda x: n_subregion.buses.loc[x, "country"])
    storage_capacities = storage_capacities.groupby(storage_capacities.index).sum()

    generation_pypsa = ((pd.concat([gen_capacities, storage_capacities], axis=1)) / 1e6).fillna(0).round(2)

    # Aggregate fossil fuel, hydro, and wind generation
    generation_pypsa["fossil fuels"] = generation_pypsa[[
        "CCGT", "OCGT", "coal"]].sum(axis=1)
    generation_pypsa["hydro"] = generation_pypsa[["hydro", "ror", "PHS"]].sum(axis=1)
    generation_pypsa["wind"] = generation_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum(axis=1)

    generation_pypsa.drop(
        columns=["CCGT", "OCGT", "coal", "ror", "PHS","offwind-ac", "offwind-dc", "onwind"], inplace=True)

    return generation_pypsa

def preprocess_demand(n_subregion):
    demand_df = n_subregion.loads_t.p.T.sum(axis=1)
    demand_df.index = demand_df.index.map(lambda x: n_subregion.buses.loc[x, "country"])
    demand_df = demand_df.groupby(demand_df.index).sum()
    demand_df.index.name = "country"
    demand_df = demand_df.to_frame().rename(columns={0: "demand"})

    return demand_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "subregion_analysis",
            configfile="configs/calibration/config.base.yaml",
        )

    configure_logging(snakemake)
    
    network = pypsa.Network(snakemake.input.p_network)
    path_shapes = snakemake.input.path_shapes
    distance_crs =  "EPSG:3857"

    n = nearest_shape(network, path_shapes, distance_crs)
    n.export_to_netcdf(snakemake.output.network)

    capacities_df = preprocess_capacities(n)
    generation_df = preprocess_generation(n)
    demand_df = preprocess_demand(n)

    fig1 = px.bar(capacities_df, barmode='stack', text_auto='.1f', orientation='h')
    fig1.update_layout(width=1000, yaxis_title='Installed capacity PyPSA (GW)')
    fig1.write_image(snakemake.output.installed_capacity_plot)

    fig2 = px.bar(generation_df, barmode='stack', text_auto='.1f', orientation='h')
    fig2.update_layout(width=1000, yaxis_title='Generation capacity PyPSA (TWh)')
    fig2.write_image(snakemake.output.generation_capacity_plot)

    fig3 = px.bar(demand_df, barmode='stack', text_auto='.1f', orientation='v')
    fig3.update_layout(width=1000, yaxis_title='Demand PyPSA (TWh)')
    fig3.write_image(snakemake.output.demand_plot)