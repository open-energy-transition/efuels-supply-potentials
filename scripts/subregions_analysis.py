# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import pandas as pd
import numpy as np
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


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "preprocess_subregions_data",
            configfile="configs/calibration/config.base.yaml",
            simpl="",
            ll="copt",
            clusters=10,
            opts="Co2L-24H",
            sopts="24H",
            planning_horizons=2020,
            discountrate="0.071",
            demand="AB",
        )

    configure_logging(snakemake)
    
    network = snakemake.input.network
    path_shapes = snakemake.input.path_shapes
    distance_crs =  "EPSG:3857"

    n = nearest_shape(network, path_shapes, distance_crs)
    n.export_to_netcdf(snakemake.output.network)

    capacities_df = preprocess_capacities(n)

    fig1 = px.bar(capacities_df, barmode='stack', text_auto='.1f', orientation='h')
    fig1.update_layout(width=1000, yaxis_title='Installed capacity PyPSA (GW)')
    fig1.write_image("testplot.png")