# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import datetime as dt
import pandas as pd
import geopandas as gpd
import numpy as np
import pypsa
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, \
                            configure_logging, get_colors


def parse_inputs():
    """
    Load all input data  
    Parameters
    ----------
    Returns
    -------
    df_ba_demand: pandas dataframe
        Balancing Authority demand profiles
    gdf_ba_shape: geopandas dataframe
        Balancing Authority shapes
    df_utility_demand: geopandas dataframe
        Output of preprocess_demand_data - demand mapped to utility level shapes and holes
    pypsa_network: pypsa
        network to obtain pypsa bus information
    """

    df_ba_demand1 = pd.read_csv(snakemake.input.BA_demand_path1, index_col="period")
    df_ba_demand2 = pd.read_csv(snakemake.input.BA_demand_path2, index_col="period")
    df_ba_demand = df_ba_demand1._append(df_ba_demand2)
    df_ba_demand = df_ba_demand[~df_ba_demand.index.duplicated(keep="first")]
    df_ba_demand = df_ba_demand.replace(0, np.nan)
    df_ba_demand = df_ba_demand.dropna(axis=1)

    gdf_ba_shape = gpd.read_file(snakemake.input.BA_shape_path)
    gdf_ba_shape = gdf_ba_shape.to_crs(3857)

    df_utility_demand = gpd.read_file(snakemake.input.utility_demand_path)
    df_utility_demand.rename(columns={"index_right": "index_right_1"}, inplace=True)
    df_utility_demand = df_utility_demand.to_crs(3857)

    pypsa_network = pypsa.Network(snakemake.input.base_network)

    return df_ba_demand, gdf_ba_shape, df_utility_demand, pypsa_network


def build_demand_profiles(df_utility_demand, df_ba_demand, gdf_ba_shape, pypsa_network):
    """
    Build spatiotemporal demand profiles 
    Parameters
    ----------
    df_utility_demand: geopandas dataframe
        Output of preprocess_demand_data - demand mapped to utility level shapes and holes
    df_ba_demand: pandas dataframe
        Balancing Authority demand profiles
    gdf_ba_shape: geopandas dataframe
        Balancing Authority shapes
    pypsa_network: pypsa
        network to obtain pypsa bus information
    Returns
    -------
    df_demand_bus_timeshifted: pandas dataframe
        bus-wise demand profiles
    """

    # Obtaining the centroids of the Utility demands
    df_utility_centroid = df_utility_demand.copy()
    df_utility_centroid.geometry = df_utility_centroid.geometry.centroid

    # Removing those shapes which do not have a representation in the demand timeseries data
    demand_columns = df_ba_demand.columns
    demand_columns = [x.split("_")[1] for x in demand_columns if x.endswith("_D")]
    drop_shape_rows = list(set(gdf_ba_shape["EIAcode"].tolist()) - set(demand_columns))
    gdf_ba_shape_filtered = gdf_ba_shape[~gdf_ba_shape["EIAcode"].isin(drop_shape_rows)]
    gdf_ba_shape_filtered.geometry = gdf_ba_shape_filtered.geometry.centroid
    gdf_ba_shape_filtered["color_ba"] = get_colors(len(gdf_ba_shape_filtered))

    df_utility_centroid = gpd.sjoin_nearest(
        df_utility_centroid, gdf_ba_shape_filtered, how="left"
    )
    df_utility_centroid.rename(columns={"index_right": "index_right_2"}, inplace=True)

    # temporal scaling factor
    df_utility_centroid["temp_scale"] = df_utility_centroid.apply(
        lambda x: df_ba_demand[f"E_{x['EIAcode']}_D"].sum()
        / 1e3
        / x["Sales (Megawatthours)"],
        axis=1,
    )

    # Mapping demand utilities to nearest PyPSA bus
    df_reqd = pypsa_network.buses.query('carrier == "AC"')
    pypsa_gpd = gpd.GeoDataFrame(
        df_reqd, geometry=gpd.points_from_xy(df_reqd.x, df_reqd.y), crs=4326
    )
    pypsa_gpd = pypsa_gpd.to_crs(3857)
    pypsa_gpd["color"] = get_colors(len(pypsa_gpd))

    df_utility_centroid = gpd.sjoin_nearest(df_utility_centroid, pypsa_gpd, how="left")
    df_utility_centroid.rename(columns={"index_right": "PyPSA_bus"}, inplace=True)

    df_demand_bus = pd.DataFrame(
        index=pd.to_datetime(df_ba_demand.index),
        columns=df_reqd.index.tolist(),
        data=0.0,
    )

    for col in df_demand_bus.columns:
        utility_rows = df_utility_centroid.query("PyPSA_bus == @col")
        for i in np.arange(0, len(utility_rows)):
            row = utility_rows.iloc[i]
            if not np.isnan(row["temp_scale"]):
                demand_data = df_ba_demand[f"E_{row['EIAcode']}_D"].copy()
                demand_data /= row["temp_scale"]
                demand_data /= 1000  # Converting to MWh
                df_demand_bus[col] += demand_data.tolist()

    # The EIA profiles start at 6:00:00 hours on 1/1 instead of 00:00:00 hours - rolling over the time series to start at 00:00 hours
    df_demand_bus_timeshifted = df_demand_bus[-9:-3]._append(df_demand_bus[:-9])
    df_demand_bus_timeshifted = df_demand_bus_timeshifted[:8760]
    df_demand_bus_timeshifted.index = pypsa_network.snapshots
    df_demand_bus_timeshifted.index.name = "time"

    return df_demand_bus_timeshifted


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_demand_profiles_from_eia",
            configfile="configs/calibration/config.base.yaml",
        )

    configure_logging(snakemake)

    # set relevant paths
    plot_path = "plots/demand_modelling"
    os.makedirs(plot_path, exist_ok=True)


    df_ba_demand, gdf_ba_shape, df_utility_demand, pypsa_network = parse_inputs()

    df_demand_profiles = build_demand_profiles(
        df_utility_demand, df_ba_demand, gdf_ba_shape, pypsa_network
    )

    df_demand_profiles.to_csv(snakemake.output.demand_profile_path)
