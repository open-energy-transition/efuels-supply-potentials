# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import numpy as np
import logging
import pypsa
import geopandas as gpd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger

from pypsa.linopf import (
    define_constraints,
    define_variables,
    get_var,
    ilopf,
    join_exprs,
    linexpr,
    network_lopf,
)

logger = create_logger(__name__)

def load_network(path):
    """
    Load the network from the given path
    """
    n = pypsa.Network(path)
    n.optimize.create_model()

    return n


def process_targets_data(path, carrier):
    df = pd.read_csv(path)
    df.rename(columns={"Unnamed: 0": "state"}, inplace=True)
    df = df.melt(id_vars="state", var_name="year", value_name="target")
    df["carrier"] = ", ".join(carrier)

    return df


def attach_state_to_buses(network, path_shapes, distance_crs):
    """
    Attach state to buses
    """
    # Read the shapefile using geopandas
    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes["ISO_1"] = shapes["ISO_1"].apply(lambda x: x.split("-")[1])
    shapes.rename(columns={"ISO_1": "State"}, inplace=True)

    pypsa_gpd = gpd.GeoDataFrame(
            network.buses, 
            geometry=gpd.points_from_xy(network.buses.x, network.buses.y), 
            crs=4326
        )

    bus_cols = network.buses.columns
    bus_cols = list(bus_cols) + ["State"]

    st_buses = gpd.sjoin_nearest(shapes, pypsa_gpd, how="right")[bus_cols]

    network.buses["state"] = st_buses["State"]

    return network


def add_constraints(network, constraints_df):
    for _, constraint_row in constraints_df.iterrows():
        region_list = [constraint_row.state.strip()]
        region_buses = network.buses[network.buses.state.isin(region_list)]

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in constraint_row.carrier.split(",")]


        # Filter region generators
        region_gens = network.generators[network.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if not region_gens_eligible.empty:
            p_eligible = network.model["Generator-p"].sel(
                snapshot=constraint_row.year,
                Generator=region_gens_eligible.index,
            )

            # power level buses
            pwr_buses = n.buses[(n.buses.carrier == "AC") & (n.buses.index.isin(region_buses.index))]
            # links delievering power within the region
            # removes any transmission links
            pwr_links = n.links[(n.links.bus0.isin(pwr_buses.index)) & ~(n.links.bus1.isin(pwr_buses.index))]
            region_demand = n.model["Link-p"].sel(period=constraint_row.planning_horizon, Link=pwr_links.index)

            lhs = p_eligible.sum() - (constraint_row.target * region_demand.sum())
            rhs = 0

            # Add constraint
            network.model.add_constraints(
                lhs >= rhs,
                name=f"GlobalConstraint-{constraint_row.state}_{constraint_row.year}_rps_limit",
            )
            logger.info(f"Added RPS {constraint_row.name} for {constraint_row.planning_horizon}.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_res_constraints",
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
    
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    res_carriers = ["solar", "onwind", "offwind-ac", "offwind-dc", "hydro", "geothermal"]
    ces_carriers = res_carriers + ["nuclear"]

    ces_data = process_targets_data(
        snakemake.input.ces_path,
        ces_carriers,
    )

    res_data = process_targets_data(
        snakemake.input.res_path,
        res_carriers,
    )

    path_shapes = snakemake.input.gadm_shape_path # retrieve from snakemake
    # distance_crs =  "EPSG:3857" # pick from config

    n_path = snakemake.input.network
    n = load_network(n_path)
    n = attach_state_to_buses(n, path_shapes, snakemake.params.distance_crs)

    planning_horizons = config["scenario"]["planning_horizons"]
    ces_data = ces_data[(ces_data["year"].isin(planning_horizons))
                    & (ces_data["target"] > 0.0)
                    & (ces_data["state"].isin(n.buses["state"].unique()))]

    add_constraints(n, ces_data)

    n.export_to_netcdf(snakemake.output[0])
