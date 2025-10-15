# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../submodules/pypsa-earth/scripts/")))
import pandas as pd
import numpy as np
from types import SimpleNamespace
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, \
                            configure_logging, load_network
from add_custom_existing_baseyear import add_build_year_to_new_assets, set_lifetimes
from _helpers import prepare_costs


logger = create_logger(__name__)
spatial = SimpleNamespace()


def add_brownfield(n, n_p, year, planning_horizon_p):
    logger.info(f"Preparing brownfield for the year {year}")

    # set optimised capacities of previous horizon as minimum for lines
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    # set optimised capacities of previous horizon as minimum for DC links
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)

    # reduce non-infinity p_nom_max for all extendable components of brownfield by p_nom of previous horizon
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"
        extendable_assets = c.df[(c.df[f"{attr}_nom_extendable"]) & 
                                 (c.df.build_year == year) &
                                 (c.df[f"{attr}_nom_max"] != np.inf)]
        # loop over extendable assets and reduce their p_nom_max by the p_nom of the previous horizon
        for idx in extendable_assets.index:
            # get common asset name (eg. US0 0 onwind)
            asset_name = idx.split(f"-{year}")[0]
            # get total p_nom_opt installed previously
            asset_prev_p_nom_opt = c.df[c.df.index.str.contains(asset_name)]["p_nom_opt"].sum()
            # reduce p_nom_max of the extendable asset by installed capacity
            c.df.loc[idx, f"{attr}_nom_max"] -= asset_prev_p_nom_opt
            # clip p_nom_max by lower bound to 0
            c.df.loc[idx, f"{attr}_nom_max"] = max(0, c.df.loc[idx, f"{attr}_nom_max"])

        # set p_nom = 0 and p_nom_min = 0 for all assets with build_year == year
        c.df.loc[c.df.build_year == year, f"{attr}_nom"] = 0
        c.df.loc[c.df.build_year == year, f"{attr}_nom_min"] = 0


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_custom_brownfield",
            simpl="",
            clusters="10",
            ll="copt",
            opts="24H",
            planning_horizons="2035",
            sopts="24H",
            discountrate=0.071,
            demand="AB",
            h2export="10",
            configfile="configs/scenarios/config.myopic.yaml"
        )

    configure_logging(snakemake)

    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    # current year and previous horizon
    year = int(snakemake.wildcards.planning_horizons)
    planning_horizon_p = int(snakemake.params.planning_horizon_p)

    # load the prenetwork for the brownfield
    n = load_network(snakemake.input.network)

    # read costs assumptions
    Nyears = n.snapshot_weightings.generators.sum() / 8760.0
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.config["costs"],
        snakemake.params.costs["output_currency"],
        snakemake.params.costs["fill_values"],
        Nyears,
        snakemake.params.costs["default_USD_to_EUR"],
        reference_year=snakemake.config["costs"].get("reference_year", 2020),
    )

    # set lifetime for nuclear, geothermal, and ror generators manually to non-infinity values
    set_lifetimes(n, costs)

    # add build_year to new assets
    add_build_year_to_new_assets(n, year)

    # load the solved network for previous horizon
    n_p = load_network(snakemake.input.network_p)

    # add brownfield assets from the previous solved network
    add_brownfield(n, n_p, year, planning_horizon_p)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
