# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import pandas as pd
import numpy as np
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, \
                            configure_logging, load_network

logger = create_logger(__name__)


def add_ekerosene_buses(n):
    """
    Adds e-kerosene buses, carrier, stores, and links to oil buses
    """
    oil_buses = n.buses.query("carrier == 'oil'")

    ekerosene_buses = [b.replace("oil", "e-kerosene") for b in oil_buses.index]
    n.madd(
        "Bus",
        ekerosene_buses,
        location=oil_buses.location.values,
        carrier="e-kerosene",
    )

    if "e-kerosene" not in n.carriers.index:
        n.add(
            "Carrier",
            "e-kerosene",
            co2_emissions=n.carriers.loc["oil", "co2_emissions"],
        )

    n.madd(
        "Store",
        [b + " Store" for b in ekerosene_buses],
        bus=ekerosene_buses,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="e-kerosene",
    )

    n.madd(
        "Link",
        [b + "-to-oil" for b in ekerosene_buses],
        bus0=ekerosene_buses,
        bus1=oil_buses.index,
        carrier="e-kerosene-to-oil",
        p_nom_extendable=True,
        efficiency=1.0,
    )

    logger.info("Added e-kerosene buses, carrier, stores, and oil backflow links")


def reroute_FT_output(n):
    """
    Reroutes Fischer-Tropsch output from oil buses to e-kerosene buses
    """
    ft_links = n.links.query("carrier == 'Fischer-Tropsch'").index
    n.links.loc[ft_links, "bus1"] = (
        n.links.loc[ft_links, "bus1"].str.replace("oil", "e-kerosene")
    )
    logger.info("Rerouted Fischer-Tropsch output to e-kerosene buses")


def get_dynamic_blending_rate(config):
    """
    Read SAF blending rate from saf_scenarios.csv
    """
    saf_scenario = snakemake.params.saf_scenario
    year = str(snakemake.wildcards.planning_horizons)
    df = pd.read_csv(snakemake.input.saf_scenarios, index_col=0)

    rate = float(df.loc[saf_scenario, year])
    logger.info(f"Requested SAF blending rate: {rate:.3f}")
    return rate


def redistribute_aviation_demand(n, rate):
    """
    Split aviation demand between kerosene and e-kerosene
    """
    aviation_loads = n.loads.query("carrier == 'kerosene for aviation'")

    n.loads.loc[aviation_loads.index, "p_set"] *= (1.0 - rate)

    n.madd(
        "Load",
        aviation_loads.index.str.replace("kerosene", "e-kerosene"),
        bus=aviation_loads.bus.str.replace("oil", "e-kerosene").values,
        carrier="e-kerosene for aviation",
        p_set=aviation_loads.p_set.fillna(0.0).values * rate,
    )

    logger.info(f"Applied SAF share: {rate:.3f}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "set_saf_mandate",
            configfile="configs/scenarios/config.2040.yaml",
            simpl="",
            ll="copt",
            clusters=10,
            opts="24H",
            sopts="24H",
            planning_horizons=2040,
            discountrate="0.071",
            demand="AB",
        )

    configure_logging(snakemake)

    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    n = load_network(snakemake.input.network)

    pre_ob3 = config.get("policies", {}).get("pre_ob3_tax_credits", False)
    mandate_enabled = config["saf_mandate"]["enable_mandate"]

    if pre_ob3 and mandate_enabled:
        add_ekerosene_buses(n)
        reroute_FT_output(n)

        requested_rate = get_dynamic_blending_rate(config)

        total_aviation_demand = (
            n.loads.query("carrier == 'kerosene for aviation'")["p_set"].sum()
        )

        ft_links = n.links.query("carrier == 'Fischer-Tropsch'")

        hours = n.snapshot_weightings.generators.sum()
        ft_eff = ft_links.efficiency.mean()

        max_ekerosene = ft_links.p_nom.sum() * hours * ft_eff
        max_rate = (
            max_ekerosene / total_aviation_demand
            if total_aviation_demand > 0
            else 0.0
        )

        effective_rate = min(requested_rate, max_rate)

        logger.info(
            f"Pre-OB3 SAF capped: requested={requested_rate:.3f}, "
            f"max_feasible={max_rate:.3f}, applied={effective_rate:.3f}"
        )

        redistribute_aviation_demand(n, effective_rate)

        n.links.loc[ft_links.index, "p_nom_extendable"] = False
        n.links = n.links[~n.links.carrier.isin(["e-kerosene-to-oil"])]

    else:
        add_ekerosene_buses(n)
        reroute_FT_output(n)

        if mandate_enabled:
            rate = get_dynamic_blending_rate(config)
            redistribute_aviation_demand(n, rate)

    n.export_to_netcdf(snakemake.output.modified_network)