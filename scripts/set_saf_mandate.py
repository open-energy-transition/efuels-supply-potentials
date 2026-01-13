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
    Adds e-kerosene buses and stores, and adds links between e-kerosene and oil bus
    """
    oil_buses = n.buses.query("carrier == 'oil'")

    ekerosene_buses = [x.replace("oil", "e-kerosene") for x in oil_buses.index]
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
        [bus + " Store" for bus in ekerosene_buses],
        bus=ekerosene_buses,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="e-kerosene",
    )
    logger.info("Added e-kerosene buses, carrier, and stores")

    n.madd(
        "Link",
        [x + "-to-oil" for x in ekerosene_buses],
        bus0=ekerosene_buses,
        bus1=oil_buses.index,
        carrier="e-kerosene-to-oil",
        p_nom_extendable=True,
        efficiency=1.0,
    )
    logger.info("Added links between e-kerosene and oil buses")


def reroute_FT_output(n):
    """
    Reroutes output of Fischer-Tropsch from oil to e-kerosene bus
    """
    ft_links = n.links.query("carrier == 'Fischer-Tropsch'").index

    n.links.loc[ft_links, "bus1"] = (
        n.links.loc[ft_links, "bus1"].str.replace("oil", "e-kerosene")
    )
    logger.info("Rerouted Fischer-Tropsch output from oil buses to e-kerosene buses")


def get_dynamic_blending_rate(config):
    """
    Extract the blending rate from data/saf_blending_rates/saf_scenarios.csv
    """
    saf_scenario = snakemake.params.saf_scenario
    year = str(snakemake.wildcards.planning_horizons)
    df = pd.read_csv(snakemake.input.saf_scenarios, index_col=0)

    rate = float(df.loc[saf_scenario, year])
    logger.info(f"Blending rate for scenario {saf_scenario} in {year}: {rate}")
    return rate


def redistribute_aviation_demand(n, rate):
    """
    Redistribute aviation demand to e-kerosene and kerosene based on blending rate
    """
    total_aviation_demand = n.loads.query(
        "carrier == 'kerosene for aviation'"
    )

    n.loads.loc[total_aviation_demand.index, "p_set"] *= (1 - rate)
    logger.info(
        f"Set kerosene for aviation to {(1 - rate) * 100:.1f}% of total aviation demand"
    )

    n.madd(
        "Load",
        total_aviation_demand.index.str.replace("kerosene", "e-kerosene"),
        bus=total_aviation_demand.bus.str.replace("oil", "e-kerosene").values,
        carrier="e-kerosene for aviation",
        p_set=total_aviation_demand.p_set.fillna(0).values * rate,
    )
    logger.info(
        f"Added e-kerosene for aviation demand at {rate * 100:.1f}% of total aviation demand"
    )


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

    if pre_ob3:
        if mandate_enabled:
            add_ekerosene_buses(n)
            reroute_FT_output(n)

            rate = get_dynamic_blending_rate(config)
            redistribute_aviation_demand(n, rate)

            ft_links = n.links.query("carrier == 'Fischer-Tropsch'").index
            n.links.loc[ft_links, "p_nom_extendable"] = False

            n.links = n.links[~n.links.carrier.isin(["e-kerosene-to-oil"])]

        else:
            logger.info("Pre-OB3 without SAF mandate: e-kerosene disabled")

    else:
        add_ekerosene_buses(n)
        reroute_FT_output(n)

        if mandate_enabled:
            rate = get_dynamic_blending_rate(config)
            redistribute_aviation_demand(n, rate)

    n.export_to_netcdf(snakemake.output.modified_network)