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
    Adds e-kerosene buses and carrier.
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

    logger.info("Added e-kerosene buses and carrier")


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
    Split aviation demand between kerosene and e-kerosene.
    """
    aviation_loads = n.loads.query("carrier == 'kerosene for aviation'")

    # store original demand
    original_p_set = aviation_loads.p_set.copy()

    # fossil share
    n.loads.loc[aviation_loads.index, "p_set"] = original_p_set * (1.0 - rate)

    # SAF share
    n.madd(
        "Load",
        aviation_loads.index.str.replace("kerosene", "e-kerosene"),
        bus=aviation_loads.bus.str.replace("oil", "e-kerosene").values,
        carrier="e-kerosene for aviation",
        p_set=original_p_set.values * rate,
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

    mandate_enabled = config["saf_mandate"]["enable_mandate"]

    add_ekerosene_buses(n)
    reroute_FT_output(n)

    if mandate_enabled:
        rate = get_dynamic_blending_rate(config)
        redistribute_aviation_demand(n, rate)
        logger.info("SAF mandate active")
    else:
        logger.info("SAF mandate inactive")

    n.export_to_netcdf(snakemake.output.modified_network)