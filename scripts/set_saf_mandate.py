# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import pandas as pd
import numpy as np

from scripts._helper import (
    mock_snakemake,
    update_config_from_wildcards,
    create_logger,
    configure_logging,
    load_network,
)

logger = create_logger(__name__)


def add_ekerosene_buses(n, config_file=None):
    """
    Adds e-kerosene buses, carrier, stores and (optionally) links to oil buses.

    If pre-OB3 tax credits are active:
      - no e-kerosene -> oil backflow is added
      - avoids structural overproduction

    Otherwise:
      - behaviour identical to main
    """

    pre_ob3 = (
        config_file.get("policies", {})
        .get("pre_ob3_tax_credits", False)
        if config_file is not None
        else False
    )

    oil_buses = n.buses.query("carrier in 'oil'")

    ekerosene_buses = [b.replace("oil", "e-kerosene") for b in oil_buses.index]

    # e-kerosene buses
    n.madd(
        "Bus",
        ekerosene_buses,
        location=oil_buses.location.values,
        carrier="e-kerosene",
    )

    # carrier
    if "e-kerosene" not in n.carriers.index:
        n.add(
            "Carrier",
            "e-kerosene",
            co2_emissions=n.carriers.loc["oil", "co2_emissions"],
        )

    # Stores
    n.madd(
        "Store",
        [b + " Store" for b in ekerosene_buses],
        bus=ekerosene_buses,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="e-kerosene",
    )

    # backflow only if not pre-OB3
    if not pre_ob3:
        n.madd(
            "Link",
            [b + "-to-oil" for b in ekerosene_buses],
            bus0=ekerosene_buses,
            bus1=oil_buses.index,
            carrier="e-kerosene-to-oil",
            p_nom_extendable=True,
            efficiency=1.0,
        )
        logger.info("Added e-kerosene -> oil backflow links (post-OB3)")
    else:
        logger.info("Pre-OB3: e-kerosene dumping to oil DISABLED")

    logger.info("Added e-kerosene buses, carrier and stores")


def reroute_FT_output(n):
    """
    Reroute Fischer-Tropsch output from oil buses to e-kerosene buses.
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

    mandate_enabled = config["saf_mandate"]["enable_mandate"]

    add_ekerosene_buses(n, config_file=config)
    reroute_FT_output(n)

    if mandate_enabled:
        rate = get_dynamic_blending_rate(config)
        redistribute_aviation_demand(n, rate)
        logger.info("SAF mandate active")
    else:
        logger.info("SAF mandate inactive")

    n.export_to_netcdf(snakemake.output.modified_network)