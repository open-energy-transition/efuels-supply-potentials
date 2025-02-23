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
        Adds e-kerosene buses and stores
    """
    # get oil bus names
    oil_buses = n.buses.query("carrier in 'oil'")

    # add e-kerosene bus
    ekerosene_buses = [x.replace("oil", "e-kerosene") for x in oil_buses.index]
    n.madd(
        "Bus",
        ekerosene_buses,
        location=oil_buses.location,
        carrier="e-kerosene",
    )

    # add e-kerosene carrier
    n.add("Carrier", "e-kerosene", co2_emissions=n.carriers.loc["oil", "co2_emissions"])

    # add e-kerosene stores
    n.madd(
        "Store",
        [ekerosene_bus + " Store" for ekerosene_bus in ekerosene_buses],
        bus=ekerosene_buses,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="e-kerosene",
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_saf_mandate",
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

    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load the network
    n = load_network(snakemake.input.network)

    # add e-kerosene buses with store
    add_ekerosene_buses(n)

