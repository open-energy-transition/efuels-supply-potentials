# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../submodules/pypsa-earth/scripts/")))
import pandas as pd
import numpy as np
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, \
                            configure_logging, load_network
from _helpers import prepare_costs


logger = create_logger(__name__)


def add_ammonia(n):
    """
        Adds ammonia buses and stores, and adds links between ammonia and hydrogen bus
    """
    # add ammonia carrier
    n.add("Carrier", "NH3")

    # add ammonia bus
    n.madd(
        "Bus",
        nodes + " NH3",
        location=nodes,
        carrier="NH3",
    )
    logger.info("Added Ammonia buses and carrier")

    # enable ammonia production flexibility
    if "ammonia" in config["custom_industry"]["production_flexibility"]:
        # add ammonia stores
        n.madd(
            "Store",
            nodes + " ammonia store",
            bus=nodes + " NH3",
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="ammonia store",
            capital_cost=costs.at["NH3 (l) storage tank incl. liquefaction", "fixed"],
            lifetime=costs.at["NH3 (l) storage tank incl. liquefaction", "lifetime"],
        )
        logger.info("Added Ammonia stores")

    # add Haber-Bosch process to produce ammonia from hydrogen and electricity
    n.madd(
        "Link",
        nodes + " Haber-Bosch",
        bus0=nodes,
        bus1=nodes + " NH3",
        bus2=nodes + " H2",
        p_nom_extendable=True,
        carrier="Haber-Bosch",
        efficiency=1 / costs.at["Haber-Bosch", "electricity-input"],
        efficiency2=-costs.at["Haber-Bosch", "hydrogen-input"]
        / costs.at["Haber-Bosch", "electricity-input"],
        capital_cost=costs.at["Haber-Bosch", "fixed"]
        / costs.at["Haber-Bosch", "electricity-input"],
        marginal_cost=costs.at["Haber-Bosch", "VOM"]
        / costs.at["Haber-Bosch", "electricity-input"],
        lifetime=costs.at["Haber-Bosch", "lifetime"],
    )
    logger.info("Added Haber-Bosch process to produce ammonia from hydrogen and electricity")

    # add ammonia demand
    p_set = industrial_demand.loc[nodes, "ammonia"].rename(index=lambda x: x + " NH3") / nhours
    n.madd(
        "Load",
        nodes + " NH3",
        bus=nodes + " NH3",
        p_set=p_set,
        carrier="NH3",
    )
    logger.info("Added ammonia demand to ammonia buses")

    # CCS retrofit for ammonia
    if "ammonia" in snakemake.params.ccs_retrofit:
        # prepare buses for SMR CC
        SMR_CC_links = n.links.query("carrier == 'SMR'").copy().rename(index=lambda x: x + " CC")
        gas_buses = SMR_CC_links.bus0
        h2_buses = SMR_CC_links.bus1
        co2_stored_buses = gas_buses.str.replace("gas", "co2 stored")
        elec_buses = gas_buses.str.replace(" gas", "")

        # TODO: revise capital and marginal costs of SMR CC
        # compute capital and marginal costs of SMR CC
        capital_cost = (
            costs.at["SMR", "fixed"]
            + costs.at["ammonia capture retrofit", "fixed"]
            * costs.at["gas", "CO2 intensity"]
            / costs.at["ammonia capture retrofit", "capture_rate"]
        )
        # no marginal cost for SMR as there is no VOM for SMR and ammonia capture retrofit

        # add SMR CC
        n.madd(
            "Link",
            SMR_CC_links.index,
            bus0=gas_buses,
            bus1=h2_buses,
            bus2="co2 atmoshpere",
            bus3=co2_stored_buses,
            bus4=elec_buses,
            p_nom_extendable=True,
            carrier="SMR CC",
            efficiency=costs.at["SMR CC", "efficiency"],
            efficiency2=costs.at["gas", "CO2 intensity"]
            * (1 - costs.at["ammonia capture retrofit", "capture_rate"]),
            efficiency3=costs.at["gas", "CO2 intensity"]
            * costs.at["ammonia capture retrofit", "capture_rate"],
            efficiency4=-costs.at["ammonia capture retrofit", "electricity-input"]
            / costs.at["ammonia capture retrofit", "capture_rate"]
            * costs.at["gas", "CO2 intensity"],
            capital_cost=capital_cost,
            lifetime=costs.at["ammonia capture retrofit", "lifetime"],
        )
        logger.info("Added SMR CC to retrofit ammonia plants")


def add_ethanol(n):
    """
        Adds ethanol buses and stores, and adds links between ethanol and hydrogen bus
    """
    # bioethanol crop carrier
    n.add("Carrier", "bioethanol crop")

    # add bioethanol crop bus
    n.madd(
        "Bus",
        nodes + " bioethanol crop",
        location=nodes,
        carrier="bioethanol crop",
    )

    # add bioethanol crop stores
    n.madd(
        "Store",
        nodes + " bioethanol crop store",
        bus=nodes + " bioethanol crop",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="bioethanol crop store",
    )

    # add bioethanol crop generator
    # TODO: revise if marginal cost is needed
    n.madd(
        "Generator",
        nodes + " bioethanol crop",
        bus=nodes + " bioethanol crop",
        p_nom_extendable=True,
        carrier="bioethanol crop",
        marginal_cost=costs.at["bioethanol crops", "fuel"],
    )
    logger.info("Added bioethanol crop carrier, buses, generators and stores")

    # add ethanol carrier
    n.add("Carrier", "ethanol")

    # add ethanol bus
    n.madd(
        "Bus",
        nodes + " ethanol",
        location=nodes,
        carrier="ethanol",
    )
    logger.info("Added ethanol carrier and buses")

    # add links of ethanol from starch crop
    n.madd(
        "Link",
        nodes + " ethanol from starch",
        bus0=nodes + " bioethanol crop",
        bus1=nodes + " ethanol",
        p_nom_extendable=True,
        carrier="ethanol from starch",
        efficiency=costs.at["ethanol from starch crop", "efficiency"],
        capital_cost=costs.at["ethanol from starch crop", "fixed"]
        / costs.at["ethanol from starch crop", "efficiency"],
        marginal_cost=costs.at["ethanol from starch crop", "VOM"],
        lifetime=costs.at["ethanol from starch crop", "lifetime"],
    )
    logger.info("Added links to model starch-based ethanol plants")

    # add ethanol demand
    p_set = industrial_demand.loc[nodes, "ethanol"].rename(index=lambda x: x + " ethanol") / nhours
    n.madd(
        "Load",
        nodes + " ethanol",
        bus=nodes + " ethanol",
        p_set=p_set,
        carrier="ethanol",
    )
    logger.info("Added ethanol demand to ethanol buses")

    # CCS retrofit for ethanol
    if "ethanol" in snakemake.params.ccs_retrofit:
        # calculate capital and marginal costs of ethanol from starch CC
        # TODO: revise capital and marginal costs of ethanol from starch CC
        capital_cost = (
            costs.at["ethanol from starch crop", "fixed"]
            / costs.at["ethanol from starch crop", "efficiency"]
            + costs.at["ethanol capture retrofit", "fixed"]
            * costs.at["bioethanol crops", "CO2 intensity"] # TODO: we do not have it yet
            / costs.at["ethanol capture retrofit", "capture_rate"]
        )
        marginal_cost = (
            costs.at["ethanol from starch crop", "VOM"]
            # + costs.at["ethanol capture retrofit", "VOM"]
            # * costs.at["ethanol from starch crop", "CO2 intensity"]
            # / costs.at["ethanol capture retrofit", "capture_rate"]
        )

        # add ethanol from starch CC
        n.madd(
            "Link",
            nodes + " ethanol from starch CC",
            bus0=nodes + " bioethanol crop",
            bus1=nodes + " ethanol",
            bus2=nodes + " co2 stored",
            bus3=nodes,
            p_nom_extendable=True,
            carrier="ethanol from starch CC",
            efficiency=costs.at["ethanol from starch crop", "efficiency"],
            efficiency2=costs.at["bioethanol crops", "CO2 intensity"]
            * costs.at["ethanol capture retrofit", "capture_rate"], # TODO: revise conversion rate
            efficiency3=-costs.at["ethanol capture retrofit", "electricity-input"]
            * costs.at["bioethanol crops", "CO2 intensity"]
            / costs.at["ethanol capture retrofit", "capture_rate"], # TODO: revise conversion rate
            capital_cost=capital_cost,
            marginal_cost=marginal_cost,
            lifetime=costs.at["ethanol capture retrofit", "lifetime"],
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_custom_industry",
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

    # load custom industrial energy demands
    industrial_demand = pd.read_csv(snakemake.input.industrial_energy_demand_per_node, index_col=0)

    # get industry nodes
    nodes = industrial_demand.rename_axis("Bus").index

    # get number of hours and years
    nhours = n.snapshot_weightings.generators.sum()
    Nyears = nhours / 8760

    # Prepare the costs dataframe
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs["USD2013_to_EUR2013"],
        snakemake.params.costs["fill_values"],
        Nyears,
    )

    # add ammonia industry
    if config["custom_industry"]["ammonia"]:
        add_ammonia(n)

    # add ethanol industry
    if config["custom_industry"]["ethanol"]:
        add_ethanol(n)

    # save the modified network
    n.export_to_netcdf(snakemake.output.modified_network)
