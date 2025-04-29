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
        capital_cost=costs.at["ethanol from starch crop", "fixed"] # TODO: revise investment cost: we have EUR/MWh_eth not EUR/MW_eth
        * costs.at["ethanol from starch crop", "efficiency"],
        marginal_cost=costs.at["ethanol from starch crop", "VOM"], # TODO: revise VOM: generally it is EUR/MWh, but here it is %/year
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
            * costs.at["ethanol from starch crop", "efficiency"]
            + costs.at["ethanol capture retrofit", "fixed"] # TODO: revise investment cost: it is only 10.0,EUR/(tCO2/h)
            * costs.at["bioethanol crops", "CO2 intensity"] # TODO: we do not have it yet
            / costs.at["ethanol capture retrofit", "capture_rate"]
        )
        marginal_cost = (
            costs.at["ethanol from starch crop", "VOM"] # TODO: revise VOM: generally it is EUR/MWh, but here it is %/year
            # + costs.at["ethanol capture retrofit", "VOM"] # TODO: revise retrofit part too
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
            efficiency2=costs.at["bioethanol crops", "CO2 intensity"] # TODO: needs CO2 intensity for bioethanol crops
            * costs.at["ethanol capture retrofit", "capture_rate"], # TODO: revise conversion rate
            efficiency3=-costs.at["ethanol capture retrofit", "electricity-input"]
            * costs.at["bioethanol crops", "CO2 intensity"]
            / costs.at["ethanol capture retrofit", "capture_rate"], # TODO: revise conversion rate
            capital_cost=capital_cost,
            marginal_cost=marginal_cost,
            lifetime=costs.at["ethanol capture retrofit", "lifetime"],
        )


def add_steel(n):
    """
        Adds steel buses and stores, and adds links to produce iron and steel
    """
    # add iron ore carrier
    n.add("Carrier", "iron ore")

    # add iron ore bus
    n.madd(
        "Bus",
        nodes + " iron ore",
        location=nodes,
        carrier="iron ore",
    )

    # add iron ore stores
    n.madd(
        "Store",
        nodes + " iron ore store",
        bus=nodes + " iron ore",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="iron ore store",
    )

    # add iron ore generator
    n.madd(
        "Generator",
        nodes + " iron ore",
        bus=nodes + " iron ore",
        p_nom_extendable=True,
        carrier="iron ore",
    )
    logger.info("Added iron ore carrier, buses, stores and generators")

    # add DRI carrier
    n.add("Carrier", "DRI")

    # add DRI bus
    n.madd(
        "Bus",
        nodes + " DRI",
        location=nodes,
        carrier="DRI",
    )
    logger.info("Added DRI carrier and buses")

    # add DRI load
    p_set = industrial_demand.loc[nodes, "DRI + Electric arc"].rename(index=lambda x: x + " DRI") / nhours
    n.madd(
        "Load",
        nodes + " DRI",
        bus=nodes + " DRI",
        p_set=p_set,
        carrier="DRI",
    )
    logger.info("Added DRI demand to DRI buses")

    # add DRI process to produce DRI/sponge iron from gas and electricity
    # TODO: revise if marginal price is needed: no VOM is available
    # TODO: revise gas-input and electricity-input
    n.madd(
        "Link",
        nodes + " DRI",
        bus0=nodes,
        bus1=nodes + " DRI",
        bus2=nodes + " gas",
        bus3=nodes + " iron ore",
        bus4="co2 atmoshpere",
        p_nom_extendable=True,
        carrier="DRI",
        efficiency=1/costs.at["direct iron reduction furnace", "electricity-input"],
        efficiency2=-costs.at["direct iron reduction furnace", "gas-input"] # TODO: revise gas-input as it is not matching the source
        / costs.at["direct iron reduction furnace", "electricity-input"],
        efficiency3=-costs.at["direct iron reduction furnace", "ore-input"] # TODO: revise electricity-input as it is not matching the source
        / costs.at["direct iron reduction furnace", "electricity-input"],
        efficiency4=costs.at["direct iron reduction furnace", "gas-input"]
        * costs.at["gas", "CO2 intensity"] # TODO: needs CO2 intensity for direct iron reduction furnace or any other method to compute CO2 emissions
        / costs.at["direct iron reduction furnace", "electricity-input"],
        capital_cost=costs.at["direct iron reduction furnace", "fixed"]
        / costs.at["direct iron reduction furnace", "ore-input"],
        lifetime=costs.at["direct iron reduction furnace", "lifetime"],
    )
    logger.info("Added DRI process to produce steel from gas and electricity")

    # TODO: revise implementation of CCS retrofit for DRI
    # CCS retrofit for DRI
    if "steel" in snakemake.params.ccs_retrofit:
        # calculate capital and marginal costs of DRI CC
        # TODO: revise capital and marginal costs of DRI CC
        capital_cost = (
            costs.at["direct iron reduction furnace", "fixed"]
            / costs.at["direct iron reduction furnace", "electricity-input"]
            + costs.at["steel capture retrofit", "fixed"] # TODO :revise: it is in USD/tCO2 (seems marginal cost)
            * costs.at["direct iron reduction furnace", "gas-input"]
            * costs.at["gas", "CO2 intensity"]
            / costs.at["direct iron reduction furnace", "capture_rate"]
        )
        # TODO: no VOM for marginal price of DRI CC

        # add DRI CC
        # iron ore bus is not used, because bus4 is maximum what we have, electricity, gas, co2 are more important, because iron ore is just there.
        n.madd(
            "Link",
            nodes + " DRI CC",
            bus0=nodes,
            bus1=nodes + " DRI",
            bus2=nodes + " gas",
            bus3=nodes + " iron ore",
            bus4="co2 atmoshpere",
            bus5=nodes + " co2 stored",
            p_nom_extendable=True,
            carrier="DRI CC",
            efficiency=1/costs.at["direct iron reduction furnace", "electricity-input"],
            efficiency2=-costs.at["direct iron reduction furnace", "gas-input"]
            / costs.at["direct iron reduction furnace", "electricity-input"],
            efficiency3=-costs.at["direct iron reduction furnace", "ore-input"]
            / costs.at["direct iron reduction furnace", "electricity-input"],
            efficiency4=costs.at["direct iron reduction furnace", "gas-input"]
            * costs.at["gas", "CO2 intensity"] # TODO: needs CO2 intensity for direct iron reduction furnace
            * (1 - costs.at["steel capture retrofit", "capture_rate"])
            / costs.at["direct iron reduction furnace", "electricity-input"],
            efficiency5=costs.at["direct iron reduction furnace", "gas-input"]
            * costs.at["gas", "CO2 intensity"] # TODO: needs CO2 intensity for direct iron reduction furnace
            * costs.at["steel capture retrofit", "capture_rate"]
            / costs.at["direct iron reduction furnace", "electricity-input"],
            capital_cost=capital_cost,
            # marginal_cost=marginal_cost, # TODO: revise VOM
            lifetime=costs.at["steel capture retrofit", "lifetime"],
        )
        logger.info("Added DRI CC to retrofit DRI plants")

    if config["custom_industry"]["H2_DRI"]:
        # add DRI process to produce steel from hydrogen
        # TODO: revise if marginal price is needed: no VOM is available
        n.madd(
            "Link",
            nodes + " DRI H2",
            bus0=nodes,
            bus1=nodes + " DRI",
            bus2=nodes + " H2",
            p_nom_extendable=True,
            carrier="DRI H2",
            efficiency=costs.at["hydrogen direct iron reduction furnace", "electricity-input"],
            efficiency2=-costs.at["hydrogen direct iron reduction furnace", "hydrogen-input"]
            / costs.at["hydrogen direct iron reduction furnace", "electricity-input"],
            capital_cost=costs.at["hydrogen direct iron reduction furnace", "fixed"]
            / costs.at["hydrogen direct iron reduction furnace", "electricity-input"],
            lifetime=costs.at["hydrogen direct iron reduction furnace", "lifetime"],
        )
        logger.info("Added DRI process to produce steel from hydrogen")

    # add steel BF-BOF carrier
    n.add("Carrier", "steel BF-BOF")

    # add steel BF-BOF bus
    n.madd(
        "Bus",
        nodes + " steel BF-BOF",
        location=nodes,
        carrier="steel BF-BOF",
    )
    logger.info("Added steel BF-BOF carrier and buses")

    # add steel BF-BOF demand
    p_set = industrial_demand.loc[nodes, "Integrated steelworks"].rename(index=lambda x: x + " steel BF-BOF") / nhours
    n.madd(
        "Load",
        nodes + " steel BF-BOF",
        bus=nodes + " steel BF-BOF",
        p_set=p_set,
        carrier="steel BF-BOF",
    )
    logger.info("Added steel BF-BOF demand to steel BF-BOF buses")

    # add scrap steel carrier
    n.add("Carrier", "scrap steel")

    # add scrap steel bus
    n.madd(
        "Bus",
        nodes + " scrap steel",
        location=nodes,
        carrier="scrap steel",
    )

    # add scrap steel stores
    n.madd(
        "Store",
        nodes + " scrap steel store",
        bus=nodes + " scrap steel",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="scrap steel store",
    )

    # add scrap steel generator
    n.madd(
        "Generator",
        nodes + " scrap steel",
        bus=nodes + " scrap steel",
        p_nom_extendable=True,
        carrier="scrap steel",
    )
    logger.info("Added scrap steel carrier, buses, stores and generators")

    # add steel BF-BOF process to produce steel from gas and electricity
    # TODO: revise ore-input and scrap-input: ore input seems to small
    # from the source: ore-input = 1.539 t_ore/t_steel, and scrap-input = 0.051 t_scrap/t_steel
    # TODO: revise if marginal price is needed: no VOM is available
    n.madd(
        "Link",
        nodes + " steel BF-BOF",
        bus0=nodes + " coal",
        bus1=nodes + " steel BF-BOF",
        bus2=nodes + " iron ore",
        bus3=nodes + " scrap steel",
        bus4="co2 atmoshpere",
        p_nom_extendable=True,
        carrier="BF-BOF",
        efficiency=1/costs.at["blast furnace-basic oxygen furnace", "coal-input"],
        efficiency2=-costs.at["blast furnace-basic oxygen furnace", "ore-input"]
        / costs.at["blast furnace-basic oxygen furnace", "coal-input"], # TODO: ore-input needs revision
        efficiency3=-costs.at["blast furnace-basic oxygen furnace", "scrap-input"]
        / costs.at["blast furnace-basic oxygen furnace", "coal-input"],
        efficiency4=costs.at["coal", "CO2 intensity"], # TODO: needs CO2 intensity for blast furnace-basic oxygen furnace
        capital_cost=costs.at["blast furnace-basic oxygen furnace", "fixed"]
        / costs.at["blast furnace-basic oxygen furnace", "coal-input"],
        lifetime=costs.at["blast furnace-basic oxygen furnace", "lifetime"],
    )
    logger.info("Added steel BF-BOF process to produce steel from gas and electricity")

    if "steel" in snakemake.params.ccs_retrofit:
        # calculate capital and marginal costs of BF-BOF CC
        capital_cost = (
            costs.at["blast furnace-basic oxygen furnace", "fixed"]
            / costs.at["blast furnace-basic oxygen furnace", "coal-input"]
            + costs.at["steel capture retrofit", "fixed"] # TODO: revise: it is in USD/tCO2 (seems marginal cost), to use it we need to know how much it emits
            * costs.at["coal", "CO2 intensity"] # TODO: needs CO2 intensity for blast furnace-basic oxygen furnace
            / costs.at["blast furnace-basic oxygen furnace", "capture_rate"]
        )
        # no VOM is available for BF-BOF CC

        # add BF-BOF CC
        n.madd(
            "Link",
            nodes + " BF-BOF CC",
            bus0=nodes + " coal",
            bus1=nodes + " steel BF-BOF",
            bus2=nodes + " iron ore",
            bus3=nodes + " scrap steel",
            bus4="co2 atmoshpere",
            bus5=nodes + " co2 stored",
            p_nom_extendable=True,
            carrier="BF-BOF CC",
            efficiency=1/costs.at["blast furnace-basic oxygen furnace", "coal-input"],
            efficiency2=-costs.at["blast furnace-basic oxygen furnace", "ore-input"]
            / costs.at["blast furnace-basic oxygen furnace", "coal-input"],
            efficiency3=-costs.at["blast furnace-basic oxygen furnace", "scrap-input"]
            / costs.at["blast furnace-basic oxygen furnace", "coal-input"],
            efficiency4=costs.at["coal", "CO2 intensity"] # TODO: needs CO2 intensity for blast furnace-basic oxygen furnace
            * (1 - costs.at["steel capture retrofit", "capture_rate"]),
            efficiency5=costs.at["coal", "CO2 intensity"] # TODO: needs CO2 intensity for blast furnace-basic oxygen furnace
            * costs.at["steel capture retrofit", "capture_rate"],
            capital_cost=capital_cost,
            lifetime=costs.at["steel capture retrofit", "lifetime"],
        )
        logger.info("Added BF-BOF CC to retrofit BF-BOF plants")

    # add steel EAF carrier
    n.add("Carrier", "steel EAF")

    # add steel EAF bus
    n.madd(
        "Bus",
        nodes + " steel EAF",
        location=nodes,
        carrier="steel EAF",
    )
    logger.info("Added steel EAF carrier and buses")

    # add steel EAF demand
    p_set = industrial_demand.loc[nodes, "Electric arc"].rename(index=lambda x: x + " steel EAF") / nhours
    n.madd(
        "Load",
        nodes + " steel EAF",
        bus=nodes + " steel EAF",
        p_set=p_set,
        carrier="steel EAF",
    )
    logger.info("Added steel EAF demand to steel EAF buses")

    # add steel EAF process to produce steel from gas and electricity
    # TODO: revise marginal price: no VOM is available
    n.madd(
        "Link",
        nodes + " steel EAF",
        bus0=nodes,
        bus1=nodes + " steel EAF",
        bus2=nodes + " scrap",
        p_nom_extendable=True,
        carrier="EAF",
        efficiency=1/costs.at["electric arc furnace", "electricity-input"],
        efficiency2=-costs.at["electric arc furnace", "scrap-input"]
        / costs.at["electric arc furnace", "electricity-input"],
        capital_cost=costs.at["electric arc furnace", "fixed"]
        / costs.at["electric arc furnace", "electricity-input"],
        lifetime=costs.at["electric arc furnace", "lifetime"],
    )
    logger.info("Added steel EAF process to produce steel from gas and electricity")


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
    if snakemake.params.add_ammonia:
        add_ammonia(n)

    # add ethanol industry
    if snakemake.params.add_ethanol:
        add_ethanol(n)

    # add steel industry
    if snakemake.params.add_steel:
        add_steel(n)

    # save the modified network
    n.export_to_netcdf(snakemake.output.modified_network)
