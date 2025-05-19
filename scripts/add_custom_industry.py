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

    # add ammonia demand in MWh
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

        # compute capital costs of SMR CC
        capital_cost = (
            costs.at["SMR", "fixed"]
            + costs.at["ammonia carbon capture retrofit", "fixed"]
            * costs.at["gas", "CO2 intensity"]
            * costs.at["ammonia carbon capture retrofit", "capture_rate"]
        )

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
            * (1 - costs.at["ammonia carbon capture retrofit", "capture_rate"]),
            efficiency3=costs.at["gas", "CO2 intensity"]
            * costs.at["ammonia carbon capture retrofit", "capture_rate"],
            efficiency4=-costs.at["ammonia carbon capture retrofit", "electricity-input"]
            * costs.at["gas", "CO2 intensity"]
            * costs.at["ammonia carbon capture retrofit", "capture_rate"],
            capital_cost=capital_cost,
            lifetime=costs.at["ammonia carbon capture retrofit", "lifetime"],
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
        * costs.at["ethanol from starch crop", "efficiency"],
        marginal_cost=costs.at["ethanol from starch crop", "VOM"],
        lifetime=costs.at["ethanol from starch crop", "lifetime"],
    )
    logger.info("Added links to model starch-based ethanol plants")

    # add ethanol demand in MWh
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
        # calculate capital cost of ethanol from starch CC
        capital_cost = (
            costs.at["ethanol from starch crop", "fixed"]
            * costs.at["ethanol from starch crop", "efficiency"]
            + costs.at["ethanol carbon capture retrofit", "fixed"]
            * costs.at["bioethanol crops", "CO2 intensity"]
            * costs.at["ethanol carbon capture retrofit", "capture_rate"]
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
            * costs.at["ethanol carbon capture retrofit", "capture_rate"],
            efficiency3=-costs.at["ethanol carbon capture retrofit", "electricity-input"]
            * costs.at["bioethanol crops", "CO2 intensity"]
            * costs.at["ethanol carbon capture retrofit", "capture_rate"],
            capital_cost=capital_cost,
            marginal_cost=costs.at["ethanol from starch crop", "VOM"],
            lifetime=costs.at["ethanol carbon capture retrofit", "lifetime"],
        )
        logger.info("Added ethanol from starch CC plants")


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

    # add DRI load in ton
    p_set = industrial_demand.loc[nodes, "DRI + Electric arc"].rename(index=lambda x: x + " DRI") * 1e3 / nhours
    n.madd(
        "Load",
        nodes + " DRI",
        bus=nodes + " DRI",
        p_set=p_set,
        carrier="DRI",
    )
    logger.info("Added DRI demand to DRI buses")

    # add DRI process to produce DRI/sponge iron from gas
    n.madd(
        "Link",
        nodes + " DRI",
        bus0=nodes + " gas",
        bus1=nodes + " DRI",
        bus2=nodes + " iron ore",
        bus3="co2 atmoshpere",
        p_nom_extendable=True,
        carrier="DRI",
        efficiency=1/costs.at["natural gas direct iron reduction furnace", "gas-input"],
        efficiency2=-costs.at["natural gas direct iron reduction furnace", "ore-input"]
        / costs.at["natural gas direct iron reduction furnace", "gas-input"],
        efficiency3=costs.at["gas", "CO2 intensity"],
        capital_cost=costs.at["natural gas direct iron reduction furnace", "fixed"]
        / costs.at["natural gas direct iron reduction furnace", "gas-input"],
        lifetime=costs.at["natural gas direct iron reduction furnace", "lifetime"],
    )
    logger.info("Added DRI process to produce steel from gas and electricity")

    # CCS retrofit for DRI
    if "steel" in snakemake.params.ccs_retrofit:
        # calculate capital cost of DRI CC
        capital_cost = (
            costs.at["natural gas direct iron reduction furnace", "fixed"]
            / costs.at["natural gas direct iron reduction furnace", "gas-input"]
            + costs.at["steel carbon capture retrofit", "fixed"]
            * costs.at["gas", "CO2 intensity"]
            * costs.at["steel carbon capture retrofit", "capture_rate"]
        )

        # add DRI CC
        n.madd(
            "Link",
            nodes + " DRI CC",
            bus0=nodes + " gas",
            bus1=nodes + " DRI",
            bus2=nodes + " iron ore",
            bus3="co2 atmoshpere",
            bus4=nodes + " co2 stored",
            bus5=nodes,
            p_nom_extendable=True,
            carrier="DRI CC",
            efficiency=1/costs.at["natural gas direct iron reduction furnace", "gas-input"],
            efficiency2=-costs.at["natural gas direct iron reduction furnace", "ore-input"]
            / costs.at["natural gas direct iron reduction furnace", "gas-input"],
            efficiency3=costs.at["gas", "CO2 intensity"]
            * (1 - costs.at["steel carbon capture retrofit", "capture_rate"]),
            efficiency4=costs.at["gas", "CO2 intensity"]
            * costs.at["steel carbon capture retrofit", "capture_rate"],
            efficiency5=-costs.at["steel carbon capture retrofit", "electricity-input"]
            * costs.at["gas", "CO2 intensity"]
            * costs.at["steel carbon capture retrofit", "capture_rate"],
            capital_cost=capital_cost,
            lifetime=costs.at["steel carbon capture retrofit", "lifetime"],
        )
        logger.info("Added DRI CC to retrofit DRI plants")

    if config["custom_industry"]["H2_DRI"]:
        # add DRI process to produce steel from hydrogen
        n.madd(
            "Link",
            nodes + " DRI H2",
            bus0=nodes + " H2",
            bus1=nodes + " DRI",
            bus2=nodes,
            p_nom_extendable=True,
            carrier="DRI H2",
            efficiency=1/costs.at["hydrogen natural gas direct iron reduction furnace", "hydrogen-input"],
            efficiency2=-costs.at["hydrogen natural gas direct iron reduction furnace", "electricity-input"]
            / costs.at["hydrogen natural gas direct iron reduction furnace", "hydrogen-input"],
            capital_cost=costs.at["hydrogen natural gas direct iron reduction furnace", "fixed"]
            / costs.at["hydrogen natural gas direct iron reduction furnace", "hydrogen-input"],
            lifetime=costs.at["hydrogen natural gas direct iron reduction furnace", "lifetime"],
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

    # add steel BF-BOF demand in ton
    p_set = industrial_demand.loc[nodes, "Integrated steelworks"].rename(index=lambda x: x + " steel BF-BOF") * 1e3 / nhours
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

    # add steel BF-BOF process to produce steel from coal, iron ore and scrap steel
    n.madd(
        "Link",
        nodes + " BF-BOF",
        bus0=nodes + " coal",
        bus1=nodes + " steel BF-BOF",
        bus2=nodes + " iron ore",
        bus3=nodes + " scrap steel",
        bus4="co2 atmoshpere",
        p_nom_extendable=True,
        carrier="BF-BOF",
        efficiency=1/costs.at["blast furnace-basic oxygen furnace", "coal-input"],
        efficiency2=-costs.at["blast furnace-basic oxygen furnace", "ore-input"]
        / costs.at["blast furnace-basic oxygen furnace", "coal-input"],
        efficiency3=-costs.at["blast furnace-basic oxygen furnace", "scrap-input"]
        / costs.at["blast furnace-basic oxygen furnace", "coal-input"],
        efficiency4=costs.at["coal", "CO2 intensity"],
        capital_cost=costs.at["blast furnace-basic oxygen furnace", "fixed"]
        / costs.at["blast furnace-basic oxygen furnace", "coal-input"],
        lifetime=costs.at["blast furnace-basic oxygen furnace", "lifetime"],
    )
    logger.info("Added steel BF-BOF process to produce steel from gas and electricity")

    if "steel" in snakemake.params.ccs_retrofit:
        # calculate capital cost of BF-BOF CC
        capital_cost = (
            costs.at["blast furnace-basic oxygen furnace", "fixed"]
            / costs.at["blast furnace-basic oxygen furnace", "coal-input"]
            + costs.at["steel carbon capture retrofit", "fixed"]
            * costs.at["coal", "CO2 intensity"]
            * costs.at["steel carbon capture retrofit", "capture_rate"]
        )

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
            efficiency4=costs.at["coal", "CO2 intensity"]
            * (1 - costs.at["steel carbon capture retrofit", "capture_rate"]),
            efficiency5=costs.at["coal", "CO2 intensity"]
            * costs.at["steel carbon capture retrofit", "capture_rate"],
            capital_cost=capital_cost,
            lifetime=costs.at["steel carbon capture retrofit", "lifetime"],
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

    # add steel EAF demand in ton
    p_set = industrial_demand.loc[nodes, "Electric arc"].rename(index=lambda x: x + " steel EAF") * 1e3 / nhours
    n.madd(
        "Load",
        nodes + " steel EAF",
        bus=nodes + " steel EAF",
        p_set=p_set,
        carrier="steel EAF",
    )
    logger.info("Added steel EAF demand to steel EAF buses")

    # add steel EAF process to produce steel from electricity and scrap steel
    n.madd(
        "Link",
        nodes + " steel EAF",
        bus0=nodes,
        bus1=nodes + " steel EAF",
        bus2=nodes + " scrap",
        p_nom_extendable=True,
        carrier="EAF",
        efficiency=1/costs.at["electric arc furnace with hbi and scrap", "electricity-input"],
        efficiency2=-costs.at["electric arc furnace with hbi and scrap", "scrap-input"]
        / costs.at["electric arc furnace with hbi and scrap", "electricity-input"],
        capital_cost=costs.at["electric arc furnace with hbi and scrap", "fixed"]
        / costs.at["electric arc furnace with hbi and scrap", "electricity-input"],
        lifetime=costs.at["electric arc furnace with hbi and scrap", "lifetime"],
    )
    logger.info("Added steel EAF process to produce steel from gas and electricity")


def add_cement(n):
    """
        Adds cement buses and stores, and adds links to produce cement
    """
    # add cement dry clinker carrier
    n.add("Carrier", "clinker")

    # add clinker bus
    n.madd(
        "Bus",
        nodes + " clinker",
        location=nodes,
        carrier="clinker",
    )

    # add clinker stores
    n.madd(
        "Store",
        nodes + " clinker store",
        bus=nodes + " clinker",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="clinker store",
    )
    logger.info("Added cement carrier, buses, and stores")

    # add cement carrier
    n.add("Carrier", "cement")

    # add cement bus
    n.madd(
        "Bus",
        nodes + " cement",
        location=nodes,
        carrier="cement",
    )
    logger.info("Added cement carrier and buses")

    # add cement load in ton
    p_set = industrial_demand.loc[nodes, "Cement"].rename(index=lambda x: x + " cement") * 1e3 / nhours
    n.madd(
        "Load",
        nodes + " cement",
        bus=nodes + " cement",
        p_set=p_set,
        carrier="cement",
    )
    logger.info("Added cement demand to cement buses")

    # add cement dry clinker techonology
    n.madd(
        "Link",
        nodes + " dry clinker",
        bus0=nodes + " gas",
        bus1=nodes + " clinker",
        bus2=nodes,
        bus3="co2 atmoshpere",
        p_nom_extendable=True,
        carrier="dry clinker",
        efficiency=1/costs.at["cement dry clinker", "gas-input"],
        efficiency2=-costs.at["cement dry clinker", "electricity-input"]
        / costs.at["cement dry clinker", "gas-input"],
        efficiency3=costs.at["gas", "CO2 intensity"],
        capital_cost=costs.at["cement dry clinker", "fixed"]
        / costs.at["cement dry clinker", "gas-input"],
        marginal_cost=costs.at["cement dry clinker", "VOM"]
        / costs.at["cement dry clinker", "gas-input"],
        lifetime=costs.at["cement dry clinker", "lifetime"],
    )
    logger.info("Added dry clinker process to produce clinker from gas and electricity")

    # add cement finishing technology
    n.madd(
        "Link",
        nodes + " cement finishing",
        bus0=nodes + " clinker",
        bus1=nodes + " cement",
        bus2=nodes,
        p_nom_extendable=True,
        carrier="cement finishing",
        efficiency=1/costs.at["cement finishing", "clinker-input"],
        efficiency2=-costs.at["cement finishing", "electricity-input"]
        / costs.at["cement finishing", "clinker-input"],
        capital_cost=costs.at["cement finishing", "fixed"]
        / costs.at["cement finishing", "clinker-input"],
        marginal_cost=costs.at["cement finishing", "VOM"]
        / costs.at["cement finishing", "clinker-input"],
        lifetime=costs.at["cement finishing", "lifetime"],
    )
    logger.info("Added cement finishing process to produce cement from clinker and electricity")

    # add cement dry clinker CC
    if "cement" in snakemake.params.ccs_retrofit:
        # calculate capital and marginal costs of cement dry clinker CC
        capital_cost = (
            costs.at["cement dry clinker", "fixed"]
            / costs.at["cement dry clinker", "gas-input"]
            + costs.at["cement carbon capture retrofit", "fixed"]
            * costs.at["gas", "CO2 intensity"]
            * costs.at["cement carbon capture retrofit", "capture_rate"]
        )
        marginal_cost = (
            costs.at["cement dry clinker", "VOM"]
            / costs.at["cement dry clinker", "gas-input"]
        )
        # add cement dry clinker CC
        n.madd(
            "Link",
            nodes + " dry clinker CC",
            bus0=nodes + " gas",
            bus1=nodes + " clinker",
            bus2=nodes,
            bus3="co2 atmoshpere",
            bus4=nodes + " co2 stored",
            p_nom_extendable=True,
            carrier="dry clinker CC",
            efficiency=1/costs.at["cement dry clinker", "gas-input"],
            efficiency2=-costs.at["cement dry clinker", "electricity-input"]
            / costs.at["cement dry clinker", "gas-input"],
            efficiency3=costs.at["gas", "CO2 intensity"]
            * (1 - costs.at["cement carbon capture retrofit", "capture_rate"]),
            efficiency4=costs.at["gas", "CO2 intensity"]
            * costs.at["cement carbon capture retrofit", "capture_rate"],
            capital_cost=capital_cost,
            marginal_cost=marginal_cost,
            lifetime=costs.at["cement carbon capture retrofit", "lifetime"],
        )
        logger.info("Added cement dry clinker CC to retrofit cement dry clinker plants")


def extend_links(n, level):
    """
        Replace NaN for bus and efficiency of the links for selected level
    """
    # find links with efficiency of NaN for given level
    nan_techs = n.links[n.links[f"efficiency{level}"].isna()].index
    n.links.loc[nan_techs, f'bus{level}'] = ""
    n.links.loc[nan_techs, f'efficiency{level}'] = 1.0
    logger.info(f"Fill bus{level} and efficiency{level} with default values")


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

    # add cement industry
    if snakemake.params.add_cement:
        add_cement(n)

    # fill efficiency5 and bus5 for missing links if exists
    if "efficiency5" in n.links.columns:
        extend_links(n, level=5)

    # save the modified network
    n.export_to_netcdf(snakemake.output.modified_network)
