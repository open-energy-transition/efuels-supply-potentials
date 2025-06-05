# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../submodules/pypsa-earth/scripts/")))
import pandas as pd
import numpy as np
import country_converter as coco
import powerplantmatching as pm
import pypsa
import xarray as xr
from types import SimpleNamespace
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, \
                            configure_logging, load_network
from _helpers import prepare_costs
from prepare_sector_network import define_spatial


logger = create_logger(__name__)
cc = coco.CountryConverter()
idx = pd.IndexSlice
spatial = SimpleNamespace()


def add_build_year_to_new_assets(n, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    baseyear : int
        year in which optimized assets are built
    """
    # Set build_year for new assets (p_nom == 0)
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        # carriers that already exists (p_nom > 0)
        existing_carriers = c.df[c.df[attr] > 0].carrier.unique()
        # new assets with no build_year and not built before baseyear
        new_assets = c.df.index[~(c.df.carrier.isin(existing_carriers)) & (c.df.build_year == 0)]
        c.df.loc[new_assets, "build_year"] = baseyear

        # add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[new_assets] += f"-{str(baseyear)}"
        c.df.rename(index=rename, inplace=True)

        # rename time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr] = c.pnl[attr].rename(columns=rename)

        logger.info(f"Added build_year {baseyear} to new assets in {c.name} for {len(new_assets)} assets.")


def remove_powerplants(n):
    """
    Remove capacities for non-extendable conventional and renewable powerplants as 
    it is added later year by year
    """
    carriers_to_remove = {
        "Link": ["CCGT", "OCGT", "coal", "oil", "biomass"],
        "Generator": ["nuclear", "solar", "onwind", "offwind-ac", "geothermal", "ror"],
        "Store": []
    }
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        # assets to remove
        assets_to_remove = c.df[c.df.carrier.isin(carriers_to_remove[c.name])].index
        # remove assets
        n.mremove(c.name, assets_to_remove)

        logger.info(f"Removed {len(assets_to_remove)} assets from {c.name} with carriers {carriers_to_remove[c.name]}.")


def add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    grouping_years :
        intervals to group existing capacities
    costs :
        to read lifetime to estimate YearDecomissioning
    baseyear : int
    """
    logger.info(
        f"Adding power capacities installed before {baseyear} from powerplants.csv"
    )
    # read powerplants.csv
    df_agg = pd.read_csv(snakemake.input.powerplants, index_col=0)

    # drop assets which are already phased out / decommissioned
    phased_out = df_agg[df_agg["DateOut"] < baseyear].index
    df_agg.drop(phased_out, inplace=True)

    # drop hydro storage_units (Reservior and Pumped Storage) and keep only Run-Of-River from hydro
    drop_hydro_storages = df_agg[(df_agg.Fueltype == "hydro") & (df_agg.Technology.isin(["Reservoir", "Pumped Storage"]))].index
    df_agg.drop(drop_hydro_storages, inplace=True)
    df_agg.Fueltype.replace({"hydro": "ror"}, inplace=True)

    # assign clustered bus
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0).squeeze()
    busmap = pd.read_csv(snakemake.input.busmap, index_col=0).squeeze()

    inv_busmap = {}
    for k, v in busmap.items():
        inv_busmap[v] = inv_busmap.get(v, []) + [k]

    clustermaps = busmap_s.map(busmap)
    clustermaps.index = clustermaps.index.astype(int)

    df_agg["cluster_bus"] = df_agg.bus.map(clustermaps)

    df_agg["grouping_year"] = np.take(
        grouping_years, np.digitize(df_agg.DateIn, grouping_years, right=True)
    )

    # calculate (adjusted) remaining lifetime before phase-out (+1 because assuming
    # phase out date at the end of the year)
    df_agg["lifetime"] = df_agg.DateOut - df_agg["grouping_year"] + 1

    df = df_agg.pivot_table(
        index=["grouping_year", "Fueltype"],
        columns="cluster_bus",
        values="Capacity",
        aggfunc="sum",
    )

    lifetime = df_agg.pivot_table(
        index=["grouping_year", "Fueltype"],
        columns="cluster_bus",
        values="lifetime",
        aggfunc="mean",  # currently taken mean for clustering lifetimes
    )

    carrier = {
        "coal": "coal",
        "lignite": "lignite",
        "oil": "oil",
        "biomass": "biomass",
        "OCGT": "gas",
        "CCGT": "gas",
    }

    for grouping_year, generator in df.index:
        # capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[
            capacity > snakemake.params.existing_capacities["threshold_capacity"]
        ]
        suffix = "-ac" if generator == "offwind" else ""
        name_suffix = f" {generator}{suffix}-{grouping_year}"
        asset_i = capacity.index + name_suffix
        if generator in ["solar", "onwind", "offwind", "ror", "geothermal", "nuclear"]:
            # to consider electricity grid connection costs or a split between
            # solar utility and rooftop as well, rather take cost assumptions
            # from existing network than from the cost database
            capital_cost = n.generators.loc[
                n.generators.carrier == generator + suffix, "capital_cost"
            ].mean()
            marginal_cost = n.generators.loc[
                n.generators.carrier == generator + suffix, "marginal_cost"
            ].mean()
            # check if assets are already in network (e.g. for 2020)
            already_build = n.generators.index.intersection(asset_i)
            new_build = asset_i.difference(n.generators.index)
            lifetime_assets = lifetime.loc[grouping_year, generator].dropna()

            # set p_nom_min for already built assets
            if not already_build.empty:
                n.generators.loc[already_build, "p_nom_min"] = capacity.loc[
                    already_build.str.replace(name_suffix, "")
                ].values
            new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

            if "m" in snakemake.wildcards.clusters:
                for ind in new_capacity.index:
                    # existing capacities are split evenly among regions in every country
                    inv_ind = list(inv_busmap[ind])

                    # for offshore the splitting only includes coastal regions
                    inv_ind = [
                        i for i in inv_ind if (i + name_suffix) in n.generators.index
                    ]

                    p_max_pu = n.generators_t.p_max_pu[
                        [i + name_suffix for i in inv_ind]
                    ]
                    p_max_pu.columns = [i + name_suffix for i in inv_ind]

                    n.madd(
                        "Generator",
                        [i + name_suffix for i in inv_ind],
                        bus=ind,
                        carrier=generator,
                        p_nom=new_capacity[ind]
                        / len(inv_ind),  # split among regions in a country
                        marginal_cost=marginal_cost,
                        capital_cost=capital_cost,
                        efficiency=costs.at[generator, "efficiency"],
                        p_max_pu=p_max_pu,
                        build_year=grouping_year,
                        lifetime=lifetime_assets.values,
                    )

            else:
                # obtain p_max_pu from existing network
                if generator in ["ror", "nuclear"]:
                    # get static p_max_pu for ror
                    p_max_pu = n.generators.loc[
                        capacity.index + f" {generator}{suffix}"
                    ].p_max_pu.values
                else:
                    try:
                        # try to get p_max_pu from the baseyear
                        p_max_pu = n.generators_t.p_max_pu[
                            capacity.index + f" {generator}{suffix}-{baseyear}"
                        ]
                    except:
                        # get p_max_pu from the existing network
                        p_max_pu = n.generators_t.p_max_pu[
                            capacity.index + f" {generator}{suffix}"
                        ]
                    p_max_pu.rename(columns=n.generators.bus, inplace=True)

                if not new_build.empty:
                    n.madd(
                        "Generator",
                        new_capacity.index,
                        suffix=name_suffix,
                        bus=new_capacity.index,
                        carrier=generator,
                        p_nom=new_capacity,
                        marginal_cost=marginal_cost,
                        capital_cost=capital_cost,
                        efficiency=costs.at[generator, "efficiency"],
                        p_max_pu=p_max_pu,
                        build_year=grouping_year,
                        lifetime=lifetime_assets.values,
                    )
                    

        else:
            # add capacities for conventional powerplants in links
            if carrier[generator] not in vars(spatial).keys():
                logger.debug(f"Carrier type {generator} not in spatial data, skipping")
                continue
            print(f"Adding {generator} capacities for {grouping_year}")
            
            # get bus0 for the link
            bus0 = vars(spatial)[carrier[generator]].nodes
            if "Earth" not in vars(spatial)[carrier[generator]].locations:
                # select only buses for which capacity is given
                if generator == "biomass":
                    capacity_index = capacity.index + " solid " + carrier[generator]
                else:
                    capacity_index = capacity.index + " " + carrier[generator]
                bus0 = bus0.intersection(capacity_index)

            # check for missing bus
            missing_bus = pd.Index(bus0).difference(n.buses.index)
            if not missing_bus.empty:
                logger.info(f"add buses {bus0}")
                n.madd(
                    "Bus",
                    bus0,
                    carrier=generator,
                    location=vars(spatial)[carrier[generator]].locations,
                    unit="MWh_el",
                )

            # check if assets are already in network
            already_build = n.links.index.intersection(asset_i)
            new_build = asset_i.difference(n.links.index)
            lifetime_assets = lifetime.loc[grouping_year, generator].dropna()

            # set p_nom_min for already built assets
            if not already_build.empty:
                n.links.loc[already_build, "p_nom_min"] = capacity.loc[
                    already_build.str.replace(name_suffix, "")
                ].values

            # add new assets if they are not already in the network
            if not new_build.empty:
                new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

                if generator != "urban central solid biomass CHP":
                    n.madd(
                        "Link",
                        new_capacity.index,
                        suffix=name_suffix,
                        bus0=bus0,
                        bus1=new_capacity.index,
                        bus2="co2 atmosphere",
                        carrier=generator,
                        marginal_cost=costs.at[generator, "efficiency"]
                        * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                        capital_cost=costs.at[generator, "efficiency"]
                        * costs.at[generator, "fixed"],  # NB: fixed cost is per MWel
                        p_nom=new_capacity / costs.at[generator, "efficiency"],
                        efficiency=costs.at[generator, "efficiency"],
                        efficiency2=costs.at[carrier[generator], "CO2 intensity"],
                        build_year=grouping_year,
                        lifetime=lifetime_assets.loc[new_capacity.index],
                    )
                else:
                    key = "central solid biomass CHP"
                    n.madd(
                        "Link",
                        new_capacity.index,
                        suffix=name_suffix,
                        bus0=spatial.biomass.df.loc[new_capacity.index]["nodes"].values,
                        bus1=new_capacity.index,
                        bus2=new_capacity.index + " urban central heat",
                        carrier=generator,
                        p_nom=new_capacity / costs.at[key, "efficiency"],
                        capital_cost=costs.at[key, "fixed"]
                        * costs.at[key, "efficiency"],
                        marginal_cost=costs.at[key, "VOM"],
                        efficiency=costs.at[key, "efficiency"],
                        build_year=grouping_year,
                        efficiency2=costs.at[key, "efficiency-heat"],
                        lifetime=lifetime_assets.loc[new_capacity.index],
                    )
        # check if existing capacities are larger than technical potential
        existing_large = n.generators[
            n.generators["p_nom_min"] > n.generators["p_nom_max"]
        ].index
        if len(existing_large):
            logger.warning(
                f"Existing capacities larger than technical potential for {existing_large},\
                           adjust technical potential to existing capacities"
            )
            n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
                existing_large, "p_nom_min"
            ]


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_custom_existing_baseyear",
            simpl="",
            clusters="10",
            ll="copt",
            opts="24H",
            planning_horizons="2030",
            sopts="24H",
            discountrate=0.071,
            demand="AB",
            h2export="10",
            configfile="configs/scenarios/config.myopic.yaml"
        )

    configure_logging(snakemake)

    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load sector and baseyear params
    options = snakemake.params.sector
    baseyear = snakemake.params.baseyear

    # load the network for the base year
    n = load_network(snakemake.input.network)

    # add build_year to new assets
    add_build_year_to_new_assets(n, baseyear)

    # remove capacities for conventional and renewable powerplants as it is added later
    # remove_powerplants(n)

    # read costs assumptions
    Nyears = n.snapshot_weightings.generators.sum() / 8760.0
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs["USD2013_to_EUR2013"],
        snakemake.params.costs["fill_values"],
        Nyears,
    )

    # define grouping years for existing capacities
    grouping_years_power = snakemake.params.existing_capacities["grouping_years_power"]

    # define spatial resolution of carriers
    spatial = define_spatial(n.buses[n.buses.carrier == "AC"].index, options)

    # add power capacities installed before baseyear
    add_power_capacities_installed_before_baseyear(n, grouping_years_power, costs, baseyear)

    # define spatial resolution of carriers
    #spatial = define_spatial(n.buses[n.buses.carrier == "AC"].index, options)