# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Solves linear optimal power flow for a network iteratively while updating
reactances.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: /img/results.png
        :width: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning)
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the ``pyomo=False`` setting in the :func:`network.lopf` and  :func:`pypsa.linopf.ilopf` function.
Additionally, some extra constraints specified in :mod:`prepare_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances on values of corresponding flows.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors made through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
"""
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../submodules/pypsa-earth/scripts/")))
from scripts._helper import configure_logging, create_logger, mock_snakemake, update_config_from_wildcards
from _helpers import override_component_attrs
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.linopf import (
    define_constraints,
    define_variables,
    get_var,
    ilopf,
    join_exprs,
    linexpr,
    network_lopf,
)
from pypsa.linopt import define_constraints, get_var, join_exprs, linexpr
import geopandas as gpd

logger = create_logger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def prepare_network(n, solve_opts):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if "lv_limit" in n.global_constraints.index:
        n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
        n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load")
        n.madd(
            "Generator",
            n.buses.index,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            # intersect between macroeconomic and surveybased
            # willingness to pay
            # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    # if snakemake.config["foresight"] == "myopic":
    #     add_land_use_constraint(n)

    return n

def apply_tax_credits_to_network(network, ptc_path, itc_path, planning_horizon, costs, log_path=None, verbose=False):
    """
    Apply production and investment tax credits to the network.

    - PTC (Production Tax Credit) reduces marginal_cost for eligible generators and links.
    - ITC (Investment Tax Credit) reduces capital_cost for eligible storage units (batteries).

    Parameters:
        network: PyPSA network object
        ptc_path: Path to CSV file with PTC (columns: carrier, credit)
        itc_path: Path to CSV file with ITC (columns: carrier, credit)
        planning_horizon: Current planning year (int)
        costs: DataFrame containing the full cost structure, including capital_cost
        log_path: Optional path to save a log of applied modifications
        verbose: If True, print detailed logging of applied credits
    """

    modifications = []

    # Load PTC file
    ptc_df = pd.read_csv(ptc_path)
    ptc_credits = dict(zip(ptc_df["carrier"], ptc_df["credit"]))

    biomass_aliases = {
        "biomass",
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC"
    }

    carbon_capture_carriers = {
        "ethanol carbon capture retrofit",
        "ammonia carbon capture retrofit",
        "steel carbon capture retrofit",
        "cement carbon capture retrofit",
        "direct air capture"
    }

    electrolyzer_carriers = {
        "Alkaline electrolyzer large size",
        "PEM electrolyzer small size",
        "SOEC"
    }

    # Apply Production Tax Credits to GENERATORS
    for name, gen in network.generators.iterrows():
        carrier = gen.carrier
        build_year = gen.build_year
        base_cost = gen["_marginal_cost_original"]

        carrier_key = carrier
        if carrier_key not in ptc_credits:
            continue

        credit = ptc_credits[carrier_key]
        apply, scale = False, 1.0

        # allow nuclear to pass even if not in ptc_credits
        if carrier_key != "nuclear" and carrier_key not in ptc_credits:
            continue

        if carrier_key == "nuclear":
            if build_year <= 2024 and 2024 <= planning_horizon <= 2032:
                credit = ptc_credits.get("nuclear_existing", 0.0)
                apply = True
            elif 2030 <= build_year <= 2033 and planning_horizon <= build_year + 10:
                credit = ptc_credits.get("nuclear_new", 0.0)
                apply = True
#        elif carrier_key in {"solar", "onwind", "offwind-ac", "offwind-dc"}:
#            if planning_horizon <= build_year + 10 and build_year <= 2027:
#                credit = ptc_credits.get(carrier_key, 0.0)
#                apply = True
        elif carrier_key == "geothermal":
            if planning_horizon <= build_year + 10:
                if 2030 <= build_year <= 2033:
                    credit = ptc_credits.get(carrier_key, 0.0)
                    apply = True
                elif build_year == 2034:
                    credit = ptc_credits.get(carrier_key, 0.0)
                    apply, scale = True, 0.75
                elif build_year == 2035:
                    credit = ptc_credits.get(carrier_key, 0.0)
                    apply, scale = True, 0.5
        elif carrier_key in carbon_capture_carriers:
            if 2030 <= build_year <= 2033 and planning_horizon <= build_year + 12:
                credit = ptc_credits.get(carrier_key, 0.0)
                apply = True
#        elif carrier_key in electrolyzer_carriers:
#            if build_year <= 2027 and planning_horizon <= build_year + 10:
#                credit = ptc_credits.get(carrier_key, 0.0)
#                apply = True

        if apply:
            new_cost = base_cost + scale * credit
            network.generators.at[name, "marginal_cost"] = new_cost
            modifications.append({
                "component": "generator", "name": name,
                "carrier": carrier, "build_year": build_year,
                "original": base_cost, "credit": scale * credit, "final": new_cost
            })
            if verbose:
                logger.info(f"[PTC GEN] {name} | +{scale * credit:.2f}")

    # --- Apply PTC to LINKS (biomass) ---
    for name, link in network.links.iterrows():
        carrier_key = "biomass" if link.carrier in biomass_aliases else link.carrier
        if carrier_key != "biomass" or "biomass" not in ptc_credits:
            continue

        build_year = getattr(link, "build_year", planning_horizon)
        base_cost = link["_marginal_cost_original"]

        if planning_horizon <= build_year + 10:
            scale = 0.0
            if 2030 <= build_year <= 2033:
                scale = 1.0
            elif build_year == 2034:
                scale = 0.75
            elif build_year == 2035:
                scale = 0.5

            if scale > 0:
                credit = ptc_credits["biomass"]
                new_cost = base_cost + scale * credit
                network.links.at[name, "marginal_cost"] = new_cost
                modifications.append({
                    "component": "link", "name": name,
                    "carrier": link.carrier, "build_year": build_year,
                    "original": base_cost, "credit": scale * credit, "final": new_cost
                })
                if verbose:
                    logger.info(f"[PTC LINK] {name} | +{scale * credit:.2f}")

    # Apply Investment Tax Credits to STORAGE UNITS (batteries)
    if 2030 <= planning_horizon <= 2035 and os.path.exists(itc_path):
        itc_df = pd.read_csv(itc_path, index_col=0)

        for carrier, row in itc_df.iterrows():
            # Interpret credit as negative percent (e.g., -30 means 30% reduction)
            credit_factor = -row.get("credit", 0.0) / 100

            if carrier not in network.storage_units.carrier.values:
                continue

            affected = network.storage_units.query("carrier == @carrier")
            for idx, su in affected.iterrows():
                build_year = su.get("build_year", planning_horizon)

                # Determine scale based on build_year
                scale = 0.0
                if 2030 <= build_year <= 2033:
                    scale = 1.0
                elif build_year == 2034:
                    scale = 0.75
                elif build_year == 2035:
                    scale = 0.5

                if scale > 0:
                    orig = su.capital_cost
                    new = orig * (1 - scale * credit_factor)
                    network.storage_units.at[idx, "capital_cost"] = new
                    modifications.append({
                        "component": "storage_unit", "name": idx,
                        "carrier": carrier, "build_year": build_year,
                        "original": orig, "credit_factor": scale * credit_factor,
                        "final": new
                    })
                    if verbose:
                        logger.info(f"[ITC STORAGE] {idx} | -{scale * credit_factor:.0%} capital_cost")

    # --- Save log of modifications ---
    if modifications and log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        pd.DataFrame(modifications).to_csv(log_path, index=False)

def add_RPS_constraints(network, config_file):

    def process_targets_data(path, carrier, policy):
        df = pd.read_csv(path)
        df.rename(columns={"Unnamed: 0": "state"}, inplace=True)
        df = df.melt(id_vars="state", var_name="year", value_name="target")
        df["carrier"] = ", ".join(carrier)
        df["year"] = df.year.astype(int)
        df["policy"] = policy
        return df

    def attach_state_to_buses(network, path_shapes, distance_crs):
        """
        Attach state to buses
        """
        # Read the shapefile using geopandas
        shapes = gpd.read_file(path_shapes, crs=distance_crs)
        shapes["ISO_1"] = shapes["ISO_1"].apply(lambda x: x.split("-")[1])
        shapes.rename(columns={"ISO_1": "State"}, inplace=True)

        ac_dc_carriers = ["AC", "DC"]
        location_mapping = network.buses.query(
            "carrier in @ac_dc_carriers")[["x", "y"]]

        network.buses["x"] = network.buses["location"].map(
            location_mapping["x"]).fillna(0)
        network.buses["y"] = network.buses["location"].map(
            location_mapping["y"]).fillna(0)

        pypsa_gpd = gpd.GeoDataFrame(
            network.buses,
            geometry=gpd.points_from_xy(network.buses.x, network.buses.y),
            crs=4326
        )

        bus_cols = network.buses.columns
        bus_cols = list(bus_cols) + ["State"]

        st_buses = gpd.sjoin_nearest(shapes, pypsa_gpd, how="right")[bus_cols]

        network.buses["state"] = st_buses["State"]

        return network

    def filter_policy_data(df, coverage, planning_horizon):
        return df[
            (df["year"] == planning_horizon)
            & (df["target"] > 0.0)
            & (df["state"].isin(n.buses[f"{coverage}"].unique()))
        ]

    def add_constraints_to_network(res_generators_eligible, res_storages_eligible, res_links_eligible,
                                   ces_generators_eligible, conventional_links_eligible,
                                   state, policy_data, constraints_type):

        target = policy_data[policy_data.policy ==
                             f"{constraints_type}"]["target"].item()
        target_year = policy_data[policy_data.policy ==
                                  f"{constraints_type}"]["year"].item()

        # remove `low voltage` from bus name to account for solar rooftop (e.g. US0 0 low voltage to US0 0)
        res_generators_eligible["bus"] = res_generators_eligible.bus.str.replace(" low voltage", "", regex=False)
        ces_generators_eligible["bus"] = ces_generators_eligible.bus.str.replace(" low voltage", "", regex=False)

        # get RES generation
        res_generation = (
            linexpr(
                (
                    n.snapshot_weightings.generators,
                    get_var(n, "Generator", "p")[res_generators_eligible.index].T
                )
            )
            .T.groupby(res_generators_eligible.bus, axis=1)
            .apply(join_exprs)
        )

        # hydro dispatch with coefficient of (1 - target)
        hydro_dispatch_with_coefficient = (
                linexpr(
                    (
                        n.snapshot_weightings.stores * (1 - target),
                        get_var(n, "StorageUnit", "p_dispatch")[
                            res_storages_eligible.index].T,
                    )
                )
                .T.groupby(res_storages_eligible.bus, axis=1)
                .apply(join_exprs)
            )

        # if not empty, reindex to generation index and fill NaN with empty string
        if not hydro_dispatch_with_coefficient.empty:
            hydro_dispatch_with_coefficient = (
                hydro_dispatch_with_coefficient
                .reindex(res_generation.index)
                .fillna("")
            )

        # RES dispatch from links with coefficient (biomass)
        if res_links_eligible.empty:
            res_link_dispatch_with_coefficient = pd.Series("", index=res_generation.index)
        else:
            res_link_dispatch_with_coefficient = (
                (
                    linexpr(
                        (
                            (n.snapshot_weightings.stores.apply(
                                lambda r: r * n.links.loc[res_links_eligible.index].efficiency) * (1 - target)).T,
                            get_var(n, "Link", "p")[res_links_eligible.index].T
                        )
                    )
                    .T.groupby(res_links_eligible.bus1, axis=1)
                    .apply(join_exprs)
                )
                .reindex(res_generation.index)
                .fillna("")
            )

        # CES generation multiplied by target
        ces_generation_with_target = (
            linexpr(
                (
                    -n.snapshot_weightings.generators * target,
                    get_var(n, "Generator", "p")[
                        ces_generators_eligible.index].T
                )
            )
            .T.groupby(ces_generators_eligible.bus, axis=1)
            .apply(join_exprs)
        )

        conventional_generation_with_target = (
            (
                linexpr(
                    (
                        (-n.snapshot_weightings.generators.apply(
                            lambda r: r * n.links.loc[conventional_links_eligible.index].efficiency) * target).T,
                        get_var(n, "Link", "p")[conventional_links_eligible.index].T
                    )
                )
                .T.groupby(conventional_links_eligible.bus1, axis=1)
                .apply(join_exprs)
            )
            .reindex(res_generation.index)
            .fillna("")
        )

        lhs = res_generation + hydro_dispatch_with_coefficient + \
            res_link_dispatch_with_coefficient + ces_generation_with_target + \
            conventional_generation_with_target

        # group buses
        if state != "US":
            lhs_grouped = lhs.groupby(n.buses.state).sum()
        else:
            lhs_grouped = lhs.groupby(n.buses.country).sum()

        define_constraints(
            n, lhs_grouped, ">=", 0, f"{constraints_type}_{state}", "rps_limit")
        logger.info(
            f"Added {constraints_type} constraint for {state} in {target_year}.")
        
        
    # define carriers for RES and CES sources
    res_generator_carriers = [
        "solar",
        "onwind",
        "offwind-ac",
        "solar rooftop",
        "offwind-dc",
        "ror",
        "geothermal"
        ]
    res_link_carriers = []
    res_storage_carriers = ["hydro"]
    ces_generator_carriers = res_generator_carriers + ["nuclear"]

    # list of carriers for conventional generation
    conventional_link_carriers = [
        'OCGT',
        'CCGT',
        'oil',
        'coal',
        'lignite',
        'urban central gas CHP',
        'urban central gas CHP CC',
        "biomass",
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC"
    ]

    # read state policies on CES constraints
    ces_data = process_targets_data(
        snakemake.input.ces_path,
        ces_generator_carriers + res_link_carriers,
        "CES",
    )
    # read state policies on RES constraints
    res_data = process_targets_data(
        snakemake.input.res_path,
        res_generator_carriers + res_link_carriers,
        "RES",
    )
    # combine dataframes to loop through states
    policy_data = pd.concat([ces_data, res_data], ignore_index=True)

    # get gadm shape path
    path_shapes = snakemake.input.gadm_shape_path

    # map states to buses
    distance_crs = config_file["crs"]["distance_crs"]
    network = attach_state_to_buses(network, path_shapes, distance_crs)
    planning_horizon = int(snakemake.wildcards.planning_horizons)

    state_policies = config_file["policies"]["state"]
    country_policies = config_file["policies"]["country"]

    if state_policies:

        # select eligible RES/CES policies based on planning horizon, presense of target and state
        state_policy_data = filter_policy_data(policy_data, "state", planning_horizon)

        # get list of states where policies need to be applied
        state_list = state_policy_data.state.unique()

        # define CES and RES constraints
        for state in state_list:
            # get state buses
            region_buses = network.buses[network.buses.state.isin([state])]

            if region_buses.empty:
                continue

            # get region policies
            region_policy = state_policy_data[state_policy_data.state == state]

            # select eligible generators for RES and CES
            region_generators = network.generators[network.generators.bus.isin(
                region_buses.index)]
            res_generators_eligible = region_generators[region_generators.carrier.isin(
                res_generator_carriers)]
            ces_generators_eligible = region_generators[region_generators.carrier.isin(
                ces_generator_carriers)]

            # select eligible links for RES
            region_links = network.links[network.links.bus1.isin(
                region_buses.index)]
            res_links_eligible = region_links[region_links.carrier.isin(
                res_link_carriers)]

            # select eligible storage_units (hydro, not PHS) for RES
            region_storages = network.storage_units[network.storage_units.bus.isin(
                region_buses.index)]
            res_storages_eligible = region_storages[region_storages.carrier.isin(
                res_storage_carriers)]

            # select eligible conventional links
            conventional_links_eligible = region_links[region_links.carrier.isin(
                conventional_link_carriers)]

            # add RES constraint
            if "RES" in region_policy.policy.values and "RES" in state_policies:
                add_constraints_to_network(res_generators_eligible, res_storages_eligible, res_links_eligible,
                                           ces_generators_eligible, conventional_links_eligible,
                                           state, region_policy, "RES")

            # add CES constraint
            if "CES" in region_policy.policy.values and "CES" in state_policies:
                add_constraints_to_network(ces_generators_eligible, res_storages_eligible, res_links_eligible,
                                           ces_generators_eligible, conventional_links_eligible,
                                           state, region_policy, "CES")

    if country_policies:
        country_policy_data = filter_policy_data(policy_data, "country", planning_horizon)
        country_ces_generators = network.generators[network.generators.carrier.isin(
            ces_generator_carriers)]
        country_res_storages = network.storage_units[network.storage_units.carrier.isin(
            res_storage_carriers)]
        country_res_links = network.links[network.links.carrier.isin(
            res_link_carriers)]
        country_conventional_links = network.links[network.links.carrier.isin(
            conventional_link_carriers)]

        if "CES" in country_policy_data.policy.values and "CES" in country_policies:

            add_constraints_to_network(country_ces_generators, country_res_storages, country_res_links,
                                       country_ces_generators, country_conventional_links,
                                       "US", country_policy_data, "CES")


def add_CCL_constraints(n, config):
    agg_p_nom_limits = config["electricity"].get("agg_p_nom_limits")

    try:
        agg_p_nom_minmax = pd.read_csv(
            agg_p_nom_limits, index_col=list(range(2)))
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity limits per country in "
            "config['electricity']['agg_p_nom_limit']."
        )
    logger.info(
        "Adding per carrier generation capacity constraints for " "individual countries"
    )

    gen_country = n.generators.bus.map(n.buses.country)
    # cc means country and carrier
    p_nom_per_cc = (
        pd.DataFrame(
            {
                "p_nom": linexpr((1, get_var(n, "Generator", "p_nom"))),
                "country": gen_country,
                "carrier": n.generators.carrier,
            }
        )
        .dropna(subset=["p_nom"])
        .groupby(["country", "carrier"])
        .p_nom.apply(join_exprs)
    )
    minimum = agg_p_nom_minmax["min"].dropna()
    if not minimum.empty:
        minconstraint = define_constraints(
            n, p_nom_per_cc[minimum.index], ">=", minimum, "agg_p_nom", "min"
        )
    maximum = agg_p_nom_minmax["max"].dropna()
    if not maximum.empty:
        maxconstraint = define_constraints(
            n, p_nom_per_cc[maximum.index], "<=", maximum, "agg_p_nom", "max"
        )


def add_EQ_constraints(n, o, scaling=1e-1):
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    lhs_gen = (
        linexpr(
            (n.snapshot_weightings.generators *
             scaling, get_var(n, "Generator", "p").T)
        )
        .T.groupby(ggrouper, axis=1)
        .apply(join_exprs)
    )
    lhs_spill = (
        linexpr(
            (
                -n.snapshot_weightings.stores * scaling,
                get_var(n, "StorageUnit", "spill").T,
            )
        )
        .T.groupby(sgrouper, axis=1)
        .apply(join_exprs)
    )
    lhs_spill = lhs_spill.reindex(lhs_gen.index).fillna("")
    lhs = lhs_gen + lhs_spill
    define_constraints(n, lhs, ">=", rhs, "equity", "min")


def add_BAU_constraints(n, config):
    ext_c = n.generators.query("p_nom_extendable").carrier.unique()
    mincaps = pd.Series(
        config["electricity"].get("BAU_mincapacities", {
                                  key: 0 for key in ext_c})
    )
    lhs = (
        linexpr((1, get_var(n, "Generator", "p_nom")))
        .groupby(n.generators.carrier)
        .apply(join_exprs)
    )
    define_constraints(
        n, lhs, ">=", mincaps[lhs.index], "Carrier", "bau_mincaps")

    maxcaps = pd.Series(
        config["electricity"].get("BAU_maxcapacities", {
                                  key: np.inf for key in ext_c})
    )
    lhs = (
        linexpr((1, get_var(n, "Generator", "p_nom")))
        .groupby(n.generators.carrier)
        .apply(join_exprs)
    )
    define_constraints(
        n, lhs, "<=", maxcaps[lhs.index], "Carrier", "bau_maxcaps")


def add_SAFE_constraints(n, config):
    peakdemand = (
        1.0 + config["electricity"]["SAFE_reservemargin"]
    ) * n.loads_t.p_set.sum(axis=1).max()
    conv_techs = config["plotting"]["conv_techs"]
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conv_techs"
    ).p_nom.sum()
    ext_gens_i = n.generators.query(
        "carrier in @conv_techs & p_nom_extendable").index
    lhs = linexpr((1, get_var(n, "Generator", "p_nom")[ext_gens_i])).sum()
    rhs = peakdemand - exist_conv_caps
    define_constraints(n, lhs, ">=", rhs, "Safe", "mintotalcap")


def add_operational_reserve_margin_constraint(n, config):
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    reserve = get_var(n, "Generator", "r")
    lhs = linexpr((1, reserve)).sum(1)

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        renewable_capacity_variables = get_var(n, "Generator", "p_nom")[
            vres_i.intersection(ext_i)
        ]
        lhs += linexpr(
            (-EPSILON_VRES * capacity_factor, renewable_capacity_variables)
        ).sum(1)

    # Total demand at t
    demand = n.loads_t.p.sum(1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    define_constraints(n, lhs, ">=", rhs, "Reserve margin")


def update_capacity_constraint(n):
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = get_var(n, "Generator", "p")
    reserve = get_var(n, "Generator", "r")

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = linexpr((1, dispatch), (1, reserve))

    if not ext_i.empty:
        capacity_variable = get_var(n, "Generator", "p_nom")
        lhs += linexpr((-p_max_pu[ext_i], capacity_variable)).reindex(
            columns=gen_i, fill_value=""
        )

    rhs = (p_max_pu[fix_i] *
           capacity_fixed).reindex(columns=gen_i, fill_value=0)

    define_constraints(n, lhs, "<=", rhs, "Generators",
                       "updated_capacity_constraint")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.
    """

    define_variables(n, 0, np.inf, "Generator", "r",
                     axes=[sns, n.generators.index])

    add_operational_reserve_margin_constraint(n, config)

    update_capacity_constraint(n)


def add_battery_constraints(n):
    nodes = n.buses.index[n.buses.carrier == "battery"]
    if nodes.empty or ("Link", "p_nom") not in n.variables.index:
        return
    link_p_nom = get_var(n, "Link", "p_nom")

    chargers_bool = link_p_nom.index.str.contains("battery charger")
    dischargers_bool = link_p_nom.index.str.contains("battery discharger")

    if snakemake.config["foresight"] == "myopic":
        name_suffix = f"-{snakemake.wildcards.planning_horizons}"
    else:
        name_suffix = ""

    lhs = linexpr(
        (1, link_p_nom[chargers_bool]),
        (
            -n.links.loc[
                n.links.index.str.contains(f"battery discharger{name_suffix}"),
                "efficiency",
            ].values,
            link_p_nom[dischargers_bool].values,
        ),
    )
    define_constraints(n, lhs, "=", 0, "Link", "charger_ratio")


def add_RES_constraints(n, res_share):
    lgrouper = n.loads.bus.map(n.buses.country)
    ggrouper = n.generators.bus.map(n.buses.country)
    sgrouper = n.storage_units.bus.map(n.buses.country)
    cgrouper = n.links.bus0.map(n.buses.country)

    logger.warning(
        "The add_RES_constraints functionality is still work in progress. "
        "Unexpected results might be incurred, particularly if "
        "temporal clustering is applied or if an unexpected change of technologies "
        "is subject to the obtimisation."
    )

    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )

    rhs = res_share * load

    res_techs = [
        "solar",
        "onwind",
        "offwind-dc",
        "offwind-ac",
        "battery",
        "hydro",
        "ror",
    ]
    charger = ["H2 electrolysis", "battery charger"]
    discharger = ["H2 fuel cell", "battery discharger"]

    gens_i = n.generators.query("carrier in @res_techs").index
    stores_i = n.storage_units.query("carrier in @res_techs").index
    charger_i = n.links.query("carrier in @charger").index
    discharger_i = n.links.query("carrier in @discharger").index

    # Generators
    lhs_gen = (
        linexpr(
            (n.snapshot_weightings.generators,
             get_var(n, "Generator", "p")[gens_i].T)
        )
        .T.groupby(ggrouper, axis=1)
        .apply(join_exprs)
    )

    # StorageUnits
    lhs_dispatch = (
        (
            linexpr(
                (
                    n.snapshot_weightings.stores,
                    get_var(n, "StorageUnit", "p_dispatch")[stores_i].T,
                )
            )
            .T.groupby(sgrouper, axis=1)
            .apply(join_exprs)
        )
        .reindex(lhs_gen.index)
        .fillna("")
    )

    lhs_store = (
        (
            linexpr(
                (
                    -n.snapshot_weightings.stores,
                    get_var(n, "StorageUnit", "p_store")[stores_i].T,
                )
            )
            .T.groupby(sgrouper, axis=1)
            .apply(join_exprs)
        )
        .reindex(lhs_gen.index)
        .fillna("")
    )

    # Stores (or their resp. Link components)
    # Note that the variables "p0" and "p1" currently do not exist.
    # Thus, p0 and p1 must be derived from "p" (which exists), taking into account the link efficiency.
    lhs_charge = (
        (
            linexpr(
                (
                    -n.snapshot_weightings.stores,
                    get_var(n, "Link", "p")[charger_i].T,
                )
            )
            .T.groupby(cgrouper, axis=1)
            .apply(join_exprs)
        )
        .reindex(lhs_gen.index)
        .fillna("")
    )

    lhs_discharge = (
        (
            linexpr(
                (
                    n.snapshot_weightings.stores.apply(
                        lambda r: r * n.links.loc[discharger_i].efficiency
                    ),
                    get_var(n, "Link", "p")[discharger_i],
                )
            )
            .groupby(cgrouper, axis=1)
            .apply(join_exprs)
        )
        .reindex(lhs_gen.index)
        .fillna("")
    )

    # signs of resp. terms are coded in the linexpr.
    # todo: for links (lhs_charge and lhs_discharge), account for snapshot weightings
    lhs = lhs_gen + lhs_dispatch + lhs_store + lhs_charge + lhs_discharge

    define_constraints(n, lhs, "=", rhs, "RES share")


def add_land_use_constraint(n):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = (
            n.generators.loc[n.generators.carrier == carrier, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier +
                           "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def add_h2_network_cap(n, cap):
    h2_network = n.links.loc[n.links.carrier == "H2 pipeline"]
    if h2_network.index.empty or ("Link", "p_nom") not in n.variables.index:
        return
    h2_network_cap = get_var(n, "Link", "p_nom")
    subset_index = h2_network.index.intersection(h2_network_cap.index)
    lhs = linexpr(
        (h2_network.loc[subset_index, "length"], h2_network_cap[subset_index])
    ).sum()
    # lhs = linexpr((1, h2_network_cap[h2_network.index])).sum()
    rhs = cap * 1000
    define_constraints(n, lhs, "<=", rhs, "h2_network_cap")


def hydrogen_temporal_constraint(n, additionality, time_period):
    """
    Enforces temporal matching constraints for hydrogen production based on renewable energy sources.

    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network object containing the energy system model.
    additionality : bool
        If True, only new renewable energy sources built in the current planning horizon are considered.
    time_period : str
        Specifies the temporal matching period. Valid options are "hour", "month", "year", or "no_temporal_matching".

    Description:
    ------------
    This function calculates the renewable energy generation and storage dispatch over the specified time period
    and ensures that hydrogen production via electrolysis does not exceed the allowed excess of renewable energy.
    It adds constraints to the PyPSA model to enforce this temporal matching.

    Raises:
    -------
    ValueError:
        If the `time_period` is invalid or not supported.
    """
    temporal_matching_carriers = snakemake.params.temporal_matching_carriers

    allowed_excess = snakemake.config["policy_config"]["hydrogen"]["allowed_excess"]

    res_gen_index = n.generators.loc[n.generators.carrier.isin(temporal_matching_carriers)].index
    res_stor_index = n.storage_units.loc[
        n.storage_units.carrier.isin(temporal_matching_carriers)
    ].index

    if additionality:
        # get newly built generators and storage_units only
        new_gens = n.generators.loc[
            n.generators.build_year == int(snakemake.wildcards.planning_horizons)
        ].index
        new_stor = n.storage_units.loc[
            n.storage_units.build_year == int(snakemake.wildcards.planning_horizons)
        ].index
        # keep only new RES generators and storage units
        res_gen_index = res_gen_index.intersection(new_gens)
        res_stor_index = res_stor_index.intersection(new_stor)

    logger.info(
        "setting h2 export to {}ly matching constraint {} additionality".format(
            time_period, "with" if additionality else "without"
        )
    )

    weightings_gen = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_gen_index)),
        index=n.snapshots,
        columns=res_gen_index,
    )

    res = linexpr((weightings_gen, get_var(n, "Generator", "p")[res_gen_index])).sum(
        axis=1
    )

    if not res_stor_index.empty:
        weightings_stor = pd.DataFrame(
            np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_stor_index)),
            index=n.snapshots,
            columns=res_stor_index,
        )
        res += linexpr(
            (weightings_stor, get_var(n, "StorageUnit", "p_dispatch")[res_stor_index])
        ).sum(axis=1)

    if time_period == "month":
        res = res.groupby(res.index.month).sum()
    elif time_period == "year":
        res = res.groupby(res.index.year).sum()

    electrolysis_carriers = [
        'H2 Electrolysis',
        'Alkaline electrolyzer large',
        'Alkaline electrolyzer medium',
        'Alkaline electrolyzer small',
        'PEM electrolyzer',
        'SOEC'
    ]
    electrolyzers = n.links[n.links.carrier.isin(electrolysis_carriers)].index
    electrolysis = get_var(n, "Link", "p")[
        n.links.loc[electrolyzers].index
    ]
    weightings_electrolysis = pd.DataFrame(
        np.outer(
            n.snapshot_weightings["generators"], [
                1.0] * len(electrolysis.columns)
        ),
        index=n.snapshots,
        columns=electrolysis.columns,
    )

    elec_input = linexpr((-allowed_excess * weightings_electrolysis, electrolysis)).sum(
        axis=1
    )

    if time_period == "month":
        elec_input = elec_input.groupby(elec_input.index.month).sum()
    elif time_period == "year":
        elec_input = elec_input.groupby(elec_input.index.year).sum()

    # add temporal matching constraints
    for i in range(len(res.index)):
        lhs = res.iloc[i] + "\n" + elec_input.iloc[i]

        con = define_constraints(
            n, lhs, ">=", 0.0, f"RESconstraints_{i}", f"REStarget_{i}"
        )


def add_chp_constraints(n):
    electric_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links.index[electric_bool & n.links.p_nom_extendable]
    heat_ext = n.links.index[heat_bool & n.links.p_nom_extendable]

    electric_fix = n.links.index[electric_bool & ~n.links.p_nom_extendable]
    heat_fix = n.links.index[heat_bool & ~n.links.p_nom_extendable]

    link_p = get_var(n, "Link", "p")

    if not electric_ext.empty:
        link_p_nom = get_var(n, "Link", "p_nom")

        # ratio of output heat to electricity set by p_nom_ratio
        lhs = linexpr(
            (
                n.links.loc[electric_ext, "efficiency"]
                * n.links.loc[electric_ext, "p_nom_ratio"],
                link_p_nom[electric_ext],
            ),
            (-n.links.loc[heat_ext, "efficiency"].values,
             link_p_nom[heat_ext].values),
        )

        define_constraints(n, lhs, "=", 0, "chplink", "fix_p_nom_ratio")

        # top_iso_fuel_line for extendable
        lhs = linexpr(
            (1, link_p[heat_ext]),
            (1, link_p[electric_ext].values),
            (-1, link_p_nom[electric_ext].values),
        )

        define_constraints(n, lhs, "<=", 0, "chplink", "top_iso_fuel_line_ext")

    if not electric_fix.empty:
        # top_iso_fuel_line for fixed
        lhs = linexpr((1, link_p[heat_fix]), (1, link_p[electric_fix].values))

        rhs = n.links.loc[electric_fix, "p_nom"].values

        define_constraints(n, lhs, "<=", rhs, "chplink",
                           "top_iso_fuel_line_fix")

    if not electric.empty:
        # backpressure
        lhs = linexpr(
            (
                n.links.loc[electric, "c_b"].values *
                n.links.loc[heat, "efficiency"],
                link_p[heat],
            ),
            (-n.links.loc[electric, "efficiency"].values,
             link_p[electric].values),
        )

        define_constraints(n, lhs, "<=", 0, "chplink", "backpressure")


def add_co2_sequestration_limit(n, sns):
    co2_stores = n.stores.loc[n.stores.carrier == "co2 stored"].index

    if co2_stores.empty or ("Store", "e") not in n.variables.index:
        return

    vars_final_co2_stored = get_var(n, "Store", "e").loc[sns[-1], co2_stores]

    lhs = linexpr((1, vars_final_co2_stored)).sum()
    rhs = (
        n.config["sector"].get("co2_sequestration_potential", 5) * 1e6
    )  # TODO change 200 limit (Europe)

    name = "co2_sequestration_limit"
    define_constraints(
        n, lhs, "<=", rhs, "GlobalConstraint", "mu", axes=pd.Index([name]), spec=name
    )


def set_h2_colors(n):
    blue_h2 = get_var(n, "Link", "p")[
        n.links.index[n.links.index.str.contains("blue H2")]
    ]

    pink_h2 = get_var(n, "Link", "p")[
        n.links.index[n.links.index.str.contains("pink H2")]
    ]

    fuelcell_ind = n.loads[n.loads.carrier == "land transport fuel cell"].index

    other_ind = n.loads[
        (n.loads.carrier == "H2 for industry")
        | (n.loads.carrier == "H2 for shipping")
        | (n.loads.carrier == "H2")
    ].index

    load_fuelcell = (
        n.loads_t.p_set[fuelcell_ind].sum(
            axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    load_other_h2 = n.loads.loc[other_ind].p_set.sum() * 8760

    load_h2 = load_fuelcell + load_other_h2

    weightings_blue = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [
                 1.0] * len(blue_h2.columns)),
        index=n.snapshots,
        columns=blue_h2.columns,
    )

    weightings_pink = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [
                 1.0] * len(pink_h2.columns)),
        index=n.snapshots,
        columns=pink_h2.columns,
    )

    total_blue = linexpr((weightings_blue, blue_h2)).sum().sum()

    total_pink = linexpr((weightings_pink, pink_h2)).sum().sum()

    rhs_blue = load_h2 * snakemake.config["sector"]["hydrogen"]["blue_share"]
    rhs_pink = load_h2 * snakemake.config["sector"]["hydrogen"]["pink_share"]

    define_constraints(n, total_blue, "=", rhs_blue, "blue_h2_share")

    define_constraints(n, total_pink, "=", rhs_pink, "pink_h2_share")


def add_existing(n):
    if snakemake.wildcards["planning_horizons"] == "2050":
        directory = (
            "results/"
            + "Existing_capacities/"
            + snakemake.config["run"].replace("2050", "2030")
        )
        n_name = (
            snakemake.input.network.split("/")[-1]
            .replace(str(snakemake.config["scenario"]["clusters"][0]), "")
            .replace(str(snakemake.config["costs"]["discountrate"][0]), "")
            .replace("_presec", "")
            .replace(".nc", ".csv")
        )
        df = pd.read_csv(directory + "/electrolyzer_caps_" +
                         n_name, index_col=0)
        existing_electrolyzers = df.p_nom_opt.values

        h2_index = n.links[n.links.carrier == "H2 Electrolysis"].index
        n.links.loc[h2_index, "p_nom_min"] = existing_electrolyzers

        # n_name = snakemake.input.network.split("/")[-1].replace(str(snakemake.config["scenario"]["clusters"][0]), "").\
        #     replace(".nc", ".csv").replace(str(snakemake.config["costs"]["discountrate"][0]), "")
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)

        for tech in snakemake.config["custom_data"]["renewables"]:
            # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
            existing_res = df.loc[tech]
            existing_res.index = existing_res.index.str.apply(
                lambda x: x + tech)
            tech_index = n.generators[n.generators.carrier == tech].index
            n.generators.loc[tech_index, tech] = existing_res


def add_lossy_bidirectional_link_constraints(n: pypsa.components.Network) -> None:
    """
    Ensures that the two links simulating a bidirectional_link are extended the same amount.
    """

    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    # ensure that the 'reversed' column is boolean and identify all link carriers that have 'reversed' links
    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    # get the indices of all forward links (non-reversed), that have a reversed counterpart
    forward_i = n.links.loc[
        n.links.carrier.isin(
            carriers) & ~n.links.reversed & n.links.p_nom_extendable
    ].index

    # function to get backward (reversed) indices corresponding to forward links
    # this function is required to properly interact with the myopic naming scheme
    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    # get the indices of all backward links (reversed)
    backward_i = get_backward_i(forward_i)

    # get the p_nom optimization variables for the links using the get_var function
    links_p_nom = get_var(n, "Link", "p_nom")

    # only consider forward and backward links that are present in the optimization variables
    subset_forward = forward_i.intersection(links_p_nom.index)
    subset_backward = backward_i.intersection(links_p_nom.index)

    # ensure we have a matching number of forward and backward links
    if len(subset_forward) != len(subset_backward):
        raise ValueError("Mismatch between forward and backward links.")

    # define the lefthand side of the constrain p_nom (forward) - p_nom (backward) = 0
    # this ensures that the forward links always have the same maximum nominal power as their backward counterpart
    lhs = linexpr(
        (1, get_var(n, "Link", "p_nom")[backward_i].to_numpy()),
        (-1, get_var(n, "Link", "p_nom")[forward_i].to_numpy()),
    )

    # add the constraint to the PySPA model
    define_constraints(n, lhs, "=", 0, "Link-bidirectional_sync")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.linopf.network_lopf``.

    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "RES" in o:
            res_share = float(re.findall("[0-9]*\.?[0-9]+$", o)[0])
            add_RES_constraints(n, res_share)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)

    additionality = snakemake.config["policy_config"]["hydrogen"]["additionality"]
    temporal_matching_period = snakemake.config["policy_config"]["hydrogen"][
        "temporal_matching"
    ]

    if temporal_matching_period == "no_temporal_matching":
        logger.info("no h2 temporal constraint set")

    elif temporal_matching_period in ["hourly", "monthly", "yearly"]:
        temporal_matching_period = temporal_matching_period[:-2]
        hydrogen_temporal_constraint(n, additionality, temporal_matching_period)

    else:
        raise ValueError(
            'H2 export constraint is invalid, check config["policy_config"]'
        )

    if snakemake.config["sector"]["hydrogen"]["network"]:
        if snakemake.config["sector"]["hydrogen"]["network_limit"]:
            add_h2_network_cap(
                n, snakemake.config["sector"]["hydrogen"]["network_limit"]
            )

    if snakemake.config["sector"]["hydrogen"]["set_color_shares"]:
        logger.info("setting H2 color mix")
        set_h2_colors(n)

    add_co2_sequestration_limit(n, snapshots)

    if config["state_policy"] == "on" and n.generators.p_nom_extendable.any():
        add_RPS_constraints(n, config)


def solve_network(n, config, solving={}, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    solver_options = solving["solver_options"][set_of_options] if set_of_options else {
    }
    solver_name = solving["solver"]["name"]

    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get("skip_iterations", False):
        network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            **kwargs,
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            extra_functionality=extra_functionality,
            **kwargs,
        )
    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        # from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_custom_network_myopic",
            configfile="configs/scenarios/config.myopic.yaml",
            simpl="",
            ll="copt",
            clusters=10,
            opts="24H",
            sopts="24H",
            planning_horizons="2030",
            discountrate="0.071",
            demand="AB",
            h2export="10",
        )

    configure_logging(snakemake)

    tmpdir = snakemake.params.solving.get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split("-")
    solving = snakemake.params.solving

    is_sector_coupled = "sopts" in snakemake.wildcards.keys()

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=overrides)

    if snakemake.params.augmented_line_connection.get("add_to_snakefile"):
        n.lines.loc[n.lines.index.str.contains("new"), "s_nom_min"] = (
            snakemake.params.augmented_line_connection.get("min_expansion")
        )

    if (
        snakemake.config["custom_data"]["add_existing"]
        and snakemake.wildcards.planning_horizons == "2050"
        and is_sector_coupled
    ):
        add_existing(n)

    n = prepare_network(n, solving["options"])

    # Ensure marginal cost restoration and initialization of original values
    for comp_name, comp_df in [("generators", n.generators), ("links", n.links)]:
        if "_marginal_cost_original" not in comp_df.columns:
            comp_df["_marginal_cost_original"] = comp_df["marginal_cost"]
        else:
            # Set original marginal cost only for newly added rows (with NaN)
            new_rows = comp_df["_marginal_cost_original"].isna()
            comp_df.loc[new_rows, "_marginal_cost_original"] = comp_df.loc[new_rows, "marginal_cost"]

    # Restore marginal cost to original before applying new tax credits
    for comp_name, comp_df in [("generators", n.generators), ("links", n.links)]:
        if "_marginal_cost_original" in comp_df.columns:
            comp_df["marginal_cost"] = comp_df["_marginal_cost_original"]

    logger.info(f"Applying tax credits for {snakemake.wildcards.planning_horizons}")
    apply_tax_credits_to_network(
        n,
        ptc_path=snakemake.input.production_tax_credits,
        itc_path=snakemake.input.investment_tax_credits,
        planning_horizon=int(snakemake.wildcards.planning_horizons),
        costs=pd.read_csv(snakemake.input.costs, index_col=0),
        log_path=f"logs/tax_credit_modifications_{snakemake.wildcards.planning_horizons}.csv",
        verbose=False
    )

    n = solve_network(
        n,
        config=snakemake.config,
        solving=solving,
        opts=opts,
        solver_dir=tmpdir,
        solver_logfile=snakemake.log.solver,
    )
    n.meta = dict(snakemake.config, **
                  dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Objective function: {n.objective}")
    logger.info(f"Objective constant: {n.objective_constant}")