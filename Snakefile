# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from snakemake.utils import min_version
min_version("6.0")

import sys
sys.path.append("submodules/pypsa-earth")
sys.path.append("submodules/pypsa-earth/scripts")

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
from scripts._helper import BASE_PATH
from scripts.retrieve_osm_raw import osm_raw_outputs
from scripts.retrieve_osm_clean import osm_clean_outputs
from scripts.retrieve_shapes import shapes_outputs
from scripts.retrieve_osm_network import osm_network_outputs
from scripts.retrieve_renewable_profiles import renewable_profiles_outputs

HTTP = HTTPRemoteProvider()

RESULTS_DIR = "plots/results/"
PYPSA_EARTH_DIR = "submodules/pypsa-earth/"


configfile: "submodules/pypsa-earth/config.default.yaml"
configfile: "submodules/pypsa-earth/configs/bundle_config.yaml"
configfile: "configs/config.main.yaml"


wildcard_constraints:
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+(m|flex)?|all|min",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*",
    unc="[-+a-zA-Z0-9\.]*",
    planning_horizon="[0-9]{4}",
    countries="[A-Z]{2}",


run = config["run"]
RDIR = run["name"] + "/" if run.get("name") else ""
SECDIR = run["sector_name"] + "/" if run.get("sector_name") else ""
CDIR = RDIR if not run.get("shared_cutouts") else ""


module pypsa_earth:
    snakefile:
        "submodules/pypsa-earth/Snakefile"
    config:
        config
    prefix:
        "submodules/pypsa-earth"


use rule * from pypsa_earth


localrules:
    all,


rule validate:
    params:
        countries=config["countries"],
        planning_horizon=config["validation"]["planning_horizon"],
    input:
        solved_network=PYPSA_EARTH_DIR + "results/" + RDIR + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc"
    output:
        demand=RESULTS_DIR + RDIR + "demand_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        capacity=RESULTS_DIR + RDIR + "capacity_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        generation=RESULTS_DIR + RDIR + "generation_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        generation_detailed=RESULTS_DIR + RDIR + "generation_validation_detailed_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        demand_csv=RESULTS_DIR + RDIR + "demand_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
        capacity_csv=RESULTS_DIR + RDIR + "capacity_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
        generation_csv=RESULTS_DIR + RDIR + "generation_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
        generation_detailed_csv=RESULTS_DIR + RDIR + "generation_validation_detailed_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
    resources:
        mem_mb=16000,
    script:
        "plots/results_validation.py"


rule statewise_validate:
    params:
        alternative_clustering=config["cluster_options"]["alternative_clustering"],
        planning_horizon=config["validation"]["planning_horizon"],
        plots_config=config["plotting"],
    input:
        solved_network=PYPSA_EARTH_DIR + "results/" + RDIR + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc"
    output:
        demand_statewise_comparison=RESULTS_DIR + RDIR + "total_demand_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        statewise_installed_capacity_pypsa=RESULTS_DIR + RDIR + "installed_capacity_pypsa_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        statewise_installed_capacity_eia=RESULTS_DIR + RDIR + "installed_capacity_eia_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
        table_demand_statewise_comparison=RESULTS_DIR + RDIR + "total_demand_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
        table_statewise_installed_capacity_pypsa=RESULTS_DIR + RDIR + "installed_capacity_pypsa_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
        table_statewise_installed_capacity_eia=RESULTS_DIR + RDIR + "installed_capacity_eia_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
    resources:
        mem_mb=16000,
    script:
        "plots/state_analysis.py"


rule get_capacity_factor:
    params:
        alternative_clustering=config["cluster_options"]["alternative_clustering"],
    input:
        unsolved_network=PYPSA_EARTH_DIR + "networks/" + RDIR + "elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
        gadm="data/validation/gadm41_USA_1.json"
    output:
        capacity_factors=RESULTS_DIR + RDIR + "capacity_factors_s{simpl}_{clusters}_ec_l{ll}_{opts}.xlsx",
    resources:
        mem_mb=8000,
    script:
        "plots/capacity_factors.py"


rule get_capacity_factors:
    input:
        expand(RESULTS_DIR + RDIR
            + "capacity_factors_s{simpl}_{clusters}_ec_l{ll}_{opts}.xlsx",
            **config["scenario"],
        ),


if config["cluster_options"]["alternative_clustering"]:
    rule statewise_validate_all:
        input:
            expand(RESULTS_DIR + RDIR
                + "total_demand_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
                **config["scenario"],
            ),
            expand(RESULTS_DIR + RDIR
                + "installed_capacity_pypsa_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
                **config["scenario"],
            ),
            expand(RESULTS_DIR + RDIR
                + "installed_capacity_eia_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
                **config["scenario"],
            ),
            expand(RESULTS_DIR + RDIR
                + "total_demand_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
                **config["scenario"],
            ),
            expand(RESULTS_DIR + RDIR
                + "installed_capacity_pypsa_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
                **config["scenario"],
            ),
            expand(RESULTS_DIR + RDIR
                + "installed_capacity_eia_statewise_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
                **config["scenario"],
            ),


rule validate_all:
    input:
        expand(RESULTS_DIR + RDIR
            + "demand_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
            **config["scenario"],
        ),
        expand(RESULTS_DIR + RDIR
            + "capacity_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
            **config["scenario"],
        ),
        expand(RESULTS_DIR + RDIR
            + "generation_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
            **config["scenario"],
        ),
        expand(RESULTS_DIR + RDIR
            + "generation_validation_detailed_s{simpl}_{clusters}_ec_l{ll}_{opts}.png",
            **config["scenario"],
        ),
        expand(RESULTS_DIR + RDIR
            + "demand_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
            **config["scenario"],
        ),
        expand(RESULTS_DIR + RDIR
            + "capacity_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
            **config["scenario"],
        ),
        expand(RESULTS_DIR + RDIR
            + "generation_validation_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
            **config["scenario"],
        ),
        expand(RESULTS_DIR + RDIR
            + "generation_validation_detailed_s{simpl}_{clusters}_ec_l{ll}_{opts}.csv",
            **config["scenario"],
        ),


rule process_airport_data:
    input:
        fuel_data="data/airport_data/fuel_jf.csv",
        airport_data="data/airport_data/airports.csv",
        passengers_data="data/airport_data/T100_Domestic_Market_and_Segment_Data_-3591723781169319541.csv",
    output:
        statewise_output="plots/results/passengers_vs_consumption.csv",
        merged_data="plots/results/merged_airports.csv",
        consumption_per_passenger="plots/results/consumption_per_passenger.png",
        correlation_matrix="plots/results/correlation_matrix.png",
        comparision_consumption_passengers="plots/results/comparision_consumption_passengers.png",
        custom_airports_data=PYPSA_EARTH_DIR + "data/airports.csv",
    resources:
        mem_mb=3000,
    script:
        "plots/airport_data_postprocessing.py"


if config["custom_data"]["airports"]:
    ruleorder: process_airport_data > prepare_airports
else:
    ruleorder: prepare_airports > process_airport_data


if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("cutouts", False):
    rule retrieve_cutouts:
        params:
            countries=config["countries"],
        output:
            cutouts=PYPSA_EARTH_DIR+"cutouts/cutout-2013-era5.nc"
        resources:
            mem_mb=16000,
        script:
            "scripts/retrieve_cutouts.py"


use rule retrieve_cost_data_flexible from pypsa_earth with:
    input:
        HTTP.remote(
            f"raw.githubusercontent.com/open-energy-transition/technology-data/nrel_atb_usa_costs/outputs/US/costs"
            + "_{planning_horizons}.csv",
            keep_local=True,
        ),


# retrieving precomputed osm/raw data and bypassing download_osm_data rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("osm_raw", False):
    rule retrieve_osm_raw:
        params:
            destination="resources/" + RDIR,
        output:
            expand(
                "{PYPSA_EARTH_DIR}resources/{RDIR}{file}", PYPSA_EARTH_DIR=PYPSA_EARTH_DIR, RDIR=RDIR, file=osm_raw_outputs()),
        script:
            "scripts/retrieve_osm_raw.py"

    ruleorder: retrieve_osm_raw > download_osm_data


# retrieving precomputed osm/clean data and bypassing clean_osm_data rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("osm_clean", False):
    rule retrieve_osm_clean:
        params:
            destination="resources/" + RDIR,
        output:
            expand(
                "{PYPSA_EARTH_DIR}resources/{RDIR}{file}", PYPSA_EARTH_DIR=PYPSA_EARTH_DIR, RDIR=RDIR, file=osm_clean_outputs()),
        script:
            "scripts/retrieve_osm_clean.py"

    ruleorder: retrieve_osm_clean > clean_osm_data


# retrieving shapes data and bypassing build_shapes rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("shapes", False):
    rule retrieve_shapes:
        params:
            destination="resources/" + RDIR,
        output:
            expand(
                "{PYPSA_EARTH_DIR}resources/{RDIR}{file}", PYPSA_EARTH_DIR=PYPSA_EARTH_DIR, RDIR=RDIR, file=shapes_outputs()),
        script:
            "scripts/retrieve_shapes.py"

    ruleorder: retrieve_shapes > build_shapes


# retrieving base_network data and bypassing build_osm_network rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("osm_network", False):
    rule retrieve_osm_network:
        params:
            destination="resources/" + RDIR,
        output:
            expand(
                "{PYPSA_EARTH_DIR}resources/{RDIR}{file}", PYPSA_EARTH_DIR=PYPSA_EARTH_DIR, RDIR=RDIR, file=osm_network_outputs()),
        script:
            "scripts/retrieve_osm_network.py"

    ruleorder: retrieve_osm_network > build_osm_network


# retrieving base.nc and bypassing base_network rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("base_network", False):
    rule retrieve_base_network:
        output:
            PYPSA_EARTH_DIR + "networks/" + RDIR + "base.nc",
        script:
            "scripts/retrieve_base_network.py"

    ruleorder: retrieve_base_network > base_network


# retrieving renewable_profiles data and bypassing build_renewable_profiles rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("renewable_profiles", False):
    rule retrieve_renewable_profiles:
        params:
            destination="resources/" + RDIR,
            alternative_clustering=config["cluster_options"]["alternative_clustering"],
        output:
            expand(
                "{PYPSA_EARTH_DIR}resources/{RDIR}{file}", PYPSA_EARTH_DIR=PYPSA_EARTH_DIR, RDIR=RDIR, file=renewable_profiles_outputs()),
        script:
            "scripts/retrieve_renewable_profiles.py"

    ruleorder: retrieve_renewable_profiles > build_renewable_profiles


rule test_modify_prenetwork:
    input:
        prenetwork=PYPSA_EARTH_DIR + "networks/" + RDIR + "elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    output:
        network=PYPSA_EARTH_DIR + "networks/" + RDIR + "elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_mod.nc",
    resources:
        mem_mb=16000,
    script:
        "scripts/test_modify_network.py"


#use rule prepare_network from pypsa_earth with:
#    input:
#        **{k: v for k, v in rules.prepare_network.input.items() if k != "tech_costs"},


#use rule add_extra_components from pypsa_earth with:
#    input:
#        **{k: v for k, v in rules.add_extra_components.input.items()},


