# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from snakemake.utils import min_version
min_version("6.0")

import sys
sys.path.append("submodules/pypsa-earth")
sys.path.append("submodules/pypsa-earth/scripts")

from scripts._helper import BASE_PATH

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


rule retrieve_cutouts:
    params:
        countries=config["countries"],
    output:
        cutouts=PYPSA_EARTH_DIR+"cutouts/cutout-2013-era5.nc"
    resources:
        mem_mb=16000,
    script:
        "scripts/retrieve_cutouts.py"


rule use_osm_data:
    input:
        generators_csv=BASE_PATH + "/submodules/pypsa-earth/resources/" + RDIR + "osm/clean/all_clean_generators.csv",
    output:
        output_csv="resources/" + RDIR + "osm/clean/all_clean_generators.csv",
    resources:
        mem_mb=16000,
    script:
        "scripts/use_osm_data.py"


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


