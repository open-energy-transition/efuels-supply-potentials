# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from snakemake.utils import min_version
min_version("6.0")

import sys
sys.path.append("submodules/pypsa-earth")
sys.path.append("submodules/pypsa-earth/scripts")

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
from scripts._helper import renewable_profiles_outputs

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
        aviation_demand="data/icct/aviation_demand.csv",
    output:
        statewise_output="plots/results/passengers_vs_consumption.csv",
        merged_data="plots/results/merged_airports.csv",
        consumption_per_passenger="plots/results/consumption_per_passenger.png",
        correlation_matrix="plots/results/correlation_matrix.png",
        comparision_consumption_passengers="plots/results/comparision_consumption_passengers.png",
        custom_airports_data=PYPSA_EARTH_DIR + "resources/" + SECDIR + "airports.csv",
    resources:
        mem_mb=3000,
    script:
        "plots/airport_data_postprocessing.py"

rule generate_aviation_scenario:
    input:
        aviation_demand_data="data/icct/US Aviation Fuel Demand Projection_NP_0.1.xls",
    output:
        scenario_df="data/icct/aviation_demand.csv",
    resources:
        mem_mb=3000,
    script:
        "scripts/generate_aviation_scenarios.py"


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


# use rule retrieve_cost_data from pypsa_earth with:
#     input:
#         HTTP.remote(
#             f"raw.githubusercontent.com/open-energy-transition/technology-data/nrel_atb_usa_costs/outputs/US/costs"
#             + "_{year}.csv",
#             keep_local=True,
#         ),


# retrieving precomputed osm/raw data and bypassing download_osm_data rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("osm_raw", False):
    rule retrieve_osm_raw:
        params:
            destination="resources/" + RDIR,
        input:
            **{k: v for k, v in rules.download_osm_data.input.items()},
        output:
            **{k: v for k, v in rules.download_osm_data.output.items()},
        script:
            "scripts/retrieve_osm_raw.py"

    ruleorder: retrieve_osm_raw > download_osm_data


# retrieving precomputed osm/clean data and bypassing clean_osm_data rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("osm_clean", False):
    rule retrieve_osm_clean:
        params:
            destination="resources/" + RDIR,
        input:
            **{k: v for k, v in rules.clean_osm_data.input.items()},
        output:
            **{k: v for k, v in rules.clean_osm_data.output.items()},
        script:
            "scripts/retrieve_osm_clean.py"

    ruleorder: retrieve_osm_clean > clean_osm_data


# retrieving shapes data and bypassing build_shapes rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("shapes", False):
    rule retrieve_shapes:
        params:
            destination="resources/" + RDIR,
        input:
            **{k: v for k, v in rules.build_shapes.input.items()},
        output:
            **{k: v for k, v in rules.build_shapes.output.items()},
        script:
            "scripts/retrieve_shapes.py"

    ruleorder: retrieve_shapes > build_shapes


# retrieving base_network data and bypassing build_osm_network rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("osm_network", False):
    rule retrieve_osm_network:
        params:
            destination="resources/" + RDIR,
        input:
            **{k: v for k, v in rules.build_osm_network.input.items()},
        output:
            **{k: v for k, v in rules.build_osm_network.output.items()},
        script:
            "scripts/retrieve_osm_network.py"

    ruleorder: retrieve_osm_network > build_osm_network


# retrieving base.nc and bypassing base_network rule
if config["countries"] == ["US"] and config["retrieve_from_gdrive"].get("base_network", False):
    rule retrieve_base_network:
        input:
            **{k: v for k, v in rules.base_network.input.items()},
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


if (config["countries"] == ["US"]):

    use rule build_powerplants from pypsa_earth with:
        input:
            **{k: v for k, v in rules.build_powerplants.input.items()},
            powerplants_dummy_input=temp("powerplants_dummy_output.log"),

    rule retrieve_custom_powerplants:
        input:
            old_path="data/custom_powerplants.csv",
        output:
            destination=PYPSA_EARTH_DIR + "data/custom_powerplants.csv",
            powerplants_dummy_input=temp("powerplants_dummy_output.log"),
        script:
            "scripts/retrieve_powerplants.py"


if config["countries"] == ["US"]:

    use rule build_demand_profiles from pypsa_earth with:
        input:
            **{k: v for k, v in rules.build_demand_profiles.input.items() if k != "load"},
            ssp2_dummy_input=temp("ssp2_dummy_output.log"),
            load = [PYPSA_EARTH_DIR + 'data/ssp2-2.6/2030/era5_2013/NorthAmerica.csv'],

    rule retrieve_ssp2:
        params:
            nc_path=PYPSA_EARTH_DIR + "data/ssp2-2.6/2030/era5_2013/NorthAmerica.nc",
        input:
            old_path="data/NorthAmerica.csv",
        output:
            ssp2_northamerica=PYPSA_EARTH_DIR + "data/ssp2-2.6/2030/era5_2013/NorthAmerica.csv",
            ssp2_dummy_output=temp("ssp2_dummy_output.log"),
        script:
            "scripts/retrieve_ssp2.py"

            
if config["countries"] == ["US"]:

    use rule prepare_energy_totals from pypsa_earth with:
        output:
            energy_totals=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "energy_totals_{demand}_{planning_horizons}_aviation_mod.csv",

    rule modify_aviation_demand:
        input:
            aviation_demand="data/icct/aviation_demand.csv",
            energy_totals=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "energy_totals_{demand}_{planning_horizons}_aviation_mod.csv",
        output:
            energy_totals=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "energy_totals_{demand}_{planning_horizons}.csv",
        script:
            "scripts/modify_aviation_demand.py"
        

if config["demand_distribution"]["enable"]:
    rule preprocess_demand_data:
        input:
            demand_utility_path="data/demand_data/table_10_EIA_utility_sales.xlsx",
            country_gadm_path=PYPSA_EARTH_DIR + "resources/" + RDIR + "shapes/country_shapes.geojson",
            erst_path="data/demand_data/Electric_Retail_Service_Territories.geojson",
            gadm_usa_path="data/demand_data/gadm41_USA_1.json",
            eia_per_capita_path="data/demand_data/use_es_capita.xlsx",
            additional_demand_path="data/demand_data/HS861_2010-.xlsx",
        output:
            utility_demand_path="data/demand_data/ERST_mapped_demand_centroids.geojson"
        script:
            "scripts/preprocess_demand_data.py"


    rule retrieve_demand_data:
        output:
            "data/demand_data/table_10_EIA_utility_sales.xlsx",
            "data/demand_data/Electric_Retail_Service_Territories.geojson",
            "data/demand_data/gadm41_USA_1.json",
            "data/demand_data/use_es_capita.xlsx",
            "data/demand_data/HS861_2010-.xlsx",
            "data/demand_data/Balancing_Authorities.geojson",
            "data/demand_data/EIA930_2023_Jan_Jun_opt.csv",
            "data/demand_data/EIA930_2023_Jul_Dec_opt.csv",
        script:
            "scripts/retrieve_demand_data.py"


    rule build_demand_profiles_from_eia:
        input:
            BA_demand_path1="data/demand_data/EIA930_2023_Jan_Jun_opt.csv",
            BA_demand_path2="data/demand_data/EIA930_2023_Jul_Dec_opt.csv",
            BA_shape_path="data/demand_data/Balancing_Authorities.geojson",
            utility_demand_path="data/demand_data/ERST_mapped_demand_centroids.geojson",
            base_network=PYPSA_EARTH_DIR + "networks/" + RDIR + "base.nc",
        output:
            demand_profile_path=PYPSA_EARTH_DIR + "resources/" + RDIR + "demand_profiles.csv",
        script:
            "scripts/build_demand_profiles_from_eia.py"


    ruleorder: build_demand_profiles_from_eia > build_demand_profiles
      

if config["saf_mandate"]["enable"]:
    rule add_saf_mandate:
        input:
            network=PYPSA_EARTH_DIR + "results/"
            + SECDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}.nc",
        output:
            modified_network=PYPSA_EARTH_DIR + "results/"
            + SECDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_saf.nc",
        script:
            "scripts/add_saf_mandate.py"


    use rule add_export from pypsa_earth with:
        input:
            **{k: v for k, v in rules.add_export.input.items() if k != "network"},
            network=PYPSA_EARTH_DIR + "results/"
            + SECDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_saf.nc",


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


