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
CDIR = RDIR if not run.get("shared_cutouts") else ""
SECDIR = run["sector_name"] + "/" if run.get("sector_name") else ""
SDIR = config["summary_dir"].strip("/") + f"/{SECDIR}"
RESDIR = config["results_dir"].strip("/") + f"/{SECDIR}"

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


# TODO: revise retrieve cost data
use rule retrieve_cost_data from pypsa_earth with:
    input:
        HTTP.remote(
            f"raw.githubusercontent.com/open-energy-transition/technology-data/master/outputs/US/costs"
            + "_{year}.csv",
            keep_local=True,
        ),


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
            **{k: v for k, v in rules.build_powerplants.input.items() if k != "custom_powerplants"},
            custom_powerplants="data/custom_powerplants.csv",


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

    rule prepare_growth_rate_scenarios:
        input:
            source_growth_factors=lambda wildcards: f"data/US_growth_rates/{config['demand_projection']['scenario']}/growth_factors_cagr.csv",
            source_industry_growth=lambda wildcards: f"data/US_growth_rates/{config['demand_projection']['scenario']}/industry_growth_cagr.csv"
        output:
            growth_factors_cagr=PYPSA_EARTH_DIR + "data/demand/growth_factors_cagr.csv",
            industry_growth_cagr=PYPSA_EARTH_DIR + "data/demand/industry_growth_cagr.csv"
        script:
            "scripts/prepare_growth_rate_scenarios.py"

    use rule prepare_energy_totals from pypsa_earth with:
        params:
            countries=config["countries"],
            base_year=config["demand_data"]["base_year"],
            sector_options=config["sector"],
        output:
            energy_totals=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "energy_totals_{demand}_{planning_horizons}_aviation_mod.csv"

    rule modify_aviation_demand:
        input:
            aviation_demand="data/icct/aviation_demand.csv",
            energy_totals=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "energy_totals_{demand}_{planning_horizons}_aviation_mod.csv"
        output:
            energy_totals=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "energy_totals_{demand}_{planning_horizons}.csv"
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
        params:
            demand_projections="data/demand_projections/",
            demand_horizon=config["demand_projection"]["planning_horizon"],
            demand_scenario=config["demand_projection"]["scenario"],
            data_center_profiles="data/data_center_profiles/",
            geo_crs=config["crs"]["geo_crs"],
        input:
            BA_demand_path1="data/demand_data/EIA930_2023_Jan_Jun_opt.csv",
            BA_demand_path2="data/demand_data/EIA930_2023_Jul_Dec_opt.csv",
            BA_shape_path="data/demand_data/Balancing_Authorities.geojson",
            utility_demand_path="data/demand_data/ERST_mapped_demand_centroids.geojson",
            base_network=PYPSA_EARTH_DIR + "networks/" + RDIR + "base.nc",
            gadm_shape="data/demand_data/gadm41_USA_1.json",
        output:
            demand_profile_path=PYPSA_EARTH_DIR + "resources/" + RDIR + "demand_profiles.csv",
        script:
            "scripts/build_demand_profiles_from_eia.py"


    ruleorder: build_demand_profiles_from_eia > build_demand_profiles
      

if config["saf_mandate"]["ekerosene_split"]:
    rule set_saf_mandate:
        params:
            non_spatial_ekerosene=config["saf_mandate"]["non_spatial_ekerosene"],
            saf_scenario=config["saf_mandate"]["saf_scenario"],
        input:
            network=PYPSA_EARTH_DIR + "results/"
            + SECDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}.nc",
            saf_scenarios="data/saf_blending_rates/saf_scenarios.csv",
        output:
            modified_network=PYPSA_EARTH_DIR + "results/"
            + SECDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_saf.nc",
        script:
            "scripts/set_saf_mandate.py"


saf_suffix = "_saf" if config["saf_mandate"]["ekerosene_split"] else ""


if config["custom_industry"]["enable"]:
    rule build_custom_industry_demand:
        params:
            countries=config["countries"],
            add_ethanol=config["custom_industry"]["ethanol"],
            add_ammonia=config["custom_industry"]["ammonia"],
            add_steel=config["custom_industry"]["steel"],
            add_cement=config["custom_industry"]["cement"],
            gadm_layer_id=config["build_shape_options"]["gadm_layer_id"],
            alternative_clustering=config["cluster_options"]["alternative_clustering"],
            industry_database=config["custom_data"]["industry_database"],
        input:
            uscity_map="data/industry_data/uscities.csv",
            ethanol_plants="data/industry_data/ethanolcapacity.xlsx",
            ammonia_plants="data/industry_data/ammoniacapacity.xlsx",
            shapes_path=PYPSA_EARTH_DIR + "resources/"
            + RDIR
            + "bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson",
            pypsa_earth_industrial_database=PYPSA_EARTH_DIR + "data/industrial_database.csv",
            industry_growth_cagr=PYPSA_EARTH_DIR + "data/demand/industry_growth_cagr.csv",
        output:
            industrial_energy_demand_per_node=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "demand/industrial_energy_demand_per_node_elec_s{simpl}_{clusters}_{planning_horizons}_{demand}_custom_industry.csv",
        threads: 1
        resources:
            mem_mb=2000,
        script:
            "scripts/build_custom_industry_demand.py"


    rule add_custom_industry:
        params:
            costs=config["costs"],
            add_ethanol=config["custom_industry"]["ethanol"],
            add_ammonia=config["custom_industry"]["ammonia"],
            add_steel=config["custom_industry"]["steel"],
            add_cement=config["custom_industry"]["cement"],
            ccs_retrofit=config["custom_industry"]["CCS_retrofit"],
            biogenic_co2=config["custom_industry"]["biogenic_co2"],
            grid_h2=config["custom_industry"]["grid_H2"],
            other_electricity=config["custom_industry"]["other_electricity"],
            data_centers=config["demand_projection"]["data_centers_load"],
            data_center_profiles="data/data_center_profiles/",
            geo_crs=config["crs"]["geo_crs"],
        input:
            industrial_energy_demand_per_node=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "demand/industrial_energy_demand_per_node_elec_s{simpl}_{clusters}_{planning_horizons}_{demand}_custom_industry.csv",
            energy_totals=PYPSA_EARTH_DIR + "resources/"
            + SECDIR
            + "energy_totals_{demand}_{planning_horizons}.csv",
            network=lambda w: f"{PYPSA_EARTH_DIR}results/{SECDIR}prenetworks/elec_s{w.simpl}_{w.clusters}_ec_l{w.ll}_{w.opts}_{w.sopts}_{w.planning_horizons}_{w.discountrate}_{w.demand}{saf_suffix}.nc",
            costs=PYPSA_EARTH_DIR + "resources/" + RDIR + "costs_{planning_horizons}.csv",
            gadm_shape="data/demand_data/gadm41_USA_1.json",
        output:
            modified_network=PYPSA_EARTH_DIR + "results/"
            + SECDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_custom_industry.nc",
        script:
            "scripts/add_custom_industry.py"


    use rule add_export from pypsa_earth with:
        input:
            **{k: v for k, v in rules.add_export.input.items() if k != "network"},
            network=PYPSA_EARTH_DIR + "results/"
            + SECDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_custom_industry.nc",

if config["foresight"] == "overnight":
    use rule solve_sector_network from pypsa_earth with:
        input:
            **{k: v for k, v in rules.solve_sector_network.input.items() if k != "overrides"},
            overrides="data/override_component_attrs",


if config["foresight"] == "myopic":
    use rule solve_network_myopic from pypsa_earth with:
        input:
            **{k: v for k, v in rules.solve_network_myopic.input.items() if k not in ["overrides", "network"]},
            overrides="data/override_component_attrs",
            network=PYPSA_EARTH_DIR + RESDIR
            + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_custom.nc",


    rule add_custom_existing_baseyear:
        params:
            baseyear=config["scenario"]["planning_horizons"][0],
            sector=config["sector"],
            existing_capacities=config["existing_capacities"],
            costs=config["costs"],
        input:
            network=PYPSA_EARTH_DIR + RESDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
            powerplants=PYPSA_EARTH_DIR + "resources/" + RDIR + "powerplants.csv",
            busmap_s=PYPSA_EARTH_DIR + "resources/" + RDIR + "bus_regions/busmap_elec_s{simpl}.csv",
            busmap=PYPSA_EARTH_DIR + "resources/"
            + RDIR
            + "bus_regions/busmap_elec_s{simpl}_{clusters}.csv",
            costs=PYPSA_EARTH_DIR + "resources/" + RDIR + "costs_{planning_horizons}.csv",
        output:
            PYPSA_EARTH_DIR + RESDIR
            + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_custom.nc",
        wildcard_constraints:
            # TODO: The first planning_horizon needs to be aligned across scenarios
            # snakemake does not support passing functions to wildcard_constraints
            # reference: https://github.com/snakemake/snakemake/issues/2703
            planning_horizons=config["scenario"]["planning_horizons"][0],  #only applies to baseyear
        threads: 1
        resources:
            mem_mb=2000,
        log:
            PYPSA_EARTH_DIR + RESDIR
            + "logs/add_custom_existing_baseyear_elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.log",
        benchmark:
            PYPSA_EARTH_DIR + RESDIR
            +"benchmarks/add_custom_existing_baseyear/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export"
        script:
            "scripts/add_custom_existing_baseyear.py"


    ruleorder: add_custom_existing_baseyear > add_custom_brownfield


    def solved_previous_horizon(w):
        planning_horizons = config["scenario"]["planning_horizons"]
        i = planning_horizons.index(int(w.planning_horizons))
        planning_horizon_p = str(planning_horizons[i - 1])

        return (
            PYPSA_EARTH_DIR + RESDIR
            + "postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_"
            + planning_horizon_p
            + "_{discountrate}_{demand}_{h2export}export.nc"
        )

    def get_previous_planning_horizon(w):
        planning_horizons = config["scenario"]["planning_horizons"]
        i = planning_horizons.index(int(w.planning_horizons))
        return str(planning_horizons[i - 1])


    rule add_custom_brownfield:
        params:
            H2_retrofit=config["sector"]["hydrogen"],
            H2_retrofit_capacity_per_CH4=config["sector"]["hydrogen"][
                "H2_retrofit_capacity_per_CH4"
            ],
            threshold_capacity=config["existing_capacities"]["threshold_capacity"],
            snapshots=config["snapshots"],
            # drop_leap_day=config["enable"]["drop_leap_day"],
            carriers=config["electricity"]["renewable_carriers"],
            planning_horizon_p=get_previous_planning_horizon,
            costs=config["costs"],
        input:
            # unpack(input_profile_tech_brownfield),
            simplify_busmap=PYPSA_EARTH_DIR + "resources/" + RDIR + "bus_regions/busmap_elec_s{simpl}.csv",
            cluster_busmap=PYPSA_EARTH_DIR + "resources/"
            + RDIR
            + "bus_regions/busmap_elec_s{simpl}_{clusters}.csv",
            network=PYPSA_EARTH_DIR + RESDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
            network_p=solved_previous_horizon,  #solved network at previous time step
            costs=PYPSA_EARTH_DIR + "resources/" + RDIR + "costs_{planning_horizons}.csv",
        output:
            PYPSA_EARTH_DIR + RESDIR
            + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_custom.nc",
        threads: 4
        resources:
            mem_mb=10000,
        log:
            PYPSA_EARTH_DIR + RESDIR
            + "logs/add_custom_brownfield_elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.log",
        benchmark:
            (
                PYPSA_EARTH_DIR + RESDIR
                + "benchmarks/add_custom_brownfield/elec_s{simpl}_ec_{clusters}_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export"
            )
        script:
            "scripts/add_custom_brownfield.py"


    rule solve_custom_network_myopic:
        params:
            solving=config["solving"],
            foresight=config["foresight"],
            planning_horizons=config["scenario"]["planning_horizons"],
            co2_sequestration_potential=config["scenario"].get(
                "co2_sequestration_potential", 200
            ),
            augmented_line_connection=config["augmented_line_connection"],
            temporal_matching_carriers=config["policy_config"]["hydrogen"]["temporal_matching_carriers"],
        input:
            ces_path="data/current_electricity_state_policies/clean_targets.csv",
            res_path="data/current_electricity_state_policies/res_targets.csv",
            production_tax_credits="data/tax_credits/production_tax_credits.csv",
            investment_tax_credits="data/tax_credits/investment_tax_credits.csv",
            gadm_shape_path="data/demand_data/gadm41_USA_1.json",
            overrides="data/override_component_attrs",
            network=PYPSA_EARTH_DIR + RESDIR
            + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_custom.nc",
            costs=PYPSA_EARTH_DIR + "resources/" + RDIR + "costs_{planning_horizons}.csv",
            configs=PYPSA_EARTH_DIR + SDIR + "configs/config.yaml",  # included to trigger copy_config rule
        output:
            network=PYPSA_EARTH_DIR + RESDIR
            + "postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
            # config=RESDIR
            # + "configs/config.elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.yaml",
        shadow:
            "shallow"
        log:
            solver=PYPSA_EARTH_DIR + RESDIR
            + "logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_solver.log",
            python=PYPSA_EARTH_DIR + RESDIR
            + "logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_python.log",
            memory=PYPSA_EARTH_DIR + RESDIR
            + "logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_memory.log",
        threads: 25
        resources:
            mem_mb=config["solving"]["mem"],
        benchmark:
            (
                PYPSA_EARTH_DIR + RESDIR
                + "benchmarks/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export"
            )
        script:
            "scripts/solve_custom_sector_network.py"


    ruleorder: solve_custom_network_myopic > solve_network_myopic


if config["demand_distribution"]["set_custom_distribution_fees"]:
    use rule prepare_sector_network from pypsa_earth with:
        output:
            PYPSA_EARTH_DIR + RESDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_distribution_fees.nc",

    rule set_custom_distribution_fees:
        params:
            distance_crs=config["crs"]["distance_crs"],
        input:
            shape_path="data/EIA_market_module_regions/EMM_regions.geojson",
            regional_fees_path="data/EIA_market_module_regions/regional_fees.csv",
            network=PYPSA_EARTH_DIR + RESDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_distribution_fees.nc",
        output:
            PYPSA_EARTH_DIR + RESDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}.nc",
        script:
            "scripts/set_custom_distribution_fees.py"

if config["foresight"] == "overnight" and config["state_policy"] != "off":    
    rule solve_custom_sector_network:
        params:
            solving=config["solving"],
            augmented_line_connection=config["augmented_line_connection"],
            temporal_matching_carriers=config["policy_config"]["hydrogen"]["temporal_matching_carriers"],
        input:
            ces_path="data/current_electricity_state_policies/clean_targets.csv",
            res_path="data/current_electricity_state_policies/res_targets.csv",
            production_tax_credits="data/tax_credits/production_tax_credits.csv",
            investment_tax_credits="data/tax_credits/investment_tax_credits.csv",
            gadm_shape_path="data/demand_data/gadm41_USA_1.json",
            overrides="data/override_component_attrs",
            network=PYPSA_EARTH_DIR + RESDIR
            + "prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
            costs=PYPSA_EARTH_DIR + "resources/" + RDIR + "costs_{planning_horizons}.csv",
            configs=PYPSA_EARTH_DIR + SDIR + "configs/config.yaml",  # included to trigger copy_config rule
        output:
            PYPSA_EARTH_DIR + RESDIR
            + "postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
        shadow:
            "shallow"
        log:
            solver=PYPSA_EARTH_DIR + RESDIR
            + "logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_solver.log",
            python=PYPSA_EARTH_DIR + RESDIR
            + "logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_python.log",
            memory=PYPSA_EARTH_DIR + RESDIR
            + "logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_memory.log",
        threads: 25
        resources:
            mem_mb=config["solving"]["mem"],
        benchmark:
            (
                PYPSA_EARTH_DIR + RESDIR
                + "benchmarks/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export"
            )
        script:
            "scripts/solve_custom_sector_network.py"

        
    ruleorder: solve_custom_sector_network > solve_sector_network


use rule cluster_network from pypsa_earth with:
    params:
        **{k: v for k, v in rules.cluster_network.params.items() if k != "custom_busmap"},
        custom_busmap=config["enable"].get("custom_busmap", False)
    input:
        **{k: v for k, v in rules.cluster_network.input.items() if k != "custom_busmap"},
        custom_busmap=("data/custom_busmap_elec_s{simpl}_{clusters}.csv"
                       if config["enable"].get("custom_busmap", False) else []),


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


