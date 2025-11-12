<!--
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later
-->

# efuels-supply-potentials


## 1. Installation
Clone the repository including its submodules:

    git clone --recurse-submodules https://github.com/open-energy-transition/efuels-supply-potentials.git

Install the necessary dependencies using `conda` or `mamba` based on the Operating System (OS) of the machine:

    mamba env create -f envs/environment.{os}-64-pinned.yaml

* **Note!** Please check for `envs/` directory for names of pinned environment files. Supported OS list are `windows`, `linux` and `macos`.

Activate `pypsa-earth-efuels` environment:

    conda activate pypsa-earth-efuels

* **Note!** At the moment, head of the PyPSA-Earth submodule points to latest stable commit (`0fa2e39`) of `efuels-supply-potential` branch of [open-energy-transition/pypsa-earth](https://github.com/open-energy-transition/pypsa-earth/tree/efuels-supply-potentials) repository. If OS-specific installation of conda environment does not succeed, it is recommended to install general pypsa-earth environment and activate as ` mamba env create -f submodules/pypsa-earth/envs/environment.yaml` and activate by `conda activate pypsa-earth`. The detailed instructions are provided in [PyPSA-Earth Documentation](https://pypsa-earth.readthedocs.io/en/latest/installation.html).

## 2. Running scenarios

This project utilizes [`snakemake`](https://snakemake.readthedocs.io/en/stable/) to automate the execution of scripts, ensuring efficient and reproducible workflows. Configuration settings for *snakemake* are available in the `configs/config.main.yaml` file as well as scenario-specific configuration files located in `configs/`.

### 2.1. Running the base scenario

To run the power model for the base scenario of the U.S., navigate to the working directory (`.../efuels-supply-potentials/`) and use the following command:
```bash
snakemake -call solve_all_networks --configfile configs/calibration/config.base.yaml
```

* **Note!** All following snakemake commands needs to be executed in the working directory (`.../efuels-supply-potentials/`).

To run the sector-coupled model of the base scenario, execute the command:
```bash
snakemake -call solve_sector_networks --configfile configs/calibration/config.base.yaml
```

### 2.2. Running scenarios for future years

#### A. Running an individual horizon

To run the power model for the Reference scenario of the U.S., navigate to the working directory (`.../efuels-supply-potentials/`) and use the following command:
```bash
snakemake -call solve_all_networks --configfile configs/scenarios/config.20**.yaml
```

* **Note!** Configuration files for future years are currently available for 2030, 2035 and 2040 (replace the "**" in the command above with one off the mentioned years).

To run the sector-coupled model for the Reference scenario, execute the command substituting the desired year to "**" in the command below:
```bash
snakemake -call solve_sector_networks --configfile configs/scenarios/config.20**.yaml
```

#### B. Running for multiple horizons

To run the sector-coupled model for the Reference scenario using myopic optimization for the years 2030, 2035, and 2040 consecutively, execute the following:
```bash
snakemake -call solve_sector_networks_myopic --configfile configs/scenarios/config.myopic.yaml
```

## 3. Snakemake rules

|Rule name                |Config file                              |Description        |
|-------------------------|-----------------------------------------|-------------------|
|`validate_all`           |`config.base.yaml`, `config.base_AC.yaml`|Performs country-level validation comparing with EIA and Ember data|
|`statewise_validate_all` |`config.base_AC.yaml`                    |Performs statewise validation comparing with EIA data|
|`get_capacity_factors`   |Any base or scenario config file         |Estimates capacity factors for renewables|
|`process_airport_data`   | -                                       |Performs analysis on passengers and jet fuel consumption data per state and generates plots and table. Also generate custom airport data with state level based demand| 
|`generate_aviation_scenario` |Any base or scenario config file         |Generates aviation demand csv file with different future scenario| 
|`modify_aviation_demand` |Any base or scenario config file         |Switches aviation demand in energy_total to custom demand|
|`preprocess_demand_data` |Any base or scenario config file         |Preprocess utlities demand data into geojson|
|`build_demand_profiles_from_eia` |Any base or scenario config file         |Build custom demand data from eia and bypass build_demand_profiles|
|`set_saf_mandate`        |Any base or scenario config file         |Adds e-kerosene buses to enable split of aviation demand and sets SAF mandate if enabled|
|`build_custom_industry_demand` |Any base or scenario config file   |Estimates node level demands for selected custom industries (e.g. ammonia, ethanol, cement, and steel)|
|`add_custom_industry`    |Any base or scenario config file         |Adds selected custom industries into the network|
|`prepare_growth_rate_scenarios`  | Any base or scenario config file          | Allows automatic fetching of correct growth rate files according to the demand_projection scenario name                                                                |
|`solve_custom_sector_network`  | Any base or scenario config file          | Allows state/country-wise clean/RES polices to be applied as constraints. The constraints is turned on by default.                                                                |


### Retrieve rules
|Rule name                |Config file                              |Description        |
|-------------------------|-----------------------------------------|-------------------|
|`retrieve_cutouts`       |Any base or scenario config file         |Retrieves US cutouts from google drive|
|`retrieve_osm_raw`       |Any base or scenario config file         |Retrieves `resources/{RDIR}/osm/raw/` data from google drive and bypasses `download_osm_data` rule|
|`retrieve_osm_clean`     |Any base or scenario config file         |Retrieves `resources/{RDIR}/osm/clean/` data from google drive and bypasses `clean_osm_data` rule|
|`retrieve_shapes`        |Any base or scenario config file         |Retrieves `resources/{RDIR}/shapes/` data from google drive and bypasses `build_shapes` rule|
|`retrieve_osm_network`   |Any base or scenario config file         |Retrieves `resources/{RDIR}/base_network/` data from google drive and bypasses `build_osm_network` rule|
|`retrieve_base_network`  |Any base or scenario config file         |Retrieves `base.nc` data from google drive and bypasses `base_network` rule|
|`retrieve_renewable_profiles`  |Any base or scenario config file         |Retrieves `resources/{RDIR}/renewable_profiles/` data from google drive and bypasses `build_renewable_profiles` rule|
|`retrieve_custom_powerplants`  |Any base or scenario config file         |Copies `data/custom_powerplants.csv` to `submodules/pypsa-earth/data/` folder|
|`retrieve_ssp2`          |Any base or scenario config file         |Copies `data/NorthAmerica.csv` to `submodules/pypsa-earth/data/ssp2-2.6/.` directory|
|`retrieve_demand_data`          |Any base or scenario config file         |Retrieves utility demand data from google drive to `data/demand_data/*`|

* `RDIR` - scenario folder

## 4. Updates to the working branch of PyPSA-Earth submodule

### 4.1. Cherry-picking
Cherry-picking allows applying specific commits from one branch to another. We cherry-picked the important commits from upstream pypsa-earth to our project branch ([efuels-supply-potentials](https://github.com/open-energy-transition/pypsa-earth/tree/efuels-supply-potentials)). The commits of the following PRs were integrated to project branch:

1.  [PR #1369](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1369): Restore functioning of myopic optimization.
2.  [PR #1372](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1372): Scale temporal loads based on temporal resolution.
3.  [PR #1381](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1381): Remove space in rail transport oil and electricity carriers.
4.  [PR #1400](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1400): Add US-specific demand growth rates and fuel shares (Medium scenario).
5.  [PR #1410](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1410): Fix negative transport demand.
6.  [PR #1401](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1401): Fix H2 pipeline bus names.
7.  [PR #1422](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1422): Fix renamed column in transport related Wikipedia data.
8.  [PR #1428](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1428): Change source for Aluminum production data.
9.  [PR #1465](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1465): Enable power plants filtering using query.
10. [PR #1468](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1468): Include missing efficiency gains and growth rates for other energy use.
11. [PR #1479](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1479): Update link for North America cutout.
12. [PR #1486](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1486): Align `prepare_transport_data_input` with new structure of the Wikipedia page table.

Please review [a short tutorial](https://www.atlassian.com/git/tutorials/cherry-pick) on cherry-picking in Git to get more familiar with procedure.

## 4.2. Direct commits to PyPSA-Earth

1. [PR #32](https://github.com/open-energy-transition/pypsa-earth/pull/34): Disable implicit calculations and assigning of industry demands for steel and cement industries, because they are added explicitly.
2. [PR #36](https://github.com/open-energy-transition/pypsa-earth/pull/36): Introduce custom H2 production technologies.
2. [PR #38](https://github.com/open-energy-transition/pypsa-earth/pull/38): Enable correct functioning of myopic optimization.
3. [PR #40](https://github.com/open-energy-transition/pypsa-earth/pull/40): Adjusts calculation of `no_years` to properly run 2023 scenario.
4. [PR #50](https://github.com/open-energy-transition/pypsa-earth/pull/50): Introduce Universal Currency Conversion to use USD as reference currency.
5. [PR #51](https://github.com/open-energy-transition/pypsa-earth/pull/51): Add US cost configurations and split scenarios per technology group.
6. [PR #56](https://github.com/open-energy-transition/pypsa-earth/pull/56): Introduce currency conversion in `simplify_network`.
7. [PR #57](https://github.com/open-energy-transition/pypsa-earth/pull/57) [PR #58](https://github.com/open-energy-transition/pypsa-earth/pull/58) [PR #59](https://github.com/open-energy-transition/pypsa-earth/pull/59): Fix logic for currency conversion to handle past and future years and introduce clear log and warning messages.
8. [PR #63](https://github.com/open-energy-transition/pypsa-earth/pull/63): Introduce `p_max_pu` for nuclear generators (or links, if necessary) to match base year statistics and to apply it to future years.
9. [PR #65](https://github.com/open-energy-transition/pypsa-earth/pull/65): Remove lignite from default conventional carriers.
10. [PR #69](https://github.com/open-energy-transition/pypsa-earth/pull/69): Extend lifetime of nuclear power plants to 60 years
11. [PR #71](https://github.com/open-energy-transition/pypsa-earth/pull/71): Enable selection of custom busmap.
12. [PR #76](https://github.com/open-energy-transition/pypsa-earth/pull/76): Add functionality to overwrite cost attributes in sector model.
13. [PR #84](https://github.com/open-energy-transition/pypsa-earth/pull/84): Add possibility to overwrite discount rate.
14. [PR #86](https://github.com/open-energy-transition/pypsa-earth/pull/86): Use H2 Store Tank costs without compressor and add lifetime.
15. [PR #89](https://github.com/open-energy-transition/pypsa-earth/pull/89): Align cost conversion with reference year for costs in input files.
15. [PR #91](https://github.com/open-energy-transition/pypsa-earth/pull/91): Include existing batteries from `powerplants.csv`.
 
## 5. Validation

### 5.1. Country-level validation for the base scenario
To run country-level validation of the U.S. for the base scenario, navigate to the working directory (`.../efuels-supply-potentials/`) and use the following command:
```bash
snakemake -call validate_all --configfile configs/calibration/config.base.yaml
```
or base scenario with alternative clustering option (AC):
```bash
snakemake -call validate_all --configfile configs/calibration/config.base_AC.yaml
```
* **Note:** Ensure that `planning_horizon` in `configs/config.main.yaml` corresponds to a horizon of the base scenario. By default, `planning_horizon` is set to 2020, which means that results are benchmarked agains 2020's historical data.

It is possible to run validation by specifying the output file with wildcards:
``` bash
snakemake -call plots/results/US_2023/demand_validation_s_10_ec_lcopt_Co2L-24H.png --configfile configs/calibration/config.base.yaml
```
Validation results are stored in `plots/results/` directory under scenario run name (e.g. `US_2023`).

### 5.2. State-wise validation
To run state-wise validation, run:
```bash
snakemake -call statewise_validate_all --configfile configs/calibration/config.base_AC.yaml
```
