<!--
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later
-->

# efuels-supply-potentials


## 1. Installation
Clone the repository including its submodules:

    git clone --recurse-submodules https://github.com/open-energy-transition/efuels-supply-potentials.git

Install the necessary dependencies using `conda` or `mamba`:

    mamba env create -f submodules/pypsa-earth/envs/environment.yaml

Activate `pypsa-earth` environment:

    conda activate pypsa-earth

* **Note!** At the moment, head of the PyPSA-Earth submodule points to latest stable commit (`ee14fa5`) of `efuels-supply-potential` branch of [open-energy-transition/pypsa-earth](https://github.com/open-energy-transition/pypsa-earth/tree/efuels-supply-potentials) repository.

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

### 2.2. Country-level validation for the base scenario
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

### 2.3. State-wise validation
To run state-wise validation, run:
```bash
snakemake -call statewise_validate_all --configfile configs/calibration/config.base_AC.yaml
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

## 4. Cherry-picking
Cherry-picking allows applying specific commits from one branch to another. We cherry-picked the important commits from upstream pypsa-earth to our project branch ([efuels-supply-potentials](https://github.com/open-energy-transition/pypsa-earth/tree/efuels-supply-potentials)). The commits of the following PRs were integrated to project branch:

1. [PR #1372](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1372): Scale temporal loads based on temporal resolution.
2. [PR #1381](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1381): Remove space in rail transport oil and electricity carriers.
3. [PR #1400](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1400): Add US-specific demand growth rates and fuel shares (Medium scenario).

Please review [a short tutorial](https://www.atlassian.com/git/tutorials/cherry-pick) on cherry-picking in Git to get more familiar with procedure.
