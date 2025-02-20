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


## 2. Running validation

This project utilizes [`snakemake`](https://snakemake.readthedocs.io/en/stable/) to automate the execution of scripts, ensuring efficient and reproducible workflows. Configuration settings for *snakemake* are available in the `configs/config.main.yaml` file.

### 2.1. Country-level validation
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
snakemake -call plots/results/US_2021/demand_validation_s_10_ec_lcopt_Co2L-24H.png --configfile configs/calibration/config.base.yaml
```
Validation results are stored in `plots/results/` directory under scenario run name (e.g. `US_2021`).
### 2.2. State-wise validation
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
|`test_modify_prenetwork` |Any base or scenario config file         |Example rule that performs modiification of pre-network| 
|`generate_aviation_scenario` |Any base or scenario config file         |Generates aviation demand csv file with different future scenario| 
|`modify_aviation_demand` |Any base or scenario config file         |Switches aviation demand in energy_total to custom demand|
|`preprocess_demand_data` |Any base or scenario config file         |Preprocess utlities data into geojson|
|`build_demand_profiles_from_eia` |Any base or scenario config file         |Build custom demand data from eia and bypass build_demand_profiles|



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
|`retrieve_demand_data`          |Any base or scenario config file         |Retrieves demand data from google drive to `data/demand_data/*`|

* `RDIR` - scenario folder