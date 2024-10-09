# efuels-supply-potentials


## 1. Running validation

This project utilizes [`snakemake`](https://snakemake.readthedocs.io/en/stable/) to automate the execution of scripts, ensuring efficient and reproducible workflows. Configuration settings for *snakemake* are available in the `configs/config.main.yaml` file.

### 1.1. Country-level validation
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
### 1.2. State-wise validation
To run state-wise validation, run:
```bash
snakemake -call statewise_validate_all --configfile configs/calibration/config.base_AC.yaml
```
