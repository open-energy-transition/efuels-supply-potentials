# efuels-supply-potentials


## 1. Running validation

This project utilizes [`snakemake`](https://snakemake.readthedocs.io/en/stable/) to automate the execution of scripts, ensuring efficient and reproducible workflows. Configuration settings for *snakemake* are available in the `configs/config.plot.yaml` file.

To run validation for the U.S. for horizon specified in the `config.plot.yaml`, navigate to the working directory (`.../efuels-supply-potentials/`) and use the following command:
```bash
snakemake -call validate_all
```
* **Note:** Ensure that PyPSA-Earth submodule contains solved networks for the countries and horizons specified in the `config.plot.yaml` file.

It is possible to run validation by specifying the output file with wildcards:
``` bash
snakemake -call plots/results/demand_validation_10_US_2020.png
```
