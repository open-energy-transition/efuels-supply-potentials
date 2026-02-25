# Notebooks Directory

This directory contains Jupyter notebooks for analyzing PyPSA-Earth-based scenarios for the **Grid modelling to assess electrofuels supply potential – The impact of electrofuels on the US electricity grid** study and for validating model results against historical data.

Once you have cloned the `efuels-supply-potentials` repository, navigate to the `notebooks` directory.

The workflow supports **parameterized, reproducible scenario analysis** across multiple planning horizons and temporal resolutions.

---

## Directory Structure

```
notebooks/
├── _helpers.py                         # Core analysis functions
├── plotting.yaml                       # Visualization config (colors, names, categories)
│
├── scenario_analysis_single.ipynb      # Single-scenario comprehensive analysis (parameterized)
├── multiple_scenario_analysis.ipynb    # Multi-scenario comparison
├── validation_base_year.ipynb          # Validate 2023 base year vs EIA / Ember data
│
├── run_single_scenario.py              # CLI script to run single scenario analysis
├── run_multiple_scenario.py            # CLI script to run multi-scenario comparison
├── run_validation.py                   # CLI script to run base year validation
│
├── data/                               # Spatial and reference data
│   ├── gadm41_USA_1.json               # US state boundaries (GADM)
│   ├── needs_grid_regions_aggregated.geojson  # NERC grid region boundaries
│   ├── energy_charge_rate.csv          # US-specific Energy charge rate data for LCOH calculations
│   ├── EIA_market_module_regions/      # EIA Electricity Market Modules (EMM) regions definitions
│   └── validation_data/                # EIA and Ember reference datasets
│
├── results/                            # Solved PyPSA networks and analysis outputs
│   ├── base_year/                      # 2023 base year networks
│   ├── scenarios/                      # Scenario results (01–10)
│   │   ├── scenario_01/
│   │   ├── scenario_02/
│   │   ├── scenario_03/
│   │   ├── scenario_04/
│   │   └── scenario_05/ ... scenario_10/
│   │       ├── *_2030_*.nc
│   │       ├── *_2035_*.nc
│   │       └── *_2040_*.nc
│   └── tables/                         # Exported result tables (Excel / CSV)
```

---

## Quick Start


### Prerequisites

1. **Conda environment** (from parent repository):
   ```bash
   mamba env create -f submodules/pypsa-earth/envs/environment.yaml
   conda activate pypsa-earth
   ```

2. **Solved networks**:

Ensure PyPSA networks exist in `results/scenarios/` (generated via the Snakemake workflow in the parent directory).

3. **Spatial data files**:

Verify the presence of:

* `data/gadm41_USA_1.json`
* `data/needs_grid_regions_aggregated.geojson`

---

## Running the Analysis: single scenario notebook

The notebook `scenario_analysis_single.ipynb` performs the **analysis for a single, selected scenario** based on the solved networks available in `results/scenarios/`.

The recommended way to run this notebook is via the provided CLI wrapper
`run_single_scenario.py`, which handles:

- scenario selection
- temporal resolution
- scenario metadata injection
- consistent output notebook naming

---

### Prerequisites

Ensure that:

- solved PyPSA networks exist in `results/scenarios/`
- the `pypsa-earth` conda environment is activated

---

### Running one or more scenarios via CLI (recommended)

The script `run_single_scenario.py` executes
`scenario_analysis_single.ipynb` for one or more scenarios sequentially.

Example: run scenarios **1, 2, 5, 6 and 10** at **3-hour resolution**:

```bash
python run_single_scenario.py \
  --scenario-id 1 2 5 6 10 \
  --resolution 3H
```

This generates one output notebook per scenario:

```
scenario_01_3H.ipynb
scenario_02_3H.ipynb
scenario_05_3H.ipynb
scenario_06_3H.ipynb
scenario_10_3H.ipynb
```

Execution is sequential by design to avoid memory issues.
If required networks for a given scenario and resolution are missing,
the notebook will fail fast.

---

### CLI Parameters

| Parameter        | Description                            | Examples              |
|-----------------|----------------------------------------|-----------------------|
| `--scenario-id` | Scenario IDs to run (1–10)             | `1 2 5 6 10`          |
| `--resolution`  | Temporal resolution of solved networks | `1H`, `3H`, `24H`, `196H` |

---

### Interactive use (development only)

The notebook can still be opened interactively for development or debugging:

```bash
jupyter notebook scenario_analysis_single.ipynb
```

In this case, the parameters at the top of the notebook are used:

```python
# Parameters
SCENARIO_ID = "02"
RESOLUTION = "3H"
```

These values are defaults for interactive use only and are overridden
when running via `run_single_scenario.py`.


## Running the Analysis: multiple scenario notebook

The notebook `multiple_scenario_analysis.ipynb` performs **cross-scenario comparisons** based on the solved PyPSA-Earth networks available in `results/scenarios/`.

Unlike `scenario_analysis_single.ipynb`, this notebook is designed to load **multiple scenarios simultaneously**, aggregate results, and produce comparative analyses across scenarios. Consolidated results are also exported to an Excel file.

The Excel output filename includes a year tag based on the selected planning horizons:

| Year selection | Output Excel file |
|---|---|
| Single year (e.g. 2030) | `efuels_supply_potentials_results_2030.xlsx` |
| Multiple years | `efuels_supply_potentials_results_all_years.xlsx` |

---

### Parameters

The notebook is fully parameterized. Parameters are defined at the top of the notebook and can be overridden when running non-interactively.

```python
# Parameters
SCENARIO_IDS = ["01", "02", "03", "04", "07", "08", "09", "10"]
RESOLUTION = "3H"
```

* `SCENARIO_IDS` selects the scenarios included in the comparison
* `RESOLUTION` selects which solved networks are loaded (e.g. `1H`, `3H`, `24H`, `196H`)

---

### Option 1: Interactive execution (development / debugging)

For exploratory analysis or debugging, the notebook can be executed interactively:

```bash
jupyter notebook multiple_scenario_analysis.ipynb
```

In this case, the parameter values defined directly in the notebook are used.

---

### Option 2: Automated execution via CLI (recommended)

For reproducible and non-interactive execution, use the CLI wrapper `run_multiple_scenario.py`, which handles:

- scenario selection
- temporal resolution
- planning horizon years
- execution mode (all years together or separately)
- consistent output notebook naming

#### Example 1: Run all years together in one notebook

Run scenarios **1, 2, 5, 6 and 10** at **3-hour resolution** for **all years** (2030, 2035, 2040) in a single notebook execution:

```bash
python run_multiple_scenario.py \
  --scenario-id 1 2 5 6 10 \
  --resolution 3H \
  --mode all
```

This generates a single output notebook and a consolidated Excel file:

```
multiple_scenario_analysis_3H_2030_2035_2040.ipynb
efuels_supply_potentials_results_all_years.xlsx
```

#### Example 2: Run each year separately

Run the same scenarios but execute each year in a separate notebook:

```bash
python run_multiple_scenario.py \
  --scenario-id 1 2 5 6 10 \
  --resolution 3H \
  --mode each
```

This generates one output notebook and one Excel file per year:

```
multiple_scenario_analysis_3H_2030.ipynb   efuels_supply_potentials_results_2030.xlsx
multiple_scenario_analysis_3H_2035.ipynb   efuels_supply_potentials_results_2035.xlsx
multiple_scenario_analysis_3H_2040.ipynb   efuels_supply_potentials_results_2040.xlsx
```

#### Example 3: Custom year selection

Run scenarios for specific planning horizons only (e.g., 2030 and 2040):

```bash
python run_multiple_scenario.py \
  --scenario-id 1 2 5 6 10 \
  --resolution 3H \
  --years 2030 2040 \
  --mode each
```

Execution is sequential by design to avoid memory issues.
If required networks for a given scenario, year, or resolution are missing, the notebook will fail fast.

---

### CLI Parameters

| Parameter        | Required | Description                                                  | Examples                    |
|------------------|----------|--------------------------------------------------------------|-----------------------------|
| `--scenario-id`  | Yes      | Scenario IDs to include (1–10)                               | `1 2 5 6 10`                |
| `--resolution`   | Yes      | Temporal resolution of solved networks                       | `1H`, `3H`, `24H`, `196H`   |
| `--years`        | No       | Planning horizons to run (default: 2030 2035 2040)          | `2030 2035 2040` or `2040`  |
| `--mode`         | No       | Execution mode: `all` (single notebook) or `each` (separate notebooks per year), default: `each` | `all` or `each` |

## Notebook Descriptions

### Primary Analysis Notebooks

**[scenario_analysis_single.ipynb](scenario_analysis_single.ipynb)** - Comprehensive single scenario analysis including:
- System costs (CAPEX/OPEX breakdown with analysis on the impact of tax credits)
- Electricity demand breakdown (including data center demand) and profiles
- Power generation capacity and dispatch analysis by technology
- Transmission expansion analysis
- Power generation emissions analysis
- Aviation demand by state and grid region
- Levelized cost analysis for electricity, hydrogen, CO2 and e-kerosene
- Analysis of the industrial sector

**[multiple_scenario_analysis.ipynb](multiple_scenario_analysis.ipynb)** - Cross-scenario comparison:
- Compare the results above for all the selected scenarios

### Validation Notebooks

**[validation_base_year.ipynb](validation_base_year.ipynb)** - Validate 2023 base year model against:
- EIA historical generation by fuel type
- EIA installed capacity by state
- Ember climate data
- Ensure model accuracy before scenario analysis

---

## Running the Validation Notebook

The `validation_base_year.ipynb` notebook validates the base year (2023) model results against EIA and Ember datasets. It compares electricity demand, generation, and installed capacity to ensure model accuracy.

### Prerequisites

Ensure that:
- a solved base year PyPSA network exists in `results/base_year/`
- validation data files exist in `data/validation_data/` (EIA and Ember datasets)
- the `pypsa-earth` conda environment is activated

---

### Running via CLI (recommended)

The script `run_validation.py` executes the validation notebook with parameterized resolution.

Example: run validation for **3-hour resolution** base year network:

```bash
python run_validation.py --resolution 3H
```

This generates an output notebook:

```
validation_base_year_3H.ipynb
```

If the required base year network for the specified resolution is missing, the notebook will fail fast.

---

### CLI Parameters

| Parameter      | Required | Description                                    | Examples                  |
|----------------|----------|------------------------------------------------|---------------------------|
| `--resolution` | No       | Temporal resolution (default: 3H)              | `1H`, `3H`, `24H`, `196H` |

---

### Interactive use (development only)

The notebook can also be opened interactively for development or debugging:

```bash
jupyter notebook validation_base_year.ipynb
```

In this case, the parameter at the top of the notebook is used:

```python
# Parameters
RESOLUTION = "3H"
```

This value is the default for interactive use only and is overridden when running via `run_validation.py`.

---

## Scenario Structure

| #  | Scenario name                                               |
| -- | ----------------------------------------------------------- |
| 1  | Reference - No e-kerosene mandate                           |
| 2  | Reference - ReFuel EU                                       |
| 3  | Reference - ReFuel EU+                                      |
| 4  | Reference - ReFuel EU-                                      |
| 5  | Sensitivity - High climate ambition & No e-kerosene mandate |
| 6  | Sensitivity - High climate ambition & ReFuel EU             |
| 7  | Sensitivity - Optimistic electricity generation costs       |
| 8  | Sensitivity - Optimistic electrolyzer costs                 |
| 9  | Sensitivity - Conservative electrolyzer costs               |
| 10 | Sensitivity - Biogenic point-source CO2 only                |


## Additional Resources

* Main Project README: [ README.md ](../README.md)
* PyPSA documentation: [https://pypsa.readthedocs.io/](https://pypsa.readthedocs.io/)
* PyPSA-Earth documentation: [https://pypsa-meets-earth.readthedocs.io/](https://pypsa-meets-earth.readthedocs.io/)
