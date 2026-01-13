# Notebooks Directory

This directory contains Jupyter notebooks for analyzing PyPSA-Earth e-fuels scenarios and validating model results against historical data.

## Directory Structure

```
notebooks/
├── _helpers.py                         # Core analysis functions (~10K lines)
├── plotting.yaml                       # Visualization config (colors, names, categories)
│
├── scenario_analysis_single.ipynb      # Single scenario comprehensive analysis
├── multiple_scenario_analysis.ipynb    # Multi-scenario comparison
├── validation_base_year.ipynb          # Validate 2023 base year vs EIA/Ember
│
├── run_single_scenario.py              # Automate single scenario analysis
├── run_multiple_scenario.py            # Batch process multiple scenarios
│
├── data/                               # Spatial and reference data
│   ├── gadm41_USA_1.json               # US state boundaries (GADM format)
│   ├── needs_grid_regions_aggregated.geojson  # NERC grid region boundaries
│   ├── energy_charge_rate.csv          # Electricity rate data for LCOH calculations
│   ├── EIA_market_module_regions/      # EIA EMM region definitions
│   └── validation_data/                # EIA and Ember reference datasets
│
├── results/                            # Solved PyPSA networks and analysis outputs
│   ├── base_year/                      # 2023 base year networks
│   ├── scenarios/                      # Scenario results (01-10)
│   │   ├── scenario_01/
│   │   ├── scenario_02/
│   │   ├── scenario_03/
│   │   ├── scenario_04/ ... scenario_10/
│   │       ├── *_2030_*.nc             # Planning horizon networks
│   │       ├── *_2035_*.nc
│   │       └── *_2040_*.nc
│   └── tables/                         # Exported result tables (Excel/CSV)
```

## Quick Start

### Prerequisites

1. **Conda environment** (from parent directory):
   ```bash
   mamba env create -f submodules/pypsa-earth/envs/environment.yaml
   conda activate pypsa-earth
   ```

2. **Solved networks**: Ensure networks exist in `results/scenarios/` directory (generated via Snakemake workflow in parent directory)

3. **Spatial data files**: Verify presence in `data/`:
   - `gadm41_USA_1.json` (US state boundaries)
   - `needs_grid_regions_aggregated.geojson` (Grid regions)

### Running Analysis

#### Option 1: Interactive Notebook

Open and run any notebook in Jupyter:
```bash
jupyter notebook scenario_analysis_single.ipynb
```

Edit the scenario folder parameter in the first code cell:
```python
scenario_folder = "scenario_02"  # Change to desired scenario (01-10)
```

#### Option 2: Automated Execution with Papermill

**Single scenario:**
```bash
python run_single_scenario.py
```
Executes `scenario_analysis_single.ipynb` for each scenario, generating parameterized output notebooks:
- `scenario_analysis_single_01.ipynb`
- `scenario_analysis_single_02.ipynb`
- etc.

**Multiple scenarios across planning horizons:**
```bash
python run_multiple_scenario.py
```
Executes `multiple_scenario_analysis.ipynb` for each year, generating:
- `multiple_scenario_analysis_2030.ipynb`
- `multiple_scenario_analysis_2035.ipynb`
- `multiple_scenario_analysis_2040.ipynb`

## Key Files

### Core Modules

| File | Purpose |
|------|---------|
| `_helpers.py` | **Main analysis library** - network loading, data extraction, visualization, regional aggregation, cost calculations |
| `plotting.yaml` | Technology colors, nice names, visualization categories |

### Data Files

| File | Purpose | Used For |
|------|---------|----------|
| `data/gadm41_USA_1.json` | US state boundaries (GADM format) | Regional aggregation by state |
| `data/needs_grid_regions_aggregated.geojson` | NERC grid regions | Regional aggregation by grid region |
| `data/energy_charge_rate.csv` | Electricity rate data by EMM region | LCOH baseload charge calculations |
| `data/validation_data/` | Historical EIA/Ember datasets | Base year validation |
| `data/EIA_market_module_regions/` | EIA EMM region definitions | Regional analysis and mapping |

## Notebook Descriptions

### Primary Analysis Notebooks

**[scenario_analysis_single.ipynb](scenario_analysis_single.ipynb)** - Comprehensive single scenario analysis including:
- System costs (CAPEX/OPEX breakdown with tax credits)
- Generation capacity and energy by technology
- Regional generation maps (state and grid region)
- Transmission expansion analysis
- Emissions analysis
- E-fuels production costs (LCOH, LCOK)
- Aviation demand by state and grid region
- Electricity dispatch and demand profiles
- Marginal price analysis by region

**[multiple_scenario_analysis.ipynb](multiple_scenario_analysis.ipynb)** - Cross-scenario comparison:
- Compare scenarios across planning horizons (2030, 2035, 2040)
- System cost comparisons
- Technology deployment differences
- Capacity factor analysis
- Sensitivity analysis across scenarios

### Validation Notebooks

**[validation_base_year.ipynb](validation_base_year.ipynb)** - Validate 2023 base year model against:
- EIA historical generation by fuel type
- EIA installed capacity by state
- Ember climate data
- Ensure model accuracy before scenario analysis

## Scenario Structure

10 scenarios with varying assumptions:

| # | Description | e-kerosene Mandate | Demand | Electricity Costs | Electrolyzer Costs | Line Volume |
|---|-------------|-------------------|---------|-------------------|-------------------|-------------|
| 1 | No mandate | None | Medium | Moderate + credits | Medium | lv1 |
| 2 | ReFuel EU | ReFuel EU | Medium | Moderate + credits | Medium | lv1 |
| 3 | ReFuel EU+ | ReFuel EU+ | Medium | Moderate + credits | Medium | lv1 |
| 4 | ReFuel EU- | ReFuel EU- | Medium | Moderate + credits | Medium | lv1 |
| 5 | High ambition, no mandate | None | High | Moderate + credits | Medium + credits | lcopt |
| 6 | High ambition + ReFuel EU | ReFuel EU | High | Moderate + credits | Medium + credits | lcopt |
| 7 | Optimistic elec costs | ReFuel EU | Medium | Advanced + credits | Medium | lv1 |
| 8 | Optimistic electrolyzer | ReFuel EU | Medium | Moderate + credits | Low + credits | lv1 |
| 9 | Conservative electrolyzer | ReFuel EU | Medium | Moderate + credits | High + credits | lv1 |
| 10 | Biogenic CO2 only | ReFuel EU | Medium | Moderate + credits | Medium + credits | lv1 |

**Key differences:**
- Scenarios 5-6 use `lcopt` (line volume optimization), others use `lv1`
- Planning horizons: 2030, 2035, 2040

## Network File Naming Convention

```
elec_s_{clusters}_ec_{lv1|lcopt}_{constraint}_{resolution}_{year}_{co2limit}_{policy}_{export}.nc
```

**Example:**
```
elec_s_100_ec_lv1_CCL-3H_3H_2035_0.07_AB_0export.nc
```

- `s_100` - 100 spatial clusters
- `ec` - with energy components
- `lv1` - line volume optimization level 1 (or `lcopt` for optimal)
- `CCL-3H` - constraint type and resolution
- `3H` - 3-hour temporal resolution
- `2035` - planning horizon year
- `0.07` - CO2 limit (fraction of baseline)
- `AB` - policy scenario code
- `0export` - no electricity exports

## Additional Resources

- **Main Project README**: `../README.md` - Installation, Snakemake workflow, scenario configuration
- **PyPSA Documentation**: https://pypsa.readthedocs.io/ - PyPSA framework reference
- **PyPSA-Earth Documentation**: https://pypsa-meets-earth.readthedocs.io/ - PyPSA-Earth workflow documentation
