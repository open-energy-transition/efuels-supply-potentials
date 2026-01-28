# Notebooks Directory

This directory contains Jupyter notebooks for analyzing PyPSA-Earth-based scenarios for the **Grid modelling to assess electrofuels supply potential – The impact of electrofuels on the US electricity grid** study.

---

## Downloading Pre-Solved Networks

Download solved PyPSA networks from Google Drive without running the Snakemake workflow:

```bash
# Download single scenario
python download_scenario_networks.py --scenario-id 2

# Download multiple scenarios
python download_scenario_networks.py --scenario-id 1 2 5 10

# Download all scenarios (1-10)
python download_scenario_networks.py --all

# Download specific years only
python download_scenario_networks.py --scenario-id 2 --years 2030 2040

# Force re-download existing files
python download_scenario_networks.py --scenario-id 2 --force
```

Files are saved to `results/scenarios/scenario_XX/`. Existing files are skipped by default.

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
