# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import textwrap
import papermill as pm

# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser(
    description="Run scenario_analysis_single.ipynb for selected scenarios"
)
parser.add_argument(
    "--scenario-id",
    type=int,
    nargs="+",
    required=True,
    help="Scenario IDs to run (1–10)",
)
parser.add_argument(
    "--resolution",
    type=str,
    default="3H",
    help="Temporal resolution (e.g. 1H, 3H, 24H, 196H)",
)

args = parser.parse_args()
chosen_scenarios = args.scenario_id
RESOLUTION = args.resolution

# -----------------------------
# Scenario metadata
# -----------------------------
scenario_data = {
    "Hydrogen policy": {
        "Application of 45V pillars": [
            "Yes*",
            "Yes",
            "Yes",
            "Yes",
            "Yes*",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
        ],
    },
    "Aviation sector": {
        "Demand": ["Central"] * 10,
        "e-kerosene mandate": [
            "No",
            "ReFuel EU",
            "ReFuel EU +",
            "ReFuel EU -",
            "No",
            "ReFuel EU",
            "ReFuel EU",
            "ReFuel EU",
            "ReFuel EU",
            "ReFuel EU",
        ],
    },
    "Demand projections": {
        "Electrification": [
            "Medium",
            "Medium",
            "Medium",
            "Medium",
            "High",
            "High",
            "Medium",
            "Medium",
            "Medium",
            "Medium",
        ],
        "Sectoral demand": [
            "Reference",
            "Reference",
            "Reference",
            "Reference",
            "High Economic Growth",
            "High Economic Growth",
            "Reference",
            "Reference",
            "Reference",
            "Reference",
        ],
    },
    "Technology costs": {
        "Electricity generation and storage": [
            "Moderate + tax credits",
            "Moderate + tax credits",
            "Moderate + tax credits",
            "Moderate + tax credits",
            "Moderate + tax credits (IRA 2022)",
            "Moderate + tax credits (IRA 2022)",
            "Advanced + tax credits",
            "Moderate + tax credits",
            "Moderate + tax credits",
            "Moderate + tax credits",
        ],
        "Electrolysis": [
            "Medium",
            "Medium",
            "Medium",
            "Medium",
            "Medium + tax credits (IRA 2022)",
            "Medium + tax credits (IRA 2022)",
            "Medium",
            "Low",
            "High",
            "Medium",
        ],
        "DAC": ["Medium + tax credits"] * 10,
        "Point-source CO2 capture": ["High + tax credits"] * 10,
    },
    "CO2 supply constraint": {
        "CO2 supply": [
            "All point sources & DAC",
            "All point sources & DAC",
            "All point sources & DAC",
            "All point sources & DAC",
            "Biogenic point sources & DAC",
            "Biogenic point sources & DAC",
            "All point sources & DAC",
            "All point sources & DAC",
            "All point sources & DAC",
            "Biogenic point sources & DAC",
        ],
    },
    "Power sector development": {
        "Transmission capacity expansion": [
            "No new expansion",
            "No new expansion",
            "No new expansion",
            "No new expansion",
            "Cost-optimal",
            "Cost-optimal",
            "No new expansion",
            "No new expansion",
            "No new expansion",
            "No new expansion",
        ],
        "Policies for electricity generation": [
            "Current State policies",
            "Current State policies",
            "Current State policies",
            "Current State policies",
            "Current State policies + 90% clean electricity by 2040",
            "Current State policies + 90% clean electricity by 2040",
            "Current State policies",
            "Current State policies",
            "Current State policies",
            "Current State policies",
        ],
    },
}


# -----------------------------
# Helpers
# -----------------------------
def build_scenario_info(scenario_id: int) -> str:
    i = scenario_id - 1

    scenario_name = {
        1: "Reference - No e-kerosene mandate",
        2: "Reference - ReFuel EU",
        3: "Reference - ReFuel EU+",
        4: "Reference - ReFuel EU-",
        5: "Sensitivity - High climate ambition & No e-kerosene mandate",
        6: "Sensitivity - High climate ambition & ReFuel EU",
        7: "Sensitivity - Optimistic electricity generation costs",
        8: "Sensitivity - Optimistic electrolyzer costs",
        9: "Sensitivity - Conservative electrolyzer costs",
        10: "Sensitivity - Biogenic point-source CO2 only",
    }[scenario_id]

    md = f"""
# Grid modelling to assess electrofuels supply potential
## The impact of electrofuels on the US electricity grid

### Scenario {scenario_id}: {scenario_name}

| **Category** | **Item** | **Value** |
|-------------|----------|-----------|
"""

    for category, items in scenario_data.items():
        for item, values in items.items():
            md += f"| **{category}** | **{item}** | {values[i]} |\n"

    return md


# -----------------------------
# Execution (Papermill)
# -----------------------------
INPUT_NOTEBOOK = "scenario_analysis_single.ipynb"

for num in chosen_scenarios:
    if not (1 <= num <= 10):
        raise ValueError(f"Invalid scenario ID: {num}")

    output_nb = f"scenario_{num:02d}_{RESOLUTION}.ipynb"

    print(f"[run] scenario_{num:02d}_{RESOLUTION}")

    pm.execute_notebook(
        input_path=INPUT_NOTEBOOK,
        output_path=output_nb,
        parameters={
            "SCENARIO_ID": f"{num:02d}",
            "RESOLUTION": RESOLUTION,
            "SCENARIO_INFO": build_scenario_info(num),
        },
    )
