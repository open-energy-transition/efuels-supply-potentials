# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

try:
    import papermill as pm
except ImportError:
    import subprocess
    import sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "papermill"])
    import papermill as pm


"""
TODO: Change directory stucture to account for path of networks
scenario 5 and 6 will have the lcopt wildcard
other scenarios will have lv1
"""

scenario_data = {
    "Scenarios": [
        "Reference: No e-kerosene mandate",
        "Reference: ReFuel EU",
        "Reference: ReFuel EU+",
        "Reference: ReFuel EU-",
        "Reference: High climate ambition & No e-kerosene mandate",
        "Reference: High climate ambition & ReFuel EU",
        "Reference: Optimistic electricity generation costs",
        "Reference: Optimistic electrolyzer costs",
        "Reference: Conservative electrolyzer costs",
        "Reference: Biogenic point-source CO2 only"
    ],
    "Temporal matching": ["Yes"] * 10,
    "Aviation sector - Demand": ["Central"] * 10,
    "e-kerosene blending mandates": [
        "No e-kerosene mandate",
        "ReFuel EU",
        "ReFuel EU +",
        "ReFuel EU -",
        "No e-kerosene mandate",
        "ReFuel EU",
        "ReFuel EU",
        "ReFuel EU",
        "ReFuel EU",
        "ReFuel EU"
    ],
    "Demand projections - Electricity demand + EV share": [
        "Medium", "Medium", "Medium", "Medium",
        "High", "High", "Medium", "Medium", "Medium", "Medium"
    ],
    "Costs - Electricity generation (NREL ATB)": [
        "Moderate + tax credits"] * 6 + ["Advanced + tax credits"] + ["Moderate + tax credits"] * 3,
    "Costs - Electrolysis (ICCT)": [
        "Medium (no tax credits)"] * 4 + ["Medium + Tax credits"] * 3 + ["Low + tax credits", "High + tax credits", "Medium + Tax credits"
                                                                         ],
    "Costs - DAC": ["Medium + Tax credits"] * 10,
    "Costs - Point-source CO2 capture": ["High + Tax credits"] * 10,
    "Supply constraint - CO2 supply": [
        "Biogenic & non-biogenic point sources & DAC"] * 4 +
    ["Biogenic point sources & DAC"] * 2 +
    ["Biogenic & non-biogenic point sources & DAC"] * 3 +
    ["Biogenic point sources & DAC"],
    "Power sector development - Transmission expansion": [
        "No new expansion"] * 4 + ["Optimal transmission expansion"] * 2 + ["No new expansion *"] * 4,
    "State policies for electricity generation": [
        "Current policies"] * 4 + ["Current policies + 90% clean electricity by 2040"] * 2 + ["Current policies"] * 4,
    "Hourly Resolution": [
        "3-hour"] * 10
}

# scenarios_folder = ['scenario_01'] #, 'scenario_02', 'scenario_06', 'scenario_10']  # List of scenarios to analyze
# horizon_list = [2030, 2035, 2040]  # List of horizons to analyze

# Define chosen scenario numbers (1-based)
chosen_scenarios = [1, 2, 5, 6, 10]  # You can modify this list as needed

# Loop through only the chosen scenario indices
for num in chosen_scenarios:
    i = num - 1  # Convert to 0-based index
    scenario_title = scenario_data["Scenarios"][i]

    # Start markdown table for this scenario
    table = f"""## Scenario {num}: {scenario_title}
    \n This notebook reports the results of preliminary runs for the scenario {num} 
    defined in the table [here](https://docs.google.com/document/d/1ssc5ilxEhEYYjFDCo5cIAgP7zSRcO4uVUXjxbyfR88Q/edit?tab=t.0). 
    In this notebook, a single scenario is analyzed. Another notebook will be available for multi-scenario comparison.
    \n"""
    table += "| "" | "" |\n"
    table += "|-----------|-------|\n"

    # Loop through each parameter (excluding 'Scenarios')
    for param, values in scenario_data.items():
        if param != "Scenarios":
            table += f"| {param} | {values[i]} |\n"

    pm.execute_notebook(
        input_path='./scenario_analysis_single.ipynb',
        output_path=f'./scenario_analysis_single_{num:02d}.ipynb',
        parameters={
            'scenario_folder': f"scenario_{num:02d}",
            'scenario_info': table,

        }
    )
