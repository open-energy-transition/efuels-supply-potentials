# SPDX-FileCopyrightText: Open Energy Transition gGmbH
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import papermill as pm

parser = argparse.ArgumentParser(
    description="Run multiple_scenario_analysis.ipynb for selected scenarios and years"
)
parser.add_argument(
    "--scenario-id",
    type=int,
    nargs="+",
    required=True,
    help="Scenario IDs to include (1–10)",
)
parser.add_argument(
    "--resolution",
    type=str,
    required=True,
    help="Temporal resolution (e.g. 1H, 3H, 24H, 196H)",
)
parser.add_argument(
    "--years",
    type=int,
    nargs="+",
    default=[2030, 2035, 2040],
    help="Planning horizons to run",
)

args = parser.parse_args()

SCENARIO_IDS = [f"{i:02d}" for i in args.scenario_id]
RESOLUTION = args.resolution
YEARS = args.years

INPUT_NOTEBOOK = "multiple_scenario_analysis.ipynb"

for year in YEARS:
    output_nb = f"multiple_scenario_analysis_{year}_{RESOLUTION}.ipynb"

    print(
        f"[run] multiple scenarios {', '.join(SCENARIO_IDS)} "
        f"| year={year} | resolution={RESOLUTION}"
    )

    pm.execute_notebook(
        input_path=INPUT_NOTEBOOK,
        output_path=output_nb,
        parameters={
            "SCENARIO_IDS": SCENARIO_IDS,
            "RESOLUTION": RESOLUTION,
            "YEAR": year,
        },
    )

