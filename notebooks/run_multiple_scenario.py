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
    nargs="+",
    required=True,
    help="Temporal resolution(s) to include together (e.g. 1H 3H)",
)
parser.add_argument(
    "--years",
    type=int,
    nargs="+",
    default=[2030, 2035, 2040],
    help="Planning horizons to run",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["all", "each"],
    default="each",
    help="Execution mode: 'all' runs all years together in one notebook execution, 'each' runs each year separately",
)

args = parser.parse_args()

SCENARIO_IDS = [f"{i:02d}" for i in args.scenario_id]
RESOLUTIONS = [res.upper() for res in args.resolution]
YEARS = [str(year) for year in args.years]
MODE = args.mode

INPUT_NOTEBOOK = "multiple_scenario_analysis.ipynb"

if MODE == "all":
    # Run once with all years as a list
    resolution_tag = "-".join(RESOLUTIONS)
    output_nb = f"multiple_scenario_analysis_{resolution_tag}_{'_'.join(YEARS)}.ipynb"

    print(
        f"[run] multiple scenarios {', '.join(SCENARIO_IDS)} "
        f"years={', '.join(YEARS)} "
        f"resolutions={', '.join(RESOLUTIONS)} "
        f"mode=all"
    )

    pm.execute_notebook(
        input_path=INPUT_NOTEBOOK,
        output_path=output_nb,
        parameters={
            "SCENARIO_IDS": SCENARIO_IDS,
            # Backward compatibility with notebooks expecting a single RESOLUTION value.
            "RESOLUTION": RESOLUTIONS[0],
            "RESOLUTIONS": RESOLUTIONS,
            "YEARS": YEARS,
        },
    )
else:  # MODE == "each"
    # Run separately for each year
    for year in YEARS:
        resolution_tag = "-".join(RESOLUTIONS)
        output_nb = f"multiple_scenario_analysis_{resolution_tag}_{year}.ipynb"

        print(
            f"[run] multiple scenarios {', '.join(SCENARIO_IDS)} "
            f"year={year} "
            f"resolutions={', '.join(RESOLUTIONS)} "
            f"mode=each"
        )

        pm.execute_notebook(
            input_path=INPUT_NOTEBOOK,
            output_path=output_nb,
            parameters={
                "SCENARIO_IDS": SCENARIO_IDS,
                # Backward compatibility with notebooks expecting a single RESOLUTION value.
                "RESOLUTION": RESOLUTIONS[0],
                "RESOLUTIONS": RESOLUTIONS,
                "YEARS": [year],
            },
        )
