# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import sys

try:
    import papermill as pm
except ImportError:
    sys.exit(
        "papermill is required for multiple-scenario execution. "
        "Install it in the pypsa-earth environment."
    )


# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser(
    description="Run multiple_scenario_analysis.ipynb for selected scenarios"
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
    default="3H",
    help="Temporal resolution (e.g. 1H, 3H, 24H, 196H)",
)

args = parser.parse_args()

SCENARIO_IDS = [f"{i:02d}" for i in args.scenario_id]
RESOLUTION = args.resolution


# -----------------------------
# Execution
# -----------------------------
INPUT_NOTEBOOK = "multiple_scenario_analysis.ipynb"
OUTPUT_NOTEBOOK = f"multiple_scenario_analysis_{RESOLUTION}.ipynb"

print(
    f"[run] multiple scenarios {', '.join(SCENARIO_IDS)} "
    f"| resolution={RESOLUTION}"
)

pm.execute_notebook(
    input_path=INPUT_NOTEBOOK,
    output_path=OUTPUT_NOTEBOOK,
    parameters={
        "SCENARIO_IDS": SCENARIO_IDS,
        "RESOLUTION": RESOLUTION,
    },
)
