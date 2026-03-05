# SPDX-FileCopyrightText: Open Energy Transition gGmbH
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import papermill as pm

parser = argparse.ArgumentParser(
    description="Run validation_base_year.ipynb to validate base year (2023) model results against EIA and Ember data"
)
parser.add_argument(
    "--resolution",
    type=str,
    default="3H",
    help="Temporal resolution of the base year network (e.g. 1H, 3H, 24H, 196H)",
)

args = parser.parse_args()

RESOLUTION = args.resolution

INPUT_NOTEBOOK = "validation_base_year.ipynb"
OUTPUT_NOTEBOOK = f"validation_base_year_{RESOLUTION}.ipynb"

print(f"[run] base year validation resolution={RESOLUTION}")

pm.execute_notebook(
    input_path=INPUT_NOTEBOOK,
    output_path=OUTPUT_NOTEBOOK,
    parameters={
        "RESOLUTION": RESOLUTION,
    },
)
