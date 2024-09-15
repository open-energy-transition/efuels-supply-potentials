# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

RESULTS_DIR = "plots/results"


configfile: "configs/config.yaml"


wildcard_constraints:
    clusters="[0-9]+",
    planning_horizon="[0-9]{4}",


localrules:
    all,
