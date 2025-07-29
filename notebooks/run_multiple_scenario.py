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


chosen_scenarios = [1, 2, 5, 6, 10] 
horizon_list = [2030, 2035, 2040]

for horizon in horizon_list:
    pm.execute_notebook(
            input_path='./multiple_scenario_analysis.ipynb',
            output_path=f'./multiple_scenario_analysis_{horizon}.ipynb',
            parameters={
                'scenario_list': chosen_scenarios,
                'horizon_list': [horizon],
            }
        )

