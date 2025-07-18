import papermill as pm

scenarios_folder = ['scenario_01', 'scenario_02', 'scenario_06', 'scenario_10']  # List of scenarios to analyze
horizon_list = [2030, 2035, 2040]  # List of horizons to analyze

for i, scenario in enumerate(scenarios_folder):
    output_str = "_".join(scenario.split('_')[1:])
    print(f"Running scenario {i+1} of {len(scenarios_folder)}: {scenario}")
    pm.execute_notebook(
        input_path='./scenario_analysis_single.ipynb',
        output_path=f'./scenario_analysis_single_{output_str}.ipynb',
        parameters={'scenario_folder': scenario}
    ) 