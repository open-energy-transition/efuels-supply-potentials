# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
import logging
import pycountry
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from scripts._helper import mock_snakemake, update_config_from_wildcards, build_directory, \
                            load_pypsa_network, PLOTS_DIR, DATA_DIR
warnings.filterwarnings("ignore")


# TODO
# Create another plot to compare generation, capacity in details.


def load_ember_data():
    """
    Load the Ember data from a CSV file located in the data folder.

    Returns:
        pd.DataFrame: Ember data in long format.
    """
    ember_data_path = os.path.join(
        DATA_DIR, "validation/ember_yearly_full_release_long_format.csv")
    ember_data = pd.read_csv(ember_data_path)
    return ember_data


def convert_two_country_code_to_three(country_code):
    """
    Convert a two-letter country code to a three-letter ISO country code.

    Args:
        country_code (str): Two-letter country code (ISO 3166-1 alpha-2).

    Returns:
        str: Three-letter country code (ISO 3166-1 alpha-3).
    """
    country = pycountry.countries.get(alpha_2=country_code)
    return country.alpha_3


def get_demand_ember(data, country_code, year):
    """
    Get the electricity demand for a given country and year from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        country_code (str): Country code (ISO 3166-1 alpha-2).
        year (int): Year of interest.

    Returns:
        float or None: Electricity demand if found, otherwise None.
    """
    demand = data[(data["Year"] == year)
                  & (data["Country code"] == country_code)
                  & (data["Category"] == "Electricity demand")
                  & (data["Subcategory"] == "Demand")]["Value"]

    if len(demand) != 0:
        return demand.iloc[0]
    return None


def get_demand_pypsa(network):
    """
    Get the total electricity demand from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        float: Total electricity demand in TWh.
    """
    demand_pypsa = network.loads_t.p_set.multiply(
        network.snapshot_weightings.objective, axis=0).sum().sum() / 1e6
    demand_pypsa = demand_pypsa.round(4)
    return demand_pypsa


def get_installed_capacity_ember(data, three_country_code, year):
    """
    Get installed capacity by fuel type for a given country and year from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        three_country_code (str): Country code (ISO 3166-1 alpha-3).
        year (int): Year of interest.

    Returns:
        pd.DataFrame: Installed capacity by fuel type.
    """
    capacity_ember = data[
        (data["Country code"] == three_country_code)
        & (data["Year"] == year)
        & (data["Category"] == "Capacity")
        & (data["Subcategory"] == "Fuel")][["Variable", "Value"]].reset_index(drop=True)

    # Drop irrelevant rows
    drop_row = ["Other Renewables"]
    capacity_ember = capacity_ember[~capacity_ember["Variable"].isin(drop_row)]

    # Standardize fuel types
    capacity_ember = capacity_ember.replace({
        "Gas": "Fossil fuels",
        "Bioenergy": "Biomass",
        "Coal": "Fossil fuels",
        "Other Fossil": "Fossil fuels"})

    capacity_ember = capacity_ember.groupby("Variable").sum()
    capacity_ember.columns = ["Ember data"]

    return capacity_ember


def get_installed_capacity_pypsa(network):
    """
    Get installed capacity by fuel type from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        pd.DataFrame: Installed capacity by fuel type.
    """
    gen_capacities = network.generators.groupby("carrier").p_nom.sum()
    storage_capacities = network.storage_units.groupby("carrier").p_nom.sum()
    capacity_pypsa = (
        pd.concat([gen_capacities, storage_capacities], axis=0) / 1e3).round(2)

    # Define all possible carriers
    all_carriers = ["nuclear", "coal", "lignite", "CCGT",
                    "OCGT", "hydro", "ror", "PHS", "solar", "offwind-ac",
                    "offwind-dc", "onwind", "biomass"]

    # Reindex to include missing carriers
    capacity_pypsa = capacity_pypsa.reindex(all_carriers, fill_value=0)

    # Rename fuel types to match convention
    capacity_pypsa.rename(index={
        "nuclear": "Nuclear",
        "solar": "Solar",
        "biomass": "Biomass"}, inplace=True)

    # Aggregate fossil fuel and hydro capacities
    capacity_pypsa["Fossil fuels"] = capacity_pypsa[[
        "coal", "lignite", "CCGT", "OCGT"]].sum()
    capacity_pypsa["Hydro"] = capacity_pypsa[["hydro", "ror"]].sum()
    capacity_pypsa["Wind"] = capacity_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum()

    # Filter and reformat
    capacity_pypsa = capacity_pypsa.loc[["Nuclear", "Fossil fuels", "Hydro",
                                         "PHS", "Solar", "Wind", "Biomass"]]
    capacity_pypsa.name = "PyPSA data"
    capacity_pypsa = capacity_pypsa.to_frame()

    return capacity_pypsa


def get_generation_capacity_ember(data, three_country_code, year):
    """
    Get electricity generation by fuel type for a given country and year from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        three_country_code (str): Country code (ISO 3166-1 alpha-3).
        year (int): Year of interest.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    generation_ember = data[
        (data["Category"] == "Electricity generation")
        & (data["Country code"] == three_country_code)
        & (data["Year"] == year)
        & (data["Subcategory"] == "Fuel")
        & (data["Unit"] == "TWh")
    ][["Variable", "Value"]].reset_index(drop=True)

    # Drop irrelevant rows
    drop_row = ["Other Renewables"]
    generation_ember = generation_ember[~generation_ember["Variable"].isin(
        drop_row)]

    # Standardize fuel types
    generation_ember = generation_ember.replace({
        "Gas": "Fossil fuels",
        "Bioenergy": "Biomass",
        "Coal": "Fossil fuels",
        "Other Fossil": "Fossil fuels"})

    # Group by fuel type
    generation_ember = generation_ember.groupby("Variable").sum()
    generation_ember.loc["Load shedding"] = 0.0
    generation_ember.columns = ["Ember data"]

    return generation_ember


def get_generation_capacity_pypsa(network):
    """
    Get electricity generation by fuel type from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    gen_capacities = (network.generators_t
                      .p.multiply(network.snapshot_weightings.objective, axis=0)
                      .groupby(network.generators.carrier, axis=1).sum().sum())

    storage_capacities = (network.storage_units_t
                          .p.multiply(network.snapshot_weightings.objective, axis=0)
                          .groupby(network.storage_units.carrier, axis=1).sum().sum())

    # Combine generator and storage generation capacities
    generation_pypsa = (
        (pd.concat([gen_capacities, storage_capacities], axis=0)) / 1e6).round(2)

    # Define all possible carriers
    all_carriers = ["nuclear", "coal", "lignite", "CCGT", "OCGT",
                    "hydro", "ror", "PHS", "solar", "offwind-ac",
                    "offwind-dc", "onwind", "biomass", "load"]

    # Reindex to include missing carriers
    generation_pypsa = generation_pypsa.reindex(all_carriers, fill_value=0)

    # Rename fuel types to match convention
    generation_pypsa.rename(index={
        "nuclear": "Nuclear",
        "solar": "Solar",
        "biomass": "Biomass",
        "load": "Load shedding"}, inplace=True)

    # Aggregate fossil fuel, hydro, and wind generation
    generation_pypsa["Fossil fuels"] = generation_pypsa[[
        "CCGT", "OCGT", "coal", "lignite"]].sum()
    generation_pypsa["Hydro"] = generation_pypsa[["hydro", "ror"]].sum()
    generation_pypsa["Wind"] = generation_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum()

    # Adjust load shedding value
    generation_pypsa["Load shedding"] /= 1e3

    # Filter and reformat
    generation_pypsa = generation_pypsa.loc[["Nuclear", "Fossil fuels", "PHS",
                                             "Hydro", "Solar", "Wind", "Biomass",
                                             "Load shedding"]]
    generation_pypsa.name = "PyPSA data"
    generation_pypsa = generation_pypsa.to_frame()

    return generation_pypsa


def get_country_name(country_code):
    """ Input:
            country_code - two letter code of the country
        Output:
            country.name - corresponding name of the country
            country.alpha_3 - three letter code of the country
    """
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name, country.alpha_3 if country else None
    except KeyError:
        return None


def get_data_EIA(data_path, country_code, year):
    """
    Retrieves energy generation data from the EIA dataset for a specified country and year.

    Args:
        data_path (str): Path to the EIA CSV file.
        country_code (str): Two-letter or three-letter country code (ISO).
        year (int or str): Year for which energy data is requested.

    Returns:
        pd.DataFrame: DataFrame containing energy generation data for the given country and year, 
                    or None if no matching country is found.
    """

    # Load EIA data from CSV file
    data = pd.read_csv(data_path)

    # Rename the second column to 'country' for consistency
    data.rename(columns={"Unnamed: 1": "country"}, inplace=True)

    # Remove leading and trailing spaces in the 'country' column
    data["country"] = data["country"].str.strip()

    # Extract the three-letter country code from the 'API' column
    data["code_3"] = data.dropna(subset=["API"])["API"].apply(
        lambda x: x.split('-')[2] if isinstance(x,
                                                str) and len(x.split('-')) > 3 else x
    )

    # Get the official country name and three-letter country code using the provided two-letter code
    country_name, country_code3 = get_country_name(country_code)

    # Check if the three-letter country code exists in the dataset
    if country_code3 and country_code3 in data.code_3.unique():
        # Retrieve the generation data for the specified year
        result = data.query("code_3 == @country_code3")[["country", str(year)]]

    # If not found by code, search by the country name
    elif country_name and country_name in data.country.unique():
        # Find the country index and retrieve generation data
        country_index = data.query("country == @country_name").index[0]
        result = data.iloc[country_index +
                           1:country_index+18][["country", str(year)]]

    else:
        # If no match is found, return None
        result = None

    # Convert the year column to float for numeric operations
    result[str(year)] = result[str(year)].astype(float)

    return result


def preprocess_eia_data(data):
    """
    Preprocesses the EIA energy data by renaming and filtering rows and columns.

    Args:
        data (pd.DataFrame): DataFrame containing EIA energy data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame ready for analysis.
    """

    # Strip the last 13 characters (descriptive text) from the 'country' column
    data["country"] = data["country"].apply(lambda x: x[:-13].strip())

    # Set 'country' as the index of the DataFrame
    data.set_index("country", inplace=True)

    # Rename columns to provide clarity
    data.columns = ["EIA data"]

    # Rename specific rows to match more standard terms
    data.rename(index={"Hydroelectricity": "Hydro",
                       "Biomass and waste": "Biomass",
                       "Hydroelectric pumped storage": "PHS"}, inplace=True)

    # Drop unwanted renewable energy categories
    data.drop(index=["Renewables", "Non-hydroelectric renewables",
                     "Geothermal", "Solar, tide, wave, fuel cell", "Tide and wave"], inplace=True)

    # Filter the DataFrame to only include relevant energy sources
    data = data.loc[["Nuclear", "Fossil fuels",
                     "Hydro", "PHS", "Solar", "Wind", "Biomass"], :]

    return data


def preprocess_eia_data_detail(data):
    """
    Preprocesses the EIA energy data by renaming and filtering rows and columns.

    Args:
        data (pd.DataFrame): DataFrame containing EIA energy data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame ready for analysis.
    """

    # Strip the last 13 characters (descriptive text) from the 'country' column
    data["country"] = data["country"].apply(lambda x: x[:-13].strip())

    # Set 'country' as the index of the DataFrame
    data.set_index("country", inplace=True)

    # Rename columns to provide clarity
    data.columns = ["EIA data"]

    # Rename specific rows to match more standard terms
    data.rename(index={"Hydroelectricity": "Hydro",
                       "Biomass and waste": "Biomass",
                       "Hydroelectric pumped storage": "PHS"}, inplace=True)

    # Drop unwanted renewable energy categories
    data.drop(index=["Fossil fuels", "Renewables", "Non-hydroelectric renewables",
                     "Geothermal", "Solar, tide, wave, fuel cell", "Tide and wave"], inplace=True)

    # Filter the DataFrame to only include relevant energy sources
    data = data.loc[["Nuclear", "Coal", "Natural gas", "Oil",
                     "Hydro", "PHS", "Solar", "Wind", "Biomass", ], :]
    return data


def get_generation_capacity_pypsa_detail(network):
    """
    Get electricity generation by fuel type from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    gen_capacities = (network.generators_t
                      .p.multiply(network.snapshot_weightings.objective, axis=0)
                      .groupby(network.generators.carrier, axis=1).sum().sum())

    storage_capacities = (network.storage_units_t
                          .p.multiply(network.snapshot_weightings.objective, axis=0)
                          .groupby(network.storage_units.carrier, axis=1).sum().sum())

    # Combine generator and storage generation capacities
    generation_pypsa = (
        (pd.concat([gen_capacities, storage_capacities], axis=0)) / 1e6).round(2)

    # Define all possible carriers
    all_carriers = ["nuclear", "coal", "lignite", "CCGT", "OCGT",
                    "hydro", "ror", "PHS", "solar", "offwind-ac",
                    "offwind-dc", "onwind", "biomass", "load"]

    # Reindex to include missing carriers
    generation_pypsa = generation_pypsa.reindex(all_carriers, fill_value=0)

    # Rename fuel types to match convention
    generation_pypsa.rename(index={
        "nuclear": "Nuclear",
        "solar": "Solar",
        "biomass": "Biomass",
        "load": "Load shedding"}, inplace=True)

    # Aggregate fossil fuel, hydro, and wind generation
    generation_pypsa["Natural gas"] = generation_pypsa[["CCGT", "OCGT"]].sum()
    generation_pypsa["Coal"] = generation_pypsa[["coal", "lignite"]].sum()
    generation_pypsa["Hydro"] = generation_pypsa[["hydro", "ror"]].sum()
    generation_pypsa["Wind"] = generation_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum()

    # Adjust load shedding value
    generation_pypsa["Load shedding"] /= 1e3

    # Filter and reformat
    generation_pypsa = generation_pypsa.loc[["Nuclear", "Natural gas", "PHS", "Coal",
                                             "Hydro", "Solar", "Wind", "Biomass",
                                             "Load shedding"]]
    generation_pypsa.name = "PyPSA data"
    generation_pypsa = generation_pypsa.to_frame()
    return generation_pypsa


def get_generation_capacity_ember_detail(data, three_country_code, year):
    """
    Get electricity generation by fuel type for a given country and year from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        three_country_code (str): Country code (ISO 3166-1 alpha-3).
        year (int): Year of interest.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    generation_ember = data[
        (data["Category"] == "Electricity generation")
        & (data["Country code"] == three_country_code)
        & (data["Year"] == year)
        & (data["Subcategory"] == "Fuel")
        & (data["Unit"] == "TWh")
    ][["Variable", "Value"]].reset_index(drop=True)

    # Drop irrelevant rows
    drop_row = ["Other Renewables"]
    generation_ember = generation_ember[~generation_ember["Variable"].isin(
        drop_row)]

    # Standardize fuel types
    generation_ember = generation_ember.replace({
        "Gas": "Natural gas",
        "Bioenergy": "Biomass",
        # "Coal": "Fossil fuels",
        # "Other Fossil": "Fossil fuels"
    })

    # Group by fuel type
    generation_ember = generation_ember.groupby("Variable").sum()
    generation_ember.loc["Load shedding"] = 0.0
    generation_ember.columns = ["Ember data"]

    return generation_ember


def plot_demand_validation(demand_ember, pypsa_demand, EIA_demand, horizon, country_code, clusters):
    plt.figure(figsize=(8, 6))  # Set figure size
    bars = plt.bar(["PyPSA", "Ember",  "EIA"], [pypsa_demand, demand_ember, EIA_demand],
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f"Demand Validation in {country_code}, {horizon}")
    plt.ylabel("Demand (TWh)", fontsize=12)
    plt.xlabel("Data Source", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/demand_validation_{clusters}_{country_code}_{horizon}.png")


def plot_detailed_generation_validation(generation_df_, horizon, country_code, clusters):
    ax = generation_df_.plot(kind="bar", figsize=(12, 7), width=0.8, color=[
                             '#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(
        f"Comparison of Generation: PyPSA vs EMBER vs EIA, {horizon}", fontsize=16)
    plt.xlabel("Generation Type", fontsize=12)
    plt.ylabel("Generation (TWh)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", fontsize=10)
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.legend(["PyPSA", "EMBER", "EIA"], loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/generation_validation_detailed_{clusters}_{country_code}_{horizon}.png")


def plot_capacity_validation(installed_capacity_df, horizon, country_code, clusters):
    ax = installed_capacity_df.plot(kind="bar", figsize=(
        12, 7), width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(
        f"Comparison of Installed Capacity: PyPSA vs EMBER vs EIA, {horizon}", fontsize=16)
    plt.xlabel("Technology", fontsize=12)
    plt.ylabel("Capacity (GW)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", fontsize=10)
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.legend(["PyPSA", "EMBER", "EIA"], loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/capacity_validation_{clusters}_{country_code}_{horizon}.png")


def plot_generation_validation(generation_df, horizon, country_code, clusters):
    ax = generation_df.plot(kind="bar", figsize=(12, 7), width=0.8, color=[
                            '#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(
        f"Comparison of Generation: PyPSA vs EMBER vs EIA, {horizon}", fontsize=16)
    plt.xlabel("Generation Type", fontsize=12)
    plt.ylabel("Generation (TWh)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", fontsize=10)
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.legend(["PyPSA", "EMBER", "EIA"], loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/generation_validation_{clusters}_{country_code}_{horizon}.png")


def save_csv_output(data, output_name, index=False):
    """
    Args:
        data (pd.DataFrame): DataFrame to save as a CSV file.
        output_name (str): Name of the output CSV file.
    returns:
        str: Message indicating the success of the operation
    """
    data.to_csv(f"{PLOTS_DIR}/{output_name}.csv", index=f"{index}")
    return "Data saved successfully."


def run(country_code, scenario_folder, horizon, clusters):
    """
    Run the data validation workflow for the specified country and year.
    """
    # Get EIA data
    EIA_demand_path = os.path.join(
        DATA_DIR, "validation", "EIA_demands.csv")
    EIA_installed_capacities_path = os.path.join(
        DATA_DIR, "validation", "EIA_installed_capacities.csv")
    EIA_generation_path = os.path.join(
        DATA_DIR, "validation", "EIA_electricity_generation.csv")

    # Load Ember data
    ember_data = load_ember_data()

    # Load PyPSA network
    # Note: you can also load the network directly using """network = pypsa.Network(network_path)"""
    ### Fix: let the scenario folder point to the results file in pypsa-earth ###
    network = load_pypsa_network(scenario_folder)

    three_country_code = convert_two_country_code_to_three(country_code)

    # Plots directory
    build_directory(PLOTS_DIR)

    ####### DEMAND #######

    demand_ember = get_demand_ember(ember_data, three_country_code, horizon)
    pypsa_demand = get_demand_pypsa(network)

    EIA_demand = get_data_EIA(EIA_demand_path, country_code, horizon)
    EIA_demand = EIA_demand.iloc[0, 1]

    # Save the output as a CSV file
    demand_df = pd.DataFrame(
        {"PyPSA data": [pypsa_demand], "Ember data": [demand_ember], "EIA data": [EIA_demand]})
    save_csv_output(
        demand_df, f"demand_validation_{country_code}_{horizon}", index=True)

    ####### INSTALLED CAPACITY #######

    installed_capacity_ember = get_installed_capacity_ember(
        ember_data, three_country_code, horizon).round(2)
    pypsa_capacity = get_installed_capacity_pypsa(network).round(2)

    EIA_inst_capacities = get_data_EIA(
        EIA_installed_capacities_path, country_code, horizon)
    EIA_inst_capacities = preprocess_eia_data(EIA_inst_capacities).round(2)

    installed_capacity_df = pd.concat(
        [pypsa_capacity, installed_capacity_ember, EIA_inst_capacities], axis=1).fillna(0)

    # Save the output as a CSV file
    save_csv_output(installed_capacity_df,
                    f"installed_capacity_validation_{country_code}_{horizon}", index=True)

    ####### GENERATION #######

    generation_data_ember = get_generation_capacity_ember(
        ember_data, three_country_code, horizon).round(2)
    pypsa_generation = get_generation_capacity_pypsa(network).round(2)

    EIA_generation = get_data_EIA(EIA_generation_path, country_code, horizon)
    EIA_generation = preprocess_eia_data(EIA_generation).round(2)

    generation_df = pd.concat(
        [pypsa_generation, generation_data_ember, EIA_generation], axis=1).fillna(0)
    generation_df

    # Save the output as a CSV file
    save_csv_output(
        generation_df, f"generation_validation_{country_code}_{horizon}", index=True)

    generation_data_ember_ = get_generation_capacity_ember_detail(
        ember_data, three_country_code, horizon).round(2)
    EIA_generation_ = get_data_EIA(EIA_generation_path, country_code, horizon)
    EIA_generation_ = preprocess_eia_data_detail(EIA_generation_).round(2)

    pypsa_generation_ = get_generation_capacity_pypsa_detail(network).round(2)

    generation_df_ = pd.concat(
        [pypsa_generation_, generation_data_ember_, EIA_generation_], axis=1).fillna(0)

    # Save the output as a CSV file
    save_csv_output(
        generation_df_, f"generation_validation_detail_{country_code}_{horizon}", index=True)

    # Plots
    plot_demand_validation(demand_ember, pypsa_demand,
                           EIA_demand, horizon, country_code, clusters)
    plot_detailed_generation_validation(generation_df_,
                                        horizon, country_code, clusters)
    plot_capacity_validation(installed_capacity_df,
                             horizon, country_code, clusters)
    plot_generation_validation(generation_df, horizon, country_code, clusters)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "validate",
            countries="US",
            clusters="10",
            planning_horizon="2020",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(
        snakemake.config, snakemake.wildcards)

    # country, planning horizon, and number of clusters from config.plot.yaml
    country_code = config["validation"]["countries"]
    planning_horizon = config["validation"]["planning_horizon"]
    clusters = config["validation"]["clusters"]
    scenario_folder = f"{country_code}_{int(planning_horizon)+1}"
    data_horizon = int(planning_horizon)

    # Run the data validation
    run(country_code, scenario_folder, data_horizon, clusters)
    logging.info("Data validation completed successfully.")
