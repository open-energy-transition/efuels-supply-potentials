# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pypsa
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from scripts._helper import mock_snakemake, update_config_from_wildcards, \
    load_pypsa_network, DATA_DIR
    
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)


#################### FUNCTIONS ####################

def rename_carrier(x):
    if x == 'ccgt':
        return 'CCGT'
    elif x == 'phs':
        return 'PHS'
    else:
        return x


def preprocess_eia_capacity(path, year):
    rename_cols = {
        "Year": "year",
        "State Code": "state",
        "Fuel Source": "carrier",
        "Nameplate Capacity (Megawatts)": "installed_capacity"
    }

    carrier_dict = {
        "Hydroelectric": "hydro",
        "Pumped Storage": "PHS",
        "Solar Thermal and Photovoltaic": "solar",
        "Natural Gas": "gas",
        "Petroleum": "oil",
        "Wind": "wind",
        "Nuclear": "nuclear",
        "Geothermal": "geothermal",
        "Pumped Storage": "PHS",
        "Wood and Wood Derived Fuels": "biomass"}

    req_col = ['nuclear', 'coal', 'gas',
               'wind', 'solar', 'geothermal',
               'oil', 'biomass', 'hydro', 'PHS']

    eia_cap = pd.read_excel(path, skiprows=1)

    eia_cap_year = eia_cap.loc[eia_cap["Year"] == year]
    eia_cap_year = eia_cap_year[[
        "State Code", "Fuel Source", "Nameplate Capacity (Megawatts)", "Producer Type"]]

    eia_cap_year = eia_cap_year.rename(columns=rename_cols)
    eia_cap_year = eia_cap_year.replace({"carrier": carrier_dict})
    eia_cap_year = eia_cap_year.loc[eia_cap_year["Producer Type"]
                                    == 'Total Electric Power Industry']
    eia_cap_year = eia_cap_year.loc[eia_cap_year['state'] != 'US']

    eia_cap_year_grouped = eia_cap_year.groupby(
        ["carrier", "state"]).installed_capacity.sum()
    eia_cap_year_grouped = eia_cap_year_grouped.unstack(level=1, fill_value=0)
    eia_cap_year_grouped.index = eia_cap_year_grouped.index.str.lower()

    eia_cap_year_grouped.index = eia_cap_year_grouped.index.map(rename_carrier)
    eia_cap_year_grouped = eia_cap_year_grouped.T

    eia_cap_year_grouped = eia_cap_year_grouped.drop(
        ["all sources"], axis=1) / 1e3
    eia_cap_year_grouped = eia_cap_year_grouped[req_col]
    eia_cap_year_grouped.index.name = "State"
    return eia_cap_year_grouped


def load_pypsa_network(is_alternative_clustering, network_path):
    if is_alternative_clustering:
        network = pypsa.Network(network_path)
    else:
        mapped_network = os.path.join(DATA_DIR, "validation", "usa_mapped.nc")
        network = pypsa.Network(mapped_network)
    return network


def preprocess_pypsa_cap(network, state_mapping):

    req_col = ['nuclear', 'coal', 'gas',
               'wind', 'solar', 'geothermal',
               'oil', 'biomass', 'hydro', 'PHS']

    series_gen_to_use = network.generators.groupby(
        ["carrier", "bus"]).p_nom.sum()
    series_sto_to_use = network.storage_units.groupby(
        ["carrier", "bus"]).p_nom.sum()
    series_to_use = series_gen_to_use._append(series_sto_to_use)
    df = series_to_use.unstack(level=0, fill_value=0)
    df.index = df.index.str[:-3]

    # map state names to buses
    pypsa_df = df.rename(index=state_mapping)
    # group duplicate entries
    pypsa_df = pypsa_df.groupby(pypsa_df.index).sum()
    # convert from MW to GW
    pypsa_df = pypsa_df / 1e3
    pypsa_df.sort_index(inplace=True)

    wind_cols = ["onwind", "offwind-ac", "offwind-dc"]
    pypsa_df["wind"] = pypsa_df[wind_cols].sum(axis=1)

    gas_cols = ["CCGT", "OCGT"]
    pypsa_df["gas"] = pypsa_df[gas_cols].sum(axis=1)

    pypsa_df.drop(wind_cols + gas_cols, axis=1, inplace=True)
    pypsa_df.drop(["csp"], axis=1, inplace=True)

    # fill empty states with 0
    all_states = sorted(state_mapping.values())
    pypsa_df = pypsa_df.reindex(all_states, fill_value=0)
    pypsa_df.index.name = "State"
    return pypsa_df[req_col]


def get_state_mapping(path):
    gadm_state = gpd.read_file(path)
    gadm_state = gadm_state[["GID_1", "ISO_1"]]
    gadm_state.loc[:, "state"] = gadm_state["ISO_1"].str[-2:]
    gadm_state["GID_1_new"] = gadm_state["GID_1"].str.replace("USA", "US")
    gadm_state = gadm_state[["GID_1_new", "state"]]
    gadm_state = gadm_state.set_index("GID_1_new")["state"].to_dict()
    return gadm_state


def preprocess_pypsa_demand(network, state_mapping):
    # calculate statewise demand
    df = network.loads_t.p_set.multiply(network.snapshot_weightings.objective, axis=0).sum() / 1e6

    df.index = df.index.str.replace("_AC", "")
    df.index = df.index.str.replace("_DC", "")
    df.index = df.index.map(state_mapping)

    df = df.groupby(df.index).sum().to_frame()
    df.index.name = 'State'
    df.columns = ["PyPSA"]
    return df


def preprocess_eia_demand(path):
    statewise_df = pd.read_excel(path, sheet_name="Data")

    demand_df = statewise_df.loc[statewise_df['MSN'] == 'ESTXP']
    demand_df.set_index('State', inplace=True)

    # data is in million kWh (GWh) - hence dividing by 1e3 to get the data in TWh
    demand_df_2021 = demand_df[2021] / 1e3
    demand_df_2021 = demand_df_2021.to_frame()
    demand_df_2021.columns = ["EIA"]

    demand_df_2021.drop(["US"], axis=0, inplace=True)
    return demand_df_2021


def plot_installed_capacities(df, name, color):
    col_idx = df.columns
    fig1, ax1 = plt.subplots(figsize=(16,6))
    df.plot(kind='bar', stacked=True, ax=ax1, color=[color[c] for c in col_idx])
    ax1.set_ylabel('Installed capacity (GW)')
    # Calculate the total height for each bar and annotate at the top
    bar_totals = df.sum(axis=1)  # Sum across columns for each row

    # Adding the annotation at the top of each bar
    for idx, total in enumerate(bar_totals):
        ax1.text(idx, total + 0.5, f'{total:.1f}', ha='center', va='bottom', fontsize=8)

    # reverse the handles and add legend
    handles, labels = ax1.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    ax1.legend(handles, labels, ncol=1, loc="upper right")

    # Save the figure
    if name == "PyPSA":
        ax1.set_title("PyPSA")
        fig1.savefig(snakemake.output.statewise_installed_capacity_pypsa, dpi=plot_scale * 100, bbox_inches = 'tight')
    elif name == "EIA":
        ax1.set_title("EIA")
        fig1.savefig(snakemake.output.statewise_installed_capacity_eia, dpi=plot_scale * 100, bbox_inches = 'tight')


def plot_demand(df):
    fig1, ax1 = plt.subplots(figsize=(16,6))
    df.plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Demand (TWh)')

    # Save the figure
    fig1.savefig(snakemake.output.demand_statewise_comparison, dpi=plot_scale * 100, bbox_inches = 'tight')


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "statewise_validate",
            configfile="configs/calibration/config.base_AC.yaml",
            simpl="",
            ll="copt",
            opts="Co2L-24H",
            clusters="10",
        )

    # update config based on wildcards
    config = update_config_from_wildcards(
        snakemake.config, snakemake.wildcards)
    
    plot_colors = snakemake.params.plots_config["tech_colors"]

    # get planning horizon
    data_horizon = snakemake.params.planning_horizon[0]

    ################ PATHS ################
    eia_statewise_demand_path = os.path.join(
        DATA_DIR, "validation", "EIA_statewise_data", "use_all_phy.xlsx")
    eia_installed_capacity_by_state_path = os.path.join(
        DATA_DIR, "validation", "existcapacity_annual.xlsx")
    gadm_usa_json_path = os.path.join(DATA_DIR, "validation", "gadm41_USA_1.json")

    plot_scale = 1.5

    # get bus to state name mapping
    state_mapping = get_state_mapping(gadm_usa_json_path)

    # load solved network
    network = load_pypsa_network(snakemake.params.alternative_clustering, 
                                 snakemake.input.solved_network)

    # get installed capacities from EIA dataset
    eia_installed_capacity_by_state = preprocess_eia_capacity(
        eia_installed_capacity_by_state_path, data_horizon)
    
    # get installed capacities from PyPSA model
    pypsa_installed_capacity_by_state = preprocess_pypsa_cap(network, state_mapping)

    # get statewise demand from EIA dataset
    eia_statewise_demand = preprocess_eia_demand(eia_statewise_demand_path)

    # get statewise demand from PyPSA model
    pypsa_statewise_demand = preprocess_pypsa_demand(network, state_mapping)

    # concatenate EIA and PyPSA model's statewise demand data
    demand_total = pd.concat([pypsa_statewise_demand, eia_statewise_demand], axis=1)

    ################ PLOTS ################
    # plot installed capacities
    plot_installed_capacities(pypsa_installed_capacity_by_state, "PyPSA", plot_colors)
    plot_installed_capacities(eia_installed_capacity_by_state, "EIA", plot_colors)

    # plot demands
    plot_demand(demand_total)

    ################ TABLE ################
    pypsa_installed_capacity_by_state.index.name = "capacities [GW]"
    pypsa_installed_capacity_by_state.to_csv(snakemake.output.table_statewise_installed_capacity_pypsa)
    eia_installed_capacity_by_state.index.name = "capacities [GW]"
    eia_installed_capacity_by_state.to_csv(snakemake.output.table_statewise_installed_capacity_eia)
    demand_total.index.name = "demand [TWh]"
    demand_total.to_csv(snakemake.output.table_demand_statewise_comparison)
