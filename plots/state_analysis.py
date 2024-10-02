# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
import logging
import pycountry
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pypsa
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from scripts._helper import mock_snakemake, update_config_from_wildcards, build_directory, \
    load_pypsa_network, PLOTS_DIR, DATA_DIR, PYPSA_EARTH_DIR
    
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

    eia_cap_year = eia_cap.loc[eia_cap["Year"] == 2021]
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
    return eia_cap_year_grouped


def load_pypsa_network(is_alternative_clustering):
    if is_alternative_clustering:
        network_path = os.path.join(PYPSA_EARTH_DIR, "networks",
                                    "US_2021", "elec_s_10.nc")
        network = pypsa.Network(network_path)
    else:
        mapped_network = os.path.join(DATA_DIR, "validation", "usa_mapped.nc")
        network = pypsa.Network(mapped_network)
    return network


def preprocess_pypsa_cap(network, gadm_us):

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

    pypsa_df = df.merge(gadm_us, left_index=True, right_on="GID_1_new")
    pypsa_df.index = pypsa_df["state"]
    pypsa_df = pypsa_df.drop(["GID_1_new", "state"], axis=1) / 1e3
    pypsa_df.sort_index(inplace=True)

    wind_cols = [x for x in pypsa_df.columns.unique() if 'wind' in x]
    pypsa_wind_df = pypsa_df[wind_cols].agg(["sum"], axis=1)
    pypsa_wind_df.rename(columns={"sum": "wind"}, inplace=True)

    gas_cols = ["CCGT", "OCGT"]
    pypsa_gas_df = pypsa_df[gas_cols].agg(["sum"], axis=1)
    pypsa_gas_df.rename(columns={"sum": "gas"}, inplace=True)

    pypsa_merged_df = pd.concat(
        [pypsa_df, pypsa_wind_df, pypsa_gas_df], axis=1)
    pypsa_merged_df.drop(wind_cols + gas_cols, axis=1, inplace=True)
    pypsa_merged_df.drop(["csp"], axis=1, inplace=True)
    return pypsa_merged_df[req_col]


def preprocess_gadm(path):
    gadm_state = gpd.read_file(path)
    gadm_state = gadm_state[["GID_1", "ISO_1"]]
    gadm_state.loc[:, "state"] = gadm_state["ISO_1"].str[-2:]
    gadm_state["GID_1_new"] = gadm_state["GID_1"].str.replace("USA", "US")
    gadm_state = gadm_state[["GID_1_new", "state"]]
    return gadm_state


def preprocess_pypsa_demand(network_path, time_res, gadm_us):
    network = pypsa.Network(network_path)
    df = network.loads_t.p.sum() / 1e6 * time_res
    usa_state_dict = dict(gadm_us.values)

    df.index = df.index.str.replace("_AC", "")
    df.index = df.index.str.replace("_DC", "")
    df.index = df.index.map(usa_state_dict)

    df = df.reset_index().groupby('Load').sum()
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


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "statewise_validate",
            countries="US",
            clusters="10",
            planning_horizon="2020",
        )

    # update config based on wildcards
    config = update_config_from_wildcards(
        snakemake.config, snakemake.wildcards)

    planning_horizon = config["validation"]["planning_horizon"]
    data_horizon = int(planning_horizon)

    ################ PATHS ################
    eia_statewise_demand_path = os.path.join(
        DATA_DIR, "validation", "EIA_statewise_data", "use_all_phy.xlsx")
    eia_installed_capacity_by_state_path = os.path.join(
        DATA_DIR, "validation", "existcapacity_annual.xlsx")
    gadm_usa_json_path = os.path.join(DATA_DIR, "validation", "gadm41_USA_1.json")
    demand_pypsa_network_path = os.path.join(PYPSA_EARTH_DIR, "results", 
        "US_2021", "networks", "elec_s_10_ec_lcopt_Co2L-24H.nc")

    alternative_clustering = True
    time_res = 24
    plot_scale = 1.5

    eia_installed_capacity_by_state_year = preprocess_eia_capacity(
        eia_installed_capacity_by_state_path, data_horizon)
    network = load_pypsa_network(alternative_clustering)
    gadm_gdp_usa_state = preprocess_gadm(gadm_usa_json_path)
    pypsa_merged_df = preprocess_pypsa_cap(network, gadm_gdp_usa_state)

    eia_statewise_demand = preprocess_eia_demand(eia_statewise_demand_path)
    pypsa_statewise_demand = preprocess_pypsa_demand(
        demand_pypsa_network_path, time_res, gadm_gdp_usa_state)
    demand_total = pd.concat(
        [eia_statewise_demand, pypsa_statewise_demand], axis=1)

    ################ PLOTS ################
    fig1 = px.bar(pypsa_merged_df, barmode='stack', text_auto='.1f')
    fig1.update_layout(width=1000, yaxis_title='Installed capacity PyPSA (GW)')
    fig1.write_image(
        snakemake.output.statewise_installed_capacity_pypsa, scale=plot_scale)

    fig2 = px.bar(eia_installed_capacity_by_state_year,
                  barmode='stack', text_auto='.1f')
    fig2.update_layout(width=1000, yaxis_title='Installed capacity EIA (GW)')
    fig2.write_image(
        snakemake.output.statewise_installed_capacity_eia, scale=plot_scale)

    fig3 = px.bar(demand_total, barmode='group')
    fig3.update_layout(
        width=1000, yaxis_title='Annual electricity demand (TWh)')
    fig3.write_image(
        snakemake.output.demand_statewise_comparison, scale=plot_scale)
