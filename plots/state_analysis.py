import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pypsa
import re
import plotly.express as px
import plotly.graph_objects as go

import warnings
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
        network = pypsa.Network(
            "../submodules/pypsa-earth/networks/US_2021/elec_s_10.nc")
    else:
        network = pypsa.Network("../data/validation/usa_mapped.nc")
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


if __name__ == "__main__":

    alternative_clustering = True

    eia_installed_capacity_by_state_path = "../data/validation/existcapacity_annual.xlsx"
    eia_installed_capacity_by_state_year = preprocess_eia_capacity(
        eia_installed_capacity_by_state_path, 2021)

    network = load_pypsa_network(alternative_clustering)

    gadm_usa_json_path = "data/gadm41_USA_1.json"
    gadm_gdp_usa_state = preprocess_gadm(gadm_usa_json_path)

    pypsa_merged_df = preprocess_pypsa_cap(network, gadm_gdp_usa_state)

    ################ PLOTS ################
    fig1 = px.bar(pypsa_merged_df, barmode='stack', text_auto='.1f')
    fig1.update_layout(width=1000, yaxis_title='Installed capacity PyPSA (GW)')
    fig1.write_image(
        f"../plots/installed_capacity_pypsa_countrywise.png", scale=1.5)

    fig2 = px.bar(eia_installed_capacity_by_state_year,
                  barmode='stack', text_auto='.1f')
    fig2.update_layout(width=1000, yaxis_title='Installed capacity EIA (GW)')
    fig2.write_image(
        f"../plots/installed_capacity_eia_countrywise.png", scale=1.5)
