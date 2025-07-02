# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import re
import pypsa
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def attach_region_to_buses(network, path_shapes, distance_crs="EPSG:4326"):
    """
    Attach region to buses
    """
    # Read the shapefile using geopandas
    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes.rename(columns={"GRID_REGIO": "Region"}, inplace=True)

    ac_dc_carriers = ["AC", "DC"]
    location_mapping = network.buses.query(
        "carrier in @ac_dc_carriers")[["x", "y"]]

    network.buses["x"] = network.buses["location"].map(
        location_mapping["x"]).fillna(0)
    network.buses["y"] = network.buses["location"].map(
        location_mapping["y"]).fillna(0)

    pypsa_gpd = gpd.GeoDataFrame(
        network.buses,
        geometry=gpd.points_from_xy(network.buses.x, network.buses.y),
        crs=4326
    )

    bus_cols = network.buses.columns
    bus_cols = list(bus_cols) + ["region"]

    st_buses = gpd.sjoin_nearest(shapes, pypsa_gpd, how="right")

    network.buses["region"] = st_buses["Region"]

    return network


def attach_state_to_buses(network, path_shapes, distance_crs="EPSG:4326"):
    """
    Attach state to buses
    """
    # Read the shapefile using geopandas
    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes["ISO_1"] = shapes["ISO_1"].apply(lambda x: x.split("-")[1])
    shapes.rename(columns={"ISO_1": "State"}, inplace=True)

    ac_dc_carriers = ["AC", "DC"]
    location_mapping = network.buses.query(
        "carrier in @ac_dc_carriers")[["x", "y"]]

    network.buses["x"] = network.buses["location"].map(
        location_mapping["x"]).fillna(0)
    network.buses["y"] = network.buses["location"].map(
        location_mapping["y"]).fillna(0)

    pypsa_gpd = gpd.GeoDataFrame(
        network.buses,
        geometry=gpd.points_from_xy(network.buses.x, network.buses.y),
        crs=4326
    )

    bus_cols = network.buses.columns
    bus_cols = list(bus_cols) + ["State"]

    st_buses = gpd.sjoin_nearest(shapes, pypsa_gpd, how="right")

    network.buses["state"] = st_buses["State"]

    return network


def compute_demand(network):
    """
    Compute total demand by region and by state
    """
    static_load_carriers = ["rail transport electricity",
                            "agriculture electricity", "industry electricity"]
    dynamic_load_carriers = ["AC", "services electricity", "land transport EV"]

    ac_loads = network.loads.query("carrier in 'AC'").index
    ac_profile = network.loads_t.p_set[ac_loads].multiply(
        network.snapshot_weightings.objective, axis=0).sum() / 1e6
    ac_load_bus = ac_profile.to_frame().reset_index().rename(
        columns={0: "load", "Load": "region"})
    ac_load_bus["carrier"] = "AC"

    nhours = network.snapshot_weightings.objective.sum()
    static_load = network.loads.groupby(["bus", "carrier"]).sum()[
        ["p_set"]].reset_index()
    static_load_bus = static_load.query(
        "carrier in @static_load_carriers").reset_index(drop=True)
    static_load_bus['p_set'] = static_load_bus.p_set * nhours / 1e6

    services_profile = network.loads_t.p_set.filter(
        like="services electricity") / 1e6
    services_load = services_profile.multiply(network.snapshot_weightings.objective, axis=0).sum(
    ).to_frame().reset_index().rename(columns={0: "services electricity load", "Load": "bus"})
    services_load["region"] = services_load["bus"].str.extract(
        r"(US\d{1} \d{1,2})")
    services_load.rename(
        columns={"services electricity load": "load"}, inplace=True)
    services_load["carrier"] = "services electricity"

    static_load_bus["region"] = static_load_bus["bus"].str.extract(
        r"(US\d{1} \d{1,2})")
    agriculture_electricity_load = static_load_bus.query(
        "carrier == 'agriculture electricity'")
    agriculture_electricity_load.rename(
        columns={"p_set": "load"}, inplace=True)

    industry_electricity_load = static_load_bus.query(
        "carrier == 'industry electricity'")
    industry_electricity_load.rename(columns={"p_set": "load"}, inplace=True)

    rail_transport_electricity_load = static_load_bus.query(
        "carrier == 'rail transport electricity'")
    rail_transport_electricity_load.rename(
        columns={"p_set": "load"}, inplace=True)

    ev_profile = network.loads_t.p_set.filter(like="land transport EV")
    ev_load = (ev_profile.multiply(network.snapshot_weightings.objective, axis=0).sum(
    ) / 1e6).to_frame().reset_index().rename(columns={0: "load", "Load": "bus"})
    ev_load["region"] = ev_load["bus"].str.extract(r"(US\d{1} \d{1,2})")
    ev_load["carrier"] = "land transport EV"

    all_loads = pd.concat([ac_load_bus, ev_load, services_load, agriculture_electricity_load,
                          industry_electricity_load, rail_transport_electricity_load], axis=0)

    all_loads_df_grid_region = all_loads.pivot(
        index="region", columns="carrier", values="load").fillna(0).round(2)
    all_loads_df_grid_region.index = all_loads_df_grid_region.index.map(
        network.buses.region)
    all_loads_df_grid_region_sum = all_loads_df_grid_region.groupby(
        "region").sum()

    all_loads_df_state = all_loads.pivot(
        index="region", columns="carrier", values="load").fillna(0).round(2)
    all_loads_df_state.index = all_loads_df_state.index.map(
        network.buses.state)
    all_loads_df_state_sum = all_loads_df_state.groupby("region").sum()

    return all_loads_df_grid_region_sum, all_loads_df_state_sum


def compute_data_center_load(network):
    """
    Compute data center load by grid region and by state
    """

    data_center_loads = network.loads.query("carrier in 'data center'")

    data_center_loads["grid_region"] = data_center_loads.bus.map(
        network.buses.region)
    data_center_loads["state"] = data_center_loads.bus.map(network.buses.state)

    return data_center_loads


def compute_carrier_costs(network, rename_tech):
    """Compute total carrier costs by region and by state
    """
    cost_df = network.statistics(
    )[['Capital Expenditure', "Operational Expenditure"]]
    carrier_cost_df = cost_df.reset_index(level=0, drop=True).sum(
        axis=1).reset_index().rename(columns={0: 'cost'})
    carrier_cost_df.carrier = carrier_cost_df.carrier.map(rename_tech)
    grouped_carrier_cost_df = carrier_cost_df.groupby(['carrier'])[
        ['cost']].sum()

    return grouped_carrier_cost_df


def update_ac_dc_bus_coordinates(network):
    """
    For all buses with carrier 'AC' or 'DC', update their 'x' and 'y' coordinates
    based on their 'location' field and the mapping from existing AC/DC buses.
    """
    ac_dc_carriers = ["AC", "DC"]
    location_mapping = network.buses.query(
        "carrier in @ac_dc_carriers")[["x", "y"]]
    network.buses["x"] = network.buses["location"].map(
        location_mapping["x"]).fillna(0)
    network.buses["y"] = network.buses["location"].map(
        location_mapping["y"]).fillna(0)
    return network


def fill_missing_nice_names(n, nice_names):
    """
    Fill missing nice_name values in n.carriers using the provided nice_names dict.
    Prints carriers that were missing and their new nice_name.
    """
    missing = n.carriers[n.carriers.nice_name == ""].index
    for idx in missing:
        if idx in nice_names:
            n.carriers.nice_name[idx] = nice_names[idx]
        else:
            print(f"No nice_name found for '{idx}' in nice_names dictionary.")


def fill_missing_color(n, color):
    """
    Fill missing color values in n.carriers using the provided color dict.
    Prints carriers that were missing and their new color.
    """
    missing = n.carriers[n.carriers.color == ""].index
    for idx in missing:
        if idx in color:
            n.carriers.color[idx] = color[idx]
        else:
            print(f"No color found for '{idx}' in color dictionary.")


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):

        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)

        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue

            names = ifind.index[ifind == i]

            c.df.loc[names, 'location'] = names.str[:i]


def compute_h2_capacities(network):
    """
    Compute the total capacities of hydrogen-related components in the network.
    Returns a DataFrame with the total capacities for each hydrogen-related component.
    """

    h2_carriers_buses = ['Alkaline electrolyzer large',
                         'PEM electrolyzer', 'SOEC',]

    hydrogen_links = network.links.query(
        "carrier in @h2_carriers_buses").copy()
    capacity_data = hydrogen_links.merge(
        network.buses[['state']],
        left_on='bus0',
        right_index=True,
        how='left'
    )

    capacity_data['p_nom_kw'] = capacity_data['p_nom_opt'] * 1000
    h2_capacity_data = capacity_data.pivot_table(
        index='bus0',
        columns='carrier',
        values='p_nom_kw',
        fill_value=0
    )

    h2_capacity_data['state'] = h2_capacity_data.index.map(network.buses.state)
    h2_capacity_data['region'] = h2_capacity_data.index.map(
        network.buses.region)

    return h2_capacity_data


def plot_h2_capacities_bar(network):
    """
    Plot the total hydrogen electrolyzer capacity by type for each state.
    """
    h2_cap = compute_h2_capacities(network)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    h2_cap.groupby(['state'])[['Alkaline electrolyzer large', 'PEM electrolyzer', 'SOEC']].sum().plot(
        ax=ax1,
        kind='bar',
        stacked=True,
        title='Hydrogen Electrolyzer Capacity by State and Type',
        ylabel='Capacity (kW)',
        xlabel='State',
    )

    h2_cap.groupby(['region'])[['Alkaline electrolyzer large', 'PEM electrolyzer', 'SOEC']].sum().plot(
        ax=ax2,
        kind='bar',
        stacked=True,
        title='Hydrogen Electrolyzer Capacity by Region and Type',
        ylabel='Capacity (kW)',
        xlabel='Region',
    )

    plt.tight_layout()
    plt.show()


def filter_and_group_small_carriers(df, threshold=0.005):
    """
    Filters a DataFrame to group small contributors into an 'other' category.
    This function assumes df contains only non-negative values.
    """
    if df.empty or df.sum().sum() == 0:
        return pd.DataFrame(index=df.index)
    totals = df.sum()
    grand_total = totals.sum()
    significant_carriers = totals[totals / grand_total > threshold].index
    df_filtered = df[significant_carriers].copy()
    other_carriers = totals[~totals.index.isin(significant_carriers)].index
    if not other_carriers.empty:
        df_filtered['other'] = df[other_carriers].sum(axis=1)
    return df_filtered


def calculate_dispatch(n, start_date=None, end_date=None):
    snapshots_slice = slice(start_date, end_date) if start_date and end_date else slice(None)
    snapshots = n.snapshots[snapshots_slice]

    valid_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac', 'nuclear',
        'geothermal', 'ror', 'hydro',
        'gas', 'OCGT', 'CCGT', 'oil', 'coal', 'biomass', 'lignite',
    }

    gen = n.generators[n.generators.carrier.isin(valid_carriers)]
    gen_dispatch = n.generators_t.p.loc[snapshots_slice, gen.index].groupby(
        n.generators.loc[gen.index, 'carrier'], axis=1).sum()

    sto = n.storage_units[n.storage_units.carrier.isin(valid_carriers)]
    sto_dispatch = n.storage_units_t.p.loc[snapshots_slice, sto.index].groupby(
        n.storage_units.loc[sto.index, 'carrier'], axis=1).sum()

    link = n.links[n.links.carrier.isin(valid_carriers) & n.links.p_nom_opt.notnull()]
    link_dispatch = n.links_t.p1.loc[snapshots_slice, link.index].groupby(
        n.links.loc[link.index, 'carrier'], axis=1).sum()

    supply = pd.concat([gen_dispatch, sto_dispatch, link_dispatch], axis=1)
    supply = supply.groupby(supply.columns, axis=1).sum()
    supply = supply.clip(lower=0)

    freq = pd.infer_freq(snapshots)
    timestep_hours = 1 if freq is None else pd.Timedelta(freq).total_seconds() / 3600

    supply_gw = supply / 1e3  # MW → GW

    energy_mwh = supply.sum(axis=1) * timestep_hours
    total_gwh = energy_mwh.sum() / 1e3  # MWh → GWh

    return total_gwh, supply_gw

def plot_electricity_dispatch(networks, carrier_colors, start_date=None, end_date=None, ymax=None):
    summary_list = []
    max_y = 0

    # Calcola dispatch per ogni network e trova il massimo valore per scala comune
    for key, n in networks.items():
        print(f"Processing network: {key}")
        total_gwh, supply_gw = calculate_dispatch(n, start_date, end_date)
        summary_list.append({"Network": key, "Total Dispatch (GWh)": total_gwh})
        max_y = max(max_y, supply_gw.sum(axis=1).max())

    y_max_plot = ymax if ymax is not None else max_y

    fig, axes = plt.subplots(len(networks), 1, figsize=(15, 5 * len(networks)), sharex=True)

    if len(networks) == 1:
        axes = [axes]

    # Plot dispatch for each network
    for ax, (key, n) in zip(axes, networks.items()):
        _, supply_gw = calculate_dispatch(n, start_date, end_date)
        supply_gw.plot.area(
            stacked=True,
            linewidth=0,
            color=[carrier_colors.get(c, 'gray') for c in supply_gw.columns],
            ax=ax,
            legend=False
        )
        ax.set_title(f"Electricity Dispatch – {key}")
        ax.set_ylabel("Power (GW)")
        ax.set_ylim(0, y_max_plot)
        ax.grid(True)

    axes[-1].set_xlabel("Time")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Legenda unica a destra
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.93, 0.95))

    plt.show()

    return summary_list

def compute_and_plot_load(n, key="", ymax=None, start_date=None, end_date=None):
    freq = pd.infer_freq(n.loads_t.p_set.index)
    snapshot_hours = 1 if freq is None else pd.Timedelta(freq).total_seconds() / 3600

    dynamic_load_gw = n.loads_t.p_set.sum(axis=1) / 1e3
    total_dynamic_gwh = (n.loads_t.p_set.sum(axis=1) * snapshot_hours).sum() / 1e3

    static_loads = n.loads[~n.loads.index.isin(n.loads_t.p_set.columns)]
    static_load_gw = static_loads["p_set"].sum() / 1e3
    total_hours = len(n.loads_t.p_set.index) * snapshot_hours
    total_static_gwh = static_load_gw * total_hours

    # Aggiungi qui il calcolo del dispatch totale
    total_dispatch_gwh, _ = calculate_dispatch(n, start_date, end_date)

    # Plot electric load
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(dynamic_load_gw.index, dynamic_load_gw.values, label="Dynamic Load (GW)")
    ax.hlines(static_load_gw, dynamic_load_gw.index.min(), dynamic_load_gw.index.max(),
              colors="red", linestyles="--", label="Static Load (GW)")

    start = dynamic_load_gw.index.min().replace(day=1)
    end = dynamic_load_gw.index.max()
    month_starts = pd.date_range(start=start, end=end, freq='MS')

    ax.set_xlim(start, end)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_starts.strftime('%b'))
    ax.tick_params(axis='x', rotation=0)

    if ymax:
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(bottom=0)

    ax.set_title(f"Electric Load Profile - {key}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (GW)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

    # Restituisci anche il totale dispatch per riepilogo
    return {
        "key": key,
        "mean_dynamic_load_gw": dynamic_load_gw.mean(),
        "total_dynamic_gwh": total_dynamic_gwh,
        "static_load_gw": static_load_gw,
        "total_static_gwh": total_static_gwh,
        "total_dispatch_gwh": total_dispatch_gwh,
    }