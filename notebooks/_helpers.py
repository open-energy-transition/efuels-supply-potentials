# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pycountry

import pypsa
import math
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from pathlib import Path

import cartopy.crs as ccrs  # For plotting maps
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import box
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from matplotlib.patches import Wedge
import matplotlib.path as mpath
from matplotlib.patches import Patch
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerPatch
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
from IPython.display import display, HTML
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from collections import OrderedDict
from shapely.geometry import LineString
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
import yaml

import warnings
warnings.filterwarnings("ignore")


def attach_grid_region_to_buses(network, path_shapes, distance_crs="EPSG:4326"):
    """
    Attach grid region to buses
    """
    # Read the shapefile using geopandas
    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes.rename(columns={"GRID_REGIO": "Grid Region"}, inplace=True)

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
    bus_cols = list(bus_cols) + ["grid_region"]

    st_buses = gpd.sjoin_nearest(shapes, pypsa_gpd, how="right")

    network.buses.rename(columns={'region': 'emm_region'}, inplace=True)
    network.buses["grid_region"] = st_buses["Grid Region"]


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
        network.buses.grid_region)
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
        network.buses.grid_region)
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
    Compute the total capacities (in MW) of hydrogen-related components in the network.
    Returns a DataFrame with the total capacities for each hydrogen-related component.
    """
    h2_carriers_buses = [
        'Alkaline electrolyzer large', 'Alkaline electrolyzer medium', 'Alkaline electrolyzer small',
        'PEM electrolyzer', 'SOEC',
    ]

    # Filter hydrogen-related links
    hydrogen_links = network.links.query(
        "carrier in @h2_carriers_buses").copy()

    # Merge with bus metadata (state and grid_region)
    capacity_data = hydrogen_links.merge(
        network.buses[['state', 'grid_region']],
        left_on='bus0',
        right_index=True,
        how='left'
    )

    # Use p_nom_opt directly (already in MW)
    capacity_data['p_nom_mw'] = capacity_data['p_nom_opt']

    # Pivot table to aggregate capacity by carrier and bus
    h2_capacity_data = capacity_data.pivot_table(
        index='bus0',
        columns='carrier',
        values='p_nom_mw',
        aggfunc='sum',
        fill_value=0
    )

    # Add state and grid_region information
    h2_capacity_data['state'] = h2_capacity_data.index.map(network.buses.state)
    h2_capacity_data['grid_region'] = h2_capacity_data.index.map(
        network.buses.grid_region)

    return h2_capacity_data


def plot_h2_capacities_by_state(grouped, title, ymax, max_n_states, bar_width=0.5, height=5):
    """
    Plot with fixed x-axis limits to preserve bar width across different state counts.
    """
    if grouped.empty:
        print(f"Skipping plot for {title} (no data)")
        return

    state = grouped.index.tolist()
    n = len(state)
    x = np.arange(n)

    techs = grouped.columns.tolist()
    bottoms = np.zeros(n)

    fig, ax = plt.subplots(figsize=(max_n_states * bar_width * 3.5, height))

    for tech in techs:
        ax.bar(x, grouped[tech].values, bar_width, bottom=bottoms, label=tech)
        bottoms += grouped[tech].values

    # Add value labels on top of each stacked bar
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(x[i], total + 0.01 * ymax,
                    f"{total:.0f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(state, rotation=30, ha='center')
    ax.set_xlabel("State")
    ax.set_ylabel("Capacity (MW input electricity)")

    if ymax > 0:
        ax.set_ylim(0, ymax * 1.05)
    else:
        ax.set_ylim(0, 1)

    ax.set_xlim(-0.5, max_n_states - 0.5)

    ax.set_title(
        f"\nHydrogen electrolyzer capacity by State and technology - {title}\n")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

    plt.tight_layout()
    plt.show()


def plot_h2_capacities_by_grid_region(grouped, title, ymax, max_n_grid_regions, bar_width=0.5, height=5):
    """
    Plot with fixed x-axis limits to preserve bar width across different grid regions counts.
    """
    if grouped.empty:
        print(f"Skipping plot for {title} (no data)")
        return

    grid_region = grouped.index.tolist()
    n = len(grid_region)
    x = np.arange(n)

    techs = grouped.columns.tolist()
    bottoms = np.zeros(n)

    fig, ax = plt.subplots(
        figsize=(max_n_grid_regions * bar_width * 5.5, height))

    for tech in techs:
        ax.bar(x, grouped[tech].values, bar_width, bottom=bottoms, label=tech)
        bottoms += grouped[tech].values

    # Add text on top of each stacked bar
    for i in range(n):
        total = grouped.iloc[i].sum()
        ax.text(x[i], total + 0.01 * ymax,
                f"{total:.0f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(grid_region, rotation=30, ha='center')
    ax.set_xlabel("Grid Region")
    ax.set_ylabel("Capacity (MW input electricity)")

    if ymax > 0:
        ax.set_ylim(0, ymax * 1.05)
    else:
        ax.set_ylim(0, 1)

    ax.set_xlim(-0.5, max_n_grid_regions - 0.5)

    ax.set_title(f"\nElectrolyzer capacity by Grid Region and technology - {title}\n")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

    plt.tight_layout()
    plt.show()


def create_hydrogen_capacity_map(network, path_shapes, distance_crs=4326, min_capacity_mw=10):
    """
    Create a map with pie charts showing electrolyzer capacity breakdown by type for each state
    """
    if hasattr(network, 'links') and len(network.links) > 0:
        # Filter for hydrogen-related links (electrolyzers)
        hydrogen_links = network.links[
            network.links['carrier'].str.contains('electrolyzer|SOEC', case=False, na=False) |
            network.links.index.str.contains(
                'electrolyzer|SOEC', case=False, na=False)
        ].copy()

    capacity_data = hydrogen_links.merge(
        network.buses[['state']],
        left_on='bus0',  # Assuming bus0 is the electrical connection
                right_index=True,
        how='left'
    )

    # capacity_data = links_with_state

    # Convert MW to MW (keep as MW for hydrogen as capacities are typically smaller)
    capacity_data['p_nom_mw'] = capacity_data['p_nom_opt']

    print(
        f"Found hydrogen capacity data for {capacity_data['state'].nunique()} states")
    print("Electrolyzer types found:",
          capacity_data['carrier'].unique().tolist())

    # Step 2: Read and prepare shapefile
    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes["ISO_1"] = shapes["ISO_1"].apply(lambda x: x.split("-")[1])
    shapes.rename(columns={"ISO_1": "State"}, inplace=True)

    # Get state centroids for pie chart placement
    shapes_centroid = shapes.copy()
    shapes_centroid['centroid'] = shapes_centroid.geometry.centroid
    shapes_centroid['cent_x'] = shapes_centroid.centroid.x
    shapes_centroid['cent_y'] = shapes_centroid.centroid.y

    # Define colors for electrolyzer types
    unique_carriers = capacity_data['carrier'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_carriers)))
    carrier_colors = dict(zip(unique_carriers, colors))

    # Customize colors for common electrolyzer types
    custom_colors = {
        'H2 Electrolysis': '#1f77b4',           # Blue
        'alkaline': '#ff7f0e',                  # Orange
        'PEM': '#2ca02c',                       # Green
        'SOEC': '#d62728',                      # Red
        'AEL': '#9467bd',                       # Purple
        'electrolyzer': '#8c564b',              # Brown
        'hydrogen': '#e377c2',                  # Pink
        'H2': '#7f7f7f',                        # Gray
    }

    # Update carrier_colors with custom colors
    for carrier, color in custom_colors.items():
        if carrier in carrier_colors:
            carrier_colors[carrier] = color

    # Create the plot
    fig, ax = plt.subplots(figsize=(30, 20))

    # Plot the base map
    shapes.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.3)

    # Group capacity data by state
    state_capacity = capacity_data.groupby('state').agg({
        'p_nom_mw': 'sum'
    }).reset_index()

    # Filter states with minimum capacity
    states_to_plot = state_capacity['state'].tolist()

    print(
        f"Plotting {len(states_to_plot)} states with ≥{min_capacity_mw} MW input electricity")

    # Create pie charts for each state
    for state in states_to_plot:
        state_data = capacity_data[capacity_data['state'] == state]

        if len(state_data) == 0:
            continue

        # Get state centroid
        state_centroid = shapes_centroid[shapes_centroid['State'] == state]
        if len(state_centroid) == 0:
            continue

        cent_x = state_centroid['cent_x'].iloc[0]
        cent_y = state_centroid['cent_y'].iloc[0]

        # Prepare pie chart data
        sizes = state_data['p_nom_mw'].values
        labels = state_data['carrier'].values
        colors_list = [carrier_colors[carrier] for carrier in labels]

        # Calculate pie chart radius based on total capacity
        total_capacity = sizes.sum()
        # Scale radius based on capacity (adjusted for MW scale)
        max_capacity = state_capacity['p_nom_mw'].max()
        radius = 0.3 + (total_capacity / max_capacity) * 1.5

        # Create pie chart
        pie_wedges, texts = ax.pie(sizes, colors=colors_list, center=(cent_x, cent_y),
                                   radius=radius, startangle=90)

        # Add capacity label
        ax.annotate(f'{total_capacity:.0f} MW',
                    xy=(cent_x, cent_y - radius - 0.3),
                    ha='center', va='top', fontsize=12)

    # Create legend
    legend_elements = []
    for carrier, color in carrier_colors.items():
        if carrier in capacity_data['carrier'].values:
            # Clean up carrier names for legend
            display_name = carrier.replace('_', ' ').title()
            legend_elements.append(
                Line2D([0], [0], color='none', label=f'— {group} —'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=14, title='Electrolyzer Type', title_fontsize=16)

    # Step 7: Formatting - Expand map boundaries
    x_buffer = (shapes.total_bounds[2] - shapes.total_bounds[0]) * 0.1
    y_buffer = (shapes.total_bounds[3] - shapes.total_bounds[1]) * 0.1

    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Installed Electrolyzer Capacity by State and Type',
                 fontsize=24, pad=30)

    # Add subtitle
    ax.text(0.5, 0.02, f'Note: Only states with ≥{min_capacity_mw} MW electrolyzer capacity are shown',
            transform=ax.transAxes, ha='center', fontsize=14, style='italic')

    plt.tight_layout()
    return fig, ax, capacity_data


def print_hydrogen_capacity_summary(capacity_data):
    """Print summary statistics of the hydrogen capacity data"""
    if len(capacity_data) == 0:
        print("No hydrogen capacity data to summarize.")
        return

    print("=== HYDROGEN ELECTROLYZER CAPACITY SUMMARY ===")
    print(
        f"Total installed hydrogen capacity: {capacity_data['p_nom'].sum():.1f} MW")
    print(
        f"Number of states with hydrogen capacity: {capacity_data['state'].nunique()}")
    print(
        f"Number of electrolyzer types: {capacity_data['carrier'].nunique()}")

    print("\n=== TOP 10 STATES BY HYDROGEN CAPACITY ===")
    state_totals = capacity_data.groupby(
        'state')['p_nom'].sum().sort_values(ascending=False)
    for i, (state, capacity) in enumerate(state_totals.head(10).items()):
        print(f"{i+1:2d}. {state}: {capacity:.1f} MW")

    print("\n=== ELECTROLYZER TYPE MIX (NATIONAL) ===")
    carrier_totals = capacity_data.groupby(
        'carrier')['p_nom'].sum().sort_values(ascending=False)
    total_national = carrier_totals.sum()
    for carrier, capacity in carrier_totals.items():
        print(
            f"{carrier:25s}: {capacity:8.1f} MW ({capacity/total_national*100:5.1f}%)")


def create_ft_capacity_by_state_map(network, path_shapes, network_name="Network", distance_crs=4326, min_capacity_gw=0.1, year_title=True):
    """
    Create a geographic map with simple round circles showing FT capacity per state in gigawatts (GW input H2).
    """

    year_match = re.search(r'\d{4}', network_name)
    year_str = f" – {year_match.group()}" if year_match else ""

    ft_links = network.links[
        network.links['carrier'].str.contains('FT|Fischer|Tropsch', case=False, na=False) |
        network.links.index.str.contains(
            'FT|Fischer|Tropsch', case=False, na=False)
    ].copy()

    if ft_links.empty:
        print(f"No FT links found in the network: {network_name}")
        return None, None, None

    links_with_state = ft_links.merge(
        network.buses[['state', 'x', 'y']],
        left_on='bus0',
        right_index=True,
        how='left'
    )
    links_with_state['p_nom_gw'] = links_with_state['p_nom_opt'] / 1000

    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes["ISO_1"] = shapes["ISO_1"].apply(lambda x: x.split("-")[1])
    shapes.rename(columns={"ISO_1": "State"}, inplace=True)

    state_capacity = links_with_state.groupby(
        'state').agg({'p_nom_gw': 'sum'}).reset_index()
    states_to_plot = state_capacity[state_capacity['p_nom_gw']
                                    >= min_capacity_gw]['state'].tolist()

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={
                           "projection": ccrs.PlateCarree()})
    bbox = box(-130, 20, -65, 50)
    shapes_clip = shapes.to_crs(epsg=4326).clip(bbox)
    shapes_clip.plot(ax=ax, facecolor='whitesmoke',
                     edgecolor='gray', alpha=0.7, linewidth=0.5)

    lon_min, lon_max = -130, -65
    lat_min, lat_max = 20, 50
    pie_scale = 0.2

    for state in states_to_plot:
        data = links_with_state[links_with_state['state'] == state]
        if data.empty:
            continue
        x = data['x'].mean()
        y = data['y'].mean()
        if pd.isna(x) or pd.isna(y):
            continue
        if not (lon_min < x < lon_max and lat_min < y < lat_max):
            continue

        total_gw = data['p_nom_gw'].sum()
        if total_gw == 0:
            continue

        radius = np.clip(total_gw * pie_scale, 0.1, 3.5)
        circle = plt.Circle((x, y), radius, color='#B22222', alpha=0.6,
                            transform=ccrs.PlateCarree(), zorder=4, linewidth=1)

        ax.add_patch(circle)

        ax.text(x, y - radius - 0.3, f'{total_gw:.2f} GW',
                ha='center', va='top', fontsize=9, fontweight='normal',
                bbox=dict(facecolor='white', edgecolor='gray',
                          boxstyle='round,pad=0.2'),
                transform=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.autoscale(False)
    ax.set_position([0.05, 0.05, 0.9, 0.9])

    ax.set_title(
        f"Fischer-Tropsch Capacity by State (GW input H2){year_str if year_title else network_name}", fontsize=12)
    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    return fig, ax, links_with_state


def create_ft_capacity_by_grid_region_map(network, path_shapes, network_name="Network", distance_crs=4326, min_capacity_gw=0.1, year_title=True):
    """
    Create a map showing total FT capacity per grid region (GW input H2) as full circles with linear radius scaling.
    """

    # Extract year from network name
    year_match = re.search(r'\d{4}', network_name)
    year_str = f" – {year_match.group()}" if year_match else ""

    # Filter FT-related links
    ft_links = network.links[
        network.links['carrier'].str.contains('FT|Fischer|Tropsch', case=False, na=False) |
        network.links.index.str.contains(
            'FT|Fischer|Tropsch', case=False, na=False)
    ].copy()

    if ft_links.empty:
        print(f"No FT links found in the network: {network_name}")
        return None, None, None

    # Merge link data with grid_region, x, y from bus0
    links_with_grid_region = ft_links.merge(
        network.buses[["grid_region", "x", "y"]],
        left_on="bus0",
        right_index=True,
        how="left"
    )
    links_with_grid_region["p_nom_gw"] = links_with_grid_region["p_nom_opt"] / 1000

    # Aggregate capacity per grid region
    grid_region_capacity = links_with_grid_region.groupby("grid_region").agg(
        total_gw=("p_nom_gw", "sum"),
        x=("x", "mean"),
        y=("y", "mean")
    ).reset_index()

    # Filter small values
    grid_region_capacity_filtered = grid_region_capacity[grid_region_capacity["total_gw"]
                                                         >= min_capacity_gw]

    # Set up map
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={
                           "projection": ccrs.PlateCarree()})
    bbox = box(-130, 20, -60, 50)
    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes = shapes.to_crs(epsg=4326).clip(bbox)
    shapes.plot(ax=ax, facecolor='whitesmoke',
                edgecolor='gray', alpha=0.7, linewidth=0.5)

    # Plot circles with linear scaling
    pie_scale = 0.2  # degrees per GW
    min_radius = 0.1
    max_radius = 3.5

    for _, row in grid_region_capacity_filtered.iterrows():
        x, y, total_gw = row["x"], row["y"], row["total_gw"]
        radius = np.clip(total_gw * pie_scale, min_radius, max_radius)

        circle = plt.Circle((x, y), radius,
                            facecolor='#B22222', edgecolor='gray', alpha=0.6,
                            linewidth=1, transform=ccrs.PlateCarree(), zorder=4)
        ax.add_patch(circle)

        ax.text(x, y - radius - 0.3, f'{total_gw:.2f} GW',
                ha='center', va='top', fontsize=9, fontweight='normal',
                bbox=dict(facecolor='white', edgecolor='gray',
                          boxstyle='round,pad=0.2'),
                transform=ccrs.PlateCarree())

    ax.set_extent([-130, -65, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(
        f"Fischer-Tropsch Capacity by Grid Region (GW input H2){year_str if year_title else network_name}", fontsize=12, pad=20)
    ax.axis('off')
    plt.tight_layout()

    return fig, ax, grid_region_capacity


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
    # Select time window
    snapshots_slice = slice(start_date, end_date) if start_date and end_date else slice(None)
    snapshots = n.snapshots[snapshots_slice]

    timestep_hours = (snapshots[1] - snapshots[0]).total_seconds() / 3600

    gen_and_sto_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac',
        'nuclear', 'geothermal', 'ror', 'hydro', 'solar rooftop',
    }
    link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', "biomass CHP", "gas CHP"]

    # identify electric buses
    electric_buses = set(
        n.buses.index[
            ~n.buses.carrier.str.contains("heat|gas|H2|oil|coal", case=False, na=False)
        ]
    )

    # Generators
    gen = n.generators[n.generators.carrier.isin(gen_and_sto_carriers)]
    gen_p = n.generators_t.p.loc[snapshots_slice, gen.index].clip(lower=0)
    gen_dispatch = gen_p.groupby(gen['carrier'], axis=1).sum()

    # Storage units
    sto = n.storage_units[n.storage_units.carrier.isin(gen_and_sto_carriers)]
    sto_p = n.storage_units_t.p.loc[snapshots_slice, sto.index].clip(lower=0)
    sto_dispatch = sto_p.groupby(sto['carrier'], axis=1).sum()

    # Links: conventional generation
    link_frames = []
    for carrier in link_carriers:
        links = n.links[(n.links.carrier == carrier) & (n.links.bus1.isin(electric_buses))]
        if links.empty:
            continue
        p1 = n.links_t.p1.loc[snapshots_slice, links.index].clip(upper=0)
        p1_positive = -p1
        df = p1_positive.groupby(links['carrier'], axis=1).sum()
        link_frames.append(df)

    # Battery
    battery_links = n.links[n.links.carrier == "battery discharger"]
    if not battery_links.empty:
        p1 = n.links_t.p1.loc[snapshots_slice, battery_links.index].clip(upper=0)
        battery_dispatch = -p1.groupby(battery_links['carrier'], axis=1).sum()
        battery_dispatch.columns = ["battery discharger"]
        link_frames.append(battery_dispatch)

    link_dispatch = pd.concat(link_frames, axis=1) if link_frames else pd.DataFrame(index=snapshots)

    # Combine everything
    supply = pd.concat([gen_dispatch, sto_dispatch, link_dispatch], axis=1)
    supply = supply.groupby(supply.columns, axis=1).sum().clip(lower=0)

    # Convert
    supply_gw = supply / 1e3
    energy_mwh = supply.sum(axis=1) * timestep_hours
    total_gwh = energy_mwh.sum() / 1e3

    return total_gwh, supply_gw


def plot_electricity_dispatch(networks, carrier_colors, start_date=None, end_date=None, ymax=None):
    summary_list = []
    max_y = 0

    # First pass: calculate ymax (for plot scaling) and total GWh
    for key, n in networks.items():
        print(f"Processing network: {key}")
        total_gwh, supply_gw = calculate_dispatch(n, start_date, end_date)
        summary_list.append({"Network": key, "Total Dispatch (GWh)": total_gwh})
        max_y = max(max_y, supply_gw.sum(axis=1).max())

    # Use provided ymax or max across all networks
    y_max_plot = ymax if ymax is not None else max_y

    # Create one subplot per network
    fig, axes = plt.subplots(len(networks), 1, figsize=(22, 5 * len(networks)), sharex=True)
    if len(networks) == 1:
        axes = [axes]

    # Order of technologies in the stacked plot
    ordered_columns = [
        'nuclear',
        'coal',
        'biomass', 'biomass CHP', 'gas CHP',
        'CCGT', 'OCGT', 'oil',
        'hydro', 'ror',
        'geothermal',
        'solar', 'solar rooftop', 'csp',
        'onwind', 'offwind-ac', 'offwind-dc',
        'battery discharger'
    ]

    # Loop through networks and plot each one
    for ax, (key, n) in zip(axes, networks.items()):
        _, supply_gw = calculate_dispatch(n, start_date, end_date)

        # Convert index to datetime and resample daily averages
        supply_gw.index = pd.to_datetime(supply_gw.index)
        supply_gw = supply_gw.resample("24H").mean()

        # Keep only carriers present in both ordered_columns and supply_gw
        supply_gw = supply_gw[[c for c in ordered_columns if c in supply_gw.columns]]

        # Stacked area plot
        supply_gw.plot.area(
            ax=ax,
            stacked=True,
            linewidth=0,
            color=[carrier_colors.get(c, 'gray') for c in supply_gw.columns],
            legend=False
        )

        # Title and axes formatting
        ax.set_title(f"Electricity dispatch – {key}")
        ax.set_ylabel("Power (GW)")
        ax.set_ylim(0, y_max_plot)
        ax.grid(True)

        # Legend: only keep carriers with nonzero total generation
        handles, labels = ax.get_legend_handles_labels()
        sums = supply_gw.sum()
        filtered = [(h, l) for h, l in zip(handles, labels) if sums.get(l, 0) > 0]

        if filtered:
            handles, labels = zip(*filtered)
            ax.legend(
                handles, labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                title='Technology',
                fontsize='small',
                title_fontsize='medium'
            )

    # Label x-axis for the bottom plot
    axes[-1].set_xlabel("Time")
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.show()

    return summary_list




def compute_and_plot_load(n, key="", ymax=None, start_date=None, end_date=None):
    freq = pd.infer_freq(n.loads_t.p_set.index)
    snapshots = n.snapshots
    snapshot_hours = (snapshots[1] - snapshots[0]).total_seconds() / 3600

    dynamic_load_gw = n.loads_t.p_set.sum(axis=1) / 1e3
    total_dynamic_gwh = (n.loads_t.p_set.sum(
        axis=1) * snapshot_hours).sum() / 1e3

    static_loads = n.loads[~n.loads.index.isin(n.loads_t.p_set.columns)]
    static_load_gw = static_loads["p_set"].sum() / 1e3
    total_hours = len(n.loads_t.p_set.index) * snapshot_hours
    total_static_gwh = static_load_gw * total_hours

    total_dispatch_gwh, _ = calculate_dispatch(n, start_date, end_date)

    # Plot electric load
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dynamic_load_gw.index, dynamic_load_gw.values,
            label="Dynamic Load (GW)")
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

    return {
        "key": key,
        "mean_dynamic_load_gw": dynamic_load_gw.mean(),
        "total_dynamic_gwh": total_dynamic_gwh,
        "static_load_gw": static_load_gw,
        "total_static_gwh": total_static_gwh,
        "total_dispatch_gwh": total_dispatch_gwh,
    }


def calculate_lcoe_summary_and_map(n, shapes):
    snapshot_weights = n.snapshot_weightings.generators

    gen_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac',
        'nuclear', 'geothermal', 'ror', 'hydro', 'solar rooftop',
    }

    storage_carriers = {
        'battery storage', 'hydro', 'PHS'
    }

    link_carriers = [
        'coal', 'oil', 'OCGT', 'CCGT', 'biomass', 'lignite',
        "urban central solid biomass CHP", "urban central gas CHP"
    ]

    electric_buses = set(n.buses[n.buses.carrier == 'AC'].index)

    # --- Generators ---
    gen = n.generators[n.generators.carrier.isin(gen_carriers)].copy()
    gen_dispatch = n.generators_t.p[gen.index].multiply(
        snapshot_weights, axis=0)
    gen['energy'] = gen_dispatch.sum()
    gen = gen[(gen.p_nom_opt > 0) & (gen.energy > 0)]
    gen['lcoe'] = (gen.capital_cost * gen.p_nom_opt +
                   gen.marginal_cost * gen.energy) / gen.energy
    gen['type'] = 'generator'

    # --- Storage units ---
    sto = n.storage_units[n.storage_units.carrier.isin(
        storage_carriers)].copy()
    sto_dispatch = n.storage_units_t.p[sto.index].clip(
        lower=0).multiply(snapshot_weights, axis=0)
    sto['energy'] = sto_dispatch.sum()
    sto = sto[(sto.p_nom_opt > 0) & (sto.energy > 0)]
    sto['lcoe'] = (sto.capital_cost * sto.p_nom_opt +
                   sto.marginal_cost * sto.energy) / sto.energy
    sto['type'] = 'storage'

    # --- Links ---
    link = n.links[
        (n.links.carrier.isin(link_carriers)) &
        (n.links.bus1.isin(electric_buses)) &
        (n.links.p_nom_opt > 0)
    ].copy()

    link_dispatch = -n.links_t.p1[link.index].clip(upper=0)
    weighted_link_dispatch = link_dispatch.multiply(snapshot_weights, axis=0)
    link['energy'] = weighted_link_dispatch.sum()

    fuel_usage = n.links_t.p0[link.index].clip(lower=0)
    weighted_fuel_usage = fuel_usage.multiply(snapshot_weights, axis=0)
    link['fuel_usage'] = weighted_fuel_usage.sum()
    link['fuel_cost'] = link.bus0.map(n.generators.marginal_cost)

    # capacity factor
    H = float(snapshot_weights.sum())
    link['CF'] = link['energy'] / (link['p_nom_opt'] * H)

    def lcoe_link(row):
        if row['energy'] <= 0:
            return np.nan
        if row['carrier'] == 'oil':
            return np.nan
        # filtro: se CF < 5% → NaN
        if row['CF'] < 0.05:
            return np.nan
        return (
            row['capital_cost'] * row['p_nom_opt']
            + row['marginal_cost'] * row['fuel_usage']
            + row['fuel_cost'] * row['fuel_usage']
        ) / row['energy']

    link['lcoe'] = link.apply(lcoe_link, axis=1)
    link['type'] = 'link'

    # --- Merge data ---
    gen_data = gen[['bus', 'carrier', 'lcoe', 'type', 'energy']]
    sto_data = sto[['bus', 'carrier', 'lcoe', 'type', 'energy']]
    link_data = link[['bus1', 'carrier', 'lcoe', 'type', 'energy']].rename(columns={
                                                                           'bus1': 'bus'})

    lcoe_data = pd.concat([gen_data, sto_data, link_data], axis=0).dropna()
    lcoe_data = lcoe_data.merge(
        n.buses[['x', 'y', 'grid_region']], left_on='bus', right_index=True)

    lcoe_by_bus = (
        lcoe_data.groupby('bus')
        .apply(lambda df: pd.Series({
            'weighted_lcoe': (df['lcoe'] * df['energy']).sum() / df['energy'].sum(),
            'x': df['x'].iloc[0],
            'y': df['y'].iloc[0],
            'grid_region': df['grid_region'].iloc[0]
        }))
        .reset_index()
    )

    region_summary = (
        lcoe_data.groupby(['grid_region', 'carrier'])
        .agg(
            dispatch_mwh=('energy', 'sum'),
            total_cost=('lcoe', lambda x: (
                x * lcoe_data.loc[x.index, 'energy']).sum())
        )
        .reset_index()
    )
    region_summary['lcoe'] = region_summary['total_cost'] / \
        region_summary['dispatch_mwh']
    region_summary['dispatch'] = region_summary['dispatch_mwh'] / 1e6

    table = region_summary.pivot(
        index='grid_region', columns='carrier', values=['lcoe', 'dispatch'])
    table.columns = [
        f"{carrier} {metric} ({'USD/MWh' if metric == 'lcoe' else 'TWh'})"
        for metric, carrier in table.columns
    ]
    table = table.reset_index()

    # filtra dispatch e sostituisci lcoe per dispatch bassi
    dispatch_cols = [col for col in table.columns if 'dispatch' in col.lower()]
    for col in dispatch_cols:
        table[col] = pd.to_numeric(table[col], errors='coerce').fillna(0.0)

    lcoe_cols = [col for col in table.columns if 'lcoe' in col.lower()]

    min_dispatch_threshold = 1  # TWh
    for lcoe_col in lcoe_cols:
        carrier = lcoe_col.split(" ")[0]
        dispatch_col = next(
            (col for col in dispatch_cols if col.startswith(carrier + " ")), None)
        if dispatch_col:
            mask = table[dispatch_col] < min_dispatch_threshold
            table.loc[mask, lcoe_col] = np.nan

    table[lcoe_cols] = table[lcoe_cols].applymap(
        lambda x: '-' if pd.isna(x) else round(x, 2))

    grid_region_weighted_lcoe = (
        lcoe_by_bus.merge(lcoe_data[['bus', 'energy']], on='bus', how='left')
        .groupby('grid_region')
        .apply(lambda df: (df['weighted_lcoe'] * df['energy']).sum() / df['energy'].sum())
    )
    table['Weighted Average LCOE (USD/MWh)'] = table['grid_region'].map(
        grid_region_weighted_lcoe).round(2)

    for col in table.columns:
        if col != 'grid_region':
            table[col] = table[col].round(
                2) if table[col].dtype != object else table[col]

    vmin = lcoe_by_bus['weighted_lcoe'].quantile(0.05)
    vmax = max(vmin, min(grid_region_weighted_lcoe.max()
               * 1.1, lcoe_by_bus['weighted_lcoe'].max()))

    geometry = [Point(xy) for xy in zip(lcoe_by_bus['x'], lcoe_by_bus['y'])]
    lcoe_gdf = gpd.GeoDataFrame(
        lcoe_by_bus, geometry=geometry, crs=shapes.crs).to_crs(shapes.crs)

    return lcoe_gdf, table, lcoe_by_bus, lcoe_data, vmin, vmax


def plot_lcoe_map_by_grid_region(lcoe_by_bus, lcoe_data, shapes, title=None, key=None, ax=None, vmin=None, vmax=None):
    grid_region_lcoe = (
        lcoe_by_bus.merge(lcoe_data[['bus', 'energy']],
                          left_on='bus', right_on='bus', how='left')
        .groupby('grid_region')
        .apply(lambda df: (df['weighted_lcoe'] * df['energy']).sum() / df['energy'].sum())
        .reset_index(name='weighted_lcoe')
    )

    shapes = shapes.rename(columns={'GRID_REGIO': 'grid_region'})
    shapes_lcoe = shapes.merge(grid_region_lcoe, on='grid_region', how='left')

    if vmin is None:
        vmin = shapes_lcoe['weighted_lcoe'].quantile(0.05)
    if vmax is None:
        vmax = shapes_lcoe['weighted_lcoe'].quantile(0.95)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={
                               'projection': ccrs.PlateCarree()})

    shapes_lcoe.plot(column='weighted_lcoe', cmap=plt.cm.get_cmap('RdYlGn_r'),
                     linewidth=0.8, edgecolor='0.8', legend=True,
                     vmin=vmin, vmax=vmax, ax=ax)

    ax.set_extent([-130, -65, 20, 55], crs=ccrs.PlateCarree())
    ax.axis('off')

    # Title handling
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            "Weighted average (plant level) LCOE per Grid Region (USD/MWh)")


def plot_h2_capacities_map(network, title, tech_colors, nice_names, regions_onshore):

    h2_carriers_links = ['H2 pipeline repurposed', 'H2 pipeline']
    h2_carriers_buses = ['Alkaline electrolyzer large',
                         'PEM electrolyzer', 'SOEC']

    net = network.copy()
    h2_capacity_data = compute_h2_capacities(net)[h2_carriers_buses]
    net.links.query("carrier in @h2_carriers_links", inplace=True)

    valid_buses = net.buses.dropna(subset=["x", "y"])
    valid_buses = valid_buses[
        (valid_buses["x"] > -200) & (valid_buses["x"] < 200) &
        (valid_buses["y"] > -90) & (valid_buses["y"] < 90)
    ]

    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={
                           "projection": ccrs.PlateCarree()})
    bbox = box(-130, 20, -60, 50)
    regions_onshore_clipped = regions_onshore.to_crs(epsg=4326).clip(bbox)
    regions_onshore_clipped.plot(ax=ax, facecolor='whitesmoke', edgecolor='gray',
                                 alpha=0.7, linewidth=0.5, zorder=0)

    line_scale = 5e2
    net.plot(ax=ax,
             bus_sizes=0,
             bus_alpha=0,
             link_widths=net.links.p_nom_opt / line_scale,
             line_colors='teal',
             link_colors='turquoise',
             color_geomap=False,
             flow=None,
             branch_components=['Link'],
             boundaries=[-130, -60, 20, 50])

    max_cap = h2_capacity_data.sum(axis=1).max()

    for bus_id, capacities in h2_capacity_data.iterrows():
        if bus_id not in valid_buses.index:
            continue
        x, y = valid_buses.loc[bus_id, ['x', 'y']]
        if not bbox.contains(gpd.points_from_xy([x], [y])[0]):
            continue

        total = capacities.sum()
        if total < 10:
            continue

        radius = np.clip(np.sqrt(total) * 0.02, 0.3, 2.0)
        colors = [tech_colors.get(c, 'gray') for c in capacities.index]

        start_angle = 0
        for val, color in zip(capacities.values, colors):
            if val == 0:
                continue
            angle = 360 * val / total
            wedge = Wedge(center=(x, y),
                          r=radius,
                          theta1=start_angle,
                          theta2=start_angle + angle,
                          facecolor=color,
                          edgecolor='k',
                          linewidth=0.3,
                          transform=ccrs.PlateCarree()._as_mpl_transform(ax),
                          zorder=5)
            ax.add_patch(wedge)
            start_angle += angle

        ax.text(x, y + radius + 0.3,
                f"{total:.1e} MW",
                transform=ccrs.PlateCarree(),
                fontsize=8,
                ha='center',
                va='bottom',
                zorder=6,
                bbox=dict(facecolor='white', edgecolor='gray',
                          boxstyle='round,pad=0.2', alpha=0.7))

    # Legends
    legend_anchor_x = 1.05
    bold_fp = FontProperties(weight='bold', size=10)

    # Electrolyzer Capacity Legend
    legend_caps = [1e1, 1e2, 1e3]
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='gray',
                   markersize=np.clip(np.sqrt(cap) * 0.02, 0.3, 2.0) * 20,
                   alpha=0.4, linestyle='None', label=f'{cap:.0e} MW')
        for cap in legend_caps
    ]
    cap_legend = ax.legend(handles=legend_elements,
                           title="Electrolyzer Capacity",
                           title_fontproperties=bold_fp,
                           fontsize=9,
                           loc='upper left',
                           bbox_to_anchor=(legend_anchor_x, 1),
                           frameon=False,
                           labelspacing=2.2,
                           handletextpad=1.0)

    # H2 pipeline legend
    link_caps_MW = [10, 100, 1000]
    # Apply a visual scaling factor to improve legend visibility
    legend_line_scale_factor = 2.5  # Adjust this if needed

    link_patches = [
        mlines.Line2D([], [], color='turquoise',
                      linewidth=(cap / line_scale) * legend_line_scale_factor,
                      label=f"{cap} MW")
        for cap in link_caps_MW
    ]

    line_legend = ax.legend(handles=link_patches,
                            title="H2 Pipeline",
                            title_fontproperties=FontProperties(weight='bold'),
                            fontsize=9,
                            loc='upper left',
                            bbox_to_anchor=(legend_anchor_x, 0.60),
                            frameon=False,
                            labelspacing=1.0)

    # Technology legend
    carrier_handles = [
        mpatches.Patch(color=tech_colors.get(c, 'gray'),
                       label=nice_names.get(c, c))
        for c in sorted(h2_capacity_data.columns)
        if h2_capacity_data[c].sum() > 0
    ]
    tech_legend = ax.legend(handles=carrier_handles,
                            title="Electrolyzer technologies",
                            title_fontproperties=FontProperties(weight='bold'),
                            fontsize=9,
                            loc='upper left',
                            bbox_to_anchor=(legend_anchor_x, 0.34),
                            frameon=False,
                            labelspacing=1.0)

    tech_legend._legend_title_box._text.set_ha("left")

    # Add in order
    ax.add_artist(cap_legend)
    ax.add_artist(line_legend)
    ax.add_artist(tech_legend)

    ax.set_extent([-130, -65, 20, 55], crs=ccrs.PlateCarree())

    ax.set_title(f'Installed electrolyzer capacity (MW input electricity) - {title} (only nodes ≥ 10 MW)\n')
    plt.tight_layout()
    plt.show()


def plot_lcoh_maps_by_grid_region_lcoe(
    networks,
    shapes,
    h2_carriers,
    regional_fees,
    emm_mapping,
    output_threshold=1.0,
):
    """
    Plot weighted average LCOH incl. Transmission fees (USD/kg),
    using LCOE-based electricity prices from additional renewable generators.
    Transmission fees are weighted by H2 output.
    """

    all_results = []

    # Normalize region column in shapefile
    for col in ["Grid Region", "GRID_REGIO", "grid_region"]:
        if col in shapes.columns:
            shapes = shapes.rename(columns={col: "grid_region"})
            break
    else:
        raise KeyError("No 'grid_region' column found in shapes GeoDataFrame")

    for year, net in networks.items():
        scen_year = int(re.search(r"\d{4}", str(year)).group())

        # Select electrolyzers
        links = net.links[net.links.carrier.isin(h2_carriers)]
        if links.empty:
            continue

        p0 = net.links_t.p0[links.index]  # electricity input (MW)
        p1 = net.links_t.p1[links.index]  # H2 output (MW)
        w = net.snapshot_weightings.generators

        cons_MWh = (p0).clip(lower=0).multiply(w, axis=0)   # MWh electricity
        h2_MWh = (-p1).clip(lower=0).multiply(w, axis=0)  # MWh H2
        h2_out = h2_MWh.sum()

        valid = h2_out > output_threshold
        if valid.sum() == 0:
            continue

        # Transmission fee mapping
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh"]
        ].set_index("region")

        df = pd.DataFrame({
            "bus": links.loc[valid, "bus0"],
            "h2_out": h2_out[valid]
        })
        df["grid_region"] = df["bus"].map(net.buses["grid_region"])
        df["EMM"] = df["grid_region"].map(emm_mapping)
        fee_trans = df["EMM"].map(fee_map["Transmission nom USD/MWh"])

        # Convert to USD/kg H2 (weighted)
        elec_rate = cons_MWh.loc[:, valid].sum(
            axis=0) / h2_out[valid]  # MWh el / MWh H2
        fee_trans_kg = (fee_trans * elec_rate / 33).reindex(df.index)

        # --- Compute LCOE for renewables built in scen_year ---
        allowed_carriers = ["csp", "solar", "onwind",
                            "offwind-ac", "offwind-dc", "ror", "hydro", "nuclear"]
        gens = net.generators[net.generators.carrier.isin(
            allowed_carriers)].copy()
        gens = gens[gens["build_year"] == scen_year]  # additionality
        if gens.empty:
            continue

        # Capacity factors
        gen_dispatch = net.generators_t.p[gens.index].multiply(w, axis=0).sum()
        gen_energy = gen_dispatch.sum()
        if gen_energy <= 0:
            continue

        capex = (gens.capital_cost * gens.p_nom_opt).sum()
        opex = (gens.marginal_cost * gen_dispatch).sum()
        lcoe_val = (capex + opex) / gen_energy  # USD/MWh el

        # Assign same LCOE to all buses
        df["LCOE elec"] = lcoe_val

        # LCOH incl. transmission
        df["LCOH incl. Transmission"] = (
            df["LCOE elec"] * elec_rate / 33) + fee_trans_kg
        df["year"] = scen_year

        all_results.append(df.dropna(subset=["grid_region"]))

    if not all_results:
        print("No valid data.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    # Weighted average by region
    region_lcoh = (
        all_df.groupby(["grid_region", "year"])
        .apply(lambda g: pd.Series({
            "weighted_lcoh": (g["LCOH incl. Transmission"] * g["h2_out"]).sum() / g["h2_out"].sum()
        }))
        .reset_index()
    )

    # Merge with shapes for plotting
    plot_df = shapes.merge(region_lcoh, on="grid_region", how="left")
    vmin = plot_df["weighted_lcoh"].quantile(0.05)
    vmax = plot_df["weighted_lcoh"].quantile(0.95)

    for y in sorted(region_lcoh.year.unique()):
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={
                               "projection": ccrs.PlateCarree()})
        year_df = plot_df[plot_df.year == y]
        year_df.plot(
            column="weighted_lcoh",
            cmap="RdYlGn_r",
            linewidth=0.8,
            edgecolor="0.8",
            legend=True,
            legend_kwds={"label": "LCOH incl. Transmission (USD/kg)"},
            vmin=vmin, vmax=vmax, ax=ax
        )
        ax.set_extent([-130, -65, 20, 55])
        ax.axis("off")
        ax.set_title(
            f"LCOH (incl. Transmission fees, elec. cost = LCOE) – {y}")
        plt.show()


def plot_lcoh_maps_by_grid_region_marginal(
    networks, shapes, h2_carriers, regional_fees, emm_mapping,
    output_threshold=1.0
):
    """
    Plot weighted average LCOH incl. Transmission fees (USD/kg),
    using marginal electricity prices. Transmission fees are weighted by H2 output.

    Notes
    -----
    - Electricity cost is computed from nodal marginal prices at the electrolyzer bus,
      weighted by electricity consumption and normalized per kg H2.
    """

    all_results = []

    # Normalize region column in shapes
    for col in ["Grid Region", "GRID_REGIO", "grid_region"]:
        if col in shapes.columns:
            shapes = shapes.rename(columns={col: "grid_region"})
            break
    else:
        raise KeyError("No 'grid_region' column found in shapes GeoDataFrame")

    for year, net in networks.items():
        scen_year = int(re.search(r"\d{4}", str(year)).group())

        links = net.links[net.links.carrier.isin(h2_carriers)]
        if links.empty:
            continue

        # Electricity input and H2 output
        p0 = net.links_t.p0[links.index]  # electricity input (MW)
        p1 = net.links_t.p1[links.index]  # H2 output (MW)
        w = net.snapshot_weightings.generators

        cons_MWh = p0.clip(lower=0).multiply(w, axis=0)   # MWh el
        h2_MWh = (-p1).clip(lower=0).multiply(w, axis=0)  # MWh H2
        h2_out = h2_MWh.sum()

        valid = h2_out > output_threshold
        if valid.sum() == 0:
            continue

        # Transmission fee mapping
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh"]
        ].set_index("region")

        df = pd.DataFrame({
            "bus": links.loc[valid, "bus0"],
            "h2_out": h2_out[valid]
        })
        df["grid_region"] = df["bus"].map(net.buses["grid_region"])
        df["EMM"] = df["grid_region"].map(emm_mapping)
        fee_trans = df["EMM"].map(fee_map["Transmission nom USD/MWh"])

        # Electricity consumption rate per H2 output
        elec_rate = cons_MWh.loc[:, valid].sum(
            axis=0) / h2_out[valid]  # MWh el / MWh H2

        # ---- Electricity cost from marginal prices ----
        elec_costs = {}
        for l in valid.index[valid]:
            bus = links.at[l, "bus0"]
            prices = net.buses_t.marginal_price[bus]
            cons = cons_MWh[l]
            elec_costs[l] = (cons * prices).sum()

        elec_cost_series = pd.Series(elec_costs)
        elec_val = elec_cost_series / h2_out[valid] / 33.0  # USD/kg H2

        # Add transmission fees
        fee_trans_kg = (fee_trans * elec_rate / 33).reindex(df.index)
        df["LCOH incl. Transmission"] = elec_val + fee_trans_kg
        df["year"] = scen_year

        all_results.append(df.dropna(subset=["grid_region"]))

    if not all_results:
        print("No valid data.")
        return

    all_df = pd.concat(all_results, ignore_index=True)
    region_lcoh = (
        all_df.groupby(["grid_region", "year"])
        .apply(lambda g: pd.Series({
            "weighted_lcoh": (g["LCOH incl. Transmission"] * g["h2_out"]).sum() / g["h2_out"].sum()
        }))
        .reset_index()
    )

    plot_df = shapes.merge(region_lcoh, on="grid_region", how="left")
    vmin = plot_df["weighted_lcoh"].quantile(0.05)
    vmax = plot_df["weighted_lcoh"].quantile(0.95)

    for y in sorted(region_lcoh.year.unique()):
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={
                               "projection": ccrs.PlateCarree()})
        year_df = plot_df[plot_df.year == y]
        year_df.plot(
            column="weighted_lcoh", cmap="RdYlGn_r",
            linewidth=0.8, edgecolor="0.8", legend=True,
            legend_kwds={"label": "LCOH incl. Transmission (USD/kg)"},
            vmin=vmin, vmax=vmax, ax=ax
        )
        ax.set_extent([-130, -65, 20, 55])
        ax.axis("off")
        ax.set_title(
            f"LCOH (incl. Transmission fees, elec.cost = marginal) – {y}")
        plt.show()


def calculate_lcoh_by_region(networks, h2_carriers, regional_fees, emm_mapping,
                             output_threshold=1.0, year_title=True,
                             electricity_price="marginal", planning_horizon=None):
    """
    Compute weighted average LCOH by grid region and year, including CAPEX, OPEX,
    electricity cost (marginal or LCOE with hourly matching + additionality),
    and T&D fees.

    Parameters
    ----------
    networks : dict
        Dictionary of PyPSA Network objects, keyed by year or scenario.
    h2_carriers : list
        List of link carrier names representing electrolyzers.
    regional_fees : pd.DataFrame
        Table with regional transmission & distribution fees by year.
    emm_mapping : dict
        Mapping from grid_region to EMM region.
    output_threshold : float, optional
        Minimum H2 output (MWh) to include a link in the calculation.
    year_title : bool, optional
        If True, use scenario year as key in results; else keep original key.
    electricity_price : {"marginal", "LCOE"}
        Method for electricity cost:
        - "marginal": use nodal marginal price of electricity
        - "LCOE": compute weighted LCOE from allowed additional generators
    planning_horizon : int or None
        Planning horizon year. If given and electricity_price="LCOE",
        only generators with build_year == planning_horizon are considered.
    """

    results = {}

    conv = 33.0   # MWh H2 per ton
    suffix = "USD/kg H2"

    # Allowed carriers and additionality flag (fixed)
    allowed_carriers = ["csp", "solar", "onwind", "offwind-ac", "offwind-dc",
                        "ror", "hydro", "nuclear"]
    additionality = True

    for year_key, net in networks.items():
        scen_year = int(re.search(r"\d{4}", str(year_key)).group())

        links = net.links[net.links.carrier.isin(h2_carriers)]
        if links.empty:
            continue

        # Electricity consumption and H2 output
        p0, p1 = net.links_t.p0[links.index], net.links_t.p1[links.index]
        w = net.snapshot_weightings.generators
        cons = p0.clip(lower=0).multiply(w, axis=0)       # MWh electricity
        h2 = (-p1).clip(lower=0).multiply(w, axis=0)    # MWh H2
        h2_out = h2.sum()

        valid = h2_out > output_threshold
        if valid.sum() == 0:
            continue

        capex = links.loc[valid, "capital_cost"] * \
            links.loc[valid, "p_nom_opt"]
        opex = links.loc[valid, "marginal_cost"] * \
            cons.loc[:, valid].sum(axis=0)

        # Electricity cost depending on method
        elec_cost = pd.Series(0.0, index=valid.index[valid])
        if electricity_price == "marginal":
            # Electricity cost is computed as a consumption-weighted average of nodal marginal prices:
            # for each electrolyzer, multiply its hourly electricity use (already weighted by snapshot weightings)
            # by the marginal price of the connected bus, and sum over all time steps.
            # This ensures that the effective electricity cost reflects both the temporal profile of consumption
            # and the locational marginal price at the bus where the electrolyzer is connected.

            # Consumption-weighted average nodal marginal prices
            for l in valid.index[valid]:
                bus = links.at[l, "bus0"]
                elec_cost[l] = (
                    cons[l] * net.buses_t.marginal_price[bus]).sum()

        elif electricity_price == "LCOE":
            # Electricity cost is computed as a weighted average LCOE of eligible (renewable + additional) generators
            # in the same grid region, based on their actual generation.

            # Select eligible generators
            gens = net.generators[net.generators.carrier.isin(
                allowed_carriers)].copy()
            if additionality and planning_horizon is not None:
                gens = gens[gens.build_year == planning_horizon]
            if gens.empty:
                continue

            # Weighted generation
            gen_dispatch = net.generators_t.p[gens.index].multiply(
                net.snapshot_weightings.generators, axis=0)
            gen_energy = gen_dispatch.sum()

            # Simple generator-level LCOE estimate [USD/MWh]
            # (here capital_cost is already annuitized capex + fixed opex in PyPSA)
            gen_lcoe = (gens.capital_cost +
                        gens.marginal_cost).reindex(gen_energy.index)

            # Weighted average per region
            region_lcoe = (
                gen_energy.groupby(gens.grid_region)
                .apply(lambda g: (gen_lcoe[g.index] * gen_energy[g.index]).sum() / gen_energy[g.index].sum())
            )

            # Apply to each electrolyzer
            for l in valid.index[valid]:
                bus = links.at[l, "bus0"]
                region = net.buses.at[bus, "grid_region"]
                avg_price = region_lcoe.get(region, np.nan)
                elec_cost[l] = cons[l].sum() * avg_price

        else:
            raise ValueError("electricity_price must be 'marginal' or 'LCOE'")

        out_valid = h2_out[valid]

        # Normalize to USD/kg H2
        capex_val = capex / out_valid / conv
        opex_val = opex / out_valid / conv
        elec_val = elec_cost / out_valid / conv

        df = pd.DataFrame({
            f"Electrolysis CAPEX ({suffix})": capex_val,
            f"Electrolysis OPEX ({suffix})": opex_val,
            f"Electricity ({suffix})": elec_val,
            "h2_out": out_valid,
            "bus": links.loc[valid, "bus0"]
        })
        df["grid_region"] = df["bus"].map(net.buses["grid_region"])

        # Transmission & distribution fees
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh", "Distribution nom USD/MWh"]
        ].set_index("region")
        df["EMM"] = df["grid_region"].map(emm_mapping)

        fee_trans = df["EMM"].map(fee_map["Transmission nom USD/MWh"])
        fee_dist = df["EMM"].map(fee_map["Distribution nom USD/MWh"])
        elec_rate = cons.loc[:, valid].sum(
            axis=0) / out_valid   # MWh el / MWh H2
        fee_trans_val = (fee_trans * elec_rate / conv).reindex(df.index)
        fee_dist_val = (fee_dist * elec_rate / conv).reindex(df.index)

        # LCOH breakdown
        df[f"LCOH (excl. T&D fees) ({suffix})"] = (
            df[f"Electrolysis CAPEX ({suffix})"] +
            df[f"Electrolysis OPEX ({suffix})"] +
            df[f"Electricity ({suffix})"]
        )
        df[f"LCOH + Transmission fees ({suffix})"] = df[f"LCOH (excl. T&D fees) ({suffix})"] + \
            fee_trans_val
        df[f"LCOH + T&D fees ({suffix})"] = df[f"LCOH + Transmission fees ({suffix})"] + fee_dist_val

        # Dispatch label
        dispatch_val = df["h2_out"].sum() * conv / 1000   # tons
        dispatch_label = "Total Hydrogen Dispatch (tons)"

        # Weighted averages by grid region
        region_summary = (
            df.groupby("grid_region")
            .apply(lambda g: pd.Series({
                f"Electrolysis CAPEX ({suffix})": (g[f"Electrolysis CAPEX ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Electrolysis OPEX ({suffix})":  (g[f"Electrolysis OPEX ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Electricity ({suffix})":        (g[f"Electricity ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"LCOH (excl. T&D fees) ({suffix})": (g[f"LCOH (excl. T&D fees) ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Transmission fees ({suffix})":   (fee_trans_val * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"LCOH + Transmission fees ({suffix})": (g[f"LCOH + Transmission fees ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Distribution fees ({suffix})":   (fee_dist_val * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"LCOH + T&D fees ({suffix})": (g[f"LCOH + T&D fees ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                dispatch_label: dispatch_val
            }))
            .reset_index().rename(columns={"grid_region": "Grid Region"})
        )

        # Round and store
        region_summary = region_summary.round(2)
        key = str(scen_year) if year_title else year_key
        results[key] = region_summary

    return results


def calculate_weighted_lcoh_table_by_year(networks, h2_carriers, regional_fees, emm_mapping,
                                          output_threshold=1.0, year_title=True):
    """
    Weighted cost breakdown per region and year (USD/kg & USD/MWh).
    """
    results = {}
    for year_key, net in networks.items():
        scen_year = int(re.search(r"\d{4}", str(year_key)).group())
        df = calculate_lcoh_by_region({year_key: net}, h2_carriers, regional_fees, emm_mapping,
                                      output_threshold, year_title)
        results.update(df)
    return results


def calculate_total_generation_by_carrier(network, start_date=None, end_date=None):

    # Time setup
    snapshots_slice = slice(
        start_date, end_date) if start_date and end_date else slice(None)
    snapshots = network.snapshots[snapshots_slice]
    timestep_h = (snapshots[1] - snapshots[0]).total_seconds() / 3600

    # Define relevant carriers ===
    gen_and_sto_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac', 'nuclear',
        'geothermal', 'ror', 'hydro', 'solar rooftop'
    }
    link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', 'lignite',
                     "urban central solid biomass CHP", "urban central gas CHP",
                    "battery discharger"
                    ]

    # Identify electric buses
    electric_buses = set(
        network.buses.index[
            ~network.buses.carrier.str.contains(
                "heat|gas|H2|oil|coal", case=False, na=False)
        ]
    )

    # Generators
    gen = network.generators[network.generators.carrier.isin(
        gen_and_sto_carriers)]
    gen_p = network.generators_t.p.loc[snapshots_slice, gen.index].clip(
        lower=0)
    gen_dispatch = gen_p.groupby(gen['carrier'], axis=1).sum()
    gen_energy_mwh = gen_dispatch.sum() * timestep_h

    # Storage units
    sto = network.storage_units[network.storage_units.carrier.isin(
        gen_and_sto_carriers)]
    sto_p = network.storage_units_t.p.loc[snapshots_slice, sto.index].clip(
        lower=0)
    sto_dispatch = sto_p.groupby(sto['carrier'], axis=1).sum()
    sto_energy_mwh = sto_dispatch.sum() * timestep_h

    # Link-based generation
    link_energy_twh = {}

    for carrier in link_carriers:
        links = network.links[
            (network.links.carrier == carrier) &
            (network.links.bus1.isin(electric_buses))
        ]

        if links.empty:
            link_energy_twh[carrier] = 0.0
            continue

        p1 = network.links_t.p1.loc[snapshots_slice, links.index]
        p1_positive = -p1.clip(upper=0)
        energy_mwh = p1_positive.sum().sum() * timestep_h
        link_energy_twh[carrier] = energy_mwh / 1e6  # MWh → TWh

    link_dispatch = pd.Series(link_energy_twh)

    # Combine all sources
    total_energy_twh = pd.concat([
        gen_energy_mwh / 1e6,    # MW → TWh
        sto_energy_mwh / 1e6,
        link_dispatch
    ])

    total_energy_twh = total_energy_twh.groupby(total_energy_twh.index).sum()
    total_energy_twh = total_energy_twh[total_energy_twh > 0].round(2)
    total_energy_twh = total_energy_twh.sort_values(ascending=False)

    return total_energy_twh


def plot_hydrogen_dispatch(networks, h2_carriers, output_threshold=1.0, year_title=True):
    """
    Plot hourly hydrogen dispatch per carrier (stacked area plot) for each network in the input dictionary.
    All plots share the same y-axis scale.

    Parameters:
    - networks: dict of PyPSA networks (e.g. {'scenario_2025': network, ...})
    - h2_carriers: list of hydrogen-producing carrier names (e.g. ['PEM', 'SOEC'])
    - output_threshold: minimum total energy (MWh) to include a link in the analysis
    """
    # First pass: find global max dispatch to fix y-axis scale
    global_max = 0
    dispatch_series_by_network = {}

    for key, network in networks.items():
        h2_links = network.links[network.links.carrier.isin(h2_carriers)]
        if h2_links.empty:
            continue

        link_ids = h2_links.index
        p1 = network.links_t.p1[link_ids]
        h2_output = -p1  # Flip sign: PyPSA convention

        data = {}
        for carrier in h2_carriers:
            carrier_links = h2_links[h2_links.carrier == carrier].index
            if carrier_links.empty:
                continue

            output = h2_output[carrier_links]
            output = output.loc[:, output.sum() > output_threshold]
            if output.empty:
                continue

            data[carrier] = output.sum(axis=1)

        if not data:
            continue

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        df = df.resample('24H').mean()
        df = df * 0.03  # Convert to tons
        dispatch_series_by_network[key] = df

        max_dispatch = df.sum(axis=1).max()
        if max_dispatch > global_max:
            global_max = max_dispatch

    if not dispatch_series_by_network:
        print("No valid hydrogen dispatch data found.")
        return pd.DataFrame()

    merged_dispatch_df = pd.concat(dispatch_series_by_network, axis=1)

    # Second pass: generate plots with fixed y-axis
    for key, df in dispatch_series_by_network.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        df.plot.area(ax=ax, linewidth=0)
        year = key[-4:]  # Extract the year
        ax.set_title(f"Electricity Dispatch – {year if year_title else key}")
        ax.set_title(
            f"Hydrogen Dispatch by technology – {year if year_title else key}", fontsize=14)
        ax.set_ylabel("Hydrogen Dispatch (tons/hour)")
        ax.set_xlabel("Time")
        ax.set_ylim(0, global_max * 1.05)  # add 5% headroom
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        start = df.index.min().replace(day=1)
        end = df.index.max()
        month_starts = pd.date_range(start=start, end=end, freq='MS')

        ax.set_xlim(start, end)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_starts.strftime('%b'))
        ax.tick_params(axis='x', rotation=0)

        ax.legend(
            title="Technology",
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    return merged_dispatch_df


def analyze_ft_costs_by_region(networks: dict, year_title=True):
    """
    Compute and display total Fischer-Tropsch fuel production and
    total marginal cost (USD/MWh) by grid region for each network.
    """
    for name, n in networks.items():
        # Identify Fischer-Tropsch links that are built or extendable with capacity
        ft_links = n.links[
            (n.links.carrier.str.contains("Fischer", case=False, na=False)) &
            (
                (n.links.get("p_nom_opt", 0) > 0) |
                ((n.links.get("p_nom", 0) > 0) &
                 (n.links.get("p_nom_extendable", False) == False))
            )
        ].copy()

        if ft_links.empty:
            print(f"\n{name}: No active Fischer-Tropsch links found.")
            continue

        # Filter out links that don't appear in all links_t.p* time series
        ft_link_ids = [
            link for link in ft_links.index
            if all(link in getattr(n.links_t, attr).columns for attr in ["p0", "p1", "p2", "p3"])
        ]

        if not ft_link_ids:
            print(f"\n{name}: No Fischer-Tropsch links with time series data.")
            continue

        # Extract hourly marginal prices for input buses (H2, CO2, electricity)
        price_dict = {}
        for link in ft_link_ids:
            price_dict[link] = {
                "h2_price":   n.buses_t.marginal_price[ft_links.at[link, "bus0"]],
                "co2_price":  n.buses_t.marginal_price[ft_links.at[link, "bus2"]],
                "elec_price": n.buses_t.marginal_price[ft_links.at[link, "bus3"]],
            }

        # Multiply prices × flows to compute total input cost per link
        p0 = n.links_t.p0[ft_link_ids]
        p2 = n.links_t.p2[ft_link_ids]
        p3 = n.links_t.p3[ft_link_ids]
        p1 = n.links_t.p1[ft_link_ids]

        marginal_cost_inputs = {}
        for link in ft_link_ids:
            cost_h2 = (p0[link] * price_dict[link]["h2_price"]).sum()
            cost_co2 = (p2[link] * price_dict[link]["co2_price"]).sum()
            cost_elec = (p3[link] * price_dict[link]["elec_price"]).sum()
            total_input_cost = cost_h2 + cost_co2 + cost_elec
            marginal_cost_inputs[link] = total_input_cost

        # Compute total marginal cost per MWh of fuel output (input + technical cost)
        marginal_cost_total = {}
        for link in ft_link_ids:
            output_mwh = -p1[link].sum()
            if output_mwh <= 0:
                continue
            tech_cost = ft_links.at[link, "marginal_cost"] * output_mwh
            total_cost_per_mwh = (
                marginal_cost_inputs[link] + tech_cost) / output_mwh
            marginal_cost_total[link] = {
                "bus": ft_links.at[link, "bus1"],
                "production (MWh)": output_mwh,
                "marginal_cost_total (USD/MWh)": total_cost_per_mwh
            }

        # Map each link's output bus to its grid_region
        df_links = pd.DataFrame.from_dict(marginal_cost_total, orient="index")
        bus_to_region = n.buses["grid_region"].to_dict()
        df_links["grid_region"] = df_links["bus"].map(bus_to_region)
        df_links = df_links.dropna(subset=["grid_region"])

        # Aggregate by grid_region (sum production, weighted average of cost)
        grouped = df_links.groupby("grid_region")
        sum_prod = grouped["production (MWh)"].sum()
        weighted_cost = grouped.apply(
            lambda g: (g["marginal_cost_total (USD/MWh)"] *
                       g["production (MWh)"]).sum() / g["production (MWh)"].sum()
        )

        df_region_result = pd.DataFrame({
            "production (MWh)": sum_prod,
            "marginal_cost_total (USD/MWh)": weighted_cost
        })

        # Round to 2 decimals
        df_region_result = df_region_result.round(2)

        # Reset index to make 'grid_region' a visible column
        df_region_result = df_region_result.reset_index()

        # Rename columns
        df_region_result = df_region_result.rename(columns={
            "grid_region": "Grid region",
            "production (MWh)": "Production (MWh)",
            "marginal_cost_total (USD/MWh)": "e-kerosene marginal cost (USD/MWh)",
        })

        # Format numbers and hide index
        styled = df_region_result.style.format({
            "Production (MWh)": "{:,.2f}",
            "e-kerosene marginal cost (USD/MWh)": "{:,.2f}"
        }).hide(axis="index")

        # Extract year from network name
        match = re.search(r"\d{4}", name)
        year = match.group() if match else "unknown"

        print(f"\nYear: {year if year_title else name}\n")
        display(styled)


def compute_aviation_fuel_demand(networks,
                                 include_scenario: bool = False,
                                 scenario_as_index: bool = False,
                                 wide: bool = False):
    """
    Compute kerosene / e-kerosene demand per scenario.

    Parameters
    ----------
    networks : dict[str, pypsa.Network]
    include_scenario : bool
        (Ignored when wide=True; kept for backward compatibility in long mode)
    scenario_as_index : bool
        (Ignored when wide=True)
    wide : bool
        If True returns a wide table with MultiIndex columns (Scenario, Metric) and a single Year column.

    Returns
    -------
    pandas.DataFrame
    """
    import re
    results = {}

    for name, n in networks.items():
        m = re.search(r'(?:scenario_(\d{2})|Base)_(\d{4})', name)
        if m:
            scenario = f"scenario_{m.group(1)}" if m.group(1) else "Base"
            year = int(m.group(2))
        else:
            scenario = name
            digits = ''.join(filter(str.isdigit, name[-4:]))
            year = int(digits) if digits.isdigit() else None

        kerosene_idx = n.loads.index[n.loads.carrier ==
                                     "kerosene for aviation"]
        ekerosene_idx = n.loads.index[n.loads.carrier ==
                                      "e-kerosene for aviation"]

        if kerosene_idx.empty and ekerosene_idx.empty:
            kerosene_twh = 0.0
            ekerosene_twh = 0.0
        else:
            w = n.snapshot_weightings.generators
            kerosene_mwh = (n.loads_t.p[kerosene_idx].multiply(w, axis=0).sum().sum()
                            if len(kerosene_idx) else 0.0)
            ekerosene_mwh = (n.loads_t.p[ekerosene_idx].multiply(w, axis=0).sum().sum()
                             if len(ekerosene_idx) else 0.0)
            kerosene_twh = kerosene_mwh / 1e6
            ekerosene_twh = ekerosene_mwh / 1e6

        results[name] = {
            "Scenario": scenario,
            "Year": year,
            "Kerosene (TWh)": kerosene_twh,
            "e-Kerosene (TWh)": ekerosene_twh,
        }

    df = pd.DataFrame.from_dict(results, orient="index").reset_index(drop=True)

    df["Total (TWh)"] = df["Kerosene (TWh)"] + df["e-Kerosene (TWh)"]
    df["e-Kerosene Share (%)"] = df.apply(
        lambda r: 0.0 if r["Total (TWh)"] == 0 else 100 *
        r["e-Kerosene (TWh)"] / r["Total (TWh)"],
        axis=1
    )

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].applymap(lambda x: 0.0 if abs(x) < 1e-9 else x)

    if not wide:
        if not include_scenario:
            df = df.drop(columns=["Scenario"])
        else:
            if scenario_as_index:
                df = df.set_index("Scenario")
        df = df.sort_values(
            ["Year"] + (["Scenario"] if include_scenario and not scenario_as_index else []))
        return df.reset_index(drop=not scenario_as_index)

    # Wide: columns (Scenario, Metric), single Year column
    metrics = ["Kerosene (TWh)", "e-Kerosene (TWh)",
               "Total (TWh)", "e-Kerosene Share (%)"]
    wide_df = df.pivot_table(
        index="Year", columns="Scenario", values=metrics, aggfunc="first")

    # Current pivot gives (Metric, Scenario); swap to (Scenario, Metric)
    wide_df.columns = wide_df.columns.swaplevel(0, 1)
    wide_df = wide_df.sort_index(axis=1, level=[0, 1])

    wide_df = wide_df.reset_index()  # keep single Year column
    return wide_df


def compute_emissions_from_links(net, net_definition="atmosphere"):
    """
    Compute CO2 flows (atmosphere, captured, sequestered) by process.

    Columns:
      - CO2 to Atmosphere (Mt CO2/year): flows to 'co2 atmosphere' (typo-safe)
      - CO2 Captured (Mt CO2/year): flows to generic CO2 buses incl. 'co2 stored'/'co2 storage'
      - CO2 Sequestered (Mt CO2/year): flows to 'co2 geological sequestration'
      - Net CO2 Emissions (Mt CO2/year):
          * 'atmosphere'  -> Atmosphere - Sequestered  (recommended; DAC negative)
          * 'neutral'     -> Atmosphere + Captured - Sequestered (DAC = 0)

    Units: Mt CO2/year
    """

    results = []
    bus_cols = [c for c in net.links.columns if re.fullmatch(r"bus\d+", c)]

    for link_name, row in net.links.iterrows():
        carrier = str(row["carrier"]).strip()

        co2_atm = 0.0
        co2_cap = 0.0
        co2_seq = 0.0

        for j, bus_col in enumerate(bus_cols):
            bus_val = str(row[bus_col]).lower().strip()
            p_col = f"p{j}"
            if p_col not in net.links_t or link_name not in net.links_t[p_col]:
                continue

            # tCO2/year (assuming 1 MWh on CO2 buses ~ 1 tCO2)
            flow = net.links_t[p_col][link_name].mean() * 8760.0

            # Order matters: atmosphere -> sequestration -> stored/storage -> generic CO2
            if ("co2 atmosphere" in bus_val) or ("co2 atmoshpere" in bus_val):
                co2_atm -= flow
            elif "co2 geological sequestration" in bus_val:
                co2_seq -= flow
            elif ("co2 stored" in bus_val) or ("co2 storage" in bus_val):
                co2_cap -= flow
            elif "co2" in bus_val:
                co2_cap -= flow

        results.append({
            "Process": carrier,
            "CO2 to Atmosphere (Mt CO2/year)": co2_atm / 1e6,
            "CO2 Captured (Mt CO2/year)": co2_cap / 1e6,
            "CO2 Sequestered (Mt CO2/year)": co2_seq / 1e6,
        })

    df = pd.DataFrame(results)
    summary = df.groupby("Process", as_index=False).sum()

    if net_definition == "neutral":
        net = (summary["CO2 to Atmosphere (Mt CO2/year)"] +
               summary["CO2 Captured (Mt CO2/year)"] -
               summary["CO2 Sequestered (Mt CO2/year)"])
    else:  # 'atmosphere' (default)
        net = (summary["CO2 to Atmosphere (Mt CO2/year)"] -
               summary["CO2 Sequestered (Mt CO2/year)"])

    summary["Net CO2 Emissions (Mt CO2/year)"] = net

    # Round numeric columns
    for c in summary.columns:
        if c != "Process":
            summary[c] = summary[c].round(2)

    return summary


def compute_emissions_grouped(net, carrier_groups):
    """
    Aggregate CO2 flows (captured, to atmosphere, sequestered) by carrier groups.
    Net emissions = Atmosphere - Sequestered.
    Units: Mt CO2/year
    """

    results = []
    bus_cols = [
        col for col in net.links.columns if re.fullmatch(r"bus\d+", col)]

    for link_name, row in net.links.iterrows():
        carrier = row["carrier"]

        # skip if not in any group
        if not any(carrier in carriers for carriers in carrier_groups.values()):
            continue

        # skip technical storage carriers
        if "storage" in carrier.lower():
            continue

        co2_atmos, co2_captured, co2_sequestered = 0.0, 0.0, 0.0

        for j, bus_col in enumerate(bus_cols):
            bus_val = str(row[bus_col]).lower().strip()
            p_col = f"p{j}"
            if p_col not in net.links_t or link_name not in net.links_t[p_col]:
                continue

            flow = net.links_t[p_col][link_name].mean() * 8760

            if "co2 atmosphere" in bus_val or "co2 atmoshpere" in bus_val:
                co2_atmos -= flow
            elif "geological sequestration" in bus_val:
                co2_sequestered -= flow
            elif "co2" in bus_val:
                co2_captured -= flow

        results.append({
            "carrier": carrier,
            "co2_atmosphere": co2_atmos,
            "co2_captured": co2_captured,
            "co2_sequestered": co2_sequestered
        })

    df = pd.DataFrame(results)

    group_results = []
    for group_name, group_carriers in carrier_groups.items():
        group_df = df[df["carrier"].isin(group_carriers)]
        if group_df.empty:
            continue

        atm = group_df["co2_atmosphere"].sum()
        captured = group_df["co2_captured"].sum()
        sequestered = group_df["co2_sequestered"].sum()

        group_results.append({
            "carrier group": group_name,
            "CO2 to Atmosphere (Mt CO2/year)": atm / 1e6,
            "CO2 Captured (Mt CO2/year)": captured / 1e6,
            "CO2 Sequestered (Mt CO2/year)": sequestered / 1e6,
            "Net CO2 Emissions (Mt CO2/year)": (atm - sequestered) / 1e6
        })

    return pd.DataFrame(group_results).round(2)


def compute_emissions_by_state(net, carrier_groups):
    """
    Compute CO2 flows (to atmosphere, stored/sequestered) by State and process group.
    Net emissions = Atmosphere - Sequestered.
    Units: Mt CO2/year
    """

    results = []
    bus_cols = [
        col for col in net.links.columns if re.fullmatch(r"bus\d+", col)]

    for link_name, row in net.links.iterrows():
        carrier = row["carrier"]

        # assign process group
        group = next((g for g, carriers in carrier_groups.items()
                      if carrier in carriers), None)
        if group is None:
            continue

        co2_atmos = 0.0
        co2_sequestered = 0.0

        for j, bus_col in enumerate(bus_cols):
            bus_val = str(row[bus_col]).lower().strip()
            p_col = f"p{j}"
            if p_col not in net.links_t or link_name not in net.links_t[p_col]:
                continue

            flow = net.links_t[p_col][link_name].mean() * 8760

            if "co2 atmosphere" in bus_val or "co2 atmoshpere" in bus_val:
                co2_atmos -= flow
            elif "geological sequestration" in bus_val or "co2 stored" in bus_val:
                co2_sequestered -= flow

        # infer State from connected buses
        state = "Unknown"
        for bus_col in bus_cols:
            bus = row[bus_col]
            if bus in net.buses.index and "state" in net.buses.columns:
                s = net.buses.loc[bus, "state"]
                if pd.notna(s) and s != "Unknown":
                    state = s
                    break

        results.append({
            "State": state,
            "group": group,
            "CO2 to Atmosphere (Mt CO2/year)": co2_atmos,
            "CO2 Sequestered (Mt CO2/year)": co2_sequestered
        })

    df = pd.DataFrame(results)

    if df.empty:
        return pd.DataFrame(columns=[
            "State", "group",
            "CO2 to Atmosphere (Mt CO2/year)",
            "CO2 Sequestered (Mt CO2/year)",
            "Net CO2 Emissions (Mt CO2/year)"
        ])

    summary = (
        df.groupby(["State", "group"])
        [["CO2 to Atmosphere (Mt CO2/year)", "CO2 Sequestered (Mt CO2/year)"]]
        .sum()
        .reset_index()
    )

    summary["Net CO2 Emissions (Mt CO2/year)"] = (
        summary["CO2 to Atmosphere (Mt CO2/year)"] -
        summary["CO2 Sequestered (Mt CO2/year)"]
    )

    for c in ["CO2 to Atmosphere (Mt CO2/year)",
              "CO2 Sequestered (Mt CO2/year)",
              "Net CO2 Emissions (Mt CO2/year)"]:
        summary[c] = (summary[c] / 1e6).round(2)

    return summary


def plot_emissions_maps_by_group(
    all_state_emissions,
    path_shapes,
    title,
    column: str = "Net CO2 Emissions (Mt CO2/year)",
    vmin=None,
    vmax=None
):
    """
    Plot CO2 emissions maps by process group and State.

    Parameters
    ----------
    all_state_emissions : pd.DataFrame
        Output from compute_emissions_by_state (must contain 'State' and 'group').
    path_shapes : str
        Path to state shapefile/geojson.
    title : str
        Title of the figure (e.g. scenario or year).
    column : str
        Column to plot (default: "Net CO2 Emissions (Mt CO2/year)").
    vmin, vmax : float
        Color scale limits.
    """

    if column not in all_state_emissions.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {list(all_state_emissions.columns)}"
        )

    # Load shapefile and enforce CRS
    gdf_states = gpd.read_file(path_shapes).to_crs("EPSG:4326")
    gdf_states["State"] = gdf_states["ISO_1"].str[-2:]

    groups = all_state_emissions["group"].unique()
    n = len(groups)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(8 * ncols, 6 * nrows))
    axes = axes.flat if n > 1 else [axes]

    for i, group in enumerate(groups):
        ax = axes[i]
        df_group = all_state_emissions[all_state_emissions["group"] == group].copy(
        )

        merged = gdf_states.merge(df_group, on="State", how="left")

        # Replace exact zeros with NaN so they are plotted as white
        plot_col = f"{column}_plot"
        merged[plot_col] = merged[column].replace(0, np.nan)

        merged.plot(
            column=plot_col,
            # red = positive (emissions), green = negative (removals)
            cmap="RdYlGn_r",
            legend=True,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            # white = 0 or missing
            missing_kwds={"color": "white", "label": "0 or no data"},
            edgecolor="black"
        )

        ax.set_title(f"{group}", fontsize=12)
        ax.set_xlim([-130, -65])
        ax.set_ylim([20, 55])
        ax.axis("off")

        # format legend
        leg = ax.get_legend()
        if leg:
            leg.set_bbox_to_anchor((1.05, 0.5))
            leg.set_title("Mt CO2/year", prop={"size": 8})
            for t in leg.get_texts():
                t.set_fontsize(8)

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{column} by process group and State – {title}",
        fontsize=14
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.85)
    plt.show()


def evaluate_res_ces_by_state(networks, ces, res, ces_carriers, res_carriers, multiple_scenarios=False):

    results = {}

    for name, network in networks.items():
        year = int(name[-4:])
        year_str = str(year)

        snapshots = network.snapshots
        timestep_h = (snapshots[1] - snapshots[0]).total_seconds() / 3600
        snapshots_slice = slice(None)

        gen_and_sto_carriers = {
            'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac', 'nuclear',
            'geothermal', 'ror', 'hydro', 'solar rooftop'
        }
        link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', 'lignite']

        electric_buses = set(
            network.buses.index[
                ~network.buses.carrier.str.contains(
                    "heat|gas|H2|oil|coal", case=False, na=False)
            ]
        )

        # Generators
        gen = network.generators[network.generators.carrier.isin(
            gen_and_sto_carriers)].copy()
        gen["state"] = gen["bus"].map(network.buses["state"])
        gen = gen[gen["state"].notna()]

        gen_p = network.generators_t.p.loc[snapshots_slice, gen.index].clip(
            lower=0)
        gen_energy = gen_p.multiply(timestep_h).sum()  # MWh per generator
        gen_energy = gen_energy.to_frame(name="energy_mwh")
        gen_energy["carrier"] = gen.loc[gen_energy.index, "carrier"]
        gen_energy["state"] = gen.loc[gen_energy.index, "state"]

        # Storage
        sto = network.storage_units[network.storage_units.carrier.isin(
            gen_and_sto_carriers)].copy()
        sto["state"] = sto["bus"].map(network.buses["state"])
        sto = sto[sto["state"].notna()]

        sto_p = network.storage_units_t.p.loc[snapshots_slice, sto.index].clip(
            lower=0)
        sto_energy = sto_p.multiply(timestep_h).sum()
        sto_energy = sto_energy.to_frame(name="energy_mwh")
        sto_energy["carrier"] = sto.loc[sto_energy.index, "carrier"]
        sto_energy["state"] = sto.loc[sto_energy.index, "state"]

        # Links
        link_data = []
        for i, link in network.links.iterrows():
            if (
                link["carrier"] in link_carriers and
                link["bus1"] in electric_buses and
                pd.notna(network.buses.loc[link["bus1"], "state"])
            ):
                p1 = -network.links_t.p1.loc[snapshots_slice, i].clip(upper=0)
                energy_mwh = p1.sum() * timestep_h
                link_data.append({
                    "carrier": link["carrier"],
                    "state": network.buses.loc[link["bus1"], "state"],
                    "energy_mwh": energy_mwh
                })

        link_energy = pd.DataFrame(link_data)

        # Combine all generations
        all_energy = pd.concat([
            gen_energy[["carrier", "state", "energy_mwh"]],
            sto_energy[["carrier", "state", "energy_mwh"]],
            link_energy[["carrier", "state", "energy_mwh"]]
        ])

        # Aggregate by State
        state_totals = all_energy.groupby("state")["energy_mwh"].sum()
        state_ces = all_energy[all_energy["carrier"].isin(
            ces_carriers)].groupby("state")["energy_mwh"].sum()
        state_res = all_energy[all_energy["carrier"].isin(
            res_carriers)].groupby("state")["energy_mwh"].sum()

        df = pd.DataFrame({
            "Total (MWh)": state_totals,
            "CES_energy": state_ces,
            "RES_energy": state_res
        }).fillna(0)

        df["% CES"] = 100 * df["CES_energy"] / df["Total (MWh)"]
        df["% RES"] = 100 * df["RES_energy"] / df["Total (MWh)"]

        # Targets
        if year_str in ces.columns:
            df["% CES target"] = df.index.map(
                lambda state: ces[year_str].get(state, float("nan")))
        else:
            df["% CES target"] = float("nan")

        if year_str in res.columns:
            df["% RES target"] = df.index.map(
                lambda state: res[year_str].get(state, float("nan")))
        else:
            df["% RES target"] = float("nan")

        df["% RES target"] = df["% RES target"].apply(
            lambda x: "N/A" if pd.isna(x) else round(x * 100, 2))
        df["% CES target"] = df["% CES target"].apply(
            lambda x: "N/A" if pd.isna(x) else round(x * 100, 2))

        df = df[["% RES", "% RES target", "% CES", "% CES target"]].round(2)
        if multiple_scenarios:
            results[name] = df.sort_index()
        else:
            results[year] = df.sort_index()
    return results


def plot_network_generation_and_transmission(n, key, tech_colors, nice_names, regions_onshore, title_year=True):

    # Define generation and link carriers
    gen_carriers = {
        "onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop",
        "csp", "nuclear", "geothermal", "ror", "PHS", "battery discharger"
    }
    link_carriers = {
        "OCGT", "CCGT", "coal", "oil", "biomass", "urban central solid biomass CHP",
        "urban central gas CHP", "battery discharger"
    }

    # Generator and storage capacity
    gen_p_nom_opt = n.generators[n.generators.carrier.isin(gen_carriers)]
    gen_p_nom_opt = gen_p_nom_opt.groupby(["bus", "carrier"]).p_nom_opt.sum()

    sto_p_nom_opt = n.storage_units[n.storage_units.carrier.isin(gen_carriers)]
    sto_p_nom_opt = sto_p_nom_opt.groupby(["bus", "carrier"]).p_nom_opt.sum()

    # Link capacity (scaled by efficiency)
    link_mask = (
        n.links.efficiency.notnull()
        & (n.links.p_nom_opt > 0)
        & n.links.carrier.isin(link_carriers)
    )
    electricity_links = n.links[link_mask].copy()
    electricity_links["electric_output"] = electricity_links.p_nom_opt * electricity_links.efficiency
    link_p_nom_opt = electricity_links.groupby(["bus1", "carrier"]).electric_output.sum()
    link_p_nom_opt.index = link_p_nom_opt.index.set_names(["bus", "carrier"])

    # Combine all sources
    bus_carrier_capacity = pd.concat([gen_p_nom_opt, sto_p_nom_opt, link_p_nom_opt])
    bus_carrier_capacity = bus_carrier_capacity.groupby(level=[0, 1]).sum()
    bus_carrier_capacity = bus_carrier_capacity[bus_carrier_capacity > 0]

    # Keep only buses with valid coordinates
    valid_buses = n.buses.dropna(subset=["x", "y"])
    valid_buses = valid_buses[
        (valid_buses["x"] > -200) & (valid_buses["x"] < 200) &
        (valid_buses["y"] > -90) & (valid_buses["y"] < 90)
    ]

    # Normalize bus names (remove " low voltage")
    def normalize_bus_name(bus_name):
        return bus_name.replace(" low voltage", "")

    bus_carrier_capacity = bus_carrier_capacity.reset_index()
    bus_carrier_capacity['bus'] = bus_carrier_capacity['bus'].apply(normalize_bus_name)
    bus_carrier_capacity['carrier'] = bus_carrier_capacity['carrier'].replace({
        'offwind-ac': 'offwind',
        'offwind-dc': 'offwind'
    })
    bus_carrier_capacity = bus_carrier_capacity.groupby(['bus', 'carrier'], as_index=False).sum()
    bus_carrier_capacity = bus_carrier_capacity.set_index(['bus', 'carrier']).squeeze()
    capacity_df = bus_carrier_capacity.unstack(fill_value=0)
    capacity_df = capacity_df.loc[capacity_df.index.intersection(valid_buses.index)]

    # Setup map background
    fig, ax = plt.subplots(figsize=(28, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    bbox = box(-130, 20, -60, 50)
    regions_onshore_clipped = regions_onshore.to_crs(epsg=4326).clip(bbox)

    regions_onshore_clipped.plot(
        ax=ax,
        facecolor='whitesmoke',
        edgecolor='gray',
        alpha=0.7,
        linewidth=0.5,
        zorder=0,
    )

    # Plot only DC links 
    original_links = n.links.copy()
    n.links = n.links[n.links.carrier == "DC"]

    line_scale = 5e3
    n.plot(
        ax=ax,
        bus_sizes=0,
        bus_alpha=0,
        line_widths=n.lines.s_nom_opt / line_scale,
        link_widths=n.links.p_nom_opt / line_scale,
        line_colors='teal',
        link_colors='turquoise',
        color_geomap=False,
        flow=None,
    )
    n.links = original_links

    # Draw pie charts at bus locations 
    pie_scale = 0.003
    for bus_id, capacities in capacity_df.iterrows():
        x, y = valid_buses.loc[bus_id, ['x', 'y']]
        if not bbox.contains(gpd.points_from_xy([x], [y])[0]):
            continue
        values = capacities.values
        total = values.sum()
        if total == 0:
            continue
        size = np.clip(np.sqrt(total) * pie_scale, 0.1, 1.5)
        colors = [tech_colors.get(c, 'gray') for c in capacities.index]
        start_angle = 0
        for val, color in zip(values, colors):
            if val == 0:
                continue
            angle = 360 * val / total
            wedge = Wedge(
                center=(x, y),
                r=size,
                theta1=start_angle,
                theta2=start_angle + angle,
                facecolor=color,
                edgecolor='k',
                linewidth=0.3,
                transform=ccrs.PlateCarree()._as_mpl_transform(ax),
                zorder=5,
            )
            ax.add_patch(wedge)
            start_angle += angle

    # Legends 

    # Bus capacity (marker size legend)
    bus_caps = [5, 10, 50]
    bus_marker_sizes = [np.sqrt(cap) * pie_scale * 1000 for cap in bus_caps]
    bus_patches = [
        mlines.Line2D([], [], linestyle='None', marker='o', color='gray',
                      markersize=size, label=f"{cap} GW", markerfacecolor='gray',
                      alpha=0.5)
        for cap, size in zip(bus_caps, bus_marker_sizes)
    ]
    bus_legend = ax.legend(
        handles=bus_patches,
        title="Bus Capacity",
        title_fontsize=12,
        fontsize=10,
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1.085, 1.0),
        labelspacing=1.4,
    )

    # AC line capacity
    ac_caps = [5e3, 20e3, 50e3]
    ac_patches = [
        mlines.Line2D([], [], color='teal', linewidth=cap / line_scale, label=f"{int(cap/1e3)} GW")
        for cap in ac_caps
    ]
    ac_legend = ax.legend(
        handles=ac_patches,
        title="AC Line Capacity",
        title_fontsize=12,
        fontsize=10,
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1.1, 0.83),
        labelspacing=1.1
    )

    # DC link capacity
    dc_caps = [2e3, 5e3, 10e3]
    dc_patches = [
        mlines.Line2D([], [], color='turquoise', linewidth=cap / line_scale, label=f"{int(cap/1e3)} GW")
        for cap in dc_caps
    ]
    dc_legend = ax.legend(
        handles=dc_patches,
        title="DC Link Capacity",
        title_fontsize=12,
        fontsize=10,
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1.1, 0.68),
        labelspacing=1.1
    )

        # Carrier legend (force preferred order with nice_names)
    preferred_order = [
        "Coal", "Gas CCGT", "Gas OCGT", "Gas CHP",
        "Oil", "Nuclear",
        "Biomass", "Biomass CHP",
        "Conventional hydro", "Run-of-River hydro", "Pumped hydro storage",
        "Utility-scale solar", "Rooftop solar", "CSP",
        "Onshore wind", "Offshore wind",
        "Battery"
    ]

    # Map raw carriers to pretty names
    carriers_present = {
        nice_names.get(c, c): c
        for c in capacity_df.columns if capacity_df[c].sum() > 0
    }

    # Keep only the carriers that are in the preferred order and present in data
    ordered_carriers = [
        c for c in preferred_order if c in carriers_present.keys()
    ]

    # Build handles with the correct color from the raw key
    carrier_handles = [
        mpatches.Patch(
            color=tech_colors.get(carriers_present[c], "gray"),
            label=c
        )
        for c in ordered_carriers
    ]

    carrier_legend = ax.legend(
        handles=carrier_handles,
        title="Technology",
        title_fontsize=13,
        fontsize=11,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.125, 0.54),
        ncol=2,
        labelspacing=1.05,
    )

    ax.add_artist(bus_legend)
    ax.add_artist(ac_legend)
    ax.add_artist(dc_legend)
    ax.add_artist(carrier_legend)

    # Set map extent
    ax.set_extent([-125, -65, 20, 55], crs=ccrs.PlateCarree())
    ax.autoscale(False)

    # Title
    year = key[-4:]
    ax.set_title(
        f"Installed electricity generation and transmission capacity – {year if title_year else key}", 
        fontsize=14
    )

    plt.tight_layout()
    plt.show()



def compute_installed_capacity_by_carrier(
    networks,
    nice_names=None,
    display_result=True,
    column_year=True
):
    totals_by_carrier = {}

    for name, net in networks.items():
        # Conventional generator and storage carriers
        gen_carriers = {
            "onwind", "offwind", "solar", "solar rooftop",
            "csp", "nuclear", "geothermal", "ror", "PHS", "hydro",
        }
        link_carriers = {
            "OCGT", "CCGT", "coal", "oil", "biomass",
            "urban central solid biomass CHP", "urban central gas CHP",
            "battery discharger"
        }

        # Generators
        gen = net.generators.copy()
        gen = gen[gen.carrier.isin(gen_carriers)]
        gen_totals = gen.groupby("carrier")["p_nom_opt"].sum()

        # Storage units
        sto = net.storage_units.copy()
        sto = sto[sto.carrier.isin(gen_carriers)]
        sto_totals = sto.groupby("carrier")["p_nom_opt"].sum()

        # Links (efficiency-scaled output)
        links = net.links.copy()
        mask = (
            links.efficiency.notnull()
            & (links.p_nom_opt > 0)
            & links.carrier.isin(link_carriers)
        )
        links = links[mask]

        links_totals = links.groupby("carrier").apply(
            lambda df: (df["p_nom_opt"] * df["efficiency"]).sum()
        )

        # Combine all contributions
        all_totals = pd.concat([gen_totals, sto_totals, links_totals])
        all_totals = all_totals.groupby(all_totals.index).sum()
        all_totals = all_totals[all_totals > 0]
        totals_by_carrier[name] = all_totals

    # Assemble final dataframe
    carrier_capacity_df = pd.DataFrame(totals_by_carrier).fillna(0)

    # Extract years and sort
    if column_year:
        carrier_capacity_df.columns = [int(name[-4:]) for name in carrier_capacity_df.columns]
    carrier_capacity_df = carrier_capacity_df[sorted(carrier_capacity_df.columns)]

    # Convert to GW
    carrier_capacity_df = (carrier_capacity_df / 1000).round(2)

    # Rename index if nice_names is provided
    if nice_names:
        carrier_capacity_df = carrier_capacity_df.rename(index=nice_names)

    # Apply preferred order
    preferred_order = [
        "Coal", "Gas CCGT", "Gas OCGT", "Gas CHP",
        "Oil", "Nuclear",
        "Biomass", "Biomass CHP", "Geothermal",        
        "Conventional hydro", "Run-of-River hydro", "Pumped hydro storage",
        "Onshore wind", "Offshore wind",
        "Utility-scale solar", "Rooftop solar", "CSP",
        "Battery"
    ]
    available = carrier_capacity_df.index.tolist()
    ordered_index = [c for c in preferred_order if c in available] + \
                    [c for c in available if c not in preferred_order]
    carrier_capacity_df = carrier_capacity_df.loc[ordered_index]

    if display_result:
        print("\nInstalled capacity by technology (GW)\n")
        display(carrier_capacity_df)

    return carrier_capacity_df



def compute_system_costs(network, rename_capex, rename_opex, name_tag):

    costs_raw = network.statistics(
    )[['Capital Expenditure', 'Operational Expenditure']]
    year_str = name_tag[-4:]

    # CAPEX
    capex_raw = costs_raw[['Capital Expenditure']].reset_index()
    capex_raw['tech_label'] = capex_raw['carrier'].map(
        rename_capex).fillna(capex_raw['carrier'])
    capex_raw['main_category'] = capex_raw['tech_label']

    capex_grouped = capex_raw.groupby('tech_label', as_index=False).agg({
        'Capital Expenditure': 'sum',
        'main_category': 'first'
    })
    capex_grouped['cost_type'] = 'Capital expenditure'
    capex_grouped.rename(
        columns={'Capital Expenditure': 'cost_billion'}, inplace=True)
    capex_grouped['cost_billion'] /= 1e9
    capex_grouped['year'] = year_str
    capex_grouped['scenario'] = name_tag

    # OPEX
    opex_raw = costs_raw[['Operational Expenditure']].reset_index()
    opex_raw['tech_label'] = opex_raw['carrier'].map(
        rename_opex).fillna(opex_raw['carrier'])
    opex_raw['main_category'] = opex_raw['tech_label']

    opex_grouped = opex_raw.groupby('tech_label', as_index=False).agg({
        'Operational Expenditure': 'sum',
        'main_category': 'first'
    })
    opex_grouped['cost_type'] = 'Operational expenditure'
    opex_grouped.rename(
        columns={'Operational Expenditure': 'cost_billion'}, inplace=True)
    opex_grouped['cost_billion'] /= 1e9
    opex_grouped['year'] = year_str
    opex_grouped['scenario'] = name_tag

    # Additional OPEX from link-based conventional generators
    carriers_of_interest = ["coal", "gas", "oil", "biomass"]
    results = []
    for carrier in carriers_of_interest:
        links = network.links[network.links.carrier == carrier]
        for link_id in links.index:
            try:
                p0 = network.links_t.p0[link_id]  # fuel input (negative)
                bus0 = links.loc[link_id, 'bus0']
                fuel_price = network.buses_t.marginal_price[bus0]
                weightings = network.snapshot_weightings['objective']

                # Fuel cost (positive)
                fuel_cost = (p0 * fuel_price * weightings).sum()

                # Other OPEX (marginal cost of link, positive)
                marginal_cost = links.loc[link_id, 'marginal_cost']
                other_opex = (p0 * marginal_cost * weightings).sum()

                total_opex = fuel_cost + other_opex

                results.append({
                    'tech_label': f'{carrier} (power)',
                    'main_category': f'{carrier} (power)',
                    'cost_type': 'Operational expenditure',
                    'cost_billion': total_opex / 1e9,
                    'year': year_str,
                    'scenario': name_tag
                })
            except KeyError:
                continue

    link_opex_df = pd.DataFrame(results)

    # Apply renaming also to link-based OPEX
    link_opex_df['tech_label'] = link_opex_df['tech_label'].replace(
        rename_opex)
    link_opex_df['main_category'] = link_opex_df['tech_label']

    return pd.concat([capex_grouped, opex_grouped, link_opex_df], ignore_index=True)


def assign_macro_category(row, categories_capex, categories_opex):
    if row['cost_type'] == 'Capital expenditure':
        return categories_capex.get(row['tech_label'], 'Other')
    elif row['cost_type'] == 'Operational expenditure':
        return categories_opex.get(row['tech_label'], 'Other')
    else:
        return 'Other'


def calculate_total_inputs_outputs_ft(
    networks: dict,
    ft_carrier: str = "Fischer-Tropsch",
    year_index: bool = True,
    include_scenario: bool = False,
    scenario_regex: str = r"(scenario_\d{2})",
    keep_empty: bool = False,
    wide: bool = False,
    scenario_first_level: bool = True,
):
    """
    Calculate Fischer-Tropsch (FT) process inputs/outputs per network.

    For each network (scenario-year):
      - Used electricity (TWh)    : links_t.p3
      - Used hydrogen (TWh / t)   : links_t.p0
      - Used CO2 (Mt)             : links_t.p2  (sum of flow; negative convention => absolute used)
      - Produced e-kerosene (TWh) : links_t.p1  (output taken as negative of p1 aggregated)

    Parameters
    ----------
    networks : dict
        {name: pypsa.Network}
    ft_carrier : str
        Link carrier name to filter Fischer-Tropsch units.
    year_index : bool
        If True (default) the output keeps a numeric Year column (and sorts by it).
        If False, original network name retained in 'Year' column (for backward compatibility).
    include_scenario : bool
        If True, adds a 'Scenario' column (parsed with scenario_regex; 'Base' if not found).
    scenario_regex : str
        Regex to extract scenario label from the network name.
    keep_empty : bool
        If True, include rows for scenarios without FT links (filled with zeros).
    wide : bool
        If True, returns a pivoted (wide) table:
            - Index: Year (or Scenario) depending on flags
            - Columns: metrics (or MultiIndex with scenario + metric)
    scenario_first_level : bool
        If wide=True and include_scenario=True: choose MultiIndex order:
            True  => (Scenario, Metric)
            False => (Metric, Scenario)

    Returns
    -------
    pd.DataFrame
        Long or wide formatted table with FT energy/material balances.

    Notes
    -----
    - Previous behavior preserved when include_scenario=False, wide=False.
    - Hydrogen tons conversion: TWh -> kWh -> kg -> t  (1 TWh = 1e9 kWh; LHV ≈ 33 kWh/kg).
    """

    rows = []

    for name, net in networks.items():
        # Extract scenario
        scenario_match = re.search(
            scenario_regex, name) if include_scenario else None
        scenario = scenario_match.group(1) if (
            scenario_match and include_scenario) else ("Base" if include_scenario else None)

        ft_links = net.links[net.links.carrier == ft_carrier]
        if ft_links.empty:
            if keep_empty:
                # Extract year anyway
                year_match = re.search(r"\d{4}", name)
                yr_val = int(year_match.group()) if year_match else None
                rows.append({
                    "Year": (yr_val if year_index else name),
                    "Scenario": scenario,
                    "Used electricity (TWh)": 0.0,
                    "Used hydrogen (TWh)": 0.0,
                    "Used hydrogen (t)": 0.0,
                    "Used CO2 (Mt)": 0.0,
                    "Produced e-kerosene (TWh)": 0.0,
                })
            continue

        ft_ids = ft_links.index

        # Timestep hours
        timestep_h = (
            (net.snapshots[1] - net.snapshots[0]).total_seconds() / 3600
            if len(net.snapshots) > 1 else 1.0
        )

        def _energy(df, ids):
            sel = df.reindex(columns=ids, fill_value=0.0)
            return (sel * timestep_h).sum().sum() / 1e6  # MWh -> TWh

        elec_twh = _energy(net.links_t.p3, ft_ids)
        h2_twh = _energy(net.links_t.p0, ft_ids)
        kerosene_twh_raw = _energy(net.links_t.p1, ft_ids)
        # p1 likely negative for output (depending on sign convention). Keep positive production:
        kerosene_twh = -kerosene_twh_raw

        def _co2(df, ids):
            sel = df.reindex(columns=ids, fill_value=0.0)
            # Sum raw (can be negative). For clarity assume negative means consumption:
            total = (sel * timestep_h).sum().sum() / 1e6  # t -> Mt
            # Use absolute if negative is consumption; keep sign if desired. Here make it positive usage.
            return -total if total < 0 else total

        co2_mt = _co2(net.links_t.p2, ft_ids)

        # Hydrogen tons
        h2_tons = h2_twh * 1e9 / 33 / 1000  # TWh -> kWh -> kg -> t

        year_match = re.search(r"\d{4}", name)
        if not year_match:
            continue
        year_val = int(year_match.group())

        row = {
            "Year": (year_val if year_index else name),
            "Used electricity (TWh)": elec_twh,
            "Used hydrogen (TWh)": h2_twh,
            "Used hydrogen (t)": h2_tons,
            "Used CO2 (Mt)": co2_mt,
            "Produced e-kerosene (TWh)": kerosene_twh,
        }
        if include_scenario:
            row["Scenario"] = scenario
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "Year",
            *(["Scenario"] if include_scenario else []),
            "Used electricity (TWh)",
            "Used hydrogen (TWh)",
            "Used hydrogen (t)",
            "Used CO2 (Mt)",
            "Produced e-kerosene (TWh)"
        ])

    df = pd.DataFrame(rows)

    # Ensure proper dtypes
    if year_index and "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Sort
    sort_cols = []
    if include_scenario:
        sort_cols.append("Scenario")
    if "Year" in df.columns:
        sort_cols.append("Year")
    if sort_cols:
        df = df.sort_values(sort_cols)

    if not wide:
        return df.reset_index(drop=True)

    # Wide formatting
    value_cols = [
        "Used electricity (TWh)",
        "Used hydrogen (TWh)",
        "Used hydrogen (t)",
        "Used CO2 (Mt)",
        "Produced e-kerosene (TWh)"
    ]

    if include_scenario:
        if scenario_first_level:
            # Columns: Scenario -> Metric
            pivot_index = "Year" if year_index else None
            if pivot_index:
                wide_df = df.pivot(index=pivot_index,
                                   columns="Scenario", values=value_cols)
                # Reorder to Scenario first => (Scenario, Metric)
                wide_df = wide_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
            else:
                # No year index: just aggregate by scenario (single row per scenario)
                wide_df = df.groupby("Scenario")[value_cols].sum()
        else:
            # Columns: Metric -> Scenario
            pivot_index = "Year" if year_index else None
            if pivot_index:
                wide_df = df.pivot(index=pivot_index,
                                   columns="Scenario", values=value_cols)
            else:
                wide_df = df.groupby("Scenario")[value_cols].sum().T
        return wide_df

    # include_scenario False in wide mode
    if year_index:
        wide_df = df.set_index("Year")[value_cols]
    else:
        # 'Year' holds original name when year_index=False
        wide_df = df.set_index("Year")[value_cols]
    return wide_df


def compute_ekerosene_production_cost_by_region(
    networks: dict,
    year_title: bool = True,
    aggregate: bool = False,
    wide: bool = False,
    scenario_regex: str = r"(?:scenario_(\d{2})|Base)",
    scenario_first_level: bool = True,
    min_production_twh: float = 1e-3,
    return_weighted_average: bool = False,
    expected_scenarios: list | None = None,
    expected_years: list | None = None,
    fill_cost_with: float | None = None,  # None → NaN
):
    """
    Extended: ensure ALL expected scenarios and years (even with no production)
    appear in the output (long and wide). Missing combinations get:
        Production (TWh) = 0
        Costs = NaN (or fill_cost_with if provided)

    Parameters added
    ----------------
    expected_scenarios : list | None
        Full list of scenarios to include (e.g. ['Base','scenario_01',...]).
        If None -> inferred from networks.
    expected_years : list | None
        Full list of model years to include (e.g. [2023,2030,2035,2040]).
        If None -> inferred from networks.
    fill_cost_with : float | None
        Value to fill missing cost metrics. Default None keeps NaN.
    """
    import re
    import pandas as pd
    from itertools import product

    all_rows = []

    # Infer scenarios / years if not provided
    inferred_scenarios = set()
    inferred_years = set()

    for name in networks.keys():
        m = re.search(r'(?:scenario_(\d{2})|Base)_(\d{4})', name)
        if not m:
            continue
        scen = f"scenario_{m.group(1)}" if m.group(1) else "Base"
        yr = int(m.group(2))
        inferred_scenarios.add(scen)
        inferred_years.add(yr)

    if expected_scenarios is None:
        expected_scenarios = sorted(
            inferred_scenarios, key=lambda x: (x != "Base", x))
    if expected_years is None:
        expected_years = sorted(inferred_years)

    # --- Energy content conversion for e-kerosene ---
    # 1 MWh = 3600 MJ
    # Energy density of kerosene ≈ 34 MJ/L
    # 1 US gallon = 3.78541 L
    # Energy per liter = 34 / 3600 = 0.009444... MWh/L
    # Energy per gallon = 0.009444... × 3.78541 ≈ 0.0357 MWh/gallon
    MWH_PER_GALLON = 34.0 / 3600.0 * 3.78541

    cost_cols = [
        "Electricity cost (USD/MWh e-kerosene)",
        "Hydrogen cost (USD/MWh e-kerosene)",
        "CO2 cost (USD/MWh e-kerosene)",
        "Total production cost (USD/MWh e-kerosene)",
        "Total cost (USD/gallon e-kerosene)",
    ]
    metrics_all = ["Production (TWh)"] + cost_cols

    for name, net in networks.items():
        year_match = re.search(r"\d{4}", name)
        if not year_match:
            continue
        year = int(year_match.group())

        scen_match = re.search(scenario_regex, name)
        if scen_match:
            scen_raw = scen_match.group(0)
            scenario = "Base" if "Base" in scen_raw else scen_raw
        else:
            scenario = "Base"

        ft_links = net.links[
            (net.links.carrier.str.contains("Fischer-Tropsch", case=False, na=False)) &
            (
                (net.links.get("p_nom_opt", 0) > 0) |
                ((net.links.get("p_nom", 0) > 0) &
                 (net.links.get("p_nom_extendable", False) == False))
            )
        ]
        if ft_links.empty:
            continue

        ft_link_ids = [
            l for l in ft_links.index
            if all(
                hasattr(net.links_t, p) and (
                    l in getattr(net.links_t, p).columns)
                for p in ["p0", "p1", "p2", "p3"]
            )
        ]
        if not ft_link_ids:
            continue

        timestep_hours = (
            (net.snapshots[1] - net.snapshots[0]).total_seconds() / 3600
            if len(net.snapshots) > 1 else 1.0
        )

        for link in ft_link_ids:
            try:
                region = net.buses.at[ft_links.at[link, "bus1"], "grid_region"]
            except KeyError:
                continue
            if pd.isna(region):
                continue

            try:
                elec_price = net.buses_t.marginal_price[ft_links.at[link, "bus3"]]
                h2_price = net.buses_t.marginal_price[ft_links.at[link, "bus0"]]
                co2_price = net.buses_t.marginal_price[ft_links.at[link, "bus2"]]
            except KeyError:
                continue

            p1 = -net.links_t.p1[link] * timestep_hours  # product out (MWh)
            p3 = net.links_t.p3[link] * timestep_hours  # elec in
            p0 = net.links_t.p0[link] * timestep_hours  # H2 in
            p2 = net.links_t.p2[link].clip(
                upper=0) * timestep_hours  # CO2 (t, negative)

            prod_twh = p1.sum() / 1e6
            if prod_twh < min_production_twh:
                continue
            fuel_output_safe = p1.sum()
            if fuel_output_safe <= 0:
                continue

            elec_cost = (p3 * elec_price).sum() / fuel_output_safe
            h2_cost = (p0 * h2_price).sum() / fuel_output_safe
            co2_cost = (-p2 * co2_price).sum() / fuel_output_safe
            total_cost = elec_cost + h2_cost + co2_cost
            total_cost_gallon = total_cost * MWH_PER_GALLON

            all_rows.append({
                "Scenario": scenario,
                "Year": year,
                "Grid Region": region,
                "Production (TWh)": prod_twh,
                "Electricity cost (USD/MWh e-kerosene)": elec_cost,
                "Hydrogen cost (USD/MWh e-kerosene)": h2_cost,
                "CO2 cost (USD/MWh e-kerosene)": co2_cost,
                "Total production cost (USD/MWh e-kerosene)": total_cost,
                "Total cost (USD/gallon e-kerosene)": total_cost_gallon,
            })

    # ---------------- legacy print mode ----------------
    if not (aggregate or wide):
        # unchanged legacy behavior (only for networks with data)
        if not all_rows:
            print("No e-kerosene production found.")
            return
        df_all = pd.DataFrame(all_rows)
        for (scen, yr), df_sub in df_all.groupby(["Scenario", "Year"]):
            def wavg(g, col):
                return (g[col]*g["Production (TWh)"]).sum()/g["Production (TWh)"].sum()
            grouped = df_sub.groupby("Grid Region").apply(
                lambda g: pd.Series({
                    "Production (TWh)": g["Production (TWh)"].sum(),
                    **{c: wavg(g, c) for c in cost_cols}
                })
            )
            grouped = grouped[grouped["Production (TWh)"]
                              >= min_production_twh]
            if grouped.empty:
                continue
            print(f"\n{yr if year_title else scen+'_'+str(yr)}:\n")
            display(
                grouped.round(2).style.format({
                    "Production (TWh)": "{:,.2f}",
                    **{c: "{:,.2f}" for c in cost_cols}
                }).hide(axis="index")
            )
            total_prod = grouped["Production (TWh)"].sum()
            if total_prod > 0:
                w_cost = (grouped["Total production cost (USD/MWh e-kerosene)"] *
                          grouped["Production (TWh)"]).sum()/total_prod
                w_cost_gallon = w_cost * MWH_PER_GALLON
                print(
                    f"Weighted average production cost: {w_cost:.2f} USD/MWh ({w_cost_gallon:.2f} USD/gallon)")
        return

    # ---------------- aggregated (long) ----------------
    if not all_rows:
        # build empty frame with expected combos
        base = []
        for scen, yr in product(expected_scenarios, expected_years):
            base.append({
                "Scenario": scen,
                "Year": yr,
                "Grid Region": None,
                "Production (TWh)": 0.0,
                **{c: (fill_cost_with if fill_cost_with is not None else float('nan')) for c in cost_cols}
            })
        df_empty = pd.DataFrame(base)
        return df_empty if not wide else pd.DataFrame()

    df_all = pd.DataFrame(all_rows)

    # weighted aggregation per Scenario-Year-Region
    def wavg_group(g, col):
        return (g[col]*g["Production (TWh)"]).sum()/g["Production (TWh)"].sum()

    grouped = (
        df_all.groupby(["Scenario", "Year", "Grid Region"])
        .apply(lambda g: pd.Series({
            "Production (TWh)": g["Production (TWh)"].sum(),
            **{c: wavg_group(g, c) for c in cost_cols}
        }))
        .reset_index()
    )

    # Collect all grid regions encountered
    grid_regions_all = sorted(grouped["Grid Region"].unique())

    # Insert missing scenario-year-region combinations
    existing_keys = set(
        zip(grouped.Scenario, grouped.Year, grouped["Grid Region"]))
    missing_rows = []
    for scen, yr, reg in product(expected_scenarios, expected_years, grid_regions_all):
        if (scen, yr, reg) not in existing_keys:
            missing_rows.append({
                "Scenario": scen,
                "Year": yr,
                "Grid Region": reg,
                "Production (TWh)": 0.0,
                **{c: (fill_cost_with if fill_cost_with is not None else float('nan')) for c in cost_cols}
            })
    if missing_rows:
        grouped = pd.concat(
            [grouped, pd.DataFrame(missing_rows)], ignore_index=True)

    # Sort
    grouped = grouped.sort_values(["Scenario", "Year", "Grid Region"])

    if wide:
        # Pivot: index -> (Grid Region, Year) if multiple years
        multi = grouped.pivot_table(
            index=["Grid Region", "Year"],
            columns="Scenario",
            values=metrics_all,
            aggfunc="first"
        )

        # Reorder columns so scenarios in expected_scenarios order
        # Current columns: (metric, scenario) -> swap if needed
        if scenario_first_level:
            # we want (Scenario, Metric)
            multi = multi.swaplevel(0, 1, axis=1)

        # Ensure all scenarios present
        # Build full column MultiIndex
        if scenario_first_level:
            full_cols = pd.MultiIndex.from_product(
                [expected_scenarios, metrics_all],
                names=multi.columns.names
            )
        else:
            full_cols = pd.MultiIndex.from_product(
                [metrics_all, expected_scenarios],
                names=multi.columns.names
            )

        multi = multi.reindex(columns=full_cols)

        # If years missing, add them
        current_years = sorted({idx[1] for idx in multi.index})
        missing_years = [y for y in expected_years if y not in current_years]
        if missing_years:
            # create empty rows for each grid region
            grids = sorted({idx[0] for idx in multi.index})
            add_index = pd.MultiIndex.from_product(
                [grids, missing_years],
                names=multi.index.names
            )
            empty_df = pd.DataFrame(
                0.0,
                index=add_index,
                columns=multi.columns
            )
            # For cost columns set NaN (or fill value)
            if scenario_first_level:
                for scen in expected_scenarios:
                    for cost_col in cost_cols:
                        col = (scen, cost_col)
                        if fill_cost_with is None:
                            empty_df[col] = float('nan')
                        else:
                            empty_df[col] = fill_cost_with
            else:
                for cost_col in cost_cols:
                    for scen in expected_scenarios:
                        col = (cost_col, scen)
                        if fill_cost_with is None:
                            empty_df[col] = float('nan')
                        else:
                            empty_df[col] = fill_cost_with
            multi = pd.concat([multi, empty_df]).sort_index()

        # Drop Year level if single expected year
        if len(expected_years) == 1:
            multi.index = multi.index.droplevel("Year")

        return multi if not return_weighted_average else (multi, None)

    # Long form
    return grouped if not return_weighted_average else (grouped, None)


#### VALIDATION HELPERS FUNCTIONS #####

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
                     "Solar, tide, wave, fuel cell", "Tide and wave"], inplace=True)

    # Filter the DataFrame to only include relevant energy sources
    data = data.loc[["Nuclear", "Coal", "Natural gas", "Oil", "Geothermal",
                     "Hydro", "PHS", "Solar", "Wind", "Biomass"], :]
    return data


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
        # "Gas": "Fossil fuels",
        "Bioenergy": "Biomass",
        # "Coal": "Fossil fuels",
        "Other Fossil": "Fossil fuels"
    })

    capacity_ember = capacity_ember.groupby("Variable").sum()
    capacity_ember.columns = ["Ember data"]

    return capacity_ember


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
                     "Solar, tide, wave, fuel cell", "Tide and wave"], inplace=True)

    # Filter the DataFrame to only include relevant energy sources
    data = data.loc[["Nuclear", "Fossil fuels", "Geothermal",
                     "Hydro", "PHS", "Solar", "Wind", "Biomass"], :]

    return data


def get_demand_pypsa(network):
    """
    Get the total electricity demand from the PyPSA-Earth network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        float: Total electricity demand in TWh.
    """
    demand_pypsa = network.loads_t.p_set.multiply(
        network.snapshot_weightings.objective, axis=0).sum().sum() / 1e6
    demand_pypsa = demand_pypsa.round(4)
    return demand_pypsa


def preprocess_eia_demand(path, horizon):
    statewise_df = pd.read_excel(path, sheet_name="Data")

    demand_df = statewise_df.loc[statewise_df['MSN'] == 'ESTXP']
    demand_df.set_index('State', inplace=True)

    # data is in million kWh (GWh) - hence dividing by 1e3 to get the data in TWh
    demand_df = demand_df[int(horizon)] / 1e3
    demand_df = demand_df.to_frame()
    demand_df.columns = ["EIA"]

    demand_df.drop(["US"], axis=0, inplace=True)
    return demand_df


def plot_stacked_costs_by_year_plotly(cost_data, cost_type_label, tech_colors=None, index='year'):
    # Filter data
    data_filtered = cost_data[
        (cost_data['cost_type'] == cost_type_label) &
        (cost_data['cost_billion'] != 0)
    ].copy()

    if data_filtered.empty:
        print("No data to plot.")
        return

    # Pivot table: index x tech_label
    pivot_table = data_filtered.pivot_table(
        index=index,
        columns='tech_label',
        values='cost_billion',
        aggfunc='sum'
    ).fillna(0)

    # Mapping: tech_label → macro category / main category
    label_to_macro = data_filtered.set_index(
        'tech_label')['macro_category'].to_dict()
    label_to_category = data_filtered.set_index(
        'tech_label')['main_category'].to_dict()

    # Desired macro-category order
    desired_macro_order = [
        'Hydrogen & e-fuels', 'Biofuels synthesis', 'DAC', 'End-uses', 'Industry',
        'Power & heat generation', 'Storage', 'Transmission & distribution',
        'Emissions', 'Other'
    ]
    macro_order_map = {macro: i for i, macro in enumerate(desired_macro_order)}

    # Sort tech labels by macro_category + appearance order
    all_labels = data_filtered['tech_label'].drop_duplicates().tolist()
    ordered_labels = sorted(
        all_labels,
        key=lambda lbl: (macro_order_map.get(
            label_to_macro.get(lbl, 'Other'), 999), all_labels.index(lbl))
    )

    # Reorder pivot table
    pivot_table = pivot_table[ordered_labels[::-1]]  # reverse for stacking

    # Assign colors
    def get_color(label):
        category = label_to_category.get(label, label)
        return tech_colors.get(category, '#999999') if tech_colors else '#999999'

    color_values = {label: get_color(label) for label in pivot_table.columns}

    # Create Plotly figure
    fig = go.Figure()

    x_vals = pivot_table.index.astype(str)

    # One trace per tech — works with negative values + interactive legend
    for label in pivot_table.columns:
        y_series = pivot_table[label]
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_series,
            name=label,
            marker=dict(color=color_values[label]),
            hovertemplate=f"%{{x}}<br>{label}: %{{y:.2f}}B USD<extra></extra>"
        ))

    # Macro-category legend block (annotation)
    grouped_labels = defaultdict(list)
    for label in ordered_labels:
        macro = label_to_macro.get(label, 'Other')
        grouped_labels[macro].append(label)

    legend_text = ""
    for macro in desired_macro_order:
        if macro in grouped_labels:
            legend_text += f"<b>{macro}</b><br>"
            for label in grouped_labels[macro]:
                color = color_values[label]
                legend_text += f"<span style='color:{color}'>▇</span> {label}<br>"

    # Add annotation for grouped legend
    fig.add_annotation(
        text=legend_text,
        showarrow=False,
        align="left",
        xref="paper", yref="paper",
        x=1.25, y=1,
        bordercolor='black',
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.95)",
        font=dict(size=14),
    )

    # Add 0-line for clarity
    fig.add_shape(
        type='line',
        xref='paper', x0=0, x1=1,
        yref='y', y0=0, y1=0,
        line=dict(color='black', width=1)
    )

    fig.update_layout(
        barmode="relative",
        title=dict(
            text=f"{cost_type_label} - Total system costs",
            font=dict(size=16)  # Titolo del grafico
        ),
        xaxis=dict(
            title=dict(text="Years (-)", font=dict(size=12)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text=f"{cost_type_label} (Billion USD)",
                       font=dict(size=12)),
            tickfont=dict(size=12)
        ),
        template="plotly_white",
        width=1400,
        height=700,
        margin=dict(l=40, r=300, t=50, b=50),
        legend_title=dict(text="Technologies", font=dict(size=14)),
        legend=dict(font=dict(size=12), traceorder='reversed'),
        showlegend=False,
    )

    fig.show()


def plot_float_bar_lcoe_dispatch_ranges(table_df, key, nice_names, use_scenario_names=False):
    import re
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract year from the key using regex
    year_match = re.search(r'\d{4}', key)
    year_str = year_match.group() if year_match else "Year N/A"

    carrier_list = [
        'CCGT lcoe (USD/MWh)', 'OCGT lcoe (USD/MWh)', 'coal lcoe (USD/MWh)', 'nuclear lcoe (USD/MWh)',
        'oil lcoe (USD/MWh)', 'urban central gas CHP lcoe (USD/MWh)',
        'urban central solid biomass CHP lcoe (USD/MWh)', 'biomass lcoe (USD/MWh)', 'geothermal lcoe (USD/MWh)',
        'hydro lcoe (USD/MWh)', 'onwind lcoe (USD/MWh)', 'ror lcoe (USD/MWh)', 'solar lcoe (USD/MWh)',
        'solar rooftop lcoe (USD/MWh)',
    ]

    buffer_left = 100
    buffer_right = 20

    global_min = table_df.xs('min', axis=1, level=1).min().min()
    global_max = table_df.xs('max', axis=1, level=1).max().max()

    x_min = min(-50, global_min - buffer_left)
    x_max = global_max + buffer_right

    regions = table_df.index.tolist()
    n_regions = len(regions)

    # Subplot grid size (2 columns)
    ncols = 2
    nrows = math.ceil(n_regions / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
        16, nrows * 5), constrained_layout=True)
    axs = axs.flatten()

    for idx, region in enumerate(regions):
        ax = axs[idx]

        # Filter only available carriers
        available_carriers = [
            c for c in carrier_list if c in table_df.columns.get_level_values(0)]
        if not available_carriers:
            ax.set_title(f"{region} - No carriers available", fontsize=12)
            ax.axis("off")
            continue

        table_lcoe_df = table_df[
            table_df.columns[table_df.columns.get_level_values(
                0).str.contains('lcoe')]
        ][available_carriers]

        table_lcoe_df_region = table_lcoe_df.loc[region, :]

        lcoe_tech_list = table_lcoe_df_region.xs('max', level=1).index

        for i, (start, end) in enumerate(zip(
            table_lcoe_df_region.xs('min', level=1).values,
            table_lcoe_df_region.xs('max', level=1).values
        )):
            str_attach = any(np.abs([start, end]) > 1e-3)
            width = end - start
            ax.broken_barh([(start, width)], (i - 0.4, 0.8),
                           hatch='///', edgecolor='white')
            start_label = f"${round(start, 2)}" if str_attach else ""
            end_label = f"${round(start + width, 2)}" if str_attach else ""
            ax.text(start - .7, i, start_label,
                    va='center', ha='right', fontsize=9)
            ax.text(start + width + .7, i, end_label,
                    va='center', ha='left', fontsize=9)

        raw_labels = [label.replace(" lcoe", "").replace(
            " (USD/MWh)", "") for label in lcoe_tech_list]
        clean_labels = [nice_names.get(lbl, lbl) for lbl in raw_labels]

        ax.set_yticks(range(len(lcoe_tech_list)))
        ax.set_yticklabels(clean_labels, fontsize=10)
        ax.set_xlabel("LCOE (USD/MWh)", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_title(
            f"\n{region} - {key if use_scenario_names else year_str}", fontsize=12)
        ax.grid(linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=9)

    # Hide any unused axes
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.show()


def compute_line_expansion_capacity(n):
    """
    Computes the line expansion capacity grouped by grid region.

    Parameters:
    n (pypsa.Network): The PyPSA network object.

    Returns:
    pd.DataFrame: DataFrame with line expansion capacity by grid region.
    """
    # Ensure 'grid_region' is present in lines
    if "grid_region" not in n.lines.columns:
        n.lines["grid_region"] = n.lines.bus0.map(n.buses.grid_region)

    # Ensure 'state' is present in lines
    if "state" not in n.lines.columns:
        n.lines["state"] = n.lines.bus0.map(n.buses.state)

    # Group by 'grid_region' and 'state' then sum the capacities
    line_exp_cap_grid = n.lines.groupby(
        "grid_region")[["s_nom", 's_nom_opt']].sum() / 1e3  # Convert to GW
    line_exp_cap_state = n.lines.groupby(
        "state")[["s_nom", 's_nom_opt']].sum() / 1e3  # Convert to GW

    return line_exp_cap_grid, line_exp_cap_state


def preprocess_res_ces_share_eia(eia_gen_data):
    eia_gen_data = eia_gen_data[eia_gen_data["YEAR"] == 2023]
    eia_gen_data = eia_gen_data[eia_gen_data["STATE"] != "US-Total"]
    eia_gen_data = eia_gen_data[eia_gen_data['TYPE OF PRODUCER']
                                == 'Total Electric Power Industry']
    eia_gen_data = eia_gen_data[(eia_gen_data["ENERGY SOURCE"] != "Total")
                                & (eia_gen_data["ENERGY SOURCE"] != "Other")]
    eia_gen_data.replace({"ENERGY SOURCE": {"Coal": "coal",
                                            "Hydroelectric Conventional": "hydro",
                                            "Pumped Storage": "PHS",
                                            "Solar Thermal and Photovoltaic": "solar",
                                            "Natural Gas": "gas",
                                            "Petroleum": "oil",
                                            "Wind": "wind",
                                            "Nuclear": "nuclear",
                                            "Geothermal": "geothermal",
                                            "Pumped Storage": "PHS",
                                            "Wood and Wood Derived Fuels": "biomass",
                                            "Other Biomass": "biomass",
                                            "Other Gases": "gas"}}, inplace=True)

    eia_gen_data["GENERATION (TWh)"] = eia_gen_data["GENERATION (Megawatthours)"] / 1e6
    eia_gen_data_df = eia_gen_data.groupby(["STATE", "ENERGY SOURCE"])[
        ["GENERATION (TWh)"]].sum().unstack(fill_value=0)
    eia_gen_data_df.columns = eia_gen_data_df.columns.droplevel(0)

    eia_res_carriers = ["solar", "wind",
                        "hydro", "geothermal", "biomass", "PHS"]
    eia_ces_carriers = eia_res_carriers + ["nuclear"]

    res_total = eia_gen_data_df[eia_res_carriers].sum(axis=1)
    ces_total = eia_gen_data_df[eia_ces_carriers].sum(axis=1)
    all_total = eia_gen_data_df[['PHS', 'biomass', 'coal', 'gas', 'geothermal', 'hydro', 'nuclear',
                                 'oil', 'solar', 'wind']].sum(axis=1)

    eia_gen_data_df["% Actual RES"] = (res_total / all_total) * 100
    eia_gen_data_df["% Actual CES"] = (ces_total / all_total) * 100

    return eia_gen_data_df


def compute_links_only_costs(network, name_tag):
    """
    Compute costs for power generation only:
    - Include: Fossil fuel links + all other generators (solar, wind, nuclear, etc.)
    - Exclude: Fossil fuel generators (coal, oil, gas, biomass) - these are end-uses
    """
    year_str = name_tag[-4:]

    # Get statistics separated by component type
    costs_detailed = network.statistics(
        groupby=None)[['Capital Expenditure', 'Operational Expenditure']]

    fossil_carriers = ["coal", "gas", "oil", "biomass"]

    # Build the final dataset by combining the right components
    final_results = []

    # 1. Add ALL Link costs (including fossil links for power generation)
    try:
        link_costs = costs_detailed.loc['Link'].reset_index()
        link_costs['tech_label'] = link_costs['carrier']

        # CAPEX from links
        link_capex = link_costs.groupby('tech_label', as_index=False).agg({
            'Capital Expenditure': 'sum'
        })
        for _, row in link_capex.iterrows():
            final_results.append({
                'tech_label': row['tech_label'],
                'cost_type': 'Capital expenditure',
                'cost_billion': row['Capital Expenditure'] / 1e9,
                'year': year_str,
                'scenario': name_tag
            })

        # OPEX from links
        link_opex = link_costs.groupby('tech_label', as_index=False).agg({
            'Operational Expenditure': 'sum'
        })
        for _, row in link_opex.iterrows():
            final_results.append({
                'tech_label': row['tech_label'],
                'cost_type': 'Operational expenditure',
                'cost_billion': row['Operational Expenditure'] / 1e9,
                'year': year_str,
                'scenario': name_tag
            })

    except KeyError:
        pass  # No links found

    # 2. Add NON-FOSSIL Generator costs (solar, wind, nuclear, etc.)
    try:
        gen_costs = costs_detailed.loc['Generator'].reset_index()
        gen_costs['tech_label'] = gen_costs['carrier']

        # Filter out fossil generators (keep only non-fossil generators)
        non_fossil_gen = gen_costs[~gen_costs['tech_label'].isin(
            fossil_carriers)]

        if len(non_fossil_gen) > 0:
            # CAPEX from non-fossil generators
            gen_capex = non_fossil_gen.groupby('tech_label', as_index=False).agg({
                'Capital Expenditure': 'sum'
            })
            for _, row in gen_capex.iterrows():
                final_results.append({
                    'tech_label': row['tech_label'],
                    'cost_type': 'Capital expenditure',
                    'cost_billion': row['Capital Expenditure'] / 1e9,
                    'year': year_str,
                    'scenario': name_tag
                })

            # OPEX from non-fossil generators
            gen_opex = non_fossil_gen.groupby('tech_label', as_index=False).agg({
                'Operational Expenditure': 'sum'
            })
            for _, row in gen_opex.iterrows():
                final_results.append({
                    'tech_label': row['tech_label'],
                    'cost_type': 'Operational expenditure',
                    'cost_billion': row['Operational Expenditure'] / 1e9,
                    'year': year_str,
                    'scenario': name_tag
                })

    except KeyError:
        pass  # No generators found

    # 3. Calculate and add FUEL COSTS for fossil fuel links
    fuel_cost_adjustments = {}

    for carrier in fossil_carriers:
        links = network.links[network.links.carrier == carrier]
        total_fuel_cost = 0

        for link_id in links.index:
            try:
                p0 = network.links_t.p0[link_id]
                fuel_bus = links.loc[link_id, 'bus0']
                fuel_price = network.buses_t.marginal_price[fuel_bus]
                weightings = network.snapshot_weightings['objective']

                # Calculate fuel cost (positive)
                fuel_cost = (p0 * fuel_price * weightings).sum()
                total_fuel_cost += fuel_cost

            except KeyError:
                continue

        if total_fuel_cost > 0:
            fuel_cost_adjustments[carrier] = total_fuel_cost / 1e9

    # 4. Modify fossil fuel link OPEX to add fuel costs and rename to (power)
    df_results = pd.DataFrame(final_results)

    # Find fossil link OPEX entries and modify them
    for carrier in fossil_carriers:
        if carrier in fuel_cost_adjustments:
            # Find the OPEX entry for this fossil carrier
            mask = (df_results['tech_label'] == carrier) & (
                df_results['cost_type'] == 'Operational expenditure')
            if mask.any():
                # Add fuel costs to existing OPEX
                df_results.loc[mask,
                               'cost_billion'] += fuel_cost_adjustments[carrier]
                # Rename to (power) version
                df_results.loc[mask, 'tech_label'] = f'{carrier} (power)'

    return df_results


def identify_power_generation_technologies(rename_techs_capex, rename_techs_opex, categories_capex, categories_opex):
    """
    Identify technologies for power generation only (including (power) versions of conventional fuels)
    """
    power_gen_techs = set()

    # Check CAPEX mappings
    for original_tech, intermediate_category in rename_techs_capex.items():
        if categories_capex.get(intermediate_category) == 'Power & heat generation':
            if intermediate_category != 'Heating':  # Exclude heating
                # Convert conventional fuels to (power) format
                if original_tech in ['coal', 'gas', 'oil', 'biomass']:
                    power_gen_techs.add(f'{original_tech} (power)')
                else:
                    power_gen_techs.add(original_tech)

    # Check OPEX mappings
    for original_tech, intermediate_category in rename_techs_opex.items():
        if categories_opex.get(intermediate_category) == 'Power & heat generation':
            if intermediate_category != 'Heating':  # Exclude heating
                # Convert conventional fuels to (power) format
                if original_tech in ['coal', 'gas', 'oil', 'biomass']:
                    power_gen_techs.add(f'{original_tech} (power)')
                else:
                    power_gen_techs.add(original_tech)

    return power_gen_techs


def plot_power_generation_details(cost_data, cost_type_label, power_techs,
                                  tech_colors=None, nice_names=None, tech_order=None, index='year'):
    """
    Plot interactive detailed breakdown of Power & heat generation technologies showing original tech_labels

    Parameters:
    - cost_data: DataFrame with cost data containing original tech_labels
    - cost_type_label: str, "Capital expenditure" or "Operational expenditure"  
    - power_techs: set of technology names that belong to Power & heat generation
    - tech_colors: dict, mapping from original tech_labels to colors
    - nice_names: dict, mapping from original tech_labels to display names
    """

    # Filter for Power & heat generation technologies
    power_data = cost_data[
        cost_data['tech_label'].isin(power_techs) &
        (cost_data['cost_type'] == cost_type_label) &
        (cost_data['cost_billion'] != 0)
    ].copy()

    # Aggregate technologies with the same name (e.g., Offshore Wind AC + DC)
    power_data = power_data.groupby(
        ['tech_label', 'cost_type', 'year', 'scenario'],
        as_index=False
    ).agg({
        'cost_billion': 'sum'
    })

    if power_data.empty:
        return

    # Create pivot table: years x technologies
    pivot_table = power_data.pivot_table(
        index=index,
        columns='tech_label',
        values='cost_billion',
        aggfunc='sum'
    ).fillna(0)

    # Sort technologies by total cost (largest first)
    if tech_order:
        available_techs = set(pivot_table.columns)
        ordered_techs = [
            tech for tech in tech_order if tech in available_techs]
        remaining_techs = available_techs - set(ordered_techs)
        ordered_techs.extend(sorted(remaining_techs))
    else:
        tech_totals = pivot_table.sum().sort_values(ascending=False)
        ordered_techs = tech_totals.index.tolist()

    # Get colors for each technology
    def get_tech_color(original_tech_label):
        if tech_colors and original_tech_label in tech_colors:
            return tech_colors[original_tech_label]
        # Try some variations if exact match not found
        elif tech_colors:
            for variant in [original_tech_label.lower(), original_tech_label.title()]:
                if variant in tech_colors:
                    return tech_colors[variant]
        # Default color if not found
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
        return colors[hash(original_tech_label) % len(colors)]

    # Create interactive plotly chart
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add traces for each technology (in order of importance)
    # Only exclude technologies that are ALL zeros
    for tech in ordered_techs:
        y_values = pivot_table[tech]
        # Skip only if ALL values are exactly zero
        if (y_values == 0).all():
            continue

        display_name = nice_names.get(tech, tech) if nice_names else tech
        color = get_tech_color(tech)

        fig.add_trace(go.Bar(
            name=display_name,
            x=pivot_table.index.astype(str),
            y=y_values,
            marker_color=color,
            hovertemplate=f"%{{x}}<br>{display_name}: %{{y:.2f}}B USD<extra></extra>"
        ))

    # Update layout for interactivity
    fig.update_layout(
        barmode='relative',  # Handle negative values correctly
        title=dict(
            text=f'Power Generation - {cost_type_label}',
            font=dict(size=16)
        ),
        xaxis=dict(
            title=dict(text="Years", font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text=f"{cost_type_label} (Billion USD)",
                       font=dict(size=12)),
            tickfont=dict(size=12)
        ),
        template="plotly_white",
        width=1400,
        height=700,
        margin=dict(l=40, r=300, t=50, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12),
            traceorder='reversed'
        )
    )

    # Add horizontal line at zero
    fig.add_hline(y=0, line_width=1, line_color="black")

    return fig


def compute_h2_efuels_costs(network, name_tag):
    """
    Compute costs for H2 and e-fuels technologies (links only)
    """
    year_str = name_tag[-4:]

    # Define H2 and e-fuels technologies
    h2_efuels_carriers = ["Alkaline electrolyzer large",
                          "Fischer-Tropsch", "PEM electrolyzer", "SOEC"]

    # Get statistics separated by component type
    costs_detailed = network.statistics(
        groupby=None)[['Capital Expenditure', 'Operational Expenditure']]

    final_results = []

    # Get ONLY Link costs for H2/e-fuels technologies
    try:
        link_costs = costs_detailed.loc['Link'].reset_index()
        link_costs['tech_label'] = link_costs['carrier']

        # Filter for H2/e-fuels technologies only
        h2_efuels_links = link_costs[link_costs['tech_label'].isin(
            h2_efuels_carriers)]

        if len(h2_efuels_links) > 0:
            # CAPEX from H2/e-fuels links
            link_capex = h2_efuels_links.groupby('tech_label', as_index=False).agg({
                'Capital Expenditure': 'sum'
            })
            for _, row in link_capex.iterrows():
                final_results.append({
                    'tech_label': row['tech_label'],
                    'cost_type': 'Capital expenditure',
                    'cost_billion': row['Capital Expenditure'] / 1e9,
                    'year': year_str,
                    'scenario': name_tag
                })

            # OPEX from H2/e-fuels links
            link_opex = h2_efuels_links.groupby('tech_label', as_index=False).agg({
                'Operational Expenditure': 'sum'
            })
            for _, row in link_opex.iterrows():
                final_results.append({
                    'tech_label': row['tech_label'],
                    'cost_type': 'Operational expenditure',
                    'cost_billion': row['Operational Expenditure'] / 1e9,
                    'year': year_str,
                    'scenario': name_tag
                })

    except KeyError:
        pass  # No links found

    return pd.DataFrame(final_results)


def calculate_lcoh_by_region(networks, h2_carriers, regional_fees, emm_mapping,
                             output_threshold=1.0, year_title=True,
                             electricity_price="marginal", grid_region_lcoe=None):
    """
    Compute weighted average LCOH by grid region and year, including CAPEX, OPEX,
    electricity cost (marginal or LCOE), and T&D fees.

    Parameters
    ----------
    electricity_price : {"marginal", "LCOE"}
        Method for electricity cost:
        - "marginal": use nodal marginal price of electricity
        - "LCOE":     use weighted average LCOE per grid region (requires grid_region_lcoe)
    grid_region_lcoe : pd.Series
        Weighted average LCOE (USD/MWh el) per grid region, indexed by "grid_region".
        Required if electricity_price="LCOE".
    """
    results = {}

    conv = 33.0   # MWh H2 per ton
    suffix = "USD/kg H2"

    for year_key, net in networks.items():
        scen_year = int(re.search(r"\d{4}", str(year_key)).group())

        links = net.links[net.links.carrier.isin(h2_carriers)]
        if links.empty:
            continue

        # Electricity consumption and H2 output
        p0, p1 = net.links_t.p0[links.index], net.links_t.p1[links.index]
        w = net.snapshot_weightings.generators
        cons = p0.clip(lower=0).multiply(w, axis=0)       # MWh electricity
        h2 = (-p1).clip(lower=0).multiply(w, axis=0)    # MWh H2
        h2_out = h2.sum()

        valid = h2_out > output_threshold
        if valid.sum() == 0:
            continue

        capex = links.loc[valid, "capital_cost"] * \
            links.loc[valid, "p_nom_opt"]
        opex = links.loc[valid, "marginal_cost"] * \
            cons.loc[:, valid].sum(axis=0)

        # Electricity cost depending on method
        elec_cost = pd.Series(0.0, index=valid.index[valid])
        if electricity_price == "marginal":
            for l in valid.index[valid]:
                bus = links.at[l, "bus0"]
                elec_cost[l] = (
                    cons[l] * net.buses_t.marginal_price[bus]).sum()
        elif electricity_price == "LCOE":
            if grid_region_lcoe is None:
                raise ValueError(
                    "grid_region_lcoe must be provided when electricity_price='LCOE'")
            for l in valid.index[valid]:
                bus = links.at[l, "bus0"]
                region = net.buses.at[bus, "grid_region"]
                avg_price = grid_region_lcoe.get(region, np.nan)
                elec_cost[l] = cons[l].sum() * avg_price
        else:
            raise ValueError("electricity_price must be 'marginal' or 'LCOE'")

        out_valid = h2_out[valid]

        # Normalize to USD/kg H2
        capex_val = capex / out_valid / conv
        opex_val = opex / out_valid / conv
        elec_val = elec_cost / out_valid / conv

        df = pd.DataFrame({
            f"Electrolysis CAPEX ({suffix})": capex_val,
            f"Electrolysis OPEX ({suffix})": opex_val,
            f"Electricity ({suffix})": elec_val,
            "h2_out": out_valid,
            "bus": links.loc[valid, "bus0"]
        })
        df["grid_region"] = df["bus"].map(net.buses["grid_region"])

        # Transmission & distribution fees
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh", "Distribution nom USD/MWh"]
        ].set_index("region")
        df["EMM"] = df["grid_region"].map(emm_mapping)

        fee_trans = df["EMM"].map(fee_map["Transmission nom USD/MWh"])
        fee_dist = df["EMM"].map(fee_map["Distribution nom USD/MWh"])
        elec_rate = cons.loc[:, valid].sum(
            axis=0) / out_valid   # MWh el / MWh H2
        fee_trans_val = (fee_trans * elec_rate / conv).reindex(df.index)
        fee_dist_val = (fee_dist * elec_rate / conv).reindex(df.index)

        # LCOH breakdown
        df[f"LCOH (excl. T&D fees) ({suffix})"] = (
            df[f"Electrolysis CAPEX ({suffix})"] +
            df[f"Electrolysis OPEX ({suffix})"] +
            df[f"Electricity ({suffix})"]
        )
        df[f"LCOH + Transmission fees ({suffix})"] = df[f"LCOH (excl. T&D fees) ({suffix})"] + \
            fee_trans_val
        df[f"LCOH + T&D fees ({suffix})"] = df[f"LCOH + Transmission fees ({suffix})"] + fee_dist_val

        # Dispatch label per regione
        dispatch_label = "Hydrogen Dispatch (tons per region)"

        # Weighted averages by grid region + dispatch per regione
        region_summary = (
            df.groupby("grid_region")
            .apply(lambda g: pd.Series({
                f"Electrolysis CAPEX ({suffix})": (g[f"Electrolysis CAPEX ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Electrolysis OPEX ({suffix})":  (g[f"Electrolysis OPEX ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Electricity ({suffix})":        (g[f"Electricity ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"LCOH (excl. T&D fees) ({suffix})": (g[f"LCOH (excl. T&D fees) ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Transmission fees ({suffix})":   (fee_trans_val.loc[g.index] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"LCOH + Transmission fees ({suffix})": (g[f"LCOH + Transmission fees ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"Distribution fees ({suffix})":   (fee_dist_val.loc[g.index] * g["h2_out"]).sum() / g["h2_out"].sum(),
                f"LCOH + T&D fees ({suffix})": (g[f"LCOH + T&D fees ({suffix})"] * g["h2_out"]).sum() / g["h2_out"].sum(),
                dispatch_label: g["h2_out"].sum() * conv /
                1000   # tons per regione
            }))
            .reset_index().rename(columns={"grid_region": "Grid Region"})
        )

        # Round and display
        region_summary = region_summary.round(2)
        key = str(scen_year) if year_title else year_key
        results[key] = region_summary

    return results


def plot_h2_efuels_details(cost_data, cost_type_label, tech_colors=None, tech_order=None, index='year'):
    """
    Plot interactive detailed breakdown of H2 and e-fuels technologies
    """

    # Filter for H2/e-fuels technologies
    h2_efuels_data = cost_data[
        (cost_data['cost_type'] == cost_type_label) &
        # Only values > 1 million USD
        (cost_data['cost_billion'].abs() > 0.001)
    ].copy()

    if h2_efuels_data.empty:
        print(f"No data for {cost_type_label}")
        return

    # Create pivot table: years x technologies
    pivot_table = h2_efuels_data.pivot_table(
        index=index,
        columns='tech_label',
        values='cost_billion',
        aggfunc='sum'
    ).fillna(0)

    # Order technologies
    if tech_order:
        available_techs = set(pivot_table.columns)
        ordered_techs = [
            tech for tech in tech_order if tech in available_techs]
        remaining_techs = available_techs - set(ordered_techs)
        ordered_techs.extend(sorted(remaining_techs))
    else:
        tech_totals = pivot_table.sum().sort_values(ascending=False)
        ordered_techs = tech_totals.index.tolist()

    # Get colors for each technology
    def get_tech_color(tech_label):
        if tech_colors and tech_label in tech_colors:
            return tech_colors[tech_label]
        # Default colors matching your figure + magenta for Fischer-Tropsch
        default_colors = {
            "Alkaline electrolyzer large": "#1f77b4",  # Blue
            "PEM electrolyzer": "#2ca02c",             # Green
            "SOEC": "#d62728",                         # Red
            "Fischer-Tropsch": "#e81cd0"               # Magenta
        }
        return default_colors.get(tech_label, "#999999")

    # Create interactive plotly chart
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add traces for each technology
    for tech in ordered_techs:
        y_values = pivot_table[tech]
        # Skip if all values are zero
        if (y_values == 0).all():
            continue

        color = get_tech_color(tech)

        fig.add_trace(go.Bar(
            name=tech,
            x=pivot_table.index.astype(str),
            y=y_values,
            marker_color=color,
            hovertemplate=f"%{{x}}<br>{tech}: %{{y:.2f}}B USD<extra></extra>"
        ))

    # Update layout
    fig.update_layout(
        barmode='relative',
        title=dict(
            text=f'H2 & e-fuels Technologies - {cost_type_label}',
            font=dict(size=16)
        ),
        xaxis=dict(
            title=dict(text="Years", font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text=f"{cost_type_label} (Billion USD)",
                       font=dict(size=14)),
            tickfont=dict(size=12),
            tickformat=".2f"  # Force decimal format, no scientific notation
        ),
        template="plotly_white",
        width=1200,
        height=600,
        margin=dict(l=40, r=200, t=50, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12),
            traceorder='reversed'
        )
    )

    # Add horizontal line at zero
    fig.add_hline(y=0, line_width=1, line_color="black")

    return fig


def create_h2_efuels_analysis(networks, index='year'):
    """
    Create complete analysis for H2 and e-fuels
    """

    # Compute costs for all networks
    all_h2_efuels_costs = []
    for name_tag, network in networks.items():
        df_costs = compute_h2_efuels_costs(network, name_tag)
        all_h2_efuels_costs.append(df_costs)

    df_h2_efuels_costs = pd.concat(all_h2_efuels_costs, ignore_index=True)

    # Define technology order (H2 first, then e-fuels)
    h2_efuels_order = [
        "Alkaline electrolyzer large",  # H2
        "PEM electrolyzer",             # H2
        "SOEC",                         # H2
        "Fischer-Tropsch"               # e-fuels
    ]

    # Define colors matching your figure + magenta for Fischer-Tropsch
    h2_efuels_colors = {
        "Alkaline electrolyzer large": "#1f77b4",  # Blue (from your figure)
        "PEM electrolyzer": "#2ca02c",             # Green (from your figure)
        "SOEC": "#d62728",                         # Red (from your figure)
        "Fischer-Tropsch": "#e81cd0"               # Magenta
    }

    # Create CAPEX plot
    fig1 = plot_h2_efuels_details(
        df_h2_efuels_costs,
        "Capital expenditure",
        tech_colors=h2_efuels_colors,
        tech_order=h2_efuels_order,
        index=index
    )

    # Create OPEX plot
    fig2 = plot_h2_efuels_details(
        df_h2_efuels_costs,
        "Operational expenditure",
        tech_colors=h2_efuels_colors,
        tech_order=h2_efuels_order,
        index=index
    )

    # Show plots
    if fig1:
        fig1.show()
    if fig2:
        fig2.show()

    return df_h2_efuels_costs


def hourly_matching_plot(networks, year_title=True):
    for idx, (network_name, network) in enumerate(networks.items()):
        year_str = network_name.split("_")[-1]
        # define additionality
        additionality = True

        # calculate electrolyzers consumption
        electrolysis_carrier = [
            'H2 Electrolysis',
            'Alkaline electrolyzer large',
            'Alkaline electrolyzer medium',
            'Alkaline electrolyzer small',
            'PEM electrolyzer',
            'SOEC'
        ]

        electrolyzers = network.links[network.links.carrier.isin(
            electrolysis_carrier)].index
        electrolyzers_consumption = network.links_t.p0[electrolyzers].multiply(
            network.snapshot_weightings.objective, axis=0
        ).sum(axis=1)

        # calculate RES generation
        res_carriers = [
            "csp",
            "solar",
            "onwind",
            "offwind-ac",
            "offwind-dc",
            "ror",
        ]
        res_stor_techs = ["hydro"]

        # get RES generators and storage units
        res_gens = network.generators.query("carrier in @res_carriers").index
        res_storages = network.storage_units.query(
            "carrier in @res_stor_techs").index

        if additionality:
            # get new generators and storage_units
            new_gens = network.generators.loc[
                network.generators.build_year == int(year_str)
            ].index
            new_stor = network.storage_units.loc[
                network.storage_units.build_year == int(year_str)
            ].index
            # keep only new RES generators and storage units
            res_gens = res_gens.intersection(new_gens)
            res_storages = res_storages.intersection(new_stor)

        # calculate RES generation
        res_generation = network.generators_t.p[res_gens].multiply(
            network.snapshot_weightings.objective, axis=0
        ).sum(axis=1)
        res_storages_dispatch = network.storage_units_t.p[res_storages].multiply(
            network.snapshot_weightings.objective, axis=0
        ).sum(axis=1)
        res_generation_total = res_generation + res_storages_dispatch

        compare_df = pd.concat(
            [res_generation_total, electrolyzers_consumption], axis=1)
        compare_df.rename(
            columns={0: "RES generation", 1: "Electrolyzer consumption"}, inplace=True)
        # compare_df = compare_df.resample("D").mean()

        fig, ax = plt.subplots(figsize=(10, 2.5))
        (
            compare_df[["RES generation", "Electrolyzer consumption"]]
            .div(network.snapshot_weightings.objective, axis=0)
            .div(1e3).resample("D").mean().plot(ax=ax)
        )
        ax.set_ylabel("GW")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel(None)
        ax.set_title(year_str if year_title else network_name)
        ax.legend(loc="lower left")


def preprocess_res_ces_share_eia_region(eia_gen_data, grid_regions):
    """
    Aggregate EIA 2023 generation data by grid region.
    Uses the mapping in grid_regions_to_states.xlsx.
    Returns a DataFrame with % Actual RES and % Actual CES for each grid region.
    """

    # Filter EIA data
    eia_gen_data = eia_gen_data[eia_gen_data["YEAR"] == 2023]
    eia_gen_data = eia_gen_data[eia_gen_data["STATE"] != "US-Total"]
    eia_gen_data = eia_gen_data[eia_gen_data['TYPE OF PRODUCER']
                                == 'Total Electric Power Industry']
    eia_gen_data = eia_gen_data[
        (eia_gen_data["ENERGY SOURCE"] != "Total") &
        (eia_gen_data["ENERGY SOURCE"] != "Other")
    ]

    # Normalize energy source names
    eia_gen_data.replace({"ENERGY SOURCE": {
        "Coal": "coal",
        "Hydroelectric Conventional": "hydro",
        "Pumped Storage": "PHS",
        "Solar Thermal and Photovoltaic": "solar",
        "Natural Gas": "gas",
        "Petroleum": "oil",
        "Wind": "wind",
        "Nuclear": "nuclear",
        "Geothermal": "geothermal",
        "Wood and Wood Derived Fuels": "biomass",
        "Other Biomass": "biomass",
        "Other Gases": "gas"
    }}, inplace=True)

    # Convert to TWh
    eia_gen_data["GENERATION (TWh)"] = eia_gen_data["GENERATION (Megawatthours)"] / 1e6

    # Pivot: state × source
    eia_state_df = (
        eia_gen_data.groupby(["STATE", "ENERGY SOURCE"])[["GENERATION (TWh)"]]
        .sum()
        .unstack(fill_value=0)
    )
    eia_state_df.columns = eia_state_df.columns.droplevel(0)

    # Merge with grid region mapping
    eia_with_region = eia_state_df.reset_index().rename(columns={
        "STATE": "State"})
    grid_regions = grid_regions.copy()
    grid_regions["States"] = grid_regions["States"].str.strip().str.upper()
    eia_with_region = eia_with_region.merge(
        grid_regions, left_on="State", right_on="States")

    # Aggregate by grid region
    region_agg = eia_with_region.groupby("Grid region").sum(numeric_only=True)

    # Compute % RES and % CES
    eia_res_carriers = ["solar", "wind",
                        "hydro", "geothermal", "biomass", "PHS"]
    eia_ces_carriers = eia_res_carriers + ["nuclear"]

    total = region_agg.sum(axis=1)
    res_total = region_agg[eia_res_carriers].sum(axis=1)
    ces_total = region_agg[eia_ces_carriers].sum(axis=1)

    region_agg["% Actual RES"] = (res_total / total) * 100
    region_agg["% Actual CES"] = (ces_total / total) * 100

    return region_agg[["% Actual RES", "% Actual CES"]]


def evaluate_res_ces_by_region(networks, ces_carriers, res_carriers):
    """
    Evaluate RES and CES shares from the model, aggregated by grid region.
    Assumes `attach_grid_region_to_buses` has been called on each network.
    Returns results[(scenario, year)] = df.
    """

    results = {}

    for name, network in networks.items():
        # Extract scenario and year from network name (e.g., "scenario_01_2030" -> scenario="scenario_01", year=2030)
        match = re.search(r'(?:scenario_(\d{2})|Base)_(\d{4})', name)
        if match:
            if match.group(1):
                scenario = f"scenario_{match.group(1)}"
            else:
                scenario = "Base"
            year = int(match.group(2))
        else:
            print(
                f"Warning: Skipping network '{name}' - does not match expected format (e.g., 'scenario_01_2030' or 'Base_2023').")
            continue

        snapshots = network.snapshots
        timestep_h = (snapshots[1] - snapshots[0]).total_seconds() / 3600
        snapshots_slice = slice(None)

        gen_and_sto_carriers = {
            'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac', 'nuclear',
            'geothermal', 'ror', 'hydro', 'solar rooftop'
        }
        link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', 'lignite']

        # Generators
        gen = network.generators[network.generators.carrier.isin(
            gen_and_sto_carriers)].copy()
        gen["grid_region"] = gen["bus"].map(network.buses["grid_region"])
        gen = gen[gen["grid_region"].notna()]

        gen_p = network.generators_t.p.loc[snapshots_slice, gen.index].clip(
            lower=0)
        gen_energy = gen_p.multiply(timestep_h).sum()
        gen_energy = gen_energy.to_frame(name="energy_mwh")
        gen_energy["carrier"] = gen.loc[gen_energy.index, "carrier"]
        gen_energy["grid_region"] = gen.loc[gen_energy.index, "grid_region"]

        # Storage
        sto = network.storage_units[network.storage_units.carrier.isin(
            gen_and_sto_carriers)].copy()
        sto["grid_region"] = sto["bus"].map(network.buses["grid_region"])
        sto = sto[sto["grid_region"].notna()]

        sto_p = network.storage_units_t.p.loc[snapshots_slice, sto.index].clip(
            lower=0)
        sto_energy = sto_p.multiply(timestep_h).sum()
        sto_energy = sto_energy.to_frame(name="energy_mwh")
        sto_energy["carrier"] = sto.loc[sto_energy.index, "carrier"]
        sto_energy["grid_region"] = sto.loc[sto_energy.index, "grid_region"]

        # Links
        link_data = []
        for i, link in network.links.iterrows():
            if (
                link["carrier"] in link_carriers and
                pd.notna(network.buses.loc[link["bus1"], "grid_region"])
            ):
                p1 = -network.links_t.p1.loc[snapshots_slice, i].clip(upper=0)
                energy_mwh = p1.sum() * timestep_h
                link_data.append({
                    "carrier": link["carrier"],
                    "grid_region": network.buses.loc[link["bus1"], "grid_region"],
                    "energy_mwh": energy_mwh
                })

        link_energy = pd.DataFrame(link_data)

        # Combine
        all_energy = pd.concat([
            gen_energy[["carrier", "grid_region", "energy_mwh"]],
            sto_energy[["carrier", "grid_region", "energy_mwh"]],
            link_energy[["carrier", "grid_region", "energy_mwh"]]
        ])

        # Aggregate by grid region
        region_totals = all_energy.groupby("grid_region")["energy_mwh"].sum()
        region_ces = all_energy[all_energy["carrier"].isin(
            ces_carriers)].groupby("grid_region")["energy_mwh"].sum()
        region_res = all_energy[all_energy["carrier"].isin(
            res_carriers)].groupby("grid_region")["energy_mwh"].sum()

        df = pd.DataFrame({
            "Total (MWh)": region_totals,
            "CES_energy": region_ces,
            "RES_energy": region_res
        }).fillna(0)

        df["% CES"] = 100 * df["CES_energy"] / df["Total (MWh)"]
        df["% RES"] = 100 * df["RES_energy"] / df["Total (MWh)"]

        df = df[["Total (MWh)", "% RES", "% CES"]].round(2)

        results[(scenario, year)] = df.sort_index()

    return results


# Function to format values
def fmt_2dp_or_na(v):
    if isinstance(v, str) and v.strip().upper() == "N/A":
        return "N/A"
    if pd.isna(v):
        return ""
    try:
        return f"{float(v):.2f}"
    except Exception:
        return v

# Function for deviation-based coloring (2023 only)


def deviation_color(a, b):
    """
    Color by absolute percent deviation |a - b| / b:
      - Green  -> ≤ 10%
      - Yellow -> > 10% and ≤ 20%
      - Red    -> > 20%
      - None   -> if b is 'N/A', NaN, or zero
    """
    try:
        if isinstance(b, str) and b.strip().upper() == "N/A":
            return ''
        a_val = float(a)
        b_val = float(b)
        if not pd.notna(b_val) or b_val == 0:
            return ''  # avoid invalid/zero baseline
        deviation = abs((a_val - b_val) / b_val) * 100.0

        if deviation <= 10:
            return 'background-color:#d4edda;'   # green
        elif deviation <= 20:
            return 'background-color:#fff3cd;'   # yellow
        else:
            return 'background-color:#f8d7da;'   # red
    except Exception:
        return ''


# Simple green/red for future years
def simple_color(a, b):
    """
    Returns green if a >= b, red if a < b, none if N/A or not numeric.
    """
    try:
        if isinstance(b, str) and b.strip().upper() == "N/A":
            return ''
        return 'background-color:#d4edda;' if float(a) >= float(b) else 'background-color:#f8d7da;'
    except:
        return ''


def get_us_from_eia(eia_generation_data):
    """
    Compute national (US) RES and CES shares from EIA data (2023).
    Weighted by total generation (TWh) of each state.
    """
    eia_state = preprocess_res_ces_share_eia(eia_generation_data)

    # Get total generation per state in TWh
    eia_gen_data = eia_generation_data[
        (eia_generation_data["YEAR"] == 2023) &
        (eia_generation_data["STATE"] != "US-Total") &
        (eia_generation_data['TYPE OF PRODUCER']
         == 'Total Electric Power Industry')
    ].copy()
    eia_gen_data["GENERATION (TWh)"] = eia_gen_data["GENERATION (Megawatthours)"] / 1e6
    total_by_state = eia_gen_data.groupby("STATE")["GENERATION (TWh)"].sum()

    # Weighted average
    us_res = (eia_state["% Actual RES"] *
              total_by_state).sum() / total_by_state.sum()
    us_ces = (eia_state["% Actual CES"] *
              total_by_state).sum() / total_by_state.sum()

    return round(us_res, 2), round(us_ces, 2)


def preprocess_res_ces_share_grid_region(eia_gen_data=None, grid_regions=None,
                                         file_path="./validation_data/generation_grid_regions.xlsx",
                                         sheet_name="Generation (TWh)"):
    """
    Drop-in replacement for preprocess_res_ces_share_eia_region.
    Ignores eia_gen_data and grid_regions, and instead loads
    precomputed grid-region data in TWh from generation_grid_regions.xlsx.
    """

    import pandas as pd
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if "Region" in df.columns and "Grid Region" not in df.columns:
        df = df.rename(columns={"Region": "Grid Region"})

    res_carriers = ["Solar", "Wind", "Hydro", "Geothermal", "Biomass"]
    ces_carriers = res_carriers + ["Nuclear"]

    total = df[["Coal", "Gas", "Oil", "Nuclear",
                "Other"] + res_carriers].sum(axis=1)
    res_total = df[res_carriers].sum(axis=1)
    ces_total = df[ces_carriers].sum(axis=1)

    result = df[["Grid Region"]].copy()
    result["% Actual RES"] = (res_total / total) * 100
    result["% Actual CES"] = (ces_total / total) * 100

    return result.set_index("Grid Region")


def display_grid_region_results(networks, ces, res, ces_carriers, res_carriers):
    """
    Show RES/CES results for grid regions using pre-aggregated Excel file
    (generation_grid_regions.xlsx). Adds columns for absolute generation (TWh),
    regional shares (%), and a national total row (U.S.).
    """

    res_by_region = evaluate_res_ces_by_region(
        networks,
        ces_carriers=ces_carriers,
        res_carriers=res_carriers
    )

    legend_html = """
    <div style="padding:10px; margin-bottom:15px; border:1px solid #ccc; border-radius:5px; width: fit-content;">
    <strong>Legend</strong>
    <ul style="margin:5px 0; padding-left:20px;">
        <li style="background-color:#d4edda; padding:2px;">Diff. ≤ ±10%</li>
        <li style="background-color:#fff3cd; padding:2px;">10% < Diff. ≤ 20%</li>
        <li style="background-color:#f8d7da; padding:2px;">Diff. > 20%</li>
    </ul>
    </div>
    """

    html_blocks = []
    cols_per_row = 2

    for (scenario, yr) in sorted(res_by_region.keys()):
        df_year = res_by_region[(scenario, yr)].copy()

        if yr == 2023:
            # Load actuals from Excel
            eia_region = preprocess_res_ces_share_grid_region()

            excel_df = pd.read_excel(
                "./validation_data/generation_grid_regions.xlsx",
                sheet_name="Generation (TWh)"
            )
            if "Region" in excel_df.columns and "Grid Region" not in excel_df.columns:
                excel_df = excel_df.rename(columns={"Region": "Grid Region"})
            excel_df = excel_df.set_index("Grid Region")

            # Add stats total generation
            eia_region = eia_region.join(excel_df[["Net generation (TWh)"]])

            # Add model total generation (TWh)
            df_year["Model generation (TWh)"] = df_year["Total (MWh)"] / 1e6

            # Merge with stats
            df_year = df_year.merge(
                eia_region, left_index=True, right_index=True)
            df_year = df_year.rename(
                columns={"Net generation (TWh)": "Stats generation (TWh)"})

            # Regional shares
            df_year["% Model generation share"] = (
                df_year["Model generation (TWh)"] /
                df_year["Model generation (TWh)"].sum() * 100
            )
            df_year["% Stats generation share"] = (
                df_year["Stats generation (TWh)"] /
                df_year["Stats generation (TWh)"].sum() * 100
            )

            # Add national total row (U.S.)
            totals = pd.Series({
                "% RES": (df_year["% RES"] * df_year["Model generation (TWh)"]).sum() / df_year["Model generation (TWh)"].sum(),
                "% Actual RES": (df_year["% Actual RES"] * df_year["Stats generation (TWh)"]).sum() / df_year["Stats generation (TWh)"].sum(),
                "% CES": (df_year["% CES"] * df_year["Model generation (TWh)"]).sum() / df_year["Model generation (TWh)"].sum(),
                "% Actual CES": (df_year["% Actual CES"] * df_year["Stats generation (TWh)"]).sum() / df_year["Stats generation (TWh)"].sum(),
                "Model generation (TWh)": df_year["Model generation (TWh)"].sum(),
                "Stats generation (TWh)": df_year["Stats generation (TWh)"].sum(),
                "% Model generation share": 100.0,
                "% Stats generation share": 100.0
            }, name="U.S.")
            df_year = pd.concat([df_year, totals.to_frame().T])

            df_disp = df_year[[
                "% RES", "% Actual RES",
                "% CES", "% Actual CES",
                "Model generation (TWh)", "Stats generation (TWh)",
                "% Model generation share", "% Stats generation share"
            ]].round(2)

            df_disp = df_disp.reset_index().rename(
                columns={"index": "Grid Region"}).set_index("Grid Region")

            def style_row(row):
                styles = []
                for col in df_disp.columns:
                    if "RES" in col:
                        styles.append(deviation_color(
                            row.get("% RES"), row.get("% Actual RES")))
                    elif "CES" in col:
                        styles.append(deviation_color(
                            row.get("% CES"), row.get("% Actual CES")))
                    elif "generation" in col or "share" in col:
                        styles.append(deviation_color(
                            row.get("Model generation (TWh)"), row.get("Stats generation (TWh)")))
                    else:
                        styles.append("")
                return styles

            styled_df = (
                df_disp.style
                .apply(style_row, axis=1)
                .format(fmt_2dp_or_na)
                .set_table_styles([{'selector': 'th.row_heading', 'props': 'font-weight:bold;'}])
            )

            # Force wide table (no wrapping)
            df_html = styled_df.to_html() + legend_html
            df_html = f"<div style='overflow-x:auto; white-space:nowrap;'>{df_html}</div>"

        else:
            expected_cols = ['% RES', '% RES target', '% CES', '% CES target']
            cols_present = [c for c in expected_cols if c in df_year.columns]
            df_year = df_year.reindex(columns=cols_present).round(2)

            # Add model total generation for future years
            if "Total (MWh)" in df_year.columns:
                df_year["Model generation (TWh)"] = df_year["Total (MWh)"] / 1e6
                df_year["% Model generation share"] = (
                    df_year["Model generation (TWh)"] /
                    df_year["Model generation (TWh)"].sum() * 100
                )

            totals = pd.Series({c: df_year[c].mean()
                               for c in df_year.columns}, name="U.S.")
            df_year = pd.concat([df_year, totals.to_frame().T])

            df_disp = df_year.reset_index().rename(
                columns={"index": "Grid Region"}).set_index("Grid Region")

            def style_row(row):
                styles = []
                for col in df_disp.columns:
                    if col.startswith('% RES'):
                        styles.append(simple_color(
                            row.get('% RES'), row.get('% RES target')))
                    elif col.startswith('% CES'):
                        styles.append(simple_color(
                            row.get('% CES'), row.get('% CES target')))
                    else:
                        styles.append("")
                return styles

            styled_df = (
                df_disp.style
                .apply(style_row, axis=1)
                .format(fmt_2dp_or_na)
                .set_table_styles([{'selector': 'th.row_heading', 'props': 'font-weight:bold;'}])
            )
            df_html = styled_df.to_html()

        block = f"""
        <div style="flex:1; padding:20px; min-width:300px;">
            <h4 style="text-align:left;">Year: {yr}</h4>
            {df_html}
        </div>
        """
        html_blocks.append(block)

    rows = [
        "<div style='display:flex; gap:10px; flex-wrap:wrap;'>" +
        "".join(html_blocks[i:i+cols_per_row]) +
        "</div>"
        for i in range(0, len(html_blocks), cols_per_row)
    ]

    for row in rows:
        display(HTML(row))


def attach_emm_region_to_buses(network, path_shape, distance_crs):
    """
    Attach EMM region to buses
    """
    # Read the shapefile using geopandas
    shape = gpd.read_file(path_shape, crs=distance_crs)
    # shape.rename(columns={"GRID_REGIO": "region"}, inplace=True)

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

    network_columns = network.buses.columns
    bus_cols = [*network_columns, "subregion"]

    st_buses = gpd.sjoin_nearest(shape, pypsa_gpd, how="right")[bus_cols]

    network.buses["emm_region"] = st_buses["subregion"]


def compute_ekerosene_by_region(
    networks: dict,
    regional_fees: pd.DataFrame,
    year_title: bool = True,
    p_nom_threshold: float = 1e-3,
    unit: str = "MWh",  # "MWh" or "gal"
):
    """
    Compute input rates, input prices, T&D fees and total unit cost for
    Fischer-Tropsch e-kerosene plants by grid region.
    """

    MWH_PER_GAL = (34.0 / 3600.0) * 3.78541  # MWh per gallon
    conv = 1.0 if unit == "MWh" else MWH_PER_GAL
    suffix = f"USD/{unit} e-ker"

    for name, net in networks.items():
        m = re.search(r"\d{4}", str(name))
        scen_year = int(m.group()) if m else 2030

        emm_mapping = dict(zip(net.buses.grid_region, net.buses.emm_region))

        ft = net.links[net.links.carrier.str.contains(
            "Fischer-Tropsch", case=False, na=False)].copy()
        if ft.empty:
            continue

        cap_series = np.where(
            ft.get("p_nom_extendable", False),
            ft.get("p_nom_opt", 0.0),
            ft.get("p_nom", 0.0),
        )
        ft = ft[pd.Series(cap_series, index=ft.index) > p_nom_threshold]
        if ft.empty:
            continue

        needed_cols = ("p0", "p1", "p2", "p3")
        ft_ids = [l for l in ft.index if all(l in getattr(
            net.links_t, c).columns for c in needed_cols)]
        if not ft_ids:
            continue
        ft = ft.loc[ft_ids]

        dt_h = (net.snapshots[1] - net.snapshots[0]).total_seconds() / \
            3600.0 if len(net.snapshots) > 1 else 1.0
        rows = []

        for link in ft.index:
            try:
                region = net.buses.at[ft.at[link, "bus1"], "grid_region"]
            except KeyError:
                continue

            bus_map = {}
            for i in range(4):
                b = ft.at[link, f"bus{i}"]
                if b not in net.buses.index:
                    continue
                carrier = net.buses.at[b, "carrier"]
                bus_map[carrier] = i

            out_MWh = (-net.links_t.p1[link] * dt_h).sum()
            if out_MWh <= 0:
                continue

            def get_flow_and_price(carrier_key):
                if carrier_key not in bus_map:
                    return 0.0, None
                idx = bus_map[carrier_key]
                flow = (net.links_t[f"p{idx}"][link] * dt_h).sum()
                price = net.buses_t.marginal_price[ft.at[link, f"bus{idx}"]]
                return flow, price

            elec_in, p_elec = get_flow_and_price("AC")
            h2_in,   p_h2 = get_flow_and_price("grid H2")
            co2_in,  p_co2 = get_flow_and_price("co2 stored")

            r_elec = elec_in / out_MWh if out_MWh > 0 else 0.0
            r_h2 = h2_in / out_MWh if out_MWh > 0 else 0.0
            r_co2 = co2_in / out_MWh if out_MWh > 0 else 0.0

            def avg_price(flow_series, price_series, total_flow):
                return (flow_series * price_series * dt_h).sum() / total_flow if total_flow > 0 else 0.0

            avg_p_elec = avg_price(
                net.links_t[f"p{bus_map['AC']}"][link], p_elec, elec_in) if "AC" in bus_map else 0.0
            avg_p_h2 = avg_price(
                net.links_t[f"p{bus_map['grid H2']}"][link], p_h2, h2_in) if "grid H2" in bus_map else 0.0
            avg_p_co2 = avg_price(
                net.links_t[f"p{bus_map['co2 stored']}"][link], p_co2, co2_in) if "co2 stored" in bus_map else 0.0

            c_elec = avg_p_elec * r_elec / conv
            c_h2 = avg_p_h2 * r_h2 / conv
            c_co2 = avg_p_co2 * r_co2 / conv

            rows.append({
                "Grid Region": region,
                "Production (TWh)": out_MWh / 1e6,
                f"Electricity rate (MWh el / MWh e-ker)": r_elec,
                f"H2 rate (MWh H2 / MWh e-ker)":          r_h2,
                f"CO2 rate (tCO2 / MWh e-ker)":           r_co2,
                f"Electricity cost ({suffix})": c_elec,
                f"Hydrogen cost ({suffix})":    c_h2,
                f"CO2 cost ({suffix})":         c_co2,
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        def wavg(group, col):
            return (group[col] * group["Production (TWh)"]).sum() / group["Production (TWh)"].sum()

        g = (
            df.groupby("Grid Region")
              .apply(lambda x: pd.Series({
                  "Production (TWh)": x["Production (TWh)"].sum(),
                  f"Electricity rate (MWh el / MWh e-ker)": wavg(x, f"Electricity rate (MWh el / MWh e-ker)"),
                  f"H2 rate (MWh H2 / MWh e-ker)":          wavg(x, f"H2 rate (MWh H2 / MWh e-ker)"),
                  f"CO2 rate (tCO2 / MWh e-ker)":           wavg(x, f"CO2 rate (tCO2 / MWh e-ker)"),
                  f"Electricity cost ({suffix})":           wavg(x, f"Electricity cost ({suffix})"),
                  f"Hydrogen cost ({suffix})":              wavg(x, f"Hydrogen cost ({suffix})"),
                  f"CO2 cost ({suffix})":                   wavg(x, f"CO2 cost ({suffix})"),
              }))
            .reset_index()   # keep Grid Region, drop numeric index
        )

        g["EMM Region"] = g["Grid Region"].map(emm_mapping)
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh", "Distribution nom USD/MWh"]
        ].set_index("region")

        g[f"Transmission fees ({suffix})"] = (
            g["EMM Region"].map(fee_map["Transmission nom USD/MWh"]) *
            g[f"Electricity rate (MWh el / MWh e-ker)"] / conv
        )
        g[f"Distribution fees ({suffix})"] = (
            g["EMM Region"].map(fee_map["Distribution nom USD/MWh"]) *
            g[f"Electricity rate (MWh el / MWh e-ker)"] / conv
        )
        g.drop(columns=["EMM Region"], inplace=True)

        g[f"LCO e-kerosene incl. T&D ({suffix})"] = (
            g[f"Electricity cost ({suffix})"]
            + g[f"Hydrogen cost ({suffix})"]
            + g[f"CO2 cost ({suffix})"]
            + g[f"Transmission fees ({suffix})"]
            + g[f"Distribution fees ({suffix})"]
        )

        title = re.search(r"\d{4}", str(name)).group() if year_title and re.search(
            r"\d{4}", str(name)) else str(name)
        print(f"\n{title} ({unit} view):")
        display(g.round(3))


def compute_ekerosene_by_region(
    networks: dict,
    regional_fees: pd.DataFrame,
    year_title: bool = True,
    p_nom_threshold: float = 1e-3,
    unit: str = "MWh",  # "MWh" or "gal"
):
    """
    Compute input rates, input prices, T&D fees and total unit cost for
    Fischer-Tropsch e-kerosene plants by grid region.
    """

    MWH_PER_GAL = (34.0 / 3600.0) * 3.78541  # MWh per gallon
    conv = 1.0 if unit == "MWh" else MWH_PER_GAL
    suffix = f"USD/{unit} e-ker"

    for name, net in networks.items():
        m = re.search(r"\d{4}", str(name))
        scen_year = int(m.group()) if m else 2030

        emm_mapping = dict(zip(net.buses.grid_region, net.buses.emm_region))

        ft = net.links[net.links.carrier.str.contains(
            "Fischer-Tropsch", case=False, na=False)].copy()
        if ft.empty:
            continue

        cap_series = np.where(
            ft.get("p_nom_extendable", False),
            ft.get("p_nom_opt", 0.0),
            ft.get("p_nom", 0.0),
        )
        ft = ft[pd.Series(cap_series, index=ft.index) > p_nom_threshold]
        if ft.empty:
            continue

        needed_cols = ("p0", "p1", "p2", "p3")
        ft_ids = [l for l in ft.index if all(l in getattr(
            net.links_t, c).columns for c in needed_cols)]
        if not ft_ids:
            continue
        ft = ft.loc[ft_ids]

        dt_h = (net.snapshots[1] - net.snapshots[0]).total_seconds() / \
            3600.0 if len(net.snapshots) > 1 else 1.0
        rows = []

        for link in ft.index:
            try:
                region = net.buses.at[ft.at[link, "bus1"], "grid_region"]
            except KeyError:
                continue

            bus_map = {}
            for i in range(4):
                b = ft.at[link, f"bus{i}"]
                if b not in net.buses.index:
                    continue
                carrier = net.buses.at[b, "carrier"]
                bus_map[carrier] = i

            out_MWh = (-net.links_t.p1[link] * dt_h).sum()
            if out_MWh <= 0:
                continue

            def get_flow_and_price(carrier_key):
                if carrier_key not in bus_map:
                    return 0.0, None
                idx = bus_map[carrier_key]
                flow = (net.links_t[f"p{idx}"][link] * dt_h).sum()
                price = net.buses_t.marginal_price[ft.at[link, f"bus{idx}"]]
                return flow, price

            elec_in, p_elec = get_flow_and_price("AC")
            h2_in,   p_h2 = get_flow_and_price("grid H2")
            co2_in,  p_co2 = get_flow_and_price("co2 stored")

            r_elec = elec_in / out_MWh if out_MWh > 0 else 0.0
            r_h2 = h2_in / out_MWh if out_MWh > 0 else 0.0
            r_co2 = co2_in / out_MWh if out_MWh > 0 else 0.0

            def avg_price(flow_series, price_series, total_flow):
                return (flow_series * price_series * dt_h).sum() / total_flow if total_flow > 0 else 0.0

            avg_p_elec = avg_price(
                net.links_t[f"p{bus_map['AC']}"][link], p_elec, elec_in) if "AC" in bus_map else 0.0
            avg_p_h2 = avg_price(
                net.links_t[f"p{bus_map['grid H2']}"][link], p_h2, h2_in) if "grid H2" in bus_map else 0.0
            avg_p_co2 = avg_price(
                net.links_t[f"p{bus_map['co2 stored']}"][link], p_co2, co2_in) if "co2 stored" in bus_map else 0.0

            c_elec = avg_p_elec * r_elec / conv
            c_h2 = avg_p_h2 * r_h2 / conv
            c_co2 = avg_p_co2 * r_co2 / conv

            rows.append({
                "Grid Region": region,
                "Production (TWh)": out_MWh / 1e6,
                f"Electricity rate (MWh el / MWh e-ker)": r_elec,
                f"H2 rate (MWh H2 / MWh e-ker)":          r_h2,
                f"CO2 rate (tCO2 / MWh e-ker)":           r_co2,
                f"Electricity cost ({suffix})": c_elec,
                f"Hydrogen cost ({suffix})":    c_h2,
                f"CO2 cost ({suffix})":         c_co2,
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        def wavg(group, col):
            return (group[col] * group["Production (TWh)"]).sum() / group["Production (TWh)"].sum()

        g = (
            df.groupby("Grid Region")
              .apply(lambda x: pd.Series({
                  "Production (TWh)": x["Production (TWh)"].sum(),
                  f"Electricity rate (MWh el / MWh e-ker)": wavg(x, f"Electricity rate (MWh el / MWh e-ker)"),
                  f"H2 rate (MWh H2 / MWh e-ker)":          wavg(x, f"H2 rate (MWh H2 / MWh e-ker)"),
                  f"CO2 rate (tCO2 / MWh e-ker)":           wavg(x, f"CO2 rate (tCO2 / MWh e-ker)"),
                  f"Electricity cost ({suffix})":           wavg(x, f"Electricity cost ({suffix})"),
                  f"Hydrogen cost ({suffix})":              wavg(x, f"Hydrogen cost ({suffix})"),
                  f"CO2 cost ({suffix})":                   wavg(x, f"CO2 cost ({suffix})"),
              }))
            .reset_index()   # keep Grid Region, drop numeric index
        )

        g["EMM Region"] = g["Grid Region"].map(emm_mapping)
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh", "Distribution nom USD/MWh"]
        ].set_index("region")

        g[f"Transmission fees ({suffix})"] = (
            g["EMM Region"].map(fee_map["Transmission nom USD/MWh"]) *
            g[f"Electricity rate (MWh el / MWh e-ker)"] / conv
        )
        g[f"Distribution fees ({suffix})"] = (
            g["EMM Region"].map(fee_map["Distribution nom USD/MWh"]) *
            g[f"Electricity rate (MWh el / MWh e-ker)"] / conv
        )
        g.drop(columns=["EMM Region"], inplace=True)

        g[f"LCO e-kerosene incl. T&D ({suffix})"] = (
            g[f"Electricity cost ({suffix})"]
            + g[f"Hydrogen cost ({suffix})"]
            + g[f"CO2 cost ({suffix})"]
            + g[f"Transmission fees ({suffix})"]
            + g[f"Distribution fees ({suffix})"]
        )

        title = re.search(r"\d{4}", str(name)).group() if year_title and re.search(
            r"\d{4}", str(name)) else str(name)
        print(f"\n{title} ({unit} view):")
        display(g.round(3))


def compute_ft_capacity_factor(
    networks: dict,
    carrier_regex: str = "Fischer-Tropsch",
    p_nom_threshold: float = 1.0,   # MW
    year_title: bool = True,
    round_digits: int = 2,
    output_threshold: float = 1.0,  # MWh
):
    """
    Compute annual capacity factor of Fischer-Tropsch plants by grid region.

    Definition:
    - Capacity factor (%) = H2 input at bus0 (MWh) /
                            (Installed capacity at bus0 (MW) * total weighted hours)

    Notes:
    - p_nom / p_nom_opt always refer to bus0 (PyPSA convention).
    - p0 (bus0) = H2 input (MWh).
    - p1 (bus1) = fuel output (MWh).
    """

    results = {}

    for year_key, n in networks.items():
        # extract year
        m = re.search(r"\d{4}", str(year_key))
        if not m:
            continue
        scen_year = int(m.group())
        key = str(scen_year) if year_title else year_key

        # select FT links
        ft = n.links[n.links.carrier.str.contains(
            carrier_regex, case=False, na=False)].copy()
        if ft.empty:
            continue

        # capacity series (MW, bus0 reference)
        cap_series = pd.Series(
            np.where(
                ft.get("p_nom_extendable", False),
                ft.get("p_nom_opt", 0.0),
                ft.get("p_nom", 0.0),
            ),
            index=ft.index,
        ).astype(float)

        # filter by capacity threshold
        ft = ft[cap_series > p_nom_threshold]
        cap_series = cap_series.loc[ft.index]
        if ft.empty:
            continue

        # snapshot weighting (hours per snapshot)
        w = n.snapshot_weightings.generators
        total_hours = float(w.sum())

        # H2 input (bus0, MWh)
        p0 = n.links_t.p0[ft.index]
        energy_h2_in = (p0).clip(lower=0).multiply(w, axis=0).sum(axis=0)

        # Fuel output (bus1, MWh, for reporting only)
        p1 = n.links_t.p1[ft.index]
        energy_out = (-p1).clip(lower=0).multiply(w, axis=0).sum(axis=0)

        # add region info
        ft["grid_region"] = ft["bus1"].map(n.buses["grid_region"])
        df_links = pd.DataFrame({
            "Link": ft.index,
            "Grid Region": ft["grid_region"],
            "Capacity (MW)": cap_series,
            "Hydrogen input (MWh)": energy_h2_in,
            "Fuel output (MWh)": energy_out,
        }).dropna()

        # aggregate by region
        region_summary = (
            df_links.groupby("Grid Region")
            .apply(lambda g: pd.Series({
                "Capacity (GW input H2)": g["Capacity (MW)"].sum() / 1e3,
                "Hydrogen input (MWh)": g["Hydrogen input (MWh)"].sum(),
                "Fuel output (MWh)": g["Fuel output (MWh)"].sum(),
                "Capacity factor (%)": (
                    g["Hydrogen input (MWh)"].sum()
                    / (g["Capacity (MW)"].sum() * total_hours) * 100
                ),
            }))
            .reset_index()
        )

        # skip if negligible production
        if region_summary["Fuel output (MWh)"].sum() < output_threshold:
            continue

        results[key] = region_summary.round(round_digits)

    return results


def compute_LCO_ekerosene_by_region(
    networks: dict,
    fx_2020: float,
    fx_recent: float,
    regional_fees: pd.DataFrame,
    emm_mapping: dict,
    unit: str = "gal",   # "gal" or "MWh"
    year_title: bool = True,
    p_nom_threshold: float = 1e-3,
    electricity_price: str = "marginal",   # "marginal" or "lcoe"
    hydrogen_price: str = "marginal",      # "marginal" or "lcoh"
    co2_price: str = "marginal",           # "marginal" or "lcoc"
    lcoe_by_region=None,                   # Series or dict
    lcoh_by_region: dict = None,           # required if hydrogen_price="lcoh"
    lcoc_by_region: dict = None,           # required if co2_price="lcoc"
    verbose=True,
):
    """
    Levelized cost of e-kerosene by grid region (USD/gal or USD/MWh e-ker).
    - Input rates = consumed energy / MWh e-ker (production-weighted).
    - Input prices = consumption-weighted marginal price (or LCOE/LCOH/LCOC if requested).
    - Autodetects flow sign per port (robust to sign conventions).
    """

    def get_year(data_dict, scen_year: int) -> str:
        """Return scen_year if available, otherwise fallback (only 2023→2020)."""
        if not data_dict:
            raise ValueError("Dataset is empty, cannot determine year.")
        if str(scen_year) in data_dict:
            return str(scen_year)
        if scen_year == 2023 and "2020" in data_dict:
            print("Year 2023 not found in dataset, using 2020 instead.")
            return "2020"
        raise KeyError(
            f"Year {scen_year} not found and no fallback available. Keys: {list(data_dict.keys())}")

    # Conversion MWh ↔ gallon
    MWH_PER_GALLON = (34.0 / 3600.0) * 3.78541
    conv = MWH_PER_GALLON if unit == "gal" else 1.0
    suffix = f"USD/{unit} e-ker"

    results = {}

    # VOM trajectory (EUR)
    vom_eur_points = {2020: 5.6360, 2025: 5.0512,
                      2030: 4.4663, 2035: 3.9346, 2040: 3.4029}
    years_sorted = np.array(sorted(vom_eur_points.keys()))
    values_sorted = np.array([vom_eur_points[y] for y in years_sorted])

    def vom_usd_for_year(y: int) -> float:
        vom_eur = float(np.interp(y, years_sorted, values_sorted))
        fx = fx_2020 if y == 2020 else fx_recent
        return vom_eur * fx

    for name, net in networks.items():
        scen_year = int(re.search(r"\d{4}", str(name)).group())

        # Select Fischer-Tropsch links
        ft = net.links[net.links.carrier.str.contains(
            "Fischer-Tropsch", case=False, na=False)].copy()
        if ft.empty:
            continue

        # Capacity filter
        cap_series = np.where(ft.get("p_nom_extendable", False),
                              ft.get("p_nom_opt", 0.0), ft.get("p_nom", 0.0))
        ft = ft[pd.Series(cap_series, index=ft.index) > p_nom_threshold]
        if ft.empty:
            continue

        # Snapshot duration [h]
        dt_h = (net.snapshots[1] - net.snapshots[0]).total_seconds() / \
            3600.0 if len(net.snapshots) > 1 else 1.0

        def energy_in(series):
            """MWh consumed at that port."""
            e_pos = (series.clip(lower=0) * dt_h).sum()
            e_neg = ((-series).clip(lower=0) * dt_h).sum()
            return ((-series).clip(lower=0) * dt_h) if e_neg >= e_pos else (series.clip(lower=0) * dt_h)

        def energy_out(series):
            """MWh produced at that port."""
            e_pos = (series.clip(lower=0) * dt_h).sum()
            e_neg = ((-series).clip(lower=0) * dt_h).sum()
            return (series.clip(lower=0) * dt_h) if e_pos >= e_neg else ((-series).clip(lower=0) * dt_h)

        rows = []
        for link in ft.index:
            try:
                region = net.buses.at[ft.at[link, "bus1"], "grid_region"]
            except KeyError:
                continue

            out_MWh = energy_out(net.links_t.p1[link]).sum()
            if out_MWh <= 0:
                continue

            elec_cons = energy_in(net.links_t.p3[link])  # AC
            h2_cons = energy_in(net.links_t.p0[link])  # H2
            co2_cons = energy_in(net.links_t.p2[link])  # CO2

            r_elec = elec_cons.sum() / out_MWh
            r_h2 = h2_cons.sum() / out_MWh
            r_co2 = co2_cons.sum() / out_MWh

            # --- electricity price ---
            if electricity_price == "marginal":
                p_elec = net.buses_t.marginal_price[ft.at[link, "bus3"]]
                avg_p_elec = (elec_cons * p_elec).sum() / \
                    elec_cons.sum() if elec_cons.sum() > 0 else 0.0
            elif electricity_price == "lcoe":
                if isinstance(lcoe_by_region, dict):
                    if not lcoe_by_region:
                        avg_p_elec = 0.0
                    else:
                        year = get_year(lcoe_by_region, scen_year)
                        avg_p_elec = lcoe_by_region[year].loc[region]
                else:
                    avg_p_elec = lcoe_by_region.loc[region]
            else:
                raise ValueError(
                    "electricity_price must be 'marginal' or 'lcoe'")

            # --- hydrogen price ---
            if hydrogen_price == "marginal":
                p_h2 = net.buses_t.marginal_price[ft.at[link, "bus0"]]
                avg_p_h2 = (h2_cons * p_h2).sum() / \
                    h2_cons.sum() if h2_cons.sum() > 0 else 0.0
            elif hydrogen_price == "lcoh":
                if not lcoh_by_region:
                    avg_p_h2 = 0.0
                else:
                    year = get_year(lcoh_by_region, scen_year)
                    avg_p_h2 = (
                        lcoh_by_region[year]
                        .set_index("Grid Region")
                        .at[region, "LCOH + T&D fees (USD/kg H2)"]
                        * 33.0
                    )
            else:
                raise ValueError("hydrogen_price must be 'marginal' or 'lcoh'")

            # --- CO2 price ---
            if co2_price == "marginal":
                p_co2 = net.buses_t.marginal_price[ft.at[link, "bus2"]]
                avg_p_co2 = (co2_cons * p_co2).sum() / \
                    co2_cons.sum() if co2_cons.sum() > 0 else 0.0
            elif co2_price == "lcoc":
                if not lcoc_by_region:
                    avg_p_co2 = 0.0
                elif scen_year == 2023:
                    # Special case: no CO₂ captured in 2023
                    avg_p_co2 = 0.0
                elif str(scen_year) in lcoc_by_region:
                    lcoc_df = lcoc_by_region[str(
                        scen_year)].set_index("Grid Region")
                    avg_p_co2 = lcoc_df.at[region,
                                           "LCOC incl. T&D fees (USD/tCO2)"] if region in lcoc_df.index else 0.0
                else:
                    raise KeyError(
                        f"LCOC dataset does not contain year {scen_year}. Available: {list(lcoc_by_region.keys())}")
            else:
                raise ValueError("co2_price must be 'marginal' or 'lcoc'")

            # --- cost components ---
            c_elec = avg_p_elec * r_elec * conv
            c_h2 = avg_p_h2 * r_h2 * conv
            c_co2 = avg_p_co2 * r_co2 * conv

            cap_cost = float(ft.at[link, "capital_cost"])
            cap_MW = float(ft.at[link, "p_nom_opt"] if ft.at[link,
                           "p_nom_extendable"] else ft.at[link, "p_nom"])
            c_capex = ((cap_cost * cap_MW) / out_MWh) * conv

            c_vom = vom_usd_for_year(scen_year) * conv
            lco_excl_TD = c_elec + c_h2 + c_co2 + c_capex + c_vom

            rows.append({
                "Grid Region": region,
                "Production (TWh)": out_MWh / 1e6,
                "Electricity rate (MWh el / MWh e-ker)": r_elec,
                "H2 rate (MWh H2 / MWh e-ker)": r_h2,
                "CO2 rate (tCO2 / MWh e-ker)": r_co2,
                "Electricity price (USD/MWh el)": avg_p_elec,
                "Hydrogen price (USD/MWh H2)":   avg_p_h2,
                "CO2 price (USD/tCO2)":          avg_p_co2,
                f"Electricity cost ({suffix})": c_elec,
                f"Hydrogen cost ({suffix})":    c_h2,
                f"CO2 cost ({suffix})":         c_co2,
                f"CAPEX ({suffix})":            c_capex,
                f"VOM ({suffix})":              c_vom,
                f"LCO e-kerosene (excl. T&D fees) ({suffix})": lco_excl_TD,
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        # Skip scenario if total production is zero
        if df["Production (TWh)"].sum() <= 1e-3:
            continue

        def wavg(group, col):
            return (group[col] * group["Production (TWh)"]).sum() / group["Production (TWh)"].sum()

        g = (
            df.groupby("Grid Region")
              .apply(lambda x: pd.Series({
                  "Production (TWh)": x["Production (TWh)"].sum(),
                  "Electricity rate (MWh el / MWh e-ker)": wavg(x, "Electricity rate (MWh el / MWh e-ker)"),
                  "H2 rate (MWh H2 / MWh e-ker)":          wavg(x, "H2 rate (MWh H2 / MWh e-ker)"),
                  "CO2 rate (tCO2 / MWh e-ker)":           wavg(x, "CO2 rate (tCO2 / MWh e-ker)"),
                  "Electricity price (USD/MWh el)":        wavg(x, "Electricity price (USD/MWh el)"),
                  "Hydrogen price (USD/MWh H2)":           wavg(x, "Hydrogen price (USD/MWh H2)"),
                  "CO2 price (USD/tCO2)":                  wavg(x, "CO2 price (USD/tCO2)"),
                  f"Electricity cost ({suffix})":          wavg(x, f"Electricity cost ({suffix})"),
                  f"Hydrogen cost ({suffix})":             wavg(x, f"Hydrogen cost ({suffix})"),
                  f"CO2 cost ({suffix})":                  wavg(x, f"CO2 cost ({suffix})"),
                  f"CAPEX ({suffix})":                     wavg(x, f"CAPEX ({suffix})"),
                  f"VOM ({suffix})":                       wavg(x, f"VOM ({suffix})"),
                  f"LCO e-kerosene (excl. T&D fees) ({suffix})": wavg(x, f"LCO e-kerosene (excl. T&D fees) ({suffix})"),
              }))
            .reset_index()
        )

        g["EMM Region"] = g["Grid Region"].map(emm_mapping)
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh", "Distribution nom USD/MWh"]
        ].set_index("region")

        g[f"Transmission fees ({suffix})"] = (
            g["EMM Region"].map(fee_map["Transmission nom USD/MWh"]) *
            g["Electricity rate (MWh el / MWh e-ker)"] * conv
        )
        g[f"Distribution fees ({suffix})"] = (
            g["EMM Region"].map(fee_map["Distribution nom USD/MWh"]) *
            g["Electricity rate (MWh el / MWh e-ker)"] * conv
        )
        g.drop(columns=["EMM Region"], inplace=True)

        g[f"LCO e-kerosene (incl. T&D fees) ({suffix})"] = (
            g[f"LCO e-kerosene (excl. T&D fees) ({suffix})"] +
            g[f"Transmission fees ({suffix})"] +
            g[f"Distribution fees ({suffix})"]
        )

        results[f"{scen_year if year_title else str(name)}"] = g

        if verbose:
            tot_prod = g["Production (TWh)"].sum()
            wavg_cost = (
                g[f"LCO e-kerosene (incl. T&D fees) ({suffix})"] * g["Production (TWh)"]).sum() / tot_prod

            title = re.search(r"\d{4}", str(
                name)).group() if year_title else str(name)
            print(f"\n{title}:")
            print(
                f"Weighted average LCO e-kerosene (incl. T&D): {wavg_cost:.2f} {suffix}")
            # scientific notation
            print(f"Total production: {tot_prod:.2f} TWh\n")

            numeric_cols = g.select_dtypes(include="number").columns
            fmt = {col: "{:.2f}" for col in numeric_cols}
            display(g.style.format(fmt).hide(axis="index"))

    return results


def compute_LCOC_by_region(
    networks: dict,
    regional_fees: pd.DataFrame,
    emm_mapping: dict,
    electricity_price: str = "marginal",   # or "lcoe"
    lcoe_by_region: pd.Series = None,      # required if electricity_price="lcoe"
    year_title: bool = True,
    captured_threshold_mt: float = 1e-6,   # MtCO2 threshold
    verbose: bool = True                   # if False, no print/display
):
    """
    Compute Levelized Cost of CO2 Capture (LCOC) by grid region.
    - Results aggregated by grid region, weighted by captured CO2.
    - Includes both excl. and incl. T&D fees.
    - Units: USD/tCO2
    """

    if verbose:
        print("Note: LCOC is computed per ton of CO2 captured.")

    results = {}

    for name, net in networks.items():
        scen_year = int(re.search(r"\d{4}", str(name)).group())

        # --- detect CCS/DAC links ---
        mask = net.links.carrier.str.contains(
            r"(DAC|CC(?!GT))", case=False, na=False)
        ccs = net.links[mask].copy()
        if ccs.empty:
            continue

        rows = []
        dt_h = (
            (net.snapshots[1] - net.snapshots[0]).total_seconds() / 3600.0
            if len(net.snapshots) > 1 else 1.0
        )

        def energy_flow(series):
            pos = (series.clip(lower=0) * dt_h).sum()
            neg = ((-series).clip(lower=0) * dt_h).sum()
            return series.clip(lower=0) * dt_h if pos >= neg else (-series).clip(lower=0) * dt_h

        for link in ccs.index:
            captured, sequestered = 0.0, 0.0
            elec_series, elec_bus = None, None

            for j in range(6):
                col = f"p{j}"
                if col not in net.links_t or link not in net.links_t[col]:
                    continue

                series = net.links_t[col][link]
                bus = net.links.at[link, f"bus{j}"]

                if "co2" in bus.lower():
                    flow = series.sum()
                    if flow < 0:
                        captured += -flow * dt_h
                        if "storage" in bus.lower() or "sequest" in bus.lower():
                            sequestered += -flow * dt_h
                elif isinstance(bus, str) and bus.strip() and "co2" not in bus.lower():
                    if re.match(r"US\d+(\s\d+)?$", bus):
                        elec_series = energy_flow(series)
                        elec_bus = bus

            if captured <= 0:
                continue

            try:
                region = net.buses.at[ccs.at[link, "bus0"], "grid_region"]
            except KeyError:
                continue

            cap_cost = float(ccs.at[link, "capital_cost"])
            cap_mw = float(
                ccs.at[link, "p_nom_opt"] if ccs.at[link, "p_nom_extendable"]
                else ccs.at[link, "p_nom"]
            )
            c_capex = (cap_cost * cap_mw) / captured

            if elec_series is not None and elec_series.sum() > 0:
                elec_rate = elec_series.sum() / captured
                if electricity_price == "marginal":
                    p_elec = net.buses_t.marginal_price[elec_bus]
                    avg_p_elec = (elec_series * p_elec).sum() / \
                        elec_series.sum()
                elif electricity_price == "lcoe":
                    avg_p_elec = lcoe_by_region.loc[region]
                else:
                    raise ValueError(
                        "electricity_price must be 'marginal' or 'lcoe'")
                c_elec = avg_p_elec * elec_rate
            else:
                elec_rate, avg_p_elec, c_elec = 0.0, 0.0, 0.0

            lco_excl = c_capex + c_elec

            rows.append({
                "Grid Region": region,
                "Captured CO2 (Mt)": captured / 1e6,
                "Sequestered CO2 (Mt)": sequestered / 1e6,
                "CAPEX (USD/tCO2)": c_capex,
                "Electricity rate (MWh el / tCO2)": elec_rate,
                "Electricity price (USD/MWh el)": avg_p_elec,
                "Electricity cost (USD/tCO2)": c_elec,
                "LCOC excl. T&D fees (USD/tCO2)": lco_excl,
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        def wavg(group, col):
            return (group[col] * group["Captured CO2 (Mt)"]).sum() / group["Captured CO2 (Mt)"].sum()

        g = (
            df.groupby("Grid Region")
              .apply(lambda x: pd.Series({
                  "Captured CO2 (Mt)": x["Captured CO2 (Mt)"].sum(),
                  "Sequestered CO2 (Mt)": x["Sequestered CO2 (Mt)"].sum(),
                  "CAPEX (USD/tCO2)": wavg(x, "CAPEX (USD/tCO2)"),
                  "Electricity rate (MWh el / tCO2)": wavg(x, "Electricity rate (MWh el / tCO2)"),
                  "Electricity price (USD/MWh el)": wavg(x, "Electricity price (USD/MWh el)"),
                  "Electricity cost (USD/tCO2)": wavg(x, "Electricity cost (USD/tCO2)"),
                  "LCOC excl. T&D fees (USD/tCO2)": wavg(x, "LCOC excl. T&D fees (USD/tCO2)"),
              }))
            .reset_index()
        )

        g = g[g["Captured CO2 (Mt)"] > captured_threshold_mt]
        if g.empty:
            continue

        g["EMM Region"] = g["Grid Region"].map(emm_mapping)
        fee_map = regional_fees.loc[
            regional_fees["Year"] == scen_year,
            ["region", "Transmission nom USD/MWh", "Distribution nom USD/MWh"]
        ].set_index("region")

        g["Transmission fee (USD/MWh)"] = g["EMM Region"].map(
            fee_map["Transmission nom USD/MWh"])
        g["Distribution fee (USD/MWh)"] = g["EMM Region"].map(
            fee_map["Distribution nom USD/MWh"])

        g["Transmission cost (USD/tCO2)"] = g["Transmission fee (USD/MWh)"] * \
            g["Electricity rate (MWh el / tCO2)"]
        g["Distribution cost (USD/tCO2)"] = g["Distribution fee (USD/MWh)"] * \
            g["Electricity rate (MWh el / tCO2)"]

        g["LCOC incl. T&D fees (USD/tCO2)"] = (
            g["LCOC excl. T&D fees (USD/tCO2)"] +
            g["Transmission cost (USD/tCO2)"] +
            g["Distribution cost (USD/tCO2)"]
        )

        g.drop(columns=["EMM Region"], inplace=True)
        results[f"{scen_year if year_title else str(name)}"] = g

        if verbose:
            tot_captured = g["Captured CO2 (Mt)"].sum()
            wavg_cost = (g["LCOC incl. T&D fees (USD/tCO2)"] *
                         g["Captured CO2 (Mt)"]).sum() / tot_captured
            title = re.search(r"\d{4}", str(
                name)).group() if year_title else str(name)
            print(f"\nYear: {title}")
            print(f"Total captured CO2: {tot_captured:.2f} Mt")
            print(
                f"Weighted average LCOC (incl. T&D fees): {wavg_cost:.2f} USD/tCO2\n")
            numeric_cols = g.select_dtypes(include="number").columns
            fmt = {col: "{:.2f}" for col in numeric_cols}
            display(g.style.format(fmt).hide(axis="index"))

    return results


def calculate_LCOC_by_region(
    networks: dict,
    regional_fees: pd.DataFrame,
    emm_mapping: dict,
    electricity_price: str = "marginal",   # or "lcoe"
    lcoe_by_region: pd.Series = None,      # required if electricity_price="lcoe"
    year_title: bool = True,
    captured_threshold_mt: float = 1e-6,   # MtCO2 threshold
    verbose: bool = False
) -> dict:
    """
    Lightweight wrapper around compute_LCOC_by_region.
    Returns the results dictionary {year: DataFrame}.

    If verbose=True, compute_LCOC_by_region will also print summaries.
    """
    return compute_LCOC_by_region(
        networks=networks,
        regional_fees=regional_fees,
        emm_mapping=emm_mapping,
        electricity_price=electricity_price,
        lcoe_by_region=lcoe_by_region,
        year_title=year_title,
        captured_threshold_mt=captured_threshold_mt,
        verbose=verbose
    )


def save_to_excel_with_formatting(df, sheet_name, title, excel_file_path, freeze_pane="B3"):
    # local import to parse column letters
    from openpyxl.utils import column_index_from_string

    with pd.ExcelWriter(excel_file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, startrow=1)

        # Get the worksheet for formatting
        worksheet = writer.sheets[sheet_name]

        # Add a title for df summary
        worksheet['A1'] = title
        worksheet['A1'].font = Font(bold=True, size=14, color="2F4F4F")
        worksheet['A1'].alignment = Alignment(
            horizontal="center", vertical="center")

        extra_col = df.index.nlevels
        max_col = len(df.columns) + extra_col  # include index columns

        if max_col > 1:
            worksheet.merge_cells(
                start_row=1, start_column=1, end_row=1, end_column=max_col)

        # Format headers (row 2: MultiIndex headers)
        header_fill = PatternFill(
            start_color="2F4F4F", end_color="2F4F4F", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True, size=10)
        header_alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True)
        border_thin = Border(left=Side(style='thin'), right=Side(
            style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

        for col in range(1, max_col + 1):
            cell = worksheet.cell(row=2, column=col)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment
            cell.border = border_thin

        # Format data cells
        extra_row = getattr(df.columns, "nlevels", 1)
        max_row = len(df) + extra_row + 2  # +2 for title and headers
        data_font = Font(size=10)
        data_alignment = Alignment(horizontal="center", vertical="center")

        for row in range(3, max_row + 1):
            for col in range(1, max_col + 1):
                cell = worksheet.cell(row=row, column=col)
                cell.font = data_font
                cell.alignment = data_alignment
                cell.border = border_thin

        # Auto-adjust column widths
        for col in range(1, max_col + 1):
            column_letter = get_column_letter(col)
            max_length = 0
            for row in range(2, min(max_row + 1, 100)):  # Sample first 100 rows
                try:
                    cell_value = str(worksheet.cell(row=row, column=col).value)
                    max_length = max(max_length, len(cell_value))
                except:
                    pass
            adjusted_width = min(max(max_length + 2, 10), 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Derive frozen rows/columns from freeze_pane (e.g., "B3" -> rows 1..2 and col 1 frozen)
        try:
            m = re.match(r"([A-Za-z]+)(\d+)", str(freeze_pane))
            freeze_col_idx = column_index_from_string(m.group(1)) if m else 2
            freeze_row_idx = int(m.group(2)) if m else 3
        except Exception:
            freeze_col_idx, freeze_row_idx = 2, 3  # sensible fallback for "B3"

        # Bold the frozen header rows (above the horizontal split), preserving existing header colors
        for r in range(1, max(1, freeze_row_idx)):
            for c in range(1, max_col + 1):
                cell = worksheet.cell(row=r, column=c)
                try:
                    cell.font = cell.font.copy(bold=True)
                except Exception:
                    cell.font = Font(bold=True, size=(
                        cell.font.sz if cell.font else 10))

        # Bold the frozen index columns (to the left of the vertical split) for data rows
        for r in range(3, max_row + 1):  # data area; header rows already bold
            for c in range(1, max(1, freeze_col_idx)):
                cell = worksheet.cell(row=r, column=c)
                try:
                    cell.font = cell.font.copy(bold=True)
                except Exception:
                    cell.font = Font(bold=True, size=data_font.size)

        # Freeze panes for better navigation
        worksheet.freeze_panes = worksheet[freeze_pane]


def compare_h2_kerosene_production(network, plot=True, network_name="Network", plot_threshold_gw=1e-3):
    """
    Compare H2 and e-kerosene production from a PyPSA network.

    - Production time series are expressed in GW (daily average power).
    - Installed capacity is expressed in GW.
    - Total production is in MWh, computed using snapshot_weightings (accounts for non-hourly resolution).
    - The plot is only shown if at least one daily average exceeds plot_threshold_gw (GW).
    """

    # Define hydrogen carriers
    h2_carriers = [
        "Alkaline electrolyzer large",
        "Alkaline electrolyzer medium",
        "Alkaline electrolyzer small",
        "PEM electrolyzer",
        "SOEC",
    ]

    # FT (e-kerosene) links
    ft_links = network.links[
        network.links['carrier'].str.contains('FT|Fischer|Tropsch', case=False, na=False) |
        network.links.index.str.contains(
            'FT|Fischer|Tropsch', case=False, na=False)
    ].copy()

    # H2 links
    h2_links = network.links[network.links.carrier.isin(h2_carriers)].copy()

    # Helper: sum p1 across given links, robust to empty indices
    def sum_p1(idx):
        if len(idx) == 0:
            try:
                base = network.links_t.p1.iloc[:, :1].copy()
                base.iloc[:, 0] = 0.0
                return base.iloc[:, 0]
            except Exception:
                return pd.Series(dtype=float, index=network.snapshots)
        return np.multiply(-1, network.links_t.p1[idx]).sum(axis=1)

    # Productions in MW (per snapshot)
    h2_prod_mw = sum_p1(h2_links.index)
    kerosene_prod_mw = sum_p1(ft_links.index)

    # Daily averages in GW
    h2_prod_gw = h2_prod_mw.resample("D").mean() / 1e3
    kerosene_prod_gw = kerosene_prod_mw.resample("D").mean() / 1e3

    # Snapshot durations (in hours)
    weights = network.snapshot_weightings["generators"]

    # Summaries: total production in MWh, installed capacity in GW
    h2_summary = {
        'total_production_MWh': (h2_prod_mw * weights).sum(),
        'installed_capacity_GW': (h2_links.p_nom_opt.sum() if 'p_nom_opt' in h2_links else h2_links.p_nom.sum()) / 1e3,
    }
    kerosene_summary = {
        'total_production_MWh': (kerosene_prod_mw * weights).sum(),
        'installed_capacity_GW': (ft_links.p_nom_opt.sum() if 'p_nom_opt' in ft_links else ft_links.p_nom.sum()) / 1e3,
    }

    # Comparison table
    comparison_data = {
        'Metric': ['Total Production (MWh)', 'Installed Capacity (GW)'],
        'Hydrogen': [h2_summary['total_production_MWh'], h2_summary['installed_capacity_GW']],
        'E-Kerosene': [kerosene_summary['total_production_MWh'], kerosene_summary['installed_capacity_GW']],
    }
    comparison_table = pd.DataFrame(comparison_data)

    # Plot in GW if above threshold
    if plot:
        avg_h2_gw = h2_prod_gw.mean()
        avg_kerosene_gw = kerosene_prod_gw.mean()
        if (avg_h2_gw > plot_threshold_gw) or (avg_kerosene_gw > plot_threshold_gw):
            fig, ax = plt.subplots(figsize=(15, 5))
            h2_prod_gw.plot(ax=ax, label='Hydrogen production', alpha=0.8)
            kerosene_prod_gw.plot(
                ax=ax, label='E-Kerosene production', alpha=0.8)

            ax.set_title(
                f'Hydrogen vs e-kerosene Production (GW) - {network_name}')
            ax.set_xlabel('')
            ax.set_ylabel('Production (GW)')

            # x-axis formatting: monthly ticks with abbreviated month names
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

            # legend outside
            ax.legend(bbox_to_anchor=(1.02, 1),
                      loc='upper left', borderaxespad=0.)
            ax.grid(True, alpha=0.3)
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend
            plt.show()
        else:
            print(
                f"\nSkipped {network_name}: both daily-average productions are below {plot_threshold_gw*1000:.1f} MW.\n")

    return {
        # Series in GW (daily average)
        'h2_production': h2_prod_gw,
        # Series in GW (daily average)
        'kerosene_production': kerosene_prod_gw,
        'h2_summary': h2_summary,                  # Totals in MWh, capacity in GW
        'kerosene_summary': kerosene_summary,      # Totals in MWh, capacity in GW
        'comparison_table': comparison_table
    }



def compute_capacity_factor_electrolysis(
    networks: dict,
    p_nom_threshold: float = 1.0,   # MW
    year_title: bool = True,
    round_digits: int = 2,
    output_threshold: float = 1.0,  # MWh
):
    """
    Compute annual capacity factor of H2 electrolysers by grid region.

    Definition:
    - Capacity factor (%) = Electricity input at bus0 (MWh) /
                            (Installed capacity at bus0 (MW) * total weighted hours)

    Notes:
    - p_nom / p_nom_opt always refer to bus0 (PyPSA convention).
    - p0 (bus0) = electricity input (MW, positive).
    - p1 (bus1) = hydrogen output (MW, negative) → reported only.
    """

    # Identify electrolyser carriers automatically
    h2_carriers = [
        c for c in pd.unique(
            pd.concat([n.links.carrier for n in networks.values()])
        )
        if "electrolyzer" in c.lower() or "soec" in c.lower()
    ]

    results = {}

    for year_key, n in networks.items():
        # extract year
        m = re.search(r"\d{4}", str(year_key))
        if not m:
            continue
        scen_year = int(m.group())
        key = str(scen_year) if year_title else year_key

        # select electrolysers
        links = n.links[n.links.carrier.isin(h2_carriers)].copy()
        if links.empty:
            continue

        # installed capacity (MW, bus0 reference)
        cap_series = pd.Series(
            np.where(
                links.get("p_nom_extendable", False),
                links.get("p_nom_opt", 0.0),
                links.get("p_nom", 0.0),
            ),
            index=links.index,
        ).astype(float)

        # filter by capacity threshold
        links = links[cap_series > p_nom_threshold]
        cap_series = cap_series.loc[links.index]
        if links.empty:
            continue

        # snapshot weighting
        w = n.snapshot_weightings.generators
        total_hours = float(w.sum())

        # electricity input (MWh, bus0)
        p0 = n.links_t.p0[links.index]
        energy_in = (p0).clip(lower=0).multiply(w, axis=0).sum(axis=0)

        # hydrogen output (MWh, bus1, for reporting only)
        p1 = n.links_t.p1[links.index]
        energy_out = (-p1).clip(lower=0).multiply(w, axis=0).sum(axis=0)

        # add grid region info
        links["grid_region"] = links["bus1"].map(n.buses["grid_region"])
        df_links = pd.DataFrame({
            "Link": links.index,
            "Grid Region": links["grid_region"],
            "Capacity (MW)": cap_series,
            "Electricity input (MWh)": energy_in,
            "Hydrogen output (MWh)": energy_out,
        }).dropna()

        # aggregate by region
        region_summary = (
            df_links.groupby("Grid Region")
            .apply(lambda g: pd.Series({
                "Capacity (GW input electricity)": g["Capacity (MW)"].sum() / 1e3,
                "Electricity input (MWh)": g["Electricity input (MWh)"].sum(),
                "Hydrogen output (MWh)": g["Hydrogen output (MWh)"].sum(),
                "Capacity factor (%)": (
                    g["Electricity input (MWh)"].sum()
                    / (g["Capacity (MW)"].sum() * total_hours) * 100
                ),
            }))
            .reset_index()
        )

        # skip if negligible production
        if region_summary["Hydrogen output (MWh)"].sum() < output_threshold:
            continue

        results[key] = region_summary.round(round_digits)

    return results

def _get_snapshot_weighting_series(network):
    """Return a snapshot weighting series aligned with the network snapshots."""

    weightings = getattr(network, "snapshot_weightings", None)
    if weightings is None:
        return pd.Series(1.0, index=network.snapshots)

    if isinstance(weightings, pd.Series):
        return weightings.reindex(network.snapshots, fill_value=0.0)

    for column in ["objective", "generators", "stores", "weights"]:
        if column in weightings.columns:
            return weightings[column].reindex(network.snapshots, fill_value=0.0)

    return weightings.iloc[:, 0].reindex(network.snapshots, fill_value=0.0)


def _get_aviation_loads_with_energy(network):
    """Collect aviation loads enriched with geographic metadata and annual energy (TWh)."""

    if "carrier" not in network.loads:
        return pd.DataFrame()

    aviation_mask = network.loads["carrier"].str.contains(
        "kerosene for aviation", case=False, na=False
    )
    aviation_loads = network.loads[aviation_mask].copy()

    if aviation_loads.empty:
        return pd.DataFrame()

    weights = _get_snapshot_weighting_series(network)
    loads_p = network.loads_t.p[aviation_loads.index]
    loads_p = loads_p.reindex(weights.index).fillna(0.0)

    energy_mwh = loads_p.mul(weights, axis=0).sum()
    average_mw = energy_mwh / weights.sum() if weights.sum() else energy_mwh
    peak_mw = loads_p.max()

    buses_info = network.buses[["state", "grid_region", "x", "y"]]
    aviation_loads = aviation_loads.join(buses_info, on="bus", how="left")
    aviation_loads["energy_MWh"] = energy_mwh
    aviation_loads["energy_TWh"] = aviation_loads["energy_MWh"] / 1e6

    return aviation_loads


def _aggregate_aviation_demand(network, level):
    """Aggregate aviation demand by the requested spatial level (state or grid_region)."""

    loads = _get_aviation_loads_with_energy(network)
    if loads.empty or level not in loads.columns:
        return pd.DataFrame(columns=[level, "energy_TWh", "lon", "lat", "share_pct"])

    loads = loads.dropna(subset=[level])
    if loads.empty:
        return pd.DataFrame(columns=[level, "energy_TWh", "lon", "lat", "share_pct"])

    aggregated = (
        loads.groupby(level)
        .agg(
            energy_TWh=("energy_TWh", "sum"),
            lon=("x", "mean"),
            lat=("y", "mean"),
        )
        .reset_index()
    )

    total_twh = aggregated["energy_TWh"].sum()
    if total_twh:
        aggregated["share_pct"] = aggregated["energy_TWh"] / total_twh * 100
    else:
        aggregated["share_pct"] = 0.0

    aggregated = aggregated.sort_values("energy_TWh", ascending=False)
    return aggregated


def compute_aviation_demand_table(network, level="state"):
    """Return a tidy aviation demand table for the requested aggregation level."""

    aggregation = _aggregate_aviation_demand(network, level)

    label = {
        "state": "State",
        "grid_region": "Grid Region"
    }.get(level, level.title())

    if aggregation.empty:
        return pd.DataFrame(columns=[label, "Fuel demand (TWh)", "Share (%)"])

    table = aggregation[[level, "energy_TWh",
                         "share_pct"]].copy()
    table = table.rename(columns={
        level: label,
        "energy_TWh": "Fuel demand (TWh)",
        "share_pct": "Share (%)",
    })

    return table


def create_aviation_demand_by_state_map(network, path_shapes, network_name="Network", distance_crs=4326, min_demand_twh=1.0, year_title=True):
    """Plot aviation demand aggregated by state (TWh/year) as scaled circles.

    Alaska and Hawaii remain in the returned tables, but they are omitted from the map.
    """

    state_df = _aggregate_aviation_demand(network, "state")
    if state_df.empty:
        print(f"No aviation loads found in the network: {network_name}")
        return None, None, state_df

    states_to_plot = state_df[state_df["energy_TWh"] >= min_demand_twh]
    if states_to_plot.empty:
        print(
            f"Aviation demand below {min_demand_twh} TWh for all states in {network_name}.")
        return None, None, state_df

    contiguous_mask = (
        states_to_plot["lon"].between(-130, -65, inclusive="both")
        & states_to_plot["lat"].between(20, 50, inclusive="both")
    )
    states_to_plot_contiguous = states_to_plot[contiguous_mask]
    if states_to_plot_contiguous.empty:
        print("No contiguous US states meet the plotting threshold; tables still include all states.")
        return None, None, state_df

    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes = shapes.to_crs(epsg=4326)
    bbox = box(-130, 20, -65, 50)
    shapes_clip = shapes.clip(bbox)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={
                           "projection": ccrs.PlateCarree()})
    shapes_clip.plot(ax=ax, facecolor="whitesmoke",
                     edgecolor="gray", alpha=0.7, linewidth=0.5)

    pie_scale = 0.02
    min_radius = 0.1
    max_radius = 3.5

    for _, row in states_to_plot_contiguous.iterrows():
        x, y = row["lon"], row["lat"]
        if pd.isna(x) or pd.isna(y):
            continue

        radius = np.clip(row["energy_TWh"] * pie_scale, min_radius, max_radius)
        circle = plt.Circle(
            (x, y), radius,
            facecolor="#1f77b4", edgecolor="gray", alpha=0.65,
            linewidth=1, transform=ccrs.PlateCarree(), zorder=4
        )
        ax.add_patch(circle)

        ax.text(
            x,
            y - radius - 0.3,
            f"{row['energy_TWh']:.1f} TWh",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray",
                      boxstyle="round,pad=0.2"),
            transform=ccrs.PlateCarree(),
        )

    year_match = re.search(r"\d{4}", network_name)
    year_str = f" – {year_match.group()}" if year_match else ""

    ax.set_extent([-130, -65, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(
        f"Aviation fuel demand by State (TWh/year){year_str if year_title else f' – {network_name}'}",
        fontsize=12,
        pad=20,
    )
    ax.axis("off")
    plt.tight_layout()

    return fig, ax, state_df


def create_aviation_demand_by_grid_region_map(network, path_shapes, network_name="Network", distance_crs=4326, min_demand_twh=5.0, year_title=True):
    """Plot aviation demand aggregated by grid region (TWh/year) as scaled circles."""

    region_df = _aggregate_aviation_demand(network, "grid_region")
    if region_df.empty:
        print(
            f"No aviation loads with grid region found in the network: {network_name}")
        return None, None, region_df

    regions_to_plot = region_df[region_df["energy_TWh"] >= min_demand_twh]
    if regions_to_plot.empty:
        print(
            f"Aviation demand below {min_demand_twh} TWh for all regions in {network_name}.")
        return None, None, region_df

    contiguous_mask = (
        regions_to_plot["lon"].between(-130, -65, inclusive="both")
        & regions_to_plot["lat"].between(20, 50, inclusive="both")
    )
    regions_to_plot_contiguous = regions_to_plot[contiguous_mask]
    if regions_to_plot_contiguous.empty:
        print("No contiguous US grid regions meet the plotting threshold; tables still include all regions.")
        return None, None, region_df

    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes = shapes.to_crs(epsg=4326)
    bbox = box(-130, 20, -60, 50)
    shapes_clip = shapes.clip(bbox)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={
                           "projection": ccrs.PlateCarree()})
    shapes_clip.plot(ax=ax, facecolor="whitesmoke",
                     edgecolor="gray", alpha=0.7, linewidth=0.5)

    pie_scale = 0.03
    min_radius = 0.15
    max_radius = 3.8

    for _, row in regions_to_plot_contiguous.iterrows():
        x, y = row["lon"], row["lat"]
        if pd.isna(x) or pd.isna(y):
            continue

        radius = np.clip(row["energy_TWh"] * pie_scale, min_radius, max_radius)
        circle = plt.Circle(
            (x, y), radius,
            facecolor="#1f77b4", edgecolor="gray", alpha=0.65,
            linewidth=1, transform=ccrs.PlateCarree(), zorder=4
        )
        ax.add_patch(circle)

        ax.text(
            x,
            y - radius - 0.3,
            f"{row['energy_TWh']:.1f} TWh",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray",
                      boxstyle="round,pad=0.2"),
            transform=ccrs.PlateCarree(),
        )

    year_match = re.search(r"\d{4}", network_name)
    year_str = f" – {year_match.group()}" if year_match else ""

    ax.set_extent([-130, -65, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(
        f"Aviation fuel demand by Grid Region (TWh/year){year_str if year_title else f' – {network_name}'}",
        fontsize=12,
        pad=20,
    )
    ax.axis("off")
    plt.tight_layout()

    return fig, ax, region_df


def calculate_demand_profile(network):
        """Calculate total electricity demand profile from network."""
        # Industrial processes consuming AC electricity
        target_processes = [
            "SMR CC", "Haber-Bosch", "ethanol from starch", "ethanol from starch CC",
            "DRI", "DRI CC", "DRI H2", "BF-BOF", "BF-BOF CC", "EAF",
            "dry clinker", "cement finishing", "dry clinker CC"
        ]
        
        # Static and dynamic loads
        static_load_carriers = ["rail transport electricity", "agriculture electricity", "industry electricity"]
        
        # Static loads (constant profile)
        static_totals = (
            network.loads.groupby("carrier").sum().p_set
            .reindex(static_load_carriers)
            .fillna(0)
        )
        static_sum = static_totals.sum()  # MW
        static_profile = pd.Series(static_sum, index=network.snapshots)
        
        # Industrial AC consumption
        process_links = network.links[network.links.carrier.isin(target_processes)]
        ac_input_links = process_links[process_links.bus0.map(network.buses.carrier) == "AC"].index
        ind_ac_profile = network.links_t.p0[ac_input_links].sum(axis=1) if len(ac_input_links) > 0 else 0
        
        # Non-industrial AC loads
        ac_loads = network.loads[network.loads.carrier == "AC"]
        industrial_ac_buses = network.links.loc[ac_input_links, "bus0"].unique() if len(ac_input_links) > 0 else []
        ac_non_ind_idx = ac_loads[~ac_loads.bus.isin(industrial_ac_buses)].index
        ac_profile = network.loads_t.p_set[ac_non_ind_idx.intersection(network.loads_t.p_set.columns)].sum(axis=1)
        
        # Services and EVs
        serv_idx = [i for i in network.loads[network.loads.carrier == "services electricity"].index
                    if i in network.loads_t.p_set.columns]
        ev_idx = [i for i in network.loads[network.loads.carrier == "land transport EV"].index
                  if i in network.loads_t.p_set.columns]
        serv_profile = network.loads_t.p_set[serv_idx].sum(axis=1) if serv_idx else 0
        ev_profile = network.loads_t.p_set[ev_idx].sum(axis=1) if ev_idx else 0
        
        # Data centers (constant profile)
        data_center_sum = network.loads.loc[network.loads.carrier == "data center", "p_set"].sum()
        dc_profile = pd.Series(data_center_sum, index=network.snapshots)
        
        # Other electricity
        other_idx = [i for i in network.loads[network.loads.carrier == "other electricity"].index
                     if i in network.loads_t.p_set.columns]
        other_profile = network.loads_t.p_set[other_idx].sum(axis=1) if other_idx else 0
        
        # Total demand profile (convert to GW, keep positive for plotting)
        total_demand = (
            static_profile + abs(ind_ac_profile) + ac_profile + 
            serv_profile + ev_profile + dc_profile + other_profile
        ) / 1000
        
        return total_demand


def plot_electricity_dispatch(networks, tech_colors, nice_names, 
                                        title_year=True, return_data=False):
    """
    Plot electricity dispatch with demand for multiple networks with stacked area charts.
    
    Parameters:
    -----------
    networks : dict
        Dictionary of network names to PyPSA network objects
    tech_colors : dict
        Technology color mapping
    nice_names : dict
        Technology name mapping for legend
    title_year : bool, default True
        If True, extract year from key for title. If False, use full key name
    return_data : bool, default False
        Whether to return the collected dispatch tables and demand data
    
    Returns:
    --------
    dict or None
        If return_data=True, returns dict with 'dispatch' and 'demand' keys
    """
    
    
    collected_dispatch_tables = {}
    collected_demand_tables = {}
    summary_list = []
    max_y = 0
    
    # Calculate dispatch and demand for all networks and find global max
    for key, n in networks.items():
        total_gwh, supply_gw = calculate_dispatch(n)
        demand_profile = calculate_demand_profile(n)
        
        summary_list.append({"Network": key, "Total Dispatch (GWh)": total_gwh})
        max_y = max(max_y, supply_gw.sum(axis=1).max(), demand_profile.max())
    
    # Add some margin
    ymax = max_y * 1.05
    
    # Create subplots
    fig, axes = plt.subplots(len(networks), 1, figsize=(22, 5 * len(networks)))
    
    # Handle single network case
    if len(networks) == 1:
        axes = [axes]
    
    # Define technology order
    ordered_columns = [
        'nuclear',
        'coal',
        'biomass',
        'CCGT',
        'OCGT',
        'oil',
        'hydro',
        'ror',
        'geothermal',
        'gas CHP',
        'biomass CHP',
        'solar',
        'solar rooftop',
        'csp',
        'onwind',
        'offwind-ac',
        'offwind-dc',
        'battery discharger'
    ]
    
    # Plot each network
    for ax, (key, n) in zip(axes, networks.items()):
        # Calculate dispatch
        _, supply_gw = calculate_dispatch(n)
        supply_gw.index = pd.to_datetime(supply_gw.index)
        supply_gw = supply_gw.resample('24H').mean()
        
        # Calculate demand
        demand_profile = calculate_demand_profile(n)
        demand_profile.index = pd.to_datetime(demand_profile.index)
        demand_daily = demand_profile.resample('24H').mean()
        
        # Filter and order columns
        supply_gw = supply_gw[[c for c in ordered_columns if c in supply_gw.columns]]
        collected_dispatch_tables[key] = supply_gw
        collected_demand_tables[key] = demand_daily
        
        # Create stacked area plot for generation
        supply_gw.plot.area(
            ax=ax,
            stacked=True,
            linewidth=0,
            color=[tech_colors.get(c, 'gray') for c in supply_gw.columns],
            legend=False
        )
        
        # Plot demand as line (positive values)
        demand_daily.plot(
            ax=ax,
            color='red',
            linewidth=2,
            linestyle='-',
            label='Total Demand',
            alpha=0.8
        )
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)
        
        # Set title based on title_year parameter
        if title_year:
            year = key[-4:]  # Extract the year
            title = f"Electricity Dispatch & Demand – {year}"
        else:
            title = f"Electricity Dispatch & Demand – {key}"
        
        ax.set_title(title)
        ax.set_ylabel("Power (GW)")
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis formatting
        start = supply_gw.index.min().replace(day=1)
        end = supply_gw.index.max()
        month_starts = pd.date_range(start=start, end=end, freq='MS')
        
        ax.set_xlim(start, end)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_starts.strftime('%b'))
        ax.tick_params(axis='x', which='both', labelbottom=True)
        
        # Create legend for technologies with non-zero values + demand
        handles, labels = ax.get_legend_handles_labels()
        sums = supply_gw.sum()
        
        # Filter out zero generation technologies but keep demand
        filtered = [(h, l) for h, l in zip(handles, labels) 
                   if sums.get(l, 0) > 0 or l == 'Total Demand']
        
        if filtered:
            handles, labels = zip(*filtered)
            pretty_labels = [nice_names.get(label, label) if label != 'Total Demand' 
                           else label for label in labels]
            
            ax.legend(
                handles, pretty_labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                title='Technology',
                fontsize='small',
                title_fontsize='medium'
            )
    
    # Set x-label for bottom subplot
    axes[-1].set_xlabel("Time (months)")
    plt.tight_layout(rect=[0, 0.05, 0.80, 1])
    plt.show()
    
    # Return data if requested
    if return_data:
        return {
            'dispatch': collected_dispatch_tables,
            'demand': collected_demand_tables
        }