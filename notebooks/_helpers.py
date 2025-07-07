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
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from pathlib import Path

import cartopy.crs as ccrs  # For plotting maps
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import box
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from matplotlib.patches import Wedge
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerPatch
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties

import plotly.express as px
import plotly.graph_objects as go

from shapely.geometry import LineString
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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

    capacity_data['p_nom_kw'] = capacity_data['p_nom_opt']  # MW
    h2_capacity_data = capacity_data.pivot_table(
        index='bus0',
        columns='carrier',
        values='p_nom_kw',
        fill_value=0
    )

    h2_capacity_data['state'] = h2_capacity_data.index.map(network.buses.state)
    h2_capacity_data['region'] = h2_capacity_data.index.map(
        network.buses.grid_region)

    return h2_capacity_data


def plot_h2_capacities_bar(network, title):
    """
    Plot the total hydrogen electrolyzer capacity by type for each state.
    """
    h2_cap = compute_h2_capacities(network)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    h2_cap.groupby(['state'])[['Alkaline electrolyzer large', 'PEM electrolyzer', 'SOEC']].sum().plot(
        ax=ax1,
        kind='bar',
        stacked=True,
        title=f'Hydrogen Electrolyzer Capacity by State and Type: {title}',
        ylabel='Capacity (kW)',
        xlabel='State',
    )

    h2_cap.groupby(['region'])[['Alkaline electrolyzer large', 'PEM electrolyzer', 'SOEC']].sum().plot(
        ax=ax2,
        kind='bar',
        stacked=True,
        title=f'Hydrogen Electrolyzer Capacity by Region and Type: {title}',
        ylabel='Capacity (MW)',
        xlabel='Region',
    )

    plt.tight_layout()
    plt.show()


def create_hydrogen_capacity_map(network, path_shapes, distance_crs=4326, min_capacity_mw=10):
    """
    Create a map with pie charts showing hydrogen electrolyzer capacity breakdown by type for each state
    """
    if hasattr(network, 'links') and len(network.links) > 0:
        # Filter for hydrogen-related links (electrolyzers)
        # Common naming patterns for electrolyzers in PyPSA
        hydrogen_links = network.links[
            network.links['carrier'].str.contains('H2|hydrogen|electroly|SOEC', case=False, na=False) |
            network.links.index.str.contains(
                'H2|hydrogen|electroly|SOEC', case=False, na=False)
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

    # Step 3: Define colors for electrolyzer types
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

    # Step 4: Create the plot
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
        f"Plotting {len(states_to_plot)} states with ≥{min_capacity_mw} MW hydrogen capacity")

    # Step 5: Create pie charts for each state
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
                    ha='center', va='top', fontsize=12, fontweight='bold')

    # Step 6: Create legend
    legend_elements = []
    for carrier, color in carrier_colors.items():
        if carrier in capacity_data['carrier'].values:
            # Clean up carrier names for legend
            display_name = carrier.replace('_', ' ').title()
            legend_elements.append(mpatches.Patch(
                color=color, label=display_name))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=14, title='Electrolyzer Type', title_fontsize=16)

    # Step 7: Formatting - Expand map boundaries
    x_buffer = (shapes.total_bounds[2] - shapes.total_bounds[0]) * 0.1
    y_buffer = (shapes.total_bounds[3] - shapes.total_bounds[1]) * 0.1

    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Installed Hydrogen Electrolyzer Capacity by State and Type',
                 fontsize=24, fontweight='bold', pad=30)

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


def create_ft_capacity_by_state_map(network, path_shapes, network_name="Network", distance_crs=4326, min_capacity_gw=0.01):
    """
    Create a geographic map with simple round circles showing FT capacity per state in gigawatts (GW).
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

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    bbox = box(-130, 20, -65, 50)
    shapes_clip = shapes.to_crs(epsg=4326).clip(bbox)
    shapes_clip.plot(ax=ax, facecolor='whitesmoke',
                     edgecolor='gray', alpha=0.7, linewidth=0.5)

    lon_min, lon_max = -130, -65
    lat_min, lat_max = 20, 50
    pie_scale = 0.3

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
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'),
                transform=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.autoscale(False)
    ax.set_position([0.05, 0.05, 0.9, 0.9])

    ax.set_title(
        f"Fischer-Tropsch Capacity by State (GW){year_str}", fontsize=14)
    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    return fig, ax, links_with_state


def create_ft_capacity_by_grid_region_map(network, path_shapes, network_name="Network", distance_crs=4326, min_capacity_gw=0.01):
    """
    Create a map showing total FT capacity per grid region (GW) as full circles with linear radius scaling.
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
    grid_region_capacity = grid_region_capacity[grid_region_capacity["total_gw"]
                                                >= min_capacity_gw]

    # Set up map
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    bbox = box(-130, 20, -60, 50)
    shapes = gpd.read_file(path_shapes, crs=distance_crs)
    shapes = shapes.to_crs(epsg=4326).clip(bbox)
    shapes.plot(ax=ax, facecolor='whitesmoke',
                edgecolor='gray', alpha=0.7, linewidth=0.5)

    # Plot circles with linear scaling
    pie_scale = 0.3  # degrees per GW
    min_radius = 0.1
    max_radius = 3.5

    for _, row in grid_region_capacity.iterrows():
        x, y, total_gw = row["x"], row["y"], row["total_gw"]
        radius = np.clip(total_gw * pie_scale, min_radius, max_radius)

        circle = plt.Circle((x, y), radius,
                            facecolor='#B22222', edgecolor='gray', alpha=0.6,
                            linewidth=1, transform=ccrs.PlateCarree(), zorder=4)
        ax.add_patch(circle)

        ax.text(x, y - radius - 0.3, f'{total_gw:.2f} GW',
                ha='center', va='top', fontsize=9, fontweight='normal',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'),
                transform=ccrs.PlateCarree())

    ax.set_extent([-130, -65, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(
        f"Fischer-Tropsch Capacity by Grid Region (GW){year_str}", fontsize=16, pad=20)
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

    # Duration of each timestep
    timestep_hours = (snapshots[1] - snapshots[0]).total_seconds() / 3600

    # Define valid carriers
    gen_and_sto_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac', 'nuclear',
        'geothermal', 'ror', 'hydro'
    }
    link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass']
    valid_carriers = gen_and_sto_carriers.union(link_carriers)

    # Identify electric buses
    electric_buses = set(
        n.buses.index[
            ~n.buses.carrier.str.contains(
                "heat|gas|H2|oil|coal", case=False, na=False)
        ]
    )

    # Generators
    gen = n.generators[n.generators.carrier.isin(gen_and_sto_carriers)]
    gen_p = n.generators_t.p.loc[snapshots_slice, gen.index].clip(lower=0)
    gen_dispatch = gen_p.groupby(gen['carrier'], axis=1).sum()

    # Storage
    sto = n.storage_units[n.storage_units.carrier.isin(gen_and_sto_carriers)]
    sto_p = n.storage_units_t.p.loc[snapshots_slice, sto.index].clip(lower=0)
    sto_dispatch = sto_p.groupby(sto['carrier'], axis=1).sum()

    # Links: only from conventional carriers and to electric buses
    link_frames = []
    for carrier in link_carriers:
        links = n.links[
            (n.links.carrier == carrier) &
            (n.links.bus1.isin(electric_buses))
        ]
        if links.empty:
            continue
        p1 = n.links_t.p1.loc[snapshots_slice, links.index].clip(upper=0)
        p1_positive = -p1
        df = p1_positive.groupby(links['carrier'], axis=1).sum()
        link_frames.append(df)

    link_dispatch = pd.concat(
        link_frames, axis=1) if link_frames else pd.DataFrame(index=snapshots)

    # Combine all sources
    supply = pd.concat([gen_dispatch, sto_dispatch, link_dispatch], axis=1)
    supply = supply.groupby(supply.columns, axis=1).sum()
    supply = supply.clip(lower=0)

    # Convert to GW and GWh
    supply_gw = supply / 1e3  # MW → GW
    energy_mwh = supply.sum(axis=1) * timestep_hours
    total_gwh = energy_mwh.sum() / 1e3  # → GWh

    return total_gwh, supply_gw


def plot_electricity_dispatch(networks, carrier_colors, start_date=None, end_date=None, ymax=None):
    summary_list = []
    max_y = 0

    for key, n in networks.items():
        print(f"Processing network: {key}")
        total_gwh, supply_gw = calculate_dispatch(n, start_date, end_date)
        summary_list.append(
            {"Network": key, "Total Dispatch (GWh)": total_gwh})
        max_y = max(max_y, supply_gw.sum(axis=1).max())

    y_max_plot = ymax if ymax is not None else max_y

    fig, axes = plt.subplots(len(networks), 1, figsize=(
        22, 5 * len(networks)), sharex=True)

    if len(networks) == 1:
        axes = [axes]

    for ax, (key, n) in zip(axes, networks.items()):
        _, supply_gw = calculate_dispatch(n, start_date, end_date)
        supply_gw.index = pd.to_datetime(supply_gw.index)
        supply_gw = supply_gw.resample('24H').mean()

        supply_gw.plot.area(
            ax=ax,
            stacked=True,
            linewidth=0,
            color=[carrier_colors.get(c, 'gray') for c in supply_gw.columns],
            legend=False
        )
        ax.set_title(f"Electricity Dispatch – {key}")
        ax.set_ylabel("Power (GW)")
        ax.set_ylim(0, y_max_plot)
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            title='Carrier',
            fontsize='small',
            title_fontsize='medium'
        )

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

    valid_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac', 'nuclear',
        'geothermal', 'ror', 'hydro',
        'gas', 'OCGT', 'CCGT', 'oil', 'coal', 'biomass', 'lignite',
    }

    # Identify electric buses (exclude heat, gas, H2, oil, coal buses)
    electric_buses = set(n.buses.query("carrier == 'AC'").index)

    # Generators
    gen = n.generators[n.generators.carrier.isin(valid_carriers)].copy()
    gen_dispatch = n.generators_t.p[gen.index].multiply(
        snapshot_weights, axis=0)
    gen['energy'] = gen_dispatch.sum()
    gen = gen[(gen.p_nom_opt > 0) & (gen.energy > 0)]
    gen['lcoe'] = (gen.capital_cost * gen.p_nom_opt +
                   gen.marginal_cost * gen.energy) / gen.energy
    gen['type'] = 'generator'

    # Storage units
    sto = n.storage_units[n.storage_units.carrier.isin(valid_carriers)].copy()
    sto_dispatch = n.storage_units_t.p[sto.index].clip(
        lower=0).multiply(snapshot_weights, axis=0)
    sto['energy'] = sto_dispatch.sum()
    sto = sto[(sto.p_nom_opt > 0) & (sto.energy > 0)]
    sto['lcoe'] = (sto.capital_cost * sto.p_nom_opt +
                   sto.marginal_cost * sto.energy) / sto.energy
    sto['type'] = 'storage'

    # Links
    # Select only links going to electric buses and with p_nom_opt > 0
    link = n.links[
        (n.links.carrier.isin(valid_carriers)) &
        (n.links.bus1.isin(electric_buses)) &
        (n.links.p_nom_opt > 0)
    ].copy()

    # Take only actual output (p1 < 0 → output), clip to keep only negative, and make positive
    link_dispatch = -n.links_t.p1[link.index].clip(upper=0)

    # Multiply by snapshot weights (if needed)
    weighted_link_dispatch = link_dispatch.multiply(snapshot_weights, axis=0)

    # Sum energy
    link['energy'] = weighted_link_dispatch.sum()
    link = link[(link.p_nom_opt > 0) & (link.energy > 0)]
    link['lcoe'] = (link.capital_cost * link.p_nom_opt +
                    link.marginal_cost * link.energy) / link.energy
    link['type'] = 'link'

    # Combine and merge
    gen_data = gen[['bus', 'carrier', 'lcoe', 'type', 'energy']]
    sto_data = sto[['bus', 'carrier', 'lcoe', 'type', 'energy']]
    link_data = link[['bus1', 'carrier', 'lcoe', 'type', 'energy']].rename(columns={
                                                                           'bus1': 'bus'})

    lcoe_data = pd.concat([gen_data, sto_data, link_data], axis=0).dropna()
    lcoe_data = lcoe_data.merge(
        n.buses[['x', 'y', 'grid_region']], left_on='bus', right_index=True)

    # Weighted mean LCOE per bus
    def weighted_avg(df):
        return (df['lcoe'] * df['energy']).sum() / df['energy'].sum()

    lcoe_by_bus = (
        lcoe_data.groupby('bus')
        .apply(lambda df: pd.Series({
            'weighted_lcoe': weighted_avg(df),
            'x': df['x'].iloc[0],
            'y': df['y'].iloc[0],
            'grid_region': df['grid_region'].iloc[0]
        }))
        .reset_index()
    )

    # Aggregate by grid_region and carrier for weighted avg LCOE and dispatch
    region_summary = (
        lcoe_data.groupby(['grid_region', 'carrier'])
        .agg(
            dispatch_mwh=('energy', 'sum'),
            total_cost=('lcoe', lambda x: (
                x * lcoe_data.loc[x.index, 'energy']).sum())
        )
        .reset_index()
    )

    # Calculate weighted avg LCOE per carrier & grid_region
    region_summary['lcoe'] = region_summary['total_cost'] / \
        region_summary['dispatch_mwh']

    # Convert dispatch to TWh
    region_summary['dispatch'] = region_summary['dispatch_mwh'] / 1e6

    # Calculate weighted average LCOE per grid_region (all carriers)
    weighted_avg_grid_region = (
        region_summary.groupby('grid_region').apply(
            lambda df: (df['dispatch_mwh'] * df['lcoe']
                        ).sum() / df['dispatch_mwh'].sum()
        )
    )

    # Pivot to wide table for carriers
    table = region_summary.pivot(
        index='grid_region', columns='carrier', values=['lcoe', 'dispatch'])
    table.columns = [
        f"{carrier} {metric} [{'USD/MWh' if metric == 'lcoe' else 'TWh'}]" for metric, carrier in table.columns]
    table = table.reset_index()

    # Replace NaN in dispatch columns only
    dispatch_cols = [col for col in table.columns if 'dispatch' in col.lower()]
    table[dispatch_cols] = table[dispatch_cols].fillna(0.0)

    # Add weighted average LCOE per grid_region as a simple column
    table['Weighted Average LCOE (USD/MWh)'] = table['grid_region'].map(
        weighted_avg_grid_region).round(2)

    # Round dispatch and lcoe to 2 decimals
    for col in table.columns:
        if col != 'grid_region':
            table[col] = table[col].round(2)

    # Color scale limits for the map based on weighted averages
    vmin = lcoe_by_bus['weighted_lcoe'].quantile(0.05)
    vmax = max(vmin, min(weighted_avg_grid_region.max()
               * 1.1, lcoe_by_bus['weighted_lcoe'].max()))

    # GeoDataFrame for plotting
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

    if title:
        ax.set_title(title)
    elif key:
        ax.set_title(f"LCOE Map for {key}")
    else:
        ax.set_title("Weighted Average LCOE (USD/MWh) per Grid Region")


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
                           labelspacing=2.5,
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

    ax.set_extent([-130, -60, 20, 50], crs=ccrs.PlateCarree())

    ax.set_title(f'Installed electrolyzer capacity - {title} (only nodes ≥ 10 MW)\n')
    plt.tight_layout()
    plt.show()


def plot_lcoh_maps_by_grid_region(networks, shapes, h2_carriers, output_threshold=1.0):
    """
    Compute and plot weighted average LCOH per grid region for each year in USD/kg H2.

    Parameters:
    - networks: dict of PyPSA networks (year -> network)
    - shapes: GeoDataFrame with grid region geometries
    - h2_carriers: list of hydrogen production carrier names
    - output_threshold: minimum hydrogen output (MWh) to include a link in calculations
    """

    all_results = []

    # Normalize the grid_region column name
    for col in ["Grid Region", "GRID_REGIO", "grid_region"]:
        if col in shapes.columns:
            shapes = shapes.rename(columns={col: "grid_region"})
            break
    else:
        raise KeyError("No 'grid_region' column found in shapes GeoDataFrame")

    for year, network in networks.items():

        hydrogen_links = network.links[network.links.carrier.isin(h2_carriers)]
        if hydrogen_links.empty:
            print("  No valid H2 links for year {year}, skipping.")
            continue

        link_ids = hydrogen_links.index

        # Energy flows: electricity in (MWh), H2 out (MWh)
        p0 = network.links_t.p0[link_ids]
        p1 = network.links_t.p1[link_ids]
        h2_output = -p1.sum()  # flip sign: production is negative in PyPSA

        # Filter valid links by output threshold
        valid_links = h2_output > output_threshold
        if valid_links.sum() == 0:
            print(
                f"  No links with H2 output > {output_threshold} MWh in {year}, skipping.")
            continue

        # Electricity prices by input bus
        prices = pd.DataFrame(index=p0.index, columns=link_ids)
        for link in link_ids:
            bus = hydrogen_links.loc[link, "bus0"]
            prices[link] = network.buses_t.marginal_price[bus]

        # Calculate electricity cost (USD) per link (sum over time)
        elec_cost = (p0.loc[:, valid_links] * prices.loc[:, valid_links]).sum()

        capex = hydrogen_links.loc[valid_links, "capital_cost"]
        opex = hydrogen_links.loc[valid_links, "marginal_cost"]
        h2_output_valid = h2_output[valid_links]

        with np.errstate(divide="ignore", invalid="ignore"):
            # Calculate LCOH in USD/MWh H2
            lcoh = (capex + opex + elec_cost) / h2_output_valid
            lcoh = lcoh.replace([np.inf, -np.inf], np.nan)

            # Convert USD/MWh to USD/kg (1 MWh H2 ≈ 33.33 kg)
            lcoh_kg = lcoh / 33.33

        df = pd.DataFrame({
            "lcoh": lcoh_kg,
            "h2_output": h2_output_valid,
            "bus": hydrogen_links.loc[valid_links, "bus0"]
        })

        df["grid_region"] = df["bus"].map(network.buses["grid_region"])
        df["year"] = year
        df = df.dropna(subset=["grid_region", "lcoh", "h2_output"])

        all_results.append(df)

    if not all_results:
        print("No valid data for LCOH plotting.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    region_lcoh = (
        all_df.groupby(["grid_region", "year"])
        .apply(lambda g: pd.Series({
            "weighted_lcoh": (g["lcoh"] * g["h2_output"]).sum() / g["h2_output"].sum()
        }))
        .reset_index()
    )

    plot_df = shapes.merge(region_lcoh, on="grid_region", how="left")

# Compute global vmin and vmax over all years
    vmin = plot_df["weighted_lcoh"].quantile(0.05)
    vmax = plot_df["weighted_lcoh"].quantile(0.95)

    for year in sorted(region_lcoh.year.unique()):
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={
                               "projection": ccrs.PlateCarree()})

        year_df = plot_df[plot_df.year == year]

        year_df.plot(
            column="weighted_lcoh",
            cmap="RdYlGn_r",
            linewidth=0.8,
            edgecolor="0.8",
            legend=True,
            legend_kwds={
                "label": "LCOH (USD/kg H2)",
                "orientation": "vertical"
            },
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        ax.set_extent([-130, -65, 20, 55], crs=ccrs.PlateCarree())
        ax.axis("off")
        ax.set_title(f"LCOH – {year.split('_')[-1]}", fontsize=14)
        plt.show()


def calculate_weighted_lcoh_table_by_year(networks, h2_carriers, output_threshold=1.0):
    """
    Calculate weighted average LCOH (USD/kg) and total hydrogen dispatch (kg)
    per grid region for each year, matching the logic of the plotting function.

    Parameters:
    - networks: dict of PyPSA networks {year: network}
    - h2_carriers: list of hydrogen carrier names
    - output_threshold: minimum hydrogen output (MWh) to include a link

    Returns:
    - dict of DataFrames keyed by year string, each DataFrame contains:
      ['grid_region', 'Weighted Average LCOH (USD/kg)', 'Total Hydrogen Dispatch (kg)']
    """
    import re
    all_results = {}

    for year_key, network in networks.items():
        # Extract simple year string matching plot title style
        year_str = str(year_key)
        simple_year = year_str.split('_')[-1]  # e.g. 'network_2025' → '2025'

        hydrogen_links = network.links[network.links.carrier.isin(h2_carriers)]
        if hydrogen_links.empty:
            continue

        link_ids = hydrogen_links.index
        p0 = network.links_t.p0[link_ids]
        p1 = network.links_t.p1[link_ids]
        h2_output = -p1.sum()

        valid_links = h2_output > output_threshold
        if valid_links.sum() == 0:
            continue

        prices = pd.DataFrame(index=p0.index, columns=link_ids)
        for link in link_ids:
            bus = hydrogen_links.loc[link, "bus0"]
            prices[link] = network.buses_t.marginal_price[bus]

        elec_cost = (p0.loc[:, valid_links] * prices.loc[:, valid_links]).sum()
        capex = hydrogen_links.loc[valid_links, "capital_cost"]
        opex = hydrogen_links.loc[valid_links, "marginal_cost"]
        h2_output_valid = h2_output[valid_links]

        with np.errstate(divide="ignore", invalid="ignore"):
            lcoh = (capex + opex + elec_cost) / h2_output_valid
            lcoh = lcoh.replace([np.inf, -np.inf], np.nan)
            lcoh_kg = lcoh / 33.33

        df_bus = pd.DataFrame({
            "lcoh": lcoh_kg,
            "h2_output": h2_output_valid,
            "bus": hydrogen_links.loc[valid_links, "bus0"]
        })

        df_bus["grid_region"] = df_bus["bus"].map(network.buses["grid_region"])
        df_bus = df_bus.dropna(subset=["grid_region", "lcoh", "h2_output"])

        region_summary = (
            df_bus.groupby('grid_region')
            .apply(lambda g: pd.Series({
                'Weighted Average LCOH (USD/kg)': (g['lcoh'] * g['h2_output']).sum() / g['h2_output'].sum(),
                # MWh --> tons
                'Total Hydrogen Dispatch (tons)': g['h2_output'].sum() * 33.33 / 1000
            }))
            .reset_index()
        )

        all_results[simple_year] = region_summary.round(2)

    return all_results


def calculate_total_generation_by_carrier(network, start_date=None, end_date=None):
    import pandas as pd

    # === 1. Time setup ===
    snapshots_slice = slice(
        start_date, end_date) if start_date and end_date else slice(None)
    snapshots = network.snapshots[snapshots_slice]
    timestep_h = (snapshots[1] - snapshots[0]).total_seconds() / 3600

    # === 2. Define relevant carriers ===
    gen_and_sto_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac', 'nuclear',
        'geothermal', 'ror', 'hydro'
    }
    link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', 'lignite']

    # === 3. Identify electric buses ===
    electric_buses = set(
        network.buses.index[
            ~network.buses.carrier.str.contains(
                "heat|gas|H2|oil|coal", case=False, na=False)
        ]
    )

    # === 4. Generators ===
    gen = network.generators[network.generators.carrier.isin(
        gen_and_sto_carriers)]
    gen_p = network.generators_t.p.loc[snapshots_slice, gen.index].clip(
        lower=0)
    gen_dispatch = gen_p.groupby(gen['carrier'], axis=1).sum()
    gen_energy_mwh = gen_dispatch.sum() * timestep_h

    # === 5. Storage units ===
    sto = network.storage_units[network.storage_units.carrier.isin(
        gen_and_sto_carriers)]
    sto_p = network.storage_units_t.p.loc[snapshots_slice, sto.index].clip(
        lower=0)
    sto_dispatch = sto_p.groupby(sto['carrier'], axis=1).sum()
    sto_energy_mwh = sto_dispatch.sum() * timestep_h

    # === 6. Link-based generation ===
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

    # === 7. Combine all sources ===
    total_energy_twh = pd.concat([
        gen_energy_mwh / 1e6,    # MW → TWh
        sto_energy_mwh / 1e6,
        link_dispatch
    ])

    total_energy_twh = total_energy_twh.groupby(total_energy_twh.index).sum()
    total_energy_twh = total_energy_twh[total_energy_twh > 0].round(2)
    total_energy_twh = total_energy_twh.sort_values(ascending=False)

    return total_energy_twh


def plot_hydrogen_dispatch(networks, h2_carriers, output_threshold=1.0):
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
        return

    # Second pass: generate plots with fixed y-axis
    for key, df in dispatch_series_by_network.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        df.plot.area(ax=ax, linewidth=0)
        year = key[-4:]  # Extract the year
        ax.set_title(f"Electricity Dispatch – {year}")
        ax.set_title(f"Hydrogen Dispatch by technology – {year}", fontsize=14)
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


def analyze_ft_costs_by_region(networks: dict):
    """
    Compute and display total Fischer-Tropsch fuel production and
    total marginal cost (USD/MWh) by grid region for each network.
    """
    for name, n in networks.items():
        # Identify Fischer-Tropsch links
        ft_links = n.links[n.links.carrier.str.contains("Fischer", case=False, na=False)].copy()
        if ft_links.empty:
            print(f"\n{name}: No Fischer-Tropsch links found.")
            continue

        ft_link_ids = ft_links.index

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
                "production [MWh]": output_mwh,
                "marginal_cost_total [USD/MWh]": total_cost_per_mwh
            }

        # Map each link's output bus to its grid_region
        df_links = pd.DataFrame.from_dict(marginal_cost_total, orient="index")
        bus_to_region = n.buses["grid_region"].to_dict()
        df_links["grid_region"] = df_links["bus"].map(bus_to_region)
        df_links = df_links.dropna(subset=["grid_region"])

        # Aggregate by grid_region (sum production, weighted average of cost)
        grouped = df_links.groupby("grid_region")
        sum_prod = grouped["production [MWh]"].sum()
        weighted_cost = grouped.apply(
            lambda g: (g["marginal_cost_total [USD/MWh]"] *
                       g["production [MWh]"]).sum() / g["production [MWh]"].sum()
        )

        df_region_result = pd.DataFrame({
            "production [MWh]": sum_prod,
            "marginal_cost_total [USD/MWh]": weighted_cost
        })

        # Round to 2 decimals
        df_region_result = df_region_result.round(2)

        # Reset index to make 'grid_region' a visible column
        df_region_result = df_region_result.reset_index()

        # Rename columns
        df_region_result = df_region_result.rename(columns={
            "grid_region": "Grid region",
            "production [MWh]": "Production (MWh)",
            "marginal_cost_total [USD/MWh]": "e-kerosene marginal cost (USD/MWh)"
        })

        # Format numbers and hide index
        styled = df_region_result.style.format({
            "Production (MWh)": "{:,.2f}",
            "e-kerosene marginal cost (USD/MWh)": "{:,.2f}"
        }).hide(axis="index")

        # Extract year from network name
        match = re.search(r"\d{4}", name)
        year = match.group() if match else "unknown"

        print(f"\nYear: {year}\n")
        display(styled)


def compute_aviation_fuel_demand(networks):
    results = {}

    for name, n in networks.items():
        # Extract year
        year = ''.join(filter(str.isdigit, name[-4:]))

        # Trova i load per ciascun carrier
        kerosene_load_names = n.loads[n.loads.carrier ==
                                      "kerosene for aviation"].index
        ekerosene_load_names = n.loads[n.loads.carrier ==
                                       "e-kerosene for aviation"].index

        # Timestep duration
        weightings = n.snapshot_weightings.generators

        # Energy in MWh
        kerosene_mwh = n.loads_t.p[kerosene_load_names].multiply(
            weightings, axis=0).sum().sum()
        ekerosene_mwh = n.loads_t.p[ekerosene_load_names].multiply(
            weightings, axis=0).sum().sum()

        # Conversion in TWh
        kerosene_twh = kerosene_mwh / 1e6
        ekerosene_twh = ekerosene_mwh / 1e6

        results[year] = {
            "Kerosene (TWh)": kerosene_twh,
            "e-Kerosene (TWh)": ekerosene_twh
        }

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Year"
    df.reset_index(inplace=True)

    # Totali e percentuali
    df["Total (TWh)"] = df["Kerosene (TWh)"] + df["e-Kerosene (TWh)"]
    df["e-Kerosene Share (%)"] = (df["e-Kerosene (TWh)"] /
                                  df["Total (TWh)"]) * 100

    # Remove values close to zero
    df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').applymap(
        lambda x: 0 if abs(x) < 1e-6 else x
    )

    return df


def compute_emissions_from_links(net):
    import pandas as pd
    import re

    results = []

    bus_cols = [
        col for col in net.links.columns if re.fullmatch(r"bus\d+", col)]

    for i, row in net.links.iterrows():
        carrier = row["carrier"]
        link_name = i
        co2_atmosphere = 0.0
        co2_stored = 0.0

        for j, bus_col in enumerate(bus_cols):
            bus_val = str(row[bus_col]).lower().strip()
            p_col = f"p{j}"

            if p_col not in net.links_t or link_name not in net.links_t[p_col]:
                continue

            flow = net.links_t[p_col][link_name].mean() * 8760  # MWh/year

            if "co2 atmosphere" in bus_val or "co2 atmoshpere" in bus_val:
                co2_atmosphere -= flow
            elif "co2 stored" in bus_val:
                co2_stored += flow

        results.append({
            "link": link_name,
            "carrier": carrier,
            "co2_atmosphere": co2_atmosphere,
            "co2_stored": co2_stored,
        })

    df = pd.DataFrame(results)
    df = df[(df["co2_atmosphere"] != 0) | (df["co2_stored"] != 0)]

    # Group per carrier and convert in Mt
    summary = df.groupby("carrier")[
        ["co2_atmosphere", "co2_stored"]].sum().reset_index()
    summary["net_emissions"] = summary["co2_atmosphere"] - \
        summary["co2_stored"]

    summary[["co2_atmosphere", "co2_stored", "net_emissions"]] = (
        summary[["co2_atmosphere", "co2_stored", "net_emissions"]] / 1e6
    ).round(2)

    summary = summary.rename(columns={
        "co2_atmosphere": "co2_atmosphere [Mt CO2]",
        "co2_stored": "co2_stored [Mt CO2]",
        "net_emissions": "net_emissions [Mt CO2]"
    })

    return summary


def compute_emissions_grouped(net, carrier_groups):
    import pandas as pd
    import re

    results = []

    bus_cols = [
        col for col in net.links.columns if re.fullmatch(r"bus\d+", col)]

    for i, row in net.links.iterrows():
        carrier = row["carrier"]
        link_name = i
        co2_atmosphere = 0.0
        co2_stored = 0.0

        for j, bus_col in enumerate(bus_cols):
            bus_val = str(row[bus_col]).lower().strip()
            p_col = f"p{j}"

            if p_col not in net.links_t or link_name not in net.links_t[p_col]:
                continue

            flow = net.links_t[p_col][link_name].mean() * 8760  # MWh/year

            if "co2 atmosphere" in bus_val or "co2 atmoshpere" in bus_val:
                co2_atmosphere -= flow
            elif "co2 stored" in bus_val:
                co2_stored += flow

        results.append({
            "link": link_name,
            "carrier": carrier,
            "co2_atmosphere": co2_atmosphere,
            "co2_stored": co2_stored,
        })

    df = pd.DataFrame(results)

    all_grouped_carriers = set(sum(carrier_groups.values(), []))
    df = df[df["carrier"].isin(all_grouped_carriers)]

    carrier_summary = df.groupby(
        "carrier")[["co2_atmosphere", "co2_stored"]].sum().reset_index()

    group_results = []
    for group_name, group_carriers in carrier_groups.items():
        group_df = carrier_summary[carrier_summary["carrier"].isin(
            group_carriers)]
        co2_atm = group_df["co2_atmosphere"].sum()
        co2_stored = group_df["co2_stored"].sum()
        net_emissions = co2_atm - co2_stored

        group_results.append({
            "carrier_group": group_name,
            "co2_atmosphere [Mt CO2]": round(co2_atm / 1e6, 2),
            "co2_stored [Mt CO2]": round(co2_stored / 1e6, 2),
            "net_emissions [Mt CO2]": round(net_emissions / 1e6, 2)
        })

    return pd.DataFrame(group_results)


def compute_emissions_by_state(net, carrier_groups):

    results = []

    bus_cols = [
        col for col in net.links.columns if re.fullmatch(r"bus\d+", col)]

    for i, row in net.links.iterrows():
        carrier = row["carrier"]
        link_name = i
        co2_atmos = 0.0
        co2_stored = 0.0

        group = next((g for g, carriers in carrier_groups.items()
                     if carrier in carriers), None)
        if group is None:
            continue

        for j, bus_col in enumerate(bus_cols):
            bus_val = str(row[bus_col]).lower().strip()
            p_col = f"p{j}"

            if p_col not in net.links_t or link_name not in net.links_t[p_col]:
                continue

            flow = net.links_t[p_col][link_name].mean() * 8760  # Mt CO2/year

            if "co2 atmosphere" in bus_val or "co2 atmoshpere" in bus_val:
                co2_atmos -= flow
            elif "co2 stored" in bus_val:
                co2_stored += flow

        state = "Unknown"
        for bus_col in bus_cols:
            bus = row[bus_col]
            if bus in net.buses.index:
                s = net.buses.loc[bus, "state"]
                if pd.notna(s) and s != "Unknown":
                    state = s
                    break

        results.append({
            "state": state,
            "group": group,
            "co2_atmosphere": co2_atmos,
            "co2_stored": co2_stored
        })

    df = pd.DataFrame(results)

    summary = df.groupby(["state", "group"])[
        ["co2_atmosphere", "co2_stored"]].sum().reset_index()
    summary["net_emissions"] = summary["co2_atmosphere"] - \
        summary["co2_stored"]

    summary[["co2_atmosphere", "co2_stored", "net_emissions"]] = (
        summary[["co2_atmosphere", "co2_stored", "net_emissions"]] /
        1e6  # Convert to MtCO2
    ).round(2)

    return summary


def plot_emissions_maps_by_group(all_state_emissions, path_shapes, title):

    # Upload shapefile and force CRS
    gdf_states = gpd.read_file(path_shapes).to_crs("EPSG:4326")
    gdf_states["State"] = gdf_states["ISO_1"].str[-2:]

    groups = all_state_emissions["group"].unique()
    n = len(groups)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(8 * ncols, 8 * nrows))
    axes = axes.flat if n > 1 else [axes]

    for i, group in enumerate(groups):
        ax = axes[i]
        df_group = all_state_emissions[all_state_emissions["group"] == group].copy(
        )
        df_group = df_group.rename(columns={"State": "State"})

        merged = gdf_states.merge(df_group, on="State", how="left")

        merged.plot(
            column="net_emissions",
            cmap="Reds",
            legend=True,
            ax=ax,
            missing_kwds={"color": "lightgrey", "label": "No data"},
            edgecolor="black"
        )

        ax.set_title(f"{group}", fontsize=12)
        ax.set_xlim([-180, -65])
        ax.set_ylim([15, 75])
        ax.axis("off")

        leg = ax.get_legend()
        if leg:
            leg.set_bbox_to_anchor((1, 0.5))
            for t in leg.get_texts():
                t.set_fontsize(8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Net Emissions by process and State [MtCO2/yr] - {title}", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
