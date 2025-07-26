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
    hydrogen_links = network.links.query("carrier in @h2_carriers_buses").copy()

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
        fill_value=0
    )

    # Add state and grid_region information
    h2_capacity_data['state'] = h2_capacity_data.index.map(network.buses.state)
    h2_capacity_data['grid_region'] = h2_capacity_data.index.map(network.buses.grid_region)

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
            ax.text(x[i], total + 0.01 * ymax, f"{total:.0f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(state, rotation=30, ha='center')
    ax.set_xlabel("State")
    ax.set_ylabel("Capacity (MW)")

    if ymax > 0:
        ax.set_ylim(0, ymax * 1.05)
    else:
        ax.set_ylim(0, 1)

    ax.set_xlim(-0.5, max_n_states - 0.5)

    ax.set_title(f"\nHydrogen electrolyzer capacity by State and technology - {title}\n")
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

    fig, ax = plt.subplots(figsize=(max_n_grid_regions * bar_width * 5.5, height))

    for tech in techs:
        ax.bar(x, grouped[tech].values, bar_width, bottom=bottoms, label=tech)
        bottoms += grouped[tech].values

    # Add text on top of each stacked bar
    for i in range(n):
        total = grouped.iloc[i].sum()
        ax.text(x[i], total + 0.01 * ymax, f"{total:.0f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(grid_region, rotation=30, ha='center')
    ax.set_xlabel("Grid Region")
    ax.set_ylabel("Capacity (MW)")

    if ymax > 0:
        ax.set_ylim(0, ymax * 1.05)
    else:
        ax.set_ylim(0, 1)

    ax.set_xlim(-0.5, max_n_grid_regions - 0.5)

    ax.set_title(f"\nHydrogen electrolyzer capacity by Grid Region and technology - {title}\n")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

    plt.tight_layout()
    plt.show()


def create_hydrogen_capacity_map(network, path_shapes, distance_crs=4326, min_capacity_mw=10):
    """
    Create a map with pie charts showing hydrogen electrolyzer capacity breakdown by type for each state
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
        f"Plotting {len(states_to_plot)} states with ≥{min_capacity_mw} MW hydrogen capacity")

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
            legend_elements.append(Line2D([0], [0], color='none', label=f'— {group} —'))
            
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
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'),
                transform=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.autoscale(False)
    ax.set_position([0.05, 0.05, 0.9, 0.9])

    ax.set_title(
        f"Fischer-Tropsch Capacity by State (GW){year_str}", fontsize=12)
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
    pie_scale = 0.2  # degrees per GW
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
        f"Fischer-Tropsch Capacity by Grid Region (GW){year_str}", fontsize=12, pad=20)
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
        'geothermal', 'ror', 'hydro', 'solar rooftop',
    }
    link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', "biomass CHP", "gas CHP"]
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
        ax.set_title(f"Electricity dispatch – {key}")
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

    # Define carrier sets by component
    gen_carriers = {
        'csp', 'solar', 'onwind', 'offwind-dc', 'offwind-ac',
        'nuclear', 'geothermal', 'ror', 'hydro', 'solar rooftop',
    }

    storage_carriers = {
        'battery', 'hydro', 'PHS'  # Customize based on actual model
    }

    link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', 'lignite', "urban central solid biomass CHP", "urban central gas CHP"]

    electric_buses = set(n.buses[n.buses.carrier == 'AC'].index)

    # Generators
    gen = n.generators[n.generators.carrier.isin(gen_carriers)].copy()
    gen_dispatch = n.generators_t.p[gen.index].multiply(snapshot_weights, axis=0)
    gen['energy'] = gen_dispatch.sum()
    gen = gen[(gen.p_nom_opt > 0) & (gen.energy > 0)]
    gen['lcoe'] = (gen.capital_cost * gen.p_nom_opt + gen.marginal_cost * gen.energy) / gen.energy
    gen['type'] = 'generator'

    # Storage units
    sto = n.storage_units[n.storage_units.carrier.isin(storage_carriers)].copy()
    sto_dispatch = n.storage_units_t.p[sto.index].clip(lower=0).multiply(snapshot_weights, axis=0)
    sto['energy'] = sto_dispatch.sum()
    sto = sto[(sto.p_nom_opt > 0) & (sto.energy > 0)]
    sto['lcoe'] = (sto.capital_cost * sto.p_nom_opt + sto.marginal_cost * sto.energy) / sto.energy
    sto['type'] = 'storage'

    # Links (using p1 logic consistent with total generation)
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

    link = link[(link.p_nom_opt > 0) & (link.fuel_usage > 0)]
    link['lcoe'] = (link.capital_cost * link.p_nom_opt + link.marginal_cost * link.fuel_usage + link.fuel_cost * link.fuel_usage) / link.energy
    link['type'] = 'link'

    # Merge data
    gen_data = gen[['bus', 'carrier', 'lcoe', 'type', 'energy']]
    sto_data = sto[['bus', 'carrier', 'lcoe', 'type', 'energy']]
    link_data = link[['bus1', 'carrier', 'lcoe', 'type', 'energy']].rename(columns={'bus1': 'bus'})

    lcoe_data = pd.concat([gen_data, sto_data, link_data], axis=0).dropna()
    lcoe_data = lcoe_data.merge(n.buses[['x', 'y', 'grid_region']], left_on='bus', right_index=True)

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
            total_cost=('lcoe', lambda x: (x * lcoe_data.loc[x.index, 'energy']).sum())
        )
        .reset_index()
    )
    region_summary['lcoe'] = region_summary['total_cost'] / region_summary['dispatch_mwh']
    region_summary['dispatch'] = region_summary['dispatch_mwh'] / 1e6

    weighted_avg_grid_region = (
        region_summary.groupby('grid_region')
        .apply(lambda df: (df['dispatch_mwh'] * df['lcoe']).sum() / df['dispatch_mwh'].sum())
    )

    table = region_summary.pivot(index='grid_region', columns='carrier', values=['lcoe', 'dispatch'])
    table.columns = [
        f"{carrier} {metric} ({'USD/MWh' if metric == 'lcoe' else 'TWh'})"
        for metric, carrier in table.columns
    ]
    table = table.reset_index()

    dispatch_cols = [col for col in table.columns if 'dispatch' in col.lower()]
    for col in dispatch_cols:
        table[col] = pd.to_numeric(table[col], errors='coerce').fillna(0.0)

    lcoe_cols = [col for col in table.columns if 'lcoe' in col.lower()]

    min_dispatch_threshold = 1  # TWh
    for lcoe_col in lcoe_cols:
        carrier = lcoe_col.split(" ")[0]
        dispatch_col = next((col for col in dispatch_cols if col.startswith(carrier + " ")), None)
        if dispatch_col:
            mask = table[dispatch_col] < min_dispatch_threshold
            table.loc[mask, lcoe_col] = np.nan

    table[lcoe_cols] = table[lcoe_cols].applymap(lambda x: '-' if pd.isna(x) else round(x, 2))

    grid_region_weighted_lcoe = (
        lcoe_by_bus.merge(lcoe_data[['bus', 'energy']], on='bus', how='left')
        .groupby('grid_region')
        .apply(lambda df: (df['weighted_lcoe'] * df['energy']).sum() / df['energy'].sum())
    )
    table['Weighted Average LCOE (USD/MWh)'] = table['grid_region'].map(grid_region_weighted_lcoe).round(2)

    for col in table.columns:
        if col != 'grid_region':
            table[col] = table[col].round(2) if table[col].dtype != object else table[col]

    vmin = lcoe_by_bus['weighted_lcoe'].quantile(0.05)
    vmax = max(vmin, min(weighted_avg_grid_region.max() * 1.1, lcoe_by_bus['weighted_lcoe'].max()))

    geometry = [Point(xy) for xy in zip(lcoe_by_bus['x'], lcoe_by_bus['y'])]
    lcoe_gdf = gpd.GeoDataFrame(lcoe_by_bus, geometry=geometry, crs=shapes.crs).to_crs(shapes.crs)

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

    ax.set_extent([-130, -60, 20, 50], crs=ccrs.PlateCarree())

    ax.set_title(f'Installed electrolyzer capacity - {title} (only nodes ≥ 10 MW)\n')
    plt.tight_layout()
    plt.show()


def plot_lcoh_maps_by_grid_region(networks, shapes, h2_carriers, output_threshold=1.0, year_title=True):
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

        capex = hydrogen_links.loc[valid_links, "capital_cost"] * hydrogen_links.loc[valid_links, "p_nom_opt"]
        opex = hydrogen_links.loc[valid_links, "marginal_cost"] * p0.loc[:, valid_links].sum(axis=0)
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
        ax.set_title(f"LCOH – {year.split('_')[-1] if year_title else year}", fontsize=14)
        plt.show()


def calculate_weighted_lcoh_table_by_year(networks, h2_carriers, output_threshold=1.0, year_title=True):
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
        capex = hydrogen_links.loc[valid_links, "capital_cost"] * hydrogen_links.loc[valid_links, "p_nom_opt"]
        opex = hydrogen_links.loc[valid_links, "marginal_cost"] * p0.loc[:, valid_links].sum(axis=0)
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
        if year_title:
            all_results[simple_year] = region_summary.round(2)
        else:
            all_results[year_key] = region_summary.round(2)

    return all_results


def calculate_total_generation_by_carrier(network, start_date=None, end_date=None):
    import pandas as pd

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
    link_carriers = ['coal', 'oil', 'OCGT', 'CCGT', 'biomass', 'lignite', "urban central solid biomass CHP", "urban central gas CHP"]

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
        return

    # Second pass: generate plots with fixed y-axis
    for key, df in dispatch_series_by_network.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        df.plot.area(ax=ax, linewidth=0)
        year = key[-4:]  # Extract the year
        ax.set_title(f"Electricity Dispatch – {year if year_title else key}")
        ax.set_title(f"Hydrogen Dispatch by technology – {year if year_title else key}", fontsize=14)
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
                ((n.links.get("p_nom", 0) > 0) & (n.links.get("p_nom_extendable", False) == False))
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

        results[name] = {
            "Kerosene (TWh)": kerosene_twh,
            "e-Kerosene (TWh)": ekerosene_twh
        }

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "scenario"
    df.reset_index(inplace=True)
    df['year'] = year

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
                co2_stored -= flow

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
    stored = summary["co2_stored"]
    summary["net_emissions"] = summary["co2_atmosphere"] + stored.where(stored <= 0, -stored)

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
                co2_stored -= flow

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
    stored = summary["co2_stored"]
    summary["net_emissions"] = summary["co2_atmosphere"] + stored.where(stored <= 0, -stored)

    summary[["co2_atmosphere", "co2_stored", "net_emissions"]] = (
        summary[["co2_atmosphere", "co2_stored", "net_emissions"]] /
        1e6  # Convert to Mt CO2
    ).round(2)

    return summary


def plot_emissions_maps_by_group(all_state_emissions, path_shapes, title, vmin=None, vmax=None):

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
        df_group = all_state_emissions[all_state_emissions["group"] == group].copy()

        merged = gdf_states.merge(df_group, on="State", how="left")

        merged.plot(
            column="net_emissions",
            cmap="RdYlGn_r",
            legend=True,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            missing_kwds={"color": "lightgrey", "label": "No data"},
            edgecolor="black"
        )

        ax.set_title(f"{group}", fontsize=12)
        ax.set_xlim([-130, -65])
        ax.set_ylim([20, 55])
        ax.axis("off")

        leg = ax.get_legend()
        if leg:
            leg.set_bbox_to_anchor((1, 0.5))
            for t in leg.get_texts():
                t.set_fontsize(8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Net CO2 emissions by process and State (Mt CO2/year) - {title}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
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
                ~network.buses.carrier.str.contains("heat|gas|H2|oil|coal", case=False, na=False)
            ]
        )

        # Generators
        gen = network.generators[network.generators.carrier.isin(gen_and_sto_carriers)].copy()
        gen["state"] = gen["bus"].map(network.buses["state"])
        gen = gen[gen["state"].notna()]

        gen_p = network.generators_t.p.loc[snapshots_slice, gen.index].clip(lower=0)
        gen_energy = gen_p.multiply(timestep_h).sum()  # MWh per generator
        gen_energy = gen_energy.to_frame(name="energy_mwh")
        gen_energy["carrier"] = gen.loc[gen_energy.index, "carrier"]
        gen_energy["state"] = gen.loc[gen_energy.index, "state"]

        # Storage
        sto = network.storage_units[network.storage_units.carrier.isin(gen_and_sto_carriers)].copy()
        sto["state"] = sto["bus"].map(network.buses["state"])
        sto = sto[sto["state"].notna()]

        sto_p = network.storage_units_t.p.loc[snapshots_slice, sto.index].clip(lower=0)
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
        state_ces = all_energy[all_energy["carrier"].isin(ces_carriers)].groupby("state")["energy_mwh"].sum()
        state_res = all_energy[all_energy["carrier"].isin(res_carriers)].groupby("state")["energy_mwh"].sum()

        df = pd.DataFrame({
            "Total (MWh)": state_totals,
            "CES_energy": state_ces,
            "RES_energy": state_res
        }).fillna(0)

        df["% CES"] = 100 * df["CES_energy"] / df["Total (MWh)"]
        df["% RES"] = 100 * df["RES_energy"] / df["Total (MWh)"]

        # Targets
        if year_str in ces.columns:
            df["% CES target"] = df.index.map(lambda state: ces[year_str].get(state, float("nan")))
        else:
            df["% CES target"] = float("nan")

        if year_str in res.columns:
            df["% RES target"] = df.index.map(lambda state: res[year_str].get(state, float("nan")))
        else:
            df["% RES target"] = float("nan")

        df["% RES target"] = df["% RES target"].apply(lambda x: "N/A" if pd.isna(x) else round(x * 100, 2))
        df["% CES target"] = df["% CES target"].apply(lambda x: "N/A" if pd.isna(x) else round(x * 100, 2))

        df = df[["% RES", "% RES target", "% CES", "% CES target"]].round(2)
        if multiple_scenarios:
            results[name] = df.sort_index()
        else:
            results[year] = df.sort_index()
    return results


def plot_network_generation_and_transmission(n, key, tech_colors, nice_names, regions_onshore, title_year=True):

    # Define generation/link carriers
    gen_carriers = {
        "onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop",
        "csp", "nuclear", "geothermal", "ror", "PHS",
    }
    link_carriers = {
        "OCGT", "CCGT", "coal", "oil", "biomass", "biomass CHP", "gas CHP"
    }

    # Generator and storage capacity
    gen_p_nom_opt = n.generators[n.generators.carrier.isin(gen_carriers)]
    gen_p_nom_opt = gen_p_nom_opt.groupby(["bus", "carrier"]).p_nom_opt.sum()

    sto_p_nom_opt = n.storage_units[n.storage_units.carrier.isin(gen_carriers)]
    sto_p_nom_opt = sto_p_nom_opt.groupby(["bus", "carrier"]).p_nom_opt.sum()

    # Link capacity
    link_mask = (
        n.links.efficiency.notnull()
        & (n.links.p_nom_opt > 0)
        & n.links.carrier.isin(link_carriers)
    )
    electricity_links = n.links[link_mask].copy()
    electricity_links["electric_output"] = electricity_links.p_nom_opt * electricity_links.efficiency
    link_p_nom_opt = electricity_links.groupby(["bus1", "carrier"]).electric_output.sum()
    link_p_nom_opt.index = link_p_nom_opt.index.set_names(["bus", "carrier"])

    # Combine all
    bus_carrier_capacity = pd.concat([gen_p_nom_opt, sto_p_nom_opt, link_p_nom_opt])
    bus_carrier_capacity = bus_carrier_capacity.groupby(level=[0, 1]).sum()
    bus_carrier_capacity = bus_carrier_capacity[bus_carrier_capacity > 0]

    # Valid buses with coordinates
    valid_buses = n.buses.dropna(subset=["x", "y"])
    valid_buses = valid_buses[
        (valid_buses["x"] > -200) & (valid_buses["x"] < 200) &
        (valid_buses["y"] > -90) & (valid_buses["y"] < 90)
    ]

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

    # Setup map
    fig, ax = plt.subplots(figsize=(22, 10), subplot_kw={"projection": ccrs.PlateCarree()})
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

    # Store original links and apply filter
    original_links = n.links.copy()
    n.links = n.links[n.links.index.isin(electricity_links.index)]

    # Plot network
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

    # Draw pie charts for buses
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

    # Bus Capacity Legend using Line2D markers
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

    carrier_handles = [
        mpatches.Patch(color=tech_colors.get(c, 'gray'), label=nice_names.get(c, c))
        for c in sorted(capacity_df.columns) if capacity_df[c].sum() > 0
    ]
    carrier_legend = ax.legend(
        handles=carrier_handles,
        title="Technology",
        title_fontsize=13,
        fontsize=11,
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1.125, 0.54),
        ncol=2,
        labelspacing=1.05,
    )

    ax.add_artist(bus_legend)
    ax.add_artist(ac_legend)
    ax.add_artist(dc_legend)
    ax.add_artist(carrier_legend)

    ax.set_extent([-130, -65, 20, 50], crs=ccrs.PlateCarree())
    ax.autoscale(False)

    year = key[-4:]
    ax.set_title(f"Installed electricity generation and transmission capacity – {year if title_year else key}", fontsize=14)

    plt.tight_layout()
    plt.show()


def compute_installed_capacity_by_carrier(networks, nice_names=None, display_result=True, column_year=True):
    import pandas as pd

    totals_by_carrier = {}

    for name, net in networks.items():
        gen_carriers = {
            "onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop",
            "csp", "nuclear", "geothermal", "ror", "PHS", "hydro",
        }
        link_carriers = {
            "OCGT", "CCGT", "coal", "oil", "biomass", "biomass CHP", "gas CHP"
        }

        # Generators
        gen = net.generators.copy()
        gen['carrier'] = gen['carrier'].replace({'offwind-ac': 'offwind', 'offwind-dc': 'offwind'})
        gen = gen[gen.carrier.isin(gen_carriers)]
        gen_totals = gen.groupby('carrier')['p_nom_opt'].sum()

        # Storage
        sto = net.storage_units.copy()
        sto = sto[sto.carrier.isin(gen_carriers)]
        sto_totals = sto.groupby('carrier')['p_nom_opt'].sum()

        # Links (efficiency-scaled output)
        links = net.links.copy()
        mask = (
            links.efficiency.notnull()
            & (links.p_nom_opt > 0)
            & links.carrier.isin(link_carriers)
        )
        links = links[mask]
        links_totals = links.groupby('carrier').apply(
            lambda df: (df['p_nom_opt'] * df['efficiency']).sum()
        )

        # Combine and store
        all_totals = pd.concat([gen_totals, sto_totals, links_totals])
        all_totals = all_totals.groupby(all_totals.index).sum()
        all_totals = all_totals[all_totals > 0]
        totals_by_carrier[name] = all_totals

    # Assemble final dataframe
    carrier_capacity_df = pd.DataFrame(totals_by_carrier).fillna(0)

    # Extract years and sort
    if column_year == True:
        carrier_capacity_df.columns = [int(name[-4:]) for name in carrier_capacity_df.columns]
    carrier_capacity_df = carrier_capacity_df[sorted(carrier_capacity_df.columns)]

    # Convert to GW
    carrier_capacity_df = carrier_capacity_df / 1000
    carrier_capacity_df = carrier_capacity_df.round(2)

    # Filter rows with any nonzero value
    carrier_capacity_df = carrier_capacity_df.loc[carrier_capacity_df.sum(axis=1) > 0]

    # Rename index if nice_names is provided
    if nice_names:
        carrier_capacity_df = carrier_capacity_df.rename(index=nice_names)

    if display_result:
        print("\nInstalled capacity by technology (GW)\n")
        display(carrier_capacity_df)

    return carrier_capacity_df


def compute_system_costs(network, rename_capex, rename_opex, name_tag):
    import pandas as pd

    costs_raw = network.statistics()[['Capital Expenditure', 'Operational Expenditure']]
    year_str = name_tag[-4:]

    # CAPEX
    capex_raw = costs_raw[['Capital Expenditure']].reset_index()
    capex_raw['tech_label'] = capex_raw['carrier'].map(rename_capex).fillna(capex_raw['carrier'])
    capex_raw['main_category'] = capex_raw['tech_label']

    capex_grouped = capex_raw.groupby('tech_label', as_index=False).agg({
        'Capital Expenditure': 'sum',
        'main_category': 'first'
    })
    capex_grouped['cost_type'] = 'Capital expenditure'
    capex_grouped.rename(columns={'Capital Expenditure': 'cost_billion'}, inplace=True)
    capex_grouped['cost_billion'] /= 1e9
    capex_grouped['year'] = year_str
    capex_grouped['scenario'] = name_tag

    # OPEX
    opex_raw = costs_raw[['Operational Expenditure']].reset_index()
    opex_raw['tech_label'] = opex_raw['carrier'].map(rename_opex).fillna(opex_raw['carrier'])
    opex_raw['main_category'] = opex_raw['tech_label']

    opex_grouped = opex_raw.groupby('tech_label', as_index=False).agg({
        'Operational Expenditure': 'sum',
        'main_category': 'first'
    })
    opex_grouped['cost_type'] = 'Operational expenditure'
    opex_grouped.rename(columns={'Operational Expenditure': 'cost_billion'}, inplace=True)
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
                fuel_cost = (-p0 * fuel_price * weightings).sum()

                # Other OPEX (marginal cost of link, positive)
                marginal_cost = links.loc[link_id, 'marginal_cost']
                other_opex = (-p0 * marginal_cost * weightings).sum()

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
    link_opex_df['tech_label'] = link_opex_df['tech_label'].replace(rename_opex)
    link_opex_df['main_category'] = link_opex_df['tech_label']

    return pd.concat([capex_grouped, opex_grouped, link_opex_df], ignore_index=True)


def plot_stacked_costs_by_year(cost_data, cost_type_label, tech_colors=None, index='year'):

    # Clean data
    data_filtered = cost_data[
        (cost_data['cost_type'] == cost_type_label) &
        (cost_data['cost_billion'] != 0)
    ].copy()

    if data_filtered.empty:
        print("No data to plot.")
        return

    # Pivot table: year x tech_label
    pivot_table = data_filtered.pivot_table(
        index=index,
        columns='tech_label',
        values='cost_billion',
        aggfunc='sum'
    ).fillna(0)

    # Mapping dictionaries
    label_to_macro = data_filtered.set_index('tech_label')['macro_category'].to_dict()
    label_to_category = data_filtered.set_index('tech_label')['main_category'].to_dict()

    # Define desired macro-category order (top-to-bottom in legend and bars)
    desired_macro_order = [
        'Hydrogen & e-fuels',
        'Biofuels synthesis',
        'DAC',
        'End-uses',
        'Industry',
        'Power & heat generation',
        'Storage',
        'Transmission & distribution',
        'Emissions',
        'Other'
    ]
    macro_order_map = {macro: i for i, macro in enumerate(desired_macro_order)}

    # Final tech_label order (top to bottom)
    all_labels = data_filtered['tech_label'].drop_duplicates().tolist()
    ordered_labels = sorted(
        all_labels,
        key=lambda lbl: (macro_order_map.get(label_to_macro.get(lbl, 'Other'), 999), all_labels.index(lbl))
    )

    # Stack order is reversed (bottom to top)
    pivot_table = pivot_table[ordered_labels[::-1]]

    # Assign colors
    def get_color(label):
        category = label_to_category.get(label, label)
        return tech_colors.get(category, '#999999')
    color_values = [get_color(label) for label in pivot_table.columns]

    # Plot
    ax = pivot_table.plot(
        kind='bar',
        stacked=True,
        color=color_values,
        figsize=(12, 6)
    )
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel("Years (-)")
    ax.set_ylabel(f"{cost_type_label} (Billion USD)")
    ax.set_title(f"{cost_type_label}")
    ax.set_xticklabels(pivot_table.index, rotation=0)
    plt.tight_layout()

    # Group for legend
    grouped_labels = defaultdict(list)
    for label in ordered_labels:
        macro = label_to_macro.get(label, 'Other')
        grouped_labels[macro].append(label)

    # Order groups in legend
    legend_elements = []
    for macro in desired_macro_order:
        if macro in grouped_labels:
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label=f'— {macro} —'))
            for label in grouped_labels[macro]:
                legend_elements.append(Patch(facecolor=get_color(label), label=label))

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    # Add y-limits margin
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    plt.show()


def assign_macro_category(row, categories_capex, categories_opex):
    if row['cost_type'] == 'Capital expenditure':
        return categories_capex.get(row['tech_label'], 'Other')
    elif row['cost_type'] == 'Operational expenditure':
        return categories_opex.get(row['tech_label'], 'Other')
    else:
        return 'Other'


def calculate_total_inputs_outputs_ft(networks, ft_carrier="Fischer-Tropsch"):
    """
    Calculates input/output flows for Fischer-Tropsch across a set of PyPSA networks.
    
    For each network:
    - electricity used (TWh)
    - hydrogen used (TWh and tons)
    - CO2 used (Mt), only negative p2 flows
    - e-kerosene produced (TWh)
    
    Returns:
        pd.DataFrame with results per year.
    """
    
    results = []

    for name, net in networks.items():
        ft_links = net.links[net.links.carrier == ft_carrier]
        if ft_links.empty:
            continue

        ft_link_ids = ft_links.index

        # Determine timestep in hours
        timestep_hours = (
            (net.snapshots[1] - net.snapshots[0]).total_seconds() / 3600
            if len(net.snapshots) > 1 else 1.0
        )

        # Helper function to get total energy (TWh) across selected links
        def get_energy(df, link_ids):
            df_selected = df.reindex(columns=link_ids, fill_value=0.0)
            return (df_selected * timestep_hours).sum().sum() / 1e6  # MWh → TWh

        # Energy flows
        elec_input_twh = get_energy(net.links_t.p3, ft_link_ids)
        h2_input_twh = get_energy(net.links_t.p0, ft_link_ids)
        fuel_output_twh = get_energy(net.links_t.p1, ft_link_ids)

        def get_co2_flow(df, link_ids):
            df_selected = df.reindex(columns=link_ids, fill_value=0.0)
            return (df_selected * timestep_hours).sum().sum() / 1e6  # t → Mt
        
        co2_input_mt = get_co2_flow(net.links_t.p2, ft_link_ids)

        # Hydrogen in tons
        h2_tons = h2_input_twh * 1e9 / 33.33 / 1000  # TWh → kWh → kg → t

        # Extract year from network name
        match = re.search(r"\d{4}", name)
        if not match:
            continue
        year = int(match.group())

        results.append({
            "Year": year,
            "Used electricity (TWh)": elec_input_twh,
            "Used hydrogen (TWh)": h2_input_twh,
            "Used hydrogen (t)": h2_tons,
            "Used CO2 (Mt)": co2_input_mt,
            "Produced e-kerosene (TWh)": -fuel_output_twh,
        })

    # Compile and sort results
    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["Year"] = df["Year"].astype(int)
    return df.sort_values("Year")

def compute_ekerosene_production_cost_by_region(networks: dict):
    import pandas as pd
    import re

    for name, net in networks.items():
        year_match = re.search(r"\d{4}", name)
        if not year_match:
            continue
        year = year_match.group()

        ft_links = net.links[
            (net.links.carrier.str.contains("Fischer", case=False, na=False)) &
            (
                (net.links.get("p_nom_opt", 0) > 0) |
                ((net.links.get("p_nom", 0) > 0) & (net.links.get("p_nom_extendable", False) == False))
            )
        ]
        if ft_links.empty:
            continue

        ft_link_ids = [
            l for l in ft_links.index
            if all(l in getattr(net.links_t, p).columns for p in ["p0", "p1", "p2", "p3"])
        ]
        if not ft_link_ids:
            continue

        timestep_hours = (
            (net.snapshots[1] - net.snapshots[0]).total_seconds() / 3600
            if len(net.snapshots) > 1 else 1.0
        )

        records = []

        for link in ft_link_ids:
            try:
                region = net.buses.at[ft_links.at[link, "bus1"], "grid_region"]
            except KeyError:
                continue

            elec_price = net.buses_t.marginal_price[ft_links.at[link, "bus3"]]
            h2_price   = net.buses_t.marginal_price[ft_links.at[link, "bus0"]]
            co2_price  = net.buses_t.marginal_price[ft_links.at[link, "bus2"]]

            p1 = -net.links_t.p1[link] * timestep_hours  # Output (MWh)
            p3 =  net.links_t.p3[link] * timestep_hours  # Electricity in (MWh)
            p0 =  net.links_t.p0[link] * timestep_hours  # H2 in (MWh)
            p2 =  net.links_t.p2[link].clip(upper=0) * timestep_hours  # CO2 in (t), negative only

            prod = p1.sum() / 1e6  # MWh → TWh

            if prod < 1e-3:
                continue  # skip links with negligible production

            fuel_output_safe = p1.sum()

            elec_cost = (p3 * elec_price).sum() / fuel_output_safe
            h2_cost   = (p0 * h2_price).sum() / fuel_output_safe
            co2_cost  = (-p2 * co2_price).sum() / fuel_output_safe  # cost per MWh of product

            total_cost = elec_cost + h2_cost + co2_cost  # USD/MWh of product

            records.append({
                "Grid Region": region,
                "Production (TWh)": prod,
                "Electricity cost (USD/MWh e-kerosene)": elec_cost,
                "Hydrogen cost (USD/MWh e-kerosene)": h2_cost,
                "CO2 cost (USD/MWh e-kerosene)": co2_cost,
                "Total production cost (USD/MWh e-kerosene)": total_cost,
            })

        if not records:
            continue

        df = pd.DataFrame(records)

        # Weighted average per region based on production
        def weighted_avg(group, col):
            return (group[col] * group["Production (TWh)"]).sum() / group["Production (TWh)"].sum()

        grouped = df.groupby("Grid Region")
        df_grouped = grouped.apply(lambda g: pd.Series({
            "Production (TWh)": g["Production (TWh)"].sum(),
            "Electricity cost (USD/MWh e-kerosene)": weighted_avg(g, "Electricity cost (USD/MWh e-kerosene)"),
            "Hydrogen cost (USD/MWh e-kerosene)": weighted_avg(g, "Hydrogen cost (USD/MWh e-kerosene)"),
            "CO2 cost (USD/MWh e-kerosene)": weighted_avg(g, "CO2 cost (USD/MWh e-kerosene)"),
            "Total production cost (USD/MWh e-kerosene)": weighted_avg(g, "Total production cost (USD/MWh e-kerosene)"),
        }))

        df_grouped = df_grouped[df_grouped["Production (TWh)"] >= 1e-3]
        if df_grouped.empty:
            continue

        print(f"\n{year}:\n")
        display(
            df_grouped.round(2).style.format({
                "Production (TWh)": "{:,.2f}",
                "Electricity cost (USD/MWh e-kerosene)": "{:,.2f}",
                "Hydrogen cost (USD/MWh e-kerosene)": "{:,.2f}",
                "CO2 cost (USD/MWh e-kerosene)": "{:.2f}",
                "Total production cost (USD/MWh e-kerosene)": "{:,.2f}",
            }).hide(axis="index")
        )

        total_prod = df_grouped["Production (TWh)"].sum()
        weighted_avg_cost = (
            (df_grouped["Total production cost (USD/MWh e-kerosene)"] * df_grouped["Production (TWh)"]).sum()
            / total_prod
        )
        print(f"Weighted average production cost: {weighted_avg_cost:.2f} USD/MWh")


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
                     "Geothermal", "Solar, tide, wave, fuel cell", "Tide and wave"], inplace=True)

    # Filter the DataFrame to only include relevant energy sources
    data = data.loc[["Nuclear", "Fossil fuels",
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
    label_to_macro = data_filtered.set_index('tech_label')['macro_category'].to_dict()
    label_to_category = data_filtered.set_index('tech_label')['main_category'].to_dict()

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
        key=lambda lbl: (macro_order_map.get(label_to_macro.get(lbl, 'Other'), 999), all_labels.index(lbl))
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
            text=cost_type_label,
            font=dict(size=16)  # Titolo del grafico
        ),
        xaxis=dict(
            title=dict(text="Years (-)", font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text=f"{cost_type_label} (Billion USD)", font=dict(size=12)),
            tickfont=dict(size=2)
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

def plot_float_bar_lcoe_dispatch_ranges(table_df, key, nice_names):
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

    for region in table_df.index:
        table_lcoe_df = table_df[table_df.columns[table_df.columns.get_level_values(0).str.contains('lcoe')]][carrier_list]
        table_lcoe_df_region = table_lcoe_df.loc[region, :]

        lcoe_tech_list = table_lcoe_df_region.xs('max', level=1).index

        fig, ax = plt.subplots(figsize=(15, 8))

        for i, (start, end) in enumerate(zip(
            table_lcoe_df_region.xs('min', level=1).values,
            table_lcoe_df_region.xs('max', level=1).values
        )):
            str_attach = any(np.abs([start, end]) > 1e-3)
            width = end - start
            ax.broken_barh([(start, width)], (i - 0.4, 0.8), hatch='///', edgecolor='white')
            start_label = f"${round(start, 2)}" if str_attach else ""
            end_label = f"${round(start + width, 2)}" if str_attach else ""
            ax.text(start - .7, i, start_label, va='center', ha='right', fontsize=12)
            ax.text(start + width + .7, i, end_label, va='center', ha='left', fontsize=12)

        # Labels and formatting
        raw_labels = [label.replace(" lcoe", "").replace(" (USD/MWh)", "") for label in lcoe_tech_list]
        clean_labels = [nice_names.get(lbl, lbl) for lbl in raw_labels]
        
        ax.set_yticks(range(len(lcoe_tech_list)))
        ax.set_yticklabels(clean_labels, fontsize=12)
        ax.set_xlabel("\nLevelized Cost of Energy (USD/MWh)", fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"\n{region} - {year_str}", fontsize=14)
        ax.grid(linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        plt.show()