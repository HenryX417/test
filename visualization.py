"""
Visualization functions for evacuation simulation results.

This module contains functions to create various plots and charts
for analyzing and presenting evacuation sweep optimization results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
from building import BuildingGraph
from simulation import EvacuationSimulation
from algorithms import naive_sequential_strategy, nearest_neighbor_only
import numpy as np


# Color schemes for visualizations
ROOM_TYPE_COLORS = {
    'office': '#3498db',      # Blue
    'classroom': '#e74c3c',   # Red
    'storage': '#95a5a6',     # Gray
    'lab': '#9b59b6',         # Purple
    'daycare': '#f39c12'      # Orange
}

RESPONDER_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
]

ACTIVITY_COLORS = {
    'travel': '#3498db',
    'sweep': '#2ecc71'
}


def create_graph_layout(building: BuildingGraph) -> Dict[str, Tuple[float, float]]:
    """
    Create a layout for the building graph using NetworkX.

    Args:
        building: BuildingGraph object

    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes
    for room_id in building.rooms.keys():
        G.add_node(room_id)
    for exit_id in building.exits:
        G.add_node(exit_id)

    # Add edges
    for edge in building.edges:
        G.add_edge(edge.start, edge.end, weight=edge.distance)

    # Use spring layout for automatic positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    return pos


def plot_floor_plan(building: BuildingGraph, scenario_name: str, output_dir: str = '/mnt/user-data/outputs'):
    """
    Create clean graph visualization of building floor plan.

    Features:
    - Nodes sized by sweep time
    - Nodes colored by type
    - Edge labels showing weights
    - Exits clearly marked (square nodes)

    Args:
        building: BuildingGraph object
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create layout
    pos = create_graph_layout(building)

    # Prepare node data
    node_colors = []
    node_sizes = []
    node_shapes = []

    all_nodes = list(building.rooms.keys()) + building.exits

    for node in all_nodes:
        room = building.get_room(node)
        if room:
            # Room node
            node_colors.append(ROOM_TYPE_COLORS.get(room.type, '#95a5a6'))
            sweep_time = room.calculate_sweep_time()
            node_sizes.append(sweep_time * 10)  # Scale for visibility
        else:
            # Exit node
            node_colors.append('#27ae60')
            node_sizes.append(500)

    # Draw edges
    for edge in building.edges:
        if edge.start in pos and edge.end in pos:
            x_vals = [pos[edge.start][0], pos[edge.end][0]]
            y_vals = [pos[edge.start][1], pos[edge.end][1]]

            # Different line styles for different edge types
            linestyle = '-'
            linewidth = 1.5
            if edge.edge_type == 'stair':
                linestyle = '--'
                linewidth = 2.5

            ax.plot(x_vals, y_vals, 'k-', alpha=0.3, linestyle=linestyle, linewidth=linewidth)

            # Add edge label (distance)
            mid_x = (x_vals[0] + x_vals[1]) / 2
            mid_y = (y_vals[0] + y_vals[1]) / 2
            ax.text(mid_x, mid_y, f'{edge.distance:.0f}m',
                   fontsize=7, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Draw nodes
    for i, node in enumerate(all_nodes):
        if node not in pos:
            continue

        x, y = pos[node]
        room = building.get_room(node)

        if room:
            # Room node (circle)
            circle = plt.Circle((x, y), radius=0.08, color=node_colors[i],
                              alpha=0.7, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, node.replace('Office', 'O').replace('Classroom', 'C').replace('Lab', 'L'),
                   fontsize=8, ha='center', va='center', fontweight='bold', zorder=3)
        else:
            # Exit node (square)
            square = mpatches.Rectangle((x - 0.08, y - 0.08), 0.16, 0.16,
                                       color=node_colors[i], alpha=0.8, zorder=2)
            ax.add_patch(square)
            ax.text(x, y, node.replace('Exit', 'E'),
                   fontsize=8, ha='center', va='center', fontweight='bold',
                   color='white', zorder=3)

    # Create legend
    legend_elements = [
        mpatches.Patch(color=color, label=room_type.capitalize())
        for room_type, color in ROOM_TYPE_COLORS.items()
    ]
    legend_elements.append(mpatches.Patch(color='#27ae60', label='Exit'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title(f'Floor Plan - {scenario_name}', fontsize=16, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/floor_plan_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_assignment(
    building: BuildingGraph,
    assignments: Dict[int, List[str]],
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Visualize room assignments to responders.

    Same layout as floor plan but rooms colored by assigned responder.

    Args:
        building: BuildingGraph object
        assignments: Dictionary mapping responder_id to room list
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create layout
    pos = create_graph_layout(building)

    # Create reverse mapping (room -> responder)
    room_to_responder = {}
    for resp_id, rooms in assignments.items():
        for room_id in rooms:
            room_to_responder[room_id] = resp_id

    all_nodes = list(building.rooms.keys()) + building.exits

    # Draw edges (lighter)
    for edge in building.edges:
        if edge.start in pos and edge.end in pos:
            x_vals = [pos[edge.start][0], pos[edge.end][0]]
            y_vals = [pos[edge.start][1], pos[edge.end][1]]
            ax.plot(x_vals, y_vals, 'k-', alpha=0.2, linewidth=1)

    # Draw nodes
    for node in all_nodes:
        if node not in pos:
            continue

        x, y = pos[node]
        room = building.get_room(node)

        if room:
            # Room node - color by responder
            resp_id = room_to_responder.get(node, -1)
            if resp_id >= 0:
                color = RESPONDER_COLORS[resp_id % len(RESPONDER_COLORS)]
            else:
                color = '#95a5a6'  # Unassigned

            circle = plt.Circle((x, y), radius=0.08, color=color, alpha=0.7, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, node.replace('Office', 'O').replace('Classroom', 'C').replace('Lab', 'L'),
                   fontsize=8, ha='center', va='center', fontweight='bold', zorder=3)
        else:
            # Exit node (square)
            square = mpatches.Rectangle((x - 0.08, y - 0.08), 0.16, 0.16,
                                       color='#27ae60', alpha=0.8, zorder=2)
            ax.add_patch(square)
            ax.text(x, y, node.replace('Exit', 'E'),
                   fontsize=8, ha='center', va='center', fontweight='bold',
                   color='white', zorder=3)

    # Create legend
    num_responders = len(assignments)
    legend_elements = [
        mpatches.Patch(color=RESPONDER_COLORS[i % len(RESPONDER_COLORS)],
                      label=f'Responder {i}')
        for i in range(num_responders)
    ]
    legend_elements.append(mpatches.Patch(color='#27ae60', label='Exit'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title(f'Room Assignments - {scenario_name}', fontsize=16, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_assignment_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_optimal_paths(
    building: BuildingGraph,
    paths: Dict[int, Tuple[List[str], float]],
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Visualize optimal paths for each responder.

    Floor plan with arrows showing paths, each responder gets different color/style.

    Args:
        building: BuildingGraph object
        paths: Dictionary mapping responder_id to (path, time)
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create layout
    pos = create_graph_layout(building)

    all_nodes = list(building.rooms.keys()) + building.exits

    # Draw edges (very light)
    for edge in building.edges:
        if edge.start in pos and edge.end in pos:
            x_vals = [pos[edge.start][0], pos[edge.end][0]]
            y_vals = [pos[edge.start][1], pos[edge.end][1]]
            ax.plot(x_vals, y_vals, 'k-', alpha=0.1, linewidth=0.5)

    # Draw base nodes
    for node in all_nodes:
        if node not in pos:
            continue

        x, y = pos[node]
        room = building.get_room(node)

        if room:
            circle = plt.Circle((x, y), radius=0.06, color='#ecf0f1', alpha=0.5, zorder=1)
            ax.add_patch(circle)
        else:
            square = mpatches.Rectangle((x - 0.06, y - 0.06), 0.12, 0.12,
                                       color='#27ae60', alpha=0.6, zorder=1)
            ax.add_patch(square)

    # Draw paths
    for resp_id, (path, _) in paths.items():
        if len(path) <= 1:
            continue

        color = RESPONDER_COLORS[resp_id % len(RESPONDER_COLORS)]

        for i in range(len(path) - 1):
            if path[i] not in pos or path[i+1] not in pos:
                continue

            x1, y1 = pos[path[i]]
            x2, y2 = pos[path[i+1]]

            # Draw arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color=color, alpha=0.7))

            # Add sequence number
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, str(i+1),
                   fontsize=7, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='circle,pad=0.2', facecolor=color, alpha=0.5))

    # Overlay node labels
    for node in all_nodes:
        if node not in pos:
            continue

        x, y = pos[node]
        label = node.replace('Office', 'O').replace('Classroom', 'C').replace('Lab', 'L').replace('Exit', 'E')
        ax.text(x, y, label, fontsize=7, ha='center', va='center', fontweight='bold', zorder=5)

    # Create legend
    legend_elements = [
        mpatches.Patch(color=RESPONDER_COLORS[i % len(RESPONDER_COLORS)],
                      label=f'Responder {i}')
        for i in range(len(paths))
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title(f'Optimal Paths - {scenario_name}', fontsize=16, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimal_paths_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_gantt_chart(
    timeline: Dict[int, List[Tuple[float, float, str, str]]],
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Create Gantt chart showing responder timelines.

    Horizontal bar chart showing:
    - Each responder on Y-axis
    - Time on X-axis
    - Bars colored by activity (travel=blue, sweep=green)

    Args:
        timeline: Dictionary mapping responder_id to list of activities
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    responder_ids = sorted(timeline.keys())
    y_positions = {resp_id: i for i, resp_id in enumerate(responder_ids)}

    max_time = 0

    for resp_id, activities in timeline.items():
        y_pos = y_positions[resp_id]

        for start_time, end_time, activity_type, location in activities:
            duration = end_time - start_time
            max_time = max(max_time, end_time)

            color = ACTIVITY_COLORS.get(activity_type, '#95a5a6')

            # Draw bar
            ax.barh(y_pos, duration, left=start_time, height=0.6,
                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

            # Add label if bar is wide enough
            if duration > max_time * 0.02:  # Only label if >2% of total time
                label_text = location.split(' -> ')[-1] if activity_type == 'travel' else location
                label_text = label_text.replace('Office', 'O').replace('Classroom', 'C')
                ax.text(start_time + duration/2, y_pos, label_text,
                       fontsize=7, ha='center', va='center', fontweight='bold')

    # Formatting
    ax.set_yticks(range(len(responder_ids)))
    ax.set_yticklabels([f'Responder {i}' for i in responder_ids])
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Responder', fontsize=12, fontweight='bold')
    ax.set_title(f'Evacuation Timeline - {scenario_name}', fontsize=16, fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(color=ACTIVITY_COLORS['travel'], label='Travel'),
        mpatches.Patch(color=ACTIVITY_COLORS['sweep'], label='Sweep')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max_time * 1.05)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/gantt_chart_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_responder_comparison(
    scenario: BuildingGraph,
    scenario_name: str,
    max_responders: int = 6,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Plot total time vs. number of responders.

    Shows diminishing returns as more responders are added.

    Args:
        scenario: BuildingGraph object
        scenario_name: Name for the scenario
        max_responders: Maximum number of responders to test
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    responder_counts = list(range(1, max_responders + 1))
    times = []

    for num_resp in responder_counts:
        sim = EvacuationSimulation(scenario, num_resp)
        sim.run()
        times.append(sim.get_total_time())

    ax.plot(responder_counts, times, 'o-', linewidth=2.5, markersize=8,
           color='#3498db', label='Total Time')

    ax.set_xlabel('Number of Responders', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Evacuation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Responder Count vs. Evacuation Time - {scenario_name}',
                fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add value labels
    for x, y in zip(responder_counts, times):
        ax.text(x, y, f'{y:.0f}s', fontsize=9, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/responder_comparison_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_sensitivity_analysis(
    scenario: BuildingGraph,
    scenario_name: str,
    num_responders: int = 3,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Plot sensitivity analysis showing effects of various parameters.

    Multiple subplots showing:
    - Total time vs. walking speed
    - Total time vs. visibility

    Args:
        scenario: BuildingGraph object
        scenario_name: Name for the scenario
        num_responders: Number of responders to use
        output_dir: Directory to save output
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Walking speed sensitivity
    speeds = np.linspace(0.5, 2.5, 10)
    speed_times = []

    for speed in speeds:
        sim = EvacuationSimulation(scenario, num_responders)
        sim.run(walking_speed=speed, visibility=1.0)
        speed_times.append(sim.get_total_time())

    axes[0].plot(speeds, speed_times, 'o-', linewidth=2.5, markersize=8, color='#e74c3c')
    axes[0].set_xlabel('Walking Speed (m/s)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Total Time (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_title('Effect of Walking Speed', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Visibility sensitivity
    visibilities = np.linspace(0.3, 1.0, 10)
    visibility_times = []

    for vis in visibilities:
        sim = EvacuationSimulation(scenario, num_responders)
        sim.run(walking_speed=1.5, visibility=vis)
        visibility_times.append(sim.get_total_time())

    axes[1].plot(visibilities, visibility_times, 'o-', linewidth=2.5, markersize=8, color='#9b59b6')
    axes[1].set_xlabel('Visibility Factor', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Total Time (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_title('Effect of Visibility', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Sensitivity Analysis - {scenario_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensitivity_analysis_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_room_properties_table(building: BuildingGraph) -> pd.DataFrame:
    """
    Generate DataFrame with all room properties.

    Args:
        building: BuildingGraph object

    Returns:
        DataFrame with room properties
    """
    data = []

    for room_id, room in building.rooms.items():
        data.append({
            'Room ID': room_id,
            'Type': room.type,
            'Size (sq ft)': room.size,
            'Occupants': room.occupant_count,
            'Occupant Type': room.occupant_type,
            'Priority': room.priority,
            'Sweep Time (s)': round(room.calculate_sweep_time(), 2)
        })

    return pd.DataFrame(data)


def generate_comparison_table(
    scenarios: Dict[str, BuildingGraph],
    num_responders_list: List[int]
) -> pd.DataFrame:
    """
    Compare algorithms and responder counts across scenarios.

    Args:
        scenarios: Dictionary mapping scenario names to BuildingGraph objects
        num_responders_list: List of responder counts to test

    Returns:
        DataFrame with comparison results
    """
    data = []

    for scenario_name, building in scenarios.items():
        for num_resp in num_responders_list:
            # Our algorithm
            sim = EvacuationSimulation(building, num_resp)
            sim.run()
            our_time = sim.get_total_time()

            # Baseline: naive sequential
            _, naive_paths = naive_sequential_strategy(building, num_resp)
            naive_time = max(time for _, time in naive_paths.values()) if naive_paths else 0

            # Baseline: nearest neighbor only
            _, nn_paths = nearest_neighbor_only(building, num_resp)
            nn_time = max(time for _, time in nn_paths.values()) if nn_paths else 0

            # Calculate improvements
            naive_improvement = ((naive_time - our_time) / naive_time * 100) if naive_time > 0 else 0
            nn_improvement = ((nn_time - our_time) / nn_time * 100) if nn_time > 0 else 0

            data.append({
                'Scenario': scenario_name,
                'Responders': num_resp,
                'Strategy': 'Optimized (Ours)',
                'Time (s)': round(our_time, 2),
                'Improvement vs Naive (%)': round(naive_improvement, 2)
            })

            data.append({
                'Scenario': scenario_name,
                'Responders': num_resp,
                'Strategy': 'Nearest Neighbor',
                'Time (s)': round(nn_time, 2),
                'Improvement vs Naive (%)': round((naive_time - nn_time) / naive_time * 100 if naive_time > 0 else 0, 2)
            })

            data.append({
                'Scenario': scenario_name,
                'Responders': num_resp,
                'Strategy': 'Naive Sequential',
                'Time (s)': round(naive_time, 2),
                'Improvement vs Naive (%)': 0.0
            })

    return pd.DataFrame(data)


def generate_edge_weights_table(building: BuildingGraph) -> pd.DataFrame:
    """
    Generate table of key edges with their weights.

    Args:
        building: BuildingGraph object

    Returns:
        DataFrame with edge information
    """
    data = []

    for edge in building.edges:
        data.append({
            'Start': edge.start,
            'End': edge.end,
            'Distance (m)': edge.distance,
            'Type': edge.edge_type,
            'Travel Time @ 1.5 m/s (s)': round(edge.calculate_travel_time(1.5), 2)
        })

    df = pd.DataFrame(data)
    # Remove duplicates (since edges are bidirectional)
    df = df.drop_duplicates(subset=['Distance (m)', 'Type'])

    return df.head(20)  # Return top 20 edges
