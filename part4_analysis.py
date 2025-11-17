"""
Part 4 Extensions: Priority-based evacuation and advanced analysis.

This module implements HiMCM Part 4 requirements:
- Priority-based room assignment for disaster scenarios
- Comparison visualizations (standard vs priority mode)
- Technology recommendations
- Scalability analysis
"""

from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from building import BuildingGraph
from simulation import EvacuationSimulation
from visualization import RESPONDER_COLORS, ACTIVITY_COLORS


def compare_priority_modes(
    building: BuildingGraph,
    num_responders: int = 3,
    walking_speed: float = 1.5,
    visibility: float = 1.0,
    output_dir: str = '/mnt/user-data/outputs'
) -> Tuple[Dict, Dict]:
    """
    Compare standard vs priority-based evacuation modes.

    Runs two simulations side-by-side:
    1. Standard mode (use_priority=False): Nearest-neighbor assignment
    2. Priority mode (use_priority=True): High-priority rooms assigned first

    Args:
        building: BuildingGraph with varied room priorities
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor
        output_dir: Directory to save outputs

    Returns:
        Tuple of (standard_results, priority_results) where each is a dict with:
            - total_time: Total evacuation time
            - timeline: Responder timeline
            - priority_5_time: Time to clear all priority-5 rooms
            - priority_4_time: Time to clear all priority-4+ rooms
    """
    # Run standard mode
    sim_standard = EvacuationSimulation(building, num_responders)
    sim_standard.run(walking_speed=walking_speed, visibility=visibility, use_priority=False)

    # Run priority mode
    sim_priority = EvacuationSimulation(building, num_responders)
    sim_priority.run(walking_speed=walking_speed, visibility=visibility, use_priority=True)

    # Analyze results
    standard_results = analyze_priority_clearance(building, sim_standard.get_timeline())
    standard_results['total_time'] = sim_standard.get_total_time()
    standard_results['simulation'] = sim_standard

    priority_results = analyze_priority_clearance(building, sim_priority.get_timeline())
    priority_results['total_time'] = sim_priority.get_total_time()
    priority_results['simulation'] = sim_priority

    return standard_results, priority_results


def analyze_priority_clearance(
    building: BuildingGraph,
    timeline: Dict[int, List[Tuple[float, float, str, str]]]
) -> Dict:
    """
    Analyze when high-priority rooms are cleared.

    Args:
        building: BuildingGraph object
        timeline: Timeline from simulation

    Returns:
        Dict with clearance times for each priority level
    """
    results = {
        'priority_5_time': 0,  # Time to clear ALL P5 rooms (last one)
        'priority_4_time': 0,
        'priority_3_time': 0,
        'all_critical_time': 0,  # Time to clear all priority 4+ rooms
        'first_critical_time': 0,  # Time to FIRST critical room (P4+)
        'avg_p5_time': 0,  # Average time to clear P5 rooms
        'avg_critical_time': 0,  # Average time to clear P4+ rooms
    }

    # Track when each room is swept
    room_sweep_times = {}

    for resp_id, activities in timeline.items():
        for start_time, end_time, activity_type, location in activities:
            if activity_type == 'sweep':
                room_sweep_times[location] = end_time

    # Find max/min/avg time for each priority level
    priority_5_rooms = [rid for rid, room in building.rooms.items() if room.priority == 5]
    priority_4_rooms = [rid for rid, room in building.rooms.items() if room.priority >= 4]
    priority_3_rooms = [rid for rid, room in building.rooms.items() if room.priority >= 3]

    if priority_5_rooms:
        p5_times = [room_sweep_times.get(rid, 0) for rid in priority_5_rooms]
        results['priority_5_time'] = max(p5_times)  # Last P5 room
        results['avg_p5_time'] = sum(p5_times) / len(p5_times)  # Average P5

    if priority_4_rooms:
        p4_times = [room_sweep_times.get(rid, 0) for rid in priority_4_rooms]
        results['all_critical_time'] = max(p4_times)  # Last P4+ room
        results['first_critical_time'] = min(p4_times)  # First P4+ room
        results['avg_critical_time'] = sum(p4_times) / len(p4_times)  # Average P4+

    if priority_3_rooms:
        results['priority_3_time'] = max(
            room_sweep_times.get(rid, 0) for rid in priority_3_rooms
        )

    results['room_sweep_times'] = room_sweep_times

    return results


def plot_priority_comparison(
    building: BuildingGraph,
    standard_results: Dict,
    priority_results: Dict,
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Create side-by-side Gantt chart comparing standard vs priority modes.

    Args:
        building: BuildingGraph object
        standard_results: Results from standard mode
        priority_results: Results from priority mode
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Get timelines
    timeline_std = standard_results['simulation'].get_timeline()
    timeline_pri = priority_results['simulation'].get_timeline()

    # Plot standard mode
    plot_priority_timeline(ax1, building, timeline_std, "Standard Mode (Nearest-Neighbor)")

    # Plot priority mode
    plot_priority_timeline(ax2, building, timeline_pri, "Priority Mode (High-Risk First)")

    # Add comparison metrics
    fig.suptitle(f'Priority Mode Comparison - {scenario_name}', fontsize=16, fontweight='bold')

    # Add text box with key metrics
    # Calculate improvements
    first_crit_improve = ((standard_results['first_critical_time'] - priority_results['first_critical_time']) / standard_results['first_critical_time'] * 100) if standard_results['first_critical_time'] > 0 else 0
    avg_p5_improve = ((standard_results['avg_p5_time'] - priority_results['avg_p5_time']) / standard_results['avg_p5_time'] * 100) if standard_results['avg_p5_time'] > 0 else 0
    avg_crit_improve = ((standard_results['avg_critical_time'] - priority_results['avg_critical_time']) / standard_results['avg_critical_time'] * 100) if standard_results['avg_critical_time'] > 0 else 0

    metrics_text = f"""
    STANDARD MODE:
    First Critical Room: {standard_results['first_critical_time']:.1f}s
    Avg Critical (P5): {standard_results['avg_p5_time']:.1f}s
    Avg High-Risk (P4+): {standard_results['avg_critical_time']:.1f}s
    Total Time: {standard_results['total_time']:.1f}s

    PRIORITY MODE:
    First Critical Room: {priority_results['first_critical_time']:.1f}s
    Avg Critical (P5): {priority_results['avg_p5_time']:.1f}s
    Avg High-Risk (P4+): {priority_results['avg_critical_time']:.1f}s
    Total Time: {priority_results['total_time']:.1f}s

    IMPROVEMENTS:
    First Critical: {first_crit_improve:+.1f}%
    Avg P5 Time: {avg_p5_improve:+.1f}%
    Avg P4+ Time: {avg_crit_improve:+.1f}%
    """

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    plt.savefig(f'{output_dir}/priority_comparison_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_priority_timeline(ax, building: BuildingGraph, timeline: Dict, title: str):
    """
    Plot a single priority-aware timeline (helper for side-by-side comparison).

    Args:
        ax: Matplotlib axis
        building: BuildingGraph object
        timeline: Timeline dict
        title: Title for this subplot
    """
    responder_ids = sorted(timeline.keys())
    y_positions = {resp_id: i for i, resp_id in enumerate(responder_ids)}

    # Define priority colors (warmer colors = higher priority)
    priority_colors = {
        5: '#d32f2f',  # Red (critical)
        4: '#f57c00',  # Orange (high)
        3: '#fbc02d',  # Yellow (medium)
        2: '#7cb342',  # Green (low)
        1: '#0288d1',  # Blue (very low)
    }

    max_time = 0

    for resp_id, activities in timeline.items():
        y_pos = y_positions[resp_id]

        for start_time, end_time, activity_type, location in activities:
            duration = end_time - start_time
            max_time = max(max_time, end_time)

            if activity_type == 'sweep':
                # Color by room priority
                room = building.get_room(location)
                if room:
                    color = priority_colors.get(room.priority, '#95a5a6')
                    edge_color = 'black'
                    edge_width = 1.5
                else:
                    color = ACTIVITY_COLORS.get('sweep', '#27ae60')
                    edge_color = 'black'
                    edge_width = 0.5
            else:
                # Travel is blue
                color = ACTIVITY_COLORS.get('travel', '#3498db')
                edge_color = 'black'
                edge_width = 0.5

            # Draw bar
            ax.barh(y_pos, duration, left=start_time, height=0.6,
                   color=color, alpha=0.8, edgecolor=edge_color, linewidth=edge_width)

            # Add label if bar is wide enough
            if duration > max_time * 0.02:
                label_text = location.split(' -> ')[-1] if activity_type == 'travel' else location
                label_text = label_text.replace('Kindergarten', 'K').replace('Daycare', 'DC').replace('Classroom', 'C').replace('Office', 'O').replace('Storage', 'S').replace('Lab', 'L')
                ax.text(start_time + duration/2, y_pos, label_text,
                       fontsize=8, ha='center', va='center', fontweight='bold')

    # Formatting
    ax.set_yticks(range(len(responder_ids)))
    ax.set_yticklabels([f'Resp {i}' for i in responder_ids])
    ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max_time * 1.05)

    # Add priority legend
    legend_elements = [
        mpatches.Patch(color=priority_colors[5], label='Priority 5 (Critical)'),
        mpatches.Patch(color=priority_colors[4], label='Priority 4 (High)'),
        mpatches.Patch(color=priority_colors[3], label='Priority 3 (Medium)'),
        mpatches.Patch(color=priority_colors[2], label='Priority 2 (Low)'),
        mpatches.Patch(color=ACTIVITY_COLORS['travel'], label='Travel'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def plot_priority_time_savings(
    standard_results: Dict,
    priority_results: Dict,
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Create bar chart showing time savings for high-priority rooms.

    Args:
        standard_results: Results from standard mode
        priority_results: Results from priority mode
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['First\nCritical Room', 'Avg Priority 5\n(Critical)', 'Avg Priority 4+\n(High-Risk)', 'Total\nEvacuation']
    standard_times = [
        standard_results['first_critical_time'],
        standard_results['avg_p5_time'],
        standard_results['avg_critical_time'],
        standard_results['total_time']
    ]
    priority_times = [
        priority_results['first_critical_time'],
        priority_results['avg_p5_time'],
        priority_results['avg_critical_time'],
        priority_results['total_time']
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, standard_times, width, label='Standard Mode',
                   color='#95a5a6', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, priority_times, width, label='Priority Mode',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}s',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add improvement percentages
    for i, (std_time, pri_time) in enumerate(zip(standard_times, priority_times)):
        if std_time > 0:
            improvement = (std_time - pri_time) / std_time * 100
            color = 'green' if improvement > 0 else 'red'
            symbol = '▼' if improvement > 0 else '▲'
            ax.text(i, max(std_time, pri_time) * 1.1,
                   f'{symbol} {abs(improvement):.1f}%',
                   ha='center', fontsize=11, fontweight='bold', color=color)

    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Priority Mode Impact on Critical Room Clearance - {scenario_name}',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/priority_time_savings_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
