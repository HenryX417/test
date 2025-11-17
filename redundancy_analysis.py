"""
Safety Redundancy Analysis for Emergency Evacuation.

This module demonstrates and tests the double-checking redundancy
feature for critical rooms (Part 4 Extension - Safety First).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple
from building import BuildingGraph
from algorithms import assign_rooms_with_redundancy, find_optimal_path
from simulation import EvacuationSimulation


def compare_redundancy_modes(
    building: BuildingGraph,
    num_responders: int = 3,
    walking_speed: float = 1.5,
    visibility: float = 1.0
) -> Tuple[Dict, Dict]:
    """
    Compare standard mode vs redundancy mode.

    Args:
        building: BuildingGraph with critical rooms
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor

    Returns:
        Tuple of (standard_results, redundancy_results)
    """
    # Standard mode (no redundancy)
    sim_standard = EvacuationSimulation(building, num_responders)
    sim_standard.run(walking_speed=walking_speed, visibility=visibility, use_priority=True)

    # Redundancy mode
    assignments_redundant = assign_rooms_with_redundancy(
        building, num_responders, walking_speed, visibility,
        use_priority=True, redundancy_level=1
    )

    # Find paths for redundancy mode
    paths_redundant = {}
    for resp_id in range(num_responders):
        if not assignments_redundant[resp_id]:
            start_exit = building.exits[resp_id % len(building.exits)]
            paths_redundant[resp_id] = ([start_exit], 0.0)
            continue

        start_exit = building.exits[resp_id % len(building.exits)]
        path, time_taken = find_optimal_path(
            assignments_redundant[resp_id],
            start_exit,
            building,
            building.exits,
            walking_speed,
            visibility
        )
        paths_redundant[resp_id] = (path, time_taken)

    # Calculate redundancy coverage
    critical_rooms = [rid for rid, room in building.rooms.items() if room.priority >= 4]

    standard_coverage = {}
    redundant_coverage = {}

    for room_id in critical_rooms:
        # Count how many responders check this room
        std_count = sum(1 for rooms in sim_standard.assignments.values() if room_id in rooms)
        red_count = sum(1 for rooms in assignments_redundant.values() if room_id in rooms)

        standard_coverage[room_id] = std_count
        redundant_coverage[room_id] = red_count

    # Calculate total evacuation times
    std_total_time = max(paths[1] for paths in sim_standard.paths.values())
    red_total_time = max(time for _, time in paths_redundant.values())

    standard_results = {
        'assignments': sim_standard.assignments,
        'paths': sim_standard.paths,
        'coverage': standard_coverage,
        'total_time': std_total_time,
        'avg_critical_checks': np.mean(list(standard_coverage.values())),
    }

    redundancy_results = {
        'assignments': assignments_redundant,
        'paths': paths_redundant,
        'coverage': redundant_coverage,
        'total_time': red_total_time,
        'avg_critical_checks': np.mean(list(redundant_coverage.values())),
    }

    return standard_results, redundancy_results


def plot_redundancy_comparison(
    building: BuildingGraph,
    standard_results: Dict,
    redundancy_results: Dict,
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Visualize redundancy comparison showing double-checking coverage.

    Args:
        building: BuildingGraph object
        standard_results: Results from standard mode
        redundancy_results: Results from redundancy mode
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    critical_rooms = [rid for rid, room in building.rooms.items() if room.priority >= 4]

    # Left plot: Standard mode
    ax1 = axes[0]
    room_labels = [f'{rid}\n(P{building.rooms[rid].priority})' for rid in critical_rooms]
    checks_std = [standard_results['coverage'][rid] for rid in critical_rooms]

    colors_std = ['#e74c3c' if c == 1 else '#27ae60' for c in checks_std]

    bars1 = ax1.bar(range(len(critical_rooms)), checks_std, color=colors_std,
                    alpha=0.7, edgecolor='black', linewidth=2)

    ax1.set_xticks(range(len(critical_rooms)))
    ax1.set_xticklabels(room_labels, fontsize=10, fontweight='bold')
    ax1.set_ylabel('Number of Checks', fontsize=11, fontweight='bold')
    ax1.set_title('Standard Mode\n(Single Check)', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(max(checks_std), 2) + 0.5])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Minimum Safety Threshold')
    ax1.legend(fontsize=9)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, checks_std)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(val)}×',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Right plot: Redundancy mode
    ax2 = axes[1]
    checks_red = [redundancy_results['coverage'][rid] for rid in critical_rooms]

    colors_red = ['#e74c3c' if c == 1 else '#27ae60' for c in checks_red]

    bars2 = ax2.bar(range(len(critical_rooms)), checks_red, color=colors_red,
                    alpha=0.7, edgecolor='black', linewidth=2)

    ax2.set_xticks(range(len(critical_rooms)))
    ax2.set_xticklabels(room_labels, fontsize=10, fontweight='bold')
    ax2.set_ylabel('Number of Checks', fontsize=11, fontweight='bold')
    ax2.set_title('Redundancy Mode\n(Double Check)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(max(checks_red), 2) + 0.5])
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Enhanced Safety Threshold')
    ax2.legend(fontsize=9)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, checks_red)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(val)}×',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Overall title
    fig.suptitle(f'Safety Redundancy Comparison - {scenario_name}\\nCritical Room Double-Checking',
                 fontsize=16, fontweight='bold')

    # Add summary text
    summary_text = f"""
    STANDARD MODE:
    Avg Checks/Critical Room: {standard_results['avg_critical_checks']:.1f}×
    Total Evacuation Time: {standard_results['total_time']:.1f}s

    REDUNDANCY MODE:
    Avg Checks/Critical Room: {redundancy_results['avg_critical_checks']:.1f}×
    Total Evacuation Time: {redundancy_results['total_time']:.1f}s

    SAFETY IMPROVEMENT: {((redundancy_results['avg_critical_checks'] - standard_results['avg_critical_checks']) / standard_results['avg_critical_checks'] * 100):.0f}% more checks
    Time Cost: {((redundancy_results['total_time'] - standard_results['total_time']) / standard_results['total_time'] * 100):+.1f}%
    """

    fig.text(0.5, -0.05, summary_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{output_dir}/redundancy_comparison_{scenario_name}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\\n✅ Redundancy comparison saved to {output_dir}/redundancy_comparison_{scenario_name}.png')


def calculate_safety_score(coverage: Dict[str, int]) -> float:
    """
    Calculate safety score based on redundancy coverage.

    Higher scores indicate better safety (more redundant checks).

    Args:
        coverage: Dictionary mapping room_id to number of checks

    Returns:
        Safety score (0-100)
    """
    if not coverage:
        return 0.0

    # Ideal is 2 checks per critical room
    ideal_checks = 2
    actual_avg = np.mean(list(coverage.values()))

    # Score is percentage of ideal coverage achieved
    score = min(100, (actual_avg / ideal_checks) * 100)

    return score
