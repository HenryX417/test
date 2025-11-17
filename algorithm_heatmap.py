"""
Algorithm Performance Heatmap Visualization.

This module creates performance heatmaps showing how evacuation time
varies across different parameters (Part 4 visualization).
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import List
from building import BuildingGraph
from simulation import EvacuationSimulation


def generate_performance_heatmap(
    building: BuildingGraph,
    walking_speeds: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0],
    visibilities: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    num_responders: int = 3,
    output_dir: str = '/mnt/user-data/outputs',
    scenario_name: str = 'Default'
):
    """
    Generate heatmap showing evacuation time vs walking speed and visibility.

    Args:
        building: BuildingGraph to test
        walking_speeds: List of walking speeds to test (m/s)
        visibilities: List of visibility factors to test
        num_responders: Number of responders
        output_dir: Directory to save output
        scenario_name: Name for the scenario
    """
    print(f'\\nGenerating performance heatmap for {scenario_name}...')

    # Create matrix to store evacuation times
    times_matrix = np.zeros((len(visibilities), len(walking_speeds)))

    # Run simulations for each combination
    for i, visibility in enumerate(visibilities):
        for j, speed in enumerate(walking_speeds):
            sim = EvacuationSimulation(building, num_responders)
            sim.run(walking_speed=speed, visibility=visibility, use_priority=False)
            times_matrix[i, j] = sim.get_total_time()

            print(f'  Testing: speed={speed:.2f} m/s, visibility={visibility:.1f} → {times_matrix[i, j]:.1f}s')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap using matplotlib
    im = ax.imshow(times_matrix, cmap='RdYlGn_r', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Evacuation Time (seconds)', fontsize=11, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(walking_speeds)))
    ax.set_yticks(np.arange(len(visibilities)))
    ax.set_xticklabels([f'{s:.2f}' for s in walking_speeds])
    ax.set_yticklabels([f'{v:.1f}' for v in visibilities])

    # Add text annotations
    for i in range(len(visibilities)):
        for j in range(len(walking_speeds)):
            text = ax.text(j, i, f'{times_matrix[i, j]:.0f}',
                          ha='center', va='center', color='black',
                          fontsize=10, fontweight='bold')

    ax.set_xlabel('Walking Speed (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Visibility Factor', fontsize=12, fontweight='bold')
    ax.set_title(f'Algorithm Performance Heatmap - {scenario_name}\\n'
                 f'Evacuation Time vs Environmental Conditions ({num_responders} Responders)',
                 fontsize=14, fontweight='bold')

    # Add performance zones
    best_idx = np.unravel_index(np.argmin(times_matrix), times_matrix.shape)
    worst_idx = np.unravel_index(np.argmax(times_matrix), times_matrix.shape)

    ax.text(best_idx[1], best_idx[0] - 0.3, '⭐ BEST',
           fontsize=12, ha='center', va='center', fontweight='bold', color='darkgreen')
    ax.text(worst_idx[1], worst_idx[0] - 0.3, '⚠️ WORST',
           fontsize=12, ha='center', va='center', fontweight='bold', color='darkred')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmap_{scenario_name}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  ✅ Heatmap saved to {output_dir}/performance_heatmap_{scenario_name}.png')

    # Print summary
    print(f'\\n  Best performance: {np.min(times_matrix):.1f}s (speed={walking_speeds[best_idx[1]]:.2f}, vis={visibilities[best_idx[0]]:.1f})')
    print(f'  Worst performance: {np.max(times_matrix):.1f}s (speed={walking_speeds[worst_idx[1]]:.2f}, vis={visibilities[worst_idx[0]]:.1f})')
    print(f'  Performance range: {((np.max(times_matrix) - np.min(times_matrix)) / np.min(times_matrix) * 100):.1f}% variation')


def generate_responder_vs_rooms_heatmap(
    buildings: List[BuildingGraph],
    room_counts: List[int],
    responder_counts: List[int] = [1, 2, 3, 4, 5],
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Generate heatmap showing evacuation time vs number of responders and rooms.

    Args:
        buildings: List of BuildingGraph objects (different sizes)
        room_counts: List of room counts (must match buildings list)
        responder_counts: List of responder counts to test
        output_dir: Directory to save output
    """
    print('\\nGenerating responders vs rooms heatmap...')

    # Create matrix
    times_matrix = np.zeros((len(room_counts), len(responder_counts)))

    # Run simulations
    for i, (building, num_rooms) in enumerate(zip(buildings, room_counts)):
        for j, num_resp in enumerate(responder_counts):
            sim = EvacuationSimulation(building, num_resp)
            sim.run(walking_speed=1.5, visibility=1.0)
            times_matrix[i, j] = sim.get_total_time()

            print(f'  Testing: {num_rooms} rooms, {num_resp} responders → {times_matrix[i, j]:.1f}s')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(times_matrix, cmap='viridis', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Evacuation Time (seconds)', fontsize=11, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(responder_counts)))
    ax.set_yticks(np.arange(len(room_counts)))
    ax.set_xticklabels(responder_counts)
    ax.set_yticklabels([f'{n} rooms' for n in room_counts])

    # Add text annotations
    for i in range(len(room_counts)):
        for j in range(len(responder_counts)):
            text = ax.text(j, i, f'{times_matrix[i, j]:.0f}',
                          ha='center', va='center', color='white',
                          fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Responders', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Size', fontsize=12, fontweight='bold')
    ax.set_title('Scalability Heatmap: Evacuation Time vs Team Size & Building Size',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scalability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  ✅ Scalability heatmap saved to {output_dir}/scalability_heatmap.png')
