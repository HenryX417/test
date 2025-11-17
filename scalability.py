"""
Scalability analysis for evacuation sweep optimization.

This module tests algorithm performance on varying building sizes
to demonstrate scalability for Part 4 extensions.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from building import BuildingGraph, Room, Edge
from simulation import EvacuationSimulation


def generate_synthetic_building(num_rooms: int, num_exits: int = 2) -> BuildingGraph:
    """
    Generate a synthetic building for scalability testing.

    Creates a grid-like floor plan with:
    - num_rooms total rooms
    - num_exits exit points (distributed at edges)
    - Realistic room sizes and connections

    Args:
        num_rooms: Number of rooms to generate
        num_exits: Number of exit points (default: 2)

    Returns:
        BuildingGraph with synthetic floor plan
    """
    building = BuildingGraph()

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_rooms)))

    # Create rooms in grid pattern
    room_types = ['Office', 'Classroom', 'Lab', 'Storage']
    room_id = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if room_id >= num_rooms:
                break

            room_type = room_types[room_id % len(room_types)]
            area = np.random.uniform(400, 1000)  # Random area 400-1000 sq ft
            occupancy = int(np.random.uniform(5, 30))

            room = Room(
                f'{room_type}{room_id + 1}',
                room_type.lower(),
                area,
                occupancy,
                'adults',
                priority=np.random.randint(1, 4)  # Random priority 1-3
            )
            building.add_room(room)
            room_id += 1

    # Create exits at grid edges
    exit_positions = [
        (0, 0),  # Top-left
        (grid_size - 1, grid_size - 1),  # Bottom-right
        (0, grid_size - 1),  # Top-right
        (grid_size - 1, 0),  # Bottom-left
    ]

    for i in range(min(num_exits, len(exit_positions))):
        x, y = exit_positions[i]
        building.add_exit(f'Exit{i + 1}')

    # Connect rooms in grid pattern
    room_grid = {}
    room_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if room_id >= num_rooms:
                break
            room_grid[(i, j)] = f'{room_types[room_id % len(room_types)]}{room_id + 1}'
            room_id += 1

    # Add edges between adjacent rooms (horizontal and vertical)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) not in room_grid:
                continue

            current = room_grid[(i, j)]

            # Connect to right neighbor
            if (i, j + 1) in room_grid:
                neighbor = room_grid[(i, j + 1)]
                distance = np.random.uniform(15, 30)  # 15-30 meters
                building.add_edge(Edge(current, neighbor, distance, 'hallway'))

            # Connect to bottom neighbor
            if (i + 1, j) in room_grid:
                neighbor = room_grid[(i + 1, j)]
                distance = np.random.uniform(15, 30)
                building.add_edge(Edge(current, neighbor, distance, 'hallway'))

    # Connect exits to multiple nearby rooms to ensure connectivity
    all_room_ids = list(building.rooms.keys())

    for i, exit_id in enumerate(building.exits):
        # Connect each exit to at least 2-3 rooms
        rooms_to_connect = min(3, len(all_room_ids))

        for j in range(rooms_to_connect):
            # Distribute connections across different rooms
            room_idx = (i * rooms_to_connect + j) % len(all_room_ids)
            room_id = all_room_ids[room_idx]
            distance = np.random.uniform(5, 20)
            building.add_edge(Edge(exit_id, room_id, distance, 'hallway'))

    return building


def run_scalability_benchmark(
    room_counts: List[int] = [6, 10, 20, 30, 50],
    num_responders: int = 3,
    walking_speed: float = 1.5,
    visibility: float = 1.0
) -> Dict[int, Dict]:
    """
    Benchmark algorithm performance across different building sizes.

    Args:
        room_counts: List of room counts to test
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor

    Returns:
        Dictionary mapping room_count to performance metrics:
            - runtime: Execution time in seconds
            - total_time: Evacuation time in seconds
            - coverage: Percentage of rooms assigned
    """
    results = {}

    for num_rooms in room_counts:
        print(f'\nBenchmarking {num_rooms} rooms...')

        # Generate synthetic building
        building = generate_synthetic_building(num_rooms, num_exits=2)

        # Time the simulation
        start_time = time.time()
        sim = EvacuationSimulation(building, num_responders)
        sim.run(walking_speed=walking_speed, visibility=visibility)
        runtime = time.time() - start_time

        # Collect metrics
        metrics = sim.get_metrics()
        results[num_rooms] = {
            'runtime': runtime,
            'total_time': sim.get_total_time(),
            'coverage': metrics['coverage'],
            'avg_responder_time': metrics['avg_responder_time'],
            'workload_std_dev': metrics['workload_std_dev']
        }

        print(f'  ✓ Runtime: {runtime:.3f}s')
        print(f'  ✓ Evacuation time: {sim.get_total_time():.1f}s')
        print(f'  ✓ Coverage: {metrics["coverage"] * 100:.1f}%')

    return results


def plot_scalability_results(
    results: Dict[int, Dict],
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Create scalability visualization showing algorithm performance vs building size.

    Args:
        results: Benchmark results from run_scalability_benchmark
        output_dir: Directory to save output
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    room_counts = sorted(results.keys())
    runtimes = [results[n]['runtime'] for n in room_counts]
    evacuation_times = [results[n]['total_time'] for n in room_counts]
    coverages = [results[n]['coverage'] * 100 for n in room_counts]
    workload_stds = [results[n]['workload_std_dev'] for n in room_counts]

    # 1. Runtime vs Building Size
    axes[0, 0].plot(room_counts, runtimes, 'o-', linewidth=2.5, markersize=8, color='#3498db')
    axes[0, 0].set_xlabel('Number of Rooms', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Algorithm Runtime (seconds)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Computational Performance', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Add complexity annotation
    if len(room_counts) >= 3:
        # Fit polynomial to show growth rate
        z = np.polyfit(room_counts, runtimes, 2)
        p = np.poly1d(z)
        axes[0, 0].plot(room_counts, p(room_counts), '--', alpha=0.5, color='red',
                       label=f'Trend (approx O(n²))')
        axes[0, 0].legend(fontsize=9)

    # 2. Evacuation Time vs Building Size
    axes[0, 1].plot(room_counts, evacuation_times, 'o-', linewidth=2.5, markersize=8, color='#e74c3c')
    axes[0, 1].set_xlabel('Number of Rooms', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Total Evacuation Time (seconds)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Evacuation Efficiency', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # 3. Coverage vs Building Size
    axes[1, 0].plot(room_counts, coverages, 'o-', linewidth=2.5, markersize=8, color='#27ae60')
    axes[1, 0].set_xlabel('Number of Rooms', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Room Coverage (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Assignment Coverage', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Coverage')
    axes[1, 0].legend(fontsize=9)

    # 4. Workload Balance vs Building Size
    axes[1, 1].plot(room_counts, workload_stds, 'o-', linewidth=2.5, markersize=8, color='#9b59b6')
    axes[1, 1].set_xlabel('Number of Rooms', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Workload Std Dev (seconds)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Load Balancing Quality', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle('Scalability Analysis: Algorithm Performance vs Building Size',
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{output_dir}/scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Scalability analysis saved to {output_dir}/scalability_analysis.png')


def plot_responder_scaling(
    building: BuildingGraph,
    responder_counts: List[int] = [1, 2, 3, 4, 5, 6],
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Analyze how performance scales with number of responders.

    Args:
        building: BuildingGraph to test
        responder_counts: List of responder counts to test
        output_dir: Directory to save output
    """
    evacuation_times = []
    runtimes = []

    for num_resp in responder_counts:
        start_time = time.time()
        sim = EvacuationSimulation(building, num_resp)
        sim.run()
        runtime = time.time() - start_time

        evacuation_times.append(sim.get_total_time())
        runtimes.append(runtime)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Evacuation time vs responders
    axes[0].plot(responder_counts, evacuation_times, 'o-', linewidth=2.5,
                markersize=8, color='#e74c3c')
    axes[0].set_xlabel('Number of Responders', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Total Evacuation Time (s)', fontsize=11, fontweight='bold')
    axes[0].set_title('Parallelization Efficiency', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Runtime vs responders
    axes[1].plot(responder_counts, runtimes, 'o-', linewidth=2.5,
                markersize=8, color='#3498db')
    axes[1].set_xlabel('Number of Responders', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Algorithm Runtime (s)', fontsize=11, fontweight='bold')
    axes[1].set_title('Computational Cost', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    fig.suptitle('Responder Scaling Analysis', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/responder_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Responder scaling analysis saved to {output_dir}/responder_scaling.png')
