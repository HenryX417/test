"""
Emergency Response Modeling Module.

This module provides functionality for modeling emergency scenarios
where certain areas are impassable, visibility is reduced, and
environmental hazards affect evacuation routing.

Part 4 Extension: Realistic Emergency Constraints
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple, Set
from copy import deepcopy

from building import BuildingGraph, Room, Edge
from simulation import EvacuationSimulation
from algorithms import assign_rooms_to_responders, find_optimal_path


def filter_blocked_areas(building: BuildingGraph) -> BuildingGraph:
    """
    Create modified building graph removing impassable areas for emergency routing.

    This function creates a filtered copy of the building that excludes:
    - Rooms marked as impassable (e.g., rooms with active fire)
    - Edges marked as blocked (e.g., debris-blocked stairwells)

    These rooms still exist in the original building (for tracking purposes)
    but cannot be traversed during emergency evacuation.

    Args:
        building: Original BuildingGraph with emergency metadata

    Returns:
        Modified BuildingGraph suitable for emergency routing
    """
    # Create a copy to modify
    emergency_building = BuildingGraph()

    # Copy exits
    for exit_id in building.exits:
        emergency_building.add_exit(exit_id)

    # Copy all node positions
    if hasattr(building, 'node_positions'):
        emergency_building.node_positions = deepcopy(building.node_positions)

    # Copy features
    if hasattr(building, 'features'):
        emergency_building.features = deepcopy(building.features)

    # Add passable rooms only
    passable_rooms = []
    blocked_rooms = []

    for room_id, room in building.rooms.items():
        is_passable = room.metadata.get('passable', True)

        if is_passable:
            # Room is passable - add to emergency graph
            emergency_building.add_room(room)
            passable_rooms.append(room_id)
        else:
            # Room is blocked - track but don't add
            blocked_rooms.append(room_id)

    # Add non-blocked edges only
    for edge in building.edges:
        is_blocked = edge.metadata.get('blocked', False)

        # Also check if edge connects to blocked room
        connects_to_blocked = (edge.node1 in blocked_rooms or edge.node2 in blocked_rooms)

        if not is_blocked and not connects_to_blocked:
            # Edge is usable - add to emergency graph
            emergency_building.add_edge(edge)

    print(f"\n🚨 Emergency Graph Filtering:")
    print(f"   Original: {len(building.rooms)} rooms, {len(building.edges)} edges")
    print(f"   Blocked rooms: {len(blocked_rooms)} → {blocked_rooms}")
    print(f"   Passable rooms: {len(passable_rooms)}")
    print(f"   Emergency graph: {len(emergency_building.rooms)} rooms, {len(emergency_building.edges)} edges")

    return emergency_building


def get_environmental_factors(building: BuildingGraph, room_id: str) -> Tuple[float, float]:
    """
    Get environmental factors for a specific room.

    Args:
        building: BuildingGraph with environmental metadata
        room_id: Room to check

    Returns:
        Tuple of (visibility_factor, speed_factor)
        - visibility_factor: 0.0 to 1.0 (affects sweep times)
        - speed_factor: 0.0 to 1.0 (affects walking speed)
    """
    if room_id not in building.rooms:
        return (1.0, 1.0)  # Exit or unknown node

    room = building.rooms[room_id]

    # Get visibility factor from room metadata
    visibility = room.metadata.get('visibility_factor', 1.0)

    # Speed factor based on smoke level
    smoke_level = room.metadata.get('smoke_level', 'none')
    speed_factor_map = {
        'none': 1.0,
        'low': 0.95,
        'medium': 0.8,
        'high': 0.7,
        'extreme': 0.5
    }
    speed_factor = speed_factor_map.get(smoke_level, 1.0)

    return (visibility, speed_factor)


def apply_environmental_factors(
    building: BuildingGraph,
    base_walking_speed: float,
    base_visibility: float
) -> Tuple[float, float]:
    """
    Calculate effective walking speed and visibility for emergency conditions.

    This averages environmental factors across all passable rooms.

    Args:
        building: BuildingGraph with environmental metadata
        base_walking_speed: Normal walking speed (m/s)
        base_visibility: Normal visibility factor

    Returns:
        Tuple of (effective_walking_speed, effective_visibility)
    """
    all_visibilities = []
    all_speed_factors = []

    for room_id in building.rooms.keys():
        vis, speed = get_environmental_factors(building, room_id)
        all_visibilities.append(vis)
        all_speed_factors.append(speed)

    avg_visibility = np.mean(all_visibilities) if all_visibilities else base_visibility
    avg_speed_factor = np.mean(all_speed_factors) if all_speed_factors else 1.0

    effective_walking_speed = base_walking_speed * avg_speed_factor
    effective_visibility = min(base_visibility, avg_visibility)

    return (effective_walking_speed, effective_visibility)


def run_emergency_comparison(
    building: BuildingGraph,
    num_responders: int = 3,
    base_walking_speed: float = 1.5,
    base_visibility: float = 1.0
) -> Dict:
    """
    Compare evacuation performance under normal vs emergency conditions.

    This runs two simulations:
    1. NORMAL: All areas accessible, standard conditions
    2. EMERGENCY: Blocked areas filtered, environmental hazards applied

    Args:
        building: BuildingGraph with emergency metadata
        num_responders: Number of responders
        base_walking_speed: Walking speed under normal conditions (m/s)
        base_visibility: Visibility under normal conditions

    Returns:
        Dictionary with comparison results:
        - normal_time: Total evacuation time (normal)
        - emergency_time: Total evacuation time (emergency)
        - time_penalty: Percentage increase due to emergency
        - blocked_rooms: List of impassable rooms
        - accessible_rooms: List of rooms that can be swept
        - environmental_impact: Summary of hazards
    """
    print("\n" + "=" * 70)
    print("EMERGENCY SCENARIO COMPARISON")
    print("=" * 70)

    # ========================================================================
    # SCENARIO 1: NORMAL CONDITIONS (baseline)
    # ========================================================================
    print("\n[1/2] Running NORMAL conditions simulation...")
    print("      (All areas accessible, no hazards)")

    # Create a "normal" version by removing emergency metadata
    normal_building = deepcopy(building)

    # Reset all rooms to passable
    for room in normal_building.rooms.values():
        room.metadata['passable'] = True
        room.metadata['visibility_factor'] = 1.0
        room.metadata['smoke_level'] = 'none'

    # Reset all edges to non-blocked
    for edge in normal_building.edges:
        edge.metadata['blocked'] = False
        edge.metadata['speed_factor'] = 1.0

    # Run normal simulation
    sim_normal = EvacuationSimulation(normal_building, num_responders)
    sim_normal.run(
        walking_speed=base_walking_speed,
        visibility=base_visibility,
        use_priority=True  # Always use priority mode for emergencies
    )
    normal_time = sim_normal.get_total_time()
    normal_assignments = sim_normal.assignments
    normal_paths = sim_normal.paths

    print(f"      ✅ Normal evacuation time: {normal_time:.1f}s")
    print(f"      All {len(normal_building.rooms)} rooms accessible")

    # ========================================================================
    # SCENARIO 2: EMERGENCY CONDITIONS (with blocked areas and hazards)
    # ========================================================================
    print("\n[2/2] Running EMERGENCY conditions simulation...")
    print("      (Blocked areas filtered, environmental hazards applied)")

    # Filter blocked areas
    emergency_building = filter_blocked_areas(building)

    # Calculate environmental factors
    emer_walking_speed, emer_visibility = apply_environmental_factors(
        emergency_building, base_walking_speed, base_visibility
    )

    print(f"      Environmental impact:")
    print(f"        Walking speed: {base_walking_speed:.2f} → {emer_walking_speed:.2f} m/s ({emer_walking_speed/base_walking_speed*100:.0f}%)")
    print(f"        Visibility: {base_visibility:.2f} → {emer_visibility:.2f} ({emer_visibility/base_visibility*100:.0f}%)")

    # Run emergency simulation
    sim_emergency = EvacuationSimulation(emergency_building, num_responders)
    sim_emergency.run(
        walking_speed=emer_walking_speed,
        visibility=emer_visibility,
        use_priority=True
    )
    emergency_time = sim_emergency.get_total_time()
    emergency_assignments = sim_emergency.assignments
    emergency_paths = sim_emergency.paths

    print(f"      ✅ Emergency evacuation time: {emergency_time:.1f}s")

    # ========================================================================
    # CALCULATE IMPACT
    # ========================================================================
    time_penalty = ((emergency_time - normal_time) / normal_time * 100) if normal_time > 0 else 0

    # Identify blocked rooms
    all_rooms = set(building.rooms.keys())
    accessible_rooms = set(emergency_building.rooms.keys())
    blocked_rooms = list(all_rooms - accessible_rooms)

    # Environmental impact summary
    env_impact = {
        'walking_speed_reduction': f"{(1 - emer_walking_speed/base_walking_speed)*100:.1f}%",
        'visibility_reduction': f"{(1 - emer_visibility/base_visibility)*100:.1f}%",
        'rooms_blocked': len(blocked_rooms),
        'rooms_accessible': len(accessible_rooms)
    }

    print("\n" + "-" * 70)
    print("EMERGENCY IMPACT SUMMARY")
    print("-" * 70)
    print(f"  Normal evacuation time:     {normal_time:.1f}s")
    print(f"  Emergency evacuation time:  {emergency_time:.1f}s")
    print(f"  Time penalty:               +{time_penalty:.1f}%")
    print(f"  Blocked rooms:              {len(blocked_rooms)} → {blocked_rooms}")
    print(f"  Accessible rooms:           {len(accessible_rooms)}/{len(all_rooms)}")
    print("=" * 70)

    return {
        'normal_time': normal_time,
        'emergency_time': emergency_time,
        'time_penalty': time_penalty,
        'blocked_rooms': blocked_rooms,
        'accessible_rooms': list(accessible_rooms),
        'environmental_impact': env_impact,
        'normal_simulation': sim_normal,
        'emergency_simulation': sim_emergency,
        'normal_building': normal_building,
        'emergency_building': emergency_building
    }


def identify_alternate_routes(
    normal_paths: Dict[int, Tuple[List[str], float]],
    emergency_paths: Dict[int, Tuple[List[str], float]]
) -> Dict[int, Dict]:
    """
    Identify where emergency routes differ from normal routes.

    Args:
        normal_paths: Paths from normal simulation
        emergency_paths: Paths from emergency simulation

    Returns:
        Dictionary mapping responder_id to route changes:
        - original_path: Path under normal conditions
        - emergency_path: Path under emergency conditions
        - differs: Boolean indicating if paths are different
        - skipped_rooms: Rooms in original but not in emergency
        - new_rooms: Rooms in emergency but not in original
    """
    route_changes = {}

    for resp_id in normal_paths.keys():
        normal_path, normal_time = normal_paths[resp_id]
        emergency_path, emergency_time = emergency_paths.get(resp_id, ([], 0))

        normal_rooms = set(normal_path)
        emergency_rooms = set(emergency_path)

        differs = (normal_path != emergency_path)
        skipped_rooms = list(normal_rooms - emergency_rooms)
        new_rooms = list(emergency_rooms - normal_rooms)

        route_changes[resp_id] = {
            'original_path': normal_path,
            'emergency_path': emergency_path,
            'differs': differs,
            'skipped_rooms': skipped_rooms,
            'new_rooms': new_rooms,
            'time_change': emergency_time - normal_time
        }

    return route_changes


def get_unreachable_rooms(
    building: BuildingGraph,
    emergency_building: BuildingGraph,
    emergency_assignments: Dict[int, List[str]]
) -> List[str]:
    """
    Identify rooms that become unreachable due to emergency conditions.

    Args:
        building: Original building
        emergency_building: Filtered emergency building
        emergency_assignments: Room assignments in emergency

    Returns:
        List of room IDs that cannot be reached
    """
    all_rooms = set(building.rooms.keys())
    emergency_accessible = set(emergency_building.rooms.keys())

    # Rooms assigned in emergency (these are reachable)
    assigned_rooms = set()
    for rooms in emergency_assignments.values():
        assigned_rooms.update(rooms)

    # Unreachable = rooms that exist but weren't assigned
    # (either blocked or isolated by blocked paths)
    unreachable = list(all_rooms - assigned_rooms)

    return unreachable


def visualize_emergency_comparison(
    comparison_results: Dict,
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Create side-by-side comparison visualization of normal vs emergency evacuation.

    Args:
        comparison_results: Results from run_emergency_comparison()
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    from visualization import plot_evacuation_timeline

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Extract data
    sim_normal = comparison_results['normal_simulation']
    sim_emergency = comparison_results['emergency_simulation']
    building_normal = comparison_results['normal_building']
    building_emergency = comparison_results['emergency_building']

    # Left plot: NORMAL conditions
    plot_evacuation_timeline(
        ax1, building_normal, sim_normal.timeline,
        f"NORMAL CONDITIONS\nAll Areas Accessible"
    )

    # Right plot: EMERGENCY conditions
    plot_evacuation_timeline(
        ax2, building_emergency, sim_emergency.timeline,
        f"EMERGENCY CONDITIONS\nBlocked Areas Filtered"
    )

    # Overall title
    fig.suptitle(
        f'Emergency Response Comparison - {scenario_name}\n'
        f'Normal: {comparison_results["normal_time"]:.1f}s  |  '
        f'Emergency: {comparison_results["emergency_time"]:.1f}s  |  '
        f'Penalty: +{comparison_results["time_penalty"]:.1f}%',
        fontsize=16, fontweight='bold'
    )

    # Add summary metrics
    metrics_text = f"""
    NORMAL CONDITIONS:
    Total Time: {comparison_results['normal_time']:.1f}s
    Rooms Accessible: {len(comparison_results['accessible_rooms']) + len(comparison_results['blocked_rooms'])}
    All routes available

    EMERGENCY CONDITIONS:
    Total Time: {comparison_results['emergency_time']:.1f}s
    Rooms Accessible: {len(comparison_results['accessible_rooms'])}
    Blocked Rooms: {len(comparison_results['blocked_rooms'])} → {comparison_results['blocked_rooms']}
    Walking Speed: {comparison_results['environmental_impact']['walking_speed_reduction']} reduction
    Visibility: {comparison_results['environmental_impact']['visibility_reduction']} reduction

    IMPACT:
    Time Penalty: +{comparison_results['time_penalty']:.1f}%
    Rooms Unreachable: {len(comparison_results['blocked_rooms'])}
    """

    fig.text(0.5, -0.05, metrics_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f'{output_dir}/emergency_comparison_{scenario_name}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Emergency comparison visualization saved:")
    print(f"   {output_dir}/emergency_comparison_{scenario_name}.png")


def plot_emergency_floor_plan(
    building: BuildingGraph,
    emergency_building: BuildingGraph,
    scenario_name: str,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Visualize floor plan showing blocked areas and hazard zones.

    Args:
        building: Original building (with all rooms)
        emergency_building: Filtered building (passable areas only)
        scenario_name: Name for the scenario
        output_dir: Directory to save output
    """
    from visualization import plot_building_graph

    fig, ax = plt.subplots(figsize=(14, 10))

    # Get blocked and accessible rooms
    all_rooms = set(building.rooms.keys())
    accessible_rooms = set(emergency_building.rooms.keys())
    blocked_rooms = all_rooms - accessible_rooms

    # Plot with custom colors for emergency status
    node_colors = {}

    for room_id in building.rooms.keys():
        if room_id in blocked_rooms:
            # Blocked room - RED
            node_colors[room_id] = '#e74c3c'
        else:
            # Check smoke level for accessible rooms
            smoke_level = building.rooms[room_id].metadata.get('smoke_level', 'none')
            if smoke_level in ['high', 'extreme']:
                node_colors[room_id] = '#e67e22'  # Orange for high smoke
            elif smoke_level == 'medium':
                node_colors[room_id] = '#f39c12'  # Yellow for medium smoke
            else:
                node_colors[room_id] = '#2ecc71'  # Green for safe

    # Plot building
    plot_building_graph(building, ax=ax, node_colors=node_colors)

    # Add legend
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label=f'BLOCKED ({len(blocked_rooms)} rooms)'),
        mpatches.Patch(color='#e67e22', label='High Smoke (accessible)'),
        mpatches.Patch(color='#f39c12', label='Medium Smoke'),
        mpatches.Patch(color='#2ecc71', label='Safe Zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title(
        f'Emergency Floor Plan - {scenario_name}\n'
        f'Blocked Areas: {len(blocked_rooms)} | Accessible: {len(accessible_rooms)}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(f'{output_dir}/emergency_floor_plan_{scenario_name}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Emergency floor plan saved:")
    print(f"   {output_dir}/emergency_floor_plan_{scenario_name}.png")
