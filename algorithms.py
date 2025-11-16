"""
Clustering and pathfinding algorithms for evacuation sweep optimization.

This module contains algorithms for:
- Assigning rooms to responders (clustering)
- Finding optimal paths through assigned rooms
- Baseline comparison strategies
"""

from typing import Dict, List, Tuple
from building import BuildingGraph, Room
import copy


def assign_rooms_to_responders(
    building: BuildingGraph,
    num_responders: int,
    walking_speed: float = 1.5,
    visibility: float = 1.0
) -> Dict[int, List[str]]:
    """
    Assign rooms to responders to balance workload using greedy algorithm.

    Algorithm:
    1. Distribute responders across exits
    2. While unassigned rooms exist:
       a. Select responder with minimum current workload
       b. Find nearest unassigned room to responder's current position
       c. Assign room, update workload and position

    Args:
        building: BuildingGraph object
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor (0.0 to 1.0)

    Returns:
        Dictionary mapping responder_id to list of assigned room IDs
    """
    # Initialize assignments
    assignments = {i: [] for i in range(num_responders)}

    # Distribute responders across exits
    responder_positions = {}
    responder_workloads = {}

    for i in range(num_responders):
        exit_idx = i % len(building.exits)
        responder_positions[i] = building.exits[exit_idx]
        responder_workloads[i] = 0.0

    # Get all rooms to assign
    unassigned_rooms = set(building.get_all_room_ids())

    # Greedy assignment
    while unassigned_rooms:
        # Find responder with minimum workload
        min_responder = min(responder_workloads.keys(), key=lambda x: responder_workloads[x])

        # Find nearest unassigned room to this responder's current position
        current_pos = responder_positions[min_responder]
        nearest_room = None
        min_distance = float('inf')

        for room_id in unassigned_rooms:
            dist = building.distance(current_pos, room_id, walking_speed)
            if dist < min_distance:
                min_distance = dist
                nearest_room = room_id

        if nearest_room is None:
            break

        # Assign room to responder
        assignments[min_responder].append(nearest_room)
        unassigned_rooms.remove(nearest_room)

        # Update responder's position and workload
        responder_positions[min_responder] = nearest_room
        room = building.get_room(nearest_room)
        sweep_time = room.calculate_sweep_time(visibility) if room else 30.0
        responder_workloads[min_responder] += min_distance + sweep_time

    return assignments


def find_optimal_path(
    assigned_rooms: List[str],
    start_exit: str,
    building: BuildingGraph,
    exits: List[str],
    walking_speed: float = 1.5
) -> Tuple[List[str], float]:
    """
    Find near-optimal path through assigned rooms using nearest neighbor + 2-opt.

    Algorithm:
    1. Nearest Neighbor: Greedily build path from start_exit
    2. Add closest exit at end
    3. 2-opt: Iteratively swap edges to reduce total path length

    Args:
        assigned_rooms: List of room IDs assigned to this responder
        start_exit: Exit where responder starts
        building: BuildingGraph object
        exits: List of all exit IDs
        walking_speed: Walking speed in m/s

    Returns:
        Tuple of (ordered path including start and end, total time)
    """
    if not assigned_rooms:
        return ([start_exit], 0.0)

    # Nearest neighbor construction
    path = [start_exit]
    remaining = set(assigned_rooms)
    current = start_exit

    while remaining:
        # Find nearest unvisited room
        nearest = None
        min_dist = float('inf')

        for room_id in remaining:
            dist = building.distance(current, room_id, walking_speed)
            if dist < min_dist:
                min_dist = dist
                nearest = room_id

        if nearest is None:
            break

        path.append(nearest)
        remaining.remove(nearest)
        current = nearest

    # Find closest exit from last room
    best_exit = start_exit
    min_exit_dist = float('inf')

    for exit_id in exits:
        dist = building.distance(current, exit_id, walking_speed)
        if dist < min_exit_dist:
            min_exit_dist = dist
            best_exit = exit_id

    path.append(best_exit)

    # Apply 2-opt improvement
    path = two_opt_improve(path, building, walking_speed)

    # Calculate total time
    total_time = calculate_path_time(path, building, walking_speed)

    return (path, total_time)


def two_opt_improve(
    path: List[str],
    building: BuildingGraph,
    walking_speed: float = 1.5,
    max_iterations: int = 100
) -> List[str]:
    """
    Apply 2-opt local search improvement to path.

    The 2-opt algorithm removes two edges and reconnects the path
    in a different way to reduce total distance.

    Args:
        path: Current path (list of node IDs)
        building: BuildingGraph object
        walking_speed: Walking speed in m/s
        max_iterations: Maximum number of improvement iterations

    Returns:
        Improved path
    """
    if len(path) <= 3:
        return path

    improved = True
    iteration = 0
    best_path = path[:]

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                # Try reversing segment [i:j+1]
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]

                # Calculate improvement
                old_dist = (building.distance(best_path[i-1], best_path[i], walking_speed) +
                           building.distance(best_path[j], best_path[j+1], walking_speed))

                new_dist = (building.distance(new_path[i-1], new_path[i], walking_speed) +
                           building.distance(new_path[j], new_path[j+1], walking_speed))

                if new_dist < old_dist:
                    best_path = new_path
                    improved = True
                    break

            if improved:
                break

    return best_path


def calculate_path_time(
    path: List[str],
    building: BuildingGraph,
    walking_speed: float = 1.5,
    visibility: float = 1.0
) -> float:
    """
    Calculate total time for a path including travel and sweep times.

    Args:
        path: List of node IDs in order
        building: BuildingGraph object
        walking_speed: Walking speed in m/s
        visibility: Visibility factor

    Returns:
        Total time in seconds
    """
    if len(path) <= 1:
        return 0.0

    total_time = 0.0

    # Add travel times between consecutive nodes
    for i in range(len(path) - 1):
        travel_time = building.distance(path[i], path[i+1], walking_speed)
        total_time += travel_time

    # Add sweep times for rooms (not exits)
    for node in path:
        room = building.get_room(node)
        if room:
            sweep_time = room.calculate_sweep_time(visibility)
            total_time += sweep_time

    return total_time


def naive_sequential_strategy(
    building: BuildingGraph,
    num_responders: int,
    walking_speed: float = 1.5,
    visibility: float = 1.0
) -> Tuple[Dict[int, List[str]], Dict[int, Tuple[List[str], float]]]:
    """
    Baseline strategy: Responders take rooms in sequential order.

    Rooms are divided evenly among responders in the order they appear.

    Args:
        building: BuildingGraph object
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor

    Returns:
        Tuple of (assignments, paths) where paths includes time
    """
    rooms = building.get_all_room_ids()
    assignments = {i: [] for i in range(num_responders)}

    # Divide rooms sequentially
    for idx, room_id in enumerate(rooms):
        responder_id = idx % num_responders
        assignments[responder_id].append(room_id)

    # Find paths for each responder
    paths = {}
    for resp_id in range(num_responders):
        if not assignments[resp_id]:
            paths[resp_id] = ([building.exits[resp_id % len(building.exits)]], 0.0)
            continue

        start_exit = building.exits[resp_id % len(building.exits)]
        path, time_taken = find_optimal_path(
            assignments[resp_id],
            start_exit,
            building,
            building.exits,
            walking_speed
        )
        paths[resp_id] = (path, time_taken)

    return (assignments, paths)


def nearest_neighbor_only(
    building: BuildingGraph,
    num_responders: int,
    walking_speed: float = 1.5,
    visibility: float = 1.0
) -> Tuple[Dict[int, List[str]], Dict[int, Tuple[List[str], float]]]:
    """
    Baseline: Simple nearest-neighbor without 2-opt improvement.

    Uses the same clustering as the main algorithm but skips 2-opt.

    Args:
        building: BuildingGraph object
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor

    Returns:
        Tuple of (assignments, paths)
    """
    # Use same clustering
    assignments = assign_rooms_to_responders(building, num_responders, walking_speed, visibility)

    # Find paths without 2-opt
    paths = {}
    for resp_id in range(num_responders):
        if not assignments[resp_id]:
            paths[resp_id] = ([building.exits[resp_id % len(building.exits)]], 0.0)
            continue

        start_exit = building.exits[resp_id % len(building.exits)]

        # Nearest neighbor only (no 2-opt)
        path = [start_exit]
        remaining = set(assignments[resp_id])
        current = start_exit

        while remaining:
            nearest = None
            min_dist = float('inf')

            for room_id in remaining:
                dist = building.distance(current, room_id, walking_speed)
                if dist < min_dist:
                    min_dist = dist
                    nearest = room_id

            if nearest is None:
                break

            path.append(nearest)
            remaining.remove(nearest)
            current = nearest

        # Add closest exit
        best_exit = start_exit
        min_exit_dist = float('inf')

        for exit_id in building.exits:
            dist = building.distance(current, exit_id, walking_speed)
            if dist < min_exit_dist:
                min_exit_dist = dist
                best_exit = exit_id

        path.append(best_exit)

        # Calculate time
        time_taken = calculate_path_time(path, building, walking_speed, visibility)
        paths[resp_id] = (path, time_taken)

    return (assignments, paths)
