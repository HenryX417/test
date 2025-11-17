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
    visibility: float = 1.0,
    use_priority: bool = False
) -> Dict[int, List[str]]:
    """
    Assign rooms to responders to balance workload using greedy algorithm.

    Algorithm:
    1. Distribute responders across exits
    2. While unassigned rooms exist:
       a. Select responder with minimum current workload
       b. Find nearest unassigned room to responder's current position
       c. Assign room, update workload and position

    Priority Mode (use_priority=True, for Part 4 extensions):
    - High-priority rooms are considered first in assignment
    - Still uses workload balancing, but prioritizes critical rooms

    Args:
        building: BuildingGraph object
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor (0.0 to 1.0)
        use_priority: If True, assign high-priority rooms first (default: False)

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
    if use_priority:
        # PART 4 EXTENSION: Sort rooms by priority (high to low) for disaster scenarios
        # This ensures critical rooms (daycare, labs) are assigned first
        all_rooms = [(rid, building.get_room(rid).priority)
                     for rid in building.get_all_room_ids()]
        all_rooms.sort(key=lambda x: x[1], reverse=True)
        unassigned_rooms = [rid for rid, _ in all_rooms]  # Ordered list
    else:
        # PARTS 1-3: Standard approach without priority considerations
        unassigned_rooms = list(building.get_all_room_ids())

    # Greedy assignment
    while unassigned_rooms:
        if use_priority:
            # PRIORITY MODE: Take next high-priority room and assign to closest responder
            # This ensures critical rooms are assigned first regardless of distance
            target_room = unassigned_rooms[0]  # First room in priority-sorted list

            # Find which responder can reach this room fastest
            min_responder = None
            min_cost = float('inf')

            for resp_id in responder_workloads.keys():
                current_pos = responder_positions[resp_id]
                travel_time = building.distance(current_pos, target_room, walking_speed)
                # Consider both travel time and current workload for balance
                total_cost = responder_workloads[resp_id] + travel_time

                if total_cost < min_cost:
                    min_cost = total_cost
                    min_responder = resp_id

            nearest_room = target_room
            min_distance = building.distance(responder_positions[min_responder], target_room, walking_speed)
        else:
            # STANDARD MODE: Pick responder with min workload, find nearest room
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

        # Remove from unassigned (works for both list and set)
        if isinstance(unassigned_rooms, list):
            unassigned_rooms.remove(nearest_room)
        else:
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

    # Apply TSP improvements: 2-opt followed by Or-opt for better local optima
    path = two_opt_improve(path, building, walking_speed)
    path = or_opt_improve(path, building, walking_speed)
    # Run 2-opt again in case Or-opt opened new opportunities
    path = two_opt_improve(path, building, walking_speed, max_iterations=50)

    # Calculate total time
    total_time = calculate_path_time(path, building, walking_speed)

    return (path, total_time)


def two_opt_improve(
    path: List[str],
    building: BuildingGraph,
    walking_speed: float = 1.5,
    max_iterations: int = 200
) -> List[str]:
    """
    Apply 2-opt local search improvement to path.

    The 2-opt algorithm removes two edges and reconnects the path
    in a different way to reduce total distance. This version runs
    more iterations for better convergence.

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
    best_cost = calculate_path_distance(best_path, building, walking_speed)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                # Try reversing segment [i:j+1]
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                new_cost = calculate_path_distance(new_path, building, walking_speed)

                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
                    improved = True
                    break

            if improved:
                break

    return best_path


def or_opt_improve(
    path: List[str],
    building: BuildingGraph,
    walking_speed: float = 1.5,
    max_iterations: int = 100
) -> List[str]:
    """
    Apply Or-opt local search improvement to path.

    Or-opt relocates a sequence of 1, 2, or 3 consecutive nodes to
    a different position in the tour. This can find improvements
    that 2-opt misses.

    Args:
        path: Current path (list of node IDs)
        building: BuildingGraph object
        walking_speed: Walking speed in m/s
        max_iterations: Maximum number of improvement iterations

    Returns:
        Improved path
    """
    if len(path) <= 4:
        return path

    improved = True
    iteration = 0
    best_path = path[:]
    best_cost = calculate_path_distance(best_path, building, walking_speed)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Try relocating sequences of length 1, 2, and 3
        for seq_len in [1, 2, 3]:
            if len(best_path) < seq_len + 3:  # Need enough nodes
                continue

            # Try removing each sequence and inserting elsewhere
            for i in range(1, len(best_path) - seq_len):
                sequence = best_path[i:i+seq_len]

                # Remove the sequence
                path_without = best_path[:i] + best_path[i+seq_len:]

                # Try inserting at each position
                for j in range(1, len(path_without)):
                    if j == i or j == i-1:  # Skip original position
                        continue

                    # Insert sequence at position j
                    new_path = path_without[:j] + sequence + path_without[j:]
                    new_cost = calculate_path_distance(new_path, building, walking_speed)

                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

    return best_path


def calculate_path_distance(
    path: List[str],
    building: BuildingGraph,
    walking_speed: float = 1.5
) -> float:
    """
    Calculate total travel distance for a path (not including sweep times).

    Args:
        path: List of node IDs in order
        building: BuildingGraph object
        walking_speed: Walking speed in m/s

    Returns:
        Total travel time in seconds
    """
    if len(path) <= 1:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        total += building.distance(path[i], path[i+1], walking_speed)

    return total


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


def add_redundancy_checks(
    assignments: Dict[int, List[str]],
    building: BuildingGraph,
    num_responders: int,
    redundancy_level: int = 1
) -> Dict[int, List[str]]:
    """
    Add redundancy by assigning critical rooms to multiple responders for double-checking.

    PART 4 EXTENSION: Safety-first approach with redundant room sweeps.

    This ensures high-priority rooms (hospitals, daycares, labs) are checked
    multiple times by different responders to minimize missed occupants.

    Args:
        assignments: Current room assignments
        building: BuildingGraph object
        num_responders: Number of responders
        redundancy_level: Number of additional checks for high-priority rooms (default: 1)

    Returns:
        Updated assignments with redundancy checks added
    """
    # Identify high-priority rooms (priority >= 4)
    critical_rooms = [rid for rid, room in building.rooms.items() if room.priority >= 4]

    # For each critical room, assign it to additional responders
    for room_id in critical_rooms:
        # Find which responder currently has this room
        primary_responder = None
        for resp_id, rooms in assignments.items():
            if room_id in rooms:
                primary_responder = resp_id
                break

        if primary_responder is None:
            continue  # Room not assigned yet

        # Assign to redundancy_level additional responders (different from primary)
        available_responders = [i for i in range(num_responders) if i != primary_responder]

        for i in range(min(redundancy_level, len(available_responders))):
            backup_responder = available_responders[i]
            if room_id not in assignments[backup_responder]:
                assignments[backup_responder].append(room_id)

    return assignments


def assign_rooms_with_redundancy(
    building: BuildingGraph,
    num_responders: int,
    walking_speed: float = 1.5,
    visibility: float = 1.0,
    use_priority: bool = False,
    redundancy_level: int = 1
) -> Dict[int, List[str]]:
    """
    Assign rooms with safety redundancy for critical areas.

    PART 4 EXTENSION: Combines priority-based assignment with redundancy checking.

    Args:
        building: BuildingGraph object
        num_responders: Number of responders
        walking_speed: Walking speed in m/s
        visibility: Visibility factor
        use_priority: If True, assign high-priority rooms first
        redundancy_level: Number of backup checks for critical rooms

    Returns:
        Dictionary mapping responder_id to list of assigned room IDs
    """
    # First, do standard assignment
    assignments = assign_rooms_to_responders(
        building, num_responders, walking_speed, visibility, use_priority
    )

    # Add redundancy checks for critical rooms
    assignments = add_redundancy_checks(
        assignments, building, num_responders, redundancy_level
    )

    return assignments
