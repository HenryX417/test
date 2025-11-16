"""
Evacuation simulation engine.

This module contains the main simulation class that orchestrates
the evacuation sweep optimization process.
"""

from typing import Dict, List, Tuple
from building import BuildingGraph
from algorithms import (
    assign_rooms_to_responders,
    find_optimal_path,
    calculate_path_time
)


class EvacuationSimulation:
    """Main simulation class for evacuation sweep optimization."""

    def __init__(self, building: BuildingGraph, num_responders: int):
        """
        Initialize evacuation simulation.

        Args:
            building: BuildingGraph object
            num_responders: Number of responders
        """
        self.building = building
        self.num_responders = num_responders
        self.assignments = None
        self.paths = None
        self.results = None
        self.walking_speed = 1.5
        self.visibility = 1.0

    def run(self, walking_speed: float = 1.5, visibility: float = 1.0):
        """
        Execute complete simulation.

        Steps:
        1. Cluster rooms (assign to responders)
        2. Find paths for each responder
        3. Calculate timelines
        4. Generate metrics

        Args:
            walking_speed: Walking speed in m/s
            visibility: Visibility factor (0.0 to 1.0)
        """
        self.walking_speed = walking_speed
        self.visibility = visibility

        # Step 1: Cluster rooms
        self.assignments = assign_rooms_to_responders(
            self.building,
            self.num_responders,
            walking_speed,
            visibility
        )

        # Step 2: Find optimal paths
        self.paths = {}
        for resp_id in range(self.num_responders):
            if not self.assignments[resp_id]:
                # No rooms assigned
                start_exit = self.building.exits[resp_id % len(self.building.exits)]
                self.paths[resp_id] = ([start_exit], 0.0)
                continue

            start_exit = self.building.exits[resp_id % len(self.building.exits)]
            path, time_taken = find_optimal_path(
                self.assignments[resp_id],
                start_exit,
                self.building,
                self.building.exits,
                walking_speed
            )
            self.paths[resp_id] = (path, time_taken)

        # Step 3: Generate detailed results
        self.results = self._generate_results()

    def _generate_results(self) -> Dict:
        """
        Generate detailed results including metrics and timelines.

        Returns:
            Dictionary containing simulation results
        """
        results = {
            'total_time': self.get_total_time(),
            'responder_times': {},
            'timelines': self.get_timeline(),
            'room_assignments': self.assignments,
            'paths': self.paths
        }

        for resp_id in range(self.num_responders):
            if resp_id in self.paths:
                results['responder_times'][resp_id] = self.paths[resp_id][1]

        return results

    def get_total_time(self) -> float:
        """
        Return max completion time across all responders.

        Returns:
            Maximum completion time in seconds
        """
        if not self.paths:
            return 0.0

        max_time = 0.0
        for resp_id in range(self.num_responders):
            if resp_id in self.paths:
                time = self.paths[resp_id][1]
                max_time = max(max_time, time)

        return max_time

    def get_timeline(self) -> Dict[int, List[Tuple[float, float, str, str]]]:
        """
        Return detailed timeline for each responder.

        Returns:
            Dictionary mapping responder_id to list of activities:
            [(start_time, end_time, activity_type, location), ...]
            where activity_type is 'travel' or 'sweep'
        """
        timelines = {}

        for resp_id in range(self.num_responders):
            if resp_id not in self.paths or not self.paths[resp_id][0]:
                timelines[resp_id] = []
                continue

            path, _ = self.paths[resp_id]
            timeline = []
            current_time = 0.0

            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]

                # Sweep current room (if it's a room, not an exit)
                room = self.building.get_room(current_node)
                if room:
                    sweep_time = room.calculate_sweep_time(self.visibility)
                    timeline.append((
                        current_time,
                        current_time + sweep_time,
                        'sweep',
                        current_node
                    ))
                    current_time += sweep_time

                # Travel to next node
                travel_time = self.building.distance(
                    current_node,
                    next_node,
                    self.walking_speed
                )
                timeline.append((
                    current_time,
                    current_time + travel_time,
                    'travel',
                    f"{current_node} -> {next_node}"
                ))
                current_time += travel_time

            # Handle last node if it's a room
            if len(path) > 0:
                last_node = path[-1]
                room = self.building.get_room(last_node)
                if room:
                    sweep_time = room.calculate_sweep_time(self.visibility)
                    timeline.append((
                        current_time,
                        current_time + sweep_time,
                        'sweep',
                        last_node
                    ))

            timelines[resp_id] = timeline

        return timelines

    def get_metrics(self) -> Dict[str, float]:
        """
        Get summary metrics for the simulation.

        Returns:
            Dictionary of metrics
        """
        if not self.results:
            return {}

        total_rooms = len(self.building.get_all_room_ids())
        assigned_rooms = sum(len(rooms) for rooms in self.assignments.values())

        metrics = {
            'total_time': self.get_total_time(),
            'num_responders': self.num_responders,
            'total_rooms': total_rooms,
            'assigned_rooms': assigned_rooms,
            'coverage': assigned_rooms / total_rooms if total_rooms > 0 else 0,
            'avg_responder_time': sum(
                self.paths[i][1] for i in range(self.num_responders) if i in self.paths
            ) / self.num_responders if self.num_responders > 0 else 0
        }

        # Calculate workload balance (standard deviation of responder times)
        times = [self.paths[i][1] for i in range(self.num_responders) if i in self.paths]
        if times:
            avg_time = sum(times) / len(times)
            variance = sum((t - avg_time) ** 2 for t in times) / len(times)
            metrics['workload_std_dev'] = variance ** 0.5
        else:
            metrics['workload_std_dev'] = 0

        return metrics

    def print_summary(self):
        """Print a summary of simulation results to console."""
        if not self.results:
            print("Simulation has not been run yet.")
            return

        print(f"\n{'='*60}")
        print(f"Evacuation Simulation Summary")
        print(f"{'='*60}")
        print(f"Number of Responders: {self.num_responders}")
        print(f"Walking Speed: {self.walking_speed} m/s")
        print(f"Visibility: {self.visibility}")
        print(f"\nTotal Evacuation Time: {self.get_total_time():.2f} seconds")

        print(f"\nResponder Details:")
        print(f"{'-'*60}")

        for resp_id in range(self.num_responders):
            if resp_id not in self.paths:
                continue

            path, time = self.paths[resp_id]
            num_rooms = len([n for n in path if self.building.get_room(n)])

            print(f"  Responder {resp_id}:")
            print(f"    - Rooms assigned: {num_rooms}")
            print(f"    - Completion time: {time:.2f} seconds")
            print(f"    - Path: {' -> '.join(path)}")

        metrics = self.get_metrics()
        print(f"\n{'='*60}")
        print(f"Metrics:")
        print(f"  - Coverage: {metrics['coverage']*100:.1f}%")
        print(f"  - Average responder time: {metrics['avg_responder_time']:.2f} seconds")
        print(f"  - Workload balance (std dev): {metrics['workload_std_dev']:.2f} seconds")
        print(f"{'='*60}\n")
