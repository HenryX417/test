"""
Building data structures for emergency evacuation sweep optimization.

This module contains the core data structures representing rooms, edges,
and building graphs used in the evacuation simulation system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import heapq


@dataclass
class Room:
    """Represents a room in the building."""
    id: str
    type: str  # 'office', 'classroom', 'storage', 'lab', 'daycare'
    size: float  # square feet
    occupant_count: int
    occupant_type: str  # 'adults', 'children', 'mixed'
    priority: int  # 1-5, higher = more urgent

    # EXTENSIBILITY: Add custom metadata for advanced features (Part 4+)
    # Examples: is_on_fire, has_gas_leak, is_blocked, temperature, smoke_density
    metadata: Dict = None

    def __post_init__(self):
        """Initialize metadata dictionary if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def set_metadata(self, key: str, value):
        """Set custom metadata for this room."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """Get custom metadata value."""
        return self.metadata.get(key, default)

    def calculate_sweep_time(self, visibility: float = 1.0) -> float:
        """
        Calculate sweep time based on room type, size, and visibility.

        Base time calculation:
        - Office: 30s + 5s per 100 sq ft
        - Classroom: 60s + 10s per 100 sq ft
        - Storage: 15s + 2s per 100 sq ft
        - Lab: 90s + 15s per 100 sq ft
        - Daycare: 120s + 20s per 100 sq ft

        Adjust by visibility factor (0.5 = half visibility, doubles time)
        Also adjust based on occupant type (children take longer)

        Args:
            visibility: Visibility factor (0.0 to 1.0)

        Returns:
            Sweep time in seconds
        """
        # Base time by room type
        base_times = {
            'office': 30,
            'classroom': 60,
            'storage': 15,
            'lab': 90,
            'daycare': 120
        }

        # Additional time per 100 sq ft
        size_factors = {
            'office': 5,
            'classroom': 10,
            'storage': 2,
            'lab': 15,
            'daycare': 20
        }

        base = base_times.get(self.type, 30)
        size_factor = size_factors.get(self.type, 5)

        # Calculate size component
        size_time = (self.size / 100.0) * size_factor

        # Total base time
        total_time = base + size_time

        # Adjust for occupant type
        occupant_multiplier = 1.0
        if self.occupant_type == 'children':
            occupant_multiplier = 1.5  # Children are harder to evacuate
        elif self.occupant_type == 'mixed':
            occupant_multiplier = 1.2

        total_time *= occupant_multiplier

        # Adjust for visibility (lower visibility = longer time)
        if visibility > 0:
            total_time /= visibility
        else:
            total_time *= 2  # Default doubling for zero visibility

        return total_time


@dataclass
class Edge:
    """Represents a connection between two nodes in the building."""
    start: str  # Also called node1 for consistency
    end: str    # Also called node2 for consistency
    distance: float  # meters
    edge_type: str  # 'hallway', 'stair', 'corridor'
    metadata: Dict[str, Any] = None  # Emergency and environmental metadata

    def __post_init__(self):
        """Initialize metadata dict if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def node1(self) -> str:
        """Alias for start node."""
        return self.start

    @property
    def node2(self) -> str:
        """Alias for end node."""
        return self.end

    def set_metadata(self, key: str, value: Any):
        """Set metadata value."""
        self.metadata[key] = value

    def calculate_travel_time(self, walking_speed: float) -> float:
        """
        Calculate travel time based on distance and walking speed.

        Args:
            walking_speed: Walking speed in m/s

        Returns:
            Travel time in seconds
        """
        # Stairs have 0.5x speed multiplier (slower movement)
        speed_multiplier = 0.5 if self.edge_type == 'stair' else 1.0

        effective_speed = walking_speed * speed_multiplier

        if effective_speed > 0:
            return self.distance / effective_speed
        else:
            return float('inf')


class BuildingGraph:
    """Represents the building as a graph structure."""

    def __init__(self):
        """Initialize an empty building graph."""
        self.rooms: Dict[str, Room] = {}
        self.exits: List[str] = []
        self.routing_nodes: List[str] = []  # Hallway junctions, stairs, etc.
        self.edges: List[Edge] = []
        self.adjacency_list: Dict[str, List[Tuple[str, Edge]]] = {}
        self.node_positions: Dict[str, Tuple[float, float]] = {}  # Explicit positions for visualization

        # EXTENSIBILITY: Global features for advanced scenarios (Part 4+)
        # Examples: disaster_type='fire'/'gas'/'earthquake', alarm_triggered=True,
        #           sprinkler_active=True, emergency_lighting=False
        self.features: Dict = {}

    def set_feature(self, key: str, value):
        """Set a global building feature."""
        self.features[key] = value

    def get_feature(self, key: str, default=None):
        """Get a global building feature."""
        return self.features.get(key, default)

    def add_room(self, room: Room):
        """
        Add a room to the building graph.

        Args:
            room: Room object to add
        """
        self.rooms[room.id] = room
        if room.id not in self.adjacency_list:
            self.adjacency_list[room.id] = []

    def add_exit(self, exit_id: str):
        """
        Add an exit to the building.

        Args:
            exit_id: ID of the exit node
        """
        if exit_id not in self.exits:
            self.exits.append(exit_id)
        if exit_id not in self.adjacency_list:
            self.adjacency_list[exit_id] = []

    def add_routing_node(self, node_id: str):
        """
        Add a routing node (hallway junction, stairs, etc.) to the building.

        Args:
            node_id: ID of the routing node
        """
        if node_id not in self.routing_nodes:
            self.routing_nodes.append(node_id)
        if node_id not in self.adjacency_list:
            self.adjacency_list[node_id] = []

    def set_node_position(self, node_id: str, x: float, y: float):
        """
        Set explicit position for a node (for visualization).

        Args:
            node_id: ID of the node
            x: X coordinate
            y: Y coordinate
        """
        self.node_positions[node_id] = (x, y)

    def add_edge(self, edge: Edge):
        """
        Add a bidirectional edge to the graph.

        Args:
            edge: Edge object to add
        """
        self.edges.append(edge)

        # Ensure nodes exist in adjacency list
        if edge.start not in self.adjacency_list:
            self.adjacency_list[edge.start] = []
        if edge.end not in self.adjacency_list:
            self.adjacency_list[edge.end] = []

        # Add bidirectional edges
        self.adjacency_list[edge.start].append((edge.end, edge))

        # Create reverse edge
        reverse_edge = Edge(edge.end, edge.start, edge.distance, edge.edge_type)
        self.adjacency_list[edge.end].append((edge.start, reverse_edge))

    def shortest_path(self, start: str, end: str, walking_speed: float = 1.5) -> Tuple[List[str], float]:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.

        Args:
            start: Starting node ID
            end: Ending node ID
            walking_speed: Walking speed in m/s

        Returns:
            Tuple of (path as list of node IDs, total travel time)

        Raises:
            ValueError: If start or end nodes don't exist
        """
        if start not in self.adjacency_list:
            raise ValueError(f"Start node '{start}' not found in building graph")
        if end not in self.adjacency_list:
            raise ValueError(f"End node '{end}' not found in building graph")

        if start == end:
            return ([start], 0.0)

        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.adjacency_list}
        distances[start] = 0
        previous = {node: None for node in self.adjacency_list}

        # Priority queue: (distance, node)
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == end:
                break

            # Check neighbors
            for neighbor, edge in self.adjacency_list.get(current_node, []):
                if neighbor in visited:
                    continue

                travel_time = edge.calculate_travel_time(walking_speed)
                new_dist = current_dist + travel_time

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        if distances[end] == float('inf'):
            raise ValueError(f"No path exists from '{start}' to '{end}'")

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()

        return (path, distances[end])

    def distance(self, node1: str, node2: str, walking_speed: float = 1.5) -> float:
        """
        Return travel time between two nodes.

        Args:
            node1: First node ID
            node2: Second node ID
            walking_speed: Walking speed in m/s

        Returns:
            Travel time in seconds
        """
        _, dist = self.shortest_path(node1, node2, walking_speed)
        return dist

    def get_all_room_ids(self) -> List[str]:
        """
        Get list of all room IDs (excluding exits).

        Returns:
            List of room IDs
        """
        return list(self.rooms.keys())

    def get_room(self, room_id: str) -> Optional[Room]:
        """
        Get room by ID.

        Args:
            room_id: Room ID

        Returns:
            Room object or None if not found
        """
        return self.rooms.get(room_id)

    def expand_path_with_intermediates(self, path: List[str], walking_speed: float = 1.5) -> List[str]:
        """
        Expand a path to include all intermediate nodes traversed.

        When a path goes from A to C, it might actually traverse through B.
        This function expands [A, C] to [A, B, C] using shortest paths.

        Args:
            path: Original path (may skip intermediate nodes)
            walking_speed: Walking speed for pathfinding

        Returns:
            Expanded path with all intermediate nodes
        """
        if len(path) <= 1:
            return path

        expanded = [path[0]]

        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]

            # Get shortest path between consecutive nodes
            shortest, _ = self.shortest_path(start, end, walking_speed)

            # Add all nodes except the first (already in expanded)
            expanded.extend(shortest[1:])

        return expanded
