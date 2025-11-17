"""
Building scenario definitions for evacuation simulations.

This module contains predefined building layouts for testing
the evacuation sweep optimization algorithms.
"""

from building import BuildingGraph, Room, Edge


def create_scenario1() -> BuildingGraph:
    """
    Create Scenario 1: Basic Single Floor.

    Layout (graph-style):
                [Off1] [Off2] [Off3]
                  |  \ / | \ /  |
          Exit1 --+----+---+----+-- Exit2
                  |  / \ | / \  |
                [Off4] [Off5] [Off6]

    - 1 floor, 6 offices, 2 exits (8 nodes total)
    - All offices are identical (200 sq ft)
    - Exits on left and right sides (vertically centered)
    - Diagonal edges show high interconnectivity
    - NO hallway or junction nodes - direct edges only
    """
    building = BuildingGraph()

    # Create 2 exits (on left and right sides)
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # Create 6 identical offices
    for i in range(1, 7):
        room = Room(
            id=f'Office{i}',
            type='office',
            size=200.0,
            occupant_count=2,
            occupant_type='adults',
            priority=3
        )
        building.add_room(room)

    # Horizontal connections - TOP ROW (10m each)
    building.add_edge(Edge('Office1', 'Office2', 10.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office3', 10.0, 'corridor'))

    # Horizontal connections - BOTTOM ROW (10m each)
    building.add_edge(Edge('Office4', 'Office5', 10.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office6', 10.0, 'corridor'))

    # Vertical connections - connecting rows (15m each)
    building.add_edge(Edge('Office1', 'Office4', 15.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office5', 15.0, 'corridor'))
    building.add_edge(Edge('Office3', 'Office6', 15.0, 'corridor'))

    # Diagonal connections - showing connectivity (18m each)
    building.add_edge(Edge('Office1', 'Office5', 18.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office4', 18.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office6', 18.0, 'corridor'))
    building.add_edge(Edge('Office3', 'Office5', 18.0, 'corridor'))

    # Exit1 (left side) connections
    building.add_edge(Edge('Exit1', 'Office1', 12.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office4', 12.0, 'hallway'))

    # Exit2 (right side) connections
    building.add_edge(Edge('Exit2', 'Office3', 12.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office6', 12.0, 'hallway'))

    # Set explicit positions for graph-style visualization
    # Top row (y = 2.0)
    building.set_node_position('Office1', 1.0, 2.0)
    building.set_node_position('Office2', 2.0, 2.0)
    building.set_node_position('Office3', 3.0, 2.0)

    # Bottom row (y = 0.0)
    building.set_node_position('Office4', 1.0, 0.0)
    building.set_node_position('Office5', 2.0, 0.0)
    building.set_node_position('Office6', 3.0, 0.0)

    # Exits (vertically centered on left and right)
    building.set_node_position('Exit1', 0.0, 1.0)  # Left side
    building.set_node_position('Exit2', 4.0, 1.0)  # Right side

    return building


def create_scenario2() -> BuildingGraph:
    """
    Create Scenario 2: School Building (Two Floors).

    Layout:
    FLOOR 1:
         O1 --- L1 --- O2
    E1 --|      |       |-- E2
         C1 --- S1 --- C2

    FLOOR 2 (above, connected via stairs):
         C3 --- C5
          |      |
         C4 --- C6

    - 2 floors, 10 rooms total, 2 exits (12 nodes)
    - Floor 1: 2 offices, 1 lab, 2 classrooms, 1 storage
    - Floor 2: 4 classrooms
    - Stairs from exits to floor 2 (weighted heavily)
    - School-like layout with central hub pattern
    """
    building = BuildingGraph()

    # Create exits (on first floor)
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # Floor 1 rooms
    floor1_rooms = [
        Room('Office1', 'office', 200.0, 2, 'adults', 3),
        Room('Lab1', 'lab', 600.0, 15, 'adults', 4),
        Room('Office2', 'office', 200.0, 2, 'adults', 3),
        Room('Classroom1', 'classroom', 400.0, 25, 'adults', 4),
        Room('Storage1', 'storage', 300.0, 0, 'adults', 2),
        Room('Classroom2', 'classroom', 400.0, 25, 'adults', 4),
    ]

    for room in floor1_rooms:
        building.add_room(room)

    # Floor 2 rooms
    floor2_rooms = [
        Room('Classroom3', 'classroom', 400.0, 25, 'adults', 4),
        Room('Classroom4', 'classroom', 400.0, 25, 'adults', 4),
        Room('Classroom5', 'classroom', 400.0, 25, 'adults', 4),
        Room('Classroom6', 'classroom', 400.0, 25, 'adults', 4),
    ]

    for room in floor2_rooms:
        building.add_room(room)

    # Floor 1: Horizontal connections (top row)
    building.add_edge(Edge('Office1', 'Lab1', 10.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Office2', 10.0, 'corridor'))

    # Floor 1: Horizontal connections (bottom row)
    building.add_edge(Edge('Classroom1', 'Storage1', 10.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Classroom2', 10.0, 'corridor'))

    # Floor 1: Exit connections
    building.add_edge(Edge('Exit1', 'Office1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Classroom1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office2', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Classroom2', 8.0, 'hallway'))

    # Floor 1: Vertical connections (connecting top row to bottom row)
    building.add_edge(Edge('Office1', 'Classroom1', 12.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Storage1', 12.0, 'corridor'))  # Middle column vertical connection
    building.add_edge(Edge('Office2', 'Classroom2', 12.0, 'corridor'))

    # Floor 1: Cross connections (through central hub)
    building.add_edge(Edge('Office1', 'Storage1', 12.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Classroom1', 12.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Storage1', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom2', 'Lab1', 12.0, 'corridor'))

    # Floor 2: Grid connections
    building.add_edge(Edge('Classroom3', 'Classroom5', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom4', 'Classroom6', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom3', 'Classroom4', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom5', 'Classroom6', 10.0, 'corridor'))

    # Floor 2: Diagonal connections for interconnectivity
    building.add_edge(Edge('Classroom3', 'Classroom6', 14.0, 'corridor'))
    building.add_edge(Edge('Classroom4', 'Classroom5', 14.0, 'corridor'))

    # Stair connections (exits to floor 2) - heavy weight (25m for stairs)
    building.add_edge(Edge('Exit1', 'Classroom3', 25.0, 'stair'))
    building.add_edge(Edge('Exit1', 'Classroom4', 25.0, 'stair'))
    building.add_edge(Edge('Exit2', 'Classroom5', 25.0, 'stair'))
    building.add_edge(Edge('Exit2', 'Classroom6', 25.0, 'stair'))

    # Set explicit positions for visualization
    # Exits (on left and right)
    building.set_node_position('Exit1', 0.0, 1.0)
    building.set_node_position('Exit2', 4.0, 1.0)

    # Floor 1 - Top row (y=2)
    building.set_node_position('Office1', 1.0, 2.0)
    building.set_node_position('Lab1', 2.0, 2.0)
    building.set_node_position('Office2', 3.0, 2.0)

    # Floor 1 - Bottom row (y=0)
    building.set_node_position('Classroom1', 1.0, 0.0)
    building.set_node_position('Storage1', 2.0, 0.0)
    building.set_node_position('Classroom2', 3.0, 0.0)

    # Floor 2 - Left column (x=1, high y for separation)
    building.set_node_position('Classroom3', 1.0, 4.5)
    building.set_node_position('Classroom4', 1.0, 3.5)

    # Floor 2 - Right column (x=3, high y)
    building.set_node_position('Classroom5', 3.0, 4.5)
    building.set_node_position('Classroom6', 3.0, 3.5)

    return building


def create_scenario3() -> BuildingGraph:
    """
    Create Scenario 3: Multi-Exit Office Complex.

    Layout (4x3 grid with storage rooms):
         O1 --- S1 --- O2
    E1 --|      |       |-- E2
         O3 --- O4 --- O5
          |      |       |
         O6 --- O7 --- O8
    E3 --|      |       |-- E4
         O9 --- S2 --- O10

    - 1 floor, 10 offices, 2 storage rooms, 4 exits (16 nodes total)
    - Highly interconnected grid with multiple egress points
    - Demonstrates optimization with many exit options
    - Varied room sizes and weights
    """
    building = BuildingGraph()

    # Create 4 exits (on all four sides)
    building.add_exit('Exit1')
    building.add_exit('Exit2')
    building.add_exit('Exit3')
    building.add_exit('Exit4')

    # Create 10 offices with varied sizes
    offices = [
        Room('Office1', 'office', 180.0, 2, 'adults', 3),
        Room('Office2', 'office', 200.0, 2, 'adults', 3),
        Room('Office3', 'office', 190.0, 2, 'adults', 3),
        Room('Office4', 'office', 220.0, 3, 'adults', 4),  # Larger, central
        Room('Office5', 'office', 200.0, 2, 'adults', 3),
        Room('Office6', 'office', 180.0, 2, 'adults', 3),
        Room('Office7', 'office', 210.0, 2, 'adults', 3),
        Room('Office8', 'office', 200.0, 2, 'adults', 3),
        Room('Office9', 'office', 180.0, 2, 'adults', 3),
        Room('Office10', 'office', 200.0, 2, 'adults', 3),
    ]

    for office in offices:
        building.add_room(office)

    # Create 2 storage rooms
    storage_rooms = [
        Room('Storage1', 'storage', 300.0, 0, 'adults', 2),
        Room('Storage2', 'storage', 300.0, 0, 'adults', 2),
    ]

    for storage in storage_rooms:
        building.add_room(storage)

    # Row 1 (top): O1 --- S1 --- O2
    building.add_edge(Edge('Office1', 'Storage1', 10.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Office2', 10.0, 'corridor'))

    # Row 2: O3 --- O4 --- O5
    building.add_edge(Edge('Office3', 'Office4', 10.0, 'corridor'))
    building.add_edge(Edge('Office4', 'Office5', 10.0, 'corridor'))

    # Row 3: O6 --- O7 --- O8
    building.add_edge(Edge('Office6', 'Office7', 10.0, 'corridor'))
    building.add_edge(Edge('Office7', 'Office8', 10.0, 'corridor'))

    # Row 4 (bottom): O9 --- S2 --- O10
    building.add_edge(Edge('Office9', 'Storage2', 10.0, 'corridor'))
    building.add_edge(Edge('Storage2', 'Office10', 10.0, 'corridor'))

    # Vertical connections (column by column)
    building.add_edge(Edge('Office1', 'Office3', 10.0, 'corridor'))
    building.add_edge(Edge('Office3', 'Office6', 10.0, 'corridor'))
    building.add_edge(Edge('Office6', 'Office9', 10.0, 'corridor'))

    building.add_edge(Edge('Storage1', 'Office4', 10.0, 'corridor'))
    building.add_edge(Edge('Office4', 'Office7', 10.0, 'corridor'))
    building.add_edge(Edge('Office7', 'Storage2', 10.0, 'corridor'))

    building.add_edge(Edge('Office2', 'Office5', 10.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office8', 10.0, 'corridor'))
    building.add_edge(Edge('Office8', 'Office10', 10.0, 'corridor'))

    # Exit connections (as specified)
    building.add_edge(Edge('Exit1', 'Office1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office3', 8.0, 'hallway'))
    # Note: E1 NOT connected to O4 (user specified)

    building.add_edge(Edge('Exit2', 'Office2', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office5', 8.0, 'hallway'))

    building.add_edge(Edge('Exit3', 'Office6', 8.0, 'hallway'))
    building.add_edge(Edge('Exit3', 'Office9', 8.0, 'hallway'))

    building.add_edge(Edge('Exit4', 'Office8', 8.0, 'hallway'))
    building.add_edge(Edge('Exit4', 'Office10', 8.0, 'hallway'))

    # Additional cross connections for high interconnectivity
    building.add_edge(Edge('Office3', 'Storage1', 12.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Office5', 12.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office4', 12.0, 'corridor'))

    building.add_edge(Edge('Office3', 'Office7', 12.0, 'corridor'))
    building.add_edge(Edge('Office4', 'Office6', 12.0, 'corridor'))
    building.add_edge(Edge('Office4', 'Office8', 12.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office7', 12.0, 'corridor'))

    building.add_edge(Edge('Office6', 'Storage2', 12.0, 'corridor'))
    building.add_edge(Edge('Office7', 'Office9', 12.0, 'corridor'))
    building.add_edge(Edge('Office7', 'Office10', 12.0, 'corridor'))
    building.add_edge(Edge('Office8', 'Storage2', 12.0, 'corridor'))

    # Set explicit positions for visualization
    # Exits on left and right edges, vertically spaced
    building.set_node_position('Exit1', 0.0, 2.5)
    building.set_node_position('Exit2', 4.0, 2.5)
    building.set_node_position('Exit3', 0.0, 0.5)
    building.set_node_position('Exit4', 4.0, 0.5)

    # Row 1 (top, y=3)
    building.set_node_position('Office1', 1.0, 3.0)
    building.set_node_position('Storage1', 2.0, 3.0)
    building.set_node_position('Office2', 3.0, 3.0)

    # Row 2 (y=2)
    building.set_node_position('Office3', 1.0, 2.0)
    building.set_node_position('Office4', 2.0, 2.0)
    building.set_node_position('Office5', 3.0, 2.0)

    # Row 3 (y=1)
    building.set_node_position('Office6', 1.0, 1.0)
    building.set_node_position('Office7', 2.0, 1.0)
    building.set_node_position('Office8', 3.0, 1.0)

    # Row 4 (bottom, y=0)
    building.set_node_position('Office9', 1.0, 0.0)
    building.set_node_position('Storage2', 2.0, 0.0)
    building.set_node_position('Office10', 3.0, 0.0)

    return building


def create_scenario4() -> BuildingGraph:
    """
    Create Scenario 4: Elementary School Emergency (Chemical Lab Fire).

    **PART 4 EXTENSION: Priority-Based Evacuation**

    Emergency Context:
        Fire in chemistry lab requires prioritizing vulnerable populations.
        Young children (kindergarten, daycare) need evacuation first.

    Layout (single floor, 10 rooms, 2 exits):
         K1 --- Lab --- K2
    E1 --|      |       |-- E2
         DC --- C1  --- C2
          |      |       |
         O1 --- S1  --- O2

    Priority Levels:
        - Priority 5 (CRITICAL): Chemistry Lab (fire source), Kindergarten classrooms
        - Priority 4 (HIGH): Daycare center
        - Priority 3 (MEDIUM): Regular classrooms
        - Priority 2 (LOW): Offices, Storage

    Rooms:
        - 1 Chemistry Lab (fire active, priority 5)
        - 2 Kindergarten classrooms (near lab, priority 5)
        - 1 Daycare center (priority 4)
        - 2 Regular classrooms (priority 3)
        - 2 Offices (priority 2)
        - 1 Storage (priority 2)

    This scenario demonstrates:
        - Impact of priority-based room assignment
        - Trade-off between total time and high-priority room clearance time
        - Practical emergency management decision-making
    """
    building = BuildingGraph()

    # Set disaster metadata
    building.set_feature('disaster_type', 'fire')
    building.set_feature('fire_source', 'Lab1')
    building.set_feature('alarm_triggered', True)

    # Create 2 exits
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # PRIORITY 5 (CRITICAL) - Fire source and nearby kindergartens
    lab = Room('Lab1', 'lab', 600.0, 15, 'adults', 5)
    lab.set_metadata('has_fire', True)
    lab.set_metadata('temperature', 150)  # degrees F
    building.add_room(lab)

    kindergarten_rooms = [
        Room('Kindergarten1', 'classroom', 400.0, 20, 'children', 5),
        Room('Kindergarten2', 'classroom', 400.0, 20, 'children', 5),
    ]
    for room in kindergarten_rooms:
        room.set_metadata('near_fire', True)
        room.set_metadata('occupant_age', 'kindergarten')
        building.add_room(room)

    # PRIORITY 4 (HIGH) - Daycare center
    daycare = Room('Daycare1', 'daycare', 500.0, 15, 'children', 4)
    daycare.set_metadata('occupant_age', 'toddler')
    building.add_room(daycare)

    # PRIORITY 3 (MEDIUM) - Regular classrooms
    classrooms = [
        Room('Classroom1', 'classroom', 400.0, 25, 'adults', 3),
        Room('Classroom2', 'classroom', 400.0, 25, 'adults', 3),
    ]
    for room in classrooms:
        building.add_room(room)

    # PRIORITY 2 (LOW) - Offices and storage
    low_priority_rooms = [
        Room('Office1', 'office', 200.0, 2, 'adults', 2),
        Room('Office2', 'office', 200.0, 2, 'adults', 2),
        Room('Storage1', 'storage', 300.0, 0, 'adults', 1),
    ]
    for room in low_priority_rooms:
        building.add_room(room)

    # Horizontal connections - Top row: K1 --- Lab --- K2
    building.add_edge(Edge('Kindergarten1', 'Lab1', 10.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Kindergarten2', 10.0, 'corridor'))

    # Horizontal connections - Middle row: DC --- C1 --- C2
    building.add_edge(Edge('Daycare1', 'Classroom1', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Classroom2', 10.0, 'corridor'))

    # Horizontal connections - Bottom row: O1 --- S1 --- O2
    building.add_edge(Edge('Office1', 'Storage1', 10.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Office2', 10.0, 'corridor'))

    # Vertical connections (column by column)
    building.add_edge(Edge('Kindergarten1', 'Daycare1', 12.0, 'corridor'))
    building.add_edge(Edge('Daycare1', 'Office1', 12.0, 'corridor'))

    building.add_edge(Edge('Lab1', 'Classroom1', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Storage1', 12.0, 'corridor'))

    building.add_edge(Edge('Kindergarten2', 'Classroom2', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom2', 'Office2', 12.0, 'corridor'))

    # Exit connections - Deliberately place OFFICES closer to exits
    # This creates tension: nearest-neighbor wants offices, priority mode wants kindergartens
    building.add_edge(Edge('Exit1', 'Office1', 5.0, 'hallway'))  # Office is CLOSER
    building.add_edge(Edge('Exit1', 'Daycare1', 10.0, 'hallway'))  # Daycare is farther
    building.add_edge(Edge('Exit1', 'Kindergarten1', 15.0, 'hallway'))  # Kindergarten is FARTHEST

    building.add_edge(Edge('Exit2', 'Office2', 5.0, 'hallway'))  # Office is CLOSER
    building.add_edge(Edge('Exit2', 'Classroom2', 10.0, 'hallway'))  # Classroom is farther
    building.add_edge(Edge('Exit2', 'Kindergarten2', 15.0, 'hallway'))  # Kindergarten is FARTHEST

    # Diagonal connections for better connectivity
    building.add_edge(Edge('Kindergarten1', 'Classroom1', 14.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Daycare1', 14.0, 'corridor'))
    building.add_edge(Edge('Kindergarten2', 'Classroom1', 14.0, 'corridor'))

    # Set explicit positions for visualization
    # Top row (y=3)
    building.set_node_position('Kindergarten1', 1.0, 3.0)
    building.set_node_position('Lab1', 2.0, 3.0)
    building.set_node_position('Kindergarten2', 3.0, 3.0)

    # Middle row (y=2)
    building.set_node_position('Daycare1', 1.0, 2.0)
    building.set_node_position('Classroom1', 2.0, 2.0)
    building.set_node_position('Classroom2', 3.0, 2.0)

    # Bottom row (y=1)
    building.set_node_position('Office1', 1.0, 1.0)
    building.set_node_position('Storage1', 2.0, 1.0)
    building.set_node_position('Office2', 3.0, 1.0)

    # Exits on left and right
    building.set_node_position('Exit1', 0.0, 2.5)
    building.set_node_position('Exit2', 4.0, 2.5)

    return building


def create_scenario5() -> BuildingGraph:
    """
    Create Scenario 5: L-Shaped Community Center (Non-Standard Layout).

    **PART 4 EXTENSION: Non-Standard Building Layout**

    This scenario demonstrates algorithm robustness on irregular floor plans
    that deviate from typical rectangular grid structures.

    Layout (L-shaped):
         G1 ========== G2
         ||            ||
    E1 --O1 -- C1 -- C2 -- O2
         ||    ||    ||
         O3 -- DC -- C3
               ||
         [S1]-[S2]  (dead end)

    Building Features (stresses algorithm):
    - L-SHAPED floor plan (not rectangular)
    - DEAD-END corridor: Storage rooms S1, S2 (must backtrack)
    - ROOM-THROUGH-ROOM: Daycare (DC) only accessible through Office3
    - UNEQUAL EXITS: Both exits clustered on west side
    - LARGE ROOMS: 2 gymnasiums (800 sq ft each, slow sweeps)
    - LONG CORRIDORS: Some edges 35m+ (significant travel time)
    - MIX OF SIZES: Gymnasiums vs small offices

    Total: 12 rooms, 2 exits, 1 floor

    Room Types:
    - 2 Gymnasiums (G1, G2) - very large (800 sq ft)
    - 1 Daycare (DC) - accessible only through Office3 (bottleneck)
    - 4 Classrooms (C1, C2, C3, and implicit in layout)
    - 3 Offices (O1, O2, O3)
    - 2 Storage rooms (S1, S2) - dead-end corridor

    Priority Assignments:
    - Daycare: Priority 5 (young children, bottleneck access)
    - Gymnasiums: Priority 4 (high occupancy, large area)
    - Classrooms: Priority 3 (standard)
    - Offices: Priority 2 (low occupancy)
    - Storage: Priority 1 (unoccupied)

    Algorithm Challenges:
    - Handles asymmetric topology (L-shape not grid)
    - Optimizes dead-end routes (storage corridor)
    - Routes through room-behind-room (Daycare via Office3)
    - Balances long corridors vs direct paths
    - Manages unequal exit distribution

    This demonstrates real-world building complexity beyond idealized grids.
    """
    building = BuildingGraph()

    # Set building features
    building.set_feature('layout_type', 'L-shaped')
    building.set_feature('floors', 1)
    building.set_feature('irregular', True)

    # Create 2 exits (both on west side - unequal distribution)
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # ========================================================================
    # GYMNASIUM WING (Top of L)
    # ========================================================================
    gym1 = Room('Gym1', 'gymnasium', 800.0, 50, 'adults', 4)
    gym1.set_metadata('wing', 'north')
    gym1.set_metadata('occupancy_type', 'high_density')
    building.add_room(gym1)

    gym2 = Room('Gym2', 'gymnasium', 800.0, 50, 'adults', 4)
    gym2.set_metadata('wing', 'north')
    gym2.set_metadata('occupancy_type', 'high_density')
    building.add_room(gym2)

    # ========================================================================
    # MAIN CORRIDOR (Horizontal part of L)
    # ========================================================================
    office1 = Room('Office1', 'office', 180.0, 2, 'adults', 2)
    office1.set_metadata('wing', 'west')
    building.add_room(office1)

    classroom1 = Room('Classroom1', 'classroom', 400.0, 25, 'adults', 3)
    classroom1.set_metadata('wing', 'central')
    building.add_room(classroom1)

    classroom2 = Room('Classroom2', 'classroom', 400.0, 25, 'adults', 3)
    classroom2.set_metadata('wing', 'central')
    building.add_room(classroom2)

    office2 = Room('Office2', 'office', 200.0, 3, 'adults', 2)
    office2.set_metadata('wing', 'east')
    building.add_room(office2)

    # ========================================================================
    # SOUTHERN WING (Vertical part of L)
    # ========================================================================
    office3 = Room('Office3', 'office', 180.0, 2, 'adults', 2)
    office3.set_metadata('wing', 'south')
    building.add_room(office3)

    # CRITICAL: Daycare accessible ONLY through Office3 (bottleneck)
    daycare = Room('Daycare', 'daycare', 500.0, 20, 'children', 5)
    daycare.set_metadata('wing', 'south')
    daycare.set_metadata('access', 'through_office3')
    daycare.set_metadata('occupant_age', 'toddler')
    building.add_room(daycare)

    classroom3 = Room('Classroom3', 'classroom', 400.0, 20, 'adults', 3)
    classroom3.set_metadata('wing', 'south')
    building.add_room(classroom3)

    # ========================================================================
    # DEAD-END STORAGE CORRIDOR (Bottom of L)
    # ========================================================================
    storage1 = Room('Storage1', 'storage', 250.0, 0, 'adults', 1)
    storage1.set_metadata('wing', 'south')
    storage1.set_metadata('corridor_type', 'dead_end')
    building.add_room(storage1)

    storage2 = Room('Storage2', 'storage', 250.0, 0, 'adults', 1)
    storage2.set_metadata('wing', 'south')
    storage2.set_metadata('corridor_type', 'dead_end')
    building.add_room(storage2)

    # ========================================================================
    # EDGES - GYMNASIUM WING (very long corridor)
    # ========================================================================
    # Gym1 to Gym2 is a VERY long hallway (35m)
    building.add_edge(Edge('Gym1', 'Gym2', 35.0, 'corridor'))

    # Gyms connect to main corridor through Office1
    building.add_edge(Edge('Gym1', 'Office1', 15.0, 'corridor'))
    building.add_edge(Edge('Gym2', 'Office1', 20.0, 'corridor'))  # Longer path from Gym2

    # ========================================================================
    # EDGES - MAIN CORRIDOR (Horizontal)
    # ========================================================================
    building.add_edge(Edge('Office1', 'Classroom1', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Classroom2', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom2', 'Office2', 12.0, 'corridor'))

    # ========================================================================
    # EDGES - SOUTHERN WING (Vertical part of L)
    # ========================================================================
    # Connection from main corridor down to south wing
    building.add_edge(Edge('Classroom1', 'Office3', 15.0, 'corridor'))
    building.add_edge(Edge('Classroom2', 'Classroom3', 15.0, 'corridor'))

    # Southern wing internal connections
    building.add_edge(Edge('Office3', 'Daycare', 10.0, 'corridor'))
    building.add_edge(Edge('Daycare', 'Classroom3', 12.0, 'corridor'))

    # CRITICAL: Daycare is ONLY accessible through Office3
    # (no direct connection from Classroom3 to Daycare in layout)
    # This creates a bottleneck that algorithm must handle

    # ========================================================================
    # EDGES - DEAD-END STORAGE CORRIDOR
    # ========================================================================
    # Dead end: Daycare -> Storage1 -> Storage2 (must backtrack)
    building.add_edge(Edge('Daycare', 'Storage1', 18.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Storage2', 10.0, 'corridor'))
    # No other connections - this is a TRUE dead end

    # ========================================================================
    # EXIT CONNECTIONS (both on west side - unequal)
    # ========================================================================
    # Exit1 connects to gymnasium wing
    building.add_edge(Edge('Exit1', 'Gym1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office1', 8.0, 'hallway'))

    # Exit2 connects to southern wing
    building.add_edge(Edge('Exit2', 'Office3', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Classroom1', 10.0, 'hallway'))

    # ========================================================================
    # SET POSITIONS FOR VISUALIZATION (L-shape)
    # ========================================================================

    # Exits (both on left/west side)
    building.set_node_position('Exit1', 0.0, 3.0)
    building.set_node_position('Exit2', 0.0, 1.0)

    # Gymnasium wing (top horizontal bar of L)
    building.set_node_position('Gym1', 1.0, 4.0)
    building.set_node_position('Gym2', 3.5, 4.0)

    # Main corridor (middle horizontal)
    building.set_node_position('Office1', 1.0, 3.0)
    building.set_node_position('Classroom1', 2.0, 3.0)
    building.set_node_position('Classroom2', 3.0, 3.0)
    building.set_node_position('Office2', 4.0, 3.0)

    # Southern wing (vertical part of L)
    building.set_node_position('Office3', 1.0, 2.0)
    building.set_node_position('Daycare', 2.0, 2.0)
    building.set_node_position('Classroom3', 3.0, 2.0)

    # Dead-end storage corridor (bottom)
    building.set_node_position('Storage1', 2.0, 1.0)
    building.set_node_position('Storage2', 2.0, 0.0)

    return building


def create_scenario6() -> BuildingGraph:
    """
    Create Scenario 6: Office Building with Active Chemical Lab Fire.

    **PART 4 EXTENSION: Emergency Response Modeling**

    Emergency Context:
        Active fire in 2nd floor chemistry lab. Fire has spread to adjacent
        storage room. Heavy smoke on Floor 2. East stairwell blocked by debris.
        Responders must route around hazards while prioritizing high-risk areas.

    Building: 2 floors, 14 rooms, 3 exits (17 nodes total)

    Emergency Conditions:
        - Lab1 (Floor 2): Active fire, IMPASSABLE (cannot enter)
        - Storage2 (Floor 2): Adjacent to fire, high heat, IMPASSABLE
        - Floor 2 rooms near fire: Heavy smoke, visibility=0.5
        - Floor 2 hallways: Smoke-filled, walking_speed×0.7
        - East Stairwell: Blocked by debris (edge removed)
        - Floor 1: Minimal impact initially

    Layout:
    FLOOR 2 (where fire is):
         O3 --- Lab1(FIRE) --- S2(BLOCKED)
          |       |              |
    E3 --+-- C3 --+-- C4 --------+-- [BLOCKED STAIR]
          |       |              |
         O4 ----- O5 ----------- C5

    FLOOR 1 (safer):
         O1 --- C1 --- O2
    E1 --|      |       |-- E2
         O6 --- C2 --- O7

    Priority Assignments:
        - Floor 2 near fire: Priority 5 (immediate danger)
        - Floor 2 other rooms: Priority 4 (smoke exposure)
        - Floor 1 rooms: Priority 3 (secondary)

    Rooms:
        - 7 Offices (mixed priorities)
        - 5 Classrooms (priorities 3-5)
        - 1 Chemistry Lab (fire source, priority 5, IMPASSABLE)
        - 1 Storage room (adjacent to fire, IMPASSABLE)

    This scenario demonstrates:
        - Emergency routing around impassable areas
        - Environmental hazard modeling (smoke, heat, blocked paths)
        - Priority-based evacuation under adverse conditions
        - Comparison: normal conditions vs emergency conditions
        - Algorithm robustness when routes are blocked
    """
    building = BuildingGraph()

    # Set disaster metadata
    building.set_feature('disaster_type', 'fire')
    building.set_feature('fire_source', 'Lab1')
    building.set_feature('fire_floor', 2)
    building.set_feature('alarm_triggered', True)
    building.set_feature('sprinklers_active', True)

    # Create 3 exits (Exit1 and Exit2 on Floor 1, Exit3 on Floor 2)
    building.add_exit('Exit1')
    building.add_exit('Exit2')
    building.add_exit('Exit3')

    # ========================================================================
    # FLOOR 1 ROOMS (Safer area, secondary priority)
    # ========================================================================
    floor1_rooms = [
        Room('Office1', 'office', 200.0, 3, 'adults', 3),
        Room('Classroom1', 'classroom', 400.0, 25, 'adults', 3),
        Room('Office2', 'office', 200.0, 3, 'adults', 3),
        Room('Office6', 'office', 200.0, 2, 'adults', 3),
        Room('Classroom2', 'classroom', 400.0, 20, 'adults', 3),
        Room('Office7', 'office', 200.0, 2, 'adults', 3),
    ]

    for room in floor1_rooms:
        room.set_metadata('floor', 1)
        room.set_metadata('smoke_level', 'none')
        building.add_room(room)

    # ========================================================================
    # FLOOR 2 ROOMS (Fire zone, high priority)
    # ========================================================================

    # FIRE SOURCE - Chemistry Lab (IMPASSABLE)
    lab1 = Room('Lab1', 'lab', 600.0, 12, 'adults', 5)
    lab1.set_metadata('floor', 2)
    lab1.set_metadata('has_fire', True)
    lab1.set_metadata('passable', False)  # CRITICAL: Cannot enter
    lab1.set_metadata('temperature', 180)  # degrees F
    lab1.set_metadata('smoke_level', 'extreme')
    building.add_room(lab1)

    # ADJACENT TO FIRE - Storage (IMPASSABLE due to heat/smoke)
    storage2 = Room('Storage2', 'storage', 300.0, 0, 'adults', 5)
    storage2.set_metadata('floor', 2)
    storage2.set_metadata('near_fire', True)
    storage2.set_metadata('passable', False)  # CRITICAL: Too dangerous
    storage2.set_metadata('temperature', 140)  # degrees F
    storage2.set_metadata('smoke_level', 'extreme')
    building.add_room(storage2)

    # NEAR FIRE - Heavy smoke zones (PASSABLE but hazardous)
    office3 = Room('Office3', 'office', 200.0, 3, 'adults', 5)
    office3.set_metadata('floor', 2)
    office3.set_metadata('near_fire', True)
    office3.set_metadata('smoke_level', 'high')
    office3.set_metadata('visibility_factor', 0.5)  # Reduced visibility
    building.add_room(office3)

    classroom3 = Room('Classroom3', 'classroom', 400.0, 25, 'adults', 5)
    classroom3.set_metadata('floor', 2)
    classroom3.set_metadata('near_fire', True)
    classroom3.set_metadata('smoke_level', 'high')
    classroom3.set_metadata('visibility_factor', 0.5)
    building.add_room(classroom3)

    # FLOOR 2 - Other rooms (smoke but farther from fire)
    floor2_other = [
        Room('Office4', 'office', 200.0, 2, 'adults', 4),
        Room('Office5', 'office', 200.0, 2, 'adults', 4),
        Room('Classroom4', 'classroom', 400.0, 20, 'adults', 4),
        Room('Classroom5', 'classroom', 400.0, 20, 'adults', 4),
    ]

    for room in floor2_other:
        room.set_metadata('floor', 2)
        room.set_metadata('smoke_level', 'medium')
        room.set_metadata('visibility_factor', 0.7)
        building.add_room(room)

    # ========================================================================
    # FLOOR 1 EDGES (Normal conditions)
    # ========================================================================

    # Floor 1: Top row connections (O1 --- C1 --- O2)
    building.add_edge(Edge('Office1', 'Classroom1', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Office2', 10.0, 'corridor'))

    # Floor 1: Bottom row connections (O6 --- C2 --- O7)
    building.add_edge(Edge('Office6', 'Classroom2', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom2', 'Office7', 10.0, 'corridor'))

    # Floor 1: Vertical connections
    building.add_edge(Edge('Office1', 'Office6', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Classroom2', 12.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office7', 12.0, 'corridor'))

    # Floor 1: Exit connections
    building.add_edge(Edge('Exit1', 'Office1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office6', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office2', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office7', 8.0, 'hallway'))

    # ========================================================================
    # FLOOR 2 EDGES (Emergency conditions - some blocked)
    # ========================================================================

    # Floor 2: Top row - FIRE ZONE
    # NOTE: Lab1 and Storage2 are IMPASSABLE, but edges exist in normal graph
    # These will be filtered out during emergency routing
    fire_zone_edge1 = Edge('Office3', 'Lab1', 10.0, 'corridor')
    fire_zone_edge1.set_metadata('smoke_level', 'high')
    fire_zone_edge1.set_metadata('blocked', True)  # Cannot traverse due to fire
    building.add_edge(fire_zone_edge1)

    fire_zone_edge2 = Edge('Lab1', 'Storage2', 10.0, 'corridor')
    fire_zone_edge2.set_metadata('smoke_level', 'extreme')
    fire_zone_edge2.set_metadata('blocked', True)  # Fire spread path
    building.add_edge(fire_zone_edge2)

    # Floor 2: Middle row connections (C3 and C4 accessible)
    smoke_edge1 = Edge('Office3', 'Classroom3', 12.0, 'corridor')
    smoke_edge1.set_metadata('smoke_level', 'high')
    smoke_edge1.set_metadata('speed_factor', 0.7)  # Slow movement through smoke
    building.add_edge(smoke_edge1)

    smoke_edge2 = Edge('Classroom3', 'Classroom4', 12.0, 'corridor')
    smoke_edge2.set_metadata('smoke_level', 'medium')
    smoke_edge2.set_metadata('speed_factor', 0.8)
    building.add_edge(smoke_edge2)

    smoke_edge3 = Edge('Classroom4', 'Storage2', 12.0, 'corridor')
    smoke_edge3.set_metadata('smoke_level', 'high')
    smoke_edge3.set_metadata('blocked', True)  # Near fire, unsafe
    building.add_edge(smoke_edge3)

    # Floor 2: Bottom row connections
    building.add_edge(Edge('Office4', 'Office5', 12.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Classroom5', 12.0, 'corridor'))

    # Floor 2: Vertical connections within floor
    building.add_edge(Edge('Office3', 'Office4', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom3', 'Office5', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom4', 'Classroom5', 10.0, 'corridor'))

    # Floor 2: Exit3 connections (emergency exit on Floor 2 west side)
    building.add_edge(Edge('Exit3', 'Office3', 8.0, 'hallway'))
    building.add_edge(Edge('Exit3', 'Classroom3', 8.0, 'hallway'))
    building.add_edge(Edge('Exit3', 'Office4', 8.0, 'hallway'))

    # ========================================================================
    # STAIRWELL CONNECTIONS (some blocked)
    # ========================================================================

    # West stairwell (OPERATIONAL) - connects Exit1 to Floor 2
    building.add_edge(Edge('Exit1', 'Office3', 20.0, 'stair'))
    building.add_edge(Edge('Exit1', 'Office4', 22.0, 'stair'))

    # Central access (OPERATIONAL) - limited capacity
    building.add_edge(Edge('Classroom1', 'Classroom3', 25.0, 'stair'))
    building.add_edge(Edge('Classroom1', 'Office5', 25.0, 'stair'))

    # East stairwell (BLOCKED by debris) - these edges are blocked
    blocked_stair1 = Edge('Exit2', 'Classroom4', 20.0, 'stair')
    blocked_stair1.set_metadata('blocked', True)
    blocked_stair1.set_metadata('blocked_reason', 'debris')
    building.add_edge(blocked_stair1)

    blocked_stair2 = Edge('Exit2', 'Classroom5', 20.0, 'stair')
    blocked_stair2.set_metadata('blocked', True)
    blocked_stair2.set_metadata('blocked_reason', 'debris')
    building.add_edge(blocked_stair2)

    # ========================================================================
    # SET POSITIONS FOR VISUALIZATION
    # ========================================================================

    # Floor 1 positions (y = 0-2)
    building.set_node_position('Exit1', 0.0, 1.0)
    building.set_node_position('Exit2', 6.0, 1.0)

    # Floor 1 top row (y=2)
    building.set_node_position('Office1', 1.0, 2.0)
    building.set_node_position('Classroom1', 2.5, 2.0)
    building.set_node_position('Office2', 4.0, 2.0)

    # Floor 1 bottom row (y=0)
    building.set_node_position('Office6', 1.0, 0.0)
    building.set_node_position('Classroom2', 2.5, 0.0)
    building.set_node_position('Office7', 4.0, 0.0)

    # Floor 2 positions (y = 4-7, separated vertically)
    building.set_node_position('Exit3', 0.0, 5.5)

    # Floor 2 top row (y=7) - FIRE ZONE
    building.set_node_position('Office3', 1.0, 7.0)
    building.set_node_position('Lab1', 2.5, 7.0)  # FIRE SOURCE
    building.set_node_position('Storage2', 4.0, 7.0)  # BLOCKED

    # Floor 2 middle row (y=5.5)
    building.set_node_position('Classroom3', 1.0, 5.5)
    building.set_node_position('Classroom4', 2.5, 5.5)
    building.set_node_position('Classroom5', 4.0, 5.5)

    # Floor 2 bottom row (y=4)
    building.set_node_position('Office4', 1.0, 4.0)
    building.set_node_position('Office5', 2.5, 4.0)

    return building


def create_scenario7() -> BuildingGraph:
    """
    Create Scenario 7: Hospital with Odorless Gas Leak.

    **PART 4 EXTENSION: Gas Emergency with Unaware Occupants**

    Emergency Context:
        Natural gas leak in basement mechanical room. Gas is odorless
        (no mercaptan added). Slowly spreading through ventilation system.
        Occupants are unaware of danger and may resist evacuation.

    Building: 2 floors, 12 rooms, 2 exits (14 nodes total)

    Emergency Conditions:
        - Gas source: Basement Mechanical Room (not directly accessible)
        - Gas-affected zones: Increasing concentrations by floor/proximity
        - NO impassable rooms (gas is invisible, can traverse all areas)
        - Occupants UNAWARE (increases sweep time by 50%)
        - Some patients bedridden (slow evacuation assistance needed)
        - Priority: Intensive care > Patient rooms > Staff areas

    Layout:
    FLOOR 2 (Upper):
         ICU1 ---- ICU2 ---- Nurse
          |         |         |
    E1 --Lab ------ OR ----- Lounge-- E2

    FLOOR 1 (Ground - gas spreading):
         PR1 ----- PR2 ----- Admin
          |         |         |
         Pharm --- Lobby --- Storage

    GAS CONCENTRATION ZONES:
        - Floor 1 (ground): HIGH gas (near basement source)
        - Floor 2 (upper): MEDIUM gas (via ventilation)
        - Stairwells: Gas rising (moderate concentration)

    Priority Assignments (medical emergency):
        - Priority 5: ICU rooms (critical patients, life support)
        - Priority 4: Operating Room, Patient Rooms (occupied)
        - Priority 3: Nurse station, Pharmacy (staff present)
        - Priority 2: Lounge, Admin, Lab (low occupancy)
        - Priority 1: Storage (unoccupied)

    Rooms:
        - 2 ICU rooms (Intensive Care Units) - priority 5
        - 1 Operating Room - priority 4
        - 2 Patient Rooms - priority 4
        - 1 Pharmacy - priority 3
        - 1 Nurse Station - priority 3
        - 1 Lab - priority 2
        - 1 Lounge - priority 2
        - 1 Admin - priority 2
        - 1 Lobby - priority 2
        - 1 Storage - priority 1

    Key Differences from Fire Scenario:
        - ALL rooms are passable (gas is invisible/traversable)
        - Sweep times INCREASED (must convince unaware occupants)
        - NO blocked paths (can use all routes)
        - Environmental hazard: Gas concentration affects responder safety
        - Focus on PRIORITY evacuation (ICU patients at highest risk)

    This demonstrates:
        - Occupant awareness modeling (unaware = longer sweeps)
        - Invisible hazard management (gas vs fire)
        - Medical facility priorities
        - Time pressure without route blockages
    """
    building = BuildingGraph()

    # Set disaster metadata
    building.set_feature('disaster_type', 'gas_leak')
    building.set_feature('gas_source', 'basement_mechanical')
    building.set_feature('gas_type', 'natural_gas')
    building.set_feature('odorless', True)
    building.set_feature('occupants_aware', False)

    # Create 2 exits
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # ========================================================================
    # FLOOR 2 ROOMS (Upper floor - moderate gas concentration)
    # ========================================================================

    # PRIORITY 5 - ICU (Intensive Care - critical patients)
    icu1 = Room('ICU1', 'icu', 500.0, 8, 'patients', 5)
    icu1.set_metadata('floor', 2)
    icu1.set_metadata('gas_concentration', 'medium')
    icu1.set_metadata('patient_mobility', 'bedridden')
    icu1.set_metadata('life_support', True)
    building.add_room(icu1)

    icu2 = Room('ICU2', 'icu', 500.0, 8, 'patients', 5)
    icu2.set_metadata('floor', 2)
    icu2.set_metadata('gas_concentration', 'medium')
    icu2.set_metadata('patient_mobility', 'bedridden')
    icu2.set_metadata('life_support', True)
    building.add_room(icu2)

    # PRIORITY 4 - Operating Room (surgery in progress)
    operating_room = Room('OperatingRoom', 'operating_room', 600.0, 10, 'mixed', 4)
    operating_room.set_metadata('floor', 2)
    operating_room.set_metadata('gas_concentration', 'medium')
    operating_room.set_metadata('surgery_active', True)
    building.add_room(operating_room)

    # PRIORITY 3 - Nurse Station
    nurse_station = Room('NurseStation', 'office', 300.0, 5, 'adults', 3)
    nurse_station.set_metadata('floor', 2)
    nurse_station.set_metadata('gas_concentration', 'medium')
    building.add_room(nurse_station)

    # PRIORITY 2 - Staff areas
    lounge = Room('Lounge', 'lounge', 250.0, 3, 'adults', 2)
    lounge.set_metadata('floor', 2)
    lounge.set_metadata('gas_concentration', 'low')
    building.add_room(lounge)

    lab = Room('Lab', 'lab', 400.0, 4, 'adults', 2)
    lab.set_metadata('floor', 2)
    lab.set_metadata('gas_concentration', 'medium')
    building.add_room(lab)

    # ========================================================================
    # FLOOR 1 ROOMS (Ground floor - high gas concentration near source)
    # ========================================================================

    # PRIORITY 4 - Patient Rooms
    patient_room1 = Room('PatientRoom1', 'patient_room', 350.0, 4, 'patients', 4)
    patient_room1.set_metadata('floor', 1)
    patient_room1.set_metadata('gas_concentration', 'high')
    patient_room1.set_metadata('patient_mobility', 'limited')
    building.add_room(patient_room1)

    patient_room2 = Room('PatientRoom2', 'patient_room', 350.0, 4, 'patients', 4)
    patient_room2.set_metadata('floor', 1)
    patient_room2.set_metadata('gas_concentration', 'high')
    patient_room2.set_metadata('patient_mobility', 'limited')
    building.add_room(patient_room2)

    # PRIORITY 3 - Pharmacy (staff + medications)
    pharmacy = Room('Pharmacy', 'pharmacy', 300.0, 3, 'adults', 3)
    pharmacy.set_metadata('floor', 1)
    pharmacy.set_metadata('gas_concentration', 'high')
    building.add_room(pharmacy)

    # PRIORITY 2 - Administrative and common areas
    admin = Room('Admin', 'office', 250.0, 3, 'adults', 2)
    admin.set_metadata('floor', 1)
    admin.set_metadata('gas_concentration', 'medium')
    building.add_room(admin)

    lobby = Room('Lobby', 'lobby', 400.0, 5, 'mixed', 2)
    lobby.set_metadata('floor', 1)
    lobby.set_metadata('gas_concentration', 'high')
    building.add_room(lobby)

    # PRIORITY 1 - Storage (unoccupied)
    storage = Room('Storage', 'storage', 200.0, 0, 'adults', 1)
    storage.set_metadata('floor', 1)
    storage.set_metadata('gas_concentration', 'medium')
    building.add_room(storage)

    # ========================================================================
    # FLOOR 2 EDGES
    # ========================================================================
    # Top row: ICU1 --- ICU2 --- Nurse
    building.add_edge(Edge('ICU1', 'ICU2', 12.0, 'corridor'))
    building.add_edge(Edge('ICU2', 'NurseStation', 12.0, 'corridor'))

    # Bottom row: Lab --- OR --- Lounge
    building.add_edge(Edge('Lab', 'OperatingRoom', 12.0, 'corridor'))
    building.add_edge(Edge('OperatingRoom', 'Lounge', 12.0, 'corridor'))

    # Vertical connections
    building.add_edge(Edge('ICU1', 'Lab', 10.0, 'corridor'))
    building.add_edge(Edge('ICU2', 'OperatingRoom', 10.0, 'corridor'))
    building.add_edge(Edge('NurseStation', 'Lounge', 10.0, 'corridor'))

    # ========================================================================
    # FLOOR 1 EDGES
    # ========================================================================
    # Top row: PR1 --- PR2 --- Admin
    building.add_edge(Edge('PatientRoom1', 'PatientRoom2', 10.0, 'corridor'))
    building.add_edge(Edge('PatientRoom2', 'Admin', 10.0, 'corridor'))

    # Bottom row: Pharm --- Lobby --- Storage
    building.add_edge(Edge('Pharmacy', 'Lobby', 12.0, 'corridor'))
    building.add_edge(Edge('Lobby', 'Storage', 12.0, 'corridor'))

    # Vertical connections
    building.add_edge(Edge('PatientRoom1', 'Pharmacy', 10.0, 'corridor'))
    building.add_edge(Edge('PatientRoom2', 'Lobby', 10.0, 'corridor'))
    building.add_edge(Edge('Admin', 'Storage', 10.0, 'corridor'))

    # ========================================================================
    # STAIRWELL CONNECTIONS (gas rising through these)
    # ========================================================================
    # West stairwell (connects ICU floor to patient floor)
    stair_west1 = Edge('ICU1', 'PatientRoom1', 20.0, 'stair')
    stair_west1.set_metadata('gas_concentration', 'medium')
    building.add_edge(stair_west1)

    stair_west2 = Edge('Lab', 'Pharmacy', 20.0, 'stair')
    stair_west2.set_metadata('gas_concentration', 'medium')
    building.add_edge(stair_west2)

    # East stairwell
    stair_east1 = Edge('NurseStation', 'Admin', 20.0, 'stair')
    stair_east1.set_metadata('gas_concentration', 'medium')
    building.add_edge(stair_east1)

    stair_east2 = Edge('Lounge', 'Storage', 20.0, 'stair')
    stair_east2.set_metadata('gas_concentration', 'low')
    building.add_edge(stair_east2)

    # ========================================================================
    # EXIT CONNECTIONS (both floors)
    # ========================================================================
    # Exit1 (west side) - connects to Floor 2 and Floor 1
    building.add_edge(Edge('Exit1', 'ICU1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Lab', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'PatientRoom1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Pharmacy', 8.0, 'hallway'))

    # Exit2 (east side) - connects to Floor 2 and Floor 1
    building.add_edge(Edge('Exit2', 'Lounge', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'NurseStation', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Admin', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Storage', 8.0, 'hallway'))

    # ========================================================================
    # SET POSITIONS FOR VISUALIZATION
    # ========================================================================
    # Exits (left and right)
    building.set_node_position('Exit1', 0.0, 2.5)
    building.set_node_position('Exit2', 5.0, 2.5)

    # Floor 2 (y = 4-5)
    # Top row (ICU)
    building.set_node_position('ICU1', 1.0, 5.0)
    building.set_node_position('ICU2', 2.0, 5.0)
    building.set_node_position('NurseStation', 3.0, 5.0)

    # Bottom row (surgical/staff)
    building.set_node_position('Lab', 1.0, 4.0)
    building.set_node_position('OperatingRoom', 2.0, 4.0)
    building.set_node_position('Lounge', 3.0, 4.0)

    # Floor 1 (y = 0-1)
    # Top row (patient rooms)
    building.set_node_position('PatientRoom1', 1.0, 1.0)
    building.set_node_position('PatientRoom2', 2.0, 1.0)
    building.set_node_position('Admin', 3.0, 1.0)

    # Bottom row (services)
    building.set_node_position('Pharmacy', 1.0, 0.0)
    building.set_node_position('Lobby', 2.0, 0.0)
    building.set_node_position('Storage', 3.0, 0.0)

    return building
