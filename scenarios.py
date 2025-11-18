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
    building.add_edge(Edge('Office1', 'Office4', 14.0, 'corridor'))  # Diagonal connection
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
    Create Scenario 4: Hospital Building with Priority-Based Evacuation (Two Floors).

    **PART 4 EXTENSION: Priority-Based Evacuation on Multi-Floor Building**

    Emergency Context:
        Hospital with critical care units on upper floor. Priority evacuation
        required for ICU patients, operating rooms, and emergency department.

    Layout:
    FLOOR 1 (Ground):
         O1 --- R1 --- O2
    E1 --|      |       |-- E2
         W1 --- S1 --- W2

    FLOOR 2 (Critical Care, connected via stairs):
         ICU1 -- OR  -- ICU2
           |     |       |
         ER1 --- ER2 -- ER3

    - 2 floors, 12 rooms total, 2 exits (14 nodes)
    - Floor 1: 2 offices, 1 reception, 2 wards, 1 storage (priority 2-3)
    - Floor 2: 2 ICUs, 1 OR, 3 ERs (priority 4-5)
    - Stairs from exits to floor 2 (weighted heavily: 25m)
    - Hospital layout with high-priority critical care upstairs

    Priority Levels:
        - Priority 5 (CRITICAL): ICU rooms, Operating Room
        - Priority 4 (HIGH): Emergency Rooms
        - Priority 3 (MEDIUM): General Wards, Reception
        - Priority 2 (LOW): Offices, Storage

    This scenario demonstrates:
        - Priority-based evacuation across multiple floors
        - Trade-off: High-priority rooms are farther (upstairs) vs low-priority closer
        - Comparison of priority mode vs. distance-optimized mode
        - Realistic hospital emergency evacuation
    """
    building = BuildingGraph()

    # Set hospital metadata
    building.set_feature('building_type', 'hospital')
    building.set_feature('floors', 2)
    building.set_feature('has_priority_rooms', True)

    # Create exits (on first floor)
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # ========================================================================
    # FLOOR 1 ROOMS (Lower Priority - Offices, Reception, Wards, Storage)
    # ========================================================================
    floor1_rooms = [
        Room('Office1', 'office', 200.0, 2, 'adults', 2),
        Room('Reception', 'reception', 300.0, 3, 'adults', 3),
        Room('Office2', 'office', 200.0, 2, 'adults', 2),
        Room('Ward1', 'ward', 400.0, 8, 'adults', 3),
        Room('Storage1', 'storage', 300.0, 0, 'adults', 1),
        Room('Ward2', 'ward', 400.0, 8, 'adults', 3),
    ]

    for room in floor1_rooms:
        room.set_metadata('floor', 1)
        building.add_room(room)

    # ========================================================================
    # FLOOR 2 ROOMS (High Priority - ICU, OR, ER)
    # ========================================================================
    # PRIORITY 5 (CRITICAL) - ICU and Operating Room
    icu_rooms = [
        Room('ICU1', 'icu', 500.0, 6, 'adults', 5),
        Room('ICU2', 'icu', 500.0, 6, 'adults', 5),
    ]
    for room in icu_rooms:
        room.set_metadata('floor', 2)
        room.set_metadata('patient_type', 'critical')
        building.add_room(room)

    operating_room = Room('OR1', 'operating_room', 600.0, 5, 'adults', 5)
    operating_room.set_metadata('floor', 2)
    operating_room.set_metadata('in_use', True)
    building.add_room(operating_room)

    # PRIORITY 4 (HIGH) - Emergency Rooms
    er_rooms = [
        Room('ER1', 'emergency', 400.0, 4, 'adults', 4),
        Room('ER2', 'emergency', 400.0, 4, 'adults', 4),
        Room('ER3', 'emergency', 400.0, 4, 'adults', 4),
    ]
    for room in er_rooms:
        room.set_metadata('floor', 2)
        room.set_metadata('patient_type', 'urgent')
        building.add_room(room)

    # ========================================================================
    # FLOOR 1 EDGES
    # ========================================================================
    # Top row: O1 --- R1 --- O2
    building.add_edge(Edge('Office1', 'Reception', 10.0, 'corridor'))
    building.add_edge(Edge('Reception', 'Office2', 10.0, 'corridor'))

    # Bottom row: W1 --- S1 --- W2
    building.add_edge(Edge('Ward1', 'Storage1', 10.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Ward2', 10.0, 'corridor'))

    # Vertical connections
    building.add_edge(Edge('Office1', 'Ward1', 12.0, 'corridor'))
    building.add_edge(Edge('Reception', 'Storage1', 12.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Ward2', 12.0, 'corridor'))

    # Cross connections
    building.add_edge(Edge('Office1', 'Storage1', 14.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Storage1', 14.0, 'corridor'))
    building.add_edge(Edge('Reception', 'Ward1', 14.0, 'corridor'))
    building.add_edge(Edge('Reception', 'Ward2', 14.0, 'corridor'))

    # Exit connections
    building.add_edge(Edge('Exit1', 'Office1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Ward1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office2', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Ward2', 8.0, 'hallway'))

    # ========================================================================
    # FLOOR 2 EDGES
    # ========================================================================
    # Top row: ICU1 --- OR1 --- ICU2
    building.add_edge(Edge('ICU1', 'OR1', 10.0, 'corridor'))
    building.add_edge(Edge('OR1', 'ICU2', 10.0, 'corridor'))

    # Bottom row: ER1 --- ER2 --- ER3
    building.add_edge(Edge('ER1', 'ER2', 10.0, 'corridor'))
    building.add_edge(Edge('ER2', 'ER3', 10.0, 'corridor'))

    # Vertical connections
    building.add_edge(Edge('ICU1', 'ER1', 12.0, 'corridor'))
    building.add_edge(Edge('OR1', 'ER2', 12.0, 'corridor'))
    building.add_edge(Edge('ICU2', 'ER3', 12.0, 'corridor'))

    # Cross connections
    building.add_edge(Edge('ICU1', 'ER2', 14.0, 'corridor'))
    building.add_edge(Edge('ICU2', 'ER2', 14.0, 'corridor'))
    building.add_edge(Edge('OR1', 'ER1', 14.0, 'corridor'))
    building.add_edge(Edge('OR1', 'ER3', 14.0, 'corridor'))

    # ========================================================================
    # STAIR CONNECTIONS (Exits to Floor 2) - Heavy weight (25m)
    # ========================================================================
    building.add_edge(Edge('Exit1', 'ICU1', 25.0, 'stair'))
    building.add_edge(Edge('Exit1', 'ER1', 25.0, 'stair'))
    building.add_edge(Edge('Exit2', 'ICU2', 25.0, 'stair'))
    building.add_edge(Edge('Exit2', 'ER3', 25.0, 'stair'))

    # ========================================================================
    # VISUALIZATION POSITIONS
    # ========================================================================
    # Exits (on left and right)
    building.set_node_position('Exit1', 0.0, 1.0)
    building.set_node_position('Exit2', 4.0, 1.0)

    # Floor 1 - Top row (y=2)
    building.set_node_position('Office1', 1.0, 2.0)
    building.set_node_position('Reception', 2.0, 2.0)
    building.set_node_position('Office2', 3.0, 2.0)

    # Floor 1 - Bottom row (y=0)
    building.set_node_position('Ward1', 1.0, 0.0)
    building.set_node_position('Storage1', 2.0, 0.0)
    building.set_node_position('Ward2', 3.0, 0.0)

    # Floor 2 - Top row (y=4.5, separated for clarity)
    building.set_node_position('ICU1', 1.0, 4.5)
    building.set_node_position('OR1', 2.0, 4.5)
    building.set_node_position('ICU2', 3.0, 4.5)

    # Floor 2 - Bottom row (y=3.5)
    building.set_node_position('ER1', 1.0, 3.5)
    building.set_node_position('ER2', 2.0, 3.5)
    building.set_node_position('ER3', 3.0, 3.5)

    return building


def create_scenario5() -> BuildingGraph:
    """
    Create Scenario 5: L-Shaped Community Center (Single Floor, Non-Rectangular).

    **PART 4 EXTENSION: Non-Standard Building Layout**

    This scenario demonstrates algorithm robustness on irregular floor plans.

    Layout (L-shaped, single floor):

    HORIZONTAL WING (Main corridor, west to east):
    E1 -- O1 -- C1 -- C2 -- C3 -- E2
           |
    VERTICAL WING (extends south):
          O2
           |
          DC (daycare)
           |
          S1 (storage, dead end)

    - 1 floor, 8 rooms, 2 exits (10 nodes total)
    - L-shaped: Horizontal wing + Vertical wing extending south from O1
    - Exit1 on west end, Exit2 on east end
    - Dead-end: Storage S1 at bottom of vertical wing

    Room Types:
    - 4 Classrooms (C1, C2, C3) + Daycare (DC) = 4 occupied rooms
    - 2 Offices (O1, O2)
    - 1 Storage (S1) - dead-end

    Priority Assignments:
    - Daycare (DC): Priority 5 (young children)
    - Classrooms: Priority 3 (standard occupancy)
    - Offices: Priority 2 (low occupancy)
    - Storage: Priority 1 (unoccupied, dead-end)

    Algorithm Challenges:
    - Non-rectangular layout (L-shape)
    - Dead-end routing (S1 requires backtracking)
    - Unbalanced wings (horizontal longer than vertical)

    This demonstrates real-world building complexity beyond idealized grids.
    """
    building = BuildingGraph()

    # Set building features
    building.set_feature('layout_type', 'L-shaped')
    building.set_feature('floors', 1)
    building.set_feature('irregular', True)

    # Create 2 exits (west and east ends)
    building.add_exit('Exit1')  # West end
    building.add_exit('Exit2')  # East end

    # ========================================================================
    # HORIZONTAL WING ROOMS (Main Corridor, west to east)
    # ========================================================================
    office1 = Room('Office1', 'office', 200.0, 2, 'adults', 2)
    office1.set_metadata('wing', 'horizontal')
    building.add_room(office1)

    classroom1 = Room('Classroom1', 'classroom', 400.0, 25, 'adults', 3)
    classroom1.set_metadata('wing', 'horizontal')
    building.add_room(classroom1)

    classroom2 = Room('Classroom2', 'classroom', 400.0, 25, 'adults', 3)
    classroom2.set_metadata('wing', 'horizontal')
    building.add_room(classroom2)

    classroom3 = Room('Classroom3', 'classroom', 400.0, 20, 'adults', 3)
    classroom3.set_metadata('wing', 'horizontal')
    building.add_room(classroom3)

    # ========================================================================
    # VERTICAL WING ROOMS (extends south from Office1)
    # ========================================================================
    office2 = Room('Office2', 'office', 200.0, 2, 'adults', 2)
    office2.set_metadata('wing', 'vertical')
    building.add_room(office2)

    daycare = Room('Daycare', 'daycare', 500.0, 20, 'children', 5)
    daycare.set_metadata('wing', 'vertical')
    daycare.set_metadata('occupant_age', 'toddler')
    building.add_room(daycare)

    storage1 = Room('Storage1', 'storage', 300.0, 0, 'adults', 1)
    storage1.set_metadata('wing', 'vertical')
    storage1.set_metadata('corridor_type', 'dead_end')
    building.add_room(storage1)

    # ========================================================================
    # EDGES - HORIZONTAL WING (Main Corridor)
    # ========================================================================
    building.add_edge(Edge('Office1', 'Classroom1', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Classroom2', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom2', 'Classroom3', 12.0, 'corridor'))

    # ========================================================================
    # EDGES - VERTICAL WING (South Extension)
    # ========================================================================
    building.add_edge(Edge('Office1', 'Office2', 12.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Daycare', 12.0, 'corridor'))
    building.add_edge(Edge('Daycare', 'Storage1', 12.0, 'corridor'))
    # Storage1 is a dead end - no other connections

    # ========================================================================
    # EXIT CONNECTIONS
    # ========================================================================
    # Exit1 (west end) connects to Office1
    building.add_edge(Edge('Exit1', 'Office1', 8.0, 'hallway'))

    # Exit2 (east end) connects to Classroom3
    building.add_edge(Edge('Exit2', 'Classroom3', 8.0, 'hallway'))

    # ========================================================================
    # SET POSITIONS FOR VISUALIZATION (L-shape)
    # ========================================================================
    # Horizontal wing (y=2.0, x increases from west to east)
    building.set_node_position('Exit1', 0.0, 2.0)  # West exit
    building.set_node_position('Office1', 1.0, 2.0)
    building.set_node_position('Classroom1', 2.0, 2.0)
    building.set_node_position('Classroom2', 3.0, 2.0)
    building.set_node_position('Classroom3', 4.0, 2.0)
    building.set_node_position('Exit2', 5.0, 2.0)  # East exit

    # Vertical wing (x=1.0, y decreases going south)
    building.set_node_position('Office2', 1.0, 1.0)
    building.set_node_position('Daycare', 1.0, 0.0)
    building.set_node_position('Storage1', 1.0, -1.0)  # Dead end at bottom

    return building



def create_scenario6() -> BuildingGraph:
    """
    Create Scenario 6: Office Building Fire Emergency (Two Floors with Blocked Room).

    **PART 4 EXTENSION: Emergency Response with Blocked Areas**

    Emergency Context:
        Fire in Lab on Floor 2. Lab is BLOCKED and cannot be entered.
        Responders must work around the blocked room to clear all other rooms.

    Layout:
    FLOOR 1:
         O1 --- C1 --- O2
    E1 --|      |       |-- E2
         O3 --- S1 --- O4

    FLOOR 2 (connected via stairs, Lab is BLOCKED):
         C2 --- LAB(BLOCKED) --- C3
          |                       |
         C4 ------- C5 ---------- C6

    - 2 floors, 12 rooms (1 blocked), 2 exits (14 nodes)
    - Floor 1: 4 offices, 1 classroom, 1 storage
    - Floor 2: 5 classrooms + 1 LAB (BLOCKED, has fire)
    - Lab on Floor 2 is impassable (marked as blocked)
    - All other rooms are accessible and connected

    Priority Levels:
        - Floor 2 classrooms near fire: Priority 5 (C2, C3)
        - Floor 2 other classrooms: Priority 4 (C4, C5, C6)
        - Floor 1 rooms: Priority 3

    This scenario demonstrates:
        - Routing around blocked/impassable rooms
        - Emergency evacuation with hazards
        - Priority-based room assignment in emergencies
        - Algorithm robustness when rooms are inaccessible
    """
    building = BuildingGraph()

    # Set emergency metadata
    building.set_feature('disaster_type', 'fire')
    building.set_feature('fire_source', 'Lab1')
    building.set_feature('fire_floor', 2)

    # Create exits (on floor 1)
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # ========================================================================
    # FLOOR 1 ROOMS (Lower priority - safer area)
    # ========================================================================
    floor1_rooms = [
        Room('Office1', 'office', 200.0, 2, 'adults', 3),
        Room('Classroom1', 'classroom', 400.0, 25, 'adults', 3),
        Room('Office2', 'office', 200.0, 2, 'adults', 3),
        Room('Office3', 'office', 200.0, 2, 'adults', 3),
        Room('Storage1', 'storage', 300.0, 0, 'adults', 2),
        Room('Office4', 'office', 200.0, 2, 'adults', 3),
    ]

    for room in floor1_rooms:
        room.set_metadata('floor', 1)
        building.add_room(room)

    # ========================================================================
    # FLOOR 2 ROOMS (Higher priority - near fire)
    # ========================================================================
    # High priority - next to fire
    floor2_high = [
        Room('Classroom2', 'classroom', 400.0, 25, 'adults', 5),
        Room('Classroom3', 'classroom', 400.0, 25, 'adults', 5),
    ]
    for room in floor2_high:
        room.set_metadata('floor', 2)
        room.set_metadata('near_fire', True)
        building.add_room(room)

    # Lab with FIRE - BLOCKED (cannot enter)
    lab = Room('Lab1', 'lab', 600.0, 0, 'adults', 5)
    lab.set_metadata('floor', 2)
    lab.set_metadata('has_fire', True)
    lab.set_metadata('passable', False)  # BLOCKED
    building.add_room(lab)

    # Medium priority - floor 2 but farther from fire
    floor2_medium = [
        Room('Classroom4', 'classroom', 400.0, 20, 'adults', 4),
        Room('Classroom5', 'classroom', 400.0, 20, 'adults', 4),
        Room('Classroom6', 'classroom', 400.0, 20, 'adults', 4),
    ]
    for room in floor2_medium:
        room.set_metadata('floor', 2)
        building.add_room(room)

    # ========================================================================
    # FLOOR 1 EDGES
    # ========================================================================
    # Top row: O1 --- C1 --- O2
    building.add_edge(Edge('Office1', 'Classroom1', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Office2', 10.0, 'corridor'))

    # Bottom row: O3 --- S1 --- O4
    building.add_edge(Edge('Office3', 'Storage1', 10.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Office4', 10.0, 'corridor'))

    # Vertical connections
    building.add_edge(Edge('Office1', 'Office3', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Storage1', 12.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office4', 12.0, 'corridor'))

    # Cross connections for Floor 1
    building.add_edge(Edge('Office1', 'Storage1', 14.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Storage1', 14.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Office3', 14.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Office4', 14.0, 'corridor'))

    # Exit connections
    building.add_edge(Edge('Exit1', 'Office1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office3', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office2', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office4', 8.0, 'hallway'))

    # ========================================================================
    # FLOOR 2 EDGES
    # ========================================================================
    # Top row: C2 --- Lab1 --- C3
    # NOTE: Lab1 is blocked, but edges exist (responders just can't enter Lab1)
    building.add_edge(Edge('Classroom2', 'Lab1', 10.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Classroom3', 10.0, 'corridor'))

    # Bottom row: C4 --- C5 --- C6
    building.add_edge(Edge('Classroom4', 'Classroom5', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom5', 'Classroom6', 10.0, 'corridor'))

    # Vertical connections
    building.add_edge(Edge('Classroom2', 'Classroom4', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom3', 'Classroom6', 12.0, 'corridor'))

    # Cross connections for redundancy (IMPORTANT - ensures connectivity even with blocked Lab)
    building.add_edge(Edge('Classroom2', 'Classroom5', 14.0, 'corridor'))
    building.add_edge(Edge('Classroom3', 'Classroom5', 14.0, 'corridor'))
    building.add_edge(Edge('Classroom4', 'Classroom6', 14.0, 'corridor'))
    # Lab1 connections to bottom row (even though Lab is blocked, edges help with routing)
    building.add_edge(Edge('Lab1', 'Classroom4', 14.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Classroom5', 12.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Classroom6', 14.0, 'corridor'))

    # ========================================================================
    # STAIR CONNECTIONS (Exits to Floor 2) - Heavy weight (25m)
    # ========================================================================
    building.add_edge(Edge('Exit1', 'Classroom2', 25.0, 'stair'))
    building.add_edge(Edge('Exit1', 'Classroom4', 25.0, 'stair'))
    building.add_edge(Edge('Exit2', 'Classroom3', 25.0, 'stair'))
    building.add_edge(Edge('Exit2', 'Classroom6', 25.0, 'stair'))

    # ========================================================================
    # VISUALIZATION POSITIONS
    # ========================================================================
    # Exits
    building.set_node_position('Exit1', 0.0, 1.0)
    building.set_node_position('Exit2', 4.0, 1.0)

    # Floor 1 - Top row (y=2)
    building.set_node_position('Office1', 1.0, 2.0)
    building.set_node_position('Classroom1', 2.0, 2.0)
    building.set_node_position('Office2', 3.0, 2.0)

    # Floor 1 - Bottom row (y=0)
    building.set_node_position('Office3', 1.0, 0.0)
    building.set_node_position('Storage1', 2.0, 0.0)
    building.set_node_position('Office4', 3.0, 0.0)

    # Floor 2 - Top row (y=4.5)
    building.set_node_position('Classroom2', 1.0, 4.5)
    building.set_node_position('Lab1', 2.0, 4.5)  # BLOCKED (fire)
    building.set_node_position('Classroom3', 3.0, 4.5)

    # Floor 2 - Bottom row (y=3.5)
    building.set_node_position('Classroom4', 1.0, 3.5)
    building.set_node_position('Classroom5', 2.0, 3.5)
    building.set_node_position('Classroom6', 3.0, 3.5)

    return building
