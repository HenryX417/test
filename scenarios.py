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

    # Floor 1: Vertical connections (offices to classrooms below)
    building.add_edge(Edge('Office1', 'Classroom1', 12.0, 'corridor'))
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
