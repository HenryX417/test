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
    Create Scenario 2: Two-Floor Mixed Use.

    Layout (2 floors):
    FLOOR 2:
    [Off2_1][Off2_2][Off2_3][Off2_4]
    [Stor1]                   [Stor2]

    FLOOR 1:
    [Class1][Class2][Class3]
    [Off1_1] [Lab1]  [Off1_2]
      ↕                ↕
     Exit1           Exit2

    - 2 floors, 12 rooms total, 2 exits (14 nodes total)
    - Floor 1: 3 classrooms, 1 lab, 2 offices
    - Floor 2: 4 offices, 2 storage rooms
    - Stairs represented as edges with higher weights
    - NO hallway or junction nodes - direct edges only
    """
    building = BuildingGraph()

    # Create exits (on first floor)
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # Floor 1: 3 classrooms, 1 lab, 2 offices
    floor1_rooms = [
        Room('Classroom1', 'classroom', 400.0, 25, 'adults', 4),
        Room('Classroom2', 'classroom', 400.0, 25, 'adults', 4),
        Room('Classroom3', 'classroom', 400.0, 25, 'adults', 4),
        Room('Lab1', 'lab', 600.0, 15, 'adults', 3),
        Room('Office1_1', 'office', 200.0, 2, 'adults', 3),
        Room('Office1_2', 'office', 200.0, 2, 'adults', 3),
    ]

    for room in floor1_rooms:
        building.add_room(room)

    # Floor 2: 4 offices, 2 storage rooms
    floor2_rooms = [
        Room('Office2_1', 'office', 200.0, 2, 'adults', 3),
        Room('Office2_2', 'office', 200.0, 2, 'adults', 3),
        Room('Office2_3', 'office', 200.0, 2, 'adults', 3),
        Room('Office2_4', 'office', 200.0, 2, 'adults', 3),
        Room('Storage1', 'storage', 300.0, 0, 'adults', 2),
        Room('Storage2', 'storage', 300.0, 0, 'adults', 2),
    ]

    for room in floor2_rooms:
        building.add_room(room)

    # Floor 1: Exit connections
    building.add_edge(Edge('Exit1', 'Classroom1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office1_1', 8.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Classroom2', 15.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Classroom3', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office1_2', 8.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Lab1', 15.0, 'hallway'))

    # Floor 1: Room-to-room connections
    building.add_edge(Edge('Classroom1', 'Classroom2', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom2', 'Classroom3', 12.0, 'corridor'))
    building.add_edge(Edge('Office1_1', 'Lab1', 12.0, 'corridor'))
    building.add_edge(Edge('Lab1', 'Office1_2', 12.0, 'corridor'))
    building.add_edge(Edge('Classroom1', 'Office1_1', 10.0, 'corridor'))
    building.add_edge(Edge('Classroom3', 'Office1_2', 10.0, 'corridor'))

    # Stair connections (Floor 1 to Floor 2) - heavy weight for stairs
    building.add_edge(Edge('Office1_1', 'Storage1', 20.0, 'stair'))
    building.add_edge(Edge('Office1_1', 'Office2_1', 20.0, 'stair'))
    building.add_edge(Edge('Office1_2', 'Storage2', 20.0, 'stair'))
    building.add_edge(Edge('Office1_2', 'Office2_4', 20.0, 'stair'))

    # Floor 2: Room-to-room connections
    building.add_edge(Edge('Office2_1', 'Office2_2', 10.0, 'corridor'))
    building.add_edge(Edge('Office2_2', 'Office2_3', 10.0, 'corridor'))
    building.add_edge(Edge('Office2_3', 'Office2_4', 10.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Office2_1', 8.0, 'corridor'))
    building.add_edge(Edge('Storage2', 'Office2_4', 8.0, 'corridor'))
    building.add_edge(Edge('Storage1', 'Office2_2', 15.0, 'corridor'))
    building.add_edge(Edge('Storage2', 'Office2_3', 15.0, 'corridor'))

    # Set explicit positions for visualization
    # Exits
    building.set_node_position('Exit1', 0, 0)
    building.set_node_position('Exit2', 6, 0)

    # Floor 1 rooms (y=0 to 1)
    building.set_node_position('Classroom1', 1, 1)
    building.set_node_position('Classroom2', 3, 1)
    building.set_node_position('Classroom3', 5, 1)
    building.set_node_position('Office1_1', 1, -1)
    building.set_node_position('Lab1', 3, -1)
    building.set_node_position('Office1_2', 5, -1)

    # Floor 2 rooms (y=2.5 to 3.5)
    building.set_node_position('Storage1', 1, 2.5)
    building.set_node_position('Office2_1', 1.5, 3.5)
    building.set_node_position('Office2_2', 2.5, 3.5)
    building.set_node_position('Office2_3', 3.5, 3.5)
    building.set_node_position('Office2_4', 4.5, 3.5)
    building.set_node_position('Storage2', 5, 2.5)

    return building


def create_scenario3() -> BuildingGraph:
    """
    Create Scenario 3: Single-Floor High-Density Office.

    Layout (grid):
    Exit1 --- [Off1][Off2][Off3] --- Exit2
               |     |     |
              [Off4][Off5][Off6]
               |     |     |
    Exit3 --- [Off7][Off8][Off9] --- Exit4
                     |
                  [Off10]

    - 1 floor, 10 offices, 4 exits (14 nodes total)
    - All offices, highly interconnected grid layout
    - Demonstrates sensitivity to constraints
    - NO hallway or junction nodes - direct edges only
    """
    building = BuildingGraph()

    # Create 4 exits
    building.add_exit('Exit1')
    building.add_exit('Exit2')
    building.add_exit('Exit3')
    building.add_exit('Exit4')

    # Create 10 offices with varied sizes
    offices = [
        Room('Office1', 'office', 180.0, 2, 'adults', 3),
        Room('Office2', 'office', 200.0, 2, 'adults', 3),
        Room('Office3', 'office', 220.0, 3, 'adults', 3),
        Room('Office4', 'office', 190.0, 2, 'adults', 3),
        Room('Office5', 'office', 250.0, 3, 'adults', 4),  # Larger, higher priority
        Room('Office6', 'office', 200.0, 2, 'adults', 3),
        Room('Office7', 'office', 180.0, 2, 'adults', 3),
        Room('Office8', 'office', 210.0, 2, 'adults', 3),
        Room('Office9', 'office', 200.0, 2, 'adults', 3),
        Room('Office10', 'office', 150.0, 1, 'adults', 2),
    ]

    for office in offices:
        building.add_room(office)

    # Exit connections - direct to nearby offices
    building.add_edge(Edge('Exit1', 'Office1', 7.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office4', 10.0, 'hallway'))
    building.add_edge(Edge('Exit1', 'Office7', 14.0, 'hallway'))

    building.add_edge(Edge('Exit2', 'Office3', 7.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office6', 10.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Office9', 14.0, 'hallway'))

    building.add_edge(Edge('Exit3', 'Office7', 7.0, 'hallway'))
    building.add_edge(Edge('Exit3', 'Office4', 14.0, 'hallway'))
    building.add_edge(Edge('Exit3', 'Office1', 18.0, 'hallway'))

    building.add_edge(Edge('Exit4', 'Office9', 7.0, 'hallway'))
    building.add_edge(Edge('Exit4', 'Office6', 14.0, 'hallway'))
    building.add_edge(Edge('Exit4', 'Office3', 18.0, 'hallway'))

    # Horizontal connections between adjacent offices (10m each)
    building.add_edge(Edge('Office1', 'Office2', 10.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office3', 10.0, 'corridor'))
    building.add_edge(Edge('Office4', 'Office5', 10.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office6', 10.0, 'corridor'))
    building.add_edge(Edge('Office7', 'Office8', 10.0, 'corridor'))
    building.add_edge(Edge('Office8', 'Office9', 10.0, 'corridor'))

    # Vertical connections between offices (10m each)
    building.add_edge(Edge('Office1', 'Office4', 10.0, 'corridor'))
    building.add_edge(Edge('Office4', 'Office7', 10.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office5', 10.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office8', 10.0, 'corridor'))
    building.add_edge(Edge('Office3', 'Office6', 10.0, 'corridor'))
    building.add_edge(Edge('Office6', 'Office9', 10.0, 'corridor'))

    # Office 10 connections (southern extension below Office8)
    building.add_edge(Edge('Office8', 'Office10', 8.0, 'corridor'))
    building.add_edge(Edge('Office7', 'Office10', 12.0, 'corridor'))
    building.add_edge(Edge('Office9', 'Office10', 12.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office10', 12.0, 'corridor'))

    # Diagonal connections for high interconnectivity
    building.add_edge(Edge('Office1', 'Office5', 14.0, 'corridor'))
    building.add_edge(Edge('Office3', 'Office5', 14.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office7', 14.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office9', 14.0, 'corridor'))

    # Set explicit positions for visualization (grid layout)
    # Exits at corners
    building.set_node_position('Exit1', 0, 2)
    building.set_node_position('Exit2', 4, 2)
    building.set_node_position('Exit3', 0, 0)
    building.set_node_position('Exit4', 4, 0)

    # Top row offices (y=2)
    building.set_node_position('Office1', 1, 2)
    building.set_node_position('Office2', 2, 2)
    building.set_node_position('Office3', 3, 2)

    # Middle row offices (y=1)
    building.set_node_position('Office4', 1, 1)
    building.set_node_position('Office5', 2, 1)
    building.set_node_position('Office6', 3, 1)

    # Bottom row offices (y=0)
    building.set_node_position('Office7', 1, 0)
    building.set_node_position('Office8', 2, 0)
    building.set_node_position('Office9', 3, 0)

    # Office 10 at bottom center (southern extension)
    building.set_node_position('Office10', 2, -0.8)

    return building
