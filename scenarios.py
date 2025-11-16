"""
Building scenario definitions for evacuation simulations.

This module contains predefined building layouts for testing
the evacuation sweep optimization algorithms.
"""

from building import BuildingGraph, Room, Edge


def create_scenario1() -> BuildingGraph:
    """
    Create Scenario 1: Basic Single Floor.

    Layout:
         Exit1
          |
    [Off1][Off2][Off3]
          |
        Hallway
          |
    [Off4][Off5][Off6]
          |
         Exit2

    - 1 floor, 6 offices, 2 exits (opposite ends)
    - Rooms: All standard offices (200 sq ft each)
    - Edges: 10m between adjacent rooms, 5m room-to-hallway
    """
    building = BuildingGraph()

    # Create exits
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # Create central hallway nodes
    building.add_exit('Hall_North')
    building.add_exit('Hall_Center')
    building.add_exit('Hall_South')

    # Create 6 offices
    offices = []
    for i in range(1, 7):
        room = Room(
            id=f'Office{i}',
            type='office',
            size=200.0,
            occupant_count=2,
            occupant_type='adults',
            priority=3
        )
        offices.append(room)
        building.add_room(room)

    # Connect Exit1 to Hall_North
    building.add_edge(Edge('Exit1', 'Hall_North', 5.0, 'hallway'))

    # Connect offices 1-3 to Hall_North
    building.add_edge(Edge('Hall_North', 'Office1', 5.0, 'hallway'))
    building.add_edge(Edge('Hall_North', 'Office2', 5.0, 'hallway'))
    building.add_edge(Edge('Hall_North', 'Office3', 5.0, 'hallway'))

    # Connect adjacent offices
    building.add_edge(Edge('Office1', 'Office2', 10.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office3', 10.0, 'corridor'))

    # Connect Hall_North to Hall_Center
    building.add_edge(Edge('Hall_North', 'Hall_Center', 8.0, 'hallway'))

    # Connect Hall_Center to Hall_South
    building.add_edge(Edge('Hall_Center', 'Hall_South', 8.0, 'hallway'))

    # Connect offices 4-6 to Hall_South
    building.add_edge(Edge('Hall_South', 'Office4', 5.0, 'hallway'))
    building.add_edge(Edge('Hall_South', 'Office5', 5.0, 'hallway'))
    building.add_edge(Edge('Hall_South', 'Office6', 5.0, 'hallway'))

    # Connect adjacent offices
    building.add_edge(Edge('Office4', 'Office5', 10.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office6', 10.0, 'corridor'))

    # Connect Exit2 to Hall_South
    building.add_edge(Edge('Hall_South', 'Exit2', 5.0, 'hallway'))

    return building


def create_scenario2() -> BuildingGraph:
    """
    Create Scenario 2: Two-Floor Mixed Use.

    - 2 floors, 12 rooms total, 2 exits (either end of first floor)
    - Floor 1: 3 classrooms, 1 lab, 2 offices
    - Floor 2: 4 offices, 2 storage rooms
    - Includes stairs connecting floors
    - More complex connectivity
    """
    building = BuildingGraph()

    # Create exits (on first floor)
    building.add_exit('Exit1')
    building.add_exit('Exit2')

    # Create hallway nodes
    building.add_exit('Hall1_West')
    building.add_exit('Hall1_Center')
    building.add_exit('Hall1_East')
    building.add_exit('Hall2_West')
    building.add_exit('Hall2_Center')
    building.add_exit('Hall2_East')

    # Stairs connecting floors
    building.add_exit('Stairs_West')
    building.add_exit('Stairs_East')

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

    # Floor 1 connections
    building.add_edge(Edge('Exit1', 'Hall1_West', 3.0, 'hallway'))
    building.add_edge(Edge('Hall1_West', 'Classroom1', 5.0, 'hallway'))
    building.add_edge(Edge('Hall1_West', 'Office1_1', 5.0, 'hallway'))
    building.add_edge(Edge('Hall1_West', 'Hall1_Center', 10.0, 'hallway'))

    building.add_edge(Edge('Hall1_Center', 'Classroom2', 5.0, 'hallway'))
    building.add_edge(Edge('Hall1_Center', 'Lab1', 5.0, 'hallway'))
    building.add_edge(Edge('Hall1_Center', 'Hall1_East', 10.0, 'hallway'))

    building.add_edge(Edge('Hall1_East', 'Classroom3', 5.0, 'hallway'))
    building.add_edge(Edge('Hall1_East', 'Office1_2', 5.0, 'hallway'))
    building.add_edge(Edge('Hall1_East', 'Exit2', 3.0, 'hallway'))

    # Stairs connections (Floor 1)
    building.add_edge(Edge('Hall1_West', 'Stairs_West', 4.0, 'hallway'))
    building.add_edge(Edge('Hall1_East', 'Stairs_East', 4.0, 'hallway'))

    # Stairs to Floor 2 (heavier weight)
    building.add_edge(Edge('Stairs_West', 'Hall2_West', 15.0, 'stair'))
    building.add_edge(Edge('Stairs_East', 'Hall2_East', 15.0, 'stair'))

    # Floor 2 connections
    building.add_edge(Edge('Hall2_West', 'Office2_1', 5.0, 'hallway'))
    building.add_edge(Edge('Hall2_West', 'Storage1', 5.0, 'hallway'))
    building.add_edge(Edge('Hall2_West', 'Hall2_Center', 10.0, 'hallway'))

    building.add_edge(Edge('Hall2_Center', 'Office2_2', 5.0, 'hallway'))
    building.add_edge(Edge('Hall2_Center', 'Office2_3', 5.0, 'hallway'))
    building.add_edge(Edge('Hall2_Center', 'Hall2_East', 10.0, 'hallway'))

    building.add_edge(Edge('Hall2_East', 'Office2_4', 5.0, 'hallway'))
    building.add_edge(Edge('Hall2_East', 'Storage2', 5.0, 'hallway'))

    # Cross connections for flexibility
    building.add_edge(Edge('Office2_1', 'Office2_2', 12.0, 'corridor'))
    building.add_edge(Edge('Office2_3', 'Office2_4', 12.0, 'corridor'))

    return building


def create_scenario3() -> BuildingGraph:
    """
    Create Scenario 3: Single-Floor High-Density Office.

    - 1 floor, 10 offices, 4 exits
    - All offices, highly interconnected grid layout
    - Demonstrates sensitivity to constraints
    - More exits allow for better optimization

    Layout (grid-like):
    Exit1 --- [Off1][Off2][Off3] --- Exit2
               |     |     |
              [Off4][Off5][Off6]
               |     |     |
    Exit3 --- [Off7][Off8][Off9] --- Exit4
                     |
                  [Off10]
    """
    building = BuildingGraph()

    # Create 4 exits
    building.add_exit('Exit1')
    building.add_exit('Exit2')
    building.add_exit('Exit3')
    building.add_exit('Exit4')

    # Create hallway junctions
    building.add_exit('Junction_NW')
    building.add_exit('Junction_NE')
    building.add_exit('Junction_SW')
    building.add_exit('Junction_SE')
    building.add_exit('Junction_Center')

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

    # Connect exits to junctions
    building.add_edge(Edge('Exit1', 'Junction_NW', 3.0, 'hallway'))
    building.add_edge(Edge('Exit2', 'Junction_NE', 3.0, 'hallway'))
    building.add_edge(Edge('Exit3', 'Junction_SW', 3.0, 'hallway'))
    building.add_edge(Edge('Exit4', 'Junction_SE', 3.0, 'hallway'))

    # Top row (Offices 1-3)
    building.add_edge(Edge('Junction_NW', 'Office1', 4.0, 'hallway'))
    building.add_edge(Edge('Office1', 'Office2', 8.0, 'corridor'))
    building.add_edge(Edge('Office2', 'Office3', 8.0, 'corridor'))
    building.add_edge(Edge('Office3', 'Junction_NE', 4.0, 'hallway'))

    # Middle row (Offices 4-6)
    building.add_edge(Edge('Junction_NW', 'Office4', 6.0, 'hallway'))
    building.add_edge(Edge('Office4', 'Office5', 8.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office6', 8.0, 'corridor'))
    building.add_edge(Edge('Office6', 'Junction_NE', 6.0, 'hallway'))

    # Connect middle row to center junction
    building.add_edge(Edge('Office5', 'Junction_Center', 5.0, 'hallway'))

    # Bottom row (Offices 7-9)
    building.add_edge(Edge('Junction_SW', 'Office7', 4.0, 'hallway'))
    building.add_edge(Edge('Office7', 'Office8', 8.0, 'corridor'))
    building.add_edge(Edge('Office8', 'Office9', 8.0, 'corridor'))
    building.add_edge(Edge('Office9', 'Junction_SE', 4.0, 'hallway'))

    # Office 10 at bottom center
    building.add_edge(Edge('Junction_Center', 'Office10', 5.0, 'hallway'))
    building.add_edge(Edge('Office10', 'Office8', 6.0, 'corridor'))

    # Vertical connections
    building.add_edge(Edge('Office1', 'Office4', 10.0, 'corridor'))
    building.add_edge(Edge('Office4', 'Office7', 10.0, 'corridor'))

    building.add_edge(Edge('Office2', 'Office5', 10.0, 'corridor'))
    building.add_edge(Edge('Office5', 'Office8', 10.0, 'corridor'))

    building.add_edge(Edge('Office3', 'Office6', 10.0, 'corridor'))
    building.add_edge(Edge('Office6', 'Office9', 10.0, 'corridor'))

    # Additional cross-connections for high interconnectivity
    building.add_edge(Edge('Junction_NW', 'Junction_NE', 15.0, 'hallway'))
    building.add_edge(Edge('Junction_SW', 'Junction_SE', 15.0, 'hallway'))
    building.add_edge(Edge('Junction_NW', 'Junction_SW', 15.0, 'hallway'))
    building.add_edge(Edge('Junction_NE', 'Junction_SE', 15.0, 'hallway'))

    return building
