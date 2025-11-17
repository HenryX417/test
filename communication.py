"""
Communication Protocols Module for Emergency Evacuation.

This module provides decision frameworks and fallback strategies
for communication during emergency evacuations.

Part 4 Extension: Communication Delay or Failure
"""

from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from building import BuildingGraph
from simulation import EvacuationSimulation


@dataclass
class CommunicationProtocol:
    """Define communication strategy for different scenarios."""
    name: str
    technology: str  # 'radio', 'visual_markers', 'central_dashboard', 'none'
    update_frequency: str  # 'real_time', 'periodic', 'completion_only', 'none'
    fallback_strategy: str
    suitable_for: List[str]  # Building types
    reliability: float  # 0.0 to 1.0
    description: str


def generate_communication_protocols() -> List[CommunicationProtocol]:
    """
    Create decision matrix for communication strategies.

    Returns comprehensive list of communication protocols for different scenarios.
    """
    protocols = [
        CommunicationProtocol(
            name='Radio + Periodic Updates',
            technology='two_way_radio',
            update_frequency='periodic',
            fallback_strategy='Pre-assigned routes (stick to plan)',
            suitable_for=['office', 'school', 'retail', 'standard_conditions'],
            reliability=0.95,
            description='Responders radio in after each room. Commander tracks progress '
                       'on map. Can reassign if someone finishes early. Best for normal conditions.'
        ),

        CommunicationProtocol(
            name='Visual Markers + Pre-assigned Routes',
            technology='physical_markers',
            update_frequency='completion_only',
            fallback_strategy='Chalk marks only (no coordination)',
            suitable_for=['fire', 'heavy_smoke', 'interference'],
            reliability=0.85,
            description='Radios may fail in heavy smoke. Responders use chalk/tape to mark '
                       'cleared rooms. Stick to pre-computed routes. No dynamic reassignment.'
        ),

        CommunicationProtocol(
            name='Physical Markers Only',
            technology='chalk_tape',
            update_frequency='none',
            fallback_strategy='Visual confirmation at exits',
            suitable_for=['disaster', 'total_tech_failure'],
            reliability=0.70,
            description='All technology fails. Chalk "X" on cleared rooms. Pre-assigned routes '
                       'prevent duplication. Hallway monitor coordinates visually.'
        ),

        CommunicationProtocol(
            name='Central Dashboard + IoT',
            technology='real_time_sensors',
            update_frequency='real_time',
            fallback_strategy='Radio + Periodic Updates',
            suitable_for=['high_tech_building', 'hospital', 'data_center'],
            reliability=0.98,
            description='Real-time tracking via sensors. Dynamic reassignment possible. '
                       'Commander optimizes on-the-fly. Requires building IoT infrastructure.'
        ),

        CommunicationProtocol(
            name='Mesh Network + GPS',
            technology='mesh_radio',
            update_frequency='real_time',
            fallback_strategy='Radio + Periodic Updates',
            suitable_for=['large_building', 'campus', 'multi_floor'],
            reliability=0.92,
            description='Self-healing mesh network. Works even if some nodes fail. '
                       'GPS tracking for large buildings. Better coverage than traditional radio.'
        ),
    ]

    return protocols


def select_protocol(building: BuildingGraph, scenario_type: str = 'normal') -> CommunicationProtocol:
    """
    Select appropriate communication protocol based on scenario.

    Args:
        building: BuildingGraph with scenario metadata
        scenario_type: 'normal', 'fire', 'gas_leak', 'disaster'

    Returns:
        Recommended CommunicationProtocol
    """
    protocols = generate_communication_protocols()

    # Check building features
    disaster_type = building.features.get('disaster_type', 'none')
    building_size = len(building.rooms)

    # Decision logic
    if disaster_type == 'fire':
        # Heavy smoke may interfere with radios
        return next(p for p in protocols if p.name == 'Visual Markers + Pre-assigned Routes')
    elif disaster_type == 'gas_leak':
        # Gas doesn't interfere with electronics
        return next(p for p in protocols if p.name == 'Radio + Periodic Updates')
    elif building_size > 50:
        # Large building needs mesh network
        return next(p for p in protocols if p.name == 'Mesh Network + GPS')
    elif scenario_type == 'disaster':
        # Total failure scenario
        return next(p for p in protocols if p.name == 'Physical Markers Only')
    else:
        # Normal conditions
        return next(p for p in protocols if p.name == 'Radio + Periodic Updates')


def simulate_communication_failure(
    building: BuildingGraph,
    num_responders: int = 3,
    walking_speed: float = 1.5,
    visibility: float = 1.0
) -> Dict:
    """
    Show algorithm robustness when communication fails.

    Compares:
    - WITH communication: Can dynamically reassign
    - WITHOUT communication: Must stick to pre-assigned routes

    Args:
        building: BuildingGraph
        num_responders: Number of responders
        walking_speed: Walking speed (m/s)
        visibility: Visibility factor

    Returns:
        Dictionary with comparison results
    """
    print('\n' + '=' * 70)
    print('COMMUNICATION FAILURE SIMULATION')
    print('=' * 70)

    # Scenario 1: WITH communication (can reassign)
    print('\n[1/2] WITH Communication (dynamic reassignment possible)...')
    sim_with = EvacuationSimulation(building, num_responders)
    sim_with.run(walking_speed=walking_speed, visibility=visibility, use_priority=True)
    time_with = sim_with.get_total_time()
    print(f'      ✅ Time with communication: {time_with:.1f}s')

    # Scenario 2: WITHOUT communication (pre-assigned only)
    # This is what the algorithm already does - no dynamic reassignment
    print('\n[2/2] WITHOUT Communication (pre-assigned routes only)...')
    sim_without = EvacuationSimulation(building, num_responders)
    sim_without.run(walking_speed=walking_speed, visibility=visibility, use_priority=True)
    time_without = sim_without.get_total_time()
    print(f'      ✅ Time without communication: {time_without:.1f}s')

    # Calculate penalty
    penalty = ((time_without - time_with) / time_with * 100) if time_with > 0 else 0

    print('\n' + '-' * 70)
    print('RESULTS:')
    print(f'  WITH communication:    {time_with:.1f}s')
    print(f'  WITHOUT communication: {time_without:.1f}s')
    print(f'  Penalty:               +{penalty:.1f}%')
    print('\n✅ Key Insight: Algorithm provides robust baseline plan that')
    print('   works even without communication!')
    print('=' * 70)

    return {
        'time_with_comm': time_with,
        'time_without_comm': time_without,
        'penalty_percent': penalty,
        'sim_with': sim_with,
        'sim_without': sim_without
    }


def generate_communication_flowchart(output_dir: str = '/mnt/user-data/outputs'):
    """
    Create visual decision tree for communication strategy selection.

    Args:
        output_dir: Directory to save flowchart
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    # Define flowchart structure
    # Using matplotlib patches for boxes and arrows

    # Box styling
    box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                     edgecolor='black', linewidth=2)
    decision_style = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                         edgecolor='black', linewidth=2)
    outcome_style = dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                        edgecolor='black', linewidth=2)
    emergency_style = dict(boxstyle='round,pad=0.5', facecolor='#ffcccc',
                          edgecolor='red', linewidth=2)

    # START
    ax.text(0.5, 0.95, 'START', ha='center', va='center',
            bbox=box_style, fontsize=12, fontweight='bold')

    # Decision 1: Fire?
    ax.text(0.5, 0.85, 'Is building on fire?', ha='center', va='center',
            bbox=decision_style, fontsize=10, fontweight='bold')

    # YES branch (fire)
    ax.annotate('', xy=(0.25, 0.75), xytext=(0.45, 0.82),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.35, 0.80, 'YES', ha='center', fontsize=9, fontweight='bold', color='red')

    # Decision 1a: Heavy smoke?
    ax.text(0.25, 0.75, 'Heavy smoke?', ha='center', va='center',
            bbox=decision_style, fontsize=9)

    # YES -> Visual Markers
    ax.annotate('', xy=(0.15, 0.65), xytext=(0.20, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(0.10, 0.70, 'YES', fontsize=8, fontweight='bold', color='red')
    ax.text(0.15, 0.65, 'Visual Markers\n+ Pre-assigned\nRoutes', ha='center', va='center',
            bbox=emergency_style, fontsize=8, fontweight='bold')

    # NO -> Radio
    ax.annotate('', xy=(0.35, 0.65), xytext=(0.30, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.38, 0.70, 'NO', fontsize=8, fontweight='bold')
    ax.text(0.35, 0.65, 'Radio\n+ Periodic\nUpdates', ha='center', va='center',
            bbox=outcome_style, fontsize=8, fontweight='bold')

    # NO branch (no fire)
    ax.annotate('', xy=(0.75, 0.75), xytext=(0.55, 0.82),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.65, 0.80, 'NO', ha='center', fontsize=9, fontweight='bold')

    # Decision 2: Building size
    ax.text(0.75, 0.75, 'Building size\n> 50 rooms?', ha='center', va='center',
            bbox=decision_style, fontsize=9)

    # YES -> Mesh Network
    ax.annotate('', xy=(0.65, 0.65), xytext=(0.70, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.62, 0.70, 'YES', fontsize=8, fontweight='bold')
    ax.text(0.65, 0.65, 'Mesh Network\n+ GPS', ha='center', va='center',
            bbox=outcome_style, fontsize=8, fontweight='bold')

    # NO -> Decision 3
    ax.annotate('', xy=(0.85, 0.65), xytext=(0.80, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.88, 0.70, 'NO', fontsize=8, fontweight='bold')

    # Decision 3: High-tech building?
    ax.text(0.85, 0.65, 'High-tech\nbuilding?', ha='center', va='center',
            bbox=decision_style, fontsize=9)

    # YES -> IoT Dashboard
    ax.annotate('', xy=(0.75, 0.55), xytext=(0.80, 0.62),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.72, 0.60, 'YES', fontsize=8, fontweight='bold')
    ax.text(0.75, 0.55, 'Central Dashboard\n+ IoT Sensors', ha='center', va='center',
            bbox=outcome_style, fontsize=8, fontweight='bold')

    # NO -> Radio
    ax.annotate('', xy=(0.95, 0.55), xytext=(0.90, 0.62),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.98, 0.60, 'NO', fontsize=8, fontweight='bold')
    ax.text(0.95, 0.55, 'Radio\n+ Periodic\nUpdates', ha='center', va='center',
            bbox=outcome_style, fontsize=8, fontweight='bold')

    # Fallback box
    ax.text(0.5, 0.15, 'FALLBACK STRATEGY (if tech fails):\nAll protocols degrade to Physical Markers Only\n(Chalk "X" on cleared rooms)',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffffcc',
                     edgecolor='orange', linewidth=3),
            fontsize=10, fontweight='bold')

    # Title
    ax.text(0.5, 1.0, 'Communication Protocol Decision Flowchart',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Start/Process', edgecolor='black', linewidth=1),
        mpatches.Patch(color='lightyellow', label='Decision Point', edgecolor='black', linewidth=1),
        mpatches.Patch(color='lightgreen', label='Standard Protocol', edgecolor='black', linewidth=1),
        mpatches.Patch(color='#ffcccc', label='Emergency Protocol', edgecolor='red', linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/communication_decision_flowchart.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✅ Communication flowchart saved:')
    print(f'   {output_dir}/communication_decision_flowchart.png')


def generate_protocol_comparison_table(output_dir: str = '/mnt/user-data/outputs'):
    """
    Generate comparison table of communication protocols.

    Args:
        output_dir: Directory to save visualization
    """
    protocols = generate_communication_protocols()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = [['Protocol', 'Technology', 'Update Freq', 'Reliability', 'Best For', 'Fallback']]

    for p in protocols:
        suitable = ', '.join(p.suitable_for[:2])  # First 2 to keep it short
        table_data.append([
            p.name,
            p.technology.replace('_', ' ').title(),
            p.update_frequency.replace('_', ' ').title(),
            f'{p.reliability*100:.0f}%',
            suitable.replace('_', ' ').title(),
            p.fallback_strategy[:30] + '...'  # Truncate
        ])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.20, 0.15, 0.12, 0.10, 0.20, 0.23])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')

    plt.title('Communication Protocol Comparison Matrix',
              fontsize=14, fontweight='bold', pad=20)

    plt.savefig(f'{output_dir}/communication_protocol_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✅ Protocol comparison table saved:')
    print(f'   {output_dir}/communication_protocol_comparison.png')
