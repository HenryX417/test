"""
Technology Integration Framework for Emergency Evacuation Systems.

This module provides recommendations for technology integration
to enhance evacuation sweep efficiency (HiMCM Part 4 Extension).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List


class Technology:
    """Represents a technology option for integration."""

    def __init__(self, name: str, category: str, cost: int, effectiveness: int,
                 implementation_time: int, description: str):
        """
        Initialize technology option.

        Args:
            name: Technology name
            category: Category (tracking, communication, sensing, automation)
            cost: Implementation cost (1-10 scale)
            effectiveness: Effectiveness improvement (1-10 scale)
            implementation_time: Time to implement in months
            description: Brief description
        """
        self.name = name
        self.category = category
        self.cost = cost
        self.effectiveness = effectiveness
        self.implementation_time = implementation_time
        self.description = description
        self.roi = effectiveness / cost  # Return on investment metric


# Technology catalog for evacuation systems
TECHNOLOGY_CATALOG = [
    # Location Tracking Technologies
    Technology(
        "RFID Badge System",
        "tracking",
        cost=6,
        effectiveness=8,
        implementation_time=3,
        description="Real-time occupant location tracking via RFID badges"
    ),
    Technology(
        "Mobile App GPS",
        "tracking",
        cost=4,
        effectiveness=7,
        implementation_time=2,
        description="Smartphone-based GPS tracking for responders and occupants"
    ),
    Technology(
        "Indoor Positioning (BLE)",
        "tracking",
        cost=7,
        effectiveness=9,
        implementation_time=4,
        description="Bluetooth Low Energy beacons for precise indoor positioning"
    ),

    # Communication Systems
    Technology(
        "Two-Way Radio Network",
        "communication",
        cost=5,
        effectiveness=8,
        implementation_time=2,
        description="Dedicated radio network for responder coordination"
    ),
    Technology(
        "Emergency Alert App",
        "communication",
        cost=3,
        effectiveness=6,
        implementation_time=1,
        description="Push notifications and real-time updates to mobile devices"
    ),
    Technology(
        "Public Address Integration",
        "communication",
        cost=4,
        effectiveness=7,
        implementation_time=2,
        description="Automated PA system for evacuation instructions"
    ),

    # Environmental Sensors
    Technology(
        "Smoke Detector Network",
        "sensing",
        cost=5,
        effectiveness=9,
        implementation_time=3,
        description="Networked smoke detectors with real-time alerts"
    ),
    Technology(
        "Thermal Imaging Cameras",
        "sensing",
        cost=8,
        effectiveness=8,
        implementation_time=3,
        description="Heat detection cameras for fire location and intensity"
    ),
    Technology(
        "Air Quality Sensors",
        "sensing",
        cost=6,
        effectiveness=7,
        implementation_time=2,
        description="CO2 and toxic gas sensors for hazard assessment"
    ),

    # Building Automation
    Technology(
        "Smart Door Control",
        "automation",
        cost=7,
        effectiveness=8,
        implementation_time=4,
        description="Automated door locking/unlocking for safe egress paths"
    ),
    Technology(
        "Emergency Lighting System",
        "automation",
        cost=5,
        effectiveness=7,
        implementation_time=2,
        description="Automated emergency lighting and exit sign illumination"
    ),
    Technology(
        "HVAC Smoke Control",
        "automation",
        cost=8,
        effectiveness=9,
        implementation_time=5,
        description="Automated HVAC adjustments to reduce smoke spread"
    ),
]


def plot_technology_matrix(
    technologies: List[Technology] = TECHNOLOGY_CATALOG,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Create technology comparison matrix showing cost vs effectiveness.

    Args:
        technologies: List of Technology objects
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Category colors
    category_colors = {
        'tracking': '#3498db',      # Blue
        'communication': '#27ae60',  # Green
        'sensing': '#e74c3c',       # Red
        'automation': '#f39c12',    # Orange
    }

    # Plot each technology as a bubble
    for tech in technologies:
        color = category_colors.get(tech.category, '#95a5a6')

        # Bubble size based on ROI
        bubble_size = tech.roi * 400

        ax.scatter(tech.cost, tech.effectiveness, s=bubble_size,
                  c=color, alpha=0.6, edgecolors='black', linewidth=2)

        # Add technology name label
        ax.annotate(tech.name, (tech.cost, tech.effectiveness),
                   fontsize=9, ha='center', va='center', fontweight='bold')

    # Add quadrant lines
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=5.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Add quadrant labels
    ax.text(2.5, 9.5, 'High Value\n(Low Cost, High Effect)', fontsize=11,
           ha='center', va='top', style='italic', alpha=0.5, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.text(8.5, 9.5, 'Premium\n(High Cost, High Effect)', fontsize=11,
           ha='center', va='top', style='italic', alpha=0.5, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    ax.text(2.5, 1.5, 'Low Priority\n(Low Cost, Low Effect)', fontsize=11,
           ha='center', va='bottom', style='italic', alpha=0.5, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    ax.text(8.5, 1.5, 'Avoid\n(High Cost, Low Effect)', fontsize=11,
           ha='center', va='bottom', style='italic', alpha=0.5, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    # Formatting
    ax.set_xlabel('Implementation Cost (1=Low, 10=High)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effectiveness (1=Low, 10=High)', fontsize=12, fontweight='bold')
    ax.set_title('Technology Integration Matrix: Cost vs Effectiveness\n(Bubble size = Return on Investment)',
                fontsize=14, fontweight='bold')

    ax.set_xlim([0, 11])
    ax.set_ylim([0, 11])
    ax.grid(alpha=0.3)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=category_colors['tracking'], label='Location Tracking'),
        mpatches.Patch(color=category_colors['communication'], label='Communication'),
        mpatches.Patch(color=category_colors['sensing'], label='Environmental Sensing'),
        mpatches.Patch(color=category_colors['automation'], label='Building Automation'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/technology_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Technology matrix saved to {output_dir}/technology_matrix.png')


def plot_technology_roadmap(
    technologies: List[Technology] = TECHNOLOGY_CATALOG,
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Create implementation roadmap showing recommended deployment timeline.

    Args:
        technologies: List of Technology objects
        output_dir: Directory to save output
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort technologies by ROI (highest first)
    sorted_techs = sorted(technologies, key=lambda t: t.roi, reverse=True)

    # Category colors
    category_colors = {
        'tracking': '#3498db',
        'communication': '#27ae60',
        'sensing': '#e74c3c',
        'automation': '#f39c12',
    }

    # Create timeline
    current_month = 0
    y_position = len(sorted_techs)

    for tech in sorted_techs:
        color = category_colors.get(tech.category, '#95a5a6')

        # Draw implementation bar
        ax.barh(y_position, tech.implementation_time, left=current_month,
               height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add technology label
        ax.text(current_month + tech.implementation_time / 2, y_position,
               f'{tech.name}\n({tech.implementation_time}mo)',
               fontsize=9, ha='center', va='center', fontweight='bold')

        # Add ROI indicator
        roi_text = f'ROI: {tech.roi:.1f}'
        ax.text(current_month - 0.5, y_position, roi_text,
               fontsize=8, ha='right', va='center', style='italic')

        y_position -= 1
        current_month += tech.implementation_time

    # Formatting
    ax.set_xlabel('Timeline (Months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Technology (Sorted by ROI)', fontsize=12, fontweight='bold')
    ax.set_title('Recommended Technology Implementation Roadmap\n(Ordered by Return on Investment)',
                fontsize=14, fontweight='bold')

    ax.set_yticks([])
    ax.set_xlim([0, current_month + 2])
    ax.grid(axis='x', alpha=0.3)

    # Add phase markers
    phase_markers = [0, 6, 12, 24]
    phase_labels = ['Start', '6mo\n(Phase 1)', '1yr\n(Phase 2)', '2yr\n(Complete)']

    for marker, label in zip(phase_markers, phase_labels):
        if marker <= current_month:
            ax.axvline(x=marker, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(marker, len(sorted_techs) + 1, label,
                   fontsize=10, ha='center', fontweight='bold', color='red')

    # Add legend
    legend_elements = [
        mpatches.Patch(color=category_colors['tracking'], label='Tracking'),
        mpatches.Patch(color=category_colors['communication'], label='Communication'),
        mpatches.Patch(color=category_colors['sensing'], label='Sensing'),
        mpatches.Patch(color=category_colors['automation'], label='Automation'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/technology_roadmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Technology roadmap saved to {output_dir}/technology_roadmap.png')


def generate_technology_recommendations() -> str:
    """
    Generate text recommendations for technology integration.

    Returns:
        Formatted recommendation text
    """
    # Calculate top technologies by ROI
    sorted_techs = sorted(TECHNOLOGY_CATALOG, key=lambda t: t.roi, reverse=True)

    report = """
TECHNOLOGY INTEGRATION RECOMMENDATIONS
======================================

PHASE 1: Quick Wins (0-6 months)
---------------------------------
High ROI technologies with fast implementation:

"""

    phase1_techs = [t for t in sorted_techs if t.implementation_time <= 2][:3]
    for i, tech in enumerate(phase1_techs, 1):
        report += f"{i}. {tech.name} ({tech.category.upper()})\n"
        report += f"   - {tech.description}\n"
        report += f"   - Cost: {tech.cost}/10, Effectiveness: {tech.effectiveness}/10, ROI: {tech.roi:.2f}\n"
        report += f"   - Implementation: {tech.implementation_time} months\n\n"

    report += """
PHASE 2: Core Capabilities (6-12 months)
-----------------------------------------
Essential systems for comprehensive coverage:

"""

    phase2_techs = [t for t in sorted_techs if 2 < t.implementation_time <= 4][:3]
    for i, tech in enumerate(phase2_techs, 1):
        report += f"{i}. {tech.name} ({tech.category.upper()})\n"
        report += f"   - {tech.description}\n"
        report += f"   - Cost: {tech.cost}/10, Effectiveness: {tech.effectiveness}/10, ROI: {tech.roi:.2f}\n"
        report += f"   - Implementation: {tech.implementation_time} months\n\n"

    report += """
PHASE 3: Advanced Integration (12-24 months)
---------------------------------------------
Premium systems for maximum effectiveness:

"""

    phase3_techs = [t for t in sorted_techs if t.implementation_time > 4][:3]
    for i, tech in enumerate(phase3_techs, 1):
        report += f"{i}. {tech.name} ({tech.category.upper()})\n"
        report += f"   - {tech.description}\n"
        report += f"   - Cost: {tech.cost}/10, Effectiveness: {tech.effectiveness}/10, ROI: {tech.roi:.2f}\n"
        report += f"   - Implementation: {tech.implementation_time} months\n\n"

    report += """
KEY RECOMMENDATIONS:
-------------------
1. Prioritize communication and tracking systems first (highest immediate impact)
2. Integrate environmental sensing early to enable priority-based algorithms
3. Building automation provides long-term efficiency gains
4. Deploy in phases to manage costs and allow for learning/adaptation
5. Ensure interoperability between systems for maximum effectiveness

ESTIMATED TOTAL TIMELINE: 24-36 months for full deployment
ESTIMATED TOTAL COST: Variable based on building size and existing infrastructure
"""

    return report
