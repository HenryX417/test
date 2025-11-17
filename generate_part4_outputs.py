"""
Master script to generate all Part 4 visualizations and outputs.

Run this script to create all competition-ready visualizations for
the HiMCM 2025 Problem A submission.
"""

from scenarios import create_scenario1, create_scenario2, create_scenario3, create_scenario4
from part4_analysis import compare_priority_modes, plot_priority_comparison, plot_priority_time_savings
from scalability import run_scalability_benchmark, plot_scalability_results, generate_synthetic_building
from technology_framework import plot_technology_matrix, plot_technology_roadmap
from redundancy_analysis import compare_redundancy_modes, plot_redundancy_comparison
from algorithm_heatmap import generate_performance_heatmap

import os


def ensure_output_dir(output_dir: str = '/mnt/user-data/outputs'):
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)


def generate_all_part4_outputs(output_dir: str = '/mnt/user-data/outputs'):
    """
    Generate all Part 4 visualizations and outputs.

    This creates:
    1. Priority mode comparison (Gantt charts, time savings)
    2. Scalability analysis (building size, responder count)
    3. Technology integration framework (matrix, roadmap)
    4. Safety redundancy analysis (double-checking)
    5. Algorithm performance heatmaps

    Args:
        output_dir: Directory to save all outputs
    """
    print('=' * 70)
    print('GENERATING ALL PART 4 VISUALIZATIONS FOR HiMCM 2025 PROBLEM A')
    print('=' * 70)

    ensure_output_dir(output_dir)

    # ========================================================================
    # 1. PRIORITY MODE COMPARISON (Extension 2.1)
    # ========================================================================
    print('\\n[1/5] Priority-Based Evacuation Analysis...')
    print('-' * 70)

    building_s4 = create_scenario4()
    standard_results, priority_results = compare_priority_modes(
        building_s4, num_responders=3
    )

    plot_priority_comparison(
        building_s4, standard_results, priority_results,
        'Scenario4_School', output_dir
    )

    plot_priority_time_savings(
        standard_results, priority_results,
        'Scenario4_School', output_dir
    )

    print(f'  ✅ Priority mode visualizations complete')
    print(f'     - Side-by-side Gantt charts')
    print(f'     - Time savings bar chart')

    # ========================================================================
    # 2. SCALABILITY ANALYSIS (Extension 2.3)
    # ========================================================================
    print('\\n[2/5] Scalability Analysis...')
    print('-' * 70)

    scalability_results = run_scalability_benchmark(
        room_counts=[6, 10, 15, 20, 30, 50],
        num_responders=3
    )

    plot_scalability_results(scalability_results, output_dir)

    print(f'  ✅ Scalability visualizations complete')
    print(f'     - Runtime vs building size')
    print(f'     - Evacuation time vs building size')
    print(f'     - Coverage and load balancing metrics')

    # ========================================================================
    # 3. TECHNOLOGY INTEGRATION FRAMEWORK (Extension 2.5)
    # ========================================================================
    print('\\n[3/5] Technology Integration Framework...')
    print('-' * 70)

    plot_technology_matrix(output_dir=output_dir)
    plot_technology_roadmap(output_dir=output_dir)

    print(f'  ✅ Technology framework visualizations complete')
    print(f'     - Cost vs effectiveness matrix')
    print(f'     - Implementation roadmap')

    # ========================================================================
    # 4. SAFETY REDUNDANCY ANALYSIS (Extension 2.4)
    # ========================================================================
    print('\\n[4/5] Safety Redundancy Analysis...')
    print('-' * 70)

    std_red, red_red = compare_redundancy_modes(building_s4, num_responders=3)

    plot_redundancy_comparison(
        building_s4, std_red, red_red,
        'Scenario4_School', output_dir
    )

    print(f'  ✅ Redundancy visualizations complete')
    print(f'     - Double-checking comparison')
    print(f'     - Safety score metrics')

    # ========================================================================
    # 5. ALGORITHM PERFORMANCE HEATMAPS
    # ========================================================================
    print('\\n[5/5] Algorithm Performance Heatmaps...')
    print('-' * 70)

    generate_performance_heatmap(
        building_s4,
        walking_speeds=[1.0, 1.25, 1.5, 1.75, 2.0],
        visibilities=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        num_responders=3,
        scenario_name='Scenario4_School',
        output_dir=output_dir
    )

    print(f'  ✅ Performance heatmaps complete')
    print(f'     - Walking speed vs visibility')

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print('\\n' + '=' * 70)
    print('ALL PART 4 VISUALIZATIONS GENERATED SUCCESSFULLY!')
    print('=' * 70)
    print(f'\\nOutput directory: {output_dir}')
    print('\\nGenerated files:')
    print('  - priority_comparison_Scenario4_School.png')
    print('  - priority_time_savings_Scenario4_School.png')
    print('  - scalability_analysis.png')
    print('  - responder_scaling.png')
    print('  - technology_matrix.png')
    print('  - technology_roadmap.png')
    print('  - redundancy_comparison_Scenario4_School.png')
    print('  - performance_heatmap_Scenario4_School.png')
    print('\\n✅ Ready for HiMCM 2025 submission!')
    print('=' * 70)


if __name__ == '__main__':
    generate_all_part4_outputs()
