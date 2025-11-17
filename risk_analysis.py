"""
Responder Shortage Risk Analysis Module.

This module generates risk assessment matrices showing how evacuation
performance varies with responder team size.

Part 4 Extension: Fewer than Optimal Responders
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from building import BuildingGraph
from simulation import EvacuationSimulation


def generate_responder_risk_matrix(
    building: BuildingGraph,
    scenario_name: str,
    responder_range: range = range(1, 7),
    time_thresholds: List[int] = [300, 600, 900],
    walking_speed: float = 1.5,
    visibility: float = 1.0,
    output_dir: str = '/mnt/user-data/outputs'
) -> Dict:
    """
    Create risk assessment matrix for different responder counts.

    Args:
        building: BuildingGraph to analyze
        scenario_name: Name for output files
        responder_range: Range of responder counts to test
        time_thresholds: Time thresholds for safety ratings [safe, marginal, unsafe]
        walking_speed: Walking speed (m/s)
        visibility: Visibility factor
        output_dir: Output directory

    Returns:
        Dictionary with analysis results
    """
    print(f'\n{"=" * 70}')
    print(f'RESPONDER SHORTAGE RISK ANALYSIS - {scenario_name}')
    print(f'{"=" * 70}')

    # Run simulations for each responder count
    results = {}
    for num_resp in responder_range:
        sim = EvacuationSimulation(building, num_responders=num_resp)
        sim.run(walking_speed=walking_speed, visibility=visibility, use_priority=True)
        total_time = sim.get_total_time()
        results[num_resp] = total_time
        print(f'  {num_resp} responders: {total_time:.1f}s')

    # Determine risk levels
    risk_matrix = {}
    for num_resp, time in results.items():
        if time <= time_thresholds[0]:
            risk_matrix[num_resp] = ('SAFE', '#2ecc71')  # Green
        elif time <= time_thresholds[1]:
            risk_matrix[num_resp] = ('ACCEPTABLE', '#3498db')  # Blue
        elif time <= time_thresholds[2]:
            risk_matrix[num_resp] = ('MARGINAL', '#f39c12')  # Orange
        else:
            risk_matrix[num_resp] = ('UNSAFE', '#e74c3c')  # Red

    # Find optimal responder count
    optimal = min(
        [r for r, (level, _) in risk_matrix.items() if level == 'SAFE'],
        default=max(responder_range)
    )

    print(f'\n📊 Risk Assessment:')
    for num_resp in responder_range:
        level, _ = risk_matrix[num_resp]
        marker = '✅' if level == 'SAFE' else '⚠️' if level in ['ACCEPTABLE', 'MARGINAL'] else '❌'
        print(f'  {marker} {num_resp} responders: {level} ({results[num_resp]:.1f}s)')

    print(f'\n💡 Recommendation: {optimal} responders minimum for SAFE operation')
    print(f'{"=" * 70}')

    # Generate heatmap visualization
    _plot_risk_heatmap(results, risk_matrix, time_thresholds, scenario_name, optimal, output_dir)

    return {
        'results': results,
        'risk_matrix': risk_matrix,
        'optimal_responders': optimal,
        'time_thresholds': time_thresholds
    }


def _plot_risk_heatmap(
    results: Dict[int, float],
    risk_matrix: Dict[int, Tuple[str, str]],
    time_thresholds: List[int],
    scenario_name: str,
    optimal: int,
    output_dir: str
):
    """
    Plot risk assessment heatmap.

    Args:
        results: Dict mapping responder count to evacuation time
        risk_matrix: Dict mapping responder count to (risk_level, color)
        time_thresholds: Time thresholds
        scenario_name: Scenario name
        optimal: Optimal responder count
        output_dir: Output directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Time vs Responders
    responders = list(results.keys())
    times = list(results.values())
    colors = [risk_matrix[r][1] for r in responders]

    bars = ax1.bar(responders, times, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)

    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{time:.0f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Add threshold lines
    ax1.axhline(y=time_thresholds[0], color='green', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Safe (<{time_thresholds[0]}s)')
    ax1.axhline(y=time_thresholds[1], color='orange', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Marginal (<{time_thresholds[1]}s)')
    ax1.axhline(y=time_thresholds[2], color='red', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Unsafe (>{time_thresholds[2]}s)')

    # Mark optimal
    if optimal in responders:
        idx = responders.index(optimal)
        ax1.scatter([optimal], [times[idx]], s=500, marker='*',
                   color='gold', edgecolor='black', linewidth=2,
                   zorder=5, label='Recommended')

    ax1.set_xlabel('Number of Responders', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Evacuation Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Evacuation Time vs Team Size', fontsize=13, fontweight='bold')
    ax1.set_xticks(responders)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Right plot: Risk matrix table
    ax2.axis('tight')
    ax2.axis('off')

    # Prepare table data
    table_data = [['Responders', f'<{time_thresholds[0]}s', f'<{time_thresholds[1]}s',
                   f'<{time_thresholds[2]}s', 'Safety Rating']]

    for num_resp in responders:
        time = results[num_resp]
        level, _ = risk_matrix[num_resp]

        # Check marks for each threshold
        check1 = '✅' if time <= time_thresholds[0] else '❌'
        check2 = '✅' if time <= time_thresholds[1] else '❌'
        check3 = '✅' if time <= time_thresholds[2] else '❌'

        marker = '⭐' if num_resp == optimal else ''
        table_data.append([
            f'{num_resp} {marker}',
            check1,
            check2,
            check3,
            level
        ])

    # Create table
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.20])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Color rows by risk level
    for i in range(1, len(table_data)):
        num_resp = responders[i - 1]
        _, color = risk_matrix[num_resp]

        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor(color)
            cell.set_alpha(0.3)

            # Make safety rating cell more prominent
            if j == 4:
                cell.set_text_props(weight='bold')
                cell.set_alpha(0.5)

    ax2.set_title('Risk Assessment Matrix', fontsize=13, fontweight='bold', pad=10)

    # Overall title
    fig.suptitle(f'Responder Shortage Risk Analysis - {scenario_name}\n'
                f'Recommendation: {optimal}+ responders for SAFE operation',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/responder_risk_matrix_{scenario_name}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Risk matrix saved:')
    print(f'   {output_dir}/responder_risk_matrix_{scenario_name}.png')


def generate_cost_benefit_analysis(
    risk_results: Dict,
    scenario_name: str,
    cost_per_responder: int = 50000,  # Annual salary
    output_dir: str = '/mnt/user-data/outputs'
):
    """
    Generate cost-benefit analysis for responder team sizing.

    Args:
        risk_results: Results from generate_responder_risk_matrix()
        scenario_name: Scenario name
        cost_per_responder: Annual cost per responder (dollars)
        output_dir: Output directory
    """
    results = risk_results['results']
    optimal = risk_results['optimal_responders']

    fig, ax = plt.subplots(figsize=(12, 7))

    responders = list(results.keys())
    costs = [r * cost_per_responder for r in responders]
    times = list(results.values())

    # Normalize time to 0-100 scale (inverted - lower is better)
    max_time = max(times)
    normalized_times = [(max_time - t) / max_time * 100 for t in times]

    # Plot
    ax2 = ax.twinx()

    # Cost bars
    bars = ax.bar(responders, costs, alpha=0.6, color='#e74c3c',
                  edgecolor='black', linewidth=2, label='Annual Cost')

    # Performance line
    line = ax2.plot(responders, normalized_times, 'o-', color='#2ecc71',
                   linewidth=3, markersize=10, label='Performance Score',
                   markeredgecolor='black', markeredgewidth=2)

    # Mark optimal
    idx = responders.index(optimal)
    ax.scatter([optimal], [costs[idx]], s=500, marker='*',
              color='gold', edgecolor='black', linewidth=2, zorder=5)
    ax2.scatter([optimal], [normalized_times[idx]], s=500, marker='*',
               color='gold', edgecolor='black', linewidth=2, zorder=5)

    # Labels
    ax.set_xlabel('Number of Responders', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Cost ($)', fontsize=12, fontweight='bold', color='#e74c3c')
    ax2.set_ylabel('Performance Score (higher is better)', fontsize=12,
                   fontweight='bold', color='#2ecc71')

    ax.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')

    ax.set_xticks(responders)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Add optimal annotation
    ax.annotate(f'Optimal: {optimal} responders\n'
               f'Cost: ${costs[idx]:,}\n'
               f'Performance: {normalized_times[idx]:.1f}/100',
               xy=(optimal, costs[idx]),
               xytext=(optimal + 0.5, costs[idx] * 1.2),
               bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', lw=2))

    plt.title(f'Cost-Benefit Analysis - {scenario_name}\n'
             f'Responder Team Size Optimization',
             fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cost_benefit_analysis_{scenario_name}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✅ Cost-benefit analysis saved:')
    print(f'   {output_dir}/cost_benefit_analysis_{scenario_name}.png')


def analyze_all_scenarios_risk(output_dir: str = '/mnt/user-data/outputs'):
    """
    Run risk analysis for all scenarios.

    Args:
        output_dir: Output directory
    """
    from scenarios import (create_scenario1, create_scenario2, create_scenario3,
                          create_scenario4, create_scenario5)

    scenarios = [
        ('Scenario1', create_scenario1()),
        ('Scenario2', create_scenario2()),
        ('Scenario3', create_scenario3()),
        ('Scenario4', create_scenario4()),
        ('Scenario5', create_scenario5()),
    ]

    print('\n' + '=' * 70)
    print('RUNNING RISK ANALYSIS FOR ALL SCENARIOS')
    print('=' * 70)

    all_results = {}

    for name, building in scenarios:
        risk_results = generate_responder_risk_matrix(
            building, name, responder_range=range(1, 7),
            output_dir=output_dir
        )
        all_results[name] = risk_results

        # Generate cost-benefit for each
        generate_cost_benefit_analysis(risk_results, name, output_dir=output_dir)

    # Summary
    print('\n' + '=' * 70)
    print('RISK ANALYSIS SUMMARY')
    print('=' * 70)
    for name, results in all_results.items():
        print(f'{name}: Recommended {results["optimal_responders"]} responders')

    return all_results
