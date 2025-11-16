"""
Main execution script for evacuation sweep optimization system.

This script runs all scenarios, generates visualizations, and produces
comparison tables.
"""

import os
from scenarios import create_scenario1, create_scenario2, create_scenario3
from simulation import EvacuationSimulation
from visualization import (
    plot_floor_plan,
    plot_cluster_assignment,
    plot_optimal_paths,
    plot_gantt_chart,
    plot_responder_comparison,
    plot_sensitivity_analysis,
    generate_room_properties_table,
    generate_comparison_table,
    generate_edge_weights_table
)


def main():
    """
    Run all three scenarios with varying responder counts.
    Generate all figures and tables.
    Save results to /mnt/user-data/outputs/
    """
    output_dir = '/mnt/user-data/outputs'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  EMERGENCY EVACUATION SWEEP OPTIMIZATION SYSTEM")
    print("="*70)

    # Create scenarios
    scenarios = {
        'Scenario_1_Single_Floor': create_scenario1(),
        'Scenario_2_Two_Floor': create_scenario2(),
        'Scenario_3_High_Density': create_scenario3()
    }

    print(f"\nCreated {len(scenarios)} building scenarios")
    print(f"Output directory: {output_dir}\n")

    # Generate room properties tables for each scenario
    print("\n" + "-"*70)
    print("Generating Room Properties Tables...")
    print("-"*70)

    for name, building in scenarios.items():
        room_table = generate_room_properties_table(building)
        csv_path = f'{output_dir}/room_properties_{name}.csv'
        room_table.to_csv(csv_path, index=False)
        print(f"  ✓ Saved room properties for {name}")

    # Generate edge weights tables
    print("\n" + "-"*70)
    print("Generating Edge Weights Tables...")
    print("-"*70)

    for name, building in scenarios.items():
        edge_table = generate_edge_weights_table(building)
        csv_path = f'{output_dir}/edge_weights_{name}.csv'
        edge_table.to_csv(csv_path, index=False)
        print(f"  ✓ Saved edge weights for {name}")

    # Run simulations for each scenario
    for name, building in scenarios.items():
        print("\n" + "="*70)
        print(f"Running {name}")
        print("="*70)

        # Generate floor plan
        print(f"\n  Generating floor plan visualization...")
        plot_floor_plan(building, name, output_dir)
        print(f"  ✓ Floor plan saved")

        # Test different responder counts
        responder_counts = [1, 2, 3, 4]

        for num_resp in responder_counts:
            print(f"\n  Testing with {num_resp} responder(s)...")

            # Run simulation
            sim = EvacuationSimulation(building, num_resp)
            sim.run()

            # Print summary to console
            sim.print_summary()

            # Generate visualizations
            scenario_label = f"{name}_{num_resp}resp"

            print(f"    Generating visualizations...")
            plot_cluster_assignment(building, sim.assignments, scenario_label, output_dir)
            plot_optimal_paths(building, sim.paths, scenario_label, output_dir)
            plot_gantt_chart(sim.get_timeline(), scenario_label, output_dir)

            print(f"    ✓ Visualizations saved for {num_resp} responder(s)")
            print(f"    ✓ Total evacuation time: {sim.get_total_time():.2f} seconds")

        # Additional analysis plots
        print(f"\n  Generating comparison and sensitivity plots...")
        plot_responder_comparison(building, name, max_responders=6, output_dir=output_dir)
        plot_sensitivity_analysis(building, name, num_responders=3, output_dir=output_dir)
        print(f"  ✓ Analysis plots saved")

    # Generate comparison table across all scenarios
    print("\n" + "="*70)
    print("Generating Comparison Table...")
    print("="*70)

    comparison_table = generate_comparison_table(scenarios, [1, 2, 3, 4])
    comparison_path = f'{output_dir}/comparison_table.csv'
    comparison_table.to_csv(comparison_path, index=False)

    print(f"\n✓ Comparison table saved to: {comparison_path}")

    # Print comparison table to console
    print("\n" + "-"*70)
    print("ALGORITHM COMPARISON RESULTS")
    print("-"*70)
    print(comparison_table.to_string(index=False))

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    for name, building in scenarios.items():
        print(f"\n{name}:")
        print(f"  - Total rooms: {len(building.get_all_room_ids())}")
        print(f"  - Total exits: {len(building.exits)}")
        print(f"  - Total edges: {len(building.edges)}")

        # Get best times
        best_times = {}
        for num_resp in [1, 2, 3, 4]:
            sim = EvacuationSimulation(building, num_resp)
            sim.run()
            best_times[num_resp] = sim.get_total_time()

        print(f"  - Best time (1 responder): {best_times[1]:.2f}s")
        print(f"  - Best time (4 responders): {best_times[4]:.2f}s")
        print(f"  - Improvement with 4 responders: {((best_times[1] - best_times[4]) / best_times[1] * 100):.1f}%")

    print("\n" + "="*70)
    print("EXECUTION COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - Floor plan diagrams (3 scenarios)")
    print("  - Cluster assignment diagrams (12 total: 3 scenarios × 4 responder counts)")
    print("  - Optimal path diagrams (12 total)")
    print("  - Gantt charts (12 total)")
    print("  - Responder comparison graphs (3 scenarios)")
    print("  - Sensitivity analysis plots (3 scenarios)")
    print("  - Room properties tables (3 CSV files)")
    print("  - Edge weights tables (3 CSV files)")
    print("  - Comparison table (1 CSV file)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
