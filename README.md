# Emergency Evacuation Sweep Optimization System

A Python simulation system to optimize emergency evacuation sweep strategies for multi-floor buildings. The system assigns responders (firefighters) to rooms and determines optimal sweep paths that minimize total clearance time.

## Overview

This system models building evacuations as a graph optimization problem, where:
- Buildings are represented as graphs with rooms and corridors
- Multiple responders work in parallel to sweep all rooms
- The goal is to minimize the maximum completion time across all responders

## Features

- **Intelligent Room Assignment**: Greedy workload-balanced clustering algorithm
- **Path Optimization**: Nearest neighbor construction with 2-opt improvement
- **Multiple Scenarios**: Three pre-built building layouts with varying complexity
- **Comprehensive Visualization**: Floor plans, cluster assignments, path diagrams, and Gantt charts
- **Performance Analysis**: Comparison against baseline strategies and sensitivity analysis

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the complete simulation system:

```bash
python main.py
```

This will:
1. Generate visualizations for all three scenarios
2. Test with 1-4 responders per scenario
3. Create comparison tables and analysis plots
4. Save all outputs to `/mnt/user-data/outputs/`

## Project Structure

```
.
├── building.py         # Core data structures (Room, Edge, BuildingGraph)
├── algorithms.py       # Clustering and pathfinding algorithms
├── simulation.py       # EvacuationSimulation class
├── scenarios.py        # Three predefined building layouts
├── visualization.py    # Plotting and table generation functions
├── main.py            # Main execution script
└── requirements.txt   # Python dependencies
```

## Scenarios

### Scenario 1: Single Floor Office
- 6 offices, 2 exits
- Simple linear layout
- Best for testing basic functionality

### Scenario 2: Two-Floor Mixed Use
- 12 rooms across 2 floors
- Mix of classrooms, labs, offices, and storage
- Includes stairs (heavier edge weights)
- Tests multi-floor optimization

### Scenario 3: High-Density Office
- 10 offices, 4 exits
- Highly interconnected grid layout
- Tests sensitivity to multiple exits

## Algorithms

### Room Assignment (Clustering)
Greedy workload-balanced algorithm:
1. Distribute responders across exits
2. Iteratively assign nearest unassigned room to least-busy responder
3. Update workload estimates including travel and sweep time

### Path Optimization
Two-phase approach:
1. **Nearest Neighbor**: Construct initial path greedily
2. **2-Opt**: Iteratively improve by reversing path segments

### Baseline Comparisons
- **Naive Sequential**: Divide rooms evenly without considering location
- **Nearest Neighbor Only**: Skip 2-opt improvement step

## Results

Key findings from simulations:

- **Multi-responder benefit**: 60-70% time reduction with 4 vs 1 responder
- **Algorithm improvement**: 7-20% better than naive baseline
- **Diminishing returns**: Marginal improvements decrease with more responders

## Output Files

The system generates:
- **Floor Plans**: Graph visualizations of building layouts (3 files)
- **Cluster Assignments**: Room-to-responder mappings (12 files)
- **Optimal Paths**: Visualization of responder routes (12 files)
- **Gantt Charts**: Timeline breakdowns (12 files)
- **Analysis Plots**: Responder comparisons and sensitivity analysis (6 files)
- **Data Tables**: Room properties, edge weights, and algorithm comparisons (7 CSV files)

Total: 52 output files

## Performance

- All simulations complete in < 5 seconds
- Efficient graph algorithms (Dijkstra's shortest path)
- Optimized visualization rendering

## Future Enhancements

Potential improvements:
- Dynamic priority adjustments during evacuation
- Real-time obstacle avoidance
- Communication constraints between responders
- Battery/stamina limitations
- Uncertainty in room occupancy

## License

This project was created for educational and research purposes.
