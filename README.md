# HiMCM 2025 Problem A - Emergency Evacuation Sweep Optimization

## 🚀 Quick Start - Just Run This!

To generate **ALL outputs** for your HiMCM 2025 submission:

```bash
python3 main.py
```

**That's it!** This single command generates everything you need (69 files in ~15 seconds).

---

## 📁 Clean File Structure

The codebase has been reorganized into **7 core modules** for maximum clarity:

### Core Modules (7 files)

1. **building.py** - Core data structures
   - `Room`, `Edge`, `BuildingGraph` classes
   - Graph representation of buildings

2. **scenarios.py** - All 7 building scenarios
   - Scenarios 1-4: Original (office, school, hospital, multi-floor)
   - Scenario 5: L-shaped layout (non-standard)
   - Scenario 6: Fire emergency (blocked areas)
   - Scenario 7: Hospital gas leak (priority mode)

3. **algorithms.py** - Pathfinding and optimization
   - Nearest-neighbor + 2-opt + Or-opt
   - TSP optimization algorithms
   - Room assignment strategies

4. **simulation.py** - Evacuation simulation engine
   - `EvacuationSimulation` class
   - Timeline generation
   - Performance metrics

5. **analysis.py** - ALL analysis functions (NEW - consolidated from 10 files!)
   - Emergency routing with blocked areas
   - Communication protocols (5 protocols)
   - Responder risk analysis
   - Occupant awareness analysis
   - Priority comparison
   - Performance heatmaps
   - Technology framework
   - Redundancy analysis
   - Scalability testing

6. **visualization.py** - Visualization functions
   - Floor plans, optimal paths, Gantt charts
   - Comparison tables and metrics

7. **main.py** - **⭐ SINGLE ENTRY POINT - RUN THIS! ⭐**
   - Orchestrates all Parts 1-4
   - Generates 69 output files
   - Creates executive summary

---

## 📊 What Gets Generated

Running `python3 main.py` generates **69 files** in `/mnt/user-data/outputs/`:

### Part 1-2: Core Scenarios
- ✅ Floor plans (Scenarios 1-4)
- ✅ Optimal paths visualizations
- ✅ Gantt charts
- ✅ Performance comparison tables

### Part 3: Sensitivity & Scalability
- ✅ Visibility sensitivity (0.5 to 1.0): 505s → 985s (95% variation)
- ✅ Walking speed sensitivity (1.0 to 2.0 m/s)
- ✅ Scalability analysis

### Part 4a: Emergency Scenarios
- ✅ Scenario 5 (L-shaped): ~431s
- ✅ Scenario 6 (Fire): Blocked areas handling
- ✅ Scenario 7 (Gas leak): Priority mode for ICU

### Part 4b-d: Extensions
- ✅ 5 communication protocols defined
- ✅ Communication decision flowchart
- ✅ Risk assessment matrices (optimal: 5 responders)
- ✅ Occupant awareness analysis
- ✅ Executive summary document

---

## 🎯 Key Results

**Algorithm Performance:**
- Scenario 1 (Office): 109s
- Scenario 2 (School): 411s
- Scenario 3 (Hospital): 197s
- Scenario 4 (Multi-floor): 505s

**Sensitivity Analysis:**
- Visibility: 95% time variation
- Speed: 3.6% time variation

**Emergency Scenarios:**
- L-shaped layout: Handles dead ends and bottlenecks
- Fire emergency: Routes around blocked areas
- Gas leak: Prioritizes ICU patients (+5% time for safety)

**Risk Analysis:**
- Optimal team size: 5 responders
- Safety thresholds: SAFE/ACCEPTABLE/MARGINAL/UNSAFE

---

## 🏆 Competitive Advantages

1. **Complete Coverage** - All 7 scenarios + all Part 4 requirements
2. **Clean Architecture** - Just 7 core files (reduced from 18!)
3. **Single Command** - `python3 main.py` does everything
4. **Quantified Results** - All claims backed by simulation data
5. **Professional Output** - 69 visualizations and analysis files
6. **Well-Organized** - Clear separation of concerns

---

## 📂 File Organization

```
/home/user/test/
├── main.py                          ⭐ RUN THIS FILE!
├── analysis.py                      All analysis (consolidated)
├── algorithms.py                    Pathfinding algorithms
├── building.py                      Data structures
├── scenarios.py                     7 building scenarios
├── simulation.py                    Simulation engine
├── visualization.py                 Plotting functions
├── generate_emergency_outputs.py   Helper (visualization)
├── generate_part4_outputs.py       Helper (visualization)
└── /mnt/user-data/outputs/         📁 69 generated files
```

---

## 🔧 Recent Major Reorganization

**Consolidated 10 scattered files into `analysis.py`:**
- `emergency.py` → `analysis.py`
- `communication.py` → `analysis.py`
- `risk_analysis.py` → `analysis.py`
- `occupant_awareness.py` → `analysis.py`
- `part4_analysis.py` → `analysis.py`
- `algorithm_heatmap.py` → `analysis.py`
- `technology_framework.py` → `analysis.py`
- `redundancy_analysis.py` → `analysis.py`
- `scalability.py` → `analysis.py`
- `generate_all_outputs.py` → Replaced by new `main.py`

**Result:** 18 Python files → 9 Python files (50% reduction!)

---

## 📋 Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `matplotlib`, `networkx`, `pandas`, `numpy`

---

## ✅ Ready for HiMCM 2025 Submission

All Parts 1-4 complete and tested.

**Just run `python3 main.py` and you're done!**

Good luck with HiMCM 2025! 🏆

---

## 📖 Additional Documentation

- `README_HOW_TO_RUN.md` - Detailed usage instructions and verification checklist
- `EXECUTIVE_SUMMARY.md` - Generated summary of all results (created when you run main.py)

---

*Emergency Evacuation Sweep Optimization System for HiMCM 2025 Problem A*
