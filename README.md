# AlgoSelect

AlgoSelect is a Python-based benchmarking and analysis framework for algorithm selection and comparison. It is designed to help researchers, educators, and students systematically evaluate a wide variety of algorithms—both deterministic (systematic) and randomized (heuristic)—across diverse computational problems.

> **This template is tailored for the [pixi.dev](https://pixi.dev) Python environment manager.**

---

## Features

- **Rich Algorithm Arsenal:** Includes 20+ classic and modern algorithms (sorting, graph, SAT, optimization, numerical, etc.), with both deterministic and randomized versions.
- **Diverse Problem Suite:** Test algorithms on a curated set of problem types, including sorting, shortest path, SAT, matrix operations, eigenvalue computation, knapsack, TSP, graph coloring, and more.
- **Pairwise Comparisons:** Run controlled experiments comparing systematic and randomized algorithms on the same problem instances.
- **Automated Experimentation:** Scripts for generating instances, running benchmarks, collecting runtime/quality metrics, and statistical analysis.
- **Visualization:** Utilities to analyze and visualize algorithm performance (e.g., heatmaps, summary statistics).

---

## Quick Start (with pixi.dev)

1. **Install pixi**  
   See [pixi.dev documentation](https://pixi.dev/docs/install/) for platform-specific instructions.

2. **Clone the repository**
   ```bash
   git clone https://github.com/Ethycs/AlgoSelect.git
   cd AlgoSelect
   ```

3. **Set up the environment**
   ```bash
   pixi install
   ```

   This will create a reproducible environment with all dependencies declared in `pixi.toml` (and/or `pyproject.toml`).

4. **Run experiments**
   ```bash
   pixi run python demos/experiment.py
   ```
   Or open the project in an IDE with the pixi environment activated.

---

## Directory Structure

- `demos/experiment.py` — Main script for defining problems, algorithms, and running experiments.
- `demos/` — Example scripts, LaTeX visualizations, and more.
- `pixi.toml` — Dependency and environment specification for pixi.
- `pyproject.toml` — (Optional) Project metadata and additional dependencies.
- `requirements.txt` — (Optional) Compatibility with other Python tools.
- Other files — Utilities, modules, and datasets.

---

## Adding New Algorithms or Problems

1. **Define the algorithm**  
   Implement your algorithm as a function in `demos/experiment.py`.
2. **Register in Arsenal**  
   Add the function to `algorithm_arsenal` and `algorithm_functions`.
3. **(For new problems)**  
   Define a `ProblemDefinition` with generator, feature extraction, and evaluator functions.

---

## Example Problems and Algorithms

- **Sorting:** Bubble Sort, Randomized Quicksort
- **Graphs:** Dijkstra, Prim, Kruskal, Max Flow, Graph Coloring
- **SAT Solving:** DPLL, Randomized SAT
- **Optimization:** Simplex, Simulated Annealing, Genetic Algorithms
- **Numerical:** Gaussian Elimination, SVD, Eigenvalues, Numerical Integration
- **Other:** Knapsack, TSP, Sudoku, Record Sorting

---

## License

MIT License

---

## Citation

If you use AlgoSelect in academic work, please cite this repository.

---

**Contributions and suggestions are welcome!**
