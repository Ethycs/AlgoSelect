import time, random, math, itertools, sys, types, textwrap, importlib, json, warnings
from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import numpy as np
import pandas as pd

# Try optional libraries
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from scipy import optimize
except ImportError:
    optimize = None

# ---------------- Core Data Structures ----------------
@dataclass
class ProblemInstance:
    name: str
    generator: Callable[[], Any]
    evaluator: Callable[[Any, Any], Dict[str, float]]

@dataclass
class Algorithm:
    name: str
    solver: Callable[[Any], Any]

@dataclass
class AlgorithmPair:
    name: str
    systematic: Algorithm
    randomized: Algorithm

@dataclass
class Measurement:
    runtime: float
    solution_quality: float

# ---------------- Problem Generators & Evaluators ----------------
def gen_int_array(n: int = 1_000):
    return random.sample(range(n * 10), n)

def eval_sorted(arr: List[int], sol: List[int]):
    correct = int(sol == sorted(arr))
    return {"solution_quality": correct}

def gen_sparse_graph(n: int = 200, p: float = 0.02):
    if nx is None:
        raise RuntimeError("networkx not installed – graph problems unavailable")
    g = nx.gnp_random_graph(n, p, directed=False)
    for (u, v) in g.edges:
        g[u][v]["weight"] = random.randint(1, 20)
    s, t = 0, n - 1
    return (g, s, t)

def eval_shortest_path(inst, sol):
    if nx is None or sol is None:
        return {"solution_quality": 0.0}
    g, s, t = inst
    try:
        opt = nx.dijkstra_path_length(g, s, t, weight="weight")
        ratio = opt / sol if sol else 0.0
        return {"solution_quality": 1.0 if math.isclose(sol, opt) else ratio}
    except Exception:
        return {"solution_quality": 0.0}

def gen_lp(n: int = 10, m: int = 20):
    if optimize is None:
        raise RuntimeError("scipy not installed – LP problem unavailable")
    A = np.random.randn(m, n)
    b = np.random.rand(m) * 10
    c = np.random.randn(n)
    return (A, b, c)

def eval_lp(inst, sol):
    status, obj = sol if sol else (1, float("-inf"))
    return {"solution_quality": -obj}

def gen_3sat(n_vars: int = 20, n_clauses: int = 90):
    assignment = {i + 1: random.choice([True, False]) for i in range(n_vars)}
    clauses = []
    for _ in range(n_clauses):
        vs = random.sample(range(1, n_vars + 1), 3)
        clause = []
        for v in vs:
            lit = v if assignment[v] else -v
            clause.append(lit)
        clauses.append(tuple(clause))
    return (n_vars, clauses)

def eval_3sat(inst, sol):
    n_vars, clauses = inst
    assign = sol or {}
    sat = all(any((lit > 0) == assign.get(abs(lit), False) for lit in clause) for clause in clauses)
    return {"solution_quality": sat}

def gen_matrix(n: int = 100):
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    return (A, B)

def eval_matmul(inst, sol):
    A, B = inst
    exact = A @ B
    approx = sol
    if approx is None:
        return {"solution_quality": 0.0}
    err = np.linalg.norm(exact - approx) / np.linalg.norm(exact)
    return {"solution_quality": 1 - err}

# ---------------- Algorithms ----------------
def bubble_sort(arr):
    a = arr.copy()
    for i in range(len(a)):
        for j in range(0, len(a) - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a

def quicksort_random(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_random(left) + mid + quicksort_random(right)

def dijkstra_solver(inst):
    if nx is None:
        raise RuntimeError("networkx missing")
    g, s, t = inst
    return nx.dijkstra_path_length(g, s, t, weight="weight")

def random_walk_sp(inst, steps: int = 5_000):
    g, s, t = inst
    node = s
    dist = 0
    for _ in range(steps):
        if node == t:
            return dist
        nbrs = list(g.neighbors(node))
        nxt = random.choice(nbrs)
        dist += g[node][nxt]["weight"]
        node = nxt
    return None

def simplex_solver(inst):
    if optimize is None:
        raise RuntimeError("scipy missing")
    A, b, c = inst
    res = optimize.linprog(-c, A_ub=A, b_ub=b, method="highs")
    return (res.status, -res.fun if res.success else float("-inf"))

def random_feasible_search(inst, tries: int = 10_000):
    A, b, c = inst
    best = float("-inf")
    for _ in range(tries):
        x = np.random.rand(len(c)) * 10
        if np.all(A @ x <= b):
            obj = c @ x
            best = max(best, obj)
    return (0, best)

def dpll_solver(inst):
    n_vars, clauses = inst
    assignment = {}

    def unit_propagate(cl, asn):
        changed = True
        while changed:
            changed = False
            units = [c for c in cl if len(c) == 1]
            for unit in units:
                lit = unit[0]
                var, val = abs(lit), lit > 0
                if var in asn and asn[var] != val:
                    return None
                asn[var] = val
                new_cl = []
                for c in cl:
                    if lit in c:
                        continue
                    if -lit in c:
                        c = tuple(l for l in c if l != -lit)
                    new_cl.append(c)
                cl = new_cl
                changed = True
        return cl

    def backtrack(cl, asn):
        cl = unit_propagate(cl, asn)
        if cl is None:
            return None
        if not cl:
            return asn
        var = abs(next(iter(cl[0])))
        for val in [True, False]:
            new_asn = asn.copy()
            new_asn[var] = val
            res = backtrack([c for c in cl if var not in map(abs, c)], new_asn)
            if res is not None:
                return res
        return None

    return backtrack(list(clauses), assignment)

def random_3sat_solver(inst, max_tries: int = 10_000):
    n_vars, clauses = inst
    for _ in range(max_tries):
        asn = {i + 1: random.choice([True, False]) for i in range(n_vars)}
        if all(any((lit > 0) == asn[abs(lit)] for lit in c) for c in clauses):
            return asn
    return None

def matmul_naive(inst):
    A, B = inst
    return A @ B

def matmul_random_sampling(inst, k: int = 20):
    A, B = inst
    n = A.shape[0]
    idx = np.random.choice(n, k)
    C = np.zeros((n, n))
    for i in idx:
        C += np.outer(A[:, i], B[i, :]) * n / k
    return C

# ---------------- Assemble Inventory ----------------
PROBLEMS = [
    ProblemInstance("IntArraySort", gen_int_array, eval_sorted),
    ProblemInstance("ShortestPathSparse", gen_sparse_graph, eval_shortest_path),
    ProblemInstance("LinearProgram", gen_lp, eval_lp),
    ProblemInstance("3SAT", gen_3sat, eval_3sat),
    ProblemInstance("MatrixMultiply", gen_matrix, eval_matmul),
]

ALGORITHM_PAIRS = [
    AlgorithmPair("SortPair", Algorithm("BubbleSort", bubble_sort), Algorithm("QuickSortRand", quicksort_random)),
    AlgorithmPair("GraphPair", Algorithm("Dijkstra", dijkstra_solver), Algorithm("RandWalkSP", random_walk_sp)),
    AlgorithmPair("LPPair", Algorithm("Simplex", simplex_solver), Algorithm("RandomSearchLP", random_feasible_search)),
    AlgorithmPair("SatPair", Algorithm("DPLL", dpll_solver), Algorithm("Rand3SAT", random_3sat_solver)),
    AlgorithmPair("MatMulPair", Algorithm("MatMulNaive", matmul_naive), Algorithm("MatMulSample", matmul_random_sampling)),
]

N_REPS = 10

def execute(algo: Algorithm, inst: Any, eval_fn: Callable) -> Measurement:
    t0 = time.perf_counter()
    try:
        sol = algo.solver(inst)
    except Exception:
        sol = None
    runtime = time.perf_counter() - t0
    quality = eval_fn(inst, sol).get("solution_quality", 0.0)
    return Measurement(runtime, quality)

def cliff_delta(x: List[float], y: List[float]):
    n_x, n_y = len(x), len(y)
    greater = sum(1 for xi in x for yi in y if xi > yi)
    lesser = sum(1 for xi in x for yi in y if xi < yi)
    return (greater - lesser) / (n_x * n_y) if n_x * n_y else float('nan')

def run_pilot():
    rng = random.Random(42)
    results = {prob.name: {pair.name: [] for pair in ALGORITHM_PAIRS} for prob in PROBLEMS}
    for prob in PROBLEMS:
        for pair in ALGORITHM_PAIRS:
            for _ in range(N_REPS):
                try:
                    inst = prob.generator()
                except Exception as e:
                    warnings.warn(f"Skipping {prob.name}: {e}")
                    continue
                chosen_algo = rng.choice([pair.systematic, pair.randomized])
                meas = execute(chosen_algo, inst, prob.evaluator)
                results[prob.name][pair.name].append({
                    "runtime": meas.runtime,
                    "quality": meas.solution_quality,
                    "algo": chosen_algo.name
                })
    return results

pilot_results = run_pilot()

# Build summary dataframe
records = []
for prob, pairs in pilot_results.items():
    for pair, runs in pairs.items():
        if not runs:  # skipped problem
            continue
        rt = [r["runtime"] for r in runs]
        q  = [r["quality"] for r in runs]
        records.append({
            "Problem": prob,
            "Pair": pair,
            "RT_mean": np.mean(rt),
            "RT_sd": np.std(rt),
            "Q_mean": np.mean(q),
            "Q_sd": np.std(q),
            "n": len(runs)
        })
summary_df = pd.DataFrame.from_records(records)

# import ace_tools as tools; tools.display_dataframe_to_user("Pilot summary", summary_df)
print("DEBUG: Skipping ace_tools display of Pilot summary_df")

# Effect size table
effect_records = []
for pair in ALGORITHM_PAIRS:
    sys_rts = []
    rand_rts = []
    for prob in PROBLEMS:
        runs = pilot_results[prob.name][pair.name]
        sys_rts.extend([r["runtime"] for r in runs if r["algo"] == pair.systematic.name])
        rand_rts.extend([r["runtime"] for r in runs if r["algo"] == pair.randomized.name])
    if sys_rts and rand_rts:
        effect_records.append({
            "Pair": pair.name,
            "Cliff_delta": cliff_delta(sys_rts, rand_rts),
            "n_sys": len(sys_rts),
            "n_rand": len(rand_rts)
        })
effect_df = pd.DataFrame(effect_records)
# tools.display_dataframe_to_user("Cliff Delta (runtime)", effect_df)
print("DEBUG: Skipping ace_tools display of Cliff Delta effect_df")
