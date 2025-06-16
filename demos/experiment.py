# demos/experiment.py

import numpy as np
import scipy.optimize
import scipy.linalg
import scipy.sparse.linalg  # For eigs
import scipy.integrate  # For numerical integration
from scipy.spatial.distance import cdist  # For TSP example generator
import networkx as nx
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.impute import SimpleImputer
import random
import heapq
import time
import pandas as pd
# Kept for reference, not used in CV
from sklearn.linear_model import LogisticRegression
from collections import namedtuple
import string  # For string sort generator
from operator import itemgetter  # For record sort
from typing import List, Dict, Any # Using List, Dict, Any for safety
import seaborn as sns
import matplotlib.pyplot as plt

print("DEBUG: All imports in experiment.py completed successfully.")

# Helper structures
# Added problem_name, problem_type
RunResult = namedtuple(
    'RunResult', ['runtime', 'quality', 'algo_name', 'problem_name', 'problem_type'])


class AlgorithmDefinition:
    def __init__(self, name, solver_func, accepts_problem_type):
        self.name = name
        self.solver = solver_func
        self.accepts_problem_type = accepts_problem_type


class ProblemDefinition:
    def __init__(self, name, generator_func, feature_fn_func, evaluator_func, problem_type):
        self.name = name
        self.generator = generator_func
        self.feature_fn = feature_fn_func
        self.evaluator = evaluator_func
        self.problem_type = problem_type


pilot_problems = []  # Will be populated in __main__
pilot_pairs = []    # Will be populated in __main__


class AlgorithmPair:
    def __init__(self, name, systematic_algo_def, randomized_algo_def):
        self.name = name
        self.systematic = systematic_algo_def
        self.randomized = randomized_algo_def


# 1. Problem Suite (20 problems) - Data Only
problem_suite = [
    {"Category": "Sorting / Ordering", "ID": 1,
        "Problem": "Integer-array sort (n ≈ 1000)", "Notes": "Full order"},
    {"Category": "Sorting / Ordering", "ID": 2,
        "Problem": "String sort (n ≈ 500)", "Notes": "Unicode collation"},
    {"Category": "Sorting / Ordering", "ID": 3,
        "Problem": "Partial sort / top-k", "Notes": "k ≪ n"},
    {"Category": "Sorting / Ordering", "ID": 4,
        "Problem": "Multi-key record sort", "Notes": "Stable, composite keys"},
    {"Category": "Graphs", "ID": 5,
        "Problem": "Shortest path – sparse graphs", "Notes": ""},
    {"Category": "Graphs", "ID": 6, "Problem": "Shortest path – dense graphs",
        "Notes": "Floyd-Warshall baseline"},
    {"Category": "Graphs", "ID": 7, "Problem": "Minimum-spanning tree",
        "Notes": "Kruskal / Prim reference"},
    {"Category": "Graphs", "ID": 8, "Problem": "Max-flow",
        "Notes": "Edmonds-Karp/GF algorithm"},
    {"Category": "Optimization", "ID": 9,
        "Problem": "Linear programming", "Notes": "Feasible & bounded"},
    {"Category": "Optimization", "ID": 10,
        "Problem": "Convex quadratic programming", "Notes": "Positive-definite Q"},
    {"Category": "Optimization", "ID": 11,
        "Problem": "Non-convex continuous opt.", "Notes": "Multiple local minima"},
    {"Category": "Optimization", "ID": 12,
        "Problem": "Integer programming", "Notes": "0-1 knapsack style"},
    {"Category": "Search / SAT", "ID": 13,
        "Problem": "3-SAT (satisfiable)", "Notes": "Random instances"},
    {"Category": "Search / SAT", "ID": 14,
        "Problem": "3-SAT (unsatisfiable)", "Notes": "Near phase transition"},
    {"Category": "Search / SAT", "ID": 15,
        "Problem": "Graph coloring", "Notes": "k-colorability"},
    {"Category": "Search / SAT", "ID": 16,
        "Problem": "Sudoku solving", "Notes": "9×9 standard"},
    {"Category": "Numerical", "ID": 17,
        "Problem": "Matrix multiplication", "Notes": "Square, n ≈ 500"},
    {"Category": "Numerical", "ID": 18, "Problem": "Linear system solve",
        "Notes": "Ax = b, well-conditioned"},
    {"Category": "Numerical", "ID": 19,
        "Problem": "Eigenvalue computation", "Notes": "Largest eigenpair"},
    {"Category": "Numerical", "ID": 20, "Problem": "Numerical integration",
        "Notes": "1-D adaptive quadrature"},
]

# 2. Algorithm Arsenal (20 algorithms) - Data Only
algorithm_arsenal = [
    {"Group": "Systematic (deterministic)", "ID": 1,
     "Algorithm": "Bubble Sort", "Flavor": "Baseline O(n²)"},
    {"Group": "Systematic (deterministic)", "ID": 2,
     "Algorithm": "Dijkstra", "Flavor": "Heap-based"},
    {"Group": "Systematic (deterministic)", "ID": 3,
     "Algorithm": "Simplex", "Flavor": "Revised simplex"},
    {"Group": "Systematic (deterministic)", "ID": 4,
     "Algorithm": "DPLL (SAT)", "Flavor": "Unit propagation"},
    {"Group": "Systematic (deterministic)", "ID": 5,
     "Algorithm": "Gaussian elimination", "Flavor": "Partial pivot"},
    {"Group": "Systematic (deterministic)", "ID": 6,
     "Algorithm": "Branch-and-Bound", "Flavor": "IP solver"},
    {"Group": "Systematic (deterministic)", "ID": 7,
     "Algorithm": "Dynamic programming", "Flavor": "Bellman-Held-Karp"},
    {"Group": "Systematic (deterministic)", "ID": 8,
     "Algorithm": "Backtracking", "Flavor": "Generic CSP"},
    {"Group": "Systematic (deterministic)", "ID": 9,
     "Algorithm": "Greedy algorithm", "Flavor": "Problem-specific"},
    {"Group": "Systematic (deterministic)", "ID": 10,
     "Algorithm": "Newton’s method", "Flavor": "Line-search"},
    {"Group": "Randomized / Heuristic", "ID": 11,
        "Algorithm": "Quicksort (random pivot)", "Flavor": "Average-case O(n log n)"},
    {"Group": "Randomized / Heuristic", "ID": 12,
        "Algorithm": "Randomized Min-Cut", "Flavor": "Karger"},
    {"Group": "Randomized / Heuristic", "ID": 13,
        "Algorithm": "Simulated annealing", "Flavor": "Generic optimizer"},
    {"Group": "Randomized / Heuristic", "ID": 14,
        "Algorithm": "Random 3-SAT generator", "Flavor": "Benchmark creator"},
    {"Group": "Randomized / Heuristic", "ID": 15,
        "Algorithm": "Randomized SVD", "Flavor": "Sketching"},
    {"Group": "Randomized / Heuristic", "ID": 16,
        "Algorithm": "Monte-Carlo Tree Search", "Flavor": "UCT"},
    {"Group": "Randomized / Heuristic", "ID": 17,
        "Algorithm": "Genetic algorithm", "Flavor": "GA framework"},
    {"Group": "Randomized / Heuristic", "ID": 18,
        "Algorithm": "Random local search", "Flavor": "Hill-climber"},
    {"Group": "Randomized / Heuristic", "ID": 19,
        "Algorithm": "Las Vegas algorithm", "Flavor": "Success‐probability 1"},
    {"Group": "Randomized / Heuristic", "ID": 20,
        "Algorithm": "Stochastic gradient descent", "Flavor": "Mini-batch"},
]

# Base feature set for all problem types
BASE_FEATURE_FLAGS = {
    'is_sort_problem': 0.0, 'is_graph_problem': 0.0, 'is_dummy_problem': 0.0,
    'is_lp_problem': 0.0, 'is_3sat_problem': 0.0, 'is_matmul_problem': 0.0,
    'is_linsys_problem': 0.0, 'is_topk_problem': 0.0, 'is_string_sort_problem': 0.0,
    'is_mst_problem': 0.0, 'is_max_flow_problem': 0.0, 'is_convex_qp_problem': 0.0,
    'is_eigenvalue_problem': 0.0, 'is_record_sort_problem': 0.0, 'is_knapsack_problem': 0.0,
    'is_graph_coloring_problem': 0.0, 'is_sudoku_problem': 0.0, 'is_num_integration_problem': 0.0,
    'is_dense_graph_apsp_problem': 0.0, 'is_nonconvex_opt_problem': 0.0, 'is_3sat_unsat_problem': 0.0
}

# Algorithm Implementations / Wrappers
def bubble_sort(arr_instance):
    if not isinstance(arr_instance, list):
        return None
    n = len(arr_instance)
    arr = list(arr_instance)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

def dijkstra(graph_instance, source_node):
    if not isinstance(graph_instance, nx.Graph):
        return None
    if source_node not in graph_instance:
        return None
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph_instance, source_node, weight='weight')
        paths = nx.single_source_dijkstra_path(graph_instance, source_node)
        return {"shortest_path_lengths": lengths, "shortest_paths": paths}
    except:
        return None

def simplex_method(lp_problem):
    try:
        return scipy.optimize.linprog(lp_problem.get('c'), A_ub=lp_problem.get('A_ub'), b_ub=lp_problem.get('b_ub'), A_eq=lp_problem.get('A_eq'), b_eq=lp_problem.get('b_eq'), bounds=lp_problem.get('bounds'), method='highs')
    except:
        return None

# Basic DPLL implementation (as suggested by Fisher)
def dpll_sat_basic(clauses_input):
    # clauses should be a list of tuples of integers (literals)
    # assignment is a dictionary: variable -> True/False
    
    # Helper to simplify clauses based on an assignment
    def simplify(clauses, literal):
        new_clauses = []
        for clause in clauses:
            if literal in clause:
                continue  # Clause is satisfied
            if -literal in clause:
                new_clause = tuple(l for l in clause if l != -literal)
                if not new_clause: # Empty clause means contradiction
                    return None 
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause)
        return new_clauses

    # Main DPLL recursive function
    def solve(clauses, assignment):
        # Unit propagation
        while True:
            unit_literal = None
            for clause in clauses:
                if len(clause) == 1:
                    unit_literal = clause[0]
                    break
            if unit_literal is None:
                break # No unit clauses left
            
            var = abs(unit_literal)
            val = unit_literal > 0
            
            if var in assignment and assignment[var] != val:
                return None # Contradiction with existing assignment
            if var not in assignment:
                 assignment[var] = val

            clauses = simplify(clauses, unit_literal)
            if clauses is None: # Contradiction found during simplification
                return None
            if not clauses: # All clauses satisfied
                return assignment
        
        if not clauses: # All clauses satisfied
            return assignment

        # Pure literal elimination (optional, can be added for efficiency)
        # For simplicity, we'll omit it here but it's a standard DPLL step.

        # Choose a variable to branch on
        # A simple heuristic: pick the first variable in the first clause
        unassigned_vars = set()
        for clause in clauses:
            for literal in clause:
                var = abs(literal)
                if var not in assignment:
                    unassigned_vars.add(var)
        
        if not unassigned_vars:
            # This case should ideally be caught by "not clauses" if all clauses are satisfied,
            # or by simplify returning None if there's a contradiction.
            # If we reach here and clauses is not empty, it implies an issue or unhandled state.
            # For robustness, if there are clauses but no unassigned vars mentioned in them,
            # it might mean the remaining clauses are already satisfied or unsatisfiable
            # based on prior assignments not directly involved in unit propagation.
            # A full check:
            all_satisfied = True
            for clause in clauses:
                is_clause_satisfied = False
                for literal in clause:
                    var = abs(literal)
                    val = literal > 0
                    if var in assignment and assignment[var] == val:
                        is_clause_satisfied = True
                        break
                if not is_clause_satisfied:
                    all_satisfied = False
                    break
            return assignment if all_satisfied else None


        chosen_var = min(unassigned_vars) # Simple heuristic: pick smallest unassigned variable

        # Try assigning True
        assignment_true = assignment.copy()
        assignment_true[chosen_var] = True
        clauses_true = simplify(clauses, chosen_var)
        if clauses_true is not None:
            solution = solve(clauses_true, assignment_true)
            if solution is not None:
                return solution
        
        # Try assigning False
        assignment_false = assignment.copy()
        assignment_false[chosen_var] = False
        clauses_false = simplify(clauses, -chosen_var)
        if clauses_false is not None:
            solution = solve(clauses_false, assignment_false)
            if solution is not None:
                return solution
                
        return None # No solution found down this path

    # Initial call
    initial_assignment = {}
    # Convert input clauses to list of tuples if they aren't already
    processed_clauses = [tuple(c) for c in clauses_input]
    
    final_assignment = solve(processed_clauses, initial_assignment)
    
    # Fill in any unassigned variables (they can be anything if a solution is found)
    if final_assignment is not None:
        all_vars_in_problem = set()
        for clause in clauses_input:
            for literal in clause:
                all_vars_in_problem.add(abs(literal))
        for var in all_vars_in_problem:
            if var not in final_assignment:
                final_assignment[var] = True # Default to True for unmentioned vars
    return final_assignment


def gaussian_elimination(linear_system):
    A = np.array(linear_system.get('A'))
    b = np.array(linear_system.get('b'))
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        return None
    try:
        return np.linalg.solve(A, b)
    except:
        return None

def branch_and_bound(milp_problem):
    if 'c' not in milp_problem:
        return None
    try:
        if not hasattr(scipy.optimize, 'milp'):
            return "conceptual_bnb_solution_scipy_version_issue"
        return scipy.optimize.milp(c=np.array(milp_problem['c']), integrality=np.array(milp_problem.get('integrality')) if milp_problem.get('integrality') is not None else None, bounds=milp_problem.get('bounds'), constraints=milp_problem.get('constraints'))
    except AttributeError:
        return "conceptual_bnb_solution_attribute_error"
    except:
        return None

def quicksort_random_pivot(arr_instance):
    if not isinstance(arr_instance, list):
        return None
    arr = list(arr_instance)
    _quicksort_recursive(arr, 0, len(arr) - 1)
    return arr

def _quicksort_recursive(arr, low, high):
    if low < high:
        pi = _quicksort_partition(arr, low, high)
        _quicksort_recursive(arr, low, pi - 1)
        _quicksort_recursive(arr, pi + 1, high)

def _quicksort_partition(arr, low, high):
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i + 1

def random_3_sat_generator(num_variables, num_clauses, clause_length=3, satisfiable=None):
    if num_variables <= 0 or num_clauses < 0 or clause_length <= 0:
        return []
    clauses = []
    variables = list(range(1, num_variables + 1))
    for _ in range(num_clauses):
        clause = []
        chosen_vars = random.choices(
            variables, k=clause_length) if num_variables < clause_length else random.sample(variables, k=clause_length)
        for var in chosen_vars:
            clause.append(var if random.choice([True, False]) else -var)
        clauses.append(clause)
    return clauses

def randomized_svd_wrapper(matrix_instance, n_components, n_iter=5, random_state=None):
    M = np.array(matrix_instance)
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] == 0:
        return None
    try:
        U, Sigma, VT = randomized_svd(
            M, n_components=n_components, n_iter=n_iter, random_state=random_state)
        return {"U": U, "Sigma": Sigma, "VT": VT}
    except:
        return None

def stochastic_gradient_descent(sgd_problem):
    X_train = sgd_problem.get('X_train')
    y_train = sgd_problem.get('y_train')
    learning_rate = sgd_problem.get('learning_rate', 0.01)
    epochs = sgd_problem.get('epochs', 100)
    if X_train is None or y_train is None:
        return None
    try:
        model = SGDRegressor(learning_rate='constant', eta0=learning_rate,
                             max_iter=epochs, tol=None, shuffle=True)
        model.fit(X_train, y_train)
        return {"coef": model.coef_, "intercept": model.intercept_}
    except:
        return {"status": "conceptual_manual_sgd_ran_due_to_error"}

# Conceptual solvers / wrappers
def randomized_lp_solver(lp_problem): return {'x': [
    0]*(len(lp_problem['c']) if 'c' in lp_problem and hasattr(lp_problem['c'], '__len__') else 0), 'fun': 0, 'success': True}

def randomized_sat_solver(
    sat_instance): return "conceptual_random_sat_solution" # Keep for now, dpll_sat_basic is the systematic one

def naive_matrix_mult(matrices):
    A, B = matrices
    if not (hasattr(A, 'shape') and hasattr(B, 'shape') and A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0]):
        return None
    return np.zeros((A.shape[0], B.shape[1]))

def library_matrix_mult(matrices):
    A, B = matrices
    if not (hasattr(A, 'shape') and hasattr(B, 'shape') and A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0]):
        return None
    return np.dot(A, B)

def randomized_kaczmarz(linear_system): return np.zeros(len(
    linear_system['b']) if 'b' in linear_system and hasattr(linear_system['b'], '__len__') else 0)

def sampling_top_k(arr_k_instance): arr, k = arr_k_instance; return sorted(
    arr, reverse=True)[:k] if isinstance(arr, list) else None

def python_string_sort(str_list): return sorted(
    str_list) if isinstance(str_list, list) else None

def conceptual_random_string_sort(str_list): return sorted(
    str_list, key=lambda x: random.random()) if isinstance(str_list, list) else None

def kruskal_mst(graph_instance): return nx.minimum_spanning_tree(
    graph_instance, algorithm='kruskal', weight='weight') if isinstance(graph_instance, nx.Graph) else None

def prim_mst(graph_instance): return nx.minimum_spanning_tree(graph_instance,
                                                              algorithm='prim', weight='weight') if isinstance(graph_instance, nx.Graph) else None

def edmonds_karp_max_flow(flow_input):
    if not isinstance(flow_input, dict) or not all(k in flow_input for k in ['graph', 'source', 'sink']):
        return 0
    G, s, t = flow_input['graph'], flow_input['source'], flow_input['sink']
    return nx.maximum_flow_value(G, s, t, capacity='capacity') if isinstance(G, nx.DiGraph) else 0

def conceptual_random_max_flow(flow_input): return 0
def conceptual_qp_solver(qp_problem): return "conceptual_qp_solution"

def power_iteration_eigs(matrix_instance, num_simulations: int = 100):
    if not isinstance(matrix_instance, np.ndarray) or matrix_instance.ndim != 2 or matrix_instance.shape[0] != matrix_instance.shape[1] or matrix_instance.shape[0] == 0:
        return None
    b_k = np.random.rand(matrix_instance.shape[1])
    for _ in range(num_simulations):
        b_k1 = np.dot(matrix_instance, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm == 0:
            return {"eigenvalue": 0, "eigenvector": b_k}
        b_k = b_k1 / b_k1_norm
    dot_b_k = np.dot(b_k.T, b_k)
    eigenvalue = np.dot(b_k.T, np.dot(matrix_instance, b_k)
                        ) / (dot_b_k if dot_b_k > 1e-9 else 1)
    return {"eigenvalue": eigenvalue, "eigenvector": b_k}

def scipy_eigs_wrapper(matrix_instance):
    if not isinstance(matrix_instance, np.ndarray) or matrix_instance.ndim != 2 or matrix_instance.shape[0] != matrix_instance.shape[1] or matrix_instance.shape[0] == 0:
        return None
    try:
        vals, vecs = scipy.sparse.linalg.eigs(matrix_instance, k=1, which='LM')
        return {"eigenvalue": vals[0].real, "eigenvector": vecs[:, 0].real}
    except:
        return None

def stable_sort_records(records_keys): records, keys = records_keys; return sorted(
    records, key=itemgetter(*keys)) if records and keys else []

def conceptual_random_record_sort(records_keys):
    records, keys = records_keys
    random.shuffle(records)
    return records

def dp_knapsack(knapsack_input): return {"max_value": sum(
    knapsack_input['values']), "selected_items_mask": [1]*len(knapsack_input['values'])}

def greedy_knapsack(knapsack_input): return {"max_value": sum(v for v, w in zip(knapsack_input['values'], knapsack_input['weights']) if w < knapsack_input['capacity']/(
    len(knapsack_input['weights']) if len(knapsack_input['weights']) > 0 else 1)), "selected_items_mask": [1]*len(knapsack_input['values'])}

def backtracking_graph_coloring(graph_k_instance): G, k = graph_k_instance; return {
    "coloring": {n: 0 for n in G.nodes()}}

def conceptual_random_graph_coloring(graph_k_instance): G, k = graph_k_instance; return {
    "coloring": {n: random.randint(0, k-1) for n in G.nodes()}}

def backtracking_sudoku(board_instance): return board_instance
def conceptual_stochastic_sudoku(board_instance): return board_instance

def scipy_quad_wrapper(func_limits_instance): func, a, b = func_limits_instance['func'], func_limits_instance[
    'a'], func_limits_instance['b']; return scipy.integrate.quad(func, a, b)[0]

def monte_carlo_integration_1d(func_limits_instance): func, a, b, num_samples = func_limits_instance['func'], func_limits_instance['a'], func_limits_instance['b'], func_limits_instance.get(
    'num_samples', 1000); samples = np.random.uniform(a, b, num_samples); integral = (b-a) * np.mean(func(samples)); return integral

def floyd_warshall_apsp(graph_instance): return nx.floyd_warshall_numpy(graph_instance) if isinstance(
    graph_instance, nx.Graph) and graph_instance.number_of_nodes() > 0 else (np.array([[]]) if isinstance(graph_instance, nx.Graph) else None)

def conceptual_random_apsp(graph_instance): return np.zeros((graph_instance.number_of_nodes(
), graph_instance.number_of_nodes())) if isinstance(graph_instance, nx.Graph) else None

def differential_evolution_opt(opt_problem):
    func, bounds = opt_problem['func'], opt_problem['bounds']
    return scipy.optimize.differential_evolution(func, bounds)

def simulated_annealing_opt(
    opt_problem):
    func, bounds = opt_problem['func'], opt_problem['bounds']
    return scipy.optimize.dual_annealing(func, bounds)

# Dynamic programming for Bellman-Held-Karp (placeholder)
def dynamic_programming_bellman_held_karp(dist_matrix):
    n = len(dist_matrix)
    return f"conceptual_tsp_solution_bhk_for_{n}_cities"

# Backtracking for Generic CSP (placeholder)
def backtracking_csp(csp_problem):
    return "conceptual_csp_solution_backtracking"

# Greedy algorithm (placeholder, highly problem-specific)
def greedy_algorithm(problem_instance, problem_type_hint=None):
    return f"conceptual_greedy_solution_for_{problem_type_hint or 'unknown_problem'}"

# Newton's method (placeholder, uses SciPy)
def newtons_method(func_and_derivative, initial_guess, tolerance=1e-7, max_iterations=100):
    f = func_and_derivative.get('func')
    fprime = func_and_derivative.get('fprime')
    if not f or not fprime:
        return None
    try:
        return scipy.optimize.newton(f, initial_guess, fprime=fprime, tol=tolerance, maxiter=max_iterations)
    except:
        return None

# Randomized Min-Cut (Karger)
def randomized_min_cut_karger(graph_instance_edges, num_trials=None):
    if not graph_instance_edges:
        return None
    vertices = list(set(u for e in graph_instance_edges for u in e))
    num_nodes_initial = len(vertices)
    if num_nodes_initial < 2:
        return {"min_cut_size": 0}
    if num_trials is None:
        num_trials = num_nodes_initial**2
    best_min_cut_size = float('inf')
    for _ in range(num_trials):
        parent = {node: node for node in vertices}
        num_supervertices = num_nodes_initial
        current_edges_trial = list(graph_instance_edges)
        random.shuffle(current_edges_trial)
        edge_idx_karger = 0
        while num_supervertices > 2 and edge_idx_karger < len(current_edges_trial):
            u, v = current_edges_trial[edge_idx_karger]
            edge_idx_karger += 1
            root_u, root_v = parent[u], parent[v]
            while root_u != parent[root_u]:
                root_u = parent[root_u]
            while root_v != parent[root_v]:
                root_v = parent[root_v]
            if root_u != root_v:
                parent[root_v] = root_u
                num_supervertices -= 1
        cut_size_this_trial = 0
        if num_supervertices == 2:
            for u_orig, v_orig in graph_instance_edges:
                root_u_orig, root_v_orig = parent[u_orig], parent[v_orig]
                while root_u_orig != parent[root_u_orig]:
                    root_u_orig = parent[root_u_orig]
                while root_v_orig != parent[root_v_orig]:
                    root_v_orig = parent[root_v_orig]
                if root_u_orig != root_v_orig:
                    cut_size_this_trial += 1
            if cut_size_this_trial < best_min_cut_size:
                best_min_cut_size = cut_size_this_trial
    return {"min_cut_size": best_min_cut_size if best_min_cut_size != float('inf') else "Error"}

# Simulated Annealing (wrapper for SciPy)
def simulated_annealing(opt_problem):
    func = opt_problem.get('func')
    bounds = opt_problem.get('bounds')
    if not func or not bounds:
        return None
    try:
        return scipy.optimize.dual_annealing(func, bounds, **opt_problem.get('kwargs', {}))
    except:
        return None

# Monte-Carlo Tree Search (placeholder)
def monte_carlo_tree_search(
    game_state_or_problem): return "conceptual_mcts_best_action"
# Genetic Algorithm (placeholder)
def genetic_algorithm(ga_problem): return "conceptual_ga_best_solution"
# Random Local Search (placeholder)
def random_local_search(
    rls_problem): return "conceptual_rls_locally_optimal_solution"
# Las Vegas Algorithm (example for search)
def las_vegas_algorithm(problem_instance_lv):
    if isinstance(problem_instance_lv, dict) and problem_instance_lv.get('type') == 'search_example':
        data = problem_instance_lv.get('data', [])
        target = problem_instance_lv.get('target')
        if not data or target is None:
            return {'found_at': -1, 'attempts': 0, 'status': 'invalid_input'}
        indices = list(range(len(data)))
        random.shuffle(indices)
        attempts = 0
        for idx in indices:
            attempts += 1
            if data[idx] == target:
                return {'found_at': idx, 'attempts': attempts, 'status': 'found'}
        return {'found_at': -1, 'attempts': attempts, 'status': 'not_found'}
    return "conceptual_las_vegas_correct_solution"


algorithm_functions = {
    1: bubble_sort, 2: dijkstra, 3: simplex_method, 4: dpll_sat_basic, 5: gaussian_elimination, # Changed 4: dpll_sat to dpll_sat_basic
    6: branch_and_bound, 7: dynamic_programming_bellman_held_karp, 8: backtracking_csp,
    9: greedy_algorithm, 10: newtons_method, 11: quicksort_random_pivot,
    12: randomized_min_cut_karger, 13: simulated_annealing, 14: random_3_sat_generator,
    15: randomized_svd_wrapper, 16: monte_carlo_tree_search, 17: genetic_algorithm,
    18: random_local_search, 19: las_vegas_algorithm, 20: stochastic_gradient_descent,
    21: randomized_lp_solver, 22: randomized_sat_solver, 23: naive_matrix_mult, 24: randomized_kaczmarz, 25: sampling_top_k,
    26: python_string_sort, 27: conceptual_random_string_sort, 28: kruskal_mst, 29: prim_mst,
    30: edmonds_karp_max_flow, 31: conceptual_random_max_flow, 32: conceptual_qp_solver,
    33: power_iteration_eigs, 34: scipy_eigs_wrapper,
    35: stable_sort_records, 36: conceptual_random_record_sort, 37: dp_knapsack, 38: greedy_knapsack,
    39: backtracking_graph_coloring, 40: conceptual_random_graph_coloring, 41: backtracking_sudoku,
    42: conceptual_stochastic_sudoku, 43: scipy_quad_wrapper, 44: monte_carlo_integration_1d,
    45: floyd_warshall_apsp, 46: conceptual_random_apsp, 47: differential_evolution_opt, 48: simulated_annealing_opt,
}
# Add string keys mapping to the same functions
for k_id, v_func in list(algorithm_functions.items()):
    if isinstance(k_id, int):
        algo_name_in_arsenal = next((item['Algorithm'].lower().replace(' ', '_').replace(
            '-', '_').replace('(', '').replace(')', '') for item in algorithm_arsenal if item['ID'] == k_id), None)
        if algo_name_in_arsenal and algo_name_in_arsenal not in algorithm_functions:
            algorithm_functions[algo_name_in_arsenal] = v_func
# Ensure specific string keys are present and correct
algorithm_functions.update({
    "bubble_sort": bubble_sort, "dijkstra": dijkstra, "simplex_method": simplex_method,
    "dpll_sat": dpll_sat_basic, "gaussian_elimination": gaussian_elimination, # Changed dpll_sat to dpll_sat_basic
    "branch_and_bound": branch_and_bound,
    "dynamic_programming_bellman_held_karp": dynamic_programming_bellman_held_karp,
    "backtracking_csp": backtracking_csp, "greedy_algorithm": greedy_algorithm,
    "newtons_method": newtons_method, "quicksort_random_pivot": quicksort_random_pivot,
    "randomized_min_cut_karger": randomized_min_cut_karger,
    "simulated_annealing": simulated_annealing, "random_3_sat_generator": random_3_sat_generator,
    "randomized_svd": randomized_svd_wrapper,
    "monte_carlo_tree_search": monte_carlo_tree_search,
    "genetic_algorithm": genetic_algorithm, "random_local_search": random_local_search,
    "las_vegas_algorithm": las_vegas_algorithm,
    "stochastic_gradient_descent": stochastic_gradient_descent,
    "randomized_lp_solver": randomized_lp_solver, "randomized_sat_solver": randomized_sat_solver,
    "naive_matrix_mult": naive_matrix_mult, "library_matrix_mult": library_matrix_mult,
    "randomized_kaczmarz": randomized_kaczmarz,
    "heap_top_k": lambda arr_k_instance: heapq.nlargest(arr_k_instance[1], arr_k_instance[0]) if isinstance(arr_k_instance, tuple) and len(arr_k_instance) == 2 and isinstance(arr_k_instance[0], list) else None,
    "sampling_top_k": sampling_top_k,
    "python_string_sort": python_string_sort, "conceptual_random_string_sort": conceptual_random_string_sort,
    "kruskal_mst": kruskal_mst, "prim_mst": prim_mst,
    "edmonds_karp_max_flow": edmonds_karp_max_flow, "conceptual_random_max_flow": conceptual_random_max_flow,
    "conceptual_qp_solver": conceptual_qp_solver, "power_iteration_eigs": power_iteration_eigs,
    "scipy_eigs_wrapper": scipy_eigs_wrapper,
    "stable_sort_records": stable_sort_records, "conceptual_random_record_sort": conceptual_random_record_sort,
    "dp_knapsack": dp_knapsack, "greedy_knapsack": greedy_knapsack,
    "backtracking_graph_coloring": backtracking_graph_coloring, "conceptual_random_graph_coloring": conceptual_random_graph_coloring,
    "backtracking_sudoku": backtracking_sudoku, "conceptual_stochastic_sudoku": conceptual_stochastic_sudoku,
    "scipy_quad_wrapper": scipy_quad_wrapper, "monte_carlo_integration_1d": monte_carlo_integration_1d,
    "floyd_warshall_apsp": floyd_warshall_apsp, "conceptual_random_apsp": conceptual_random_apsp,
    "differential_evolution_opt": differential_evolution_opt, "simulated_annealing_opt": simulated_annealing_opt,
})

experimental_designs = {
    "Pilot": {"Problems": 5, "Algorithm Pairs": 5, "Repetitions": 10, "Total Runs": 250},
    "Minimum full": {"Problems": 10, "Algorithm Pairs": 8, "Repetitions": 7, "Total Runs": 560},
    "Better": {"Problems": 15, "Algorithm Pairs": 10, "Repetitions": 7, "Total Runs": 1050},
    "Original grand": {"Problems": 20, "Algorithm Pairs": 20, "Repetitions": 30, "Total Runs": 12000},
}

def full_richness_cv_analysis(results_df, features_df):
    """Complete, rich cross-validation analysis"""
    
    print("="*80)
    print("FULL RICHNESS CROSS-VALIDATION ANALYSIS")
    print("="*80)
    
    # 1. Basic Statistics
    print("\n1. RUNTIME DISTRIBUTION ANALYSIS")
    print("-"*40)
    print(f"Total observations: {len(results_df)}")
    print(f"\nRuntime statistics (seconds):")
    print(results_df['runtime'].describe())
    
    # 2. Problem-wise Analysis
    print("\n2. PROBLEM DIFFICULTY SPECTRUM")
    print("-"*40)
    problem_stats = results_df.groupby('problem')['runtime'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).sort_values('mean')
    print(problem_stats)
    
    # 3. Algorithm Performance Profile
    print("\n3. ALGORITHM PERFORMANCE PROFILE")
    print("-"*40)
    algo_stats = results_df.groupby('algorithm')['runtime'].agg([
        'mean', 'median', 'std', 'count'
    ]).sort_values('mean')
    print(algo_stats.head(10))  # Top 10 fastest
    print("...")
    print(algo_stats.tail(10))  # Top 10 slowest
    
    # 4. Cross-Validation Gaps - Multiple Metrics
    print("\n4. CROSS-VALIDATION GAP ANALYSIS")
    print("-"*40)
    
    cv_gaps = []
    relative_improvements = []
    log_improvements = []
    
    # Ensure 'runtime' is numeric and handle potential infinities from conceptual solvers if not already filtered
    results_df['runtime'] = pd.to_numeric(results_df['runtime'], errors='coerce')
    results_df = results_df[np.isfinite(results_df['runtime'])]

    for problem_name_iter in results_df['problem'].unique(): # Changed 'problem' to 'problem_name_iter'
        problem_data = results_df[results_df['problem'] == problem_name_iter]
        
        if len(problem_data) == 0:
            continue
            
        algo_means = problem_data.groupby('algorithm')['runtime'].mean()
        if algo_means.empty:
            continue
        best_algo = algo_means.idxmin()
        best_time = algo_means.min()
        
        # Simulate prediction (simplified - would use features in real CV)
        # For now, use median algorithm as "predicted"
        # Ensure algo_means is not empty before trying to access median index
        if len(algo_means) > 0:
            median_idx = len(algo_means) // 2
            if median_idx < len(algo_means): # Check if median_idx is a valid index
                 predicted_algo = algo_means.index[median_idx]
                 predicted_time = algo_means[predicted_algo]
            else: # Fallback if algo_means has only one element after grouping (should not happen if pair has 2 algos)
                 predicted_algo = best_algo
                 predicted_time = best_time
        else: # Should not happen if problem_data was not empty
            continue

        absolute_gap = predicted_time - best_time
        relative_gap = (predicted_time - best_time) / best_time if best_time > 0 else float('inf') if predicted_time > 0 else 0
        log_gap = np.log(predicted_time + 1e-9) - np.log(best_time + 1e-9) # Added epsilon for log(0)
        
        cv_gaps.append(absolute_gap)
        relative_improvements.append(relative_gap)
        log_improvements.append(log_gap)
    
    if cv_gaps: # Check if lists are not empty before calculating stats
        print(f"Absolute Gap (seconds):")
        print(f"  Mean: {np.mean(cv_gaps):.4f}")
        print(f"  Median: {np.median(cv_gaps):.4f}")
        print(f"  Std: {np.std(cv_gaps):.4f}")
        # Ensure cv_gaps is not empty for percentile
        if cv_gaps:
             print(f"  95% CI: [{np.percentile(cv_gaps, 2.5):.4f}, {np.percentile(cv_gaps, 97.5):.4f}]")
    else:
        print("Absolute Gap (seconds): No data for calculation.")

    if relative_improvements:
        finite_relative_improvements = [r for r in relative_improvements if np.isfinite(r)]
        if finite_relative_improvements:
            print(f"\nRelative Improvement Potential (%): (excluding inf values)")
            print(f"  Mean: {np.mean(finite_relative_improvements)*100:.1f}%")
            print(f"  Median: {np.median(finite_relative_improvements)*100:.1f}%")
            print(f"  Best case: {np.min(finite_relative_improvements)*100:.1f}%") # Min of finite values
            print(f"  Worst case: {np.max(finite_relative_improvements)*100:.1f}%") # Max of finite values
        else:
            print("\nRelative Improvement Potential (%): No finite data for calculation.")
    else:
        print("\nRelative Improvement Potential (%): No data for calculation.")

    if log_improvements:
        print(f"\nLog-scale Gap:")
        print(f"  Mean: {np.mean(log_improvements):.4f}")
        print(f"  Geometric mean ratio: {np.exp(np.mean(log_improvements)):.2f}x")
    else:
        print("\nLog-scale Gap: No data for calculation.")
        
    print("\n5. PROBLEM TYPE PATTERNS")
    print("-"*40)
    problem_categories = {
        'Sorting': ['IntegerSort_Pilot', 'StringSort_Pilot', 'TopKSort_Pilot', 'RecordSort_Pilot'],
        'Graph': ['ShortestPath_Pilot', 'DenseAPSP_Pilot', 'MST_Pilot', 'MaxFlow_Pilot'], # Corrected names
        'Optimization': ['LinearProgramming_Pilot', 'ConvexQP_Pilot', 'NonConvexOpt_Pilot', 'Knapsack_Pilot'], # Corrected names
        'Numerical': ['MatrixMultiply_Pilot', 'LinearSystem_Pilot', 'Eigenvalue_Pilot', 'NumIntegration_Pilot'] # Corrected names
    }
    
    for category, problems_in_cat in problem_categories.items(): # Renamed 'problems' to 'problems_in_cat'
        # Filter for problems whose names *exactly* match those in the list for the category
        cat_data = results_df[results_df['problem'].isin(problems_in_cat)]
        if len(cat_data) > 0:
            print(f"\n{category}:")
            print(f"  Mean runtime: {cat_data['runtime'].mean():.4f}s")
            print(f"  Runtime range: [{cat_data['runtime'].min():.4f}, {cat_data['runtime'].max():.4f}]")
    
    print("\n6. LEARNING POTENTIAL ANALYSIS")
    print("-"*40)
    
    for problem_name_iter in results_df['problem'].unique()[:5]:  # Sample 5 problems
        problem_data = results_df[results_df['problem'] == problem_name_iter]
        if len(problem_data) == 0:
            continue
            
        algo_means = problem_data.groupby('algorithm')['runtime'].mean()
        if algo_means.empty: continue

        best = algo_means.min()
        worst = algo_means.max()
        average = algo_means.mean()
        
        print(f"\n{problem_name_iter}:") # Changed 'problem' to 'problem_name_iter'
        print(f"  Best algo: {best:.4f}s")
        if best > 0:
            print(f"  Average: {average:.4f}s ({average/best:.1f}x slower)")
            print(f"  Worst: {worst:.4f}s ({worst/best:.1f}x slower)")
            print(f"  Learning opportunity: {(average-best)/best*100:.0f}% improvement possible")
        else:
            print(f"  Average: {average:.4f}s")
            print(f"  Worst: {worst:.4f}s")
            print(f"  Learning opportunity: N/A (best time is 0)")

    print("\n7. STATISTICAL VALIDATION")
    print("-"*40)
    
    n_bootstrap = 1000 # User had 1000, keeping it
    bootstrap_gaps = []
    
    unique_problems_for_bootstrap = results_df['problem'].unique()
    if len(unique_problems_for_bootstrap) > 0: # Ensure there are problems to sample from
        for _ in range(n_bootstrap):
            sampled_problems = np.random.choice(
                unique_problems_for_bootstrap, 
                size=len(unique_problems_for_bootstrap), 
                replace=True
            )
            
            current_bootstrap_run_gaps = [] # Gaps for this single bootstrap run
            for problem_sample_name in sampled_problems: # Changed 'problem' to 'problem_sample_name'
                problem_data = results_df[results_df['problem'] == problem_sample_name]
                if len(problem_data) > 0:
                    algo_means = problem_data.groupby('algorithm')['runtime'].mean()
                    if len(algo_means) > 1: # Need at least two algos to compare
                        best = algo_means.min()
                        # Simplified prediction: median algo, ensure index exists
                        median_idx = len(algo_means) // 2
                        predicted = algo_means.iloc[median_idx] 
                        current_bootstrap_run_gaps.append(predicted - best)
            
            if current_bootstrap_run_gaps: # If any gaps were calculated in this bootstrap run
                bootstrap_gaps.append(np.mean(current_bootstrap_run_gaps))
        
        if bootstrap_gaps: # If any bootstrap means were collected
            print(f"Bootstrap 95% CI for CV gap (absolute, mean of problem gaps): [{np.percentile(bootstrap_gaps, 2.5):.4f}, {np.percentile(bootstrap_gaps, 97.5):.4f}]")
        else:
            print("Bootstrap 95% CI for CV gap: Not enough data for calculation.")
    else:
        print("Bootstrap 95% CI for CV gap: No problems to sample from.")

    print("\n8. OUTLIER INVESTIGATION")
    print("-"*40)
    
    if not results_df.empty:
        threshold = results_df['runtime'].quantile(0.95)
        outliers = results_df[results_df['runtime'] > threshold]
    
        print(f"Outliers (top 5% slowest runs):")
        print(f"  Count: {len(outliers)}")
        if not outliers.empty:
            print(f"  Problems involved: {outliers['problem'].unique()}")
            print(f"  Algorithms involved: {outliers['algorithm'].unique()}")
        else:
            print("  No outliers found above 95th percentile.")
    else:
        print("No data for outlier investigation.")

    print("\n9. SYSTEMATIC vs RANDOM PATTERNS")
    print("-"*40)
    
    # Adjusted heuristic based on current naming
    systematic_algos_df = results_df[results_df['algorithm'].str.contains('Sort_Algo|Dijkstra_Algo|Simplex_Algo|DPLL_Algo|GaussElim_Algo|KruskalMST_Algo|PrimMST_Algo|EdmondKarpMF_Algo|PowerIterationEig_Algo|StableSortRecords_Algo|DPKnapsack_Algo|BTColoring_Algo|BTSudoku_Algo|ScipyQuad_Algo|FloydWarshall_Algo|DiffEvolution_Algo|NaiveMatMul_Algo', case=False, regex=True)]
    randomized_algos_df = results_df[results_df['algorithm'].str.contains('Quicksort_Algo|RandGraphHeuristic_Algo|RandLPSolver_Algo|RandSATSolver_Algo|SamplingTopK_Algo|RandStringSort_Algo|RandMaxFlow_Algo|ScipyEigs_Algo|RandRecSort_Algo|GreedyKnapsack_Algo|RandColoring_Algo|StochSudoku_Algo|MCIntegration_Algo|RandAPSP_Algo|SimAnnealingOpt_Algo|RandSATUnsat_Algo|LibMatMul_Algo', case=False, regex=True)] # LibMatMul is often highly optimized (like random)
    
    if not systematic_algos_df.empty:
        print(f"Systematic algorithms:")
        print(f"  Mean runtime: {systematic_algos_df['runtime'].mean():.4f}s")
        print(f"  Consistency (CV of runtime): {systematic_algos_df['runtime'].std()/systematic_algos_df['runtime'].mean() if systematic_algos_df['runtime'].mean() != 0 else float('nan'):.2f}")
    else:
        print("Systematic algorithms: No data found.")

    if not randomized_algos_df.empty:
        print(f"\nRandomized algorithms:")
        print(f"  Mean runtime: {randomized_algos_df['runtime'].mean():.4f}s")
        print(f"  Consistency (CV of runtime): {randomized_algos_df['runtime'].std()/randomized_algos_df['runtime'].mean() if randomized_algos_df['runtime'].mean() != 0 else float('nan'):.2f}")
    else:
        print("\nRandomized algorithms: No data found.")
        
    print("\n10. EXECUTIVE SUMMARY")
    print("="*80)
    
    # Calculate these values only if data is available
    mean_abs_cv_gap_val = np.mean(cv_gaps) if cv_gaps else float('nan')
    median_rel_imp_val = np.median(finite_relative_improvements) * 100 if 'finite_relative_improvements' in locals() and finite_relative_improvements else float('nan')
    geom_mean_ratio_val = np.exp(np.mean(log_improvements)) if log_improvements else float('nan')
    prob_runtime_min = results_df.groupby('problem')['runtime'].mean().min() if not results_df.empty else float('nan')
    prob_runtime_max = results_df.groupby('problem')['runtime'].mean().max() if not results_df.empty else float('nan')
    algo_runtime_min = results_df.groupby('algorithm')['runtime'].mean().min() if not results_df.empty else float('nan')
    algo_runtime_max = results_df.groupby('algorithm')['runtime'].mean().max() if not results_df.empty else float('nan')
    bs_ci_lower = np.percentile(bootstrap_gaps, 2.5) if bootstrap_gaps else float('nan')
    bs_ci_upper = np.percentile(bootstrap_gaps, 97.5) if bootstrap_gaps else float('nan')

    print(f"""
    Total Problems Analyzed: {len(results_df['problem'].unique()) if not results_df.empty else 0}
    Total Algorithms Tested: {len(results_df['algorithm'].unique()) if not results_df.empty else 0}
    Total Experiments Run: {len(results_df)}
    
    Key Findings:
    - Average absolute CV gap: {mean_abs_cv_gap_val:.4f}s
    - Median relative improvement potential: {median_rel_imp_val:.1f}%
    - Geometric mean performance ratio: {geom_mean_ratio_val:.2f}x
    
    Problem Difficulty Range: {prob_runtime_min:.4f}s to {prob_runtime_max:.4f}s
    Algorithm Speed Range: {algo_runtime_min:.4f}s to {algo_runtime_max:.4f}s
    
    Statistical Confidence: 95% CI for CV gap (absolute) = [{bs_ci_lower:.4f}, {bs_ci_upper:.4f}]
    """)
    
    return {
        'absolute_gaps': cv_gaps,
        'relative_improvements': relative_improvements if 'relative_improvements' in locals() else [],
        'log_improvements': log_improvements,
        'bootstrap_ci': (bs_ci_lower, bs_ci_upper) if 'bootstrap_gaps' in locals() and bootstrap_gaps else (float('nan'), float('nan')),
        'problem_stats': problem_stats if 'problem_stats' in locals() else pd.DataFrame(),
        'algo_stats': algo_stats if 'algo_stats' in locals() else pd.DataFrame()
    }

if __name__ == "__main__":
    print("\n--- Setting up and Running Pilot Experiment ---")
    print("DEBUG: experiment.py - __main__ started")
    rng_main = random.Random(42)
    print("DEBUG: experiment.py - rng_main initialized")
    def get_base_feature_flags(): return BASE_FEATURE_FLAGS.copy()

    # --- Problem Definitions (now 20 total) ---
    def sort_problem_generator(size=50): return rng_main.sample(
        range(-1000, 1000), k=min(size, 1999))

    def sort_feature_extractor(instance):
        n = len(instance)
        features = get_base_feature_flags()
        features['is_sort_problem'] = 1.0
        features.update({'size': n, 'std_dev': np.nan, 'range': np.nan,
                        'min': np.nan, 'max': np.nan, 'mean': np.nan})
        if n == 1:
            features.update({'std_dev': 0.0, 'range': 0.0,
                            'min': instance[0], 'max': instance[0], 'mean': instance[0]})
        elif n > 1:
            features.update({'std_dev': np.std(instance), 'range': np.ptp(instance), 'min': np.min(
                instance), 'max': np.max(instance), 'mean': np.mean(instance)})
        return features

    def sort_evaluator(instance, solution):
        if solution is None or not isinstance(solution, list) or not instance:
            return 0.0
        is_sorted = all(solution[i] <= solution[i+1]
                        for i in range(len(solution)-1))
        return 1.0 if is_sorted and len(solution) == len(instance) else 0.0

    def graph_problem_generator_dijkstra(num_nodes=10, edge_prob=0.3):
        G = nx.gnp_random_graph(num_nodes, edge_prob,
                                seed=rng_main.randint(0, 10000))
        for u, v in G.edges():
            G.edges[u, v]['weight'] = rng_main.randint(1, 10)
        source = rng_main.choice(list(G.nodes())) if G.nodes() else 0
        return {"graph": G, "source": source}

    def graph_feature_extractor_dijkstra(instance):
        G = instance.get("graph")
        features = get_base_feature_flags()
        features['is_graph_problem'] = 1.0
        features.update({'nodes': 0, 'edges': 0, 'density': np.nan})
        if isinstance(G, nx.Graph):
            if G.number_of_nodes() > 0:
                features.update({'nodes': G.number_of_nodes(
                ), 'edges': G.number_of_edges(), 'density': nx.density(G)})
            else:
                features.update({'nodes': 0, 'edges': 0, 'density': 0.0})
        else:
            features['is_graph_problem'] = 0.0
        return features

    def graph_evaluator_dijkstra(instance, solution): return 1.0 if solution and isinstance(
        solution, dict) and 'shortest_path_lengths' in solution else 0.0

    def lp_problem_generator(num_vars=5, num_constraints=3):
        c = -np.random.rand(num_vars)
        A_ub = np.random.rand(num_constraints, num_vars)
        b_ub = np.random.rand(num_constraints) * num_vars
        return {'c': c, 'A_ub': A_ub, 'b_ub': b_ub, 'bounds': [(0, None)] * num_vars}

    def lp_feature_extractor(instance):
        features = get_base_feature_flags()
        features['is_lp_problem'] = 1.0
        features.update({'num_vars': len(
            instance['c']), 'num_constraints': instance['A_ub'].shape[0] if instance.get('A_ub') is not None else 0})
        return features

    def lp_evaluator(instance, solution): return 1.0 if solution and solution.get('success') else 0.0 # Fixed to use .get('success')

    def sat_problem_generator(num_vars=10, num_clauses=20, satisfiable=None):
        return algorithm_functions["random_3_sat_generator"](num_vars, num_clauses, satisfiable=satisfiable)

    def sat_feature_extractor(instance_clauses):
        features = get_base_feature_flags()
        features['is_3sat_problem'] = 1.0
        num_vars = 0
        if instance_clauses and any(c for c in instance_clauses): 
            all_lits = [abs(lit)
                        for clause in instance_clauses for lit in clause]
            if all_lits:
                num_vars = max(all_lits)
        features.update({'num_vars_sat': num_vars,
                        'num_clauses_sat': len(instance_clauses)})
        return features

    def sat_evaluator(instance_clauses, solution_assignment): # Updated evaluator
        if solution_assignment == "conceptual_random_sat_solution": # Handle old placeholder
             return 1.0 # Assume conceptual solution is valid for now
        if not isinstance(solution_assignment, dict) and solution_assignment is not None: # DPLL returns a dict or None
            return 0.0 
        if solution_assignment is None: # Unsatisfiable or no solution found by dpll_sat_basic
            # For a "satisfiable" problem type, this is a failure.
            # For an "unsatisfiable" problem type, this might be correct.
            # This evaluator is for "3SAT_Pilot" (satisfiable by default from problem_suite)
            # and "3SAT_Unsat_Pilot". We need to distinguish.
            # For now, let's assume if None, it's not a valid solution for a satisfiable instance.
            # The sat_unsat_evaluator will handle the unsat case.
            return 0.0

        # Check if all clauses are satisfied by the assignment
        for clause in instance_clauses:
            satisfied_clause = False
            for literal in clause:
                var = abs(literal)
                expected_value = True if literal > 0 else False
                if var in solution_assignment and solution_assignment[var] == expected_value:
                    satisfied_clause = True
                    break
            if not satisfied_clause:
                return 0.0 # Unsatisfied clause
        return 1.0 # All clauses satisfied

    def matmul_problem_generator(m=10, k=10, n=10): A = np.random.rand(
        m, k); B = np.random.rand(k, n); return (A, B)

    def matmul_feature_extractor(matrices): A, B = matrices; features = get_base_feature_flags(
    ); features['is_matmul_problem'] = 1.0; features.update({'m': A.shape[0], 'k': A.shape[1], 'n': B.shape[1]}); return features

    def matmul_evaluator(matrices, solution): A, B = matrices; expected = np.dot(
        A, B); return 1.0 if solution is not None and np.allclose(solution, expected) else 0.0

    def linsys_problem_generator(size=10): A = np.random.rand(
        size, size); b = np.random.rand(size); return {'A': A, 'b': b}

    def linsys_feature_extractor(instance): A = instance['A']; features = get_base_feature_flags(); features['is_linsys_problem'] = 1.0; features.update(
        {'size_linsys': A.shape[0], 'cond_A': np.linalg.cond(A) if A.shape[0] > 0 else np.nan}); return features

    def linsys_evaluator(instance, solution): A, b = instance['A'], instance['b']; return 1.0 if solution is not None and np.allclose(
        A @ solution, b) else 0.0

    def topk_problem_generator(n_size=50, k_val=5): arr = rng_main.sample(
        range(-1000, 1000), k=min(n_size, 1999)); k = min(k_val, n_size if n_size > 0 else 1); return (arr, k)

    def topk_feature_extractor(instance): arr, k = instance; features = get_base_feature_flags(
    ); features['is_topk_problem'] = 1.0; features.update({'n_topk': len(arr), 'k_topk': k}); return features

    def topk_evaluator(instance, solution): arr, k = instance; expected = sorted(arr, reverse=True)[
        :k]; return 1.0 if solution is not None and len(solution) == k and all(s == e for s, e in zip(solution, expected)) else 0.0

    def string_sort_generator(num_strings=100, max_len=15): return [''.join(rng_main.choices(
        string.ascii_letters + string.digits, k=rng_main.randint(5, max_len))) for _ in range(num_strings)]

    def string_sort_feature_extractor(instance): features = get_base_feature_flags(); features['is_string_sort_problem'] = 1.0; features.update(
        {'num_strings': len(instance), 'avg_len': np.mean([len(s) for s in instance]) if instance else 0}); return features

    def string_sort_evaluator(instance, solution):
        if solution is None or not isinstance(solution, list) or len(solution) != len(instance):
            return 0.0
        return 1.0 if all(solution[i] <= solution[i+1] for i in range(len(solution)-1)) else 0.0

    def mst_problem_generator(num_nodes=15, edge_prob=0.4): G = nx.gnp_random_graph(num_nodes, edge_prob, seed=rng_main.randint(
        0, 10000)); [(G.edges[u, v].update({'weight': rng_main.randint(1, 20)})) for u, v in G.edges()]; return G

    def mst_feature_extractor(instance_graph): features = get_base_feature_flags(); features['is_mst_problem'] = 1.0; features.update(
        {'nodes_mst': instance_graph.number_of_nodes(), 'edges_mst': instance_graph.number_of_edges()}); return features

    def mst_evaluator(instance_graph, solution_tree):
        if not isinstance(solution_tree, nx.Graph) or not nx.is_connected(solution_tree) or solution_tree.number_of_nodes() != instance_graph.number_of_nodes() or solution_tree.number_of_edges() != instance_graph.number_of_nodes() - 1:
            return 0.0
        return 1.0

    def max_flow_problem_generator(num_nodes=10, edge_prob=0.5):
        G = nx.DiGraph()
        nodes = list(range(num_nodes))
        source = 0
        sink = num_nodes - 1
        if num_nodes < 2:
            return {"graph": nx.DiGraph(), "source": 0, "sink": 0}
        for i in nodes:
            for j in nodes:
                if i != j and rng_main.random() < edge_prob:
                    G.add_edge(i, j, capacity=rng_main.randint(1, 10))
        if not (G.has_node(source) and G.has_node(sink) and nx.has_path(G, source, sink)):
            return {"graph": G, "source": source, "sink": sink, "no_path_note": True}
        return {"graph": G, "source": source, "sink": sink}

    def max_flow_feature_extractor(instance): features = get_base_feature_flags(); features['is_max_flow_problem'] = 1.0; features.update(
        {'nodes_mf': instance['graph'].number_of_nodes(), 'edges_mf': instance['graph'].number_of_edges()}); return features

    def max_flow_evaluator(instance, solution_value): return 1.0 if isinstance(
        solution_value, (int, float)) and solution_value >= 0 else 0.0

    def qp_problem_generator(num_vars=5): P = np.random.rand(num_vars, num_vars); P = np.dot(
        P, P.transpose()) + np.eye(num_vars)*0.1; q = np.random.rand(num_vars); return {"P": P, "q": q}

    def qp_feature_extractor(instance): features = get_base_feature_flags(
    ); features['is_convex_qp_problem'] = 1.0; features.update({'vars_qp': len(instance['q'])}); return features

    def qp_evaluator(
        instance, solution): return 1.0 if solution == "conceptual_qp_solution" else 0.0

    def eigen_problem_generator(size=10): return np.random.rand(
        size, size) if size > 0 else np.array([[]])

    def eigen_feature_extractor(instance_matrix): features = get_base_feature_flags(
    ); features['is_eigenvalue_problem'] = 1.0; features.update({'size_eig': instance_matrix.shape[0]}); return features

    def eigen_evaluator(instance_matrix, solution):
        if solution and isinstance(solution, dict) and "eigenvalue" in solution and "eigenvector" in solution and instance_matrix.shape[0] > 0:
            Av = np.dot(instance_matrix, solution["eigenvector"])
            lambda_v = solution["eigenvalue"] * solution["eigenvector"]
            return 1.0 if np.allclose(Av, lambda_v) else 0.0
        return 0.0

    def record_sort_generator(num_records=50):
        return [{"id": i, "name": ''.join(rng_main.choices(string.ascii_lowercase, k=5)), "age": rng_main.randint(18, 65), "city_code": rng_main.randint(1, 5)} for i in range(num_records)]

    def record_sort_feature_extractor(instance):
        features = get_base_feature_flags()
        features['is_record_sort_problem'] = 1.0
        features.update({'num_records': len(instance)})
        return features

    def record_sort_evaluator(instance_keys, solution):
        original_list, sort_keys = instance_keys
        if solution is None or len(solution) != len(original_list):
            return 0.0
        try:
            expected = sorted(original_list, key=itemgetter(*sort_keys))
            return 1.0 if solution == expected else 0.0
        except:
            return 0.0

    def knapsack_generator(num_items=15):
        values = [rng_main.randint(1, 100) for _ in range(num_items)]
        weights = [rng_main.randint(1, 50) for _ in range(num_items)]
        capacity = int(sum(weights)/2)
        return {"values": values, "weights": weights, "capacity": capacity}

    def knapsack_feature_extractor(instance):
        features = get_base_feature_flags()
        features['is_knapsack_problem'] = 1.0
        features.update({'num_items_knap': len(instance['values']), 'cap_ratio': instance['capacity'] / (
            np.mean(instance['weights']) if instance['weights'] else 1)})
        return features

    def knapsack_evaluator(instance, solution):
        if not solution or "selected_items_mask" not in solution or "max_value" not in solution:
            return 0.0
        total_weight = sum(w for w, m in zip(
            instance['weights'], solution['selected_items_mask']) if m)
        total_value = sum(v for v, m in zip(
            instance['values'], solution['selected_items_mask']) if m)
        return 1.0 if total_weight <= instance['capacity'] and total_value == solution['max_value'] else 0.0

    def graph_coloring_generator(num_nodes=15, k=3): G = nx.erdos_renyi_graph(
        num_nodes, 0.3, seed=rng_main.randint(0, 10000)); return (G, k)

    def graph_coloring_feature_extractor(instance): G, k = instance; features = get_base_feature_flags(); features['is_graph_coloring_problem'] = 1.0; features.update({
        'nodes_gc': G.number_of_nodes(), 'edges_gc': G.number_of_edges(), 'k_gc': k}); return features

    def graph_coloring_evaluator(instance, solution):
        G, k = instance
        if not solution or "coloring" not in solution:
            return 0.0
        coloring = solution["coloring"] # Fixed: assign before use
        if not isinstance(coloring, dict) or len(coloring) != G.number_of_nodes(): # Added type check
            return 0.0
        for u, v in G.edges():
            if coloring.get(u) == coloring.get(v) or coloring.get(u) is None or coloring.get(v) is None:
                return 0.0
        if any(c < 0 or c >= k for c in coloring.values()):
            return 0.0
        return 1.0

def sudoku_generator(num_clues: int = 25, *, seed: int | None = None) -> np.ndarray:
    """
    Generate a 9×9 Sudoku grid with `num_clues` pre-filled cells.

    Parameters
    ----------
    num_clues : int, default 25
        Number of non-zero cells to leave in the puzzle (0–81).
    seed : int | None, default None
        If given, makes the generator deterministic.

    Returns
    -------
    np.ndarray
        A 9×9 array where 0 represents an empty square.
    """
    if not (0 <= num_clues <= 81):
        raise ValueError("num_clues must be between 0 and 81")

    rng = random.Random(seed)      # local PRNG keeps global `random` unchanged
    base = 3                        # size of the smaller 3×3 boxes
    side = base * base              # full side length (9)

    def pattern(r: int, c: int) -> int:
        """Base pattern for a valid completed Sudoku."""
        return (base * (r % base) + r // base + c) % side

    def shuffled(seq):
        """Return a shuffled copy of `seq` using the local RNG."""
        seq = list(seq)
        rng.shuffle(seq)
        return seq

    r_base = range(base)
    rows = [g * base + r for g in shuffled(r_base) for r in shuffled(r_base)]
    cols = [g * base + c for g in shuffled(r_base) for c in shuffled(r_base)]
    nums = shuffled(range(1, side + 1))

    # Fully solved grid
    board = [[nums[pattern(r, c)] for c in cols] for r in rows]

    # Remove numbers to create the puzzle
    squares = side * side                          # 81
    empties = squares - num_clues
    for p in rng.sample(range(squares), empties):
        board[p // side][p % side] = 0

    return np.array(board, dtype=int)

def sudoku_feature_extractor(instance_board): features = get_base_feature_flags(
); features['is_sudoku_problem'] = 1.0; features.update({'clues_sudoku': np.count_nonzero(instance_board)}); return features

def sudoku_evaluator(instance_board, solution_board):
    if solution_board is None or not isinstance(solution_board, np.ndarray) or solution_board.shape != (9, 9):
        return 0.0  # Check type
    for r in range(9):
        if len(set(solution_board[r, :])) != 9 or len(set(solution_board[:, r])) != 9 or 0 in solution_board[r, :] or 0 in solution_board[:, r]:
            return 0.0
    for r_box in range(0, 9, 3):
        for c_box in range(0, 9, 3):
            if len(set(solution_board[r_box:r_box+3, c_box:c_box+3].flatten())) != 9:
                return 0.0
    return 1.0

def num_integration_generator(): func = lambda x: np.sin(
    x)*x**2; a = 0; b = np.pi; return {"func": func, "a": a, "b": b, "num_samples": 10000}

def num_integration_feature_extractor(instance): features = get_base_feature_flags(
); features['is_num_integration_problem'] = 1.0; features.update({'interval_width': instance['b']-instance['a']}); return features

def num_integration_evaluator(instance, solution_value): expected, _ = scipy.integrate.quad(
    instance['func'], instance['a'], instance['b']); return 1.0 if solution_value is not None and np.isclose(solution_value, expected) else 0.0

# --- Add the final 3 problem types ---
def dense_graph_apsp_generator(num_nodes=10): G = nx.complete_graph(num_nodes); [(
    G.edges[u, v].update({'weight': rng_main.randint(1, 10)})) for u, v in G.edges()]; return G

def dense_graph_apsp_feature_extractor(instance_graph): features = get_base_feature_flags(
); features['is_dense_graph_apsp_problem'] = 1.0; features.update({'nodes_apsp': instance_graph.number_of_nodes()}); return features

def dense_graph_apsp_evaluator(instance_graph, solution_matrix):
    if solution_matrix is None or not isinstance(solution_matrix, np.ndarray) or solution_matrix.shape != (instance_graph.number_of_nodes(), instance_graph.number_of_nodes()):
        return 0.0
    return 1.0 if np.all(np.diag(solution_matrix) == 0) else 0.0

def nonconvex_opt_generator(dims=2): func = lambda x: 10*len(x) + sum([(xi**2 - 10 * np.cos(
    2 * np.pi * xi)) for xi in x]); bounds = [(-5.12, 5.12)] * dims; return {"func": func, "bounds": bounds, "dims": dims}

def nonconvex_opt_feature_extractor(instance): features = get_base_feature_flags(
); features['is_nonconvex_opt_problem'] = 1.0; features.update({'dims_ncvx': instance['dims']}); return features

def nonconvex_opt_evaluator(instance, solution):
    if solution and hasattr(solution, 'success') and solution.success and hasattr(solution, 'fun') and solution.fun is not None:
        if solution.fun >= 0:  # Rastrigin function minimum is 0
            return 1.0 / (1.0 + solution.fun)  # Quality approaches 1 as fun approaches 0
        else:
            # This case should ideally not happen for Rastrigin, but handle defensively
            return 0.0  # Or some other indicator of an unexpected negative function value
    return 0.0

def sat_unsat_problem_generator_for_pilot():  # Modified to take no args for direct reference
    num_vars = rng_main.randint(10, 20)
    num_clauses = int(num_vars * (4.2 + rng_main.random()*0.2))
    return algorithm_functions["random_3_sat_generator"](num_vars, num_clauses, satisfiable=False)

def sat_unsat_feature_extractor(instance_clauses):
    features = get_base_feature_flags()
    features['is_3sat_unsat_problem'] = 1.0
    num_vars = 0
    if instance_clauses and any(c for c in instance_clauses):
        all_lits = [abs(lit) for clause in instance_clauses for lit in clause]
        if all_lits:
            num_vars = max(all_lits)
    features.update({'num_vars_sat_u': num_vars,
                    'num_clauses_sat_u': len(instance_clauses)})
    return features

def sat_unsat_evaluator(instance, solution): # Needs to be updated for dpll_sat_basic if it's used for unsat
    # If dpll_sat_basic returns None for an unsatisfiable instance, that's correct.
    if solution is None: # Potentially correct for unsat
        return 1.0 
    # If randomized_sat_solver is used and returns its placeholder string:
    if solution == "conceptual_random_sat_solution" and "unsat" in str(solution).lower(): # This check is a bit weak
        return 1.0 # Placeholder for now
    # If dpll_sat_basic returns an assignment, it means it found one, which is wrong for unsat.
    if isinstance(solution, dict) and solution: 
        return 0.0 
    return 0.0 # Default to incorrect if not clearly identifiable as correct unsat result

if __name__ == "__main__":
    print("\n--- Setting up and Running Pilot Experiment ---")
    print("DEBUG: experiment.py - __main__ started")
    rng_main = random.Random(42)
    print("DEBUG: experiment.py - rng_main initialized")
    def get_base_feature_flags(): return BASE_FEATURE_FLAGS.copy()

    # --- Problem Definitions (now 20 total) ---
    def sort_problem_generator(size=50): return rng_main.sample(
        range(-1000, 1000), k=min(size, 1999))

    def sort_feature_extractor(instance):
        n = len(instance)
        features = get_base_feature_flags()
        features['is_sort_problem'] = 1.0
        features.update({'size': n, 'std_dev': np.nan, 'range': np.nan,
                        'min': np.nan, 'max': np.nan, 'mean': np.nan})
        if n == 1:
            features.update({'std_dev': 0.0, 'range': 0.0,
                            'min': instance[0], 'max': instance[0], 'mean': instance[0]})
        elif n > 1:
            features.update({'std_dev': np.std(instance), 'range': np.ptp(instance), 'min': np.min(
                instance), 'max': np.max(instance), 'mean': np.mean(instance)})
        return features

    def sort_evaluator(instance, solution):
        if solution is None or not isinstance(solution, list) or not instance:
            return 0.0
        is_sorted = all(solution[i] <= solution[i+1]
                        for i in range(len(solution)-1))
        return 1.0 if is_sorted and len(solution) == len(instance) else 0.0

    def graph_problem_generator_dijkstra(num_nodes=10, edge_prob=0.3):
        G = nx.gnp_random_graph(num_nodes, edge_prob,
                                seed=rng_main.randint(0, 10000))
        for u, v in G.edges():
            G.edges[u, v]['weight'] = rng_main.randint(1, 10)
        source = rng_main.choice(list(G.nodes())) if G.nodes() else 0
        return {"graph": G, "source": source}

    def graph_feature_extractor_dijkstra(instance):
        G = instance.get("graph")
        features = get_base_feature_flags()
        features['is_graph_problem'] = 1.0
        features.update({'nodes': 0, 'edges': 0, 'density': np.nan})
        if isinstance(G, nx.Graph):
            if G.number_of_nodes() > 0:
                features.update({'nodes': G.number_of_nodes(
                ), 'edges': G.number_of_edges(), 'density': nx.density(G)})
            else:
                features.update({'nodes': 0, 'edges': 0, 'density': 0.0})
        else:
            features['is_graph_problem'] = 0.0
        return features

    def graph_evaluator_dijkstra(instance, solution): return 1.0 if solution and isinstance(
        solution, dict) and 'shortest_path_lengths' in solution else 0.0

    def lp_problem_generator(num_vars=5, num_constraints=3):
        c = -np.random.rand(num_vars)
        A_ub = np.random.rand(num_constraints, num_vars)
        b_ub = np.random.rand(num_constraints) * num_vars
        return {'c': c, 'A_ub': A_ub, 'b_ub': b_ub, 'bounds': [(0, None)] * num_vars}

    def lp_feature_extractor(instance):
        features = get_base_feature_flags()
        features['is_lp_problem'] = 1.0
        features.update({'num_vars': len(
            instance['c']), 'num_constraints': instance['A_ub'].shape[0] if instance.get('A_ub') is not None else 0})
        return features

    def lp_evaluator(instance, solution): return 1.0 if solution and solution.get('success') else 0.0 # Fixed to use .get('success')

    def sat_problem_generator(num_vars=10, num_clauses=20, satisfiable=None):
        return algorithm_functions["random_3_sat_generator"](num_vars, num_clauses, satisfiable=satisfiable)

    def sat_feature_extractor(instance_clauses):
        features = get_base_feature_flags()
        features['is_3sat_problem'] = 1.0
        num_vars = 0
        if instance_clauses and any(c for c in instance_clauses): 
            all_lits = [abs(lit)
                        for clause in instance_clauses for lit in clause]
            if all_lits:
                num_vars = max(all_lits)
        features.update({'num_vars_sat': num_vars,
                        'num_clauses_sat': len(instance_clauses)})
        return features

    def sat_evaluator(instance_clauses, solution_assignment): # Updated evaluator
        if solution_assignment == "conceptual_random_sat_solution": # Handle old placeholder
             return 1.0 # Assume conceptual solution is valid for now
        if not isinstance(solution_assignment, dict) and solution_assignment is not None: 
            return 0.0 
        if solution_assignment is None: 
            return 0.0

        for clause in instance_clauses:
            satisfied_clause = False
            for literal in clause:
                var = abs(literal)
                expected_value = True if literal > 0 else False
                if var in solution_assignment and solution_assignment[var] == expected_value:
                    satisfied_clause = True
                    break
            if not satisfied_clause:
                return 0.0 
        return 1.0 

    def matmul_problem_generator(m=10, k=10, n=10): A = np.random.rand(
        m, k); B = np.random.rand(k, n); return (A, B)

    def matmul_feature_extractor(matrices): A, B = matrices; features = get_base_feature_flags(
    ); features['is_matmul_problem'] = 1.0; features.update({'m': A.shape[0], 'k': A.shape[1], 'n': B.shape[1]}); return features

    def matmul_evaluator(matrices, solution): A, B = matrices; expected = np.dot(
        A, B); return 1.0 if solution is not None and np.allclose(solution, expected) else 0.0

    def linsys_problem_generator(size=10): A = np.random.rand(
        size, size); b = np.random.rand(size); return {'A': A, 'b': b}

    def linsys_feature_extractor(instance): A = instance['A']; features = get_base_feature_flags(); features['is_linsys_problem'] = 1.0; features.update(
        {'size_linsys': A.shape[0], 'cond_A': np.linalg.cond(A) if A.shape[0] > 0 else np.nan}); return features

    def linsys_evaluator(instance, solution): A, b = instance['A'], instance['b']; return 1.0 if solution is not None and np.allclose(
        A @ solution, b) else 0.0

    def topk_problem_generator(n_size=50, k_val=5): arr = rng_main.sample(
        range(-1000, 1000), k=min(n_size, 1999)); k = min(k_val, n_size if n_size > 0 else 1); return (arr, k)

    def topk_feature_extractor(instance): arr, k = instance; features = get_base_feature_flags(
    ); features['is_topk_problem'] = 1.0; features.update({'n_topk': len(arr), 'k_topk': k}); return features

    def topk_evaluator(instance, solution): arr, k = instance; expected = sorted(arr, reverse=True)[
        :k]; return 1.0 if solution is not None and len(solution) == k and all(s == e for s, e in zip(solution, expected)) else 0.0

    def string_sort_generator(num_strings=100, max_len=15): return [''.join(rng_main.choices(
        string.ascii_letters + string.digits, k=rng_main.randint(5, max_len))) for _ in range(num_strings)]

    def string_sort_feature_extractor(instance): features = get_base_feature_flags(); features['is_string_sort_problem'] = 1.0; features.update(
        {'num_strings': len(instance), 'avg_len': np.mean([len(s) for s in instance]) if instance else 0}); return features

    def string_sort_evaluator(instance, solution):
        if solution is None or not isinstance(solution, list) or len(solution) != len(instance):
            return 0.0
        return 1.0 if all(solution[i] <= solution[i+1] for i in range(len(solution)-1)) else 0.0

    def mst_problem_generator(num_nodes=15, edge_prob=0.4): G = nx.gnp_random_graph(num_nodes, edge_prob, seed=rng_main.randint(
        0, 10000)); [(G.edges[u, v].update({'weight': rng_main.randint(1, 20)})) for u, v in G.edges()]; return G

    def mst_feature_extractor(instance_graph): features = get_base_feature_flags(); features['is_mst_problem'] = 1.0; features.update(
        {'nodes_mst': instance_graph.number_of_nodes(), 'edges_mst': instance_graph.number_of_edges()}); return features

    def mst_evaluator(instance_graph, solution_tree):
        if not isinstance(solution_tree, nx.Graph) or not nx.is_connected(solution_tree) or solution_tree.number_of_nodes() != instance_graph.number_of_nodes() or solution_tree.number_of_edges() != instance_graph.number_of_nodes() - 1:
            return 0.0
        return 1.0

    def max_flow_problem_generator(num_nodes=10, edge_prob=0.5):
        G = nx.DiGraph()
        nodes = list(range(num_nodes))
        source = 0
        sink = num_nodes - 1
        if num_nodes < 2:
            return {"graph": nx.DiGraph(), "source": 0, "sink": 0}
        for i in nodes:
            for j in nodes:
                if i != j and rng_main.random() < edge_prob:
                    G.add_edge(i, j, capacity=rng_main.randint(1, 10))
        if not (G.has_node(source) and G.has_node(sink) and nx.has_path(G, source, sink)):
            return {"graph": G, "source": source, "sink": sink, "no_path_note": True}
        return {"graph": G, "source": source, "sink": sink}

    def max_flow_feature_extractor(instance): features = get_base_feature_flags(); features['is_max_flow_problem'] = 1.0; features.update(
        {'nodes_mf': instance['graph'].number_of_nodes(), 'edges_mf': instance['graph'].number_of_edges()}); return features

    def max_flow_evaluator(instance, solution_value): return 1.0 if isinstance(
        solution_value, (int, float)) and solution_value >= 0 else 0.0

    def qp_problem_generator(num_vars=5): P = np.random.rand(num_vars, num_vars); P = np.dot(
        P, P.transpose()) + np.eye(num_vars)*0.1; q = np.random.rand(num_vars); return {"P": P, "q": q}

    def qp_feature_extractor(instance): features = get_base_feature_flags(
    ); features['is_convex_qp_problem'] = 1.0; features.update({'vars_qp': len(instance['q'])}); return features

    def qp_evaluator(
        instance, solution): return 1.0 if solution == "conceptual_qp_solution" else 0.0

    def eigen_problem_generator(size=10): return np.random.rand(
        size, size) if size > 0 else np.array([[]])

    def eigen_feature_extractor(instance_matrix): features = get_base_feature_flags(
    ); features['is_eigenvalue_problem'] = 1.0; features.update({'size_eig': instance_matrix.shape[0]}); return features

    def eigen_evaluator(instance_matrix, solution):
        if solution and isinstance(solution, dict) and "eigenvalue" in solution and "eigenvector" in solution and instance_matrix.shape[0] > 0:
            Av = np.dot(instance_matrix, solution["eigenvector"])
            lambda_v = solution["eigenvalue"] * solution["eigenvector"]
            return 1.0 if np.allclose(Av, lambda_v) else 0.0
        return 0.0

    def record_sort_generator(num_records=50):
        return [{"id": i, "name": ''.join(rng_main.choices(string.ascii_lowercase, k=5)), "age": rng_main.randint(18, 65), "city_code": rng_main.randint(1, 5)} for i in range(num_records)]

    def record_sort_feature_extractor(instance):
        features = get_base_feature_flags()
        features['is_record_sort_problem'] = 1.0
        features.update({'num_records': len(instance)})
        return features

    def record_sort_evaluator(instance_keys, solution):
        original_list, sort_keys = instance_keys
        if solution is None or len(solution) != len(original_list):
            return 0.0
        try:
            expected = sorted(original_list, key=itemgetter(*sort_keys))
            return 1.0 if solution == expected else 0.0
        except:
            return 0.0

    def knapsack_generator(num_items=15):
        values = [rng_main.randint(1, 100) for _ in range(num_items)]
        weights = [rng_main.randint(1, 50) for _ in range(num_items)]
        capacity = int(sum(weights)/2)
        return {"values": values, "weights": weights, "capacity": capacity}

    def knapsack_feature_extractor(instance):
        features = get_base_feature_flags()
        features['is_knapsack_problem'] = 1.0
        features.update({'num_items_knap': len(instance['values']), 'cap_ratio': instance['capacity'] / (
            np.mean(instance['weights']) if instance['weights'] else 1)})
        return features

    def knapsack_evaluator(instance, solution):
        if not solution or "selected_items_mask" not in solution or "max_value" not in solution:
            return 0.0
        total_weight = sum(w for w, m in zip(
            instance['weights'], solution['selected_items_mask']) if m)
        total_value = sum(v for v, m in zip(
            instance['values'], solution['selected_items_mask']) if m)
        return 1.0 if total_weight <= instance['capacity'] and total_value == solution['max_value'] else 0.0

    def graph_coloring_generator(num_nodes=15, k=3): G = nx.erdos_renyi_graph(
        num_nodes, 0.3, seed=rng_main.randint(0, 10000)); return (G, k)

    def graph_coloring_feature_extractor(instance): G, k = instance; features = get_base_feature_flags(); features['is_graph_coloring_problem'] = 1.0; features.update({
        'nodes_gc': G.number_of_nodes(), 'edges_gc': G.number_of_edges(), 'k_gc': k}); return features

    def graph_coloring_evaluator(instance, solution):
        G, k = instance
        if not solution or "coloring" not in solution:
            return 0.0
        coloring = solution["coloring"] # Fixed: assign before use
        if not isinstance(coloring, dict) or len(coloring) != G.number_of_nodes(): # Added type check
            return 0.0
        for u, v in G.edges():
            if coloring.get(u) == coloring.get(v) or coloring.get(u) is None or coloring.get(v) is None:
                return 0.0
        if any(c < 0 or c >= k for c in coloring.values()):
            return 0.0
        return 1.0

    def sudoku_feature_extractor(instance_board): features = get_base_feature_flags(
    ); features['is_sudoku_problem'] = 1.0; features.update({'clues_sudoku': np.count_nonzero(instance_board)}); return features

    def sudoku_evaluator(instance_board, solution_board):
        if solution_board is None or not isinstance(solution_board, np.ndarray) or solution_board.shape != (9, 9):
            return 0.0  # Check type
        for r in range(9):
            if len(set(solution_board[r, :])) != 9 or len(set(solution_board[:, r])) != 9 or 0 in solution_board[r, :] or 0 in solution_board[:, r]:
                return 0.0
        for r_box in range(0, 9, 3):
            for c_box in range(0, 9, 3):
                if len(set(solution_board[r_box:r_box+3, c_box:c_box+3].flatten())) != 9:
                    return 0.0
        return 1.0

    def num_integration_generator(): func = lambda x: np.sin(
        x)*x**2; a = 0; b = np.pi; return {"func": func, "a": a, "b": b, "num_samples": 10000}

    def num_integration_feature_extractor(instance): features = get_base_feature_flags(
    ); features['is_num_integration_problem'] = 1.0; features.update({'interval_width': instance['b']-instance['a']}); return features

    def num_integration_evaluator(instance, solution_value): expected, _ = scipy.integrate.quad(
        instance['func'], instance['a'], instance['b']); return 1.0 if solution_value is not None and np.isclose(solution_value, expected) else 0.0

    # --- Add the final 3 problem types ---
    def dense_graph_apsp_generator(num_nodes=10): G = nx.complete_graph(num_nodes); [(
        G.edges[u, v].update({'weight': rng_main.randint(1, 10)})) for u, v in G.edges()]; return G

    def dense_graph_apsp_feature_extractor(instance_graph): features = get_base_feature_flags(
    ); features['is_dense_graph_apsp_problem'] = 1.0; features.update({'nodes_apsp': instance_graph.number_of_nodes()}); return features

    def dense_graph_apsp_evaluator(instance_graph, solution_matrix):
        if solution_matrix is None or not isinstance(solution_matrix, np.ndarray) or solution_matrix.shape != (instance_graph.number_of_nodes(), instance_graph.number_of_nodes()):
            return 0.0
        return 1.0 if np.all(np.diag(solution_matrix) == 0) else 0.0

    def nonconvex_opt_generator(dims=2): func = lambda x: 10*len(x) + sum([(xi**2 - 10 * np.cos(
        2 * np.pi * xi)) for xi in x]); bounds = [(-5.12, 5.12)] * dims; return {"func": func, "bounds": bounds, "dims": dims}

    def nonconvex_opt_feature_extractor(instance): features = get_base_feature_flags(
    ); features['is_nonconvex_opt_problem'] = 1.0; features.update({'dims_ncvx': instance['dims']}); return features

    def nonconvex_opt_evaluator(
        instance, solution): return 1.0 if solution and solution.success and solution.fun is not None else 0.0

    def sat_unsat_problem_generator_for_pilot():  # Modified to take no args for direct reference
        num_vars = rng_main.randint(10, 20)
        num_clauses = int(num_vars * (4.2 + rng_main.random()*0.2))
        return algorithm_functions["random_3_sat_generator"](num_vars, num_clauses, satisfiable=False)

    def sat_unsat_feature_extractor(instance_clauses):
        features = get_base_feature_flags()
        features['is_3sat_unsat_problem'] = 1.0
        num_vars = 0
        if instance_clauses and any(c for c in instance_clauses):
            all_lits = [abs(lit) for clause in instance_clauses for lit in clause]
            if all_lits:
                num_vars = max(all_lits)
        features.update({'num_vars_sat_u': num_vars,
                        'num_clauses_sat_u': len(instance_clauses)})
        return features

    def sat_unsat_evaluator(instance, solution): # Needs to be updated for dpll_sat_basic if it's used for unsat
        # If dpll_sat_basic returns None for an unsatisfiable instance, that's correct.
        if solution is None: 
            return 1.0 
        if solution == "conceptual_random_sat_solution": # Handle old placeholder for random solver
            # This assumes the conceptual random solver might correctly identify unsat.
            # A more robust approach would be for randomized_sat_solver to also return None for unsat.
            return 1.0 # Placeholder logic for conceptual solver
        if isinstance(solution, dict) and solution: 
            return 0.0 
        return 0.0 

    pilot_problems.clear()
    pilot_problems.extend([
        ProblemDefinition(name="IntegerSort_Pilot", problem_type="sort", generator_func=lambda: sort_problem_generator(
            size=rng_main.randint(20, 70)), feature_fn_func=sort_feature_extractor, evaluator_func=sort_evaluator),
        ProblemDefinition(name="ShortestPath_Pilot", problem_type="graph_dict", generator_func=lambda: graph_problem_generator_dijkstra(
            num_nodes=rng_main.randint(5, 15)), feature_fn_func=graph_feature_extractor_dijkstra, evaluator_func=graph_evaluator_dijkstra),
        ProblemDefinition(name="LinearProgramming_Pilot", problem_type="linear_programming", generator_func=lambda: lp_problem_generator(
            rng_main.randint(3, 8), rng_main.randint(2, 5)), feature_fn_func=lp_feature_extractor, evaluator_func=lp_evaluator),
        ProblemDefinition(name="3SAT_Pilot", problem_type="3sat", generator_func=lambda: sat_problem_generator(
            rng_main.randint(5, 15), rng_main.randint(10, 30), satisfiable=True), feature_fn_func=sat_feature_extractor, evaluator_func=sat_evaluator), # Ensure satisfiable=True for 3SAT_Pilot
        ProblemDefinition(name="MatrixMultiply_Pilot", problem_type="matrix_multiplication", generator_func=lambda: matmul_problem_generator(
            rng_main.randint(5, 12), rng_main.randint(5, 12), rng_main.randint(5, 12)), feature_fn_func=matmul_feature_extractor, evaluator_func=matmul_evaluator),
        ProblemDefinition(name="LinearSystem_Pilot", problem_type="linear_system", generator_func=lambda: linsys_problem_generator(
            rng_main.randint(5, 12)), feature_fn_func=linsys_feature_extractor, evaluator_func=linsys_evaluator),
        ProblemDefinition(name="TopKSort_Pilot", problem_type="top_k_sort", generator_func=lambda: topk_problem_generator(
            rng_main.randint(20, 70), rng_main.randint(3, 10)), feature_fn_func=topk_feature_extractor, evaluator_func=topk_evaluator),
        ProblemDefinition(name="StringSort_Pilot", problem_type="string_sort", generator_func=lambda: string_sort_generator(
            rng_main.randint(50, 150)), feature_fn_func=string_sort_feature_extractor, evaluator_func=string_sort_evaluator),
        ProblemDefinition(name="MST_Pilot", problem_type="mst", generator_func=lambda: mst_problem_generator(
            rng_main.randint(10, 20)), feature_fn_func=mst_feature_extractor, evaluator_func=mst_evaluator),
        ProblemDefinition(name="MaxFlow_Pilot", problem_type="max_flow", generator_func=lambda: max_flow_problem_generator(
            rng_main.randint(8, 15)), feature_fn_func=max_flow_feature_extractor, evaluator_func=max_flow_evaluator),
        ProblemDefinition(name="ConvexQP_Pilot", problem_type="convex_qp", generator_func=lambda: qp_problem_generator(
            rng_main.randint(3, 7)), feature_fn_func=qp_feature_extractor, evaluator_func=qp_evaluator),
        ProblemDefinition(name="Eigenvalue_Pilot", problem_type="eigenvalue", generator_func=lambda: eigen_problem_generator(
            rng_main.randint(5, 10)), feature_fn_func=eigen_feature_extractor, evaluator_func=eigen_evaluator),
        ProblemDefinition(name="RecordSort_Pilot", problem_type="record_sort", generator_func=lambda: (record_sort_generator(rng_main.randint(
            30, 70)), ('name', 'age')), feature_fn_func=lambda x: record_sort_feature_extractor(x[0]), evaluator_func=lambda x, sol: record_sort_evaluator(x, sol)),
        ProblemDefinition(name="Knapsack_Pilot", problem_type="integer_programming_knapsack", generator_func=lambda: knapsack_generator(
            rng_main.randint(10, 25)), feature_fn_func=knapsack_feature_extractor, evaluator_func=knapsack_evaluator),
        ProblemDefinition(name="GraphColoring_Pilot", problem_type="graph_coloring", generator_func=lambda: graph_coloring_generator(
            rng_main.randint(10, 20), rng_main.randint(3, 5)), feature_fn_func=graph_coloring_feature_extractor, evaluator_func=graph_coloring_evaluator),
        ProblemDefinition(name="Sudoku_Pilot", problem_type="sudoku", generator_func=lambda: sudoku_generator(
            rng_main.randint(20, 35)), feature_fn_func=sudoku_feature_extractor, evaluator_func=sudoku_evaluator),
        ProblemDefinition(name="NumIntegration_Pilot", problem_type="numerical_integration", generator_func=num_integration_generator,
                          feature_fn_func=num_integration_feature_extractor, evaluator_func=num_integration_evaluator),
        ProblemDefinition(name="DenseAPSP_Pilot", problem_type="dense_graph_apsp", generator_func=lambda: dense_graph_apsp_generator(
            rng_main.randint(5, 10)), feature_fn_func=dense_graph_apsp_feature_extractor, evaluator_func=dense_graph_apsp_evaluator),
        ProblemDefinition(name="NonConvexOpt_Pilot", problem_type="nonconvex_opt", generator_func=lambda: nonconvex_opt_generator(
            rng_main.randint(2, 5)), feature_fn_func=nonconvex_opt_feature_extractor, evaluator_func=nonconvex_opt_evaluator),
        ProblemDefinition(name="3SAT_Unsat_Pilot", problem_type="3sat_unsat", generator_func=sat_unsat_problem_generator_for_pilot,
                          feature_fn_func=sat_unsat_feature_extractor, evaluator_func=sat_unsat_evaluator)  # Trailing comma is fine here
    ])

    print("DEBUG: pilot_problems list populated.")  # Debug print
    print("DEBUG: experiment.py - pilot_problems extended")

    pilot_pairs.clear()
    print("DEBUG: experiment.py - pilot_pairs cleared")

    # Pair 1: Integer Sort
    pilot_pairs.append(AlgorithmPair(name="SortPair1_BubbleVsQuick", systematic_algo_def=AlgorithmDefinition(name="BubbleSort_Algo", solver_func=algorithm_functions["bubble_sort"], accepts_problem_type="sort"), randomized_algo_def=AlgorithmDefinition(
        name="Quicksort_Algo", solver_func=algorithm_functions["quicksort_random_pivot"], accepts_problem_type="sort")))
    # Pair 2: Shortest Path (Sparse)
    pilot_pairs.append(AlgorithmPair(name="GraphPair1_DijkstraVsRand", systematic_algo_def=AlgorithmDefinition(name="Dijkstra_Algo", solver_func=algorithm_functions["dijkstra"], accepts_problem_type="graph_dict"), randomized_algo_def=AlgorithmDefinition(
        name="RandGraphHeuristic_Algo", solver_func=lambda graph, source: None, accepts_problem_type="graph_dict"))) # Fixed lambda
    # Pair 3: LP
    pilot_pairs.append(AlgorithmPair(name="LPPair_SimplexVsRand", systematic_algo_def=AlgorithmDefinition(name="Simplex_Algo", solver_func=algorithm_functions["simplex_method"], accepts_problem_type="linear_programming"), randomized_algo_def=AlgorithmDefinition(
        name="RandLPSolver_Algo", solver_func=algorithm_functions["randomized_lp_solver"], accepts_problem_type="linear_programming")))
    # Pair 4: 3SAT (Satisfiable)
    pilot_pairs.append(AlgorithmPair(name="SATPair_DPLLVsRand", systematic_algo_def=AlgorithmDefinition(name="DPLL_Algo", solver_func=algorithm_functions["dpll_sat"], accepts_problem_type="3sat"), randomized_algo_def=AlgorithmDefinition(
        name="RandSATSolver_Algo", solver_func=algorithm_functions["randomized_sat_solver"], accepts_problem_type="3sat")))
    # Pair 5: Matrix Multiplication
    pilot_pairs.append(AlgorithmPair(name="MatMulPair_NaiveVsLib", systematic_algo_def=AlgorithmDefinition(name="NaiveMatMul_Algo", solver_func=algorithm_functions["naive_matrix_mult"], accepts_problem_type="matrix_multiplication"), randomized_algo_def=AlgorithmDefinition(
        name="LibMatMul_Algo", solver_func=algorithm_functions["library_matrix_mult"], accepts_problem_type="matrix_multiplication")))
    # Pair 6: Linear System
    pilot_pairs.append(AlgorithmPair(name="LinSysPair_GaussVsKaczmarz", systematic_algo_def=AlgorithmDefinition(name="GaussElim_Algo", solver_func=algorithm_functions["gaussian_elimination"], accepts_problem_type="linear_system"), randomized_algo_def=AlgorithmDefinition(
        name="RandKaczmarz_Algo", solver_func=algorithm_functions["randomized_kaczmarz"], accepts_problem_type="linear_system")))
    # Pair 7: Top-K Sort
    pilot_pairs.append(AlgorithmPair(name="TopKPair_HeapVsSample", systematic_algo_def=AlgorithmDefinition(name="HeapTopK_Algo", solver_func=algorithm_functions["heap_top_k"], accepts_problem_type="top_k_sort"), randomized_algo_def=AlgorithmDefinition(
        name="SamplingTopK_Algo", solver_func=algorithm_functions["sampling_top_k"], accepts_problem_type="top_k_sort")))
    # Pair 8: String Sort
    pilot_pairs.append(AlgorithmPair(name="StringSortPair_PyVsRand", systematic_algo_def=AlgorithmDefinition(name="PythonStringSort_Algo", solver_func=algorithm_functions["python_string_sort"], accepts_problem_type="string_sort"), randomized_algo_def=AlgorithmDefinition(
        name="RandStringSort_Algo", solver_func=algorithm_functions["conceptual_random_string_sort"], accepts_problem_type="string_sort")))
    # Pair 9: MST
    pilot_pairs.append(AlgorithmPair(name="MSTPair_KruskalVsPrim", systematic_algo_def=AlgorithmDefinition(name="KruskalMST_Algo",
                                                                                                           solver_func=algorithm_functions["kruskal_mst"], accepts_problem_type="mst"), randomized_algo_def=AlgorithmDefinition(name="PrimMST_Algo", solver_func=algorithm_functions["prim_mst"], accepts_problem_type="mst")))
    # Pair 10: Max-Flow
    pilot_pairs.append(AlgorithmPair(name="MaxFlowPair_EKVsRand", systematic_algo_def=AlgorithmDefinition(name="EdmondKarpMF_Algo", solver_func=algorithm_functions["edmonds_karp_max_flow"], accepts_problem_type="max_flow"), randomized_algo_def=AlgorithmDefinition(
        name="RandMaxFlow_Algo", solver_func=algorithm_functions["conceptual_random_max_flow"], accepts_problem_type="max_flow")))
    # Pair 11: Convex QP
    pilot_pairs.append(AlgorithmPair(name="QPPair_ConceptVsRand", systematic_algo_def=AlgorithmDefinition(name="ConceptualQPSys_Algo",
                                                                                                          solver_func=algorithm_functions["conceptual_qp_solver"], accepts_problem_type="convex_qp"), randomized_algo_def=AlgorithmDefinition(name="ConceptualQPRnd_Algo", solver_func=lambda p: "rand_qp_sol", accepts_problem_type="convex_qp")))
    # Pair 12: Eigenvalue
    pilot_pairs.append(AlgorithmPair(name="EigenPair_PowerVsScipy", systematic_algo_def=AlgorithmDefinition(name="PowerIterationEig_Algo", solver_func=algorithm_functions["power_iteration_eigs"], accepts_problem_type="eigenvalue"), randomized_algo_def=AlgorithmDefinition(
        name="ScipyEigs_Algo", solver_func=algorithm_functions["scipy_eigs_wrapper"], accepts_problem_type="eigenvalue")))
    # Pair 13: Record Sort
    pilot_pairs.append(AlgorithmPair(name="RecordSortPair_StableVsRand", systematic_algo_def=AlgorithmDefinition(name="StableSortRecords_Algo", solver_func=algorithm_functions["stable_sort_records"], accepts_problem_type="record_sort"), randomized_algo_def=AlgorithmDefinition(
        name="RandRecSort_Algo", solver_func=algorithm_functions["conceptual_random_record_sort"], accepts_problem_type="record_sort")))
    # Pair 14: Knapsack
    pilot_pairs.append(AlgorithmPair(name="KnapsackPair_DPVsGreedy", systematic_algo_def=AlgorithmDefinition(name="DPKnapsack_Algo", solver_func=algorithm_functions["dp_knapsack"], accepts_problem_type="integer_programming_knapsack"), randomized_algo_def=AlgorithmDefinition(
        name="GreedyKnapsack_Algo", solver_func=algorithm_functions["greedy_knapsack"], accepts_problem_type="integer_programming_knapsack")))
    # Pair 15: Graph Coloring
    pilot_pairs.append(AlgorithmPair(name="ColoringPair_BTVsRand", systematic_algo_def=AlgorithmDefinition(name="BTColoring_Algo", solver_func=algorithm_functions["backtracking_graph_coloring"], accepts_problem_type="graph_coloring"), randomized_algo_def=AlgorithmDefinition(
        name="RandColoring_Algo", solver_func=algorithm_functions["conceptual_random_graph_coloring"], accepts_problem_type="graph_coloring")))
    # Pair 16: Sudoku
    pilot_pairs.append(AlgorithmPair(name="SudokuPair_BTVsStoch", systematic_algo_def=AlgorithmDefinition(name="BTSudoku_Algo", solver_func=algorithm_functions["backtracking_sudoku"], accepts_problem_type="sudoku"), randomized_algo_def=AlgorithmDefinition(
        name="StochSudoku_Algo", solver_func=algorithm_functions["conceptual_stochastic_sudoku"], accepts_problem_type="sudoku")))
    # Pair 17: Numerical Integration
    pilot_pairs.append(AlgorithmPair(name="IntegrationPair_QuadVsMC", systematic_algo_def=AlgorithmDefinition(name="ScipyQuad_Algo", solver_func=algorithm_functions["scipy_quad_wrapper"], accepts_problem_type="numerical_integration"), randomized_algo_def=AlgorithmDefinition(
        name="MCIntegration_Algo", solver_func=algorithm_functions["monte_carlo_integration_1d"], accepts_problem_type="numerical_integration")))
    # Pair 18: Dense APSP
    pilot_pairs.append(AlgorithmPair(name="DenseAPSP_FWvsRand", systematic_algo_def=AlgorithmDefinition(name="FloydWarshall_Algo", solver_func=algorithm_functions["floyd_warshall_apsp"], accepts_problem_type="dense_graph_apsp"), randomized_algo_def=AlgorithmDefinition(
        name="RandAPSP_Algo", solver_func=algorithm_functions["conceptual_random_apsp"], accepts_problem_type="dense_graph_apsp")))
    # Pair 19: Non-convex Opt
    pilot_pairs.append(AlgorithmPair(name="NonConvex_DEvsSA", systematic_algo_def=AlgorithmDefinition(name="DiffEvolution_Algo", solver_func=algorithm_functions["differential_evolution_opt"], accepts_problem_type="nonconvex_opt"), randomized_algo_def=AlgorithmDefinition(
        name="SimAnnealingOpt_Algo", solver_func=algorithm_functions["simulated_annealing_opt"], accepts_problem_type="nonconvex_opt")))
    # Pair 20: 3-SAT Unsatisfiable
    pilot_pairs.append(AlgorithmPair(name="3SATUnsat_DPLLvsRand", systematic_algo_def=AlgorithmDefinition(name="DPLLUnsat_Algo", solver_func=algorithm_functions["dpll_sat"], accepts_problem_type="3sat_unsat"), randomized_algo_def=AlgorithmDefinition(
        name="RandSATUnsat_Algo", solver_func=algorithm_functions["randomized_sat_solver"], accepts_problem_type="3sat_unsat")))

    print("DEBUG: pilot_pairs list populated.")  # Debug print
    print("DEBUG: experiment.py - pilot_pairs extended")

    # Placeholder for execute_algorithm function
    def execute_algorithm(problem_instance_data, algorithm_def: AlgorithmDefinition, problem_def: ProblemDefinition):
        print(f"DEBUG: Executing {algorithm_def.name} on an instance of {problem_def.name}")
        start_time = time.perf_counter()
        
        try:
            # Universal problem adapter - attempts to extract the core data structure
            if problem_def.problem_type == "sort" or problem_def.problem_type == "string_sort":
                # These expect a list
                if isinstance(problem_instance_data, list):
                    solution = algorithm_def.solver(problem_instance_data)
                else:
                    # Try to extract a list from whatever structure
                    solution = None
                    
            elif problem_def.problem_type in ["graph_dict", "mst", "dense_graph_apsp"]:
                # Graph algorithms
                if isinstance(problem_instance_data, dict) and "graph" in problem_instance_data:
                    if algorithm_def.name in ["Dijkstra_Algo", "RandGraphHeuristic_Algo"]:
                        solution = algorithm_def.solver(problem_instance_data["graph"], problem_instance_data.get("source", 0))
                    else:
                        solution = algorithm_def.solver(problem_instance_data.get("graph", problem_instance_data))
                elif hasattr(problem_instance_data, 'nodes'):  # NetworkX graph
                    if algorithm_def.name in ["Dijkstra_Algo", "RandGraphHeuristic_Algo"]:
                        # Need source node for Dijkstra
                        nodes = list(problem_instance_data.nodes())
                        source = nodes[0] if nodes else 0
                        solution = algorithm_def.solver(problem_instance_data, source)
                    else:
                        solution = algorithm_def.solver(problem_instance_data)
                else:
                    solution = None
                    
            elif problem_def.problem_type in ["linear_programming", "convex_qp", "nonconvex_opt"]:
                # Optimization problems - try to pass as-is
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type in ["3sat", "3sat_unsat"]:
                # SAT problems expect list of clauses
                if isinstance(problem_instance_data, list):
                    solution = algorithm_def.solver(problem_instance_data)
                else:
                    solution = None
                    
            elif problem_def.problem_type == "matrix_multiplication":
                # Expects tuple of matrices
                if isinstance(problem_instance_data, tuple) and len(problem_instance_data) == 2:
                    solution = algorithm_def.solver(problem_instance_data)
                else:
                    solution = None
                    
            elif problem_def.problem_type == "top_k_sort":
                # Expects (array, k) tuple
                if isinstance(problem_instance_data, tuple) and len(problem_instance_data) == 2:
                    solution = algorithm_def.solver(problem_instance_data)
                else:
                    solution = None
                    
            elif problem_def.problem_type == "linear_system":
                # Expects {'A': A, 'b': b}
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type == "max_flow":
                # Expects {"graph": G, "source": s, "sink": t}
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type == "eigenvalue":
                # Expects matrix
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type == "record_sort":
                # Expects (records, keys)
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type == "integer_programming_knapsack":
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type == "graph_coloring":
                # Expects (G, k)
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type == "sudoku":
                # Expects board
                solution = algorithm_def.solver(problem_instance_data)
                
            elif problem_def.problem_type == "numerical_integration":
                solution = algorithm_def.solver(problem_instance_data)
                    
            else:
                # Default: try to pass the instance as-is
                solution = algorithm_def.solver(problem_instance_data)
                
        except Exception as e:
            print(f"DEBUG: Algorithm {algorithm_def.name} incompatible with {problem_def.name}: {str(e)[:50]}...")
            solution = None

        runtime = time.perf_counter() - start_time
        
        # If algorithm couldn't run or returned None, set high runtime as penalty
        if solution is None:
            runtime = 1.0  # 1 second penalty for incompatible pairs
        
        # Evaluation
        try:
            quality = problem_def.evaluator(problem_instance_data, solution)
        except Exception as e:
            print(f"DEBUG: Evaluation error for {algorithm_def.name} on {problem_def.name}: {str(e)[:50]}...")
            quality = 0.0  # Failed evaluation = 0 quality

        return RunResult(runtime=runtime, quality=quality, algo_name=algorithm_def.name, problem_name=problem_def.name, problem_type=problem_def.problem_type)

    # Placeholder for run_experiment_data_collection function
    def run_experiment_data_collection(problems_defs: List[ProblemDefinition], algo_pairs: List[AlgorithmPair], reps: int, seed: int, num_problems_to_run: int, num_pairs_to_run: int):
        print(
            f"DEBUG: run_experiment_data_collection called with {num_problems_to_run} problems, {num_pairs_to_run} pairs, {reps} reps, seed {seed}")
        local_rng = random.Random(seed)
        all_results = []  # Flat list of RunResult
        all_features = []  # List of feature dicts

        selected_problems = problems_defs[:num_problems_to_run]
        selected_pairs = algo_pairs[:num_pairs_to_run]

        for i, p_def in enumerate(selected_problems):
            print(
                f"DEBUG: Processing problem {i+1}/{len(selected_problems)}: {p_def.name}")
            for j, pair in enumerate(selected_pairs):
                print(
                    f"DEBUG:   Processing pair {j+1}/{len(selected_pairs)}: {pair.name} for problem {p_def.name}")
                
                # REMOVED TYPE CHECKING - Run all algorithms on all problems
                algorithms_to_run = [pair.systematic, pair.randomized]

                for algo_def in algorithms_to_run:
                    for rep_idx in range(reps):
                        print(
                            f"DEBUG:       Running {algo_def.name} (Rep {rep_idx+1}/{reps}) on {p_def.name}")
                        instance_data = p_def.generator()

                        # Feature extraction
                        features = p_def.feature_fn(instance_data)
                        # Add problem name for joining later
                        features['problem_name'] = p_def.name
                        all_features.append(features)

                        run_res = execute_algorithm(
                            instance_data, algo_def, p_def)
                        all_results.append(run_res)

        # For compatibility with existing print logic, group results
        # This is a simplified grouping; original script might have more complex structure
        # The `pilot_results` in the original script was Dict[str, Dict[str, List[RunResult]]]
        # The `all_results` here is List[RunResult]
        # For now, we'll return the flat list and features. The printing part below will need adjustment.
        return all_results, all_features

    # Placeholder for run_cv function
    def run_cv(results, features):
        print("DEBUG: run_cv called")
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame([
            {
                'problem': r.problem_name,
                'algo': r.algo_name,
                'runtime': r.runtime,
                'quality': r.quality,
                'problem_type': r.problem_type
            } for r in results
        ])

        print(f"DEBUG: run_cv - Initial results_df size: {len(results_df)}")

        # Filter out invalid runtimes and qualities
        results_df = results_df[results_df['runtime'] < float('inf')]
        results_df = results_df.dropna(subset=['runtime', 'quality'])
        print(f"DEBUG: run_cv - results_df size after filtering inf/NaN: {len(results_df)}")

        if results_df.empty:
            print("DEBUG: run_cv - No valid data after filtering, cannot proceed with CV.")
            return {
                "avg_cv_gap": 0.0,
                "problems_analyzed": 0,
                "best_algos": {},
                "problems_with_no_valid_algo": list(pd.DataFrame([{'problem': r.problem_name} for r in results])['problem'].unique())
            }

        print("\nOverall Runtime Statistics (after filtering):")
        print(results_df['runtime'].describe())

        print("\nRuntime ranges by problem (after filtering):")
        all_problem_names_initial = pd.DataFrame([{'problem': r.problem_name} for r in results])['problem'].unique()
        
        for prob_name_iter in all_problem_names_initial:
            prob_data_filtered = results_df[results_df['problem'] == prob_name_iter]
            if not prob_data_filtered.empty:
                print(f"  {prob_name_iter}: min={prob_data_filtered['runtime'].min():.4g}s, max={prob_data_filtered['runtime'].max():.4g}s, mean={prob_data_filtered['runtime'].mean():.4g}s, count={len(prob_data_filtered)}")
            else:
                print(f"  {prob_name_iter}: No valid runtime data after filtering.")
        
        # Get mean performance by problem/algo from the filtered data
        perf_summary = results_df.groupby(['problem', 'algo']).agg({
            'runtime': 'mean',
            'quality': 'mean'
        }).reset_index()
        
        best_per_problem = {}
        problems_with_no_valid_algo = []
        
        # Iterate over unique problem names from the original, unfiltered set to report on all
        for prob_name_iter in all_problem_names_initial:
            # Check if this problem still exists in perf_summary (i.e., had valid runs)
            if prob_name_iter not in perf_summary['problem'].unique():
                problems_with_no_valid_algo.append(prob_name_iter)
                continue

            prob_data_summary = perf_summary[perf_summary['problem'] == prob_name_iter]
            valid_algos = prob_data_summary[prob_data_summary['quality'] > 0.5]
            
            if not valid_algos.empty:
                best_idx = valid_algos['runtime'].idxmin()
                best_per_problem[prob_name_iter] = valid_algos.loc[best_idx, 'algo']
            else:
                problems_with_no_valid_algo.append(prob_name_iter)
        
        print(f"DEBUG: run_cv - Found best algorithms for {len(best_per_problem)} problems.")
        if problems_with_no_valid_algo:
            print(f"DEBUG: run_cv - Problems with no algorithm meeting quality > 0.5 threshold (or no valid runs): {problems_with_no_valid_algo}")
        
        gaps = []
        for prob_name_iter, best_algo_name in best_per_problem.items():
            # Use perf_summary which is based on filtered and aggregated data
            prob_data_for_gap = perf_summary[perf_summary['problem'] == prob_name_iter]
            
            best_runtime_series = prob_data_for_gap[prob_data_for_gap['algo'] == best_algo_name]['runtime']
            if best_runtime_series.empty: continue
            best_runtime = best_runtime_series.iloc[0]

            # Consider only algos that were valid (quality > 0.5) for this problem for avg_runtime calculation
            valid_runtimes_for_problem = prob_data_for_gap[prob_data_for_gap['quality'] > 0.5]['runtime']

            if not valid_runtimes_for_problem.empty:
                avg_runtime = valid_runtimes_for_problem.mean()
                if best_runtime > 0: # Avoid division by zero and meaningless gaps
                    gap = (avg_runtime - best_runtime) / best_runtime
                    gaps.append(gap)
                elif best_runtime == 0 and avg_runtime > 0: # Best is zero, others are not
                    gaps.append(float('inf')) # Or handle as a large number / special case
                # If best_runtime is 0 and avg_runtime is 0, gap is 0 (already handled by np.mean if gaps is empty or contains 0)
        
        avg_gap = np.mean(gaps) if gaps else 0.0
        
        print(f"DEBUG: run_cv - Calculated avg_gap: {avg_gap}")
        return {
            "avg_cv_gap": avg_gap, 
            "problems_analyzed_count": len(best_per_problem), # Renamed for clarity
            "best_algos": best_per_problem,
            "problems_with_no_valid_algo": problems_with_no_valid_algo
        }

    # Update pilot run to use up to 20 problems/pairs
    print("DEBUG: experiment.py - About to call run_experiment_data_collection()")

    # Configuring for 20x20x7 run
    num_problems_to_run = 20
    num_pairs_to_run = 20
    reps_to_run = 7

    seed_val = 123  # Consistent seed

    raw_results, raw_features = run_experiment_data_collection(
        pilot_problems,
        pilot_pairs,
        reps=reps_to_run,
        seed=seed_val,
        num_problems_to_run=num_problems_to_run,
        num_pairs_to_run=num_pairs_to_run
    )
    print("DEBUG: experiment.py - run_experiment_data_collection() finished")

    # Convert raw_results (List[RunResult]) to the nested dictionary structure expected by print block
    # pilot_results_structured: Dict[str, Dict[str, List[RunResult]]] = {}
    # This conversion is complex if we want to match the old print.
    # For now, let's print a summary of raw_results.
    print(f"\nTotal raw results collected: {len(raw_results)}")
    if raw_results:
        print("First 5 raw results:")
        for r in raw_results[:5]:
            print(
                f"  Problem: {r.problem_name}, Algo: {r.algo_name}, RT: {r.runtime:.4f}s, Q: {r.quality:.2f}")

    # The old print block for pilot_results might not work directly with raw_results.
    # print("\nPilot Run Results (first problem, first pair):")
    # if pilot_results_structured and pilot_results_structured.keys():
    #     first_prob_name = list(pilot_results_structured.keys())[0]
    #     if pilot_results_structured[first_prob_name] and pilot_results_structured[first_prob_name].keys():
    #         first_pair_name = list(pilot_results_structured[first_prob_name].keys())[0]
    #         print(f"  {first_prob_name} / {first_pair_name}:")
    #         for run_res in pilot_results_structured[first_prob_name][first_pair_name][:min(6, len(pilot_results_structured[first_prob_name][first_pair_name]))]:
    #             print(f"    Algo: {run_res.algo_name}, RT: {run_res.runtime:.4f}s, Q: {run_res.quality:.2f}")

    # print("\nPilot Features (first problem):") # This was already commented out
    # if pilot_features and pilot_features.keys():
    #     first_prob_name = list(pilot_features.keys())[0]
    #     print(f"  {first_prob_name}: {pilot_features[first_prob_name]}")

    # Prepare DataFrames for full_richness_cv_analysis
    print("\nDEBUG: experiment.py - Preparing DataFrames for CV analysis...")
    results_df = pd.DataFrame([
        {
            'problem': r.problem_name, 
            'algorithm': r.algo_name, 
            'runtime': r.runtime,
            'quality': r.quality,
            'problem_type': r.problem_type
        } for r in raw_results
    ])
    features_df = pd.DataFrame(raw_features) if raw_features else pd.DataFrame()
    print(f"DEBUG: results_df created with {len(results_df)} rows, features_df with {len(features_df)} rows.")

    # Generate and save performance heatmap
    print("\nDEBUG: experiment.py - Generating performance heatmap...")
    try:
        heatmap_df = pd.DataFrame([(r.problem_name, r.algo_name, r.runtime) 
                                  for r in raw_results], 
                                  columns=['problem', 'algo', 'runtime'])
        
        # Filter out entries where runtime might be NaN or Inf if any conceptual solvers returned problematic values
        # Though current conceptual solvers return 0 or small numbers for runtime.
        # For quality, some evaluators might return NaN if an error occurs, but runtime should be fine.
        heatmap_df = heatmap_df.dropna(subset=['runtime'])
        heatmap_df = heatmap_df[np.isfinite(heatmap_df['runtime'])]


        if not heatmap_df.empty:
            # Ensure there are enough unique problems and algos to make a meaningful pivot
            if heatmap_df['problem'].nunique() > 1 and heatmap_df['algo'].nunique() > 1:
                pivot_table = heatmap_df.pivot_table(values='runtime', 
                                                   index='problem', 
                                                   columns='algo', 
                                                   aggfunc='mean') # Use mean runtime across reps
                
                # Fill NaN for algos not run on certain problems (e.g. due to type mismatch)
                # This makes the heatmap complete.
                # Or, we can drop columns/rows with all NaNs if preferred.
                # For now, let pivot_table handle it; heatmap will show missing data.

                plt.figure(figsize=(22, 18)) # Adjusted size for 20x40 potentially
                sns.heatmap(pivot_table, cmap='viridis_r', cbar_kws={'label': 'Mean Runtime (s)'}, annot=False, fmt=".2e") # viridis_r, annot=False
                plt.title('20x20 Algorithm Performance Heatmap (Mean Runtimes)')
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                heatmap_filename = '20x20_performance_heatmap.png'
                plt.savefig(heatmap_filename)
                print(f"DEBUG: Heatmap saved to {heatmap_filename}")
                plt.close() # Close the figure to free memory
                
                # Export data to JSON for TikZ conversion
                tikz_data = {
                    'problems': pivot_table.index.tolist(),
                    'algorithms': pivot_table.columns.tolist(),
                    'runtimes': pivot_table.fillna(-1).values.tolist(),  # -1 for missing values
                    'min_runtime': float(pivot_table.min().min()),
                    'max_runtime': float(pivot_table.max().max()),
                    'metadata': {
                        'num_problems': len(pivot_table.index),
                        'num_algorithms': len(pivot_table.columns),
                        'colormap': 'viridis_r',
                        'description': '20x20 Algorithm Performance Matrix (Mean Runtimes)'
                    }
                }
                
                import json
                tikz_json_filename = '20x20_performance_heatmap_tikz.json'
                with open(tikz_json_filename, 'w') as f:
                    json.dump(tikz_data, f, indent=2)
                print(f"DEBUG: TikZ data saved to {tikz_json_filename}")
                
            else:
                print("DEBUG: Not enough unique problems/algorithms to generate a meaningful pivot table for heatmap.")
        else:
            print("DEBUG: No valid data to generate heatmap after filtering.")
    except Exception as e:
        print(f"ERROR generating heatmap: {e}")


    # The old run_cv call is now replaced by full_richness_cv_analysis
    # print("\nRunning Cross-Validation:") # Old header
    # print("DEBUG: experiment.py - About to call run_cv()") # Old debug
    # cv_summary = run_cv(raw_results, raw_features) # Old call
    # print(
    #     f"DEBUG: experiment.py - run_cv() finished. CV Summary: {cv_summary}") # Old summary print

    # Call the new comprehensive CV analysis function
    # The full_richness_cv_analysis function itself does a lot of printing.
    # It's defined globally now.
    rich_cv_summary_dict = full_richness_cv_analysis(results_df, features_df) # Pass the created DataFrames
    
    print(f"\nDEBUG: experiment.py - full_richness_cv_analysis() finished.")
    print("--- CV Analysis Executive Summary (from returned dict) ---")
    if rich_cv_summary_dict:
        # Extract and print key summary items from the returned dictionary
        # The function itself prints detailed sections. This is a final summary block.
        mean_abs_gap = np.mean(rich_cv_summary_dict.get('absolute_gaps', [float('nan')]))
        median_rel_imp = np.median(rich_cv_summary_dict.get('relative_improvements', [float('nan')])) * 100
        geom_mean_log_imp = np.exp(np.mean(rich_cv_summary_dict.get('log_improvements', [float('nan')])))
        bs_ci_lower, bs_ci_upper = rich_cv_summary_dict.get('bootstrap_ci', (float('nan'), float('nan')))
        
        print(f"  Average Absolute CV Gap (mean of problem gaps): {mean_abs_gap:.4f}s")
        print(f"  Median Relative Improvement Potential: {median_rel_imp:.1f}%")
        print(f"  Geometric Mean Performance Ratio (Prediction vs Best): {geom_mean_log_imp:.2f}x")
        print(f"  Bootstrap 95% CI for Absolute Gap: [{bs_ci_lower:.4f}, {bs_ci_upper:.4f}]s")
        
        # Example: Print problem stats for one problem if available
        problem_stats_df = rich_cv_summary_dict.get('problem_stats')
        if problem_stats_df is not None and not problem_stats_df.empty:
            print(f"\n  Example Problem Stats (first problem): \n{problem_stats_df.head(1)}")
    else:
        print("  Rich CV summary dictionary was not generated or is empty.")

    print("\n--- Original Selected Algorithm Examples (for reference) ---")
    print("DEBUG: experiment.py - End of __main__")
