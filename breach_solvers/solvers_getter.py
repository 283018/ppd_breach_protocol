from typing import overload, Literal
from collections import defaultdict

from .solvers import BruteSolver, GurobiSolver, AntColSolver, ScipSolver
from .solvers_abc import Solver, solver_registry


@overload
def get_solver(name: Literal['gurobi', 'gb']) -> GurobiSolver: ...

@overload
def get_solver(name: Literal['brute', 'bf']) -> BruteSolver: ...

@overload
def get_solver(name: Literal['ant_col', 'ac']) -> AntColSolver: ...

@overload
def get_solver(name: Literal['scip', 'sc']) -> ScipSolver: ...


def get_solver(name: str) -> Solver:
    """
    Retrieve a Solver instance by name, acting as a factory for creating Solver instances.

    Main solver interface provides ``.solve(task:Task,**kwargs)`` method.
    Each solver also supports ``.solve_iter(task_list:Iterable[Task],**kwargs)`` for batch processing of multiple tasks.
    Shortcut methods ``.s()`` and ``.__call__()`` are available (all require same args/kwargs as ``.solve()``).

    Supported Solvers & type code:
        - **Gurobi LP** ``{'gurobi','gb'}``: Constraint-based optimization using Gurobi via `gurobipy`.
        - **SCIP LP** ``{'scip','sc'}``: Constraint-based optimization using SCIP via `PySCIPOpt`.
        - **Brute-force** ``{'brute','bf'}``: DFS search with optional B&B pruning for full exploration or heuristic search.
        - **Ant-colony** ``{'ant_col','ac'}``: Ant Colony Optimization (ACO) metaheuristic using C++ backend.

    ---

    Keyword arguments by Solver:

    **Gurobi LP**
       - ``output_flag``: *bool* = ``False``
         Enable verbose solver output.
       - ``strict_opt``: *bool* = ``False``
         Enforce strictly optimal solutions. Raises ``OptimizationError`` if model fails or solution status is non-optimal.
    **SCIP LP**
       - ``output_flag``: *bool* = ``False``
         Enable verbose solver output.
       - ``strict_opt``: *bool* = ``False``
         Enforce strictly optimal solutions. Raises ``OptimizationError`` if model fails or solution status is non-optimal.
    **Brute-force**
       - ``to_prune``: *bool* = ``True``
         Allow branch-and-bound pruning. *Heuristic* may yield non-optimal solutions (optimal uses least buffer among max-scored).
       - ``avoid_c``: *bool* = ``False``
         Skip C++ backend (NOT RECOMMENDED: Python-Numba has recompilation overhead for varying inputs).
       - ``timeout``: *float* = ``0.0``
             If ``>0.`` set time limit on execution, if ``<=0.`` process task as normal (look ``enable_pruning``).
    **Ant-colony**
       - ``avoid_c``: *bool* = ``False``
         Skip C++ backend (NOT RECOMMENDED: Python implementation is outdated/slower).
       - ``n_ants``: *int* = ``task.matrix.size``
         Number of ants per iteration.
       - ``max_iter``: *int* = ``0``
         Max iterations; set to `INT_MAX` (2³¹-1) if ≤ 0.
       - ``stag_lim``: *int* = ``n*buffer_size*num_demons``
         Allowed iterations without improvement (modes: ``==0``: auto, ``<0``: no control, ``>0``: hard limit).
       - ``alpha``: *float* = ``0.1``
         Pheromone trail attractiveness.
       - ``beta``: *float* = ``0.4``
         Heuristic matrix importance (reward-based cell attractiveness).
       - ``evap``: *float* = ``0.475``
         Pheromone decay rate.
       - ``q``: *float* = ``250.0``
         Pheromone deposited per ant.

    .. note::
        - **Gurobi**: License restrictions may block large-scale tasks.
        - **SCIP**: Slower than Gurobi but no model size restrictions.
        - **Ant-colony**: Use `.seed()` to set RNG state for reproducibility.
        - **Brute-force**, **Ant-colony**: ``avoid_c=True`` should not be used since those methods are deprecated and inefficient.

    ----

    .. warning::
       - ***WARNING***
        - **Brute force**: Large-scale tasks may cause unreasonable runtime even with pruning, consider setting ``timeout``.
        - **Ant-colony**: Using ``stagnant_limit < 0 < n_iterations`` may cause infinite loops (2³¹-1 iterations).

    :param name: Solver type code: (`'gurobi'`, `'brute'`, `'ant_col'`, `'scip'`), including corresponding shortcuts.
    :return: Instance of the specified Solver subclass.
    """
    solver_class = solver_registry.get(name, None)
    if not solver_class:
        raise ValueError(f"Unknown solver: {name}, must be one of {list(solver_registry.keys())}")
    return solver_class()