from typing import overload, Literal

from .solvers import BruteSolver, GurobiSolver, AntColSolver, ScipSolver
from .solvers_abc import Solver, solver_registry


@overload
def get_solver(name: Literal['gurobi']) -> GurobiSolver:
    ...

@overload
def get_solver(name: Literal['brute']) -> BruteSolver:
    ...

@overload
def get_solver(name: Literal['ant_col']) -> AntColSolver:
    ...

@overload
def get_solver(name: Literal['scip']) -> ScipSolver:
    ...


# TODO: kwargs for seed
def get_solver(name: str) -> Solver:
    """
    Retrieve a Solver instance by name, acts as a factory for creating Solver instances.

    Main method for Solver family is .solve(task: Task, **kwargs), each type may support different keyword arguments (some may support none).
    Shortcut '.s()' and '.__call__()' methods for '.solve()' are available (all require same args and kwargs as .solve()).

    Solvers:
        - Brute force solver - provide both strait-forward approach with full exploration and heuristic search.
        - Linear programming solver - constraint-based solution search.
        - Ant-colony - ant-colony optimization metaheuristic approach.

    Keyword arguments:
        -----=====#####=====-----

    gurobi:
        - output_flag:bool=False
            if True allow solver to output full optimization information in console.
        - strict_opt:bool=False
            if True enforce strictly optimal solution return, will raise OptimizationError if model build failed or solution status is not optimal.
    scip:
        - output_flag:bool=False
            if True allow solver to output full optimization information in console.
        - strict_opt:bool=False
            if True enforce strictly optimal solution return, will raise OptimizationError if model build failed or solution status is not optimal.
    brute:
        - to_prune:bool=True
            if True allow B&B pruning, and best-score loop cut, essentially heuristic, that allows for non-optimal solutions (optimal solution is one, that uses the least buffer across all maximum-scored solutions).
        - avoid_c:bool=False
            if True skip call to c++ back and jump to Python/numba implementation.
    ant_col:
        - Not added yet

    :param name: code for Solver type, currently supported: ('gurobi', 'brute', 'ant_col')
    :return: instance of specific (specified by name) type Solver
    """
    solver_class = solver_registry.get(name, None)
    if not solver_class:
        raise ValueError(f"Unknown solver: {name}, must be one of {list(solver_registry.keys())}")
    return solver_class()