from typing import overload, Literal

from .solvers import BruteSolver, GurobiSolver, AntColSolver
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


# TODO: docstring kwargs update
def get_solver(name: str) -> Solver:
    """
    Get solver instance by name:

    :param name: {'gurobi', 'brute', 'ant_col'}
    :return: solver instance
    """
    solver_class = solver_registry.get(name)
    if not solver_class:
        raise ValueError(f"Unknown solver: {name}, must be one of {list(solver_registry.keys())}")
    return solver_class()