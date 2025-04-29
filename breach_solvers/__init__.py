# breach_solvers/__init__.py

from .solvers_protocol import get_solver
from .solver_gurobi import SolverGurobi
from .solver_brute import BruteSolver


__all__ = ['get_solver']