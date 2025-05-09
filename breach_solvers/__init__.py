# breach_solvers/__init__.py

from .solvers_protocol import get_solver
from .gurobi.solver_gurobi import SolverGurobi
from .brute.solver_brute import BruteSolver


__all__ = ['get_solver']