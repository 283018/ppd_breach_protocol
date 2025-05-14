# breach_solvers/__init__.py

from .solvers_protocol import get_solver
from .gurobi.solver_gurobi import GurobiSolver
from .brute.solver_brute import BruteSolver
from .ant_colony.solver_ac import AntColSolver


__all__ = ['get_solver']