from breach_solvers.ant_colony.solver_ac import AntColSolver
from breach_solvers.brute.solver_brute import BruteSolver
from breach_solvers.gurobi.solver_gurobi import GurobiSolver

from core import *
from task_generator import *
from icecream import ic
from pprint import pprint as pp

if __name__ == '__main__':
    solver_ac = AntColSolver()
    solver_ac.seed(124534562211)
    solver_bf = BruteSolver()
    solver_gb = GurobiSolver()
    factory1 = TaskFactory(474686)

    task1 = factory1.gen_manual(8, {4:4, 5:3, 6:3, 7:1}, 12)
    # ic(task1)

    sol1, time1 = solver_ac(task1)
    # sol2, time2 = solver_bf(task1)
    sol3, time3 = solver_gb(task1)

    solution_print(task1, sol1);    print("\n")
    # solution_print(task1, sol2);    print("\n")
    solution_print(task1, sol3);    print("\n")

    print(time1)
    # print(time2)
    print(time3)