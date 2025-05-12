from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from numpy import int8, array, zeros, dot
from time import perf_counter

import gurobipy as gp

# ================================================================
# This script uses Gurobi Optimizer via the 'gurobipy' Python API.
# It is intended for educational or research purposes only.
#
# A restricted-use license is bundled with 'gurobipy', which allows
# solving small models for non-commercial and non-production use.
#
# If you intend to use Gurobi beyond these limits, please obtain a
# proper academic or commercial license from:
# https://www.gurobi.com/downloads/
#
# ================================================================


# TODO: separate method + init log
@register_solver('gurobi')
class GurobiSolver(Solver):
    _allowed_kwargs = {}

    # pycharm cant reed types from dataclasses and struggling with gurobi :/
    # noinspection PyTypeChecker
    def __call__(self, task: Task, **kwargs):
        """Solve breach protol task using linear programming solver Gurobi via gurobipy API."""
        self._validate_kwargs(kwargs)

        matrix = task.matrix
        buffer_size = int(task.buffer_size) # to avoid problems with numpy.int
        demons = task.demons
        costs = task.demons_costs

        n = matrix.shape[0]
        d_amo = len(demons)
        d_lengths = array([d.size for d in demons])
        max_points = costs.sum()
        unused_cell_reward = 0.1 * (max_points / d_amo)


        # setup
        model = gp.Model("BreachProtocol")
        model.setParam('OutputFlag', 0)

        # Do mierzenia czasu pomijamy przekształcanie danych i inicjalizacje skupiamy się przede wszystkim na budowaniu modelu
        # (w naszym przypadku model w pełni zależy od inputu i nie można go używać kilkukrotnie) i jego optymalizacji
        time_start = perf_counter()

        # wybór drogi
        x = model.addVars(n, n, buffer_size, vtype=gp.GRB.BINARY, name='x')

        # co najwyżej jedna komórka wybierana w każdym kroku
        for t in range(buffer_size):
            model.addConstr(x.sum('*', '*', t) <= 1, name=f'one_cell_per_step_{t}')

        # ciągłość drogi - nie można wybrać komórki po niewybranej komórki (ale można nie wybierać na końcu)
        for t in range(1, buffer_size):
            prev = x.sum('*', '*', t - 1)
            curr = x.sum('*', '*', t)
            model.addConstr(curr <= prev, name=f'continuous_path_at_{t}')

        # komórka może być wybrana co najwyżej raz
        for i in range(n):
            for j in range(n):
                model.addConstr(x.sum(i, j, '*') <= 1, name=f'cell_{i}_{j}_once')

        # maksymalna ilość kroków
        used_buffer = x.sum('*', '*', '*')
        model.addConstr((used_buffer <= buffer_size), name='max_steps')

        # rozpoczęcie zawsze w pierwszym rzędzie - dla wszystkich rzędów poza pierwszym w kroku 0 nie można wybierać komórki
        model.addConstrs((x.sum(i, '*', 0) == 0 for i in range(1, n)), name='start_in_first_row')

        # poruszanie się po macierzy zgodnie z zasadami rząd/kolumna
        # TODO: indicators?
        for t in range(1, buffer_size):
            for i in range(n):
                for j in range(n):
                    if t % 2 == 1:
                        # dla dowolnej komórki w kroku t musi być wybrana dowolna komórka w tej samej *kolumnie* w kroku t-1
                        model.addConstr(x[i, j, t] <= x.sum('*', j, t - 1), name=f'move_rule_col_{i}_{j}_step_{t}')
                    else:
                        model.addConstr(x[i, j, t] <= x.sum(i, '*', t - 1), name=f'move_rule_row_{i}_{j}_step_{t}')

        # # bezpośrednio sekwencja buffera
        buffer_seq = [
            gp.quicksum(
                matrix[i, j] * x[i, j, t]
                for i in range(n)
                for j in range(n)
                )
            for t in range(buffer_size)
        ]

        # lista, dla każdego demona jest słownik pozycji w sekwencji buffera gdzie ten demon może się zaczynać : is_valid
        # [{pos: is_valid}, ...] ∀ demon (according to demon length)
        # TODO: really dont like this dicts list
        z = []
        for i in range(d_amo):
            curr_len = d_lengths[i]
            valid_p = buffer_size - curr_len + 1
            z_i = {}

            for p in range(valid_p):                            # ∀ valid starting pos
                z_i[p] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f'z_{i}_{p}')
            z.append(z_i)

            for p in range(valid_p):
                for s in range(curr_len):                       # ∀ symbolu demona
                    t = p + s
                    model.addGenConstrIndicator(
                        z_i[p],                         # jeśli dla tego demona dla tej pozycji startowej is_valid
                        1,                              # == 1 (True)
                        buffer_seq[t] == demons[i][s],  # to w buffer (na odpowiedniej pozycji) MUSI być symbol demona
                        name=f'indicator_demon_{i}_pos_{p}_symbol_{s}')
        

        # y variables for demon activation
        y = model.addVars(d_amo, vtype=gp.GRB.BINARY, name='y')

        # jeśli chociaż jedna z starting_pos valid to y[i] == 1
        for i in range(d_amo):
            valid_p = buffer_size - d_lengths[i] + 1

            # sekwencja demona może występować w buffer kilkakrotnie np. demon=[1, 3], buffer = [1, 3, 1, 3]
            # y[i] == 0 jeśli *żadna* z z[i][p] nie == 1
            model.addConstr(y[i] <= gp.quicksum(z[i][p] for p in range(valid_p)), name=f'active_y_{i}_at_least_one')
            # jeśli dowolna z z[i][p] == 1, y[i] też == 1
            for p in range(valid_p):
                model.addConstr(y[i] >= z[i][p], name=f'active_y_{i}_for_{p}')


        # maksymalizacja punktów (obv) + wynagrodzenie za niezużycie buffera
        model.setObjective(
            gp.quicksum(
                costs[i] * y[i] for i in range(d_amo)) +
                unused_cell_reward * (buffer_size - used_buffer),
            gp.GRB.MAXIMIZE)

        # solve
        model.optimize()

        if model.status != gp.GRB.OPTIMAL:
            raise ValueError(f"Optimization failed, model.status={model.status}")

        time_end = perf_counter()


        # result extraction
        x_path = array([(i, j) for t in range(buffer_size) for i in range(n) for j in range(n) if x[i, j, t].x > 0.5], dtype=int8)

        buffer_nums = array([int(buffer_seq[t].getValue()) for t in range(buffer_size)])

        y_active = zeros(d_amo, dtype=bool)
        for i, var in y.items():
            y_active[i] = bool(int(var.x))

        y_total_points = dot(costs, y_active)

        # from icecream import ic
        # ic('')
        # ic(task)
        # ic(x_path, buffer_nums, y_active, y_total_points)

        return Solution(x_path, buffer_nums, y_active, y_total_points), time_end - time_start



# if __name__ == '__main__':
#     from icecream import ic
#     # from core import verify_solution, solution_print
#     buffer_size1 = 8
#     demons1 = (
#         array([1, 2]),
#         array([3, 4]),
#         array([5, 1, 2])
#     )
#     d_costs1 = array([1, 2, 3])
#     matrix1 = array([
#         [3, 1, 5, 5, 3, 5, 2],
#         [5, 6, 2, 2, 5, 5, 1],
#         [5, 5, 4, 2, 6, 5, 2],
#         [1, 2, 1, 3, 2, 2, 1],
#         [1, 5, 2, 4, 6, 6, 4],
#         [3, 1, 3, 5, 5, 2, 3],
#         [3, 5, 1, 5, 6, 3, 2],
#     ])
#     gb_solver = SolverGurobi()
#     task1 = Task(matrix1, demons1, d_costs1, buffer_size1)
#     sol1 = gb_solver.solve(task1)
#     ic(sol1)
# #     # solution_print(task1, sol1)
# #     # mat_print(matrix1, sol1[0])
# #     if verify_solution(task1, sol1):
# #         solution_print(task1, sol1)
# #
# #     else:
# #         print("AAAAAAAAAAAAAAAAAA")