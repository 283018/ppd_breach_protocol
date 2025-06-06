from breach_solvers.solvers_abc import Solver, OptimizationError, register_solver
from core import Task, Solution, DUMMY_TASK, NoSolution

from numpy import int8, array, zeros, dot
from time import perf_counter
from typing import Tuple, List
from warnings import warn

from gurobipy import GRB, quicksum, Model ,Var, LinExpr, tupledict
# noinspection PyProtectedMember
from gurobipy._exception import GurobiError


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


def ensure_reset(method):
    """Gurobi solver decorator, ensures model reset on solve release."""
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self.model.reset(clearall=True)
    return wrapper


# noinspection DuplicatedCode
@register_solver('gb', 'gurobi')
class GurobiSolver(Solver):
    """Gurobi solver"""
    _allowed_kwargs = {'output_flag': bool, 'strict_opt': bool}

    model: Model

    def _warm_up(self):
        from os import devnull
        import sys

        original_stdout = sys.stdout
        sys.stdout = open(devnull, 'w')
        self.model = Model('BreachProtocol')
        sys.stdout = original_stdout

        self.model.setParam('OutputFlag', 0)

        try:
            start_init = perf_counter()
            self.solve(DUMMY_TASK)
            end_init = perf_counter()
        except Exception as e:
            raise RuntimeError(f"Error while initialization gurobi solver occurred: {e}") from e
        else:
            print(f"\rSuccessfully initialized gurobi solver in {end_init - start_init:.4} sec", flush=True)


    @ensure_reset
    def solve(self, task: Task, **kwargs) -> Tuple[Solution|NoSolution, float]:
        """
        Linear programming solver.

        Exact approach using Gurobi solver via `gurobipy` API for constraint-based modeling.
        Find optimal (if possible) or near-optimal solutions.

        ----
        .. important::
           **IMPORTANT**: Due to Gurobi license limitations, model size is restricted and large-scale tasks may fail.
        ----

        .. note::
           May return ``NoSolution`` or raise ``OptimizationError`` if model building fails or solution status is non-optimal.

        Keyword arguments:
           - **output_flag**: *bool* = ``False``
             If True, enables verbose solver output to console.
           - **strict_opt**: *bool* = ``False``
             If True, enforces strictly optimal solutions. Raises `OptimizationError` if the model fails or the solution status is non-optimal.

        :param task: ``Task`` instance.
        :param kwargs:
        :return: ``tuple``: (found ``Solution`` or ``NoSolution``, ``execution_time``)
            excluding pre- and post- calculations.
        """
        self._validate_kwargs(kwargs)
        output_flag = kwargs.get('output_flag', False)
        strict_opt = kwargs.get('strict_opt', False)

        matrix = task.matrix
        buffer_size = int(task.buffer_size)
        demons = task.demons
        costs = task.demons_costs

        n = matrix.shape[0]
        d_amo = len(demons)
        d_lengths = array([d.size for d in demons])
        # max_points = costs.sum()
        rewards_per_symbol = costs / d_lengths
        unused_cell_reward = 0.1 * rewards_per_symbol.min()

        self.model.setParam('OutputFlag', output_flag)

        time_start = perf_counter()
        x, y, buffer_seq = self._build_model(
            matrix=matrix,
            buffer_size=buffer_size,
            demons=demons,
            costs=costs,
            n=n,
            d_amo=d_amo,
            d_lengths=d_lengths,
            unused_cell_reward=unused_cell_reward,
        )
        time_mid = perf_counter()
        opt_message = self._optimize(strict_opt)
        time_end = perf_counter()

        if opt_message is not None:
            return NoSolution(reason=opt_message), 0.0

        if self.model.status != GRB.OPTIMAL:
            if strict_opt:
                raise OptimizationError("Gurobi solution status is not optimal:\n    {}".format(GRB.OPTIMAL))
            else:
                warn(f"Solution status is not optimal: model.status={self.model.status}")

        x_path = array([(i, j) for t in range(buffer_size) for i in range(n) for j in range(n) if x[i, j, t].X > 0.5], dtype=int8)
        buffer_nums = array([round(buffer_seq[t].getValue()) for t in range(buffer_size)])
        y_active = zeros(d_amo, dtype=bool)
        for i, var in y.items():
            y_active[i] = bool(round(var.X))
        y_total_points = dot(costs, y_active)

        if output_flag:
            print(f"\nBuild time: {time_mid - time_start:.6} sec \nOptimization time: {time_end - time_mid:.6} sec")

        if x_path.size == 0:
            return NoSolution(reason="No valid solution possible for given task"), 0.0

        return Solution(x_path, buffer_nums, y_active, y_total_points), time_end - time_start


    def _build_model(self, matrix, buffer_size, demons, costs, n, d_amo, d_lengths, unused_cell_reward) \
            -> Tuple[tupledict[Tuple[int, ...], Var], tupledict[int, Var], List[LinExpr]]:
        """
        Model build
        """
        x = self.model.addVars(n, n, buffer_size, vtype=GRB.BINARY, name='x')

        for t in range(buffer_size):
            self.model.addConstr(x.sum('*', '*', t) <= 1, name=f'one_cell_per_step_{t}')

        for t in range(1, buffer_size):
            prev = x.sum('*', '*', t - 1)
            curr = x.sum('*', '*', t)
            self.model.addConstr(curr <= prev, name=f'continuous_path_at_{t}')

        for i in range(n):
            for j in range(n):
                self.model.addConstr(x.sum(i, j, '*') <= 1, name=f'cell_{i}_{j}_once')

        used_buffer = x.sum('*', '*', '*')
        self.model.addConstr((used_buffer <= buffer_size), name='max_steps')

        self.model.addConstrs((x.sum(i, '*', 0) == 0 for i in range(1, n)), name='start_in_first_row')

        for t in range(1, buffer_size):
            for i in range(n):
                for j in range(n):
                    if t % 2 == 1:
                        self.model.addConstr(x[i, j, t] <= x.sum('*', j, t - 1),
                                             name=f'move_rule_col_{i}_{j}_step_{t}')
                    else:
                        self.model.addConstr(x[i, j, t] <= x.sum(i, '*', t - 1),
                                             name=f'move_rule_row_{i}_{j}_step_{t}')

        buffer_seq = [
            quicksum(
                matrix[i, j] * x[i, j, t]
                for i in range(n)
                for j in range(n)
            )
            for t in range(buffer_size)
        ]

        z = []
        for i in range(d_amo):
            curr_len = d_lengths[i]
            valid_p = buffer_size - curr_len + 1
            z_i = {}

            for p in range(valid_p):
                z_i[p] = self.model.addVar(
                    vtype=GRB.BINARY, name=f'z_{i}_{p}')
            z.append(z_i)

            for p in range(valid_p):
                for s in range(curr_len):
                    t = p + s
                    self.model.addGenConstrIndicator(
                        z_i[p],
                        1,
                        buffer_seq[t] == demons[i][s],
                        name=f'indicator_demon_{i}_pos_{p}_symbol_{s}')


        y = self.model.addVars(d_amo, vtype=GRB.BINARY, name='y')

        for i in range(d_amo):
            valid_p = buffer_size - d_lengths[i] + 1

            self.model.addConstr(y[i] <= quicksum(z[i][p] for p in range(valid_p)), name=f'active_y_{i}_at_least_one')
            for p in range(valid_p):
                self.model.addConstr(y[i] >= z[i][p], name=f'active_y_{i}_for_{p}')

        self.model.setObjective(
            quicksum(costs[i] * y[i] for i in range(d_amo))
            + unused_cell_reward * (buffer_size - used_buffer),
            GRB.MAXIMIZE,
        )

        return x, y, buffer_seq

    def _optimize(self, strict_opt) -> str|None:
        """
        Run optimization.
        :param strict_opt: if True raise error on optimization failure, otherwise return error log on failure.
        :return: optimization failure message or None if optimization succeeded.
        """
        try:
            self.model.optimize()
        except GurobiError as e:
            if strict_opt:
                raise OptimizationError("Gurobi optimization failed:\n    {}".format(e)) from e
            else:
                return str(e)
        return None


