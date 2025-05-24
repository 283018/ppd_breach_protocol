from breach_solvers.solvers_abc import Solver, OptimizationError, register_solver
from core import Task, Solution, DUMMY_TASK, NoSolution

from numpy import int8, array, zeros, dot
from time import perf_counter
from typing import Tuple, List, Dict
from warnings import warn

from pyscipopt import Model, quicksum, Expr



# ================================================================
# This script uses SCIP Optimizer via the 'PySCIPOpt' Python API.
# It is intended for educational or research purposes only.
#
# PySCIPOpt provided under such licence:
#
#
# MIT License
#
# Copyright (c) 2023 Zuse Institute Berlin (ZIB)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ================================================================



def ensure_reset(method):
    """Scip solver decorator, ensures model reset on solve release."""
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self.model.resetParams()
            self.model = Model('BreachProtocol')
    return wrapper


# noinspection DuplicatedCode
@register_solver('scip')
class ScipSolver(Solver):
    _allowed_kwargs = {'output_flag': bool, 'strict_opt': bool}

    model: Model

    def _warm_up(self):
        self.model = Model('BreachProtocol')

        self.model.hideOutput(True)

        try:
            start_init = perf_counter()
            self.solve(DUMMY_TASK)
            end_init = perf_counter()
        except Exception as e:
            raise RuntimeError(f"Error while initialization scip solver occurred: {e}") from e
        else:
            print(f"\rSuccessfully initialized scip solver in {end_init - start_init:.4} sec", flush=True)

    @ensure_reset
    def solve(self, task: Task, **kwargs) -> Tuple[Solution | NoSolution, float]:
        """
        Linear programming solver.

        Uses SCIP via PySCIPOpt API.
        Build constraints-based model to find optimal or near-optimal (if previous is not possible) solutions.

        Possible keyword arguments:
            - output_flag:bool=False - if True allow solver to output full optimization information in console.
            - strict_opt:bool=False - if True enforce strictly optimal solution return, will raise OptimizationError if model build failed or solution status is not optimal.

        :param task:
        :param kwargs:
        :return: found Solution or NoSolution, main execution time (without pre-calculation but including model build time)
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
        max_points = costs.sum()
        unused_cell_reward = 0.1 * (max_points / d_amo)

        self.model.hideOutput(not output_flag)

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

        status = self.model.getStatus()
        if status != 'optimal':
            if strict_opt:
                raise OptimizationError("Gurobi solution status is not optimal:\n    {}".format('optimal'))
            else:
                warn(f"Solution status is not optimal: model.status={status}")

        x_path = array(
            [
                (i, j)
                for t in range(buffer_size)
                for i in range(n)
                for j in range(n)
                if self.model.getVal(x[i, j, t]) > 0.5
            ],
            dtype=int8,
        )
        
        from icecream import ic
        ic(buffer_size)
        ic([self.model.getVal(buffer_seq[t]) for t in range(buffer_size)])
        
        buffer_nums = array([round(self.model.getVal(buffer_seq[t])) for t in range(buffer_size)])

        y_active = zeros(d_amo, dtype=bool)
        for i, var in enumerate(y):
            y_active[i] = bool(int(self.model.getVal(y[i])))
        y_total_points = dot(costs, y_active)

        if output_flag:
            print(f"\nBuild time: {time_mid - time_start:.6} sec \nOptimization time: {time_end - time_mid:.6} sec")

        return Solution(x_path, buffer_nums, y_active, y_total_points), time_end - time_start


    def _build_model(
        self,
        matrix,
        buffer_size,
        demons,
        costs,
        n,
        d_amo,
        d_lengths,
        unused_cell_reward,
    ) -> Tuple[Dict[Tuple[int, int, int], str], List[str], List[Expr]]:
        """
        Model build
        """

        x = {}
        for i in range(n):
            for j in range(n):
                for t in range(buffer_size):
                    x[i, j, t] = self.model.addVar(vtype="B", name=f"x_{i}_{j}_{t}")

        # CONSTRAINS
        # one cell per step
        for t in range(buffer_size):
            self.model.addCons(
                quicksum(
                    x[i, j, t] for i in range(n) for j in range(n)
                )
                <= 1,
                name=f"one_cell_per_step_{t}",
            )

        # continuous path
        for t in range(1, buffer_size):
            prev_sum = quicksum(
                x[i, j, t - 1] for i in range(n) for j in range(n)
            )
            curr_sum = quicksum(
                x[i, j, t] for i in range(n) for j in range(n)
            )
            self.model.addCons(curr_sum <= prev_sum, name=f"continuous_path_{t}")

        # cell used at max one time
        for i in range(n):
            for j in range(n):
                self.model.addCons(
                    quicksum(x[i, j, t] for t in range(buffer_size)) <= 1,
                    name=f"cell_once_{i}_{j}",
                )

        # Cax steps (probably redundant)
        used_buffer = quicksum(
            x[i, j, t]
            for i in range(n)
            for j in range(n)
            for t in range(buffer_size)
        )
        self.model.addCons(used_buffer <= buffer_size, name="max_steps")

        # start in first row
        for i in range(1, n):
            self.model.addCons(
                quicksum(x[i, j, 0] for j in range(n)) == 0,
                name=f"start_in_first_row_{i}",
            )

        # movement rules
        for t in range(1, buffer_size):
            for i in range(n):
                for j in range(n):
                    if t % 2 == 1:  # column
                        prev_col = quicksum(x[k, j, t - 1] for k in range(n))
                        self.model.addCons(
                            x[i, j, t] <= prev_col, name=f"move_rule_col_{i}_{j}_step_{t}"
                        )
                    else:  # row
                        prev_row = quicksum(x[i, k, t - 1] for k in range(n))
                        self.model.addCons(
                            x[i, j, t] <= prev_row, name=f"move_rule_row_{i}_{j}_step_{t}"
                        )

        # buffer sequence
        buffer_seq = [
            quicksum(
                matrix[i][j] * x[i, j, t]
                for i in range(n)
                for j in range(n)
            )
            for t in range(buffer_size)
        ]

        # demon activation
        z = []
        for i in range(d_amo):
            curr_len = d_lengths[i]
            valid_p = buffer_size - curr_len + 1
            z_i = {}
            for p in range(valid_p):
                z_i[p] = self.model.addVar(vtype="B", name=f"z_{i}_{p}")
            z.append(z_i)

        # constraints for demons
        # had to switch to big-M constrains, since indicator works differently
        big_m = 101
        for i in range(d_amo):
            curr_len = d_lengths[i]
            valid_p = buffer_size - curr_len + 1
            for p in range(valid_p):
                for s in range(curr_len):
                    t = p + s
                    # z[i][p] == 1 --> buffer_seq[t] == demons[i][s]
                    self.model.addCons(
                        buffer_seq[t] >= demons[i][s] - big_m * (1 - z[i][p]),
                        name=f"indicator_lb_{i}_{p}_{s}",
                    )
                    self.model.addCons(
                        buffer_seq[t] <= demons[i][s] + big_m * (1 - z[i][p]),
                        name=f"indicator_ub_{i}_{p}_{s}",
                    )

        # demon activation
        y = [self.model.addVar(vtype="B", name=f"y_{i}") for i in range(d_amo)]
        for i in range(d_amo):
            valid_p = buffer_size - d_lengths[i] + 1
            if valid_p > 0:
                self.model.addCons(
                    y[i] <= quicksum(z[i][p] for p in range(valid_p)),
                    name=f"y_upper_{i}",
                )
                for p in range(valid_p):
                    self.model.addCons(y[i] >= z[i][p], name=f"y_lower_{i}_{p}")
            else:
                self.model.addCons(y[i] == 0, name=f"y_false_{i}")

        # objective
        objective = quicksum(costs[i] * y[i] for i in range(d_amo)) + unused_cell_reward * (buffer_size - used_buffer)
        self.model.setObjective(objective, sense='maximize')

        return x, y, buffer_seq


    def _optimize(self, strict_opt) -> str|None:
        """
        Run optimization.
        :param strict_opt: if True raise error on optimization failure, otherwise return error log on failure.
        :return: optimization failure message or None if optimization succeeded.
        """
        try:
            self.model.optimize()
        except Exception as e:
            if strict_opt:
                raise OptimizationError(
                    "SCIP optimization failed:\n    {}".format(e)
                ) from e
            else:
                return str(e)
        return None









