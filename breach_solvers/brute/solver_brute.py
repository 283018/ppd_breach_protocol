from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from numpy import ndarray, integer, array, zeros, empty, full, dtype
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Tuple
from time import perf_counter

from numba import njit

from icecream import ic



#! TODO: sometimes stuck on over 8x8 matrices
# TODO: @njit + prange
@register_solver('brute')
class BruteSolver(Solver):
    _allowed_kwargs = {'to_prune'}

    def __call__(self, task:Task, **kwargs):
        self._validate_kwargs(kwargs)
        to_prune = kwargs.get("to_prune", True)

        matrix = task.matrix
        demons = task.demons
        d_costs = task.demons_costs
        buffer_s = task.buffer_size

        n = matrix.shape[0]
        max_score = d_costs.sum()
        d_lengths = array([d.size for d in demons])
        n_demons = d_lengths.size
        max_demon_len = d_lengths.max()

        # TODO: use it ig
        padded_demons = full((n_demons, max_demon_len), -1)
        for i, d in enumerate(demons):
            padded_demons[i, :d_lengths[i]] = d


        init_stack_size = 100    # TODO: not-constant

        process_num = min(n, cpu_count())

        time_start = perf_counter()

        # aktualizuje multiprocessing dla wszystkich startowych pozycji (komórek w pierwszym rzędzie)
        # TODO: sequential if ...
        with Pool(processes=process_num) as pool:
            worker = partial(self._process_start_column,
                             matrix=matrix,
                             demons=demons,
                             demons_lengths=d_lengths,
                             d_costs=d_costs,
                             buffer_size=buffer_s,
                             n=n,
                             max_score=max_score,
                             num_demons=n_demons,
                             stack_size=init_stack_size,
                             to_prune=to_prune,
                             )

            results = pool.map(worker, range(n))
        time_end = perf_counter()

        # Znajduje najlepszy wynik ze wszystkich startowych kolumn
        ic(results)
        best_score = 0
        best_path = None
        for path, score in results:
            if score > best_score or (score == best_score and path.shape[0] < best_path.shape[0]):
                best_score = score
                best_path = path

        return Solution(best_path).fill_solution(task), time_end - time_start


    @staticmethod
    # @njit
    def _process_start_column(start_col,  # starting column (setted by multiprocessing)
                              matrix: ndarray, demons: Tuple[ndarray], demons_lengths: ndarray, d_costs: ndarray, buffer_size: int | integer,  # specs of task itself
                              n: int, max_score: int, num_demons: int,  # pre-calculated parameters
                              stack_size,
                              to_prune: bool,
                              ) -> Tuple[ndarray, integer]:
        # TODO: max stack instead of 100

        max_stack = stack_size

        stack_path = empty((max_stack, buffer_size, 2), dtype='int8')
        stack_buff = empty((max_stack, buffer_size), dtype='int8')
        stack_activ = empty((max_stack, num_demons), dtype='bool_')
        stack_score = empty(max_stack, dtype='int16')
        stack_used = empty((max_stack, n, n), dtype='bool_')
        stack_length = empty(max_stack, dtype='int8')

        best_score = 0
        best_path = full((buffer_size, 2), -1, dtype='int8')
        best_path_length = 0

        start_r, start_c = 0, start_col
        start_symbol = matrix[start_r, start_c]

        # boolean array instead of bitmask
        used0 = zeros((n, n), dtype='bool_')
        used0[start_r, start_c] = True

        activ0 = zeros(num_demons, dtype='bool_')
        current_score0 = 0

        # sprawdzenie na demonów długości 1
        for i in range(num_demons):
            if demons_lengths[i] == 1 and demons[i][0] == start_symbol:
                activ0[i] = True
                current_score0 += d_costs[i]

        path0 = full((buffer_size, 2), -1, dtype='int8')
        path0[0, 0] = start_r
        path0[0, 1] = start_c

        buff0 = full(buffer_size, -1, dtype='int8')
        buff0[0] = start_symbol
        length0 = 1

        # Basically manual stack implementation by bundled parallel pre-allocated arrays
        # Setting first elements on stack
        top = 0
        stack_path[top] = path0
        stack_buff[top] = buff0
        stack_activ[top] = activ0
        stack_score[top] = current_score0
        stack_used[top] = used0
        stack_length[top] = length0
        top += 1

        while top > 0:
            top -= 1

            # pop from stack
            path = stack_path[top]
            buff = stack_buff[top]
            activ = stack_activ[top]
            score = stack_score[top]
            used = stack_used[top]
            curr_len = stack_length[top]

            if score > best_score:
                if curr_len <= buffer_size:
                    best_score = score
                    best_path_length = curr_len
                    for i in range(curr_len):
                        best_path[i, 0] = path[i, 0]
                        best_path[i, 1] = path[i, 1]
                    if best_score == max_score:
                        return best_path[:best_path_length], best_score

            # push children if can grow
            if curr_len < buffer_size:
                next_step = curr_len + 1
                last_r = path[curr_len - 1, 0]
                last_c = path[curr_len - 1, 1]

                if (next_step & 1) == 1:  # synonymic to next_step % 2 == 1
                    for c in range(n):
                        if not used[last_r, c]:
                            new_len = next_step

                            # copy-on-write arrays
                            new_path = path.copy()
                            new_path[curr_len, 0] = last_r
                            new_path[curr_len, 1] = c
                            new_buff = buff.copy()
                            new_buff[curr_len] = matrix[last_r, c]
                            new_used = used.copy()
                            new_used[last_r, c] = True
                            new_activ = activ.copy()
                            new_score = score

                            # check demons
                            for i in range(num_demons):
                                if not new_activ[i]:
                                    k = demons_lengths[i]
                                    if new_len >= k:
                                        match = True
                                        # check last k symbols
                                        for j in range(k):
                                            if new_buff[new_len - k + j] != demons[i][j]:
                                                match = False
                                                break
                                        if match:
                                            new_activ[i] = True
                                            new_score += d_costs[i]

                            # pruning
                            if to_prune:
                                rem = 0
                                for i in range(num_demons):
                                    if not new_activ[i]:
                                        rem += d_costs[i]
                                do_push = (new_score + rem > best_score)
                            else:
                                do_push = True

                            if do_push:
                                idx = top
                                stack_path[idx] = new_path
                                stack_buff[idx] = new_buff
                                stack_activ[idx] = new_activ
                                stack_score[idx] = new_score
                                stack_used[idx] = new_used
                                stack_length[idx] = new_len
                                top += 1

                else:
                    # move along column
                    for r in range(n):
                        if not used[r, last_c]:
                            new_len = next_step

                            new_path = path.copy()
                            new_path[curr_len, 0] = r
                            new_path[curr_len, 1] = last_c
                            new_buff = buff.copy()
                            new_buff[curr_len] = matrix[r, last_c]
                            new_used = used.copy()
                            new_used[r, last_c] = True
                            new_activ = activ.copy()
                            new_score = score

                            for i in range(num_demons):
                                if not new_activ[i]:
                                    k = demons_lengths[i]
                                    if new_len >= k:
                                        match = True
                                        for j in range(k):
                                            if new_buff[new_len - k + j] != demons[i][j]:
                                                match = False
                                                break
                                        if match:
                                            new_activ[i] = True
                                            new_score += d_costs[i]

                            if to_prune:
                                rem = 0
                                for i in range(num_demons):
                                    if not new_activ[i]:
                                        rem += d_costs[i]
                                do_push = (new_score + rem > best_score)
                            else:
                                do_push = True

                            if do_push:
                                idx = top
                                stack_path[idx] = new_path
                                stack_buff[idx] = new_buff
                                stack_activ[idx] = new_activ
                                stack_score[idx] = new_score
                                stack_used[idx] = new_used
                                stack_length[idx] = new_len
                                top += 1

        return best_path[:best_path_length], best_score









