from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from numpy import ndarray, integer, int8, int16, bool_, array, zeros, empty, full, dtype
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
                             demons_array=padded_demons,
                             demons_lengths=d_lengths,
                             demons_costs=d_costs,
                             buffer_size=buffer_s,
                             n=n,
                             max_score=max_score,
                             num_demons=n_demons,
                             init_stack_size=init_stack_size,
                             enable_pruning=to_prune,
                             )

            results = pool.map(worker, range(n))
        time_end = perf_counter()

        # Znajduje najlepszy wynik ze wszystkich startowych kolumn
        ic(results)
        best_score = 0
        best_path = empty((0, 2), dtype=bool_)
        for path, score in results:
            if score > best_score or (score == best_score and path.shape[0] < best_path.shape[0]):
                best_score = score
                best_path = path

        return Solution(best_path).fill_solution(task), time_end - time_start


    @staticmethod
    @njit
    def _process_start_column(start_col, matrix: ndarray, demons_array: ndarray, demons_lengths: ndarray,
                              demons_costs: ndarray, buffer_size: int|integer, n:int, max_score: int, num_demons: int,
                              init_stack_size, enable_pruning: bool, ):
        """
        Process 1 possible starting position - matrix[0, start_col], then explore possible solution (DFS search) with optional pruning
        :param start_col: index of column to start from
        :param matrix: Task.matrix
        :param demons_array: Task.demons as 2d array padded with -1
        :param demons_lengths: lengths of demons
        :param demons_costs: Task.demons_costs
        :param buffer_size: Task.buffer_size
        :param n: size of matrix
        :param max_score: max possible score ≡ d_costs.sum()
        :param num_demons: number of demons ≡ demons_array.shape[0] ≡ len(Task.demons)
        :param init_stack_size: init stack size
        :param enable_pruning: same as in __call__
        :return: tuple; path found in current branch, score from path
        """

        # Basically just manual implementation of stack using parallel arrays
        # pre-allocating parallel arrays
        max_stack = init_stack_size
        stack_path = empty((max_stack, buffer_size, 2), dtype=int8)
        stack_buff = empty((max_stack, buffer_size), dtype=int8)
        stack_activ = empty((max_stack, num_demons), dtype=bool_)
        stack_score = empty(max_stack, dtype=int16)
        stack_used = empty((max_stack, n, n), dtype=bool_)
        stack_length = empty(max_stack, dtype=int8)

        # tracking of best solution
        best_score = 0
        best_path = empty((buffer_size, 2), dtype=int8); best_path[:] = -1
        best_path_length = 0

        # starting position
        start_r, start_c = 0, start_col
        start_symbol = matrix[start_r, start_c]

        # boolean array as mask to mark used cells
        used_init = zeros((n, n), dtype=bool_)
        used_init[start_r, start_c] = True

        # initial markers for demons activations
        active_init = zeros(num_demons, dtype=bool_)
        score_init = 0
        for d in range(num_demons):
            if demons_lengths[d] == 1 and demons_array[d, 0] == start_symbol:
                active_init[d] = True
                score_init += demons_costs[d]

        # initial path and buffer sequence
        path_init = empty((buffer_size, 2), dtype=int8); best_path[:] = -1
        path_init[0, 0] = start_r
        path_init[0, 1] = start_c
        buffer_init = empty(buffer_size, dtype=int8);   buffer_init[:] = -1
        buffer_init[0] = start_symbol
        length_init = 1

        # pushing initial values onto stack
        pointer = 0
        stack_path[pointer] = path_init
        stack_buff[pointer] = buffer_init
        stack_activ[pointer] = active_init
        stack_score[pointer] = score_init
        stack_used[pointer] = used_init
        stack_length[pointer] = length_init
        pointer += 1

        # dfs search over all possible (correct) next moves
        while pointer > 0:
            pointer -= 1

            # pop from stack
            path = stack_path[pointer]
            buffer = stack_buff[pointer]
            active = stack_activ[pointer]
            curr_score = stack_score[pointer]
            used_mask = stack_used[pointer]
            curr_leng = stack_length[pointer]

            # updating best path if this solution is better
            if curr_score > best_score:
                if curr_leng <= buffer_size:
                    best_score = curr_score
                    best_path_length = curr_leng
                    for d in range(curr_leng):
                        best_path[d, 0] = path[d, 0]
                        best_path[d, 1] = path[d, 1]
                    # express exit if best possible score achieved (also pruning of some kind but necessary one)
                    if best_score == max_score:
                        return best_path[:best_path_length], best_score

            # if path can be extended
            if curr_leng < buffer_size:
                next_step = curr_leng + 1
                last_r = path[curr_leng - 1, 0]
                last_c = path[curr_leng - 1, 1]

                # row/column alternation setting
                if next_step % 2 == 1:
                    row_fixed, col_fixed = last_r, False
                else:
                    row_fixed, col_fixed = False, last_c

                iterator = range(n)
                for idx in iterator:
                    r = last_r if col_fixed else idx
                    c = idx if col_fixed else last_c


                    if not used_mask[r, c]:
                        # preparing new frames (copying current)
                        #TODO: thouse copyies looks shady
                        new_path = path.copy()
                        new_buff = buffer.copy()
                        new_used = used_mask.copy()
                        new_activ = active.copy()
                        new_score = curr_score

                        # new cell
                        new_path[curr_leng, 0] = r
                        new_path[curr_leng, 1] = c
                        new_buff[curr_leng] = matrix[r, c]
                        new_used[r, c] = True
                        new_len = next_step

                        # current demons matching
                        for d in range(num_demons):
                            if not new_activ[d]:
                                k = demons_lengths[d]
                                if new_len >= k:
                                    match = True
                                    for j in range(k):
                                        if new_buff[new_len - k + j] != demons_array[d, j]:
                                            match = False
                                            break
                                    if match:
                                        new_activ[d] = True
                                        new_score += demons_costs[d]

                        # B&B pruning
                        if enable_pruning:
                            rem = 0
                            for d in range(num_demons):
                                if not new_activ[d]:
                                    rem += demons_costs[d]
                            do_push = (new_score + rem > best_score)
                        else:
                            do_push = True

                        # puch children onto stack
                        if do_push:
                            idx = pointer
                            stack_path[idx] = new_path
                            stack_buff[idx] = new_buff
                            stack_activ[idx] = new_activ
                            stack_score[idx] = new_score
                            stack_used[idx] = new_used
                            stack_length[idx] = new_len
                            pointer += 1


        return best_path[:best_path_length], best_score









