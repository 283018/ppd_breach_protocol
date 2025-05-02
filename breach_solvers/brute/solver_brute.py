from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from numpy import ndarray, integer, array, zeros, empty, full, dtype
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Tuple
from time import perf_counter

from numba import njit

from icecream import ic


# na razie nie jest zbytnio zoptymalizowany, mimo że używa multiprocessing, zwłaszcza dla dużych macierzy
# planuje go bardziej zoptymalizować, usunąć niepotrzebne konwersje na listy
# (a w ideale w ogóle się się pozbyć i działać tylko na numpy/ewentualnie deque)
# Jeśli się uda to użyć jit-compiler z numba, który ma wbudowany multiprocessing dla pętli


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
        n_demons = len(demons)

        n = matrix.shape[0]
        max_score = d_costs.sum()
        d_lengths = array([d.size for d in demons])

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
                             stack_dtype=self.make_stack_dtype(buffer_s, n_demons, n),
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
    def make_stack_dtype(buffer_size, num_demons, n):
        return dtype([
            ("path", 'int8', (buffer_size, 2)),
            ("buff", 'int8', (buffer_size,)),
            ("activ", 'bool', (num_demons,)),
            ("score", 'int16'),
            ("used", 'bool', (n, n)),
            ("length", 'int8'),
        ])

    @staticmethod
    # @njit
    def _process_start_column(start_col,  # starting column (setted by multiprocessing)
                              matrix: ndarray, demons: Tuple[ndarray], demons_lengths: ndarray, d_costs: ndarray, buffer_size: int | integer, # specs of task itself
                              n: int, max_score: int, num_demons: int,                  # pre-calculated parameters
                              stack_dtype,
                              to_prune: bool,
                              ) -> Tuple[ndarray, integer]:
        # TODO: max stack instead of 100
        stack = empty(100, dtype=stack_dtype)

        best_score = 0
        best_path = full((buffer_size, 2), -1, dtype='int8')
        best_path_length = 0

        start_r, start_c = 0, start_col
        start_symbol = matrix[start_r, start_c]

        # boolean array instead of bitmask
        used0 = zeros((n, n), dtype='bool')
        used0[start_r, start_c] = True

        activ0 = zeros(num_demons, dtype='bool')
        current_score0 = 0

        # sprawdzenie na demonów długości 1
        for i in range(num_demons):
            if demons_lengths[i] == 1 and demons[i][0] == start_symbol:
                activ0[i] = True
                current_score0 += d_costs[i]

        path0 = full((buffer_size, 2), -1, dtype='int8')
        path0[0, 0] = start_r
        path0[0, 1] = start_c

        buffer0 = full(buffer_size, -1, dtype='int8')
        buffer0[0] = start_symbol
        length0 = 1

        # Basically manual stack implementation by bundled parallel pre-allocated arrays
        # Setting first elements on stack
        top = 0
        stack[top]['path'] = path0
        stack[top]['buff'] = buffer0
        stack[top]['activ'] = activ0
        stack[top]['score'] = current_score0
        stack[top]['used'] = used0
        stack[top]['length'] = length0
        top += 1

        while top > 0:
            top -= 1

            # pop from stack
            rec = stack[top]
            path = rec['path']
            buff = rec['buff']
            activ = rec['activ']
            score = rec['score']
            used = rec['used']
            curr_len = rec['length']

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
                            new_len = curr_len + 1

                            # copy-on-write arrays
                            new_path = path.copy()
                            new_path[curr_len] = (last_r, c)
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
                            if not to_prune:
                                do_push = True
                            else:
                                rem = 0
                                for i in range(num_demons):
                                    if not new_activ[i]:
                                        rem += d_costs[i]
                                do_push = (new_score + rem > best_score)

                            if do_push:
                                idx = top
                                stack[idx]['path'] = new_path
                                stack[idx]['buff'] = new_buff
                                stack[idx]['activ'] = new_activ
                                stack[idx]['score'] = new_score
                                stack[idx]['used'] = new_used
                                stack[idx]['length'] = new_len
                                top += 1

                else:
                    # move along column
                    for r in range(n):
                        if not used[r, last_c]:
                            new_len = curr_len + 1

                            new_path = path.copy()
                            new_path[curr_len] = (r, last_c)
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

                            if not to_prune:
                                do_push = True
                            else:
                                rem = 0
                                for i in range(num_demons):
                                    if not new_activ[i]:
                                        rem += d_costs[i]
                                do_push = (new_score + rem > best_score)

                            if do_push:
                                idx = top
                                stack[idx]['path'] = new_path
                                stack[idx]['buff'] = new_buff
                                stack[idx]['activ'] = new_activ
                                stack[idx]['score'] = new_score
                                stack[idx]['used'] = new_used
                                stack[idx]['length'] = new_len
                                top += 1

        return best_path[:best_path_length], best_score








# if __name__ == '__main__':
#     from core import solution_print
#
#     matrix1 = np.array([
#         [3, 1, 5, 5, 3, 5, 2],
#         [5, 6, 2, 2, 5, 5, 1],
#         [5, 5, 4, 2, 6, 5, 2],
#         [1, 2, 1, 3, 2, 2, 1],
#         [1, 5, 2, 4, 6, 6, 4],
#         [3, 1, 3, 5, 5, 2, 3],
#         [3, 5, 1, 5, 6, 3, 2],
#     ], dtype='int8')
#
#     demons1 = (
#         np.array([1, 2], dtype='int8'),
#         np.array([3, 4], dtype='int8'),
#         np.array([5, 1, 2], dtype='int8')
#     )
#     d_costs1 = np.array([1, 2, 3], dtype='int8')
#     buffer_size1 = 8
#
#     task1 = Task(matrix1, demons1, d_costs1, buffer_size1)
#
#     solver1 = BruteSolver()
#
#
#     sol1 = solver1(task1)
#
#     solution_print(task1, sol1)








