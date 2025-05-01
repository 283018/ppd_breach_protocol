from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from numpy import ndarray, integer, array, zeros, empty, full
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

        n = matrix.shape[0]
        max_score = d_costs.sum()
        process_num = min(n, cpu_count())
        d_lengths = array([d.size for d in demons])

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
    def _process_start_column(start_col, matrix:ndarray, demons:Tuple[ndarray], demons_lengths:ndarray, d_costs:ndarray, buffer_size:int|integer, n:int, max_score:int, to_prune:bool):
        best_score = 0
        best_path = full((buffer_size, 2), -1, dtype='int8')
        best_path_length = 0

        num_demons = len(demons)
        start_r, start_c = 0, start_col
        start_symbol = matrix[start_r, start_c]

        # boolean array instead of bitmask
        used = zeros((n, n), dtype='bool')
        used[start_r, start_c] = True

        activated = zeros(num_demons, dtype='bool')
        current_score = 0

        # sprawdzenie na demonów długości 1
        for i in range(num_demons):
            if demons_lengths[i] == 1 and demons[i][0] == start_symbol:
                activated[i] = True
                current_score += d_costs[i]

        path_array = full((buffer_size, 2), -1, dtype='int8')
        path_array[0, 0] = start_r
        path_array[0, 1] = start_c

        buffer_array = full(buffer_size, -1, dtype='int8')
        buffer_array[0] = start_symbol

        length = 1

        stack = [(
            path_array.copy(),
            buffer_array.copy(),
            activated.copy(),
            current_score,
            used.copy(),
            length,
        )]

        while len(stack) > 0:
            path, buffer, activated, score, used, curr_len = stack.pop()

            if score > best_score:
                best_score = score
                best_path_length = curr_len
                # manual copy only up to curr_len
                for i in range(curr_len):
                    best_path[i, 0] = path[i, 0]
                    best_path[i, 1] = path[i, 1]
                if best_score == max_score:
                    return best_path, best_score

            if curr_len < buffer_size:
                next_step = curr_len + 1
                last_r = path[curr_len - 1, 0]
                last_c = path[curr_len - 1, 1]

                if next_step % 2 == 1:
                    cells = []
                    for c in range(n):
                        if not used[last_r, c]:
                            cells.append((last_r, c))
                else:
                    cells = []
                    for r in range(n):
                        if not used[r, last_c]:
                            cells.append((r, last_c))

                for cell in cells:
                    r, c = cell

                    new_path = path.copy()
                    new_path[curr_len] = (r, c)

                    new_buff = buffer.copy()
                    new_buff[curr_len] = matrix[r, c]

                    new_used = used.copy()
                    new_used[r, c] = True

                    new_activ = activated.copy()
                    # print(new_activ)
                    new_score = score

                    # sprawdzenie wszystkich demonów
                    for i in range(num_demons):
                        if not new_activ[i]:
                            k = demons_lengths[i]
                            if curr_len + 1 >= k:
                                match = True
                                for j in range(k):
                                    idx = curr_len + 1 - k + j
                                    if new_buff[idx] != demons[i][j]:
                                        match = False
                                        break
                                if match:
                                    new_activ[i] = True
                                    new_score += d_costs[i]

                    # pruning jeśli nie da się pobić aktualny najlepszy wynik
                    # (wtedy niezużyty buffor nie ma znaczenia, ale przyspiesza to wyszukiwanie)
                    remaining = 0
                    for i in range(num_demons):
                        if not new_activ[i]:
                            remaining += d_costs[i]
                    if not to_prune or (new_score + remaining > best_score):
                        stack.append((
                            new_path,
                            new_buff,
                            new_activ,
                            new_score,
                            new_used,
                            curr_len + 1
                        ))

        return best_path, best_score








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








