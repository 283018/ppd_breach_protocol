from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from numpy import array, int8, zeros, empty
from multiprocessing import Pool, cpu_count
from functools import partial
from time import perf_counter


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

        time_start = perf_counter()

        # aktualizuje multiprocessing dla wszystkich startowych pozycji (komórek w pierwszym rzędzie)
        # TODO: sequential if ...
        with Pool(processes=process_num) as pool:
            worker = partial(self._process_start_column,
                             matrix=matrix,
                             demons=demons,
                             d_costs=d_costs,
                             buffer_size=buffer_s,
                             n=n,
                             max_score=max_score,
                             to_prune=to_prune,
                             )

            results = pool.map(worker, range(n))
        time_end = perf_counter()

        # Znajduje najlepszy wynik ze wszystkich startowych kolumn
        best_score = 0
        best_path = []
        for path, score in results:
            if score > best_score or (score == best_score and len(path) < len(best_path)):
                best_score = score
                best_path = path


        final_solution = array(best_path, dtype=int8) if best_path else array([], dtype=int8)
        return Solution(final_solution).fill_solution(task), time_end - time_start

    @staticmethod
    def _process_start_column(start_col, matrix, demons, d_costs, buffer_size, n, max_score, to_prune):
        best_score = 0
        best_path = []
        demons = [tuple(d) for d in demons]
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
            if len(demons[i]) == 1 and demons[i][0] == start_symbol:
                activated[i] = True
                current_score += d_costs[i]

        stack = [(
            [(start_r, start_c)],
            [start_symbol],
            activated.copy(),
            current_score,
            used.copy()
        )]

        while stack:
            path, buffer, activated, score, used = stack.pop()

            if score > best_score:
                best_score = score
                best_path = path.copy()
                if best_score == max_score:
                    return best_path, best_score

            if len(path) < buffer_size:
                next_step = len(path) + 1
                last_r, last_c = path[-1]

                if next_step % 2 == 1:  # rząd
                    cells = [(last_r, c) for c in range(n) if not used[last_r, c]]
                else:   # kolumna
                    cells = [(r, last_c) for r in range(n) if not used[r, last_c]]

                for r, c in cells:
                    new_path = path + [(r, c)]
                    new_symbol = matrix[r, c]
                    new_buffer = buffer + [new_symbol]

                    new_used = used.copy()
                    new_used[r, c] = True

                    new_activated = activated.copy()
                    new_score = score

                    # sprawdzenie wszystkich demonów
                    for i in range(num_demons):
                        if not new_activated[i]:
                            demon = demons[i]
                            k = len(demon)
                            if len(new_buffer) >= k:
                                if tuple(new_buffer[-k:]) == demon:
                                    new_activated[i] = True
                                    new_score += d_costs[i]

                    # pruning jeśli nie da się pobić aktualny najlepszy wynik
                    # (wtedy niezużyty buffor nie ma znaczenia, ale przyspiesza to wyszukiwanie)
                    remaining = sum(d_costs[i] for i in range(num_demons) if not new_activated[i])
                    if not to_prune or (new_score + remaining > best_score):
                        stack.append((new_path, new_buffer, new_activated, new_score, new_used))

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








