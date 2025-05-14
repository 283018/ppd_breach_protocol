from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from typing import Tuple, List, Optional
from dataclasses import dataclass
from time import perf_counter

# from numpy import ndarray, integer, int8, int16, bool_, array, zeros, empty
import numpy as np
from numpy.random import default_rng, Generator
from numpy import integer


from icecream import ic


@dataclass
class SolCandidate:
    """
    Temporary encapsulation for solution candidate
    """
    path: np.ndarray
    cost: int
    length: int
    buffer_seq: List[int|integer]

    def accept(self):
        return Solution(path=self.path, buffer_sequence=np.array(self.buffer_seq), total_points=self.cost)


@register_solver('ant_col')
class AntColSolver(Solver):
    _allowed_kwargs = {'n_ant':int,  'n_iterations':int,
                       'alpha': float,       'beta': float,
                       'evaporation': float,    'q': float}

    _DEFAULT_PARAMS = {'n_ants': 50, 'n_iterations': 100, 'alpha': 1.0,
                       'beta': 1.0, 'evaporation': 0.05, 'q': 1.0}

    rng: Generator


    def seed(self, seed):
        if not isinstance(seed, (int, integer)):
            raise TypeError("seed must be an integer")
        self.rng = default_rng(seed)


    def __call__(self, task: Task, **kwargs):
        # if not self._initialized:
        #     self._warm_up()

        # params setup
        self._validate_kwargs(kwargs)
        params = {**self._DEFAULT_PARAMS, **kwargs}
        n_ants = params['n_ants']
        n_iterations = params['n_iterations']

        self.alpha = params['alpha']
        self.beta = params['beta']
        self.evaporation = params['evaporation']
        self.q = params['q']

        # pre-calculations
        n = task.matrix.shape[0]
        size = task.matrix.size

        # starting pheromones and heuristic
        pheromone = np.ones((size, size), dtype=float)
        heuristic = self._get_freqs(task, n, size)
        best = None

        start = perf_counter()
        for _ in range(n_iterations):
            # construct solutions for all ants
            solutions = [self._construct_solution(task, n, pheromone, heuristic) for _ in range(n_ants)]
            # global best
            best = self._update_best(best, solutions)
            # most promising solutions
            top_solutions = self._select_top_solutions(solutions, n_ants)
            # update pheromone trails
            self._update_pheromones(top_solutions, pheromone, n)
        end = perf_counter()

        return best.accept().fill_solution(task), end - start


    @staticmethod
    def _get_freqs(task: Task, n, size):
        """
        Calculate attractiveness of each cell (flattened), based on frequency of symbols appearance
        """
        occurrence = np.zeros(100, dtype=int)
        for demon in task.demons:
            for s in demon:
                occurrence[s] += 1

        h = np.zeros(size, dtype=int)
        for idx in range(size):
            r, c = _to_shape(idx, n)
            h[idx] = occurrence[task.matrix[r, c]]
        # ic(h.reshape((n, n)))
        return h

    def _construct_solution(self, task, n, pheromone, heuristic):
        """
        Construct single path based on pheromone trails and occurrences of symbols (heuristic)
        """
        path = []
        buffer_vals = []
        visited = set()
        is_even_step = False
        current = (0, self.rng.integers(n))

        path.append(current)
        visited.add(current)
        buffer_vals.append(task.matrix[current])

        while len(path) < task.buffer_size:
            current = self._next_move(current, is_even_step, n, visited, pheromone, heuristic)

            if current is None:
                break
            path.append(current)
            visited.add(current)
            buffer_vals.append(task.matrix[current])
            is_even_step = not is_even_step

        # scoring paths
        total = 0
        seq = np.array(buffer_vals, dtype=int)
        for demon, cost in zip(task.demons, task.demons_costs):
            dlen = len(demon)
            if dlen == 0:
                continue
            for i in range(len(seq) - dlen + 1):
                if np.array_equal(seq[i:i + dlen], demon):
                    total += cost
                    break

        return SolCandidate(
            path=np.array(path, dtype=np.int8),
            cost=total,
            length=len(path),
            buffer_seq = buffer_vals
        )

    def _next_move(self, last, is_even, n, visited, pheromone, heuristic):
        """
        Select nest move according to P(ij) formula
        """
        # generating valid candidates from current position
        r, c = last
        if is_even:
            candidates = [(r, col) for col in range(n) if col != c and (r, col) not in visited]
        else:
            candidates = [(row, c) for row in range(n) if row != r and (row, c) not in visited]

        if not candidates:
            return None

        # selecting next move according to:
        # P(ij) = (τ_ij^α * ɳ_j^β) / ∑(τ_ik^α * ɳ_k^β)
        # τ - pheromone value, ɳ - heuristic (frequencies) value
        last_idx = _to_flat(*last, n)
        scores = []
        for r, c in candidates:
            idx = _to_flat(r, c, n)
            tau = pheromone[last_idx, idx] ** self.alpha
            eta = heuristic[idx] ** self.beta
            scores.append(tau * eta)

        probs = np.array(scores) / np.sum(scores)
        return candidates[self.rng.choice(len(candidates), p=probs)]


    @staticmethod
    def _update_best(current_best, candidates):
        """
        Update global best solution
        """
        if not current_best:
            return max(candidates, key=lambda x: (x.cost, -x.length*0.1))

        best_candidate = max(candidates + [current_best], key=lambda x: (x.cost, -x.length*0.1))
        return best_candidate

    @staticmethod
    def _select_top_solutions(solutions, n_ants):
        """
        Returns promising solutions for pheromone update (1, n_ants//2)
        """
        k = max(1, n_ants // 2)
        return sorted(solutions, key=lambda s: (-s.cost, s.length*0.1))[:k]

    #* TODO: need be careful with that mutability
    def _update_pheromones(self, solutions, pheromone, n):
        """
        Update pheromone trails based on best solutions
        """
        pheromone *= (1 - self.evaporation)
        for sol in solutions:
            if sol.length <= 1:
                continue
            deposit = self.q * sol.cost / sol.length
            for i in range(sol.length - 1):
                from_idx = _to_flat(*sol.path[i], n)
                to_idx = _to_flat(*sol.path[i+1], n)
                pheromone[from_idx, to_idx] += deposit



def _to_flat(r, c, n):
    """Convert 2d matrix coords to 1d index"""
    return r * n + c

def _to_shape(idx, n):
    """Convert 1D index to 2D matrix coords"""
    return divmod(idx, n)
