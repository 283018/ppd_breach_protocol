from breach_solvers.solvers_protocol import Solver, register_solver
from core import Task, Solution

from typing import Tuple, List, Optional
from dataclasses import dataclass
from time import perf_counter
import numpy as np
# from numpy import ndarray, integer, int8, int16, bool_, array, zeros, empty
# from time import perf_counter


@dataclass
class SolCandidate:
    """
    Temporary encapsulation for solution candidate
    """
    path: np.ndarray
    cost: int
    length: int

    def accept_candidate(self):
        return Solution(path=self.path, total_points=self.cost)


@register_solver('ant_col')
class AntColSolver(Solver):
    _allowed_kwargs = {'n_ant':int, 'n_iterations':int, 'alpha': float, 'beta': float, 'evaporation': float, 'q': float}

    def __call__(self, task: Task, **kwargs):
        self._validate_kwargs(kwargs)
        self.n_ants = kwargs.get('n_ants', 10)
        self.n_iterations = kwargs.get('n_iterations', 100)
        self.alpha = kwargs.get('alpha', 1.0)
        self.beta = kwargs.get('beta', 2.0)
        self.evaporation = kwargs.get('evaporation', 0.1)
        self.q = kwargs.get('q', 1.0)


        self.task = task
        self.n = task.matrix.shape[0]
        size = self.n * self.n
        self.pheromone = np.ones((size, size), dtype=float)
        self.heuristic = self._compute_heuristic()

        best: Optional[SolCandidate] = None

        start = perf_counter()
        for _ in range(self.n_iterations):
            ants_sols: List[SolCandidate] = []
            for _ in range(self.n_ants):
                sol = self._construct_solution()
                ants_sols.append(sol)
                if best is None or (sol.cost > best.cost) or (sol.cost == best.cost and sol.length < best.length):
                    best = sol
            top_k = sorted(ants_sols, key=lambda s: (-s.cost, s.length))[:max(1, self.n_ants // 2)]
            self._update_pheromones(top_k)

        end = perf_counter()

        return best.accept_candidate().fill_solution(self.task), end-start

    def _idx(self, r: int, c: int) -> int:
        return r * self.n + c

    def _rc(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.n)

    def _compute_heuristic(self) -> np.ndarray:
        freq = {symbol: 0 for symbol in range(1, 100)}
        for demon in self.task.demons:
            for s in demon:
                freq[int(s)] += 1

        h = np.zeros(self.n * self.n, dtype=float)
        for idx in range(self.n * self.n):
            r, c = self._rc(idx)
            val = int(self.task.matrix[r, c])
            h[idx] = freq[val] + 1e-6
        return h

    def _score_buffer(self, buffer: List[int]) -> int:
        total = 0
        seq = np.array(buffer, dtype=int)
        for demon, cost in zip(self.task.demons, self.task.demons_costs):
            dlen = len(demon)
            for i in range(len(seq) - dlen + 1):
                if np.array_equal(seq[i:i+dlen], demon):
                    total += int(cost)
                    break
        return total

    def _construct_solution(self) -> SolCandidate:
        n, buf_size = self.n, self.task.buffer_size
        visited = set()
        path: List[Tuple[int,int]] = []
        buffer_vals: List = []
        is_even_step = False

        candidates = [(0, c) for c in range(n)]
        move = self._choose_move(-1, candidates)
        path.append(move)
        visited.add(move)
        buffer_vals.append(self.task.matrix[move])

        while len(path) < buf_size:
            last_r, last_c = path[-1]
            if is_even_step:
                candidates = [(last_r, c) for c in range(n) if c != last_c]
            else:
                candidates = [(r, last_c) for r in range(n) if r != last_r]
            candidates = [mv for mv in candidates if mv not in visited]
            if not candidates:
                break
            move = self._choose_move(path[-1], candidates)
            path.append(move)
            visited.add(move)
            buffer_vals.append(self.task.matrix[move])
            is_even_step = not is_even_step

        cost = self._score_buffer(buffer_vals)
        return SolCandidate(path=np.array(path, dtype=np.int8), cost=cost, length=len(path))

    def _choose_move(
        self,
        last: Tuple[int,int] or int,
        candidates: List[Tuple[int,int]],
    ) -> Tuple[int,int]:
        idx_last = last if isinstance(last, int) else self._idx(*last)
        scores = []
        for (r, c) in candidates:
            idx_next = self._idx(r, c)
            tau = self.pheromone[idx_last, idx_next] ** self.alpha
            eta = self.heuristic[idx_next] ** self.beta
            scores.append(tau * eta)
        scores = np.array(scores, dtype=float)
        probs = scores / scores.sum()
        choice = np.random.choice(len(candidates), p=probs)
        return candidates[choice]

    def _update_pheromones(self, solutions: List[SolCandidate]):
        self.pheromone *= (1 - self.evaporation)
        for sol in solutions:
            deposit_amt = self.q * (sol.cost / sol.length if sol.length > 0 else 0)
            for i in range(sol.length - 1):
                a = tuple(sol.path[i])
                b = tuple(sol.path[i+1])
                idx_a, idx_b = self._idx(*a), self._idx(*b)
                self.pheromone[idx_a, idx_b] += deposit_amt