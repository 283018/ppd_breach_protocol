from breach_solvers.solvers_abc import SeedableSolver, register_solver
from core import Task, Solution, DUMMY_TASK
from cpp import ant_colony

from numpy import array, full, ones, array_equal, concatenate, bincount, ndarray, integer, int8
from numpy.random import default_rng, Generator
from numpy import sum as npsum
from dataclasses import dataclass
from time import perf_counter

from typing import List, Tuple
from numpy.typing import NDArray


RANGE_INT32 = (0, 2**32-1)

@dataclass
class SolCandidate:
    """
    Temporary encapsulation for solution candidate
        - path: ndarray
        - cost: int
        - length: int
        - buffer_seq: List[int|integer] | Tuple[int|integer] | NDArray[int8]
    """

    path: ndarray
    cost: int
    length: int
    buffer_seq: List[int|integer] | Tuple[int|integer] | NDArray[int8]

    def accept(self):
        return Solution(
            path=self.path,
            buffer_sequence=array(self.buffer_seq),
            total_points=self.cost,
        )


@register_solver("ant_col")
class AntColSolver(SeedableSolver):
    _allowed_kwargs = {
        "avoid_c_back":bool,

        "n_ant": int,
        "n_iterations": int,
        "alpha": float,
        "beta": float,
        "evaporation": float,
        "q": float,
    }

    _DEFAULT_PARAMS = {
        "avoid_c_back":False,

        "n_ants": 50,
        "n_iterations": 100,
        "alpha": 1.0,
        "beta": 1.0,
        "evaporation": 0.05,
        "q": 1.0,
    }

    rng: Generator

    def _warm_up(self):
        self.rng = default_rng()
        try:
            start_init = perf_counter()
            self.solve(DUMMY_TASK)
            end_init = perf_counter()
        except Exception as e:
            raise RuntimeError(f"Error while initialization ant-colony solver occurred: {e}") from e
        else:
            print(f"\rSuccessfully initialized ant-colony solver in {end_init - start_init:.4} sec", flush=True,)


    def seed(self, seed):
        if not isinstance(seed, (int, integer)):
            raise TypeError("seed must be an integer")
        self.rng = default_rng(seed)


    def solve(self, task: Task, **kwargs):
        """
        Ant-colony solver.

        Meta-heuristic approach, uses ant-colony optimization algorithm.
        Uses c++ module if possible.
        If c++ back fails, run python based methods, which takes ~10-12 more time

        Possible keyword arguments:
            - avoid_c:bool=False - if True skip call to c++ back and jump to pure python implementation.

            - n_ants:int - amount of ants to run in one iteration.
            - n_iterations:int - amount of iterations.
            - alpha:float - attractiveness of pheromone trail on ants.
            - beta:float - importance of heuristic matrix (attractiveness of cells with higher reward)
            - evaporation:float - pheromone decay tempo.
            - q:float - amount of pheromones each ant leave on their path

        :param task: Task instance.
        :param kwargs:
        :return: found Solution, main execution time (without pre-calculation)
        """

        self._validate_kwargs(kwargs)
        params = {**self._DEFAULT_PARAMS, **kwargs}
        avoid_c_back = params["avoid_c_back"]

        n_ants = params["n_ants"]
        n_iterations = params["n_iterations"]

        alpha = params["alpha"]
        beta = params["beta"]
        evaporation = params["evaporation"]
        q = params["q"]


        matrix = task.matrix
        demons = task.demons
        demons_costs = task.demons_costs
        buffer_size = task.buffer_size

        # pre-calculations
        n, _ = matrix.shape
        size = matrix.size

        num_demons = len(demons)
        demons_lengths = array([d.size for d in demons])
        max_demon_len = max(demons_lengths)
        padded_demons = full((num_demons, max_demon_len), -1, dtype=int8)
        for i, d in enumerate(demons):
            padded_demons[i, :d.size] = d
        flat_demons = padded_demons.ravel()

        # starting pheromones and heuristic
        pheromone = ones((size, size), dtype=float)
        heuristic = self._get_freqs(task)
        seed =  self.rng.integers(*RANGE_INT32)

        # creating args pools
        common_args = (
            matrix, demons,
            demons_costs, buffer_size,
            n_ants, n_iterations,
            n, pheromone, heuristic,
            alpha, beta,
            evaporation, q,
        )

        cpp_args = (
            matrix, flat_demons,
            demons_costs, buffer_size,
            n, num_demons,
            demons_lengths, max_demon_len,
            heuristic,
            alpha, beta,
            evaporation, q,
            seed, n_ants, n_iterations,
        )
        

        if avoid_c_back:
            print("\nJumped to pyton")
            start = perf_counter()
            best = self._run_ants(*common_args)
            end = perf_counter()
        else:
            try:
                start = perf_counter()
                res = ant_colony(*cpp_args)
                end = perf_counter()
                best = SolCandidate(*res)
            except Exception as e:
                print(f"\nError on c++ back, running python ({e})")
                start = perf_counter()
                best = self._run_ants(*common_args)
                end = perf_counter()

        return best.accept().fill_solution(task), end - start

    @staticmethod
    def _get_freqs(task):
        """
        |Precalculation|: Calculate attractiveness of each cell (flattened), based on frequency of symbols appearance
        """
        all_symbols = concatenate(task.demons)
        occurrence = bincount(all_symbols, minlength=100)
        h = occurrence[task.matrix.ravel()]

        return h

    def _run_ants(self, matrix, demons, demons_costs, buffer_size, n_ants, n_iterations, n, pheromone, heuristic, alpha, beta, evaporation, q):
        """Main loop"""
        best = None
        for _ in range(n_iterations):
            solutions = self._get_solutions_candidates(n_ants, matrix, demons, demons_costs, buffer_size, n, pheromone, heuristic, alpha, beta)
            best = self._update_best(best, solutions)
            top_solutions = self._select_top_solutions(solutions, n_ants)
            self._update_pheromones(top_solutions, pheromone, n, evaporation, q)
        return best


    def _get_solutions_candidates(self, n_ants, matrix, demons, demons_costs, buffer_size, n, pheromone, heuristic, alpha, beta):
        solutions = []
        for i in range(n_ants):
            solutions.append(self._construct_solution(matrix, demons, demons_costs, buffer_size, n, pheromone, heuristic, alpha, beta))
        return solutions


    def _construct_solution(self, matrix, demons, demons_costs, buffer_size, n, pheromone, heuristic, alpha, beta):
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
        buffer_vals.append(matrix[current])

        while len(path) < buffer_size:
            current = self._next_move(
                current, is_even_step, n, visited, pheromone, heuristic, alpha, beta)

            if current is None:
                break
            path.append(current)
            visited.add(current)
            buffer_vals.append(matrix[current])
            is_even_step = not is_even_step

        # scoring paths
        total = 0
        seq = array(buffer_vals, dtype=int)
        for demon, cost in zip(demons, demons_costs):
            dlen = len(demon)
            if dlen == 0:
                continue
            for i in range(len(seq) - dlen + 1):
                if array_equal(seq[i : i + dlen], demon):
                    total += cost
                    break

        return SolCandidate(
            path=array(path, dtype=int8),
            cost=total,
            length=len(path),
            buffer_seq=buffer_vals,
        )

    def _next_move(self, last, is_even, n, visited, pheromone, heuristic, alpha, beta):
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
            tau = pheromone[last_idx, idx] ** alpha
            eta = heuristic[idx] ** beta
            scores.append(tau * eta)

        probs = array(scores) / npsum(scores)
        return candidates[self.rng.choice(len(candidates), p=probs)]

    @staticmethod
    def _update_best(current_best, candidates):
        """
        Update global best solution
        """
        if not current_best:
            return max(candidates, key=lambda x: (x.cost, -x.length * 0.1))

        best_candidate = max(
            candidates + [current_best], key=lambda x: (x.cost, -x.length * 0.1)
        )
        return best_candidate

    @staticmethod
    def _select_top_solutions(solutions, n_ants):
        """
        Returns promising solutions for pheromone update (1, n_ants//2)
        """
        k = max(1, n_ants // 2)
        return sorted(solutions, key=lambda s: (-s.cost, s.length * 0.1))[:k]

    # * TODO: need be careful with that mutability
    @staticmethod
    def _update_pheromones(solutions, pheromone, n, evaporation, q):
        """
        Update pheromone trails based on best solutions
        """
        pheromone *= 1 - evaporation
        for sol in solutions:
            if sol.length <= 1:
                continue
            deposit = q * sol.cost / sol.length
            for i in range(sol.length - 1):
                from_idx = _to_flat(*sol.path[i], n)
                to_idx = _to_flat(*sol.path[i + 1], n)
                pheromone[from_idx, to_idx] += deposit



def _to_flat(r, c, n):
    """Convert 2d matrix coords to 1d index"""
    return r * n + c

def _to_shape(idx, n):
    """Convert 1D index to 2d matrix coords"""
    return divmod(idx, n)
