from breach_solvers.solvers_abc import SeedableSolver, register_solver
from core import Task, Solution, DUMMY_TASK, NoSolution
from cpp import ant_colony

from numpy import array, full, ones, array_equal, concatenate, bincount, ndarray, integer, int8
from numpy.random import default_rng, Generator
from numpy import sum as npsum
from dataclasses import dataclass
from time import perf_counter
from warnings import warn

from typing import List, Tuple, Self
from numpy.typing import NDArray


RANGE_UINT32 = (0, 2**32-1)
RANGE_INT32 = (-2**31, 2**31-1)

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


@register_solver('ac', 'ant_col')
class AntColSolver(SeedableSolver):
    """Ant-Colony solver"""
    _allowed_kwargs = {
        "avoid_c_back":bool,

        "n_ant": int,
        "max_iter": int,
        "stag_lim": int,

        "alpha": float,
        "beta": float,
        "evap": float,
        "q": float,
    }

    _DEFAULT_HIPERPARAMS = {
        "alpha": 0.1,
        "beta": 0.4,
        "evap": 0.475,
        "q": 250.,
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


    def seed(self, seed:int|integer=None) -> Self:
        """
        Set or re-set RNG seed for reproducibility, enabling method chaining.

        :param seed: RNG seed. If None, uses system-generated state.
        :return: Self instance
        """
        if seed is not None and not isinstance(seed, (int, integer)):
            raise TypeError("seed must be an integer")
        self.rng = default_rng(seed)
        return self


    def solve(self, task: Task, **kwargs):
        """
        Ant-colony solver.

        Meta-heuristic approach, using ant-colony optimization (ACO) algorithm.
        Uses c++ module if possible, falls back to python implementation if c++ fails.

        ----
        .. warning::
            ****WARNING****: Using ``stagnant_limit<0<n_iterations`` may cause unreasonable long loops with 2³¹-1 iterations.
        ----

        .. note::
           Python implementation is significantly slower and outdated compared to the C++ version.

        .. seealso::
            ``.seed()`` allow to set seed for reproducibility. If seed not set, uses system-generated state.

        Keyword arguments:
           - **avoid_c**: *bool* = ``False``
             If True, skips C++ backend and uses pure Python implementation.
             *NOT RECOMMENDED*: Python implementation is much slower and generally deprecated.
           - **n_ants**: *int* = ``task.matrix.size``
             Number of ants per iteration.
           - **max_iter**: *int* = ``0``
             Maximum iterations; set to `INT_MAX` (2³¹-1) if ≤ 0, stopping only when exceeding `stag_lim`.
           - **stag_lim**: *int* = ``n*buffer_size*num_demons``
             Allowed iterations without improvement.
           - **alpha**: *float* = ``0.1``
             Attractiveness of pheromone trails.
           - **beta**: *float* = ``0.4``
             Heuristic matrix importance (reward-based cell attractiveness).
           - **evap**: *float* = ``0.475``
             Pheromone decay rate.
           - **q**: *float* = ``250.0``
             Pheromone deposited per ant on their path.

        Stagnant limit modes:
            * ``==0``: Default - calculated from task parameters
            * ``<0``: No stagnation control
            * ``>0``: Hard limit on stagnant iterations

        :param task: ``Task`` instance.
        :param kwargs:
        :return: ``tuple``: (found ``Solution`` or ``NoSolution``, ``execution_time``)
            excluding pre- and post- calculations.
        """
        self._validate_kwargs(kwargs)
        params = {**self._DEFAULT_HIPERPARAMS, **kwargs}
        avoid_c_back = params.get("avoid_c_back", False)

        alpha = params["alpha"]
        beta = params["beta"]
        evaporation = params["evap"]
        q = params["q"]

        n_ants = params.get("n_ants", None)
        n_iterations = params.get("max_iter", 0)
        stagnant_limit = params.get("stag_lim", 0)

        if stagnant_limit < 0 < n_iterations:
            warn("Blah blah infinite execution blah")


        matrix = task.matrix
        demons = task.demons
        demons_costs = task.demons_costs
        buffer_size = task.buffer_size

        # pre-calculations
        n, _ = matrix.shape
        size = matrix.size

        n_ants = n_ants or size

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
        seed =  self.rng.integers(*RANGE_UINT32)

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
            stagnant_limit,
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
                warn(f"\nError on c++ back, running python\n   {e}")
                start = perf_counter()
                best = self._run_ants(*common_args)
                end = perf_counter()

        if best.path.size == 0:
            return NoSolution(reason="No valid solution possible for given task"), 0.0

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
