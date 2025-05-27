from numpy import ndarray, int8, zeros, integer
from numpy.random import Generator, SeedSequence, default_rng, PCG64

from .breach_generator import BPGen, CallableGenerator
from core import Task

from typing import List, Self


TASK_MODES = set(range(-3, 5))

class TaskFactory:
    """
    Factory for Task, callable object.

    Main method for generation is ``.gen()``, supports shortcut via ``.__call__()``
    Secondary method is ``.gen_manual()``

    .. seealso::
        ``.gen()``
          ``mode:int=0``:
            - ``-3``: Varying complication (easy)
            - ``-2``: Varying complication (medium)
            - ``-1``: Varying complication (hard)
            - ``0``:  Default, high deviation for varying complication
            - ``1``:  Set size and lengths, similar to base game BP (easy)
            - ``2``:  Set size and lengths (medium)
            - ``3``:  Set size and lengths (hard)
            - ``4``:  Set size and lengths (very hard)
          ``amount:int=1``
        ``.gen_manual()``
           ``matrix_size``:
             - Size of matrix to generate.
           ``demons_specs``:
             - Demon configuration: dict {length: count} or array[i] = count of length i.
           ``buffer_size``:
             - Buffer capacity for the generated task.
           ``matrix_mode``:
             - Mode selection (see matrix mode description).
             - ``0``: Standard minigame with base game symbols (equal chances) [3-6]
             - ``1``: DLC minigame (mostly DLC symbols, with small changes for base symbols) [5-8]
             - ``2``: Full symbol set with decreasing appearance chances [6-12]
             - ``3``: Full symbol set with equal chances (>10)
             - ``4``: *"Good luck xD"* - Not recommended
           ``amount:int=1``:
             - Optional number of tasks to generate.


    :param seed: optional, seed for main rng, rest derived from it.
    """
    _rng: Generator
    _mat_gen: CallableGenerator
    _demons_gen: CallableGenerator
    _costs_gen: CallableGenerator

    def __init__(self, seed: int|integer=None):
        self.reseed(seed)

    def reseed(self, seed: int|integer=None) -> Self:
        """
        Reset internal RNG states with optional seed, enabling method chaining.

        :param seed: RNG seed. If None, uses system-generated state.
        :return: Self instance
        """
        self._rng = default_rng(seed)

        main_ss_seed = self._rng.integers(0, 2 ** 32)
        main_ss = SeedSequence(main_ss_seed)
        child_gens_seeds = main_ss.spawn(3)
        child_gens = [Generator(PCG64(seed)) for seed in child_gens_seeds]

        # pain...
        self._mat_gen = BPGen.create('matrix', child_gens[0])
        self._demons_gen = BPGen.create('demons', child_gens[1])
        self._costs_gen = BPGen.create('demons_costs', child_gens[2])

    def __call__(self, mode: int = 0) -> Task:
        return self.gen(mode)

    def gen(self, mode: int = 0, amount:int=1) -> Task|List[Task]:
        """
        Main generator function.

        Generation modes:
            - ``-3``: Varying complication (easy)
            - ``-2``: Varying complication (medium)
            - ``-1``: Varying complication (hard)
            - ``0``:  Default, high deviation for varying complication
            - ``1``:  Set size and lengths, similar to base game BP (easy)
            - ``2``:  Set size and lengths (medium)
            - ``3``:  Set size and lengths (hard)
            - ``4``:  Set size and lengths (very hard)

        :param mode: generation mode (see list above)
        :param amount: Optional number of tasks to generate. If ``amount>1``, returns list; if ``1`` (default), returns single instance
        :return: ``Task`` instance or ``list`` of ``Task`` instances
        """

        if mode not in TASK_MODES:
            raise ValueError(f"invalid mode must be on of {TASK_MODES}")
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("amount must be positive integer")

        size = 0
        demon_specs = zeros(10, dtype=int8)
        buffer_length = 0
        mat_mode = 0

        # fixed modes (1 - 4)
        if mode in {1, 2, 3, 4}:
            if mode == 1:
                size = 6
                demon_specs[2] = 1  # 1 leng 2
                demon_specs[3] = 2  # 2 leng 3
            elif mode == 2:
                size = 8
                demon_specs[2] = 2  # 2 leng 2
                demon_specs[3] = 1  # 1 leng 3
                demon_specs[4] = 1  # 1 leng 4
            elif mode == 3:
                size = 8
                # demon_specs[2] = 1  # 1 length 2
                demon_specs[3] = 2  # 2 length 3
                demon_specs[4] = 1  # 2 length 4
                demon_specs[5] = 1  # 1 length 5
            elif mode == 4:
                size = 9
                # demon_specs[2] =   # 2 leng 2
                # demon_specs[3] = 1  # 1 leng 3
                demon_specs[4] = 3  # 2 leng 4
                demon_specs[5] = 2  # 5 leng 5
                demon_specs[6] = 2  # 1 leng 6
                demon_specs[7] = 1  # 1 leng 7

                mat_mode = 1

            active_lengths = [l for l in range(2, 10) if demon_specs[l] > 0]
            max_demon_length = max(active_lengths) if active_lengths else 0

            if mode == 1:
                buffer_length = max_demon_length + 2
            elif mode == 2:
                buffer_length = max_demon_length + 1
            elif mode == 3:
                buffer_length = max_demon_length
            elif mode == 4:
                buffer_length = max(max_demon_length - 1, 1)

        # varying modes (-3 - 0)
        else:
            available_lengths = []
            num_selected = 0

            if mode == -3:
                size = self._rng.integers(4, 7)  # 4-6 inclusive
                available_lengths = list(range(2, 4))  # lengths 2-3
                num_selected = self._rng.integers(1, 3)  # 1-2 lengths
            elif mode == -2:
                size = self._rng.integers(6, 9)  # 6-8 inclusive
                available_lengths = list(range(3, 6))  # lengths 3-5
                num_selected = self._rng.integers(2, 4)  # 2-3 lengths
            elif mode == -1:
                size = self._rng.integers(8, 11)  # 8-10 inclusive
                available_lengths = list(range(5, min(9, int(size)) + 1))  # lengths 5+
                num_selected = self._rng.integers(3, 5)  # 3-4 lengths
            elif mode == 0:
                size = self._rng.integers(4, 11)  # 4-10 inclusive
                available_lengths = list(range(2, min(9, int(size)) + 1))
                num_selected = size // 2
                if size > 6:
                    mat_mode = self._rng.choice((0, 0, 1, 1, 1))
                if size > 8:
                    mat_mode = self._rng.choice((0, 0, 1, 1, 1, 2))
                else:
                    mat_mode = 0

            # demons specs
            demon_specs = zeros(10, dtype=int8)
            selected_lengths = self._rng.choice(available_lengths,
                                                size=min(num_selected, len(available_lengths)),
                                                replace=False)

            for length in selected_lengths:
                max_demons = max(1, size // length)
                num_demons = 0
                if mode == -3:
                    num_demons = self._rng.integers(1, min(3, max_demons) + 1)
                elif mode == -2:
                    num_demons = self._rng.integers(2, min(4, max_demons) + 2)
                elif mode == -1:
                    min_demons = min(3, max_demons)
                    num_demons = self._rng.integers(min_demons, max_demons + 1)
                elif mode == 0:
                    num_demons = self._rng.integers(1, max_demons + 1)
                demon_specs[length] = num_demons

            active_lengths = [l for l in range(2, 10) if demon_specs[l] > 0]
            max_demon_length = max(active_lengths) if active_lengths else 0

            if mode == -3:
                buffer_length = max_demon_length + 2
            elif mode == -2:
                buffer_length = max_demon_length + 1
            elif mode == -1:
                buffer_length = max_demon_length
            elif mode == 0:
                buffer_length = max_demon_length + self._rng.integers(0, 3)  # noqa (numpy int vs python int)

        # Generate game data
        # matrix = self.mat_gen(size, mat_mode)
        # demons = self.demons_gen(matrix, demon_specs)
        # costs = self.costs_gen(demons)
        # return Task(matrix, demons, costs, buffer_length)

        tasks = [
            Task(
                matrix := self._mat_gen(size, mat_mode),
                demons := self._demons_gen(matrix, demon_specs),
                self._costs_gen(demons),
                buffer_length,
            ) for _ in range(amount)
        ]
        if amount == 1:
            return tasks[0]
        else:
            return tasks


    def gen_manual(self, matrix_size:int, demons_specs:dict|ndarray, buffer_size:int, matrix_mode:int=0, amount:int=1) -> Task|List[Task]:
        """
        Manually generates Task instance based on specified parameters.

        Does not verify inputs - delegates validation to generators.
        Returns single Task if `amount == 1` (default), or list of Tasks if `amount > 1`.

        Matrix mode description & recommended sizes:
           - ``0``: Standard minigame with base game symbols (equal chances) [3-6]
           - ``1``: DLC minigame (mostly DLC symbols, with small changes for base symbols) [5-8]
           - ``2``: Full symbol set with decreasing appearance chances [6-12]
           - ``3``: Full symbol set with equal chances (>10)
           - ``4``: *"Good luck xD"* - Not recommended

        :param matrix_size: Size of matrix to generate.
        :param demons_specs: Demon configuration: dict {length: count} or array[i] = count of length i.
        :param buffer_size: Buffer capacity for the generated task.
        :param matrix_mode: Mode selection (see matrix mode description above).
        :param amount: Optional number of tasks to generate.
        :return: `Task`` instance or ``list`` of ``Task`` instances
        """
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("amount must be positive integer")
        tasks = [
            Task(
                matrix := self._mat_gen(matrix_size, matrix_mode),
                demons := self._demons_gen(matrix, demons_specs),
                self._costs_gen(demons),
                buffer_size,
            ) for _ in range(amount)
        ]
        if amount == 1:
            return tasks[0]
        else:
            return tasks
