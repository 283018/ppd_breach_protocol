from numpy import ndarray, int8, zeros, integer
from numpy.random import Generator, SeedSequence, default_rng, PCG64

from .breach_generator import BPGen
from core import Task


TASK_MODES = set(range(-3, 5))

class TaskFactory:
    """
    Factory for Task, callable object.

    For __call__ supports mode parameter:
        -3: varying complication of tasks (easy)

        -2: varying complication of tasks (medium)

        -1: varying complication of tasks (hard)

        0: default, high deviation for varying complication of tasks

        1: setted size and lengths, similar to base game bp (easy)

        2: setted size and lengths (medium)

        3: setted size and lengths (hard)

        4: setted size and lengths (very hard)

    :param seed: optional, seed for main rng, rest derived from it.
    """

    def __init__(self, seed: int|integer=None):
        self.rng = default_rng(seed)

        main_ss_seed = self.rng.integers(0, 2**32)
        main_ss = SeedSequence(main_ss_seed)
        child_gens_seeds = main_ss.spawn(3)
        child_gens = [Generator(PCG64(seed)) for seed in child_gens_seeds]

        # pain...
        self.mat_gen = BPGen.create('matrix', child_gens[0])
        self.demons_gen = BPGen.create('demons', child_gens[1])
        self.costs_gen = BPGen.create('demons_costs', child_gens[2])

    def __call__(self, mode: int = 0) -> Task:
        """
        Main generator function.
        :param mode: generation mode
                    -3: varying complication of tasks (easy)
                    -2: varying complication of tasks (medium)
                    -1: varying complication of tasks (hard)
                    0: default, high deviation for varying complication of tasks
                    1: setted size and lengths, similar to base game bp (easy)
                    2: setted size and lengths (medium)
                    3: setted size and lengths (hard)
                    4: setted size and lengths (very hard)
        :return: Task instance
        :rtype: Task
        """

        if mode not in TASK_MODES:
            raise ValueError(f"invalid mode must be on of {TASK_MODES}")

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
                size = self.rng.integers(4, 7)  # 4-6 inclusive
                available_lengths = list(range(2, 4))  # lengths 2-3
                num_selected = self.rng.integers(1, 3)  # 1-2 lengths
            elif mode == -2:
                size = self.rng.integers(6, 9)  # 6-8 inclusive
                available_lengths = list(range(3, 6))  # lengths 3-5
                num_selected = self.rng.integers(2, 4)  # 2-3 lengths
            elif mode == -1:
                size = self.rng.integers(8, 11)  # 8-10 inclusive
                available_lengths = list(range(5, min(9, int(size)) + 1))  # lengths 5+
                num_selected = self.rng.integers(3, 5)  # 3-4 lengths
            elif mode == 0:
                size = self.rng.integers(4, 11)  # 4-10 inclusive
                available_lengths = list(range(2, min(9, int(size)) + 1))
                num_selected = size // 2
                if size > 6:
                    mat_mode = self.rng.choice((0, 0, 1, 1, 1))
                if size > 8:
                    mat_mode = self.rng.choice((0, 0, 1, 1, 1, 2))
                else:
                    mat_mode = 0

            # demons specs
            demon_specs = zeros(10, dtype=int8)
            selected_lengths = self.rng.choice(available_lengths,
                                               size=min(num_selected, len(available_lengths)),
                                               replace=False)

            for length in selected_lengths:
                max_demons = max(1, size // length)
                num_demons = 0
                if mode == -3:
                    num_demons = self.rng.integers(1, min(3, max_demons) + 1)
                elif mode == -2:
                    num_demons = self.rng.integers(2, min(4, max_demons) + 2)
                elif mode == -1:
                    min_demons = min(3, max_demons)
                    num_demons = self.rng.integers(min_demons, max_demons + 1)
                elif mode == 0:
                    num_demons = self.rng.integers(1, max_demons + 1)
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
                buffer_length = max_demon_length + self.rng.integers(0, 3)  # noqa (numpy int vs python int)

        # Generate game data
        matrix = self.mat_gen(size, mat_mode)
        demons = self.demons_gen(matrix, demon_specs)
        costs = self.costs_gen(demons)

        return Task(matrix, demons, costs, buffer_length)


    def gen_manual(self, matrix_size:int, demons_specs:dict|ndarray, buffer_size:int, matrix_mode:int=0) -> Task:
        """
        Manually generates Task according to given parameters.

        Does not verify inputs, delegate that to generators.

        Matrix mode description & recommended size:
            0: simulates standard minigame, uses only base game symbols with equal chances; [3-6]

            1: simulates dlc minigame, uses mostly dlc symbols, with small chances for base game symbols; [5-8]

            2: uses full set (base + dlc), slowly decrease chances of appearing for each new symbol, add step for dlc; [6-12]

            3: uses full set with equal chances for all; (>10)

            4:"good luck xD"; not recommended

        :param matrix_size:
        :param demons_specs: dict(length:count) | ndarray[i] = count of len i
        :param buffer_size:
        :param matrix_mode: (copied from GeneratorMatrix)
        :return: Task instance
        """
        matrix = self.mat_gen(matrix_size, matrix_mode)
        demons = self.demons_gen(matrix, demons_specs)
        costs = self.costs_gen(demons)
        return Task(matrix, demons, costs, buffer_size)






# if __name__ == "__main__":
#     t = TaskFactory(123123)
#     t.gen_manual()
#     from breach_solvers import SolverGurobi
#     from core import solution_print
#     from time import time, perf_counter
#
#     factory = TaskFactory(42)
#     gb_solver = SolverGurobi()
#
#     task_gen_times = []
#     solve_times = []
#     for i in range(10):
#         start_task_gen = perf_counter()
#         task1 = factory()
#         end_task_gen = perf_counter()
#
#         task_gen_time = end_task_gen - start_task_gen
#         task_gen_times.append(task_gen_time)
#
#         start_solve = perf_counter()
#         sol1 = gb_solver.solve(task1)
#         end_solve = perf_counter()
#
#         solve_time = end_solve - start_solve
#         solve_times.append(solve_time)
#
#         # solution_print(task1, sol1)
#         # print('\n'*3)
#
#     print("gen_times:")
#     print([f"{t:.6f}" for t in task_gen_times])
#     print("\nsol_times:")
#     print([f"{t:.6f}" for t in solve_times])
#
#     avg_task_gen_time = sum(task_gen_times) / len(task_gen_times)
#     avg_solve_time = sum(solve_times) / len(solve_times)
#
#     print(f"avg gen: {avg_task_gen_time:.6f} s")
#     print(f"avg sol: {avg_solve_time:.6f} s")