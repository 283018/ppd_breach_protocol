# cpp/__init__.py
# a little clumsy, but since imports made with importlib on runtime, typecheckers says "nuh-uh"

from numpy import int8, float64, int32, int64, integer
from numpy.typing import NDArray


def ant_colony(
    matrix: NDArray[int8],
    flat_demons: NDArray[int8],
    demons_costs: NDArray[int8],
    buffer_size: int,
    n: int,
    num_demons: int,
    demons_lengths: NDArray[int32],
    max_demon_len: int,
    heuristic: NDArray[float64],
    alpha: float,
    beta: float,
    evaporation: float,
    q: float,
    seed: int,
    n_ants: int,
    n_iterations: int
) -> tuple[NDArray[int8], int|integer, int|integer, NDArray[int8]]:
    """
    Solve breach protocol task with ant colony optimization, using c++ back.

    Require pre-calculating or generating parameters:
        - flat_demons
        - n
        - num_demons
        - demons_lengths
        - max_demon_len
        - heuristic
        - seed

    :param matrix: Task.matrix
    :param flat_demons: 2d ndarray, padded with '-1' demons array
    :param demons_costs: Task.demons_costs
    :param buffer_size: Task.buffer_size
    :param n: side of Task.matrix(nxn)
    :param num_demons: amount of demons
    :param demons_lengths: 1d array with original demons lengths
    :param max_demon_len: length of longest demon
    :param heuristic: 1d ndarray flattened array of Task.matrix.size with attractiveness of cells in matrix
        based on symbol frequent in Task.demons sequences
    :param alpha:
    :param beta:
    :param evaporation:
    :param q:
    :param seed: seed in range (0, 2³²-1)
    :param n_ants:
    :param n_iterations:
    :return: tuple of path as 2d ndarray (nx2), cost, length of path, buffer sequence as 1d ndarray
    """
    ...



def brute_force(
    matrix_np: NDArray[int8],
    demons_array_np: NDArray[int8],
    demons_lengths_np: NDArray[int8],
    demons_costs_np: NDArray[int8],
    buffer_size: int,
    n: int,
    max_score: int64,
    num_demons: int,
    init_stack_size: int,
    enable_pruning: bool
) -> NDArray[int32]:
    """
    Solve breach protocol task with brute force, using c++ back, runs in parallel.

    Require pre-calculating parameters:
        - demons_array_np
        - demons_lengths_np
        - n
        - max_score
        - num_demons
        - init_stack_size

    :param matrix_np: Task.matrix
    :param demons_array_np: 2d ndarray, padded with '-1' demons array
    :param demons_lengths_np: 1d array with original demons lengths
    :param demons_costs_np: Task.demons_costs
    :param buffer_size: Task.buffer_size
    :param n: side of Task.matrix(nxn)
    :param max_score: numpu.int64, sum of all demons costs (max possible score)
    :param num_demons: amount of demons
    :param init_stack_size: estimated max stack size
    :param enable_pruning:
    :return: path as 2d ndarray (nx2) dtype=numpy.int32
    """
    ...
