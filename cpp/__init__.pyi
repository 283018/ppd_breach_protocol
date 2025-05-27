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
    n_iterations: int,
    stagnant_limit: int,
) -> tuple[NDArray[int8], int|integer, int|integer, NDArray[int8]]:
    """
    Solve breach protocol task with ant colony optimization using C++ backend.

    Requires pre-calculated/generated parameters:
        - ``flat_demons``
        - ``n``
        - ``num_demons``
        - ``demons_lengths``
        - ``max_demon_len``
        - ``heuristic``
        - ``seed``

    .. note::
        - ``seed``: Must be in range (``UINT_MIN``, ``UINT_MAX``) = (``0``, ``2³²-1``)
        - ``n_iterations`` If ≤ 0, set to `INT_MAX` (2³¹-1)
        - ``stagnant_limit`` modes:
              - ``==0``: Calculated from task
              - ``<0``: No stagnation control
              - ``>0``: Hard limit

    .. warning::
        **WARNING**: Using ``stagnant_limit < 0 < n_iterations`` may cause infinite loops.

    :param matrix: ``Task.matrix`` (nxn)
    :param flat_demons: 2d array of demons, padded with ``-1`` demons array
    :param demons_costs: ``Task.demons_costs``
    :param buffer_size: ``Task.buffer_size``
    :param n: side length of ``matrix``
    :param num_demons: amount of demons
    :param demons_lengths: 1d array with original demons lengths
    :param max_demon_len: length of the longest demon
    :param heuristic: 1d flattened array of ``Task.matrix.size`` with attractiveness of cells in matrix
        based of ``Task.demons`` symbols
    :param alpha: metaheuristic hiperparametr
    :param beta: metaheuristic hiperparametr
    :param evaporation: metaheuristic hiperparametr
    :param q: metaheuristic hiperparametr
    :param seed: RNG seed
    :param n_ants: number of ants per iteration
    :param n_iterations: maximum number of iterations
    :param stagnant_limit: amount of allowed stagnant iterations
    :return: tuple: (``path`` as 2D ndarray (nx2), ``cost``, ``path_length``, ``buffer_sequence`` as 1d ndarray)
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
