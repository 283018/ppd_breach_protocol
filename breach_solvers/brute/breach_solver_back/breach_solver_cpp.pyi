from numpy import ndarray, int64

def solve_breach(

    matrix_np: ndarray,
    demons_array_np: ndarray,
    demons_lengths_np: ndarray,
    demons_costs_np: ndarray,
    buffer_size: int,
    n: int,
    max_score: int64,
    num_demons: int,
    init_stack_size: int,
    enable_pruning: bool
) -> ndarray:
    """
    Solve breach protocol task using c++ back, runs in parallel.

    Require pre-calculating parameters and padding demons array with -1 to ensure same dimensions and estimating stack size

    :param matrix_np: 2d ndarray, dtype=np.int8
    :param demons_array_np: 2d ndarray, dtype=np.int8
    :param demons_lengths_np: ndarray, dtype=np.int8
    :param demons_costs_np: ndarray, dtype=np.int8
    :param buffer_size: in
    :param n: int
    :param max_score: int64
    :param num_demons: int
    :param init_stack_size:
    :param enable_pruning:
    :return:
    """
    ...