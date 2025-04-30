from numpy import ndarray
from .breach_translator import map_breach


__all__ = ['aligned_print', 'mat_print', 'solution_print']

SUPERSCRIPT_MAP = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


def aligned_print(arraylike, *line_suffixes) -> None:
    """
    Prints 2d arraylike with aligned cells in console.
    Allow both numbers and translated to hex cells.
    Allows multiple optional suffixes to be appended to each row.
    Aligns both cells and suffixes.
    Does not verify dimensions.

    :param arraylike: 2d nested list: List[List[ int|str ]] or 2d numpy.ndarray
    :param line_suffixes: optional, each is 1d array-like of same length as given arraylike with any additional information
    """
    array_len = len(arraylike)
    for suffix in line_suffixes:
        if len(suffix) != array_len:
            raise ValueError("All line_suffixes must have same length as arraylike")

    max_len = max(len(str(cell)) for row in arraylike for cell in row)

    max_num_columns = max(len(row) for row in arraylike)
    row_width = max_len * max_num_columns + (max_num_columns - 1) * 2

    suffix_widths = []
    for suffix in line_suffixes:
        max_suffix_len = max(len(str(value)) for value in suffix)
        suffix_widths.append(max_suffix_len)

    for i, row in enumerate(arraylike):
        row_str = "  ".join(str(cell).rjust(max_len) for cell in row)

        padded_row_str = row_str.ljust(row_width + 4)

        if line_suffixes:
            suffix_parts = []
            for j, suffix in enumerate(line_suffixes):
                suffix_value = str(suffix[i])
                suffix_parts.append(suffix_value.rjust(suffix_widths[j]))
            suffix_str = "    ".join(suffix_parts)
            print(f"{padded_row_str}| {suffix_str}")
        else:
            print(padded_row_str)


def mat_print(matrix, path=None) -> None:
    """
    Prints 2d arraylike with aligned cells in console.
    Optionally with path given by coordinates (in superscript).

    Does not verify dimensions, can be used for iterables with demons sequences.
    Does not verify coordinates, nor correctness of solution.
    Allow both numbers and translated to hex cells.

    :param matrix: 2d arraylike: nested list/tuple, numpy array
    :param path: iterable of (row, col) tuples marking the path in order
    """
    if path is not None:

        if isinstance(path, ndarray):
            path = tuple(map(tuple, path))

        index_map = {coord: str(idx+1).translate(SUPERSCRIPT_MAP)
                     for idx, coord in enumerate(path)}

        # apply step index to cells if presented in path arg
        matrix_with_path = [
            [str(cell) + index_map.get((i, j), ' ') for j, cell in enumerate(row)]
            for i, row in enumerate(matrix)
        ]

        aligned_print(matrix_with_path)
    else:
        aligned_print(matrix)


# TODO: just for task, just for solution
def solution_print(task, solution, translate:bool=False) -> None:
    """
    Print given Task and Solution in console.
    :param task: core.datastructures.task.Task
    :param solution: core.datastructures.solution.Solution
    :param translate: if True prints translated matrix and demons, do not alter original instances of task and solution
    """
    # unpacking Task
    matrix = task.matrix
    demons = task.demons
    buffer_size = task.buffer_size
    demons_costs = task.demons_costs

    # unpacking Solution
    path = solution.path
    buffer_sequence = solution.buffer_sequence
    active_demons = solution.active_demons
    total_points = solution.total_points

    # counting
    demons_costs_sum = sum(demons_costs)
    used_buffer = sum(1 if int(i) != 0 else 0 for i in buffer_sequence)
    demons_amo = len(demons)
    demons_active_amo = active_demons.sum()

    if translate:
        matrix = map_breach(matrix)
        demons = map_breach(demons)
        buffer_sequence = map_breach([buffer_sequence])[0]
        active_demons = ['✓' if value else '⨯' for value in active_demons]


    # string generating
    max_len = max(len(str(cell)) for row in demons for cell in row)
    max_num_columns = max(len(row) for row in demons)
    row_width = max_len * max_num_columns + (max_num_columns - 1) * 2
    demons_active_str = f"{demons_active_amo}/{demons_amo}".ljust(row_width)
    points_str = f"{total_points}/{demons_costs_sum}"
    used_buffer_str = f"{used_buffer}/{buffer_size}"

    print()
    print("Matrix: ")
    mat_print(matrix, path=path)
    print("Demons: ")
    aligned_print(demons, demons_costs, active_demons)
    aligned_print([[demons_active_str]], [points_str])
    print("Buffer: ")
    aligned_print([buffer_sequence], [used_buffer_str])
    print()



# if __name__ == "__main__":
#     import numpy as np
#     from icecream import ic
#     # ic(SUPERSCRIPT_MAP)
#     n1 = 4
#     mat1 = np.array([i+1 for i in range(n1*n1)]).reshape((n1, n1))
#     print("#"*10)
#     mat_print(mat1)
#     print("#"*10)
#     demons1 = [np.array([1, 2, 3, 4]), np.array([7, 8, 9]), np.array([7, 8, 12]), np.array([4, 12, 6, 1]), np.array([7, 8, 9])]
#     mat_print(demons1)
#     print("#" * 10)
#     # mat2 = [['12', '2', '3'], ['4', '12', '652'], ['7', '8', '9']]
#     # mat_print(mat2)
#
#     mat = [
#         [0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]
#     ]
#
#     coords = [(0, 0), (1, 1), (2, 2), (35, 11)]
#     mat_print(mat, coords)
#     print()
#     aligned_print(demons1)
#     print()
#     costs1 = [2, 4, 5, 12, 2]
#     active1 = [True, False, True, True, True]
#     aligned_print(demons1, costs1, active1)