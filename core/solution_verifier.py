from numpy import ndarray, asarray
from core.datastructures import Task, Solution


__all__ = ['validate_solution', 'verify_solution']


# ########################################################################## #
#                                                                            #
#   whole this module is deprecated, all validation now done in Solution     #
#                                                                            #
# ########################################################################## #


def validate_solution(matrix, path):
    """
    Validates path on being valid according to movement rules
    :param matrix: 2d ndarray
    :param path: 2xn ndarray
    :return: Trie if path satisfies rules
    """
    n_rows, n_cols = matrix.shape

    if isinstance(path, ndarray):
        path = tuple(map(tuple, path))
    visited = set()

    # validate all cells in path
    for cell in path:
        row, col = cell
        # check does path is in matrix bounds
        if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
            print(f"Cell {cell} out of bounds")
            return False

        # check duplicates
        if cell in visited:
            print(f"Duplicated cell: {cell}")
            return False
        visited.add(cell)

    # validate movement rules
    for step, (current_row, current_col) in enumerate(path):
        if step == 0:
            # first move must be in first row
            if current_row != 0:
                print(f"First move in row {current_row}")
                return False
        else:
            prev_row, prev_col = path[step - 1]
            if step % 2 == 1:
                if current_col != prev_col:
                    print(f"Wrong column alternation")
                    return False
            else:
                if current_row != prev_row:
                    print(f"Wrong row alternation")
                    return False

    return True


def verify_solution(task:Task, solution:Solution):

    matrix = task.matrix
    buffer_size = task.buffer_size
    demons = task.demons
    demons_costs = task.demons_costs

    path = solution.path
    buffer_sequence = solution.buffer_sequence
    active_demons = solution.active_demons
    total_points = solution.total_points


    # first validate path (column/row, starting in first row)
    if not validate_solution(matrix, path):
        return False

    # check buffer leng if given
    if buffer_size is not None and len(path) > buffer_size:
        print(f"Actual buffer sequence longer then given buffer length")
        return False


    # check buffer sequence to path if given, generate one if not provided
    if buffer_sequence is not None:
        generated_sequence = [int(matrix[row, col]) for row, col in path]
        generated_sequence += [0] * (buffer_size - len(generated_sequence))   # adding 0s for unused buffer
        buffer_list = asarray(buffer_sequence).tolist()     # a little redundant but for now keep both types
        if buffer_list != generated_sequence:
            print(f"Path does not match buffer sequence")
            return False
    else:
        buffer_list = [matrix[cell] for cell in path]


    # verify active_demons if provided
    if active_demons is not None:

        # check if length of binary array equals to length of demons
        if demons is None:
            raise ValueError("active_demons provided without demons")
        if len(active_demons) != len(demons):
            raise ValueError(f"Demon list length does not match active_demons length")

        # check each demon's presence in buffer (given or generated from path)
        for i, demon in enumerate(demons):
            demon_list = list(demon)    # should be ok, since already checked dimensions
            demon_len = len(demon_list)
            found = False

            # check all possible starting positions in buffer
            for j in range(len(buffer_list) - demon_len + 1):
                if buffer_list[j:j + demon_len] == demon_list:
                    found = True
                    break

            if active_demons[i] != found:
                print(f"active_demons does not match found activated demons")
                return False

    return True



# if __name__ == '__main__':
#     from icecream import ic
#     import numpy as np
#
#     mat1 = np.arange(1, 17).reshape(4, 4)
#     ic(mat1)
#
#     ic('')
#     path1 = ((0, 0), (1, 0), (1, 1), (2, 1))    # valid
#     path2 = ((0, 0), (0, 1), (1, 1))     # invalid
#     ic(verify_solution(mat1, path1))
#     ic(verify_solution(mat1, path2))
#
#     ic('')
#     ic(verify_solution(mat1, path1, buffer_length=3))
#
#     ic('')
#     test_buffer = [mat1[cell] for cell in path1]
#     ic(verify_solution(mat1, path1, buffer_sequence=test_buffer))  # True
#     ic(verify_solution(mat1, path1, buffer_sequence=[1, 5, 7, 11]))  # False
#
#     ic('')
#     demons1 = (
#         np.array([1, 5, 6]),
#         [6, 10],
#         [7, 6, 10]
#     )
#     buffer1 = [mat1[cell] for cell in path1]
#     active1 = [True, True, True]
#     ic(verify_solution(mat1, path1, demons=demons1, buffer_sequence=buffer1, demons_activated=active1))
#
#     ic('')
#     ic(verify_solution(mat1, tuple()))
#
#     ic('')
#     repeated_path = ((0, 0), (1, 0), (1, 0))
#     ic(verify_solution(mat1, repeated_path))