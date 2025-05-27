from core.datastructures import Task, Solution, NoSolution
from .breach_translator import map_breach

from numpy import ndarray
from multipledispatch import dispatch

from collections.abc import Sequence


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


#* TODO: encapsulation
#? TODO: add .__str__

@dispatch(Task, (Solution, NoSolution), bool)
def bprint(task:Task, solution:Solution|NoSolution, translate:bool=False) -> None:
    """
    Print given single Task and single Solution in console.
    """
    _full_print(task, solution, translate)
    print()
@dispatch(Task, (Solution, NoSolution))
def bprint(task:Task, solution:Solution|NoSolution) -> None:
    bprint(task, solution, False)

@dispatch(Task, (Solution, NoSolution), float, bool)
def bprint(task:Task, solution:Solution|NoSolution, time, translate:bool=False) -> None:
    """
    Print given single Task, single Solution and time in console.
    """
    _full_print(task, solution, translate)
    print(f"{time:.4}\n")
    print()
@dispatch(Task, (Solution, NoSolution), float)
def bprint(task:Task, solution:Solution|NoSolution, time) -> None:
    bprint(task, solution, time, False)

@dispatch(Task, (list, tuple), bool)
def bprint(task:Task, solutions:Sequence[Solution|NoSolution], translate:bool=False) -> None:
    """
    Prints single Tasks with multiple given Solutions in console.
    """
    if not all(isinstance(sol, (Solution, NoSolution)) for sol in solutions):
        raise TypeError(f"all solution must be of type {Solution} or {NoSolution}")
    for sol in solutions:
        _full_print(task, sol, translate)
        print()
@dispatch(Task, (list, tuple))
def bprint(task:Task, solutions:Sequence[Solution|NoSolution]) -> None:
    bprint(task, solutions, False)

@dispatch((list, tuple), (list, tuple), bool)
def bprint(tasks:Sequence[Task], solutions:Sequence[Solution], translate:bool=False) -> None:
    """
    Prints multiple Tasks with multiple given Solutions in console.
    """
    if len(tasks) != len(solutions):
        raise ValueError(f"Lengths are not equal: {len(tasks)}, {len(solutions)}")
    for task, solution in zip(tasks, solutions):
        if not isinstance(task, Task) or not isinstance(solution, (Solution, NoSolution)):
            raise TypeError(f"(task, solution) pair must be of types ({Task}, {Solution}|{NoSolution}), not ({type(task)}, {type(solution)})")
        _full_print(task, solution, translate)
        print()
@dispatch((list, tuple), (list, tuple))
def bprint(tasks:Sequence[Task], solutions:Sequence[Solution]) -> None:
    bprint(tasks, solutions, False)


@dispatch(Task, bool)
def bprint(task:Task, translate:bool=False) -> None:
    """
    Prints Single Task in console.
    """
    _task_print(task, translate)
@dispatch(Task)
def bprint(task:Task) -> None:
    bprint(task, False)

@dispatch((list, tuple), bool)
def bprint(tasks:Sequence[Task], translate:bool=False) -> None:
    """
    Prints multiple Task in console.
    """
    for task in tasks:
        if not isinstance(task, Task):
            raise TypeError(f"task must be of type {Task}, not {type(task)}")
        _task_print(task, translate)
        print()
@dispatch((list, tuple))
def bprint(tasks:Sequence[Task]) -> None:
    bprint(tasks, False)

@dispatch(Task, (Solution, NoSolution), float, bool)
def bprint(task:Task, solution:Solution|NoSolution, time, translate:bool=False) -> None:
    """
    Prints single Tasks with single Solutions and time in console.
    """
    _full_print(task, solution, translate)
    print(f"{time:.4}\n")
@dispatch(Task, (Solution, NoSolution), float)
def bprint(task:Task, solution:Solution|NoSolution, time) -> None:
    bprint(task, solution, time, False)


@dispatch(Task, (list, tuple), (list, tuple), bool)
def bprint(task:Task, solutions:Sequence[Solution|NoSolution], times:Sequence[float], translate:bool=False) -> None:
    """
    Prints single Tasks with multiple given Solutions and times in console.
    """
    if len(solutions) != len(times):
        raise ValueError(f"Lengths are not equal: {len(solutions)} ,{len(times)}")
    if (not all(isinstance(sol, (Solution, NoSolution)) for sol in solutions) or
            not all(isinstance(time, float) for time in times)):
        raise TypeError(f"all solution must be of type {Solution} or {NoSolution}, and all times must be float")
    for sol in solutions:
        _full_print(task, sol, translate)
        print()
    for time in times:
        print(f"{time:.4}")
    print()
@dispatch(Task, (list, tuple), (list, tuple))
def bprint(task:Task, solutions:Sequence[Solution|NoSolution], times:Sequence[float]) -> None:
    bprint(task, solutions, times, False)


@dispatch((list, tuple), (list, tuple), (list, tuple), bool)
def bprint(tasks:Sequence[Task], solutions:Sequence[Solution|NoSolution], times:Sequence[float]) -> None:
    bprint(tasks, solutions, times, False)
@dispatch((list, tuple), (list, tuple), (list, tuple), bool)
def bprint(tasks:Sequence[Task], solutions:Sequence[Solution|NoSolution], times:Sequence[float], translate:bool=False) -> None:     # noqa
    """
    Multiple dispatched function for printing Tasks and Solution in console.

    .. note::
        - Multipledispatch does not support keyword arguments.
        - `Sequence[]` typehint is used for simplicity; accepts only `List` or `Tuple` (or subclasses) with valid structs.
        - `translate` is an optional argument in all cases.


    Accept different combinations of arguments:
        - ``Task``, ``translate``: Single task
        - ``Sequence[Task]``, ``translate``: Multiple tasks
        - ``Task``, ``Solution|NoSolution``, ``translate``: single Task and single Solution
        - ``Task``, ``Sequence[Solution|NoSolution]``, ``translate``: single Task with multiple Solutions
        - ``Sequence[Task]``, ``Sequence[Solution|NoSolution]``, ``translate``: multiple Tasks with multiple corresponding Solutions
        - ``Task``, ``Solution|NoSolution``, ``time``, ``translate``: Task, Solution, and solving time
        - ``Sequence[Task]``, ``Sequence[Solution|NoSolution]``, ``Sequence[float]``, ``translate``: Tasks with solutions and times

    :param translate: Optional translation flag for int-to-breach-hex mapping.
    """
    if len(tasks) != len(solutions) != len(times):
        raise ValueError(f"Lengths are not equal: {len(tasks)}, {len(solutions)} ,{len(times)}")
    for task, solution, time in zip(tasks, solutions, times):
        if not isinstance(task, Task) or not isinstance(solution, (Solution, NoSolution)):
            raise TypeError(f"(task, solution) pair must be of types ({Task}, {Solution}|{NoSolution}), not ({type(task)}, {type(solution)})")
        _full_print(task, solution, translate)
        print(f"Time: {time:.4}\n")



def _task_print(task: Task, translate: bool = False) -> None:
    matrix = task.matrix
    demons = task.demons
    buffer_size = task.buffer_size
    demons_costs = task.demons_costs

    if translate:
        matrix = map_breach(matrix)
        demons = map_breach(demons)

    print("Matrix: ")
    mat_print(matrix)
    print("Demons: ")
    aligned_print(demons, demons_costs)
    print(f"Buffer: {buffer_size}")


def _full_print(task, solution, translate):
    if type(solution).__name__ == 'NoSolution':
        is_no_sol = True
    elif type(solution).__name__ == 'Solution':
        is_no_sol = False
    else:
        raise NotImplementedError("Printing error, unknown solution type")

    # unpacking Task
    matrix = task.matrix
    demons = task.demons
    buffer_size = task.buffer_size
    demons_costs = task.demons_costs


    if not is_no_sol:
        # unpacking Solution
        path = solution.path
        buffer_sequence = solution.buffer_sequence
        active_demons = solution.active_demons
        total_points = solution.total_points

        # counting
        demons_costs_sum = sum(int(d) for d in demons_costs)
        used_buffer = sum(1 if int(i) != 0 else 0 for i in buffer_sequence)
        demons_amo = len(demons)
        demons_active_amo = active_demons.sum()

    else:
        path = None
        buffer_sequence = [0 for _ in range(buffer_size)]
        demons_amo = len(demons)
        active_demons = [False] * demons_amo
        total_points = 0

        demons_costs_sum = sum(int(d) for d in demons_costs)
        used_buffer = 0
        demons_active_amo = 0

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

    print("Matrix: ")
    mat_print(matrix, path=path)
    print("Demons: ")
    aligned_print(demons, demons_costs, active_demons)
    aligned_print([[demons_active_str]], [points_str])
    print("Buffer: ")
    aligned_print([buffer_sequence], [used_buffer_str])
    if is_no_sol:
        print(f"Reason:\n{solution.reason}")
