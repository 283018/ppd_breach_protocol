from dataclasses import dataclass
from typing import Optional, Self
from warnings import warn

from numpy import array_equal, full_like, int8, integer, ndarray, zeros
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(slots=True)
class Solution:
    """
    Represents breach protocol solution, only non-optional parameter is path, rest may be reconstructed with .fill_solution()

    - path: ndarray(shape=(nx2), dtype=int)
    - buffer_sequence: Optional[ndarray(dtype=int)]
    - active_demons: Optional[ndarray(dtype=bool)]
    - total_points: Optional[int|integer]
    """

    path: Optional[ndarray] = None
    buffer_sequence: Optional[ndarray] = None
    active_demons: Optional[ndarray] = None
    total_points: Optional[int|integer] = None

    def __post_init__(self):
        path = self.path
        buffer_sequence = self.buffer_sequence
        active_demons = self.active_demons
        total_points = self.total_points

        msg = ""

        if path is None:
            msg += "Solution must have path, for no-path cases use NoSolution instead\n"
        if not isinstance(path, ndarray):
            msg += "path must be ndarray\n"
        else:
            if path.size == 0:
                msg += "Solution must have path, for no-path cases use NoSolution instead\n"
            if path.ndim != 2 or path.shape[1] != 2:
                msg += "non-empty path must be 2d array with shape (n, 2)\n"
        if buffer_sequence is not None and (not isinstance(buffer_sequence, ndarray) or buffer_sequence.ndim != 1):
            msg += "buffer_sequence must be 1d ndarray\n"
        if active_demons is not None and (not isinstance(active_demons, ndarray) or active_demons.ndim != 1):
            msg += "active_demons must be 1d ndarray.\n"
        if total_points is not None and not isinstance(total_points, (int|integer)):
            msg += "total_points must be integer\n"

        if msg != "":
            raise ValueError(msg)


    def fill_solution(self, from_task) -> Self:
        """
        Fills all optional parameters based on self.path and given Task instance.
        Does not validate path.
        :param from_task: Task instance.
        :return: self
        """
        if self.buffer_sequence is None:
            path = self.path
            buffer = zeros(from_task.buffer_size, dtype=int8)

            buffer[:path.shape[0]] = from_task.matrix[path[:, 0], path[:, 1]]

            self.buffer_sequence = buffer

        if self.active_demons is None:
            buffer = self.buffer_sequence[:self.path.shape[0]]  # sliced, since unused calls by definition cant contain demons
            demons = from_task.demons
            num_demons = len(demons)
            active_demons = zeros(num_demons, dtype='bool')
            buffer_len = buffer.shape[0]

            for i in range(num_demons):
                demon = demons[i]
                d_len = demon.shape[0]
                if d_len > buffer_len:
                    continue
                windows = sliding_window_view(buffer, window_shape=d_len)
                match = (windows == demon).all(axis=1).any()    # noqa (array treated as bool by IDE)
                active_demons[i] = match

            self.active_demons = active_demons

        if self.total_points is None:
            costs = from_task.demons_costs
            points = costs @ self.active_demons
            self.total_points = points

        return self

    def copy(self) -> Self:
        return Solution(
            path = self.path.copy(),
            buffer_sequence = self.buffer_sequence.copy() if self.buffer_sequence is not None else None,
            active_demons = self.active_demons.copy() if self.active_demons is not None else None,
            total_points = self.total_points if self.total_points is not None else None,
        )

    def __copy__(self) -> Self:
        return self.copy()

    def __deepcopy__(self, memo: dict) -> Self:
        return self.copy()

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented

        if (
                self.buffer_sequence is None
                or other.buffer_sequence is None
                or self.active_demons is None
                or other.active_demons is None
                or self.total_points is None
                or other.total_points is None
        ):
            warn("Comparison of unfilled Solutions")
            return False

        return (
            array_equal(self.path, other.path)
            and array_equal(self.buffer_sequence, other.buffer_sequence)
            and array_equal(self.active_demons, other.active_demons)
            and self.total_points == other.total_points
        )

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented

        if self.total_points != other.total_points:
            return self.total_points < other.total_points

        return len(self.path) < len(other.path)


@dataclass(slots=True)
class NoSolution(Solution):
    """
    Represents errored/impossible Solutions
    """

    path: ndarray|None = None
    reason: str = None

    def __post_init__(self):
        msg = ""
        if self.path is not None:
            msg += f"Creating NoSolution from non-empty path: \n{self.path}"
        if any(val is not None for val in (self.total_points, self.active_demons, self.buffer_sequence)):
            msg += ("Created NoSolution with non-empty values of (buffer_sequence, active_demons, total_points): \n"
                    f"{self.buffer_sequence, self.active_demons, self.total_points}")
        if msg != "":
            warn(msg)
        self.path = full_like(self.path, fill_value=-1)
        self.buffer_sequence = full_like(self.buffer_sequence, fill_value=0, dtype=int8)
        self.active_demons = full_like(self.active_demons, fill_value=False, dtype=bool)
        self.total_points = 0

    def __repr__(self) -> str:
        reason = self.reason or "Unknown"
        return f"NoSolution({reason=})"

    def fill_solution(self, _from_task=None) -> Self :
        warn("Unable to fill NoSolution")
        return self







