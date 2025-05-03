from dataclasses import dataclass
from numpy import ndarray, integer, int8, zeros
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional


@dataclass
class Solution:
    """
    Represents breach protocol solution, only non-optional parameter is path, rest may be reconstructed with .fill_solution()

    path: ndarray(shape=(nx2), dtype=int) |
    buffer_sequence: Optional[ndarray(dtype=int)] |
    active_demons: Optional[ndarray(dtype=bool)] |
    total_points: Optional[int] |
    """
    path: ndarray
    buffer_sequence: Optional[ndarray] = None
    active_demons: Optional[ndarray] = None
    total_points: Optional[int|integer] = None

    def __post_init__(self):
        self._validate_inputs()

    def _validate_inputs(self):
        path = self.path
        buffer_sequence = self.buffer_sequence
        active_demons = self.active_demons
        total_points = self.total_points

        if not isinstance(path, ndarray):
            raise ValueError("path must be ndarray")
        if path.size == 0:
            pass    # TODO: logs & warnings (or maybe counting)
        else:
            if path.ndim != 2 or path.shape[1] != 2:
                raise ValueError("non-empty path must be 2d array with shape (n, 2)")
        if buffer_sequence is not None:
            if not isinstance(buffer_sequence, ndarray) or buffer_sequence.ndim != 1:
                raise ValueError("buffer_sequence must be 1d ndarray")
        if active_demons is not None:
            if not isinstance(active_demons, ndarray) or active_demons.ndim != 1:
                raise ValueError("active_demons must be 1d ndarray.")
        if total_points is not None:
            if not isinstance(total_points, (int|integer)):
                raise ValueError("total_points must be integer")


    def fill_solution(self, from_task):
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












