from dataclasses import dataclass
from typing import Tuple, Self
from numpy import ndarray, integer, issubdtype, array




@dataclass(slots=True)
class Task:
    """
    Represents single task for breach protocol minigame.

    - matrix: ndarray[ndarray(dtype=int8)]
    - demons: Tuple[ndarray(dtype=int8)]
    - demons_costs: ndarray(dtype=int8)
    - buffer_size: [int|int8]
    """
    matrix: ndarray
    demons: Tuple[ndarray, ...]
    demons_costs: ndarray
    buffer_size: int|integer

    def __post_init__(self):
        self._validate_inputs()

    def _validate_inputs(self):
        matrix = self.matrix
        buffer_size = self.buffer_size
        demons = self.demons
        demons_cost = self.demons_costs

        if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or
                not issubdtype(matrix.dtype, integer)):
            raise ValueError("matrix must be square 2d array")
        if not isinstance(buffer_size, (int, integer)) or buffer_size < 1:
            raise TypeError("buffer_size must be positive integer")
        d_amo = len(demons)
        if demons_cost.shape[0] != d_amo:
            raise ValueError("demons and d_costs must have the same length")
        for demon in demons:
            if not isinstance(demon, ndarray) or demon.ndim != 1:
                raise ValueError("Each demon must be a 1D numpy array")


    def copy(self) -> Self:
        return Task(
            matrix=self.matrix.copy(),
            demons=tuple(d.copy() for d in self.demons),
            demons_costs=self.demons_costs.copy(),
            buffer_size=self.buffer_size
        )

    def __copy__(self, memo: dict) -> Self:
        return self.copy()

    def __deepcopy__(self, memo: dict) -> Self:
        return self.copy()


DUMMY_TASK:Task = Task(
    matrix=array([[1, 2], [3, 4]], dtype='int8'),
    demons=(array([1, 3], dtype='int8'),),
    demons_costs=array([1], dtype='int8'),
    buffer_size=2
)
