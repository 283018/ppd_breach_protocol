from typing import Callable, Any, Tuple, Dict, Type, List, Iterable, Self
from numpy import ndarray, array
from abc import ABC, ABCMeta, abstractmethod
from core import Task, Solution, NoSolution


SAFE_TYPES = {int, float, str, bool}
SEQUENCE_TYPES = {list, tuple, set, ndarray}
CONSTR_MAP = {list: list, tuple: tuple, set:set, ndarray: array}


solver_registry:Dict[str, Callable] = {}


def register_solver(name: str) -> Callable[[Any], Any]:
    """Decorator to register solvers"""
    def decorator(cls):
        solver_registry[name] = cls
        return cls
    return decorator


class OptimizationError(Exception):
    """Raised when optimization of a task is not possible"""
    pass


class Solver(ABC, metaclass=ABCMeta):
    """
    Solver abstract base class
    """
    #! WARNING: only build-in python allowed to be specified as kwarg type
    _allowed_kwargs: Dict[str, Type] = {}

    def __init__(self):
        self._warm_up()

    @abstractmethod
    def _warm_up(self):
        """Solver initialization and dummy task run"""
        ...

    def __call__(self, task:Task, **kwargs:Any) -> Tuple[Solution|NoSolution, float]:
        return self.solve(task, **kwargs)

    def s(self, task:Task, **kwargs:Any) -> Tuple[Solution|NoSolution, float]:
         return self.solve(task, **kwargs)

    def solve_iter(self, task_list:Iterable[Task], **kwargs:Any) -> Tuple[List[Solution|NoSolution], List[float,]]:
        """
        Solve tasks from any iterable in loop.
        :param task_list:
        :param kwargs: kwargs accepted by self.solve
        :return: List of solutions, list of times
        """

        solved = []
        times = []
        for task in task_list:
            sol, time = self.solve(task, **kwargs)
            solved.append(sol); times.append(time)
        return solved, times


    @abstractmethod
    def solve(self, task:Task, **kwargs:Any) -> Tuple[Solution|NoSolution, float]:
        """
        'solve' is main method to solve task
        's' and '__call__' are essentially a shortcuts to 'solve', look __doc__ of subclass solve method
        """
        ...
    __call__.__doc__ = s.__doc__ = solve.__doc__


    def _validate_kwargs(self, kwargs:Dict[str, Any]) -> None:
        """Method to validate optional kwargs for solvers main method (if needed)"""
        # Access _allowed_kwargs from the subclass
        allowed_kwargs = self.__class__._allowed_kwargs
        if kwargs and not allowed_kwargs:
            raise TypeError(f"{self.__class__.__name__} does not support any keyword arguments. Received: {list(kwargs.keys())}")
        for name, value in kwargs.items():
            if name not in allowed_kwargs:
                raise TypeError(f"Unexpected keyword argument: '{name}'")

            expected_type = allowed_kwargs[name]
            if expected_type == float and isinstance(value, int):
                kwargs[name] = float(value)
                continue
            if expected_type == int and isinstance(value, float):
                kwargs[name] = int(value)
                continue
            if expected_type in SAFE_TYPES:
                try:
                    kwargs[name] = expected_type(value)
                    continue
                except (ValueError, TypeError):
                    pass
            if expected_type in SEQUENCE_TYPES and type(value) in SEQUENCE_TYPES:
                try:
                    # noinspection PyTypeChecker
                    kwargs[name] = CONSTR_MAP[expected_type](value)
                    continue
                except TypeError as e:
                    raise TypeError(f"Argument '{name}' cannot be converted to {expected_type.__name__}: {e}") from e

            raise TypeError(f"Argument '{name}' must be {expected_type.__name__} or convertable to it. "
                            f"Got {type(value).__name__} with value: {repr(value)}")



class SeedableSolver(Solver, ABC):
    """
    Abstract base class for solvers that require random number generation.
    """
    @abstractmethod
    def seed(self, value: int) -> Self:
        """Set the random number generator seed."""
        ...
