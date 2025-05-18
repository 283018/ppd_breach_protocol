from typing import Callable, Any, Tuple, Dict, Type
from abc import ABC, ABCMeta, abstractmethod
from core import Task, Solution, NoSolution



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
        for key, value in kwargs.items():
            if key not in allowed_kwargs:
                raise TypeError(f"Unexpected keyword argument: '{key}'")
            expected_type = allowed_kwargs[key]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Argument '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}"
                )



class SeedableSolver(Solver, ABC):
    """
    Abstract base class for solvers that require random number generation.
    """
    @abstractmethod
    def seed(self, value: int) -> None:
        """Set the random number generator seed."""
        ...
