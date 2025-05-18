from typing import Protocol, Callable, Any, Tuple, Dict, Type
from core import Task, Solution



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


class Solver(Protocol):
    _allowed_kwargs: Dict[str, Type] = {}

    # noinspection PyProtocol
    def __init__(self):
        self._warm_up()

    def _warm_up(self):
        raise NotImplementedError("Subclasses must implement '_warm_up'")

    def __call__(self, task:Task, **kwargs) -> Tuple[Solution, float]:
        """Call method takes instance of Task and returns instance of Solution"""
        ...

    def _validate_kwargs(self, kwargs:dict) -> None:
        """Method to validate optional kwargs for some solvers (if needed)"""
        # Access _allowed_kwargs from the subclass
        allowed_kwargs = self.__class__._allowed_kwargs
        if kwargs and not allowed_kwargs:
            raise TypeError(f"{self.__class__.__name__} does not support any keyword arguments. Received: {list(kwargs.keys())}")
        for key, value in kwargs.items():
            if key not in allowed_kwargs:
                raise TypeError(f"Unexpected keyword argument: '{key}'")
            expected_type = allowed_kwargs[key]
            if not isinstance(value, expected_type):
                raise TypeError(f"Argument '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}")


class SeedableSolver(Solver, Protocol):
    def seed(self, value: int) -> None:
        """Set the random number generator seed."""
        ...
