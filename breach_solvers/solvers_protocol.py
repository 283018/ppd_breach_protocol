from typing import Protocol, Callable, Any, Tuple, Dict, Type
from core import Task, Solution

solver_registry = {}


def register_solver(name: str) -> Callable[[Any], Any]:
    """Decorator to register solvers"""
    def decorator(cls):
        solver_registry[name] = cls
        return cls
    return decorator


class Solver(Protocol):
    _allowed_kwargs: Dict[str, Type] = {}

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
                raise TypeError(
                    f"Argument '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}")


def get_solver(name: str) -> Solver:
    """Get solver by name."""
    solver_class = solver_registry.get(name)
    if not solver_class:
        raise ValueError(f"Unknown solver: {name}, must be one of {list(solver_registry.keys())}")
    return solver_class()