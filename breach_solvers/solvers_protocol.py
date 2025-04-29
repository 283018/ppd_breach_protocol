from core import Task, Solution
from typing import Protocol, Callable, Any, Tuple

solver_registry = {}


def register_solver(name: str) -> Callable[[Any], Any]:
    """Decorator to register solvers"""
    def decorator(cls):
        solver_registry[name] = cls
        return cls
    return decorator


class Solver(Protocol):
    def __call__(self, task:Task, **kwargs) -> Tuple[Solution, float]:
        """Call method takes instance of Task and returns instance of Solution"""
        ...

    def _validate_kwargs(self, kwargs:dict):
        """Method to validate optional kwargs for some solvers (if needed)"""
        ...


def get_solver(name: str) -> Solver:
    """Get solver by name."""
    solver_class = solver_registry.get(name)
    if not solver_class:
        raise ValueError(f"Unknown solver: {name}, must be one of {list(solver_registry.keys())}")
    return solver_class()