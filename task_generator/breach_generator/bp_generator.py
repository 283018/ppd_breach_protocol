from numpy.random import Generator
from typing import Dict, Protocol, Any, Type


# After TaskFactory all this now really unnecessary, but well :/


class CallableGenerator(Protocol):
    def __call__(self, *args:Any, **kwargs:Any) -> Any:
        ...


generators_registry: Dict[str, Type['BPGen']] = {}

def register_generator(name: str):
    """
    Decorator to register generator class in global registry.
    """
    def decorator(cls):
        generators_registry[name] = cls
        return cls
    return decorator



class BPGen:
    def __init__(self, rng:Generator):
        self.rng = rng

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # This makes BPGen compatible with CallableGenerator
        # Subclasses will override with specific parameters
        raise NotImplementedError("Subclasses must implement __call__")

    @staticmethod
    def create(name: str, rng: Generator) -> 'CallableGenerator':
        """
        Factory method to create generator instance by name
        """
        if name not in generators_registry:
            raise ValueError(f"Unknown generator: {name}")
        return generators_registry[name](rng)