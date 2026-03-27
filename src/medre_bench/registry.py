"""Generic registry for models and datasets."""

from __future__ import annotations

from typing import TypeVar, Type, Callable

T = TypeVar("T")


class Registry:
    """Maps string keys to classes via decorator registration."""

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Type] = {}

    def register(self, key: str) -> Callable:
        """Decorator to register a class under a given key."""

        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._registry:
                raise ValueError(f"Key '{key}' already registered in {self.name} registry")
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> Type:
        """Retrieve a registered class by key."""
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{key}' not found in {self.name} registry. Available: [{available}]"
            )
        return self._registry[key]

    def list_available(self) -> list[str]:
        """Return sorted list of registered keys."""
        return sorted(self._registry.keys())


MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
