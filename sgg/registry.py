# sgg/registry.py
"""
Universal component registry for the SGG framework.

All extensible components (models, datasets, tasks, evaluators, losses)
are registered and discovered through this unified registry system.

Usage:
    MODEL_REGISTRY = Registry("model")

    @MODEL_REGISTRY.register('reltr')
    class RelTRAdapter(BaseModelAdapter):
        ...

    adapter = MODEL_REGISTRY.get('reltr')(config)
    print(MODEL_REGISTRY.list())  # ['reltr', 'hstrnet', ...]
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Generic

T = TypeVar('T')


class Registry(Generic[T]):
    """
    Generic registry for named component classes.

    Components are registered via decorator or explicit call,
    and discovered automatically by scanning package directories.
    """

    def __init__(self, name: str, base_class: Type[T] = object):
        self._name = name
        self._base_class = base_class
        self._entries: Dict[str, Type[T]] = {}
        self._discovered = False

    def register(self, name: Optional[str] = None):
        """
        Decorator: register a class under the given name.

        @MODEL_REGISTRY.register('reltr')
        class RelTRAdapter(BaseModelAdapter):
            ...
        """
        def _register(cls: Type[T]) -> Type[T]:
            key = name or cls.__name__.lower()
            if key in self._entries:
                raise KeyError(
                    f"{self._name} registry: '{key}' is already registered. "
                    f"Existing: {self._entries[key].__name__}"
                )
            self._entries[key] = cls
            return cls
        return _register

    def get(self, name: str) -> Type[T]:
        """Look up a registered component by name."""
        if name not in self._entries:
            available = sorted(self._entries.keys())
            raise KeyError(
                f"{self._name} registry: '{name}' not found. "
                f"Available: {available}"
            )
        return self._entries[name]

    def list(self) -> List[str]:
        """Return sorted list of all registered component names."""
        return sorted(self._entries.keys())

    def items(self):
        """Return (name, cls) pairs for all registered components."""
        return self._entries.items()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry('{self._name}', entries={self.list()})"

    def auto_discover(self, package_path: str):
        """
        Auto-discover and import all modules in a package directory.

        This scans the directory of the given package and imports all
        submodules, triggering any @register decorators they contain.

        Args:
            package_path: Dotted Python package path, e.g. 'sgg.models'
        """
        try:
            pkg = importlib.import_module(package_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {package_path} for auto_discover: {e}"
            )

        pkg_dir = Path(pkg.__file__).parent

        for module_info in pkgutil.iter_modules([str(pkg_dir)]):
            importlib.import_module(f"{package_path}.{module_info.name}")

        self._discovered = True


# ---------------------------------------------------------------------------
# Global registry instances
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Registry = Registry("model")
"""Registry for model adapters (BaseModelAdapter subclasses)."""

DATASET_REGISTRY: Registry = Registry("dataset")
"""Registry for SGG datasets (BaseSGGDataset subclasses)."""

TASK_REGISTRY: Registry = Registry("task")
"""Registry for SGG tasks (BaseTask subclasses)."""

LOSS_REGISTRY: Registry = Registry("loss")
"""Registry for loss modules."""

EVALUATOR_REGISTRY: Registry = Registry("evaluator")
"""Registry for evaluators (BaseEvaluator subclasses)."""

METRIC_ADAPTER_REGISTRY: Registry = Registry("metric_adapter")
"""Registry for metric adapter functions that convert model outputs to evaluator format."""
