"""Resolve the selected runtime method without importing unrelated runtimes.

``method_maps[key]`` retains the class lookup interface. Iterating items resolves
every class, so the CI all-method import check still discovers broken imports.
"""

from collections.abc import Mapping

from src.method_registry import METHODS, resolve


class _MethodMap(Mapping):
    def __init__(self):
        self._specs = {spec.key: spec for spec in METHODS}
        self._classes = {}

    def __getitem__(self, key):
        spec = self._specs[key]
        if key not in self._classes:
            self._classes[key] = resolve(spec)
        return self._classes[key]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)

    def __contains__(self, key):
        return key in self._specs


method_maps = _MethodMap()

__all__ = ["method_maps"]
