"""Skip guards for tests whose dependencies are not installed in every CI job.

The ``validate`` CI job runs ``python -m unittest discover`` on a bare Python
with no third-party packages, while the local ``hsg`` environment has the full
stack.  Tests that import torch/numpy must therefore bail out at *import* time
rather than failing.

unittest treats a module that raises :class:`unittest.SkipTest` during import as
a skipped module, so ``require_modules`` is called at module scope, before the
guarded third-party imports::

    import unittest

    from tests._optional import require_modules

    require_modules("torch", "lightning")

    import torch

Note the ordering: ``torch`` is imported *after* the guard, and the class it
decorates must not touch ``torch`` at class-body scope, only inside test bodies.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def unavailable(*modules: str) -> str | None:
    """Return a skip reason if any module is unimportable, else ``None``."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            return f"optional dependency {name!r} unavailable: {exc}"
    return None


def require_modules(*modules: str) -> None:
    """Raise :class:`unittest.SkipTest` unless every module can be imported."""
    reason = unavailable(*modules)
    if reason is not None:
        raise unittest.SkipTest(reason)


def skip_unless(*modules: str):
    """Class/method decorator form, for use when only *part* of a module needs it.

    ``require_modules`` skips a whole file, which is wrong when a module has
    stdlib-only tests alongside a few that need torch.  With this decorator the
    module imports cleanly and only the decorated cases skip, so the guarded
    import can stay inside the test body.
    """

    def decorate(obj):
        return unittest.skipIf(unavailable(*modules) is not None, unavailable(*modules) or "")(obj)

    return decorate
