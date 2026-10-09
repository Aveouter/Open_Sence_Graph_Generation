"""The runtime method registry, built from :mod:`src.method_registry`.

``method_maps`` is what the runtime dispatches on -- ``src/exp.py`` does
``method_maps[args.method](...)`` -- and it is the only thing callers of this
package use.  The seventeen import lines it used to need are gone: the classes
are resolved from the registry instead, so a method exists in exactly one place.

The classes are still resolved *eagerly* here.  That is deliberate and keeps
this module's behaviour unchanged: importing ``src.methods`` already pulled in
every method module, and making it lazy would hide an import error that
currently surfaces at startup rather than mid-run.
"""

from src.method_registry import METHODS, resolve

# 规范method name 为小写
method_maps = {spec.key: resolve(spec) for spec in METHODS}

__all__ = ["method_maps"]
