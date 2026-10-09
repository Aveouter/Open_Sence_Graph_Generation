"""The single source of truth for SGG method identity.

A method's name was previously written down in five tables that had already
drifted: ``src/methods/__init__.py``'s ``method_maps``, the ``--method`` choices
in ``utils/parser.py``, ``train.py``'s config-file alias map, and two tables in
``tools/ci_smoke_test.py``.  Adding a method meant editing eight to ten places,
and four CI validators existed to notice when one was missed.

Two design decisions make this module usable as that source of truth:

**Module names, not class references.**  ``cls_name`` and ``method_module`` are
strings, resolved lazily by ``src/methods/__init__.py``.  Holding the classes
directly would mean importing every method module here, which imports torch, and
``tools/ci_validate.py`` runs on the dependency-free CI job.  A checker that
cannot import the table it checks is no use, so the table holds no code.

**Nothing here imports anything but the standard library.**  The same reason.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    """One registered method.

    ``key`` is the canonical lowercase registry key.  ``cli_name`` is the
    spelling ``--method`` accepts.  ``method_module`` is a module stem under
    ``src/methods/``; the class it defines is ``cls_name``.  ``config_stems``
    names every file under ``configs/<dataset>/`` that configures this method,
    primary first -- RA-SGG ships one per task, so there are three.  ``cli_aliases``
    are additional accepted spellings for ``--method``, kept so existing commands
    and config filenames keep working.
    """

    key: str
    cli_name: str
    cls_name: str
    method_module: str
    config_stems: tuple[str, ...]
    cli_aliases: tuple[str, ...] = ()

    @property
    def config_stem(self) -> str:
        """The primary config basename, which ``train.py`` resolves by default."""
        return self.config_stems[0]


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("cvc", "CVC", "CVC_Method", "cvc_method", ("CVC",)),
    MethodSpec("egtr", "EGTR", "EGTR_Method", "egtr_method", ("EGTR",)),
    MethodSpec("flowsg", "FlowSG", "FlowSG_Method", "flowsg_method", ("FlowSG",)),
    MethodSpec(
        "gpsnet",
        "GPSNet",
        "GPSNet_Method",
        "gpsnet_method",
        ("GPS_Net",),
        ("gps_net",),
    ),
    MethodSpec("hstrnet", "HSTRNet", "HSTRNet_Method", "hstrnet_method", ("HSTRNet",)),
    MethodSpec("imp", "IMP", "IMP_Method", "imp_method", ("IMP",)),
    MethodSpec("motifs", "Motifs", "Motifs_Method", "motifs_method", ("Motifs",)),
    MethodSpec(
        "penet",
        "PENet",
        "PENet_Method",
        "penet_method",
        ("PE_NET",),
        ("pe_net",),
    ),
    MethodSpec(
        "ra_sgg",
        "RA_SGG",
        "RA_SGG_Method",
        "ra_sgg_method",
        ("RA_SGG", "RA_SGG_sgcls", "RA_SGG_sgdet"),
    ),
    MethodSpec("react", "REACT", "REACT_Method", "react_method", ("REACT",)),
    MethodSpec("reltr", "RelTR", "RelTR_Method", "reltr_method", ("RelTR",)),
    MethodSpec(
        "shagcl",
        "SHAGCL",
        "SHAGCL_Method",
        "shagcl_method",
        ("SHA_GCL",),
        ("sha_gcl",),
    ),
    MethodSpec("squat", "SQUAT", "Squat_Method", "squat_method", ("SQUAT",)),
    MethodSpec("tde", "TDE", "TDE_Method", "tde_method", ("TDE",)),
    MethodSpec(
        "transformer",
        "Transformer",
        "TransformerSGG_Method",
        "transformer_method",
        ("Transformer",),
    ),
    MethodSpec("usg", "USG", "USG_Method", "usg_method", ("USG",)),
    MethodSpec("vctree", "VCTree", "VCTree_Method", "vctree_method", ("VCTree",)),
)

BY_KEY: dict[str, MethodSpec] = {spec.key: spec for spec in METHODS}


def resolve(spec: MethodSpec):
    """Import and return the Lightning method class ``spec`` names.

    Lazy by necessity: this is the only place the registry reaches for code that
    imports torch, and it happens when a run actually needs the class.
    """
    import importlib

    module = importlib.import_module(f"src.methods.{spec.method_module}")
    return getattr(module, spec.cls_name)


def key_for_config_stem(stem: str) -> str | None:
    """The registry key a config filename (without ``.py``) belongs to.

    ``RA_SGG_sgcls`` and ``RA_SGG_sgdet`` are task variants of one method, so
    three config files resolve to one key.  This is what ``train.py`` and the
    smoke test each used to spell out separately.
    """
    lowered = stem.lower()
    for spec in METHODS:
        for candidate in spec.config_stems:
            if candidate.lower() == lowered:
                return spec.key
    return None
