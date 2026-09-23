"""Torch oracle models for the Phase IA relational-emergence study.

This subpackage is the **only** part of the relational-emergence tree allowed to
import ``torch``. Everything under ``simulator/`` and ``audits/`` stays pure
stdlib so that the Level-1 gate runs in the dependency-free CI job; the moment
one of those modules imports torch, that job stops executing the checks it
exists to run.

The oracles here **consume frozen artifacts rather than re-sampling**. They read
``oracle_dataset.jsonl``, written by
:mod:`tools.relational_emergence.experiments.run_phase1a`, and train on the
episodes recorded there. Rebuilding the twins on the model side would let the
split, the compatibility table or the pairing quietly diverge from the audited
ones, and every number downstream would then describe a different experiment
from the one that was checked. The dataset is the interface; nothing in this
subpackage imports the simulator's samplers.

Torch is imported lazily, inside functions, so that this package is importable
by the bare-Python CI job. See :mod:`tools.relational_emergence.models.oracles`.
"""
