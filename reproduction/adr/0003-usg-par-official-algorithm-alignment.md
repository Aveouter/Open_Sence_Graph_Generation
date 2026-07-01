# ADR 0003: USG-Par Official Algorithm Alignment

## Status

Accepted

## Context

The local OpenSGG USG candidate used a RelTR-shaped output interface, but several
core algorithm choices differed from the official USG implementation:

- object classification used a fixed linear classifier and softmax CE;
- relation classification used single-label CE;
- duplicate predicates for the same subject-object pair were overwritten;
- RPC manually masked diagonal pairs;
- the two-way RAC layer updated the object stream from already-updated subject states;
- SGDet metrics treated USG logits as RelTR-style softmax outputs.

These differences are algorithmic, not merely implementation style.

## Decision

Align the local image-only USG candidate with official USG semantics where the
Visual Genome box protocol allows it:

- use official CLIP-style cosine object classification;
- use sigmoid BCE object classification and sigmoid-class Hungarian matching;
- use foreground-only predicate logits with multi-hot BCE targets;
- use weighted BCE for pair confidence;
- restore official RPC top-k and two-way RAC ordering;
- route SGDet evaluation through USG-specific sigmoid scoring and VG label offsets.

Keep VG bbox L1/GIoU losses as a documented dataset adaptation because OpenSGG's
Visual Genome path does not provide the official PSG mask supervision.

## Evidence

- Paper/source: USG-Par / Universal Scene Graph Parser
- Official repo/source:
  - `usg_par/detection_head.py`
  - `usg_par/rpc.py`
  - `usg_par/losses.py`
  - `usg_par/training/targets.py`
  - `usg_par/training/loss_assembly.py`
- Local file/command:
  - `src/models/usg.py`
  - `src/core/metrics.py`
  - `configs/VisualGenome/USG.py`

## Consequences

- What can be claimed:
  - local USG candidate is algorithmically closer to the official image-path USG design;
  - current work is an `implementation_audit`.
- What cannot be claimed:
  - no USG-Par reproduction success;
  - no checkpoint-backed baseline result;
  - no direct comparability to official PSG numbers.
- What must happen next:
  - verify checkpoint identity/provenance;
  - audit preprocessing and evaluator parity;
  - only then run checkpoint-backed metrics.
