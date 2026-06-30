# Reproduction Glossary

## reproduction

Evidence-backed alignment with the original paper or official repository across
method, checkpoint, configuration, inference flow, and evaluator semantics.

## checkpoint-backed evaluation

Evaluation using a checkpoint with recorded provenance, checksum, expected
architecture/config compatibility, and an aligned evaluator.

## official protocol alignment

The preprocessing, task setting, split, metric, evaluator semantics, and
inference procedure match the paper or official repository closely enough that
the result can be compared to the reported baseline.

## pipeline smoke

A minimal import, forward pass, export, or metric-loop run. This proves only
that plumbing executes. It is not reproduction evidence.

## implementation audit

A structured comparison between local code and paper/official repository code.
It may find matches, gaps, or unknowns. It is not a result claim by itself.

## deferred baseline

A baseline whose reproduction is paused because a required artifact or alignment
proof is missing.

## unacceptable evidence

The following must not be used as reproduction evidence:

- random initialization
- tiny slices used only to prove execution
- all-zero metrics
- partial checkpoint remapping without compatibility proof
- approximate adapters without official-code parity
- modified labels, ground truth, metric semantics, or filtered failures
