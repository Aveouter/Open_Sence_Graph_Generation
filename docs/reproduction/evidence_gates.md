# Reproduction Evidence Gates

This file defines the minimum evidence required before a deferred baseline can
be reopened as a reproduction PR. It is intentionally stricter than a runnable
pipeline. Passing smoke tests, exporting JSONL, or obtaining non-crashing metric
output is useful, but it does not satisfy reproduction.

## Global Gates

Every method must satisfy these gates before any PR title, report, or matrix row
may call it reproduced, paper-aligned, or benchmark-equivalent.

| Gate | Requirement | Acceptable evidence | Non-evidence |
|---|---|---|---|
| G-1 | Official source identified | Paper citation plus official repository URL, commit, release, or archived source snapshot | A similarly named OpenSGG implementation alone |
| G-2 | Checkpoint provenance verified | Official checkpoint URL or local file with source, SHA256, expected architecture, and load log | Random init, partial remap, or unknown checkpoint origin |
| G-3 | Config parity established | Side-by-side paper/official config vs OpenSGG config for task, split, object boxes, labels, relation classes, and hyperparameters used at eval | A config that only instantiates the model |
| G-4 | Inference protocol aligned | Documented command showing the same task mode, preprocessing, object proposals or GT boxes, postprocessing, and top-K behavior | Tiny-slice export with different preprocessing or task semantics |
| G-5 | Evaluator semantics aligned | Mapping from official metric code to OpenSGG evaluator behavior, including predicate indexing, background handling, graph constraints, no-graph-constraint mode, and recall/mRecall definitions | Any evaluator change made only to improve numbers |
| G-6 | Benchmark-scale result recorded | Full intended split result, or an explicit resource-blocked report that does not claim reproduction | One-batch, two-image, all-zero, or smoke-only output |
| G-7 | Claim boundary audited | Final report lists current trusted claims and forbidden claims | A success summary without deviations |

If any global gate is missing, the status must remain one of:

- `DEFERRED_NOT_REPRODUCED`
- `IMPLEMENTATION_AUDIT`
- `PIPELINE_SMOKE_ONLY`
- `CHECKPOINT_UNAVAILABLE`
- `PROTOCOL_MISMATCH`
- `FAILED_WITH_REPORT`

## Method Gates

### FREQ

| Gate | Requirement |
|---|---|
| FREQ-1 | Verify whether the target FREQ baseline is the Neural Motifs frequency prior or another paper-specific frequency baseline. |
| FREQ-2 | Document the exact training split used to build `P(predicate | subject, object)` and confirm no validation/test labels enter the prior. |
| FREQ-3 | Compare OpenSGG `PairFrequencyBias` predicate indexing and background handling to the official Motifs implementation. |
| FREQ-4 | Run benchmark-scale PredCls evaluation with the aligned frequency prior and unchanged evaluator semantics. |

### Motifs

| Gate | Requirement |
|---|---|
| MOTIFS-1 | Pin the official Neural Motifs or Scene-Graph-Benchmark source commit used as reference. |
| MOTIFS-2 | Verify checkpoint provenance and architecture compatibility without partial-remap ambiguity. |
| MOTIFS-3 | Align object boxes/features, context LSTM dimensions, frequency bias, graph constraint, and PredCls/SGCls/SGDet mode. |
| MOTIFS-4 | Compare OpenSGG evaluator semantics against the official Motifs evaluation script. |
| MOTIFS-5 | Record benchmark-scale metrics and preserve any mismatch as deviation, not as hidden postprocessing. |

### TDE

| Gate | Requirement |
|---|---|
| TDE-1 | Pin the official `Scene-Graph-Benchmark.pytorch` source commit for unbiased SGG. |
| TDE-2 | Verify the TDE checkpoint, including causal analysis mode and statistics required by the official implementation. |
| TDE-3 | Align the OpenSGG causal subtraction path with official Total Direct Effect inference. |
| TDE-4 | Align config, object proposals or GT boxes, relation labels, graph constraints, and evaluator mode. |
| TDE-5 | Run benchmark-scale metrics or document a resource/checkpoint blocker without claiming reproduction. |

### VCTree

| Gate | Requirement |
|---|---|
| VCTREE-1 | Pin the official VCTree paper implementation or authoritative Scene-Graph-Benchmark source. |
| VCTREE-2 | Verify checkpoint provenance and tree-context architecture compatibility. |
| VCTREE-3 | Align tree construction, object/context features, predicate indexing, and evaluation task mode. |
| VCTREE-4 | Compare evaluator semantics with the official reference. |
| VCTREE-5 | Run benchmark-scale metrics or keep the method deferred with a checkpoint/protocol gap. |

### PENet

| Gate | Requirement |
|---|---|
| PENET-1 | Pin the official PENet source and paper protocol. |
| PENET-2 | Verify checkpoint provenance and architecture compatibility. |
| PENET-3 | Align predicate embedding, relation feature construction, frequency-bias usage, and task mode. |
| PENET-4 | Compare evaluator semantics with official metrics. |
| PENET-5 | Run benchmark-scale metrics or keep the method deferred with documented gaps. |

### SHA-GCL

| Gate | Requirement |
|---|---|
| SHAGCL-1 | Pin the official SHA-GCL source and paper protocol. |
| SHAGCL-2 | Verify checkpoint provenance and any auxiliary statistics required for debiasing or contrastive components. |
| SHAGCL-3 | Align semantic hierarchy, graph contrastive logic, predicate indexing, and task mode. |
| SHAGCL-4 | Compare evaluator semantics with official metrics. |
| SHAGCL-5 | Run benchmark-scale metrics or keep the method deferred with documented gaps. |

### RA-SGG

| Gate | Requirement |
|---|---|
| RASGG-1 | Pin the official RA-SGG/ReTAG source commit. |
| RASGG-2 | Verify checkpoint provenance. |
| RASGG-3 | Verify memory-bank or retrieval artifacts, including build split and update/freeze behavior. |
| RASGG-4 | Align relation-aware augmentation/retrieval logic with the official implementation. |
| RASGG-5 | Align evaluator semantics and task mode. |
| RASGG-6 | Run benchmark-scale metrics or keep the method deferred; a no-memory adapter remains audit only. |

### RelTR

| Gate | Requirement |
|---|---|
| RELTR-1 | Pin the official RelTR source commit and checkpoint release. |
| RELTR-2 | Verify checkpoint SHA256, model dimensions, query count, and backbone compatibility. |
| RELTR-3 | Align Hungarian matching, triplet construction, predicate indexing, and postprocessing. |
| RELTR-4 | Use a RelTR-compatible evaluator protocol, not a Motifs-style shortcut unless formally justified. |
| RELTR-5 | Run benchmark-scale metrics or keep the method deferred with protocol mismatch documented. |

### EGTR

| Gate | Requirement |
|---|---|
| EGTR-1 | Pin the official EGTR source commit and checkpoint release. |
| EGTR-2 | Verify checkpoint SHA256, Deformable DETR backbone, query count, and relation head dimensions. |
| EGTR-3 | Align dense query relation targets, sigmoid predicate scoring, and EGTR postprocessing. |
| EGTR-4 | Use EGTR-native evaluator semantics; do not claim parity from Motifs-style PredCls probing alone. |
| EGTR-5 | Run benchmark-scale metrics or keep the method deferred with protocol mismatch documented. |

## PR Rule

A PR may be opened while a method is deferred only if the title and body say
`audit`, `smoke`, `gap analysis`, or `deferred reproduction`. It may not say
`reproduction`, `reproduced`, `paper-aligned`, or `baseline result` unless the
global gates and the method-specific gates above are all satisfied.
