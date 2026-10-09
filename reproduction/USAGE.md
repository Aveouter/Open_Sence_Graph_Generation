# Reproduction Workflow Usage Guide

This guide explains how to use the OpenSGG reproduction workflow. It is written
for both human maintainers and AI agents.

## What This Part Is For

Use this workflow when a task claims to reproduce, audit, evaluate, or defer an
SGG baseline. The purpose is to keep evidence and claims aligned:

- official paper/repository first
- checkpoint provenance before metrics
- configuration, inference, and evaluator semantics recorded before results
- smoke/audit/gap work clearly separated from reproduction claims

Do not use this workflow to make an approximate implementation look complete.

## Required Reading

Before touching code for a baseline:

1. Read `AGENTS.md`.
2. Read `reproduction/README.md`.
3. Read `reproduction/glossary.md`.
4. Read `reproduction/issues/baseline_reproduction_issue.md`.
5. Read `reproduction/templates/pr_checklist.md`.

Codex-specific notes are in `reproduction/agents/codex.md`. Claude Code notes
are in `reproduction/agents/claude_code.md`, but Codex rules are authoritative
when the two differ.

## Baseline Workflow

For each baseline, create an auditable trail in this order:

1. Create an issue from `reproduction/issues/baseline_reproduction_issue.md`.
2. Record official paper and official repository links.
3. Record checkpoint source, checksum if available, and license/access notes.
4. Compare OpenSGG code against the official implementation.
5. Compare config, preprocessing, inference flow, and evaluator semantics.
6. Choose the correct outcome label from `reproduction/README.md`.
7. Run metrics only after the required alignment gates pass.
8. Write deviations and blockers before opening a PR.

If any required evidence is missing, stop at `implementation_audit`,
`checkpoint_unavailable`, `protocol_mismatch`, or `deferred_reproduction`.

## Evidence Gates

Use these gates before calling a result reproduction evidence:

| Gate | Required evidence |
|---|---|
| Method | local implementation matches the paper or official repo, or deviations are documented |
| Checkpoint | checkpoint identity, source, and compatibility are verified |
| Config | dataset, splits, preprocessing, and hyperparameters match the official protocol |
| Inference | prediction path and postprocessing match the official protocol |
| Evaluation | metric names, modes, and evaluator semantics match the official protocol |

Passing a smoke test does not pass these gates.

## What To Do When Evidence Is Missing

Use the precise label rather than stretching the claim:

- missing official checkpoint: `checkpoint_unavailable`
- official evaluator differs from OpenSGG: `protocol_mismatch`
- local code is only approximate: `implementation_audit`
- only import/forward/eval loop ran: `pipeline_smoke_only`
- work should wait for official inputs: `deferred_reproduction`

Write the missing evidence in the issue and in the relevant report. Do not
replace missing evidence with random initialization, tiny slices, partial
checkpoint remaps, or all-zero metrics.

## Guardrail Check

Before opening or updating a PR, run:

```bash
python tools/reproduction/run_reproduction_guardrails.py
```

This aggregate guardrail compiles reproduction tools, checks tracked report
completeness, runs reproduction guardrail unit tests, and scans claim-bearing
text. When `tools/relational_emergence/` is present, it also runs the existing
Phase IA protocol validator against `outputs/analysis/relational_emergence/`.
Protocol violations fail the guardrail. To validate another isolated analysis
root, pass `--protocol-root <path>`. Repository-boundary, artifact-provenance,
documentation, and claim checks also run through this entrypoint. It does not run
model evaluation.

CI invokes this same entrypoint after its unit-test step, so checker failures
fail the validation job rather than relying on indirect test coverage.

Missing artifact manifest:

```bash
python tools/reproduction/build_artifact_manifest.py
```

This creates `reproduction/evidence/artifact_manifest.json`, a structured list of
official inputs, checkpoints, detector weights, and memory-bank artifacts still
needed before any deferred baseline can be reconsidered for reproduction.

Individual claim guardrail:

```bash
python tools/reproduction/check_reproduction_claims.py \
  --changed-from origin/main \
  AGENTS.md reproduction <pr-body-or-release-note-files>
```

Also pass any Markdown/text files outside `reproduction/` that contain claims,
such as report drafts or copied PR body text. The check scans text only. Passing
it does not prove reproduction; it only catches common misleading phrasing.

## PR Checklist

Use `reproduction/templates/pr_checklist.md` in every reproduction-related PR.
A valid PR should be one of these:

- code alignment fix
- checkpoint-backed evaluation
- protocol alignment documentation
- implementation audit
- deferred reproduction report
- guardrail against misleading claims

The PR title and body must match the evidence. For example:

- Good: `audit(tde): record checkpoint-unavailable reproduction blocker`
- Good: `fix(freq): align prior computation with official statistics file`
- Not allowed: `reproduce tde baseline` when no verified checkpoint exists

## Agent Operating Rules

Agents must not start by trying to make a command pass. Start with source,
checkpoint, config, inference, and evaluation alignment. If alignment cannot be
verified, produce an audit or deferred report and stop.

Agents must not:

- modify labels or ground truth
- change evaluator semantics to improve numbers
- delete failed samples
- hide zero or failed metrics
- present random-init or tiny-slice output as reproduction
- call an OpenSGG adapter official without comparing it to the official repo

## Typical Commands

Claim guardrail:

```bash
python tools/reproduction/check_reproduction_claims.py \
  AGENTS.md reproduction
```

Documentation completeness guardrail for the target baseline suite:

```bash
python tools/reproduction/check_reproduction_docs.py
```

This check verifies that FREQ, TDE, VCTree, PENet, SHA-GCL, and RA-SGG have
tracked phase reports, evidence-gate audits, official-input check JSON files,
baseline matrix rows, method gate sections, the suite evidence runner, and the
tracked suite summary. It does not prove reproduction.

Python syntax check for reproduction tools:

```bash
python -m py_compile tools/reproduction/*.py
```

FREQ/TDE/VCTree input audits, when those files are present:

```bash
python tools/reproduction/check_sgb_freq_inputs.py \
  --output reproduction/evidence/freq/sgb_freq_input_check.json

python tools/reproduction/check_tde_official_inputs.py \
  --output reproduction/evidence/tde/tde_official_input_check.json

python tools/reproduction/check_vctree_official_inputs.py \
  --output reproduction/evidence/vctree/vctree_official_input_check.json
```

Full target-suite evidence gate audit:

```bash
python tools/reproduction/run_evidence_gate_checks.py
```

Exit code `2` means at least one baseline is blocked by missing official inputs
or checkpoints. Treat that as a deferred reproduction state, not as a successful
metric result.

The audit commands report blockers. A blocker is not a failure to hide; it is
the evidence needed to keep the reproduction claim honest.

## RelateAnything input preflight (issue #124)

This supported input checker audits external official source, a release snapshot,
and the VG150 test pack. It imports neither upstream code nor PyTorch, loads no
weights, downloads nothing, and runs no metrics. Keep its JSON output, external
load/split reports, downloaded assets and official checkout outside this repository.

```bash
python tools/reproduction/check_relateanything_official_inputs.py \
  --official-root <external-official-checkout> \
  --snapshot <external-model-snapshot> \
  --data-root <external-packed-vg150> \
  --protocol A1 \
  --evidence <external-input-evidence.json> \
  --output <external-output-dir>/relateanything-preflight.json
```

Omit `--evidence` for initial inventory: absent evidence is a blocker. Exit `2`
means blocked; exit `0` and `can_evaluate=true` mean the **input gate** passes
against the supplied external attestations. The checker does not independently
rerun the strict loader or dataset audit, authenticate the report's author, or
prove revision-to-file provenance from the Hub. `reproduction_ready` stays false:
method/config/inference/evaluator parity, paper-table identity, environment and
license decisions remain separate gates before an official baseline run.

The source must be a clean Git checkout at issue #124's pinned commit
`06766fdf56752ca535fc9b971fca99ce563676d0`. `--official-commit` permits an explicitly
recorded revision change; record its compatibility decision in an ADR. The checker
always resolves `text_student.pt` beside `model.pth`, and emits an argv list in
`reference_command` with `--split test --graph_constraint --limit 0`. This list is
a proposed command, not an executed or fully validated reference invocation.
Other upstream defaults must still be recorded in the evaluation protocol.

The evidence JSON has these fields. Populate them from real official load and
full-input-audit logs, never from a smoke run or just the inventory:

- `official_commit`: full source SHA, matching the clean checkout.
- `snapshot_revision`, `data_revision`: full 40-character Hub revisions.
- `snapshot_sha256`: SHA256 mapping for `model.pth`, `text_student.pt`,
  `tokenizer.json`, `tokenizer_config.json`.
- `strict_load`: `strict=true`; `missing_keys`, `unexpected_keys`,
  `shape_mismatches`, `remapped_keys` all empty arrays; `model_sha256` and
  `text_student_sha256` matching current files; `source_sha256` mapping for
  `benchmark/eval_zeroshot.py`, `relsgg/checkpoint.py`, `relsgg/eval/evaluator.py`,
  `relsgg/data/dataset.py`, `pyproject.toml`, `LICENSE`, `NOTICE`.
- `full_split`: `dataset="vg150"`, `split="test"`; `expected_images`,
  `loaded_images`, `valid_input_images` all exactly **26,404** for this official
  pack; `failed_images=0`, `limit=0`; `pack_sha256` mapping for `test/meta.json`,
  `file_names.json`, `img_meta.npy`, `boxes.npy`, `box_cats.npy`, `rels.npy`, with
  keys relative to `test/` (including `meta.json`). The external input audit must
  actually decode every image and inspect relations under the official caps;
  these are pre-inference counts, not scored denominators. Scored counts and
  trimming must be reconciled after evaluation. The expected count comes from
  the [pinned official SPEC](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/SPEC.md).

Current hashes are reported in `artifact_sha256`; their presence does not prove a
strict load, full split, provenance, or reproduction. Local input hash changes
invalidate the corresponding supplied load/split attestation. Raw-image content
is not hashed by this checker, so a pack audit must be rerun if its image assets
change even when the pack arrays are unchanged.

For A3, add `--protocol A3 --tau-calibration <external-matcher-tau.json>`.
Deployment `calibration.json` is **not** the synonym-matcher calibration. The
matcher JSON must contain `pred_embeds` identifying this snapshot's
`predicate_embeddings.npz`, and a finite cosine threshold in `chosen.tau`.
Relative `pred_embeds` paths resolve beside the calibration file, never against
the caller's working directory; prefer an absolute path in upstream-compatible
calibration records. The pinned runner expects NPZ fields `predicates` and
`embeddings`; the release fields `names`/`W` are rejected, never silently converted.
Archive member inspection checks layout only, not array values or dimensions.
An A3 `a3` evidence object must additionally bind `embedding_sha256`,
`calibration_sha256`, and absolute `checkpoint_pred_embeds`; it must attest
`vocabulary_order_verified=true` and `runner_layout_verified=true` against the
same source/model/student identified above. This does not repair the pinned
runner's released-layout incompatibility or establish matcher/inverse-mask parity.

Behavior tests use temporary synthetic files and attestations solely to exercise
the preflight decision, not as model evaluation or baseline evidence:

```bash
python -m unittest discover -s tests/reproduction -p test_relateanything_inputs.py
```

## RelateAnything ported runtime

The complete relation network, text student, scoring, region encoding, loss and
evaluator live under `src/modules/relateanything`. This Apache-2.0 component
retains its license and notices, with source identity in `UPSTREAM.json`.
Object detection and upstream research/deployment runners are separate.

Create an isolated Python 3.13 environment and install
`requirements-relateanything.txt`. Keep the existing OpenSGG environment intact.
Select the appropriate CUDA PyTorch wheels separately for GPU execution; the
recorded local verification used CPU. Run all commands below from the repo root.

```bash
conda create -p <external-runtime> python=3.13 pip
conda run -p <external-runtime> python -m pip install -r requirements-relateanything.txt
conda run -p <external-runtime> python train.py --method RelateAnything --test \
  --ckpt_path <external-snapshot>/model.pth --device cpu \
  --image <image.png> --regions <regions.json> \
  --predicate_vocabulary <predicates.json> --relation_output <external-output>/prediction.json
```

`regions.json` is a list of original-image pixel xyxy boxes; `predicates.json`
is a nonempty unique string list. Object labels are not inputs. The snapshot
must contain the released text student and tokenizer sidecars. Alternatively,
`--full_vocabulary` selects the released predicate bank. Optional `.npy` binary
masks, confidence JSON and decomposition use `--relation_masks`,
`--relation_box_scores` and `--relation_decompose`. Python callers use
`src.relateanything.RelateAnythingModel.from_checkpoint(...).predict(...)`.

Evaluation without `--image` requires the full official pack and actual external
load/split evidence. Use `--relation_official_root`, `--relation_evidence`,
`--data_root`, and `--relation_protocol A1|A3`. The CLI invokes the checker with
`--consumer opensgg --official-commit 4a07de9d06f2e3f14309753b7907cf1d3a263b08`.
This Apache parent has no `NOTICE` file; omit that key from its source evidence.
The evidence must additionally bind `consumer_sha256` to the current three
`src/relateanything*.py` facade/CLI/evaluation files reported by the checker.
For this consumer, A3 accepts the release `names`/`W` NPZ layout; the default
official-runner consumer retains the legacy schema requirement described above.
A3 still requires `--relation_tau_calibration` and bound matcher evidence.

Metrics are explicitly named `regions_A1_*` or `regions_A3_*`. Subset evaluation,
non-`model.pth` checkpoints and silent denominator losses are rejected. Outputs
retain `reproduced=false`; passing an input gate does not establish paper parity.
Full split data and matcher calibration remain prerequisites for baseline runs.

Code parity tests can include external release assets:

```bash
# Set these environment variables using your shell's syntax first:
# RELATEANYTHING_TEST_CHECKPOINT=<external-snapshot>/model.pth
# RELATEANYTHING_REFERENCE_ROOT=<external-Apache-parent-checkout>
conda run -p <external-runtime> python -m pytest tests/reproduction/test_relateanything_port.py
```

Synthetic fixtures and small inference comparisons in these tests verify code
behavior; they are not baseline evaluation evidence or reproduction results.
