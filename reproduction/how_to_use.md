# How to Use the Reproduction Workflow

This guide is for future maintainers and AI agents. It explains how to use the
`reproduction/` workflow when evaluating an OpenSGG baseline.

The key rule is simple: do not optimize for getting a command to run. Optimize
for evidence that the local method matches the original paper or official
repository.

## Quick Start

For every baseline, create a tracked status directory and fill the templates
before running evaluation:

```bash
METHOD=<method-name>
mkdir -p reproduction/baselines/$METHOD
cp reproduction/templates/baseline_status.md \
  reproduction/baselines/$METHOD/status.md
cp reproduction/templates/checkpoint_manifest.md \
  reproduction/baselines/$METHOD/checkpoint_manifest.md
cp reproduction/templates/eval_protocol.md \
  reproduction/baselines/$METHOD/eval_protocol.md
cp reproduction/issues/baseline_reproduction_issue.md \
  reproduction/baselines/$METHOD/issue.md
```

Then edit those files with evidence from the paper, official repository,
checkpoint source, local config, inference command, and evaluator.

## Step 1: Define the Claim Boundary

Before running code, decide what kind of work this is allowed to become.

Use these labels:

- `checkpoint_backed_evaluation`: verified checkpoint plus aligned protocol.
- `implementation_audit`: code comparison without sufficient result evidence.
- `pipeline_smoke_only`: import/forward/eval plumbing check only.
- `checkpoint_unavailable`: no trustworthy checkpoint exists locally or online.
- `protocol_mismatch`: local evaluator/config/inference differs from official
  protocol.
- `deferred_reproduction`: stop because required evidence is missing.

Do not choose a success-oriented label until the required evidence is present.

## Step 2: Collect Official Evidence

Fill `reproduction/baselines/<method>/issue.md`:

- paper title, table, and metric
- official repository URL and commit/tag
- official config and inference command
- official evaluator or metric implementation
- expected dataset split and task setting

If any item is unknown, write `unknown` and continue the audit. Do not fill gaps
with assumptions.

## Step 3: Verify the Checkpoint

Fill `checkpoint_manifest.md`:

- source URL or local path
- owner/source provenance
- SHA256
- expected architecture
- expected dataset/task
- expected config
- missing/unexpected/remapped keys after loading

Verdict rules:

- `trusted`: source, checksum, architecture, config, and load report are
  compatible.
- `partial`: some keys/configs map, but compatibility is not fully proven.
- `rejected`: architecture/config/provenance conflicts with the target method.
- `unavailable`: no trustworthy checkpoint exists.

Partial remapping is not enough for a method result claim.

## Step 4: Align the Evaluation Protocol

Fill `eval_protocol.md` by comparing official and local settings:

- dataset split
- object and predicate vocabularies
- preprocessing
- task mode, such as PredCls, SGCls, or SGDet
- inference inputs
- top-k and constraint settings
- evaluator semantics
- metric keys

If the evaluator differs, record `protocol_mismatch` and stop at audit unless
the PR fixes the mismatch.

## Step 5: Decide Whether to Run Metrics

Run metrics only when all required alignment evidence is present.

Allowed before full metric evaluation:

- import check
- config parse check
- checkpoint load inspection
- one-batch shape smoke labeled as `pipeline_smoke_only`

Not allowed as method evidence:

- random initialization metrics
- all-zero tiny-slice metrics
- partial checkpoint remap metrics
- output from a local approximate adapter without official-code parity

## Step 6: Record an ADR for Non-Obvious Decisions

Create an ADR from `reproduction/templates/decision_record.md` when:

- accepting an OpenSGG implementation as official-compatible
- rejecting an OpenSGG implementation as approximate
- trusting, rejecting, or deferring a checkpoint
- finding evaluator semantics mismatch
- deciding that a baseline must be deferred

Example:

```bash
cp reproduction/templates/decision_record.md \
  reproduction/adr/0003-<method>-checkpoint-verdict.md
```

Link the ADR from the baseline issue under `Decision Records`.

## Step 7: Write the Baseline Status

Update `status.md` with:

- current outcome label
- evidence table
- current blocker
- next action

If the method is deferred, say exactly what is missing:

- official checkpoint
- official config
- official evaluator
- class/vocabulary mapping
- compatible inference procedure

This status file is the source of truth for future agents.

## Step 8: Decide Whether a PR Is Appropriate

A PR is appropriate if it does one of these:

- fixes code to match the paper or official repository
- adds checkpoint-backed evaluation with provenance
- documents protocol alignment or mismatch
- adds an audit/deferred status report
- adds a guardrail against misleading claims

A PR is not appropriate as a method-result PR when the only evidence is a smoke
run, random initialization, partial checkpoint remap, or tiny-slice zero metric.

Use `reproduction/templates/pr_checklist.md` in the PR description or as a
tracked checklist file.

## Step 9: Run the Claim Guardrail

Before opening a PR, scan claim-bearing files:

```bash
python tools/reproduction/check_reproduction_claims.py \
  --changed-from origin/main \
  AGENTS.md reproduction <pr-body-or-release-note-files>
```

The script is intentionally limited. It catches obvious misleading wording, but
it does not prove that the evidence is sufficient.

## Common Scenarios

### Official Checkpoint Missing

Outcome: `checkpoint_unavailable` or `deferred_reproduction`.

Allowed output: checkpoint search notes, source audit, and a gap report.

Do not report random-init metrics as method evidence.

### Checkpoint Loads With Many Missing Keys

Outcome: usually `implementation_audit` or `deferred_reproduction`.

Record the load report in `checkpoint_manifest.md`. Trust the checkpoint only if
the missing/unexpected keys are explained and compatible with the official
method.

### OpenSGG Has a Similar Method Name

Outcome: `implementation_audit` until official-code parity is shown.

Compare modules, losses, relation heads, data preprocessing, inference behavior,
and evaluator assumptions. Same name is not enough.

### Evaluator Semantics Differ

Outcome: `protocol_mismatch`.

Open a PR only if it documents the mismatch or fixes evaluator alignment without
changing labels, ground truth, or metric semantics to improve numbers.

### Tiny Slice Is Needed for Debugging

Outcome: `pipeline_smoke_only`.

Write that the tiny slice proves plumbing only. Do not compare it to paper
numbers.

## Reviewer Checklist

Reviewers should ask:

- Is the claim label accurate?
- Are official sources cited?
- Is checkpoint provenance recorded?
- Does the config match the official protocol?
- Does inference match the official protocol?
- Does evaluator semantics match the official protocol?
- Are random-init or tiny-slice outputs clearly labeled as smoke only?
- Are all deviations recorded?
- Does the PR avoid changing labels, ground truth, or evaluator semantics to
  improve numbers?

If any answer is missing, request an audit/deferred status instead of accepting
a method-result claim.
