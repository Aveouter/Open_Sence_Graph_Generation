# AAAI-27 SGG Paper Sprint Research Decision System

This folder is the execution plan for an AAAI-27 sprint on **Predicate Orthogonality Bias** and **Fine-to-Coarse Collapse** in Scene Graph Generation (SGG).

Current scope: **docs-only planning**. Do not implement experiments from this folder until a task file is selected by a separate `goal` command.

## Start Here

Read in this order:

1. `00_overall_aaai_paper_plan.md` for paper thesis, contribution shape, one-month schedule, reviewer risks, and Go / Weak-Go / No-Go.
2. `01_novelty_and_positioning_check.md` for related-work search scope and claim boundaries.
3. `03_predicate_semantic_map_construction.md` before any experiment, because every collapse metric depends on a validated semantic map.
4. `02_experiment1_fine_to_coarse_collapse.md` for the main diagnosis protocol.
5. `PR_WORKFLOW.md` before opening any implementation PR.

Existing materials to reuse:

- `../SGG_relation_phrase_generation_idea.md`
- `../F2C_VG_Experiment1_Plan.md`
- `../RSC_VG_Experiment1_Hard_Diagnosis_Plan.md`
- `../../reports/THESIS_VALIDATION_REPORT.md`
- `../../data/VisualGenome/predicate_frequencies.json`
- `../../data/VisualGenome/composition_splits/`

## Dependency Graph

```text
semantic map
-> prediction export
-> collapse metrics
-> negative controls
-> report
-> Go/No-Go decision
-> LCompo protocol
-> method design
```

Do not start method design until the Go / Weak-Go / No-Go decision has been made from validated collapse evidence.

## Task Status Table

| Task | File | Status | Blocking | Next Goal |
|---|---|---|---|---|
| 00 | `00_overall_aaai_paper_plan.md` | pending | none | `goal "Read docs/aaai2027_plan/00_overall_aaai_paper_plan.md and update only the paper sprint status table after reviewing current evidence. Do not modify code."` |
| 01 | `01_novelty_and_positioning_check.md` | pending | none | `goal "Read docs/aaai2027_plan/01_novelty_and_positioning_check.md. Build the related-work positioning matrix only. Do not implement experiments."` |
| 02 | `02_experiment1_fine_to_coarse_collapse.md` | pending | Task 03 and Task 04 | `goal "Read docs/aaai2027_plan/02_experiment1_fine_to_coarse_collapse.md. Implement only the dry-run checks needed to validate Experiment 1 inputs and output paths."` |
| 03 | `03_predicate_semantic_map_construction.md` | done | none | `goal "Read docs/aaai2027_plan/03_predicate_semantic_map_construction.md. Implement predicate semantic map construction only. Add configs/predicate_semantic_map_vg150.json and a dry-run validator. Do not modify training code. Run the validator and prepare a PR following docs/aaai2027_plan/PR_WORKFLOW.md."` |
| 04 | `04_relation_prediction_export_pipeline.md` | pending | Task 03 | `goal "Read docs/aaai2027_plan/04_relation_prediction_export_pipeline.md. Implement relation-level prediction export for Motifs PredCLS only, with dry-run and small-sample validation."` |
| 05 | `05_collapse_metrics_and_visualization.md` | pending | Task 03 and Task 04 | `goal "Read docs/aaai2027_plan/05_collapse_metrics_and_visualization.md. Implement collapse metrics and one on-family heatmap from an existing relation_predictions.jsonl."` |
| 06 | `06_lcompo_seen_unseen_protocol.md` | pending | Task 05 | `goal "Read docs/aaai2027_plan/06_lcompo_seen_unseen_protocol.md. Connect collapse metrics to the existing LCompo-SGG seen/unseen split without changing training code."` |
| 07 | `07_baseline_reproduction_plan.md` | pending | checkpoint availability | `goal "Read docs/aaai2027_plan/07_baseline_reproduction_plan.md. Reproduce the first Motifs PredCLS smoke evaluation and document exact commands and outputs."` |
| 08 | `08_method_design_placeholder.md` | pending | Go or Weak-Go decision | `goal "Read docs/aaai2027_plan/08_method_design_placeholder.md. Draft a method design memo only after Experiment 1 reaches Go or Weak-Go."` |
| 09 | `09_ablation_and_analysis_plan.md` | pending | Task 05 | `goal "Read docs/aaai2027_plan/09_ablation_and_analysis_plan.md. Implement negative controls only, no new model method."` |
| 10 | `10_paper_writing_plan.md` | pending | Go / Weak-Go decision | `goal "Read docs/aaai2027_plan/10_paper_writing_plan.md. Create the paper skeleton and fill only evidence-backed claims."` |

Status values: `pending`, `in_progress`, `done`, `blocked`, `dropped`.

## Loop-Programming Workflow

For every implementation loop:

1. Read the selected task file and `PR_WORKFLOW.md`.
2. Execute the smallest useful change.
3. Run the task acceptance commands or dry-run.
4. Generate the expected artifact under the task's declared output path.
5. Update the task status in this README or the task file.
6. Run the Definition of Done checks in `PR_WORKFLOW.md`.
7. Open one small PR, or explicitly report the work as local-only.
8. Wait for review before starting the next dependent task.

Do not report a task as PR-submitted unless the final response includes a real PR URL. Do not report README or task markdown changes as included unless they appear in the staged or committed diff.

## Execution Completion Gate

Before reporting any implementation task as complete, verify:

1. `git status -sb --ignored` was inspected.
2. The branch name matches the task.
3. `git diff --cached --name-status` contains only intended task files.
4. `git diff --cached --check` passes.
5. The task validation commands were run.
6. Generated artifacts under `outputs/`, checkpoints, datasets, caches, and local reports are not staged.
7. A real PR URL is reported, or the work is explicitly marked local-only.

The completion report must include the branch name, commit SHA if committed, PR URL if opened, actual changed files, validation commands, and generated local artifacts that were intentionally not committed.

If `docs/` is locally ignored, documentation changes are local-only unless a human explicitly asks to include them in a PR.

## First Recommended Goal Command

```text
goal "Read docs/aaai2027_plan/03_predicate_semantic_map_construction.md. Implement predicate semantic map construction only. Add configs/predicate_semantic_map_vg150.json and a dry-run validator. Do not modify training code. Run the validator and prepare a PR following docs/aaai2027_plan/PR_WORKFLOW.md."
```

## Docs-Only PR Validation Checklist

Run these commands for this documentation PR:

```bash
find docs/aaai2027_plan -maxdepth 1 -name "*.md" | wc -l
rg "## 1\\. Goal" docs/aaai2027_plan
rg "## 12\\. Notes for Future Paper Writing" docs/aaai2027_plan
rg "Go / Weak-Go / No-Go" docs/aaai2027_plan
rg "Random parent control|Frequency-matched parent control|Semantic sibling control|Subject-object prior" docs/aaai2027_plan
rg "Allowed Claims|Forbidden Claims" docs/aaai2027_plan
git diff --name-only
```

Expected result:

- The first command prints `13`.
- The heading checks find all task files from `01` through `10`.
- The Go / Weak-Go / No-Go, negative controls, and claims checks find the required planning sections.
- `git diff --name-only` shows only Markdown documentation changes, preferably only under `docs/aaai2027_plan/`.

Do not modify:

- `src/`
- `tools/`
- `scripts/`
- `configs/`
- `data/`
- `reports/`
- `train.py`

## Review Priority

First manual review should focus on:

1. `00_overall_aaai_paper_plan.md`
2. `02_experiment1_fine_to_coarse_collapse.md`
3. `03_predicate_semantic_map_construction.md`
4. `PR_WORKFLOW.md`

The semantic map task is the most important first gate. If strong / medium / weak criteria are loose, all downstream collapse analysis becomes unstable.
