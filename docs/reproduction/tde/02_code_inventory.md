# TDE Code Inventory

Existing official-source notes:

- `01_official_sources.md`
- `02_environment.md`
- `03_code_analysis.md`
- `04_checkpoint.md`
- `05_mapping.md`
- `06_deviations.md`

OpenSGG implementation files:

- `configs/VisualGenome/TDE.py`
  - PredCls config exposing `tde_effect_type`, `tde_fusion_type`,
    `tde_spatial_for_vision`, and `tde_average_ratio`.
- `src/methods/tde_method.py`
  - `TDE_Method` extends `Motifs_Method`.
  - `TDECriterion` adds auxiliary branch losses emitted by the model.
- `src/models/motifs.py`
  - `TDEModel` implements an official-style causal predictor.
  - Includes `ctx_compress`, `vis_compress`, `PairFrequencyBias`,
    `index_with_probability`, `calculate_logits`, untreated buffers, and
    effect modes `none`, `TDE`, `NIE`, `TE`.
- `tools/analysis/export_relation_predictions.py`
  - Relation JSONL export now supports explicit `--allow_random_init` fallback.
- `src/core/metrics.py`
  - Standard PredCls metric path used for validation.

Checkpoint inventory:

- Official TDE checkpoints are documented in `04_checkpoint.md` but not
  downloaded/verified in this workspace.
- Local Motifs checkpoint is not accepted as an official TDE checkpoint.
- Local `outputs/pretrained/other/checkpoint0149.pth` is not TDE.
