# FREQ Code Inventory

Relevant existing code before integration:

- `src/models/motifs.py`
  - `PairFrequencyBias`: pair-conditioned predicate prior table.
  - `generate_object_pairs`: directed object-pair enumeration.
  - `FrequencyBias`: legacy marginal prior used by older local baselines.
- `src/methods/motifs_method.py`
  - `MotifsCriterion`: maps VG predicate IDs to model logit layout and computes
    relation cross entropy.
- `tools/analysis/export_relation_predictions.py`
  - Motifs-style relation JSONL export schema and validator.
- `tools/analysis/compute_predicate_recall_from_jsonl.py`
  - GT-aligned predicate recall and mean recall from JSONL exports.
- `src/core/metrics.py`
  - standard project metric path for PredCls recall and mean recall.

Missing before integration:

- No registered `freq` method.
- No VisualGenome `FREQ.py` config.
- CLI parser and train config alias did not accept `freq`.
- Relation exporter required checkpoints and could not run deterministic FREQ.
