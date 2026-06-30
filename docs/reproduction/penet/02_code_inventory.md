# PENet Code Inventory

OpenSGG files:

- `configs/VisualGenome/PE_NET.py`
  - PENet VisualGenome config.
- `src/methods/penet_method.py`
  - Extends `Motifs_Method` and builds `build_penet`.
- `src/models/penet.py`
  - Prototype/semantic embedding components, visual-semantic gates, predicate
    classifier, and frequency bias.
- `src/methods/__init__.py`
  - Registers method key `penet`.
- `tools/analysis/export_relation_predictions.py`
  - Used with explicit `--allow_random_init` fallback.

Checkpoint inventory:

- Search of `outputs/pretrained` found no PENet checkpoint.
