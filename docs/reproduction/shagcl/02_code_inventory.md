# SHA-GCL Code Inventory

OpenSGG files:

- `configs/VisualGenome/SHA_GCL.py`
  - SHA-GCL VisualGenome config.
- `src/methods/shagcl_method.py`
  - Extends `Motifs_Method`, builds `build_shagcl`, and adds GCL loss handling.
- `src/models/shagcl.py`
  - Co-attention, hybrid attention stack, group prototypes, GCL loss, predicate
    classifier, and frequency bias.
- `src/methods/__init__.py`
  - Registers method key `shagcl`.
- `tools/analysis/export_relation_predictions.py`
  - Used with explicit `--allow_random_init` fallback.

Checkpoint inventory:

- Search of `outputs/pretrained` found no SHA-GCL checkpoint.
