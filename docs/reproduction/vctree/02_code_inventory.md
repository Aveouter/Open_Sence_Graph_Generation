# VCTree Code Inventory

OpenSGG files:

- `configs/VisualGenome/VCTree.py`
  - VCTree VisualGenome config.
- `src/methods/vctree_method.py`
  - Extends `Motifs_Method` and swaps model construction to `build_vctree`.
- `src/models/vctree.py`
  - Binary TreeLSTM, tree construction, context encoding, and predicate head.
- `src/methods/__init__.py`
  - Registers method key `vctree`.
- `tools/analysis/export_relation_predictions.py`
  - Used with explicit `--allow_random_init` fallback.

Checkpoint inventory:

- Search of `outputs/pretrained` found Motifs, RelTR, EGTR, and miscellaneous
  checkpoints, but no VCTree checkpoint.
