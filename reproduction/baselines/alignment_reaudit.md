# Original-Alignment Reaudit

The earlier baseline-suite PRs were closed after a stricter audit. The accepted
standard is now:

1. Match the original paper protocol or official repository implementation.
2. Prefer official/local checkpoints; do not train.
3. Treat random initialization, tiny-slice zero metrics, partial checkpoint
   remapping, and minimal adapters as pipeline evidence only.
4. If original alignment cannot be established, mark the method deferred rather
   than reproduced.

Closed PRs:

- FREQ: #59
- Motifs: #60
- TDE: #61
- VCTree: #62
- PENet: #63
- SHA-GCL: #64
- RA-SGG: #65
- RelTR: #66
- EGTR: #67

Current conclusion:

- No method in this tracker is currently claimed as paper-aligned reproduction.
- The previous outputs may help diagnose integration, but they must not be used
  as reproduction claims.
