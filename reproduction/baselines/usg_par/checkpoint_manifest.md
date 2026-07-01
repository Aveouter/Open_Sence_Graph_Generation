# Checkpoint Manifest: USG-Par

## Identity

- File/path: unavailable
- Source URL: unavailable
- Source owner: unavailable
- Download command: unavailable
- Download date: unavailable
- SHA256: unavailable
- File size: unavailable

## Expected Compatibility

- Method: USG-Par
- Architecture: official USG-Par image path
- Dataset: official PSG unless a Visual Genome checkpoint is documented
- Task: SGDet
- Config: `configs/psg.yaml` plus documented dataset adaptation
- Number of object classes: dataset-specific; local VG path uses 150 foreground + no-object
- Number of predicate classes: local USG path uses 50 foreground predicates

## Load Report

- Loader command: not run
- Missing keys: not assessed
- Unexpected keys: not assessed
- Shape mismatches: not assessed
- Remapped keys: none
- Verdict: rejected

## Notes

No checkpoint-backed metric can be claimed from the current local implementation.
