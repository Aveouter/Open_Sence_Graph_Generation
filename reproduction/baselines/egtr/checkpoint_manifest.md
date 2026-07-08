# Checkpoint Manifest: EGTR

## Identity

- File/path: `checkpoints/egtr/egtr_vg.ckpt`
- Resolved path:
  `outputs/pretrained/egtr/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/epoch=03-validation_loss=1.71.ckpt`
- Source archive: `outputs/pretrained/egtr/egtr_vg.tar.gz`
- Official source link recorded by upstream README: Google Drive VG trained EGTR checkpoint
- Archive SHA256: `39e9f9ee9ff755ad4233f68a30333f8da756db3f2d95a236e3e1b5e142d5d2fe`
- Checkpoint SHA256: `8926640facecd2784160c221c048dd4209df7d623ceb3c9eda289a11cb1f84d5`
- Checkpoint epoch/global step: epoch `3`, global step `3608`

## Expected Compatibility

- Method: EGTR
- Architecture: `SenseTime/deformable-detr`
- Dataset: Visual Genome
- Task: SGDet
- Object logits: 150 foreground classes, 0-indexed in EGTR
- Predicate logits: 50 foreground predicates, 0-indexed in EGTR
- Number of object queries: 200
- Image preprocessing: min size 800, max size 1333

## Load Report

- Loader command: full VG DDP evaluation command in `parity_report.md`
- Missing keys: 0
- Unexpected keys: 0
- Shape mismatches: none reported
- Remapped keys: official Lightning `model.` prefix removed by loader
- Verdict: accepted
