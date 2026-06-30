# EGTR Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#67](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/67), branch `features-codex/repro-egtr` -> `feat/analysis-tools`)

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- Local archive: `outputs/pretrained/egtr/egtr_vg.tar.gz`.
- Archive SHA256: `39e9f9ee9ff755ad4233f68a30333f8da756db3f2d95a236e3e1b5e142d5d2fe`.
- Extracted checkpoint: `outputs/pretrained/egtr/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/epoch=03-validation_loss=1.71.ckpt`.
- Checkpoint SHA256: `8926640facecd2784160c221c048dd4209df7d623ceb3c9eda289a11cb1f84d5`.
- Synthetic forward smoke passed under the EGTR wrapper.
- Checkpoint-backed GT-aligned relation JSONL export passed on 2 rows.
- Hidden/GT-aligned predicate recall: R@1=0.0, R@5=0.5, R@10=1.0; mR@1=0.0, mR@5=0.5, mR@10=1.0.
- Direct Motifs-style PredCls metric probing is not claimed for EGTR because EGTR runs SGDet query inference and requires EGTR-formatted targets for its training loss path.

Next action: review draft PR [#67](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/67) and merge when ready.
