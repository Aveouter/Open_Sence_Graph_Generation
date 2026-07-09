# Checkpoint Manifest: PENet SGDet

## Identity

- File/path: `outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth`
- Source URL: official Google Drive id `1Ed6PkATiig0xpFuQYL-G5trFifhPpc0C`
- Source owner: VL-Group/PENET
- Download command: manual download, recorded by local file provenance
- Download date: local file timestamp `2023-04-02 09:15:21 +0000`
- SHA256: `ca7009b404f845ed989f799d1dc426be28a33db89b6b8d16649309d338948183`
- File size: `2627548389`

## Expected Compatibility

- Method: PENet / `PrototypeEmbeddingNetwork`
- Architecture: ResNeXt-101-32x8d FPN detector plus PE-NET relation predictor
- Dataset: Visual Genome
- Task: SGDet
- Config: `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`
- Number of object classes: 151 including background
- Number of predicate classes: 51 including background

## Load Report

- Loader command: `train.py --test --method PENet --config_file configs/VisualGenome/PE_NET.py --ckpt_path .../PE-NET_SGDet/model_final.pth`
- Missing keys: not accepted for relation/detector extractor parity tests
- Unexpected keys: ignored only when outside the local adapter scope
- Shape mismatches: not accepted for mapped tensors
- Remapped keys:
  - `backbone.body.*` to local backbone
  - `fpn.*` to local FPN
  - `roi_heads.relation.box_feature_extractor.*` to `relation_box_extractor`
  - `roi_heads.box.feature_extractor.*` to `detector_box_extractor`
  - `rpn.head.*` to local SGDet proposal generator
  - `roi_heads.box.predictor.*` to local detector box predictor
  - `roi_heads.relation.union_feature_extractor.*` to local union extractor
  - `roi_heads.relation.predictor.*` to local `PENetContext`
- Verdict: trusted for mapped tensors; partial for full-process reproduction until metrics align

## Notes

Tensor-level tests assert exact equality for relation and detector ROI extractor
weights after loading from the official SGDet checkpoint.
