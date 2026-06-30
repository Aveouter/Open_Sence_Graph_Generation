# SHA-GCL Final Reproduction Report

**Superseded by stricter alignment audit:** this report documents audit/smoke
evidence only. SHA-GCL is currently `DEFERRED_NOT_REPRODUCED` because official
inputs and protocol parity are missing.

## 1. 方法简介

SHA-GCL uses stacked hybrid attention and group collaborative learning for
unbiased scene graph relation prediction.

## 2. 官方来源

Primary source: `Stacked Hybrid-Attention and Group Collaborative Learning for
Unbiased Scene Graph Generation`, Dong et al., CVPR 2022.

Official repository: `https://github.com/dongxingning/SHA-GCL-for-SGG`,
inspected locally at commit `8acfb818a0b2a88f9dd4a8ed64591ef856e66bf5`.

## 3. OpenSGG 集成状态

OpenSGG contains SHA-GCL in `src/models/shagcl.py`,
`src/methods/shagcl_method.py`, and `configs/VisualGenome/SHA_GCL.py`.

The local adapter is not official `TransLike_GCL` parity. Official SHA-GCL
uses `Hybrid-Attention`, fixed predicate group splits, `FrequencyBias_GCL`, and
KL-logit knowledge transfer; OpenSGG uses a simplified group-prototype loss.

## 4. Standard SGG Metrics

Random-init fallback PredCls metric slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

## 5. Paper-Number Alignment

Not claimed. The official README lists a VG PredCls checkpoint link, but no
trusted local checkpoint, pretrained detector, or official VG inputs are
available in this workspace.

## 6. Hidden-Positive Metrics

Random-init fallback JSONL export passed validation on 2 rows. R@1/R@5/R@10
and mR@1/mR@5/mR@10 were 0.0 / 0.0 / 1.0.

## 7. Deviations

No official checkpoint; no official pretrained detector; missing official VG
inputs; local implementation is not official `TransLike_GCL` parity;
random-init fallback; tiny CPU slice; local hidden-eval tooling.

## 8. 当前可信 Claim

SHA-GCL can be instantiated and tested under `conda hsg`; its outputs work with
the standard metric and hidden/GT-aligned JSONL paths. The official repo and
commit are pinned, and missing reproduction inputs are recorded in
`shagcl_official_input_check.json`.

## 9. 不能声称的内容

Do not claim paper-level SHA-GCL reproduction numbers, checkpoint-backed
performance, or official implementation parity.

## 10. 对本项目的作用

SHA-GCL is available as a documented P1 baseline with implementation audit and
no-training fallback smoke evidence.

## 11. 下一步建议

Provide official VG inputs, the pretrained detector, and the trusted
`SHA_GCL_VG_PredCls` checkpoint. Then run official evaluation first and compare
OpenSGG only after config/evaluator parity is established.
