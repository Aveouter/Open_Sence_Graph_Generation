# SHA-GCL Final Reproduction Report

## 1. 方法简介

SHA-GCL uses stacked hybrid attention and group collaborative learning for
unbiased scene graph relation prediction.

## 2. 官方来源

Primary source: `Stacked Hybrid-Attention and Group Collaborative Learning for
Unbiased Scene Graph Generation`, Dong et al., CVPR 2022.

## 3. OpenSGG 集成状态

OpenSGG contains SHA-GCL in `src/models/shagcl.py`,
`src/methods/shagcl_method.py`, and `configs/VisualGenome/SHA_GCL.py`.

## 4. Standard SGG Metrics

Random-init fallback PredCls metric slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

## 5. Paper-Number Alignment

Not claimed. No verified checkpoint was available.

## 6. Hidden-Positive Metrics

Random-init fallback JSONL export passed validation on 2 rows. R@1/R@5/R@10
and mR@1/mR@5/mR@10 were 0.0 / 0.0 / 1.0.

## 7. Deviations

No checkpoint; random-init fallback; tiny CPU slice; local hidden-eval tooling;
GitHub PR blocked by remote/proxy tooling.

## 8. 当前可信 Claim

SHA-GCL can be instantiated and tested under `conda hsg`; its outputs work with
the standard metric and hidden/GT-aligned JSONL paths.

## 9. 不能声称的内容

Do not claim paper-level SHA-GCL reproduction numbers or checkpoint-backed
performance.

## 10. 对本项目的作用

SHA-GCL is available as a documented P1 baseline with no-training fallback test
evidence.

## 11. 下一步建议

Proceed to P1 RA-SGG using checkpoint-first, test-only policy and disambiguate
the local `REACT` method from canonical RA-SGG.
