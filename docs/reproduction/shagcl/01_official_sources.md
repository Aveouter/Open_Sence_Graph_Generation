# SHA-GCL Official Sources

Primary source:

- `Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene
  Graph Generation`, Dong et al., CVPR 2022.
- Official repository: `https://github.com/dongxingning/SHA-GCL-for-SGG`
- Local checkout: `/workspace/external/shagcl_official/SHA-GCL-for-SGG`
- Commit inspected: `8acfb818a0b2a88f9dd4a8ed64591ef856e66bf5`

Implementation interpretation:

- Official VG PredCls path uses `TransLike_GCL`, `Hybrid-Attention`,
  `divide4`, and `KL_logit_TopDown`.
- Official GCL includes grouped auxiliary classifiers, `FrequencyBias_GCL`, and
  KL-logit knowledge transfer.

Checkpoint status:

- Official README lists a OneDrive link for `SHA_GCL_VG_PredCls`.
- Other trained models from the paper require contacting the authors.
- No local trusted SHA-GCL checkpoint was found in
  `outputs/pretrained/shagcl_official`.
- No paper-number alignment is claimed.
