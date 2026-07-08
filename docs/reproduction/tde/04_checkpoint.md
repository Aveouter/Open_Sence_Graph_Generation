# TDE Reproduction: Checkpoint Status

Last updated: 2026-06-22.

## Requirement

Strict reproduction must use official checkpoints. A checkpoint is not accepted
until the file path, source URL, file size, and SHA256 are recorded.

## Official Checkpoint Sources

Official README provides causal MOTIFS-SUM checkpoints:

| Protocol | URL | Status |
|---|---|---|
| SGDet | https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602 | pending download |
| SGCls | https://1drv.ms/u/s!AmRLLNf6bzcir9xyuLO_I8TSZ6kfyQ?e=Y5686s | pending download |
| PredCls | https://1drv.ms/u/s!AmRLLNf6bzcir9xx725wYjN7lytynA?e=0B65Ws | pending download |

Alternate official bundle mirrors:

- Baidu: https://pan.baidu.com/s/1oyPQBDHXMQ5Tsl0jy5OzgA, extraction code `1234`
- Weiyun: https://share.weiyun.com/ViTWrFxG

## Current Local Checkpoints

The current OpenSGG workspace contains several pretrained files, but none has
yet been verified as an official TDE checkpoint:

| Path | SHA256 | Inspection result | Use for TDE? |
|---|---|---|---|
| `outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth` | `7c4fda21f779e6947d042043567e682c2e2e058c2ee49e63dcebb3c65741f551` | SGB-style Motifs checkpoint; has `context_layer.untreated_*` and `freq_bias`, but no `avg_post_ctx`, `untreated_feat`, `vis_compress`, or `ctx_compress` | not accepted as official TDE |
| `outputs/pretrained/other/checkpoint0149.pth` | `2b3601e5cc5d835d9538d78b698dd7b3f4f1646bdb0e5c459d2cbe959d5906b1` | Transformer-style keys; no `roi_heads.relation.predictor`, `untreated`, `avg_post_ctx`, `vis_compress`, `ctx_compress`, or `freq_bias` | not TDE |
| `outputs/pretrained/reltr/reltr_vg.pth` | not computed in this phase | RelTR checkpoint | not TDE |
| `outputs/pretrained/egtr/egtr_vg.tar.gz` | not computed in this phase | EGTR archive | not TDE |

Local inspection commands:

```bash
sha256sum outputs/pretrained/other/checkpoint0149.pth \
  outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth

python - <<'PY'
import argparse
import torch
from torch.serialization import safe_globals

for path in [
    "outputs/pretrained/other/checkpoint0149.pth",
    "outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth",
]:
    with safe_globals([argparse.Namespace]):
        ckpt = torch.load(path, map_location="cpu")
    model = ckpt.get("model", ckpt.get("state_dict", ckpt))
    keys = list(model.keys())
    for pattern in [
        "roi_heads.relation.predictor",
        "untreated",
        "avg_post_ctx",
        "vis_compress",
        "ctx_compress",
        "freq_bias",
    ]:
        print(path, pattern, sum(pattern in key for key in keys))
PY
```

## Download Target Layout

Use a separate official-checkpoint directory:

```text
outputs/pretrained/tde_official/
  sgdet/
  sgcls/
  predcls/
```

Expected records after download:

```text
outputs/pretrained/tde_official/<protocol>/<downloaded-file>
outputs/pretrained/tde_official/<protocol>/SHA256SUMS.txt
outputs/pretrained/tde_official/<protocol>/download_url.txt
```

## Download Procedure

OneDrive shared links may require browser or scripted URL resolution. Do not
silently substitute a non-official mirror. If OneDrive fails, use the official
Baidu/Weiyun alternate links and record the exact source.

Observed on 2026-06-22 from this workspace:

- `curl -L -I` reaches each `1drv.ms` URL and receives a `301 Moved Permanently`
  redirect to `onedrive.live.com`.
- The follow-up TLS handshake fails with
  `SSL routines::unexpected eof while reading`.
- Python `urllib.request` fails with the same SSL EOF.
- Adding `download=1` to the shared link still redirects to `onedrive.live.com`
  and fails at the same TLS step.
- The workspace proxy variables point to `http://127.0.0.1:7890`.
- `curl --noproxy '*'` can reach the first `1drv.ms` redirect, but direct
  connection to `onedrive.live.com:443` times out.
- Microsoft Graph public-share forms return authentication failures rather than
  public download content in this environment.
- Direct OneDrive download URLs reconstructed from observed `resid` values also
  hit the same `onedrive.live.com` TLS EOF path.
- A PredCls probe with `curl -L -k --http1.1 --max-time 25 -A 'Mozilla/5.0'`
  still fails before payload download with `curl: (35)` and
  `http_code=000`.

Observed direct resource IDs from OneDrive redirects:

| Protocol | Direct resource ID |
|---|---|
| PredCls | `22376FFAD72C4B64!781937` |

Latest PredCls probe output:

```text
effective_url=https://onedrive.live.com/:u:/g/personal/22376FFAD72C4B64/s!AmRLLNf6bzcir9xx725wYjN7lytynA?resid=22376FFAD72C4B64!781937&e=0B65Ws&migratedtospo=true&redeem=...
errormsg=error:0A000126:SSL routines::unexpected eof while reading
size_download=0
```
| SGCls | `22376FFAD72C4B64!781938` |
| SGDet | `22376FFAD72C4B64!781947` |

This means the links are still structured as live OneDrive share URLs, but
command-line download is not yet resolved in this environment. Next attempts
should use one of:

1. Browser download, then copy the resulting files into
   `outputs/pretrained/tde_official/<protocol>/`.
2. A OneDrive share-link resolver that extracts the authenticated direct
   download URL.
3. Official Baidu/Weiyun mirrors, recording the exact mirror and file names.

Do not use unofficial Google Drive, Gitee, or third-party model mirrors unless
their files are cryptographically verified against an official checkpoint. Such
files may be useful clues for debugging but do not satisfy the official
checkpoint requirement.

For each checkpoint:

```bash
mkdir -p outputs/pretrained/tde_official/predcls
# Download official file into this directory.
ls -lh outputs/pretrained/tde_official/predcls
sha256sum outputs/pretrained/tde_official/predcls/* > outputs/pretrained/tde_official/predcls/SHA256SUMS.txt
```

## Checkpoint Inspection Procedure

After download, inspect keys without modifying the file:

```bash
conda activate tde_official
python - <<'PY'
import torch
path = "outputs/pretrained/tde_official/predcls/model_final.pth"
ckpt = torch.load(path, map_location="cpu")
print(type(ckpt))
if isinstance(ckpt, dict):
    print(ckpt.keys())
    model = ckpt.get("model", ckpt)
    keys = list(model.keys())
    print("num_keys", len(keys))
    for key in keys[:50]:
        print(key)
    required = [
        "roi_heads.relation.predictor.avg_post_ctx",
        "roi_heads.relation.predictor.untreated_feat",
        "roi_heads.relation.predictor.context_layer.untreated_dcd_feat",
        "roi_heads.relation.predictor.context_layer.untreated_obj_feat",
        "roi_heads.relation.predictor.context_layer.untreated_edg_feat",
    ]
    for key in required:
        print(key, key in model)
PY
```

Required evidence:

- Predictor keys match `CausalAnalysisPredictor`, not `MotifPredictor`.
- Untreated moving-average buffers are present.
- Config/protocol is recoverable from checkpoint directory or run command.

## Acceptance Criteria

Checkpoint phase is complete only when this table is filled:

| Protocol | Path | Size | SHA256 | Download URL | Official predictor verified |
|---|---|---:|---|---|---|
| SGDet | TBD | TBD | TBD | official URL | no |
| SGCls | TBD | TBD | TBD | official URL | no |
| PredCls | TBD | TBD | TBD | official URL | no |

## Current Status

Pending. The official checkpoint files have not yet been downloaded or hashed in
this workspace.
