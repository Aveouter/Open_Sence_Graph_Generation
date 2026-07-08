# TDE Reproduction: Environment Plan

Last updated: 2026-06-22.

## Requirement

Official TDE reproduction must not pollute the current OpenSGG environment. The
official code is based on the older `maskrcnn-benchmark` stack and should be
run in an isolated environment.

## Current OpenSGG Project Environment Snapshot

Captured in `/workspace/Item_code/OpenSGG` with the user-specified `hsg`
environment:

```text
Conda env: hsg
Python: 3.10.8
Python executable: /root/miniconda3/envs/hsg/bin/python
Platform: Linux-3.10.0-693.el7.x86_64-x86_64-with-glibc2.35
Torch: 2.5.1+cu121
torchvision: 0.20.1+cu121
Torch CUDA runtime: 12.1
CUDA available: True
GPU: NVIDIA A100-PCIE-40GB
yacs: import ok
OpenCV: 4.13.0
tqdm: 4.67.3
h5py: 3.12.1
numpy: 2.2.6
matplotlib: missing in hsg
```

This environment should be used for current OpenSGG commands. It is not a
strict official TDE reproduction environment because the official repository is
an older `maskrcnn-benchmark` codebase targeting Python 3.7, PyTorch 1.4, and
CUDA 10.1.

## Official Environment Target

From official `INSTALL.md`:

```text
Python <= 3.8
PyTorch >= 1.2; official author environment: torch 1.4.0 + CUDA 10.1
torchvision >= 0.4; official author environment: torchvision 0.5.0 + CUDA 10.1
cocoapi
yacs
matplotlib
GCC >= 4.9
OpenCV
apex
```

## Proposed Isolated Environment

Preferred exact-match conda environment:

```bash
conda create -n tde_official python=3.7 -y
conda activate tde_official

conda install -y ipython scipy h5py
pip install ninja yacs cython matplotlib tqdm opencv-python overrides
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

## Isolated Environment Attempt Log

Current isolated env state:

```text
Conda env: tde_official
Python: 3.7.16
PyTorch: not installed
torchvision: not installed
cudatoolkit: not installed
cudnn: not installed
```

Commands already attempted:

```bash
conda create -n tde_official python=3.7 -y
conda run -n tde_official python --version
```

Result:

```text
Python 3.7.16
```

The exact official install command failed because current conda channel
metadata no longer exposes `torchvision==0.5.0` through the requested solver:

```bash
conda install -n tde_official -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

Observed failure:

```text
PackagesNotFoundError: torchvision==0.5.0
```

An attempted fallback install of `pytorch==1.4.0 cudatoolkit=10.1` began
downloading packages but was intentionally interrupted after the user clarified
that the active project environment is `hsg`. No partial PyTorch/CUDA packages
are installed in `tde_official` according to:

```bash
conda list -n tde_official | rg '^(python|pytorch|torchvision|cudatoolkit|cudnn|numpy|mkl)\s'
```

Current output:

```text
python                     3.7.16           h7a1cb2a_0
```

Recommended next environment attempt, if continuing official-repo execution on
this machine:

```bash
conda install -n tde_official -y pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
conda run -n tde_official pip install --no-deps torchvision==0.5.0
```

This keeps `hsg` untouched while using the pip wheel only for torchvision after
the matching PyTorch package is installed.

Install cocoapi:

```bash
mkdir -p /tmp/tde_deps
cd /tmp/tde_deps
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

Install Apex matched to older PyTorch:

```bash
cd /tmp/tde_deps
git clone https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac
python setup.py install --cuda_ext --cpp_ext
```

Install official Scene Graph Benchmark:

```bash
mkdir -p /workspace/external/tde_official
cd /workspace/external/tde_official
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd Scene-Graph-Benchmark.pytorch
git checkout ceb71fa88461c2a97a6258a80f47669d89207296
python setup.py build develop
```

Repository clone status:

```text
Path: /workspace/external/tde_official/Scene-Graph-Benchmark.pytorch
Commit: ceb71fa88461c2a97a6258a80f47669d89207296
Build status: not built; PyTorch 1.4/torchvision 0.5 official environment is not installed yet
```

## Feasibility Notes

- The current machine reports CUDA unavailable from the active Python
  environment. The official stack may still run on CPU for limited import and
  config checks, but full SGDet/SGCls/PredCls reproduction will likely require a
  compatible GPU runtime.
- PyTorch 1.4/CUDA 10.1 binaries may not work with the current system driver
  if the driver is too new/old relative to runtime constraints. Verify with:

```bash
conda activate tde_official
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Environment Acceptance Criteria

This phase is not complete until all checks pass inside `tde_official`:

```bash
python --version
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda)"
python -c "import yacs, cv2, matplotlib, tqdm, h5py; print('deps ok')"
python -c "import maskrcnn_benchmark; print(maskrcnn_benchmark.__file__)"
python -c "from maskrcnn_benchmark.config import cfg; print('cfg ok')"
```

For GPU reproduction:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## Non-Pollution Rule

Do not install official TDE dependencies into the current OpenSGG environment.
Use one of:

- `conda activate tde_official`
- a Docker image pinned to Python 3.7, PyTorch 1.4, CUDA 10.1
- an external machine with the same environment recorded in this file
