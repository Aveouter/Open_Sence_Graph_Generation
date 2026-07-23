# Docker Environment

This guide sets up an OpenSGG development container for smoke tests, training,
and evaluation. It does not provide datasets, checkpoints, or reproduction
evidence by itself; mount those assets explicitly and keep reproduction claims
within the repository's evidence standard.

## Requirements

- Docker Engine with BuildKit enabled.
- NVIDIA Container Toolkit for GPU runs.
- A host NVIDIA driver compatible with the CUDA image used by the Dockerfile.

The default image uses CUDA 12.1, Python from `environment.yml`, and PyTorch
`2.5.1`/torchvision `0.20.1` from the CUDA 12.1 wheel index. This matches the
known-good local container used for OpenSGG development at the time this guide
was added: Ubuntu 22.04, Python 3.10.8, PyTorch `2.5.1+cu121`, torchvision
`0.20.1+cu121`, NVIDIA driver `550.163.01`.

## Build

```bash
docker build -t opensgg:dev .
```

By default the Dockerfile rewrites the conda channel list to `conda-forge` and
`defaults` inside the build, which avoids depending on a region-specific mirror.
To keep the channels exactly as written in `environment.yml`, run:

```bash
docker build \
  --build-arg CONDA_CHANNEL_MODE=repo \
  -t opensgg:dev .
```

To target a different PyTorch wheel index or version:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  --build-arg TORCH_VERSION=2.5.1 \
  --build-arg TORCHVISION_VERSION=0.20.1 \
  -t opensgg:cu118 .
```

## Run

Use Docker directly:

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -v "$PWD":/workspace/OpenSGG \
  -v "$PWD/data":/workspace/OpenSGG/data \
  -v "$PWD/outputs":/workspace/OpenSGG/outputs \
  opensgg:dev bash
```

Or use Compose:

```bash
docker compose run --rm opensgg bash
```

For GPU access with Compose, add the GPU override:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm opensgg bash
```

Inside the container, run Python through the `hsg` conda environment:

```bash
conda run --no-capture-output -n hsg python train.py --help
conda run --no-capture-output -n hsg python train.py \
  --method EGTR --dataname VisualGenome --gpus 0 --batch_size 4
```

## Data And Outputs

The `.dockerignore` file excludes `data/`, `checkpoints/`, `outputs/`, local
reports, and crash dumps from the build context. Keep large assets on the host
and mount them into the container:

```bash
OPENSGG_DATA_DIR=/path/to/VisualGenome-root \
OPENSGG_OUTPUTS_DIR=/path/to/opensgg-outputs \
docker compose run --rm opensgg bash
```

Expected Visual Genome layout is the same as the main README:

```text
data/VisualGenome/
  images/
  train.json
  val.json
  test.json
  rel.json
```

## Smoke Check

After building, a lightweight import check is:

```bash
docker run --rm opensgg:dev \
  conda run --no-capture-output -n hsg python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

For a GPU sanity check, expose one device first:

```bash
docker run --rm --gpus '"device=0"' opensgg:dev \
  conda run --no-capture-output -n hsg python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

This only verifies the container runtime. It is not baseline reproduction
evidence and should not be reported as model performance.
