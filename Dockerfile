FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG CONDA_CHANNEL_MODE=community

ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      libglib2.0-0 \
      libgl1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      tini \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

WORKDIR /workspace/OpenSGG

COPY environment.yml /tmp/environment.yml

RUN if [[ "${CONDA_CHANNEL_MODE}" == "community" ]]; then \
      awk 'BEGIN { skip=0 } \
        /^channels:/ { print "channels:"; print "  - conda-forge"; print "  - defaults"; skip=1; next } \
        skip && /^  - / { next } \
        skip && /^$/ { next } \
        { skip=0; print }' /tmp/environment.yml > /tmp/environment.docker.yml \
      && mv /tmp/environment.docker.yml /tmp/environment.yml; \
    fi \
    && conda env create -f /tmp/environment.yml \
    && conda run -n hsg python -m pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${PYTORCH_INDEX_URL}" \
    && conda clean -afy

COPY . .

RUN conda run -n hsg python -c 'import torch; print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")'

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["conda", "run", "--no-capture-output", "-n", "hsg", "python", "train.py", "--help"]
