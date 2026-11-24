# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

FROM nvidia/cuda:12.8.2-devel-ubuntu22.04

# System libraries and build tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libegl1 \
        libeigen3-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libvulkan1 \
        libxext6 \
        libxrender1 \
        ninja-build \
        rsync \
        swig \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# GPU configuration
COPY --from=uv /uv /usr/local/bin/uv
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

# Environment setup
ENV CUDA_HOME=/usr/local/cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    PYOPENGL_PLATFORM=egl \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    TORCH_EXTENSIONS_DIR=/tmp/torch-extensions \
    MS_ASSET_DIR=/data/.maniskill \
    VIRTUAL_ENV=/app/venv \
    UV_PROJECT_ENVIRONMENT=/app/venv \
    PATH=/app/venv/bin:$PATH

# Create the venv for the checkout supplied at runtime.
WORKDIR /app
RUN uv venv --python 3.11 "$VIRTUAL_ENV" \
    && chmod -R a+rwX "$VIRTUAL_ENV" \
    && uv cache clean

CMD ["bash"]
