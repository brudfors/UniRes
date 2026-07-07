FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3-pip git && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch first using the CUDA 12.6 wheel index
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Copy source code (Docker context must be project root)
COPY . .

# Install unires package and its dependencies.
# Pin build tools: nitorch's pinned commit calls the old positional distutils
# compiler signature Compiler(None, dry_run, force), which setuptools 81 removed
# when it reorganised the distutils compiler classes. Keep setuptools < 75 and
# cython < 3 to stay in the working window (was unpinned; broke via setuptools drift).
RUN pip install --upgrade pip && pip install "setuptools>=74,<75" wheel "cython<3.0"
RUN pip install numpy==1.26.0
# CUDA compute capability to compile nitorch kernels for; override to match your
# GPU, e.g. docker build --build-arg TORCH_CUDA_ARCH_LIST=8.0 (A100). Find yours
# with: nvidia-smi --query-gpu=compute_cap --format=csv
ARG TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV NI_COMPILED_BACKEND=C
RUN pip install --no-build-isolation .
