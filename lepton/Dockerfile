# syntax=docker/dockerfile:1.7
# UniRes on an NVIDIA Lepton dev pod (8xA100-80GB).
#
# Built on the VALIDATED UniRes recipe (see repo-root Dockerfile): CUDA 12.6-devel
# base + torch 2.6.0/cu126 + the COMPILED nitorch backend, so nitorch's fragile
# pinned 2021 commit compiles reliably (matching nvcc <-> torch CUDA). Adds the
# DL-prior deps (deepinv, torch-pinned), a baked training stack, and Claude Code +
# dev tooling for the on-pod workflow. NO Docker-in-Docker (UniRes is plain Python).
#
# The `unires` package itself is NOT baked -- the repo (branch dl-prior-red) is
# pulled + editable-installed at runtime on the pod; only the heavy, immutable
# environment (torch, compiled nitorch, deepinv, training stack) is baked here.
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
# Claude Code "brains" + toolchain caches live on the persistent /workspace (NFS)
# mount so they survive pod restarts.
ENV CLAUDE_CONFIG_DIR=/workspace/.claude \
    UV_CACHE_DIR=/workspace/.cache/uv \
    UV_LINK_MODE=copy \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCH_HOME=/workspace/.cache/torch \
    TOKENIZERS_PARALLELISM=false

# --------------------------------------------------------------------------- #
# System deps + dev tools (main + universe). bubblewrap is used by Claude Code's
# sandbox-runtime; python3.10 matches the validated recipe.
# --------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository universe && apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git git-lfs wget unzip tmux vim htop rsync \
      build-essential pkg-config \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      bubblewrap && \
    git lfs install --system && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Node 20 + Claude Code (+ sandbox runtime). apply-seccomp lives in one of the
# two @anthropic-ai packages depending on version; chmod it wherever it landed.
# --------------------------------------------------------------------------- #
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @anthropic-ai/claude-code @anthropic-ai/sandbox-runtime && \
    find /usr/lib/node_modules/@anthropic-ai -type f -name "apply-seccomp" -exec chmod +x {} \; && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# uv (fast installer) + ripgrep (Claude Code sandbox-runtime hard-requires `rg`).
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
RUN curl -fsSL -o /tmp/rg.tar.gz \
      https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf /tmp/rg.tar.gz -C /tmp && \
    install -m 0755 /tmp/ripgrep-14.1.1-x86_64-unknown-linux-musl/rg /usr/local/bin/rg && \
    rm -rf /tmp/rg.tar.gz /tmp/ripgrep-14.1.1-x86_64-unknown-linux-musl

# =========================================================================== #
# UniRes environment in an isolated venv. Mirrors the validated root Dockerfile:
# torch 2.6.0/cu126 -> build pins -> COMPILED nitorch -> deepinv (torch-pinned).
# =========================================================================== #
ENV PATH="/opt/unires-venv/bin:${PATH}"
RUN python3.10 -m venv /opt/unires-venv && pip install --upgrade pip

# PyTorch first (CUDA 12.6 wheels), before the fragile compiled installs.
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Build pins: nitorch's pinned 2021 commit calls the old positional distutils
# Compiler(None, dry_run, force) signature that setuptools 81 removed; cython<3
# keeps the same working window. (Same reasoning as the repo-root Dockerfile.)
RUN pip install "setuptools>=74,<75" wheel "cython<3.0" && pip install numpy==1.26.0

# Compile the nitorch C/CUDA backend for the target arch(es).
#   8.0 = A100 (the Lepton target); 8.9 = Ada (so a local laptop build can
#   EXECUTE-verify the compiled backend, not just import it).
# Override for other GPUs, e.g. --build-arg TORCH_CUDA_ARCH_LIST=9.0 (H100).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV NI_COMPILED_BACKEND=C
# Parallelise the nitorch C/CUDA compile (many kernels x2 arches). Sized for the
# build host's RAM, not core count -- CUDA compile jobs are memory-heavy.
ENV MAX_JOBS=8
RUN pip install --no-build-isolation \
      "nitorch[all]@git+https://github.com/balbasty/nitorch#8067d60542642a39ab6c6eb5e1157373a9d3dcc3"

# deepinv (DRUNet denoiser prior). A constraints file forbids any torch/torchvision
# change while resolving deepinv -- a naive `pip install deepinv` otherwise drags in
# a newer torchvision that upgrades torch and BREAKS the compiled nitorch backend.
RUN printf 'torch==2.6.0\ntorchvision==0.21.0\n' > /tmp/uc.txt && \
    pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install deepinv -c /tmp/uc.txt && \
    python -c "import torch, deepinv; assert torch.__version__.startswith('2.6.0'), torch.__version__; print('deepinv OK, torch', torch.__version__)" && \
    rm -f /tmp/uc.txt

# Baked training stack (denoiser training) + demo/benchmark convenience.
# Same constraint so torch stays pinned at 2.6.0.
RUN printf 'torch==2.6.0\ntorchvision==0.21.0\n' > /tmp/uc.txt && \
    pip install monai h5py tensorboard nibabel matplotlib -c /tmp/uc.txt && \
    python -c "import torch; assert torch.__version__.startswith('2.6.0'), torch.__version__" && \
    rm -f /tmp/uc.txt

# NOTE: `unires` is intentionally NOT installed here. On the pod, pull branch
# dl-prior-red and install editable, reusing this baked (compiled) nitorch:
#   NI_COMPILED_BACKEND=C pip install -e . --no-deps --no-build-isolation
# `--no-deps` stops pip from re-cloning/recompiling the nitorch git pin in setup.py.

# --------------------------------------------------------------------------- #
# Dev-pod essentials (no DinD).
# --------------------------------------------------------------------------- #
WORKDIR /workspace
RUN git config --global user.email "mbrudfors@nvidia.com" && \
    git config --global user.name "Mikael Brudfors" && \
    git config --global core.editor vim
# /dev/shm is tens of GB at runtime (great for pip/tmp); restore Claude settings
# from the persistent mount; convenience tmux alias.
RUN echo 'export TMPDIR=/dev/shm' >> /root/.bashrc && \
    echo 'mkdir -p /workspace/.claude' >> /root/.bashrc && \
    echo '[ -f /workspace/.claude/restore-settings.sh ] && source /workspace/.claude/restore-settings.sh' >> /root/.bashrc && \
    echo 'alias claude-dev="tmux attach -t claude || tmux new -s claude -c /workspace \"claude\""' >> /root/.bashrc

COPY --chmod=0755 <<'ENTRY' /usr/local/bin/entrypoint.sh
#!/bin/bash
# Non-interactive workloads (exec'd without a login shell) also get these.
export TMPDIR=/dev/shm
mkdir -p /workspace/.claude
[ -f /workspace/.claude/restore-settings.sh ] && source /workspace/.claude/restore-settings.sh
exec "$@"
ENTRY

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
