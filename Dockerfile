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

# Install unires package and its dependencies
RUN pip install --upgrade pip setuptools wheel cython
RUN pip install numpy==1.26.0
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV NI_COMPILED_BACKEND=C
RUN pip install --no-build-isolation .

# Set the default command to unires CLI tool (as installed by console_scripts)
ENTRYPOINT ["unires"]