FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
   build-essential \
   python3 \
   python3-dev \
   python3-pip \
   git

RUN pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install numpy nibabel==4.0.2

WORKDIR /workspace
RUN git clone -n https://github.com/balbasty/nitorch.git
WORKDIR /workspace/nitorch
RUN git checkout ad022e529a3631967f9103431231689bd12f6242
ARG MKL_SERVICE_FORCE_INTEL=1
ARG NI_COMPILED_BACKEND="C"
RUN pip install .

WORKDIR /workspace