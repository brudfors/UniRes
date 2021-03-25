# Unified Super-Resolution of Neuroimaging Data on the GPU
<img style="float: right;" src="https://github.com/brudfors/UniRes/blob/master/figures/example_2.png" width="100%" height="100%"> 

This repository implements a unified model for super-resolving neuroimaging data 
(MRI and CT scans), which combines: super-resolution with a multi-channel denoising 
prior, rigid registration and a correction for interleaved slice acquisition. 
The archetype use-case is when having multiple scans of the same subject 
(e.g., T1w, T2w and FLAIR MRIs) and an analysis requires these scans to be 
represented on the same grid (i.e., having the same image size, affine matrix 
and voxel size). By default, the model reconstructs 1 mm isotropic images 
with a field-of-view that contains all input scans; however, this voxel 
size can be customised with the possibility of sub-millimetric reconstuctions. 
The model additionally supports multiple repeats of each MR sequence. 
The implementation is written in *PyTorch* and should therefore execute fast 
on the GPU. The software can be run either through **Docker** -- which ensures 
the correct library and OS versions are used, plus requires no compilation -- 
or directly by interfacing with the **Python** code. Both of these ways are 
described next. 

An installation-free demo of UniRes is available in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z6AsRLpTQxn-3myxOIKBRf6Lv2unqJ6G?usp=sharing "UniRes Colab Demo")

## 1. Python

### 1.1. Dependencies
The *NITorch* package is required to use *UniRes*; simply follow the quickstart 
guide on its GitHub page: https://github.com/balbasty/nitorch.

Next, activate a *NITorch* virtual environment and move to the *UniRes* project 
directory. Then install the package using either setuptools or pip:
``` bash
cd /path/to/unires
pip install .
``` 

### 1.2. Example use case

Running *UniRes* should be straight forward. Let's say you have three 
thick-sliced MR images: `T1.nii.gz`, `T2.nii.gz` and `PD.nii.gz`, then 
simply run `unires.py` in the terminal as:
``` bash
python unires.py T1.nii.gz T2.nii.gz PD.nii.gz
```
Three 1 mm isotropic images are written to the same folder as the input
 data, prefixed `'ur_'`. 
 
Algorithm options can be displayed by:
``` bash
python unires.py --help
```
As an example, the voxel size of the super-resolved data is here set to
 1.5 mm isotropic:
``` bash
python unires.py --vx 1.5 T1.nii.gz T2.nii.gz PD.nii.gz
```

## 2. Running through NVIDIA Docker
This section describes setting up *UniRes* to run using NVIDIA's Docker 
engine on Ubuntu. As long as the NVIDIA Docker engine can be installed on 
MS Windows or Mac OSX there is no reason that these operating systems could 
not also be used. However, setting it up for Windows and OSX is not 
described here.

### 2.1. Install NVIDIA driver and Docker
Make sure that you have **installed** the **NVIDIA driver** and **Docker 
engine** for **your Linux distribution** (you do not need to install the 
CUDA Toolkit on the host system). These commands should install Docker:
``` bash
curl https://get.docker.com | sh
sudo systemctl start docker && sudo systemctl enable docker
```
Regarding the NVIDIA driver, I personally like the installation guide in 
[1] (step 2). Although this guide is targeted at Ubuntu 19.04, it should 
generalise to other Debian/Ubuntu versions (I used it for Ubuntu 18.04). 

### 2.2. Install the NVIDIA Docker engine
Execute the following commands to install the NVIDIA Docker engine:
``` bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```
Next, edit/create `/etc/docker/daemon.json` with content (e.g., by `sudo vim /etc/docker/daemon.json`):
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```
then do:
``` bash
sudo systemctl restart docker
```
Finally, test that it works by starting *nvidia-smi* in a Docker container:
``` bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 2.3. Build UniRes Docker image
`cd` to the content of the `docker` folder and run the following command to 
build the *UniRes* image:
``` bash
docker build --rm --tag unires:1.0 .
```
If there are permission issues, the following can help:
``` bash
sudo chmod 666 /var/run/docker.sock
```
Now you can run *UniRes* via Docker containers started from the `unires:1.0` 
image!

### 2.4. Process MRI scans through Docker container
Let's say you have a folder named `data` in your current working directory, 
which contains two MR images of the same subject: `T1.nii.gz`, `PD.nii.gz`. 
You can then process these two scans with *UniRes* by executing:
``` bash
docker run -it --rm -v $PWD/data:/home/docker/app/data unires:1.0 data/T1.nii.gz data/PD.nii.gz
```
When the algorithm has finished, you will find the processed scans in
 the same `data` folder, prefixed `'ur_'`.

## 3. References
1. Brudfors M, Balbastre Y, Nachev P, Ashburner J.
   A Tool for Super-Resolving Multimodal Clinical MRI.
   2019 arXiv preprint arXiv:1909.01140. 

2. Brudfors M, Balbastre Y, Nachev P, Ashburner J.
   MRI Super-Resolution Using Multi-channel Total Variation.
   In Annual Conference on Medical Image Understanding and Analysis
   2018 Jul 9 (pp. 217-228). Springer, Cham.   
