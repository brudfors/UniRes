# UniRes: Unified Super-Resolution of Neuroimaging Data
<img style="float: right;" src="https://github.com/brudfors/UniRes/blob/master/figures/example_2.png" width="100%" height="100%"> 

This repository implements a unified model for super-resolving neuroimaging data (MRI and CT scans), which combines: super-resolution with a multi-channel denoising prior, rigid registration and correction for interleaved slice acquisition. The archetype use-case is when having multiple scans of the same subject (e.g., T1w, T2w and FLAIR MRIs) and an analysis requires these scans to be represented on the same grid (i.e., having the same image size, affine matrix and voxel size). By default, the model reconstructs 1 mm isotropic images with a field-of-view that contains all input scans; however, this voxel size can be customised with the possibility of sub-millimetric reconstuctions. The model additionally supports multiple repeats of each MR sequence. The implementation is written in *PyTorch* and should therefore execute fast on the GPU. The software can be run either through **Docker** -- which ensures the correct library and OS versions are used, plus requires no compilation -- or directly by interfacing with the **Python** code. Both of these ways are described next.

## 1. Python

### 1.1. Dependencies
The *NITorch* package is required to use *UniRes*; simply follow the quickstart guide on its GitHub page: https://github.com/balbasty/nitorch.

Next, activate a *NITorch* virtual environment and move to the *UniRes* project directory. Then install the package using either setuptools or pip:
``` bash
cd /path/to/unires
python setupy.py install/develop
# OR
pip install .
``` 

### 1.2. Example use case

Running *UniRes* should be straight forward. Let's say you have three thick-sliced MR images: `T1.nii.gz`, `T2.nii.gz` and `PD.nii.gz`, then simply run `unires.py` in the terminal as:
``` bash
python unires.py T1.nii.gz T2.nii.gz PD.nii.gz
```
Three 1 mm isotropic images are written to the same folder as the input data, prefixed *y_*.

### 1.3. Advanced use

The algorithm estimates the necessary parameters from the input data, so it should, hopefully, work well out-the-box. However, a user might want to change some of the defaults, like slice-profile, slice-gap, or scale the regularisation a bit. Furthermore, instead of giving, e.g., nifti files via the command line tool (`fit.py`) it might be more desirable to interact with the nires code directly (maybe as part of some pipeline), working with the image data as `torch.tensor`. The following code snippet shows an example of how to do this:
``` python
from unires.run import (init, fit)
from unires.struct import settings

# Algorithm settings
s = settings()
s.reg_scl = 32  # scale regularisation
s.max_iter = 512  # maximum number of algorithm iterations
s.tolerance = 1e-4  # algorithm stopping tolerance
s.prefix = 'ur_'  # prefix of reconstructed images
s.dir_out = 'path'  # output directory to write reconstructions
s.vx = 0.8  # voxel size of reconstruced images
s.gap = 0.1  # slice-gap
s.inplane_ip = 0  # in-plane slice profile (0=rect|1=tri|2=gauss)
s.profile_tp = 0  # through-plane slice profile (0=rect|1=tri|2=gauss)
s.write_out = False  # write reconstruced images?

# Init UniRes
x, y, s = init(pth, s)
# pth is a list of strings (nibabel compatible paths)

# Fit UniRes
dat_y, mat_y, pth_y, R, _, _ = fit(x, y, s)
# Outputs are: reconstructed data (dat_y), output affine (mat_y), 
# paths to reconstructions (pth_y) and rigid transformation matrices (R).
```
More details of algorithm settings can be found in the declaration of
 the dataclass `settings()` in `struct.py`.

## 2. Running through NVIDIA Docker
This section describes setting up *UniRes* to run using NVIDIA's Docker engine on Ubuntu. As long as the NVIDIA Docker engine can be installed on MS Windows or Mac OSX there is no reason that these operating systems could not also be used. However, setting it up for Windows and OSX is not described here.

### 2.1. Install NVIDIA driver and Docker
Make sure that you have **installed** the **NVIDIA driver** and **Docker engine** for **your Linux distribution** (you do not need to install the CUDA Toolkit on the host system). These commands should install Docker:
``` bash
curl https://get.docker.com | sh
sudo systemctl start docker && sudo systemctl enable docker
```
Regarding the NVIDIA driver, I personally like the installation guide in [1] (step 2). Although this guide is targeted at Ubuntu 19.04, it should generalise to other Debian/Ubuntu versions (I used it for Ubuntu 18.04). 

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
`cd` to the content of the `docker` folder and run the following command to build the *UniRes* image:
``` bash
docker build --rm --tag unires:1.0 .
```
If there are permission issues, the following can help:
``` bash
sudo chmod 666 /var/run/docker.sock
```
Now you can run *UniRes* via Docker containers started from the `unires:1.0` image!

### 2.4. Process MRI scans through Docker container
Let's say you have a folder named `data` in your current working directory, which contains two MR images of the same subject: `T1.nii.gz`, `PD.nii.gz`. You can then process these two scans with *UniRes* by executing:
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
