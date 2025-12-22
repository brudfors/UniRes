# Unified Super-Resolution in PyTorch

This repository implements an algorithm based on a unified model for super-resolving medical images (MR images and CT scans).

<img style="float: right;" src="https://github.com/brudfors/UniRes/blob/master/figures/example_2.png" width="100%" height="100%">

The algorithm combines:

* Image super-resolution with a multi-channel denoising prior
* Rigid registration
* Correction for interleaved slice acquisition

These parameters are fit using an alternating optimization method, which iteratively converges to the optimal values. An initial registration step additionally ensures that all input images are well aligned before optimization begins.

The algorithm is best used when having multiple scans of the same subject (e.g., T1w, T2w and FLAIR MRIs) and an analysis requires these scans to be represented on the same grid (i.e., having the same image size, affine matrix and voxel size).

By default, the model reconstructs 1 mm isotropic images with a field-of-view that contains all input scans; however, this voxel size can be customised with the possibility of **sub-millimetric reconstuctions**. The model additionally supports multiple repeats of each MR sequence. There is an option that makes images registered and defined on the same grid, **across subjects**, where the grid size is optimal from a CNN fitting perspective.

See instructions below for both local and Docker installation. Additionally, there are Jupyter notebooks in the `demos` folder showing sample functionality.

## 1. Local Install

Follow the below instrutions to install `UniRes` locally (i.e., bare metal).

Note that the algorithm runs faster if `nitorch` (dependency) uses its compiled backend (see Section 1.1.1). Howevever, the compile time is quite slow, but only required once.

### 1.1. Installation
Clone `UniRes`:
```shell
git clone https://github.com/brudfors/UniRes
```
Then `cd` into the `UniRes`.

We recommend you use the `nitorch` compiled backend (see Section 1.1.1), but if you want a faster install simply do:
```shell
pip install -e .
```

#### 1.1.1. Using `nitorch` compiled backend

Prerequisites are that CUDA is installed and that `nvcc` is on the system path, and that your `torch` installation was built with the same CUDA version.

For example, if:
```shell
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:34:21_PDT_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
```
then install `torch` like:
```shell
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```
(the correct `torch` install command can be found under *Install Torch* at [https://pytorch.org/](https://pytorch.org/))

Now install `UniRes` with:
```shell
NI_COMPILED_BACKEND="C" pip install --no-build-isolation -e .
```

### 1.2. Example usage
Note that these examples are only for demonstration purposes, as the [BrainWeb](https://brainweb.bic.mni.mcgill.ca/brainweb/) images used are already 1 mm isotropic and noise free.

Super-resolve and align three MR images to 1 mm isotropic voxels:
``` shell
unires --vx 1.0 data/t1_icbm_normal_1mm_pn0_rf0.nii.gz data/t2_icbm_normal_1mm_pn0_rf0.nii.gz data/pd_icbm_normal_1mm_pn0_rf0.nii.gz
```
The processed images are written to the same folder as the input data, prefixed `'u_'`.

Instead of super-resolution it is possible to instead use a trilinear reslice:
``` shell
unires --linear --vx 1.0 data/t1_icbm_normal_1mm_pn0_rf0.nii.gz data/t2_icbm_normal_1mm_pn0_rf0.nii.gz data/pd_icbm_normal_1mm_pn0_rf0.nii.gz
```

It is also possible to make images aligned and defined on the same grid **across subjects**, where the grid size is optimal from a CNN fitting perspective:
``` shell
unires --common_output data/t1_icbm_normal_1mm_pn0_rf0.nii.gz data/t2_icbm_normal_1mm_pn0_rf0.nii.gz data/pd_icbm_normal_1mm_pn0_rf0.nii.gz
```

Although the model's hyper parameters are estimated from the data, adjusting the scaling of the regularisation can sometimes be required:
``` shell
unires --reg_scl 10 data/t1_icbm_normal_1mm_pn0_rf0.nii.gz data/t2_icbm_normal_1mm_pn0_rf0.nii.gz data/pd_icbm_normal_1mm_pn0_rf0.nii.gz
```

Finally, it is possible to do denoising with `UniRes`:
``` shell
unires --denoising data/t1_icbm_normal_1mm_pn0_rf0.nii.gz
```
if the data has negative values (e.g., CT) add the `--ct` flag.

There are plenty of other options that can be seen with:
``` shell
unires --help
```

## 2. Docker Install

This section describes how to run `UniRes` using Docker.

Prerequisites are that the NVIDIA GPU driver and the NVIDIA Container Toolkit are installed on the host machine.

### 2.1. Build UniRes Docker image
First, build the `UniRes` Docker image with:
``` shell
docker build --rm --tag unires:latest .
```
The build will use the compiled backend of `nitorch`, meaning it can take quite some time for the build to complete.

If you get an error that the host GPU and its driver are not available, make sure that the environment variable `TORCH_CUDA_ARCH_LIST` in the `Dockerfile` includes a [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) supported by your GPU. You can see the compute capability of your GPU with:
```sh
 nvidia-smi --query-gpu=compute_cap --format=csv
```

### 2.2. Example usage
Note that this example is only for demonstration purposes, as the [BrainWeb](https://brainweb.bic.mni.mcgill.ca/brainweb/) images used are already 1 mm isotropic and noise free.

Process the three simulated BrainWeb MR images in the `data` folder:
``` shell
docker run -it --rm --gpus all -v $PWD/data:/data unires:latest unires --vx 1.0 /data/t1_icbm_normal_1mm_pn0_rf0.nii.gz /data/t2_icbm_normal_1mm_pn0_rf0.nii.gz /data/pd_icbm_normal_1mm_pn0_rf0.nii.gz
```
When the algorithm has finished, you will find the processed scans in the same `data` folder, prefixed `'u_'`.

To easily test the demo notebooks in the `demos` folder, use VS Code and the "Dev Containers: Reopen in Container" command.

## 3. References
```BibTex
@inproceedings{brudfors2018mri,
  title={MRI super-resolution using multi-channel total variation},
  author={Brudfors, Mikael and Balbastre, Ya{\"e}l and Nachev, Parashkev and Ashburner, John},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={217--228},
  year={2018},
  organization={Springer}
}

@article{brudfors2019tool,
  title={A Tool for Super-Resolving Multimodal Clinical MRI},
  author={Brudfors, Mikael and Balbastre, Yael and Nachev, Parashkev and Ashburner, John},
  journal={arXiv preprint arXiv:1909.01140},
  year={2019}
}
```
