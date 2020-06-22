# nires: Neuroimaging super-resolution with PyTorch

A model for super-resolving neuroimaging data on the GPU (MRI and CT). The archetype use-case is when having multiple scans of the same subject (e.g., T1w, T2w and FLAIR MRIs) and an analysis requires these scans to be represented on the same grid (i.e., having the same image size, affine matrix and voxel size). By default, the model reconstructs 1 mm isotropic images with a field-of-view that contains all input scans; however, this voxel size can be customised with the possibility of sub-millimetric reconstuctions. The code is based on the algorithm described in the papers:

     Brudfors M, Balbastre Y, Nachev P, Ashburner J.
     A Tool for Super-Resolving Multimodal Clinical MRI.
     2019 arXiv preprint arXiv:1909.01140.     
     
     Brudfors M, Balbastre Y, Nachev P, Ashburner J.
     MRI Super-Resolution Using Multi-channel Total Variation.
     In Annual Conference on Medical Image Understanding and Analysis
     2018 Jul 9 (pp. 217-228). Springer, Cham.   
     
The model additionally supports multiple repeats of the same MR sequence.

The implementation is written in PyTorch and should therefore run fast on the GPU. It is possible to run it also on the CPU, but GPU is strongly encouraged..

## A simple example

The nitorch package is required to fit the model; simply follow the quickstart guide on its GitHub page. Once the nitroch environment has been activated, simply do:
```
python fit image1.nii image2.nii image3.nii ... --vx 1.0
```
The 1 mm isotropic images are written to the same folder as the input images, prefixed `'y'`.
