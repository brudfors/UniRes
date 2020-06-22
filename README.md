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

The `'nitorch`' package is required to fit the model; simply follow the quickstart guide on its GitHub page. Once the nitroch environment has been activated, simply do:
```
python fit image1.nii image2.nii image3.nii ... --vx 1.0
```
The 1 mm isotropic images are written to the same folder as the input images, prefixed `'y'`.

## Further customisation

The algorithm estimates the necessary parameters from the input data, so it should, hopefully, work well out-the-box. However, a user might want to change some of the defaults, like slice-profile, slice-gap, or scale the regularisation a bit. Furthermore, instead of giving, e.g., nifti files via the command line tool (`'fit.py`') it might be more desirable to interact with the nires code directly (maybe as part of some pipeline), working with the image data as torch.tensor. The following code snippet shows an example of how to do this:
```
import nibabel as nib
from nires import Model
from nires import Settings

# Algorithm settings
s = Settings()
s.reg_scl = 10  # scale regularisation
s.max_iter = 512  # maximum number of algorithm iterations
s.tolerance = 1e-4  # algorithm stopping tolerance
s.prefix = 'y_'  # prefix of reconstructed images
s.dir_out = 'path'  # output directory to write reconstructions
s.vx = 0.8  # voxel size of reconstruced images
s.gap = 0.1  # slice-gap
s.inplane_ip = 0  # in-plane slice profile (0=rect|1=tri|2=gauss)
s.profile_tp = 0  # through-plane slice profile (0=rect|1=tri|2=gauss)
s.write_out = False  # write reconstruced images?

# Load files in list 'p' into a data array using nibabel.
# Each element of this array should contain the image data and
# the corresponding affine matrix (as [[dat, mat], ...]).
data = []
for c in range(len(p)):
    nii = nib.load(p[c])
    # Get affine matrix
    mat = nii.affine
    # Get image data
    dat = nii.get_fdata()
    data.append([dat, mat])

# Init super-resolution model
model = Model(data, s)

# Fit superres model
y, mat, p_y = model.fit()
# Outputs are: reconstructed data (y), output affine (mat), paths to reconstructions (p_y)
```
More details of the algorithm settings can be found in the declaration of the dataclass `'Settings()`' in `'nires.py`'.
