"""
Simple example of using UniRes API to preprocess a single image.
Good for testing and debugging.
"""
from nitorch import io
import torch
from unires.struct import settings
from unires.run import preproc


# Parameters
device = 'cuda:0'
pth_x = "./data/t1_icbm_normal_1mm_pn0_rf0.nii.gz"

# Data can be given either as...
 
# a file path or ...
data = pth_x
write_out = True

# ... a list [tensor, tensor] with image data and affine matrix
nii_x = io.map(pth_x)
x = nii_x.fdata().to(device)
eye = torch.eye(4, device=device)
data = [[x, eye]]
write_out = False

# Settings
s = settings()
s.vx = 1.0
s.write_out = write_out
s.reg_scl = 1e0
s.ct = False
s.show_hyperpar = True

# Run algorithm
y_hat, _, _ = preproc(data, sett=s)