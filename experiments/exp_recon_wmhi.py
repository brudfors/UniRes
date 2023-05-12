import glob
import os
from pprint import pprint
from unires.run import preproc
from exp_utils import get_sett

# Parameters
# ----------
# wmh scans
ddata = '/workspace/data/data/wmhi'
# torch device
device = 'cuda'
# output stuff
dir_results = '/workspace/data/experiments/wmhi/recons'
os.makedirs(dir_results, exist_ok=True)
# testing?
test = False

# Get images
# ----------
pths = [f for f in sorted(glob.glob(os.path.join(ddata, "*.nii")))]  # get niftis in ddata folder
pths = [pths[i:i + 3] for i in range(0, len(pths), 3)]  # split into list of lists, separating subjects
print("=" * 32)
print(f"len(pths)]={len(pths)}")
pprint(pths[0])
print("=" * 32)
print()

# Reconstruct
# ----------
for i, pth_x in enumerate(pths):
    
    print("=" * 32)
    print(f"pth_x[0]={pth_x[0]}\npth_x[1]={pth_x[1]}\npth_x[2]={pth_x[2]}")
    print("=" * 32)

    # get image data and label image
    x = [pth_x[1], pth_x[0]]
    l = (pth_x[2], (1, 0))  

    # reslice
    _, _ , _ = preproc(x, get_sett(device, dir_results, (256,) * 3, sr=False, label=l))

    # super-resolve    
    _, _ , _ = preproc(x, get_sett(device, dir_results, (256,) * 3, sr=True, label=l))
    
    if test:  break