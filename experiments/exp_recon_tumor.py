import glob
import os
from pathlib import Path
from pprint import pprint
from unires.run import preproc
from exp_utils import get_sett

# Parameters
# ----------
# tumour scans
ddata_lgg = '/workspace/data/data/tumor/lgg'
ddata_gbm = '/workspace/data/data/tumor/gbm'
# torch device
device = 'cuda'
# output stuff
dir_results_lgg = '/workspace/data/experiments/tumor/recons/LGG'
dir_results_gbm = '/workspace/data/experiments/tumor/recons/GBM'
os.makedirs(dir_results_lgg, exist_ok=True)
os.makedirs(dir_results_gbm, exist_ok=True)
# testing?
test = False

# Get images
# ----------
pths_lgg = [str(f) for f in sorted(Path(ddata_lgg).rglob('*.nii'))]  # get niftis in ddata folder
pths_lgg = [pths_lgg[i:i + 3] for i in range(0, len(pths_lgg), 3)]  # split into list of lists, separating subjects
pths_gbm = [str(f) for f in sorted(Path(ddata_gbm).rglob('*.nii'))]  # get niftis in ddata folder
pths_gbm = [pths_gbm[i:i + 3] for i in range(0, len(pths_gbm), 3)]  # split into list of lists, separating subjects

# LGG
# Reconstruct
# ----------
for i, pth_x in enumerate(pths_lgg):

    print("=" * 32)
    print(f"pth_x[0]={pth_x[0]}\npth_x[1]={pth_x[1]}\npth_x[2]={pth_x[2]}")
    print("=" * 32)

    folder = os.path.dirname(pth_x[0]).split('/')[-1]
    
    # reslice
    _, _ , _ = preproc(pth_x, get_sett(device, os.path.join(dir_results_lgg, folder), (192,) * 3, sr=False))

    # super-resolve
    _, _ , _ = preproc(pth_x, get_sett(device, os.path.join(dir_results_lgg, folder), (192,) * 3, sr=True))
    
    if test:  break

# GBM
# Reconstruct
# ----------
for i, pth_x in enumerate(pths_gbm):

    print("=" * 32)
    print(f"pth_x[0]={pth_x[0]}\npth_x[1]={pth_x[1]}\npth_x[2]={pth_x[2]}")
    print("=" * 32)

    folder = os.path.dirname(pth_x[0]).split('/')[-1]
    
    # reslice
    _, _ , _ = preproc(pth_x, get_sett(device, os.path.join(dir_results_gbm, folder), (192,) * 3, sr=False))

    # super-resolve
    _, _ , _ = preproc(pth_x, get_sett(device, os.path.join(dir_results_gbm, folder), (192,) * 3, sr=True))
    
    if test:  break