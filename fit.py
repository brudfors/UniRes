"""
Script to process MRI or CT scans with UnRes.

Example usage:
    python fit.py T1.nii T2.nii Flair.nii

Results will be written in the same folder, prefixed 'y_'.
Default settings should work well.

@author: brudfors@gmail.com
"""


from argparse import ArgumentParser
from unres import Model
from unres import Settings
import sys
import torch

def fit(pth, device, dir_out, plot_conv, print_info, reg_scl,
        show_hyperpar, show_jtv, tolerance, unified_rigid, vx):

    # CPU/GPU?
    device = torch.device("cpu" if not torch.cuda.is_available() else device)

    # Algorithm settings
    s = Settings()  # Get default settings
    s.device = device
    if dir_out is not None: s.dir_out = dir_out
    if plot_conv is not None: s.plot_conv = plot_conv
    if print_info is not None: s.print_info = print_info
    if reg_scl is not None: s.reg_scl = reg_scl
    if show_hyperpar is not None: s.show_hyperpar = show_hyperpar
    if show_jtv is not None: s.show_jtv = show_jtv
    if tolerance is not None: s.tolerance = tolerance
    if unified_rigid is not None: s.unified_rigid = unified_rigid
    if vx is not None: s.vx = vx

    # Init algorithm
    model = Model(pth, s)

    # Start algorithm
    model.fit()


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Compulsory arguments
    parser.add_argument("pth",
                        type=str,
                        nargs='+',
                        help="<Required> nibabel compatible path(s) to subject MRIs/CT")
    # Optional arguments
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="PyTorch device (default: cuda:0)",)
    parser.add_argument("--dir_out",
                        type=str,
                        default=None,
                        help="Directory to write output, if None uses same as input (output is prefixed 'y_')")
    parser.add_argument("--plot_conv",
                        type=bool,
                        default=None,
                        help="Use matplotlib to plot convergence in real-time")
    parser.add_argument("--print_info",
                        type=int,
                        default=None,
                        help="Print progress to terminal (0, 1, 2)")
    parser.add_argument("--reg_scl",
                        type=float,
                        default=None,
                        help="Scale regularisation estimate")
    parser.add_argument("--show_hyperpar",
                        type=bool,
                        default=None,
                        help="Use matplotlib to visualise hyper-parameter estimates")
    parser.add_argument("--show_jtv",
                        type=bool,
                        default=None,
                        help="Show the joint total variation (JTV)")
    parser.add_argument("--tolerance",
                        type=float,
                        default=None,
                        help="Algorithm tolerance, if zero, run to max_iter")
    parser.add_argument("--unified_rigid",
                        type=bool,
                        default=None,
                        help="Do unified rigid registration")
    parser.add_argument("--vx",
                        type=float,
                        default=None,
                        help="Reconstruction voxel size (if None, set automatically)")
    # Apply
    args = parser.parse_args()
    fit(**vars(args))
