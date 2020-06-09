
"""
Script to process MRI or CT scans (as nifti files) with mriqplus.

Example usage:
    python run_mriqplus.py T1.nii T2.nii Flair.nii --do_sr=True

"""


from argparse import ArgumentParser
import niiproc
from niiproc import NiiProc
import sys
import torch

def apply(pth, device, dir_out, dir_nitorch, plot_conv, print_info,
          reg_scl, show_hyperpar, tolerance, vx):

    # CPU/GPU?
    device = torch.device("cpu" if not torch.cuda.is_available() else device)

    # Algorithm settings
    s = niiproc.Settings  # Get default settings
    s.device = device
    if dir_out is not None: s.dir_out = dir_out
    if dir_nitorch is not None: s.dir_nitorch = dir_nitorch
    if plot_conv is not None: s.plot_conv = plot_conv
    if print_info is not None: s.print_info = print_info
    if reg_scl is not None: s.reg_scl = reg_scl
    if show_hyperpar is not None: s.show_hyperpar = show_hyperpar
    if tolerance is not None: s.tolerance = tolerance
    if vx is not None: s.vx = vx

    if dir_nitorch:
        # Set path to nitorch
        sys.path.append(dir_nitorch)

    # Init algorithm
    model = NiiProc(pth, s)

    # Start algorithm
    model.fit()


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Compulsory arguments
    parser.add_argument("pth",
                        type=str,
                        nargs='+',
                        help="<Required> Path to subject MRIs/CT (.nii/.nii.gz)")
    # Optional arguments
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="PyTorch device (default: cuda:0)",)
    parser.add_argument("--dir_out",
                        type=str,
                        default=None,
                        help="Directory to write output, if None uses same as input (output is prefixed 'y_')")
    parser.add_argument("--dir_nitorch",
                        type=str,
                        default=None,
                        help="Path to nitorch directory (https://github.com/balbasty/nitorch)")
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
    parser.add_argument("--tolerance",
                        type=float,
                        default=None,
                        help="Algorithm tolerance, if zero, run to max_iter")
    parser.add_argument("--vx",
                        type=float,
                        default=None,
                        help="Reconstruction voxel size (if None, set automatically)")
    # Apply
    args = parser.parse_args()
    apply(**vars(args))
