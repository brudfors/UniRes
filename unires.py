"""Script to process MRI and/or CT scans with UniRes.

Example usage:
    python fit.py T1.nii T2.nii Flair.nii

Results will be written in the same folder, prefixed 'y_'.
Default settings should work well.

References:
    Brudfors M, Balbastre Y, Nachev P, Ashburner J.
    A Tool for Super-Resolving Multimodal Clinical MRI.
    2019 arXiv preprint arXiv:1909.01140.

    Brudfors M, Balbastre Y, Nachev P, Ashburner J.
    MRI Super-Resolution Using Multi-channel Total Variation.
    In Annual Conference on Medical Image Understanding and Analysis
    2018 Jul 9 (pp. 217-228). Springer, Cham.

@author: brudfors@gmail.com
"""


from argparse import ArgumentParser
import torch
from unires.struct import settings
from unires.run import (init, fit)


def _run(pth, device, dir_out, plot_conv, print_info, reg_scl,
         show_hyperpar, show_jtv, tolerance, unified_rigid, vx, linear,
         crop, do_res_origin, do_atlas_align, atlas_rigid, write_out):
    """Fit UniRes model from the command line.

    Returns
    ----------
    dat_y (torch.tensor): Reconstructed image data as float32, (dim_y, C).
    mat_y (torch.tensor): Reconstructed affine matrix, (4, 4).
    pth_y ([str, ...]): Paths to reconstructed images.

    """
    # CPU/GPU?
    device = torch.device("cpu" if not torch.cuda.is_available() else device)
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    # Algorithm settings
    s = settings()  # Get default settings
    s.device = device
    if dir_out is not None: s.dir_out = dir_out
    if plot_conv is not None: s.plot_conv = plot_conv
    if print_info is not None: s.print_info = print_info
    if reg_scl is not None: s.reg_scl = reg_scl
    if show_hyperpar is not None: s.show_hyperpar = show_hyperpar
    if show_jtv is not None: s.show_jtv = show_jtv
    if tolerance is not None: s.tolerance = tolerance
    if unified_rigid is not None: s.unified_rigid = unified_rigid
    if crop is not None: s.crop = crop
    if vx is not None: s.vx = vx
    if do_res_origin is not None: s.do_res_origin = do_res_origin
    if do_atlas_align is not None: s.do_atlas_align = do_atlas_align
    if atlas_rigid is not None: s.atlas_rigid = atlas_rigid
    if write_out is not None: s.atlas_rigid = write_out
    if linear:
        s.max_iter = 0
        s.prefix = 'l' + s.prefix

    # Init UniRes
    x, y, s = init(pth, s)

    # Fit UniRes
    dat_y, mat_y, pth_y, _, _, _ = fit(x, y, s)

    return dat_y, mat_y, pth_y


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
                        default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument("--dir_out",
                        type=str,
                        default=None,
                        help="Directory to write output, if None uses same as input (output is prefixed 'ur_')")
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
    parser.add_argument("--linear",
                        type=bool,
                        default=None,
                        help="Reslice using trilinear interpolation (no super-resolution)")
    parser.add_argument("--crop",
                        type=bool,
                        default=None,
                        help="Crop input images' FOV to brain in the NITorch atlas")
    parser.add_argument("--do_res_origin",
                        type=bool,
                        default=None,
                        help="Resets origin, if CT data")
    parser.add_argument("--do_atlas_align",
                        type=bool,
                        default=None,
                        help="Align images to an atlas space")
    parser.add_argument("--atlas_rigid",
                        type=bool,
                        default=None,
                        help="Rigid or rigid+isotropic scaling alignment to atlas")
    parser.add_argument("--write_out",
                        type=bool,
                        default=None,
                        help="Write reconstructed output images")
    # Apply
    args = parser.parse_args()
    _run(**vars(args))
