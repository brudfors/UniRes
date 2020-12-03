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


def _run(pth, atlas_align, atlas_rigid, crop, device, dir_out,
         linear, plot_conv, prefix, print_info, reg_scl, res_origin, sched,
         show_hyperpar, show_jtv, tolerance, unified_rigid, vx,
         write_out):
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
    s.dir_out = dir_out
    s.plot_conv = plot_conv
    s.do_print = print_info
    s.reg_scl = reg_scl
    s.show_hyperpar = show_hyperpar
    s.show_jtv = show_jtv
    s.tolerance = tolerance
    s.unified_rigid = unified_rigid
    s.crop = crop
    s.vx = vx
    s.do_res_origin = res_origin
    s.do_atlas_align = atlas_align
    s.atlas_rigid = atlas_rigid
    s.atlas_rigid = write_out
    s.sched_num = sched
    s.prefix = prefix
    if linear:
        s.max_iter = 0
        s.prefix = 'l' + s.prefix

    # Init UniRes
    x, y, s = init(pth, s)

    # Fit UniRes
    dat_y, mat_y, pth_y, _, _, _ = fit(x, y, s)

    return dat_y, mat_y, pth_y


if __name__ == "__main__":
    # UniRes default settings
    s = settings()
    # Build parser
    parser = ArgumentParser()
    # Compulsory arguments
    parser.add_argument("pth",
                        type=str,
                        nargs='+',
                        help="<Required> nibabel compatible path(s) to "
                             "subject MRIs/CTs.")
    # Optional arguments
    #
    parser.add_argument("--atlas_align",
                        action='store_true',
                        help="Align images to an atlas space ("
                             "default=" + str(s.do_atlas_align) + ").")
    parser.add_argument('--no-atlas_align', dest='atlas_align',
                        action='store_false')
    parser.set_defaults(atlas_align=s.do_atlas_align)
    #
    parser.add_argument("--atlas_rigid",
                        action='store_true',
                        help="Rigid, else rigid+isotropic, alignment to "
                             "atlas (default=" + str(s.atlas_rigid) + ").")
    parser.add_argument('--no-atlas_rigid', dest='atlas_rigid',
                        action='store_false')
    parser.set_defaults(atlas_rigid=s.atlas_rigid)
    #
    parser.add_argument("--crop",
                        action='store_true',
                        help="Crop input images' FOV to brain in the "
                             "NITorch atlas (default=" + str(s.crop) +
                             ").")
    parser.add_argument('--no-crop', dest='crop',
                        action='store_false')
    parser.set_defaults(crop=s.crop)
    #
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="PyTorch device (default='cuda').")
    #
    parser.add_argument("--dir_out",
                        type=str,
                        default=s.dir_out,
                        help="Directory to write output. Default is same as "
                             "as input data.")
    #
    parser.add_argument("--linear",
                        action='store_true',
                        help="Reslice using trilinear interpolation, i.e.,"
                             "no super-resolution (default=False).")
    parser.add_argument('--no-linear', dest='linear',
                        action='store_false')
    parser.set_defaults(linear=False)
    #
    parser.add_argument("--plot_conv",
                        action='store_true',
                        help="Use matplotlib to plot convergence in "
                             "real-time (default=" + str(s.plot_conv) +
                             ").")
    parser.add_argument('--no-plot_conv', dest='plot_conv',
                        action='store_false')
    parser.set_defaults(plot_conv=s.plot_conv)
    #
    parser.add_argument("--prefix",
                        type=str,
                        default=s.prefix,
                        help="Output image(s) prefix (default=" + str(
                            s.prefix) + ").")
    #
    parser.add_argument("--print_info",
                        type=int,
                        default=s.do_print,
                        help="Print progress to terminal (0, 1, 2; "
                             "default=" + str(s.do_print) + ").")
    #
    parser.add_argument("--reg_scl",
                        type=float,
                        default=s.reg_scl,
                        help="Scale regularisation estimate (default=" +
                             str(s.reg_scl) + ").")
    #
    parser.add_argument("--res_origin",
                        action='store_true',
                        help="Resets origin, if CT data (default=" +
                             str(s.do_res_origin) + ").")
    parser.add_argument('--no-res_origin', dest='res_origin',
                        action='store_false')
    parser.set_defaults(res_origin=s.do_res_origin)
    #
    parser.add_argument("--sched",
                        type=int,
                        default=s.sched_num,
                        help="Number of coarse-to-fine scalings ("
                             "default=" + str(s.sched_num) + ").")
    #
    parser.add_argument("--show_hyperpar",
                        action='store_true',
                        help="Use matplotlib to visualise "
                             "hyper-parameter estimates (default=" +
                             str(s.show_hyperpar) + ").")
    parser.add_argument('--no-show_hyperpar', dest='show_hyperpar',
                        action='store_false')
    parser.set_defaults(show_hyperpar=s.show_hyperpar)
    #
    parser.add_argument("--show_jtv",
                        action='store_true',
                        help="Show the joint total variation ("
                             "default=" + str(s.show_jtv) + ").")
    parser.add_argument('--no-show_jtv', dest='show_jtv',
                        action='store_false')
    parser.set_defaults(show_jtv=s.show_jtv)
    #
    parser.add_argument("--tolerance",
                        type=float,
                        default=s.tolerance,
                        help="Algorithm tolerance, if zero, run to "
                             "max_iter (default=" + str(s.tolerance) +
                             ").")
    #
    parser.add_argument("--unified_rigid",
                        action='store_true',
                        help="Do unified rigid registration ("
                             "default=" + str(s.unified_rigid) + ").")
    parser.add_argument('--no-unified_rigid', dest='unified_rigid',
                        action='store_false')
    parser.set_defaults(unified_rigid=s.unified_rigid)
    #
    parser.add_argument("--vx",
                        type=float,
                        default=s.vx,
                        help="Reconstruction voxel size (default=" +
                             str(s.vx) + ").")
    #
    parser.add_argument("--write_out",
                        action='store_true',
                        help="Write reconstructed output images ("
                             "default=" + str(s.write_out) + ").")
    parser.add_argument('--no-write_out', dest='write_out',
                        action='store_false')
    parser.set_defaults(write_out=s.write_out)
    # Apply
    args = parser.parse_args()
    _run(**vars(args))
