from argparse import ArgumentParser
import torch
from unires.struct import settings
from unires.run import preproc


def _preproc(pth, atlas_rigid, common_output, denoising, device, dir_out, fov, label_file,
             label_channel_index, label_repeat_index, linear, plot_conv, prefix,
             print_info, reg_scl, res_origin, scale, sched, show_hyperpar, show_jtv,
             tolerance, unified_rigid, vx, write_out, ct, crop):
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
    if isinstance(label_file, str):
        s.label = (label_file, (label_channel_index, label_repeat_index))
    s.show_hyperpar = show_hyperpar
    s.show_jtv = show_jtv
    s.tolerance = tolerance
    s.unified_rigid = unified_rigid
    s.common_output = common_output
    s.vx = vx
    s.do_res_origin = res_origin
    s.write_out = write_out
    s.sched_num = sched
    s.prefix = prefix
    s.scaling = scale
    s.fov = fov
    s.ct = ct
    s.crop = crop
    if linear:
        s.max_iter = 0    
    if denoising:
        s.vx = 0

    # Run UniRes
    dat_y, mat_y, pth_y = preproc(pth, s)

    return dat_y, mat_y, pth_y


def run():
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
    parser.add_argument("--atlas_rigid",
                        action='store_true',
                        help="Rigid, else rigid+isotropic, alignment to "
                             "atlas [default=" + str(s.atlas_rigid) + "].")
    parser.add_argument('--no-atlas_rigid', dest='atlas_rigid',
                        action='store_false')
    parser.set_defaults(atlas_rigid=s.atlas_rigid)
    #
    parser.add_argument("--common_output",
                        action='store_true',
                        help="Makes recons aligned with same grid, across subjects "
                             "[default=" + str(s.common_output) + "].")
    parser.add_argument('--no-common_output', dest='common_output',
                        action='store_false')
    parser.set_defaults(common_output=s.common_output)
    #
    parser.add_argument("--ct",
                        action='store_true',
                        help="Data could be CT (if contain negative values) "
                             "[default=" + str(s.ct) +
                             "].")
    parser.add_argument('--no-ct', dest='ct',
                        action='store_false')
    parser.set_defaults(ct=s.ct)
    #
    parser.add_argument("--crop",
                        action='store_true',
                        help="Crop field-of-view "
                             "[default=" + str(s.crop) + "].")
    parser.add_argument('--no-crop', dest='crop',
                        action='store_false')
    parser.set_defaults(crop=s.crop)
    parser.add_argument("--denoising",
                        action='store_true',
                        default=False,
                        help="Apply denoising to input data ")
    #
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="PyTorch device [default='cuda'].")
    #
    parser.add_argument("--dir_out",
                        type=str,
                        default=s.dir_out,
                        help="Directory to write output. Default is same "
                             "as input data.")
    #
    parser.add_argument("--fov",
                        type=str,
                        default=s.fov,
                        help="If crop, uses this field-of-view ('brain'|'head')")
    #
    parser.add_argument("--label_file",
                        type=str,
                        default=None,
                        help="Path to manual label file,"
                             " Nearest Neighbour interpolation will be applied to"
                             " it [default= None")
    #
    parser.add_argument("--label_channel_index",
                        type=int,
                        default=0,
                        help=" Channel index for label"
                             "[default=0]")
    #
    parser.add_argument("--label_repeat_index",
                        type=int,
                        default=0,
                        help=" Repeat index for label"
                             "[default=0]")
    #
    parser.add_argument("--linear",
                        action='store_true',
                        help="Reslice using trilinear interpolation, i.e.,"
                             "no super-resolution [default=False].")
    parser.add_argument('--no-linear', dest='linear',
                        action='store_false')
    parser.set_defaults(linear=False)
    #
    parser.add_argument("--plot_conv",
                        action='store_true',
                        help="Use matplotlib to plot convergence in "
                             "real-time [default=" + str(s.plot_conv) +
                             "].")
    parser.add_argument('--no-plot_conv', dest='plot_conv',
                        action='store_false')
    parser.set_defaults(plot_conv=s.plot_conv)
    #
    parser.add_argument("--prefix",
                        type=str,
                        default=s.prefix,
                        help="Output image(s) prefix [default=" + str(
                            s.prefix) + "].")
    #
    parser.add_argument("--print_info",
                        type=int,
                        default=s.do_print,
                        help="Print progress to terminal [0, 1, 2; "
                             "default=" + str(s.do_print) + "].")
    #
    parser.add_argument("--reg_scl",
                        type=float,
                        default=s.reg_scl,
                        help="Scale regularisation estimate [default=" +
                             str(s.reg_scl) + "].")
    #
    parser.add_argument("--res_origin",
                        action='store_true',
                        help="Resets origin, if CT data [default=" +
                             str(s.do_res_origin) + "].")
    parser.add_argument('--no-res_origin', dest='res_origin',
                        action='store_false')
    parser.set_defaults(res_origin=s.do_res_origin)
    #
    parser.add_argument("--scale",
                        action='store_true',
                        help="Optimise even/odd slice scaling [default=" +
                             str(s.scaling) + "].")
    parser.add_argument('--no-scale', dest='scale',
                        action='store_false')
    parser.set_defaults(scale=s.scaling)
    #
    parser.add_argument("--sched",
                        type=int,
                        default=s.sched_num,
                        help="Number of coarse-to-fine scalings ["
                             "default=" + str(s.sched_num) + "].")
    #
    parser.add_argument("--show_hyperpar",
                        action='store_true',
                        help="Use matplotlib to visualise "
                             "hyper-parameter estimates [default=" +
                             str(s.show_hyperpar) + "].")
    parser.add_argument('--no-show_hyperpar', dest='show_hyperpar',
                        action='store_false')
    parser.set_defaults(show_hyperpar=s.show_hyperpar)
    #
    parser.add_argument("--show_jtv",
                        action='store_true',
                        help="Show the joint total variation ["
                             "default=" + str(s.show_jtv) + "].")
    parser.add_argument('--no-show_jtv', dest='show_jtv',
                        action='store_false')
    parser.set_defaults(show_jtv=s.show_jtv)
    #
    parser.add_argument("--tolerance",
                        type=float,
                        default=s.tolerance,
                        help="Algorithm tolerance, if zero, run to "
                             "max_iter [default=" + str(s.tolerance) +
                             "].")
    #
    parser.add_argument("--unified_rigid",
                        action='store_true',
                        help="Do unified rigid registration ["
                             "default=" + str(s.unified_rigid) + "].")
    parser.add_argument('--no-unified_rigid', dest='unified_rigid',
                        action='store_false')
    parser.set_defaults(unified_rigid=s.unified_rigid)
    #
    parser.add_argument("--vx",
                        type=float,
                        default=s.vx,
                        help="Reconstruction voxel size [default=" +
                             str(s.vx) + "].")
    #
    parser.add_argument("--write_out",
                        action='store_true',
                        help="Write reconstructed output images ["
                             "default=" + str(s.write_out) + "].")
    parser.add_argument('--no-write_out', dest='write_out',
                        action='store_false')
    parser.set_defaults(write_out=s.write_out)
    #
    args = parser.parse_args()
    # Run UniRes
    _preproc(**vars(args))
