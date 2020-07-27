# -*- coding: utf-8 -*-
""" UniRes: Unified Super-Resolution of Neuroimaging Data in PyTorch.

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

from dataclasses import dataclass
from datetime import datetime
import math
import nibabel as nib
from nitorch.kernels import smooth
from nitorch.spatial import grid_pull, grid_push, voxsize, im_gradient, im_divergence, grid_grad
from nitorch.spm import affine, mean_space, noise_estimate, affine_basis, dexpm, identity, estimate_fwhm
from nitorch.optim import cg, get_gain, plot_convergence
from nitorch.utils import show_slices, round
import os
from timeit import default_timer as timer
import torch
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True


@dataclass
class Input:
    """ Algorithm input.

    """
    dat = None
    dim = None
    ct = None
    mat = None
    mu = None
    po = None
    sd = None
    tau = None
    head = None
    fname = None
    direc = None
    nam = None
    rigid_q = None


@dataclass
class Output:
    """ Algorithm output.

    """
    dat = None
    dim = None
    lam = None
    mat = None


@dataclass
class ProjOp:
    """ Encodes a projection operator.

    """
    dim_x = None
    mat_x = None
    vx_x = None
    dim_y = None
    mat_y = None
    vx_y = None
    dim_yx = None
    mat_yx = None
    ratio = None
    smo_ker = None
    rigid = None
    scl = None
    dim_thick = None
    D_x = None
    D_y = None

@dataclass
class Settings:
    """ Algorithm settings.

    """
    alpha: float = 1.0  # Relaxation parameter 0 < alpha < 2, alpha < 1: under-relaxation, alpha > 1: over-relaxation
    cgs_max_iter: int = 128  # Max conjugate gradient (CG) iterations for solving for y
    cgs_tol: float = 1e-2  # CG tolerance for solving for y
    cgs_verbose: bool = False  # CG verbosity (0, 1)
    device: str = 'cuda'  # PyTorch device name
    gr_diff: str = 'forward'  # Gradient difference operator (forward|backward|central)
    dir_out: str = None  # Directory to write output, if None uses same as input (output is prefixed 'y_')
    do_proj = None  # Use projection matrices, defined in format_output()
    gap: float = 0.0  # Slice gap, between 0 and 1
    has_ct: bool = False  # Data could be CT (but data must contain negative values)
    mat: torch.Tensor = None  # Observed image(s) affine matrix. OBS: Data needs to be given as 4D array
    max_iter: int = 512  # Max algorithm iterations
    method = None  # Method name (super-resolution|denoising), defined in format_output()
    mod_prct: float = 0.0  # Amount to crop mean space, between 0 and 1 (faster, but could loss out on data)
    prefix: str = 'y_'  # Prefix for reconstructed image(s)
    do_print: int = 1  # Print progress to terminal (0, 1, 2)
    plot_conv: bool = False  # Use matplotlib to plot convergence in real-time
    profile_ip: int = 0  # In-plane slice profile (0=rect|1=tri|2=gauss)
    profile_tp: int = 0  # Through-plane slice profile (0=rect|1=tri|2=gauss)
    reg_scl: float = 20.0  # Scale regularisation estimate (for coarse-to-fine scaling, give as list of floats)
    rho: float = None  # ADMM step-size, if None -> estimate is made
    rho_scl: float = 1.0  # Scaling of ADMM step-size
    rigid_basis = None  # Rigid transformation basis, defined in init_reg()
    rigid_mod: int = 4  # Update rigidt every rigid_mod iteration
    rigid_sched_max: int = 13  # Start scaling at 2^rigid_sched_max
    rigid_samp: int = 1  # Level of sub-sampling for estimating rigid registration parameters
    scaling: bool = True  # Optimise even/odd slice scaling
    show_hyperpar: bool = False  # Use matplotlib to visualise hyper-parameter estimates
    show_jtv: bool = False  # Show the joint total variation (JTV)
    tolerance: float = 1e-4  # Algorithm tolerance, if zero, run to max_iter
    unified_rigid: bool = True  # Do unified rigid registration
    vx: float = 1.0  # Reconstruction voxel size (if None, set automatically)
    write_jtv: bool = False  # Write JTV to nifti
    write_out: bool = True  # Write reconstructed output images


def init(data, sett=Settings()):
    """ Model initialiser.

        This is the entry point to the algorithm, it takes a bunch of nifti files
        as a list of paths (.nii|.nii.gz) and initialises input, output and projection
        operator objects. Settings are changed by editing the Settings() object and
        providing it to this constructor. If not given, default settings are used
        (see Settings() dataclass).

    Args:
        data
            (list): Path(s) to data, e.g:
                [*/T1.nii, */T2.nii, ...]
                or if multiple repeats:
                [[*/T1_1.nii, */T1_2.nii, ...],
                 [*/T2_1.nii, */T2_2.nii, ...], ...]
            (list): Image data and affine matrix(ces), e.g:
                [[T1_dat, T1_mat], [T2_dat, T2_mat], ...]
                where T1_dat is a array|torch.tensor with the image data and T1_mat
                is the image's corresponding affine matrix (array|torch.tensor).
                If multiple repeats:
                [[[T1_1_dat, T1_1_mat], [T1_2_dat, T1_2_mat]],
                 [[T2_1_dat, T2_1_mat], [T2_2_dat, T2_2_mat]], ...]

        sett (Settings(), optional): Algorithm settings. Described in Settings() class.

    """
    # Read and format data
    x = read_data(data, sett)
    del data
    
    # Estimate model hyper-parameters
    x = estimate_hyperpar(x, sett)
    
    # Init registration
    x, sett = init_reg(x, sett)

    # Format output
    y, sett = format_output(x, sett)

    # Define projection matrices
    x = proj_info_add(x, y, sett)

    if False:
        # Check adjointness of A and At operators
        check_adjoint(po=x[0][0].po, method=sett.method, dtype=torch.float64)

    return x, y, sett


def fit(x, y, sett):
    """ Fit model.

        This runs the iterative denoising/super-resolution algorithm and,
        at the end, writes the reconstructed images to disk. If the maximum number
        of iterations are set to zero, the initial guesses of the reconstructed
        images will be written to disk (acquired with b-spline interpolation), no
        denoising/super-resolution will be applied.

    Returns:
        y (torch.tensor): Reconstructed image data as float32, (dim_y, C).
        mat (torch.tensor): Reconstructed affine matrix, (4, 4).
        pth_y ([str, ...]): Paths to reconstructed images.
        R (torch.tensor): Rigid matrices (4, 4, N).

    """
    # Total number of observations
    N = sum([len(xn) for xn in x])

    # Initial guess of reconstructed images (y)
    y = init_y(x, y, sett)

    # Sanity check scaling parameter
    if not isinstance(sett.reg_scl, torch.Tensor):
        sett.reg_scl = torch.tensor(sett.reg_scl, dtype=torch.float32, device=sett.device)
        sett.reg_scl = sett.reg_scl.reshape(1)

    # For unified registration, defines a coarse-to-fine scaling of regularisation
    sett = set_sched(sett)

    # For visualisation
    fig_ax_nll = None
    fig_ax_jtv = None

    # Scale lambda
    cnt_scl = 0
    for c in range(len(x)):
        y[c].lam = sett.reg_scl[cnt_scl] * y[c].lam0

    # Get ADMM step-size
    rho = step_size(x, y, sett, verbose=True)

    if sett.max_iter > 0:
        # Get ADMM variables (only if algorithm is run)
        z, w = alloc_admm_vars(y, sett)

    # ----------
    # ITERATE:
    # Updates model in an alternating fashion, until a convergence threshold is met
    # on the model negative log-likelihood.
    # ----------
    next_reg_scl = False  # For registration, makes sure there is at least two registration updates for each scale level
    obj = torch.zeros(sett.max_iter, 3, dtype=torch.float64, device=sett.device)
    tmp = torch.zeros_like(y[0].dat)  # for holding rhs in y-update, and jtv in u-update
    t_iter = timer() if sett.do_print else 0
    for n_iter in range(sett.max_iter):

        if n_iter == 0:
            t00 = print_info('fit-start', sett, len(x), N, sett.device,
                             sett.max_iter, sett.tolerance)  # PRINT

        # ----------
        # UPDATE: image
        # ----------
        y, z, w, tmp, obj = update_admm(x, y, z, w, rho, tmp, obj, n_iter, sett)

        # Show JTV
        if sett.show_jtv:
            fig_ax_jtv = show_slices(img=tmp, fig_ax=fig_ax_jtv, title='JTV',
                                     cmap='coolwarm', fig_num=98)

        # ----------
        # Check convergence
        # ----------
        if sett.plot_conv:  # Plot algorithm convergence
            fig_ax_nll = plot_convergence(vals=obj[:n_iter + 1, :], fig_ax=fig_ax_nll, fig_num=99)
        gain = get_gain(obj[:n_iter + 1, 0], monotonicity='decreasing')
        t_iter = print_info('fit-ll', sett, 'y', n_iter, obj[n_iter, :], gain, t_iter)
        # Converged?
        if cnt_scl >= (sett.reg_scl.numel() - 1) and ((gain.abs() < sett.tolerance) or (n_iter >= (sett.max_iter - 1))):
            _ = print_info('fit-finish', sett, t00, n_iter)
            break  # Finished

        # ----------
        # UPDATE: even/odd scaling
        # ----------
        if sett.scaling and sett.reg_scl[cnt_scl] <= 256:

            t0 = print_info('fit-update', sett, 's', n_iter)  # PRINT
            # Do update
            x, _ = update_scaling(x, y, sett, max_niter_gn=1, num_linesearch=4, verbose=0)
            _ = print_info('fit-done', sett, t0)  # PRINT
            # # Print parameter estimates
            # _ = print_info('scl-param', sett, x, t0)

        # ----------
        # UPDATE: rigid_q (not every iteration)
        # ----------
        if sett.unified_rigid and 0 < n_iter and cnt_scl < (sett.reg_scl.numel() - 1) \
            and ((n_iter + 1) % sett.rigid_mod) == 0:

            t0 = print_info('fit-update', sett, 'q', n_iter)  # PRINT
            x, _ = update_rigid(x, y, sett, mean_correct=True, max_niter_gn=1, num_linesearch=4,
                             verbose=0, samp=sett.rigid_samp)
            _ = print_info('fit-done', sett, t0)  # PRINT
            # # Print parameter estimates
            # _ = print_info('reg-param', sett, x, t0)
            if gain.abs() < 1e-3:
                if not next_reg_scl:
                    next_reg_scl = True
                else:
                    next_reg_scl = False
                    cnt_scl += 1
                    # Coarse-to-fine scaling of lambda
                    for c in range(len(x)):
                        y[c].lam = sett.reg_scl[cnt_scl] * y[c].lam0
                    # Also update ADMM step-size
                    rho = step_size(x, y, sett)

    # ----------
    # Get rigid matrices
    # ----------
    R = torch.zeros((4, 4, N), device=sett.device, dtype=torch.float64)
    n = 0
    for c in range(len(x)):
        num_cn = len(x[c])
        for cn in range(num_cn):
            R[..., n] = dexpm(x[c][cn].rigid_q, sett.rigid_basis)[0]
            n += 1

    # ----------
    # Process reconstruction results
    # ----------
    y, mat, pth_y = write_data(x, y, sett, jtv=tmp)

    return y, mat, pth_y, R


def update_admm(x, y, z, w, rho, tmp, obj, n_iter, sett):
    """


    """
    # Parameters
    vx_y = voxsize(y[0].mat).float()  # Output voxel size
    bound_grad = 'constant'
    # Constants
    tiny = torch.tensor(1e-7, dtype=torch.float32, device=sett.device)
    one = torch.tensor(1, dtype=torch.float32, device=sett.device)
    # Over/under-relaxation parameter
    alpha = torch.tensor(sett.alpha, device=sett.device, dtype=torch.float32)

    # ----------
    # UPDATE: y
    # ----------
    t0 = print_info('fit-update', sett, 'y', n_iter)  # PRINT
    for c in range(len(x)):  # Loop over channels
        # RHS
        tmp[:] = 0
        for n in range(len(x[c])):  # Loop over observations of channel 'c'
            # _ = print_info('int', sett, n)  # PRINT
            tmp += x[c][n].tau * proj('At', x[c][n].dat, x[c], y[c], sett, rho, n=n)

        # Divergence
        div = w[c, ...] - rho * z[c, ...]
        div = im_divergence(div, vx=vx_y, bound=bound_grad, which=sett.gr_diff)
        tmp -= y[c].lam * div

        # Invert y = lhs\tmp by conjugate gradients
        lhs = lambda i: proj('AtA', i, x[c], y[c], sett, rho, vx_y=vx_y, bound__DtD=bound_grad, gr_diff=sett.gr_diff)
        cg(A=lhs, b=tmp, x=y[c].dat,
           verbose=sett.cgs_verbose,
           max_iter=sett.cgs_max_iter,
           stop='residuals',
           inplace=True,
           tolerance=sett.cgs_tol)  # OBS: y[c].dat is here updated in-place

        _ = print_info('int', sett, c)  # PRINT

    _ = print_info('fit-done', sett, t0)  # PRINT

    # ----------
    # Compute model objective function
    # ----------
    if sett.tolerance > 0:
        obj[n_iter, 0], obj[n_iter, 1], obj[n_iter, 2] \
            = compute_nll(x, y, sett, rho, bound=bound_grad, gr_diff=sett.gr_diff)  # nl_pyx, nl_pxy, nl_py

    # ----------
    # UPDATE: z
    # ----------
    if alpha != 1:  # Use over/under-relaxation
        z_old = z.clone()
    t0 = print_info('fit-update', sett, 'z', n_iter)  # PRINT
    tmp[:] = 0
    for c in range(len(x)):
        Dy = y[c].lam * im_gradient(y[c].dat, vx=vx_y, bound=bound_grad, which=sett.gr_diff)
        if alpha != 1:  # Use over/under-relaxation
            Dy = alpha * Dy + (one - alpha) * z_old[c, ...]
        tmp += torch.sum((w[c, ...] / rho + Dy) ** 2, dim=0)
    tmp.sqrt_()  # in-place
    tmp = ((tmp - one / rho).clamp_min(0)) / (tmp + tiny)

    for c in range(len(x)):
        Dy = y[c].lam * im_gradient(y[c].dat, vx=vx_y, bound=bound_grad, which=sett.gr_diff)
        if alpha != 1:  # Use over/under-relaxation
            Dy = alpha * Dy + (one - alpha) * z_old[c, ...]
        for d in range(Dy.shape[0]):
            z[c, d, ...] = tmp * (w[c, d, ...] / rho + Dy[d, ...])
    _ = print_info('fit-done', sett, t0)  # PRINT

    # ----------
    # UPDATE: w
    # ----------
    t0 = print_info('fit-update', sett, 'w', n_iter)  # PRINT
    for c in range(len(x)):  # Loop over channels
        Dy = y[c].lam * im_gradient(y[c].dat, vx=vx_y, bound=bound_grad, which=sett.gr_diff)
        if alpha != 1:  # Use over/under-relaxation
            Dy = alpha * Dy + (one - alpha) * z_old[c, ...]
        w[c, ...] += rho * (Dy - z[c, ...])
        _ = print_info('int', sett, c)  # PRINT
    _ = print_info('fit-done', sett, t0)  # PRINT

    return y, z, w, tmp, obj


def all_mat_dim_vx(x, sett):
    """ Get all images affine matrices, dimensions and voxel sizes (as numpy arrays).

    Returns:
        all_mat (torch.tensor): Image orientation matrices (4, 4, N).
        Dim (torch.tensor): Image dimensions (3, N).
        all_vx (torch.tensor): Image voxel sizes (3, N).

    """
    N = sum([len(xn) for xn in x])
    all_mat = torch.zeros((4, 4, N), device=sett.device, dtype=torch.float64)
    all_dim = torch.zeros((3, N), device=sett.device, dtype=torch.float64)
    all_vx = torch.zeros((3, N), device=sett.device, dtype=torch.float64)

    cnt = 0
    for c in range(len(x)):
        for n in range(len(x[c])):
            all_mat[..., cnt] = x[c][n].mat
            all_dim[..., cnt] = torch.tensor(x[c][n].dim, device=sett.device, dtype=torch.float64)
            all_vx[..., cnt] = voxsize(x[c][n].mat)
            cnt += 1

    return all_mat, all_dim, all_vx


def alloc_admm_vars(y, sett):
    """ Get ADMM variables z and w.

    Returns:
        z (torch.tensor()): (C, 3, dim_y)
        w (torch.tensor()): (C, 3, dim_y)

    """
    # Parse function settings/parameters
    dim_y = y[0].dim
    dim = (len(y), 3) + dim_y
    # Allocate
    z = torch.zeros(dim, dtype=torch.float32, device=sett.device)
    w = torch.zeros(dim, dtype=torch.float32, device=sett.device)

    return z, w


def compute_nll(x, y, sett, rho, sum_dtype=torch.float64, bound='constant', gr_diff='forward'):
    """ Compute negative model log-likelihood.

    Args:
        rho (torch.Tensor): ADMM step size.
        sum_dtype (torch.dtype): Defaults to torch.float64.
        bound (str, optional): Bound for gradient/divergence calculation, defaults to
            constant zero.
        gr_diff (str, optional): Gradient difference operator, defaults to 'forward'.

    Returns:
        nll_yx (torch.tensor()): Negative log-posterior
        nll_xy (torch.tensor()): Negative log-likelihood.
        nll_y (torch.tensor()): Negative log-prior.

    """
    vx_y = voxsize(y[0].mat).float()
    nll_xy = torch.tensor(0, device=sett.device, dtype=torch.float64)
    for c in range(len(x)):
        # Neg. log-likelihood term
        for n in range(len(x[c])):
            msk = x[c][n].dat != 0
            nll_xy += 0.5 * x[c][n].tau * torch.sum((x[c][n].dat[msk] -
                                                    proj('A', y[c].dat, x[c], y[c], sett, rho, n=n)[msk]) ** 2,
                                                    dtype=sum_dtype)
        # Neg. log-prior term
        Dy = y[c].lam * im_gradient(y[c].dat, vx=vx_y, bound=bound, which=gr_diff)
        if c > 0:
            nll_y += torch.sum(Dy ** 2, dim=0)
        else:
            nll_y = torch.sum(Dy ** 2, dim=0)

    nll_y = torch.sum(torch.sqrt(nll_y), dtype=sum_dtype)

    return nll_xy + nll_y, nll_xy, nll_y


def DtD(dat, vx_y, bound='constant', gr_diff='forward'):
    """ Computes the divergence of the gradient.

    Args:
        dat (torch.tensor()): A tensor (dim_y).
        vx_y (tuple(float)): Output voxel size.
        bound (str, optional): Bound for gradient/divergence calculation, defaults to
            constant zero.
        gr_diff (str, optional): Gradient difference operator, defaults to 'forward'.

    Returns:
          div (torch.tensor()): Dt(D(dat)) (dim_y).

    """
    dat = im_gradient(dat, vx=vx_y, bound=bound, which=gr_diff)
    dat = im_divergence(dat, vx=vx_y, bound=bound, which=gr_diff)
    
    return dat


def estimate_hyperpar(x, sett):
    """ Estimate noise precision (tau) and mean brain
        intensity (mu) of each observed image.

    Args:
        x (Input()): Input data.

    Returns:
        tau (list): List of C torch.tensor(float) with noise precision of each MR image.
        lam (torch.tensor(float)): The parameter lambda (1, C).

    """
    # Print info to screen
    t0 = print_info('hyper_par', sett)

    # Do estimation
    cnt = 0
    for c in range(len(x)):
        for n in range(len(x[c])):
            # Get data
            dat = x[c][n].dat
            if x[c][n].ct:
                # Estimate noise sd from estimate of FWHM
                sd_bg = estimate_fwhm(dat, voxsize(x[c][n].mat), mn=20, mx=50)[1]
                mu_bg = torch.tensor(0.0, device=dat.device, dtype=dat.dtype)
                mu_fg = torch.tensor(2000.0, device=dat.device, dtype=dat.dtype)
            else:
                # Get noise and foreground statistics
                sd_bg, sd_fg, mu_bg, mu_fg = noise_estimate(dat, num_class=2, show_fit=sett.show_hyperpar, 
                                                            fig_num=100 + cnt)
            # Set values
            x[c][n].sd = sd_bg.float()
            x[c][n].tau = 1 / sd_bg.float() ** 2
            x[c][n].mu = torch.abs(mu_fg.float() - mu_bg.float())
            cnt += 1

    # Print info to screen
    print_info('hyper_par', sett, x, t0)

    return x


def format_output(x, sett):
    """ Construct algorithm output struct. See Output() dataclass.

    Returns:
        y (Output()): Algorithm output struct(s).

    """
    one = torch.tensor(1.0, device=sett.device, dtype=torch.float64)
    vx_y = sett.vx
    if vx_y is not None:
        if isinstance(vx_y, int):
            vx_y = float(vx_y)
        if isinstance(vx_y, float):
            vx_y = (vx_y,) * 3
        vx_y = torch.tensor(vx_y, dtype=torch.float64, device=sett.device)

    # Get all orientation matrices and dimensions
    all_mat, all_dim, all_vx = all_mat_dim_vx(x, sett)
    N = all_mat.shape[-1]  # Total number of observations

    if N == 1:
        # Disable unified rigid registration
        sett.unified_rigid = False

    # Check if all input images have the same fov/vx
    mat_same = True
    dim_same = True
    vx_same = True
    for n in range(1, all_mat.shape[2]):
        mat_same = mat_same & torch.equal(round(all_mat[..., n - 1], 3), round(all_mat[..., n], 3))
        dim_same = dim_same & torch.equal(round(all_dim[..., n - 1], 3), round(all_dim[..., n], 3))
        vx_same = vx_same & torch.equal(round(all_vx[..., n - 1], 3), round(all_vx[..., n], 3))

    # Decide if super-resolving and/or projection is necessary
    do_sr = True
    sett.do_proj = True
    if vx_y is None and ((N == 1) or vx_same):  # One image, voxel size not given
        vx_y = all_vx[..., 0]

    if vx_same and (torch.abs(all_vx[..., 0] - vx_y) < 1e-3).all():
        # All input images have same voxel size, and output voxel size is the also the same
        do_sr = False
        if mat_same and dim_same and not sett.unified_rigid:
            # All input images have the same FOV
            mat = all_mat[..., 0]
            dim = all_dim[..., 0]
            sett.do_proj = False

    if do_sr or sett.do_proj:
        # Get FOV of mean space
        if mat_same and do_sr:
            D = torch.diag(torch.cat((vx_y / all_vx[:, 0], one[..., None])))
            mat = all_mat[..., 0].mm(D)
            mat[:3, 3] = mat[:3, 3] + 0.5*(vx_y - all_vx[:, 0])
            dim = D.inverse()[:3, :3].mm(all_dim[:, 0].reshape((3, 1))).ceil().squeeze()
        else:
            # Mean space from several images
            dim, mat, _ = mean_space(all_mat, all_dim, vx=vx_y, mod_prct=-sett.mod_prct)
    if do_sr:
        sett.method = 'super-resolution'
    else:
        sett.method = 'denoising'

    # Optimise even/odd scaling parameter?
    if sett.method == 'denoising' or (N == 1 and x[0][0].ct):
        sett.scaling = False

    dim = tuple(dim.int().tolist())
    _ = print_info('mean-space', sett, dim, mat)

    # CT lambda fudge factor
    ff_ct = 0.5  # Just a single CT
    if N > 1:
        # CT and MRIs
        ff_ct = 8.0

    # Assign output
    y = []
    for c in range(len(x)):
        y.append(Output())
        # Regularisation (lambda) for channel c
        mu_c = torch.zeros(len(x[c]), dtype=torch.float32, device=sett.device)
        for n in range(len(x[c])):
            mu_c[n] = x[c][n].mu
        y[c].lam0 = 1 / torch.mean(mu_c)
        y[c].lam = 1 / torch.mean(mu_c)  # To facilitate rescaling
        if x[c][0].ct:
            y[c].lam0 *= ff_ct
            y[c].lam *= ff_ct
        # Output image(s) dimension and orientation matrix
        y[c].dim = dim
        y[c].mat = mat.double().to(sett.device)

    return y, sett


def init_reg(x, sett):
    """ Initialise rigid registration.

    """
    # Init rigid basis
    sett.rigid_basis = affine_basis('SE', 3, device=sett.device, dtype=torch.float64)
    for c in range(len(x)):  # Loop over channels
        for n in range(len(x[c])):  # Loop over observations of channel c
            x[c][n].rigid_q = torch.zeros(6, device=sett.device, dtype=torch.float64)

    return x, sett


def init_y(x, y, sett):
    """ Make initial guesses of reconstucted image(s) using b-spline interpolation,
        with averaging if more than one observation per channel.

    """
    dim_y = x[0][0].po.dim_y
    mat_y = x[0][0].po.mat_y
    for c in range(len(x)):
        dat_y = torch.zeros(dim_y, dtype=torch.float32, device=sett.device)
        num_x = 1.0
        if sett.max_iter == 0:
            # No superres will be performed, so generate the isotropic images using b-splines
            num_x = len(x[c])
            for n in range(num_x):
                # Get image data
                dat = x[c][n].dat[None, None, ...]
                # Make output grid
                mat = mat_y.solve(x[c][n].po.mat_x)[0]  # mat_x\mat_y
                grid = affine(x[c][n].po.dim_y, mat, device=dat.device, dtype=dat.dtype)
                # Do interpolation
                mn = torch.min(dat)
                mx = torch.max(dat)
                dat = grid_pull(dat, grid, bound='zero', extrapolate=False, interpolation=1)
                dat[dat < mn] = mn
                dat[dat > mx] = mx
                dat_y = dat_y + dat[0, 0, ...]
        y[c].dat = dat_y / num_x

    return y


def print_info(info, sett, *argv):
    """ Print algorithm info to terminal.

    Args:
        info (string): What to print.

    """
    if not sett.do_print:
        return 0

    if sett.do_print >= 1:
        if info == 'fit-finish':
            print(' {} finished in {:0.5f} seconds and '
                  '{} iterations\n'.format(sett.method, timer() - argv[0], argv[1] + 1))
        elif info in 'fit-ll':
            print('{:3} - Convergence ({} | {:0.1f} s)  | nlyx={:0.4f}, nlxy={:0.4f}, nly={:0.4f} '
                  'gain={:0.7f}'.format(argv[1], argv[0], timer() - argv[4], argv[2][0], argv[2][1], argv[2][2],
                                        argv[3]))
        elif info == 'fit-start':
            print('\nStarting {} \n{} | C={} | N={} | device={} | '
                  'max_iter={} | tol={}'.format(sett.method, datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                argv[0], argv[1], argv[2], argv[3], argv[4]))
        elif info in 'step_size':
            print('\nADMM step-size={:0.4f}'.format(argv[0]))
        elif info in 'reg-param':
            print('Rigid registration fit:')
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print('c={} n={} | q={}'.format(c, n, round(argv[0][c][n].rigid_q, 4).cpu().tolist()))
            print('')
        elif info in 'scl-param':
            print('Scale fit:')
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print('c={} n={} | exp(s)={}'.format(c, n, round(argv[0][c][n].po.scl.exp(), 4)))
            print('')
        elif info == 'hyper_par':
            if len(argv) == 2:
                print('completed in {:0.5f} seconds:'.format(timer() - argv[1]))
                for c in range(len(argv[0])):
                    print('c={} | tau='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:0.7f}'.format(argv[0][c][n].tau), end=' ')
                    print('| sd='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:0.7f}'.format(argv[0][c][n].sd), end=' ')
                    print('| mu='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:0.7f}'.format(argv[0][c][n].mu), end=' ')
                    print('| ct='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{}'.format(argv[0][c][n].ct), end=' ')
                    print()
            else:
                print('\nEstimating model hyper-parameters...', end='')
        elif info == 'mean-space':
            vx_y = voxsize(argv[1])
            vx_y = tuple(vx_y.tolist())
            print('\nMean space | dim={}, vx_y={}'.format(argv[0], vx_y))
    if sett.do_print >= 2:
        if info == 'fit-done':
            print('(completed in {:0.5f} seconds)'.format(timer() - argv[0]))
        elif info == 'fit-update':
            print('{:3} - Updating {:2}   | '.format(argv[1] + 1, argv[0]), end='')
        elif info == 'int':
            print('{}'.format(argv[0]), end=' ')

    return timer()


def proj(operator, dat, x, y, sett, rho, n=0, vx_y=None, bound__DtD='constant', gr_diff='forward'):
    """ Projects image data by A, At or AtA.

    Args:
        operator (string): Either 'A', 'At ' or 'AtA'.
        dat (torch.Rensor): Image data (dim_x|dim_y).
        rho (torch.Tensor): ADMM step size.
        n (int): Observation index, defaults to 0.
        vx_y (tuple(float)): Output voxel size.
        bound__DtD (str, optional): Bound for gradient/divergence calculation, defaults to
            constant zero.
        gr_diff (str, optional): Gradient difference operator, defaults to 'forward'.

    Returns:
        dat (torch.tensor()): Projected image data (dim_y|dim_x).

    """
    if operator == 'AtA':
        if not sett.do_proj:  # return dat
            operator = 'none'
        dat1 = rho * y.lam ** 2 * DtD(dat, vx_y=vx_y, bound=bound__DtD, gr_diff=gr_diff)
        dat = dat[None, None, ...]
        dat = x[n].tau * proj_apply(operator, sett.method, dat, x[n].po)
        for n1 in range(1, len(x)):
            dat = dat + x[n1].tau * proj_apply(operator, sett.method, dat, x[n1].po)
        dat = dat[0, 0, ...]
        dat += dat1
    else:  # A, At
        if not sett.do_proj:  # return dat
            operator = 'none'
        dat = dat[None, None, ...]
        dat = proj_apply(operator, sett.method, dat, x[n].po)
        dat = dat[0, 0, ...]

    return dat


def proj_info_add(x, y, sett):
    """ Adds a projection matrix encoding to each input (x).

    """
    # Build each projection operator
    for c in range(len(x)):
        dim_y = y[c].dim
        mat_y = y[c].mat
        for n in range(len(x[c])):
            # Get rigid matrix
            rigid = dexpm(x[c][n].rigid_q, sett.rigid_basis)[0]
            # Define projection operator
            x[c][n].po = proj_info(dim_y, mat_y, x[c][n].dim, x[c][n].mat,
                                   prof_ip=sett.profile_ip, prof_tp=sett.profile_tp,
                                   gap=sett.gap, device=sett.device, rigid=rigid)

    return x


def read_data(data, sett):
    """ Parse input data into algorithm input struct(s).

    Args:
        data

    Returns:
        x (Input()): Algorithm input struct(s).

    """
    # Sanity check
    mat_vol = sett.mat
    if isinstance(data, str):
        nii = nib.load(data)
        dim = nii.shape
        if len(dim) > 3:
            # Data is path to 4D nifti
            data = nii.get_fdata()
            mat_vol = nii.affine
    try:
        data.shape
        data = data[..., None]
        data = data[:, :, :, :, 0]
        if mat_vol is None:
            raise ValueError('Image data given as array, please also provide affine matrix in sett.mat!')
    except AttributeError:
        pass
    if isinstance(data, str):
        data = [data]

    # Number of channels
    if mat_vol is not None:
        C = data.shape[3]
    else:
        C = len(data)

    x = []
    for c in range(C):  # Loop over channels
        x.append([])
        x[c] = []
        if mat_vol is None and isinstance(data[c], list) and (isinstance(data[c][0], str) or isinstance(data[c][0], list)):
            # Possibly multiple repeats per channel
            for n in range(len(data[c])):  # Loop over observations of channel c
                x[c].append(Input())
                # Get data
                dat, dim, mat, fname, direc, nam, head, ct, _ = \
                    read_image(data[c][n], sett.device, is_ct=sett.has_ct)
                # Assign
                x[c][n].dat = dat
                x[c][n].dim = dim
                x[c][n].mat = mat
                x[c][n].fname = fname
                x[c][n].direc = direc
                x[c][n].nam = nam
                x[c][n].head = head
                x[c][n].ct = ct
        else:
            # One repeat per channel
            n = 0
            x[c].append(Input())
            # Get data
            if mat_vol is not None:
                dat, dim, mat, fname, direc, nam, head, ct, _ = \
                    read_image([data[..., c], mat_vol], sett.device, is_ct=sett.has_ct)
            else:
                dat, dim, mat, fname, direc, nam, head, ct, _ = \
                    read_image(data[c], sett.device, is_ct=sett.has_ct)
            # Assign
            x[c][n].dat = dat
            x[c][n].dim = dim
            x[c][n].mat = mat
            x[c][n].fname = fname
            x[c][n].direc = direc
            x[c][n].nam = nam
            x[c][n].head = head
            x[c][n].ct = ct

    return x


def rigid_match(method, dat_x, dat_y, po, tau, rigid, CtC=None, diff=False, verbose=0):
    """ Computes the rigid matching term, and its gradient and Hessian (if requested).

    Args:
        dat_x (torch.tensor): Observed data (X0, Y0, Z0).
        dat_y (torch.tensor): Reconstructed data (X1, Y1, Z1).
        po (ProjOp): Projection operator.
        tau (torch.tensor): Noice precision.
        CtC (torch.tensor, optional): CtC(ones), used for super-res gradient calculation.
            Defaults to None.
        rigid (torch.tensor): Rigid transformation matrix (4, 4).
        diff (bool, optional): Compute derivatives, defaults to False.
        verbose (bool, optional): Show registration results, defaults to 0.
            0: No verbose
            1: Print convergence info to console
            2: Plot registration results using matplotlib

    Returns:
        ll (torch.tensor): Log-likelihood.
        gr (torch.tensor): Gradient (dim_x, 3).
        Hes (torch.tensor): Hessian (dim_x, 6).

    """
    # Projection info
    mat_x = po.mat_x
    mat_y = po.mat_y
    mat_yx = po.mat_yx
    dim_x = po.dim_x
    dim_yx = po.dim_yx
    ratio = po.ratio
    smo_ker = po.smo_ker
    dim_thick = po.dim_thick
    scl = po.scl

    # Init output
    ll = None
    gr = None
    Hes = None

    bound = 'dct2'
    interpolation = 1
    if method == 'super-resolution':
        extrapolate = True
        dim = dim_yx
        mat = mat_yx
    elif method == 'denoising':
        extrapolate = False
        dim = dim_x
        mat = mat_x

    # Get grid
    mat = rigid.mm(mat).solve(mat_y)[0]  # mat_y\rigid*mat
    grid = affine(dim, mat, device=dat_x.device, dtype=torch.float32)

    # Warp y and compute spatial derivatives
    dat_yx = grid_pull(dat_y, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)[0, 0, ...]
    if method == 'super-resolution':
        dat_yx = F.conv3d(dat_yx[None, None, ...], smo_ker, stride=ratio)[0, 0, ...]
        if scl != 0:
            dat_yx = apply_scaling(dat_yx, scl, dim_thick)
    if diff:
        gr = grid_grad(dat_y, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)[0, 0, ...]

    if verbose >= 2:  # Show images
        show_slices(torch.stack((dat_x, dat_yx, (dat_x - dat_yx) ** 2), 3),
                    fig_num=666, colorbar=False, flip=False)

    # Double and mask
    msk = dat_x != 0

    # Compute matching term
    ll = 0.5 * tau * torch.sum((dat_x[msk] - dat_yx[msk]) ** 2, dtype=torch.float64)

    if diff:
        # Difference
        diff = dat_yx - dat_x
        msk = msk & (dat_yx != 0)
        diff[~msk] = 0
        # Hessian
        Hes = torch.zeros(dim + (6,), device=dat_x.device, dtype=torch.float32)
        Hes[:, :, :, 0] = gr[:, :, :, 0] * gr[:, :, :, 0]
        Hes[:, :, :, 1] = gr[:, :, :, 1] * gr[:, :, :, 1]
        Hes[:, :, :, 2] = gr[:, :, :, 2] * gr[:, :, :, 2]
        Hes[:, :, :, 3] = gr[:, :, :, 0] * gr[:, :, :, 1]
        Hes[:, :, :, 4] = gr[:, :, :, 0] * gr[:, :, :, 2]
        Hes[:, :, :, 5] = gr[:, :, :, 1] * gr[:, :, :, 2]
        if method == 'super-resolution':
            Hes *= CtC[..., None]
            diff = F.conv_transpose3d(diff[None, None, ...], smo_ker, stride=ratio)[0, 0, ...]
        # Gradient
        gr *= diff[..., None]

    return ll, gr, Hes


def step_size(x, y, sett, verbose=False):
    """ ADMM step size (rho) from image statistics.

    Args:
        verbose (bool, optional): Defaults to False.

    Returns:
        rho (torch.tensor()): Step size.

    """
    rho = sett.rho
    if rho is not None:
        rho = torch.tensor(rho, device=sett.device, dtype=torch.float32)
    else:
        N = sum([len(xn) for xn in x])
        all_lam = torch.zeros(len(x), dtype=torch.float32, device=sett.device)
        all_tau = torch.zeros(N, dtype=torch.float32, device=sett.device)
        cnt = 0
        for c in range(len(x)):
            all_lam[c] = y[c].lam
            for n in range(len(x[c])):
                all_tau[cnt] = x[c][n].tau
                cnt += 1
        rho = sett.rho_scl * torch.sqrt(torch.mean(all_tau)) / torch.mean(all_lam)
    if verbose:
        _ = print_info('step_size', sett, rho)  # PRINT

    return rho


def update_scaling(x, y, sett, max_niter_gn=1, num_linesearch=4, verbose=0):
    """ Updates an even/odd slice scaling parameter using Gauss-Newton
        optimisation.

    Args:
        verbose (bool, optional): Verbose for testing, defaults to False.

    Returns:
        sll (torch.tensor): Log-likelihood.

    """
    # Update rigid parameters, for all input images
    sll = torch.tensor(0, device=sett.device, dtype=torch.float64)
    ll = torch.tensor(0, device=sett.device, dtype=torch.float64)
    for c in range(len(x)):  # Loop over channels
        for n_x in range(len(x[c])):  # Loop over repeats
            if x[c][n_x].ct:
                # Do not optimise scaling for CT data
                continue
            # Parameters
            dim_thick = x[c][n_x].po.dim_thick
            tau = x[c][n_x].tau
            scl = x[c][n_x].po.scl
            smo_ker = x[c][n_x].po.smo_ker
            dim_thick = x[c][n_x].po.dim_thick
            ratio = x[c][n_x].po.ratio
            dim = x[c][n_x].po.dim_yx
            mat_yx = x[c][n_x].po.mat_yx
            mat_y = x[c][n_x].po.mat_y
            rigid = dexpm(x[c][n_x].rigid_q, sett.rigid_basis)[0]
            mat = rigid.mm(mat_yx).solve(mat_y)[0]  # mat_y\rigid*mat_yx
            # Observed data
            dat_x = x[c][n_x].dat
            msk = dat_x != 0
            # Get even/odd data
            xo = even_odd(dat_x, 'odd', dim_thick)
            mo = even_odd(msk, 'odd', dim_thick)
            xo = xo[mo]
            xe = even_odd(dat_x, 'even', dim_thick)
            me = even_odd(msk, 'even', dim_thick)
            xe = xe[me]
            # Get reconstruction (without scaling)
            grid = affine(dim, mat, device=sett.device, dtype=torch.float32)
            dat_y = grid_pull(y[c].dat[None, None, ...], grid, bound='dct2', extrapolate=True)
            dat_y = F.conv3d(dat_y, smo_ker, stride=ratio)[0, 0, ...]
            # Apply scaling
            dat_y = apply_scaling(dat_y, scl, dim_thick)

            for n_gn in range(max_niter_gn):  # Loop over Gauss-Newton iterations
                # Log-likelihood
                ll = 0.5 * tau * torch.sum((dat_x[msk] - dat_y[msk]) ** 2, dtype=torch.float64)

                if verbose >= 2:  # Show images
                    show_slices(torch.stack((dat_x, dat_y, (dat_x - dat_y) ** 2), 3),
                                fig_num=666, colorbar=False, flip=False)

                # Get even/odd data
                yo = even_odd(dat_y, 'odd', dim_thick)
                yo = yo[mo]
                ye = even_odd(dat_y, 'even', dim_thick)
                ye = ye[me]

                # Gradient
                gr = tau * (torch.sum(ye * (xe - ye), dtype=torch.float64)
                            - torch.sum(yo * (xo - yo), dtype=torch.float64))

                # Hessian
                Hes = tau * (torch.sum(ye ** 2, dtype=torch.float64)
                             + torch.sum(yo ** 2, dtype=torch.float64))

                # Compute Gauss-Newton update step
                Update = gr / Hes

                # Do update..
                old_scl = scl.clone()
                old_ll = ll.clone()
                armijo = torch.tensor(1.0, device=sett.device, dtype=old_scl.dtype)
                if num_linesearch == 0:
                    # ..without a line-search
                    scl = old_scl - armijo * Update
                    if verbose >= 1:
                        print('c={}, n={}, gn={} | exp(s)={}'
                              .format(c, n_x, n_gn, round(s.exp(), 5)))
                else:
                    # ..using a line-search
                    for n_ls in range(num_linesearch):
                        # Take step
                        scl = old_scl - armijo * Update
                        # Apply scaling
                        dat_y = apply_scaling(dat_y, scl - old_scl, dim_thick)
                        # Compute matching term
                        ll = 0.5 * tau * torch.sum((dat_x[msk] - dat_y[msk]) ** 2, dtype=torch.float64)

                        if verbose >= 2:  # Show images
                            show_slices(torch.stack((dat_x, dat_y, (dat_x - dat_y) ** 2), 3),
                                        fig_num=666, colorbar=False, flip=False)

                        # Matching improved?
                        if ll < old_ll:
                            # Better fit!
                            if verbose >= 1:
                                print(
                                    'c={}, n={}, gn={}, ls={} | :) ll={:0.2f}, ll-oll={:0.2f} | exp(s)={} armijo={}'
                                    .format(c, n_x, n_gn, n_ls, ll, ll - old_ll, round(scl.exp(), 5),
                                            round(armijo, 4)))
                            break
                        else:
                            # Reset parameters
                            scl = old_scl
                            ll = old_ll
                            armijo *= 0.5
                            if verbose >= 1 and n_ls == num_linesearch - 1:
                                print(
                                    'c={}, n={}, gn={}, ls={} | :( ll={:0.2f}, ll-oll={:0.2f} | exp(s)={} armijo={}'
                                    .format(c, n_x, n_gn, n_ls, ll, ll - old_ll, round(old_scl.exp(), 5),
                                            round(armijo, 4)))
            # Update scaling in projection operator
            x[c][n_x].po.scl = scl
            # Accumulate neg log-lik
            sll += ll

    return x, sll


def update_rigid(x, y, sett, mean_correct=True, max_niter_gn=1, num_linesearch=4, verbose=0, samp=3):
    """ Updates each input image's specific registration parameters:
            x[c][n].rigid_q
        using a Gauss-Newton optimisation. After the parameters have
        been updated also the rigid matrix:
            x[c][n].po.rigid
        is updated

    Args:
        mean_correct (bool, optional): Mean-correct rigid parameters,
            defaults to True.
        max_niter_gn (int, optional): Max Gauss-Newton iterations, defaults to 1.
        num_linesearch (int, optional): Max line-search iterations, defaults to 4.
        verbose (bool, optional): Show registration results, defaults to 0.
            0: No verbose
            1: Print convergence info to console
            2: Plot registration results using matplotlib
        samp (int, optional): Sub-sample data, defaults to 3.

    Returns:
        sll (torch.tensor): Log-likelihood.

    """
    # Update rigid parameters, for all input images
    sll = torch.tensor(0, device=sett.device, dtype=torch.float64)
    for c in range(len(x)):  # Loop over channels
        # # FOR TESTING
        # from nitorch.spm import matrix
        # mat_re = matrix(torch.tensor([-8, 6, -3, 0, 0, 0], device=sett.device, dtype=torch.float64))
        # mat_y = y[c].mat
        # mat_y[:3, 3] = mat_re[:3, 3] + mat_y[:3, 3]
        # mat_y[:3, :3] = mat_re[:3, :3].mm(mat_y[:3, :3])
        # y[c].mat = mat_y
        # update_rigid_channel(c, sett.rigid_basis, verbose=2, max_niter_gn=16, samp=3)
        x[c], sllc = update_rigid_channel(x[c], y[c], sett, max_niter_gn=max_niter_gn,
                                          num_linesearch=num_linesearch, verbose=verbose,
                                          samp=samp)
        sll += sllc

    # Mean correct the rigid-body transforms
    if mean_correct:
        num_q = sett.rigid_basis.shape[2]  # Number of registration parameters
        sum_q = torch.zeros(num_q, device=sett.device, dtype=torch.float64)
        num_q = 0.0
        # Sum q parameters
        for c in range(len(x)):  # Loop over channels
            for n in range(len(x[c])):  # Loop over observations of channel c
                sum_q += x[c][n].rigid_q
                num_q += 1
        # Compute mean
        mean_q = sum_q / num_q
        # Subtract mean (mean correct)
        for c in range(len(x)):  # Loop over channels
            for n in range(len(x[c])):  # Loop over observations of channel c
                # if torch.sum(x[c][n].rigid_q) != 0:
                x[c][n].rigid_q -= mean_q

        # Update rigid transformations
        for c in range(len(x)):  # Loop over channels
            for n in range(len(x[c])):  # Loop over observations of channel c
                rigid = dexpm(x[c][n].rigid_q, sett.rigid_basis)[0]
                x[c][n].po.rigid = rigid

    return x, sll


def update_rigid_channel(xc, yc, sett, max_niter_gn=1, num_linesearch=4,
                         verbose=0, samp=3):
    """ Updates the rigid parameters for all images of one channel.

    Args:
        c (int): Channel index.
        rigid_basis (torch.tensor)
        max_niter_gn (int, optional): Max Gauss-Newton iterations, defaults to 1.
        num_linesearch (int, optional): Max line-search iterations, defaults to 4.
        verbose (bool, optional): Show registration results, defaults to 0.
            0: No verbose
            1: Print convergence info to console
            2: Plot registration results using matplotlib
        samp (int, optional): Sub-sample data, defaults to 3.

    Returns:
        sll (torch.tensor): Log-likelihood.

    """
    # Parameters
    device = yc.dat.device
    method = sett.method
    num_q = sett.rigid_basis.shape[2]
    lkp = [[0, 3, 4], [3, 1, 5], [4, 5, 2]]
    one = torch.tensor(1.0, device=device, dtype=torch.float64)

    sll = torch.tensor(0, device=device, dtype=torch.float64)
    for n_x in range(len(xc)):  # Loop over repeats

        # Lowres image data
        dat_x = xc[n_x].dat[None, None, ...]
        # Parameters
        q = xc[n_x].rigid_q
        tau = xc[n_x].tau
        armijo = torch.tensor(1.0, device=device, dtype=q.dtype)
        po = proj_info(xc[n_x].po.dim_y, xc[n_x].po.mat_y, xc[n_x].po.dim_x,
                            xc[n_x].po.mat_x, rigid=xc[n_x].po.rigid, prof_ip=sett.profile_ip,
                            prof_tp=sett.profile_tp, gap=sett.gap, device=device,
                            scl=xc[n_x].po.scl, samp=samp)

        # Superres or denoising?
        if method == 'super-resolution':
            dim = po.dim_yx
            mat = po.mat_yx
        elif method == 'denoising':
            dim = po.dim_x
            mat = po.mat_x

        # Do sub-sampling?
        if samp > 0 and po.D_x is not None:
            # Lowres
            grid = affine(po.dim_x, po.D_x, device=device, dtype=torch.float32)
            dat_x = grid_pull(xc[n_x].dat[None, None, ...], grid, bound='zero',
                              extrapolate=False, interpolation=0)[0, 0, ...]
            if n_x == 0 and po.D_y is not None:
                # Highres (only for superres)
                grid = affine(po.dim_y, po.D_y, device=device, dtype=torch.float32)
                dat_y = grid_pull(yc.dat[None, None, ...], grid, bound='zero',
                                  extrapolate=False, interpolation=0)
            else:
                dat_y = yc.dat[None, None, ...]
        else:
            dat_x = xc[n_x].dat
            dat_y = yc.dat[None, None, ...]

        # Pre-compute super-resolution Hessian (CtC)?
        CtC = None
        if method == 'super-resolution':
            CtC = F.conv3d(torch.ones((1, 1,) + dim, device=device, dtype=torch.float32),
                           po.smo_ker, stride=po.ratio)
            CtC = F.conv_transpose3d(CtC, po.smo_ker, stride=po.ratio)[0, 0, ...]

        # Get identity grid
        id_x = identity(dim, dtype=torch.float32, device=device, jitter=False)

        for n_gn in range(max_niter_gn):  # Loop over Gauss-Newton iterations

            # Differentiate Rq w.r.t. q (store in d_rigid_q)
            rigid, d_rigid = dexpm(q, sett.rigid_basis, diff=True)
            d_rigid_q = torch.zeros(4, 4, num_q, device=device, dtype=torch.float64)
            for i in range(num_q):
                d_rigid_q[:, :, i] = d_rigid[:, :, i].mm(mat).solve(po.mat_y)[0]  # mat_y\d_rigid*mat

            # Compute gradient and Hessian
            gr = torch.zeros(num_q, 1, device=device, dtype=torch.float64)
            Hes = torch.zeros(num_q, num_q, device=device, dtype=torch.float64)

            # Compute matching-term part (log-likelihood)
            ll, gr_m, Hes_m = rigid_match(sett.method, dat_x, dat_y, po, tau, rigid,
                                          diff=True, verbose=verbose, CtC=CtC)

            # Multiply with d_rigid_q (chain-rule)
            dAff = []
            for i in range(num_q):
                dAff.append([])
                for d in range(3):
                    dAff[i].append(d_rigid_q[d, 0, i] * id_x[:, :, :, 0] + \
                                   d_rigid_q[d, 1, i] * id_x[:, :, :, 1] + \
                                   d_rigid_q[d, 2, i] * id_x[:, :, :, 2] + \
                                   d_rigid_q[d, 3, i])

            # Add d_rigid_q to gradient
            for d in range(3):
                for i in range(num_q):
                    gr[i] += torch.sum(gr_m[:, :, :, d] * dAff[i][d], dtype=torch.float64)

            # Add d_rigid_q to Hessian
            for d1 in range(3):
                for d2 in range(3):
                    for i1 in range(num_q):
                        tmp1 = Hes_m[:, :, :, lkp[d1][d2]] * dAff[i1][d1]
                        for i2 in range(i1, num_q):
                            Hes[i1, i2] += torch.sum(tmp1 * dAff[i2][d2], dtype=torch.float64)

            # Fill in missing triangle
            for i1 in range(num_q):
                for i2 in range(i1 + 1, num_q):
                    Hes[i2, i1] = Hes[i1, i2]

            # # Regularise diagonal of Hessian
            # Hes += 1e-5*Hes.diag().max()*torch.eye(num_q, dtype=Hes.dtype, device=device)

            # Compute Gauss-Newton update step
            Update = gr.solve(Hes)[0][:, 0]

            # Do update..
            old_ll = ll.clone()
            old_q = q.clone()
            old_rigid = rigid.clone()
            if num_linesearch == 0:
                # ..without a line-search
                q = old_q - armijo * Update
                rigid = dexpm(q, sett.rigid_basis)[0]
                if verbose >= 1:
                    print('c={}, n={}, gn={} | q={}'.format(c, n_x, n_gn, round(q, 7).tolist()))
            else:
                # ..using a line-search
                for n_ls in range(num_linesearch):
                    # Take step
                    q = old_q - armijo * Update
                    # Compute matching term
                    rigid = dexpm(q, sett.rigid_basis)[0]
                    ll = rigid_match(sett.method, dat_x, dat_y, po, tau, rigid,
                                     verbose=verbose)[0]
                    # Matching improved?
                    if ll < old_ll:
                        # Better fit!
                        armijo = torch.min(1.25 * armijo, one)
                        if verbose >= 1:
                            print('c={}, n={}, gn={}, ls={} | :) ll={:0.2f}, ll-oll={:0.2f} | q={} armijo={}'
                                  .format(c, n_x, n_gn, n_ls, ll, ll - old_ll, round(q, 7).tolist(),
                                          round(armijo, 4)))
                        break
                    else:
                        # Reset parameters
                        ll = old_ll
                        q = old_q
                        rigid = old_rigid
                        armijo *= 0.5
                        if n_ls == num_linesearch - 1 and verbose >= 1:
                            print('c={}, n={}, gn={}, ls={} | :( ll={:0.2f}, ll-oll={:0.2f} | q={} armijo={}'
                                  .format(c, n_x, n_gn, n_ls, ll, ll - old_ll, round(q, 7).tolist(),
                                          round(armijo, 4)))
        # Assign
        xc[n_x].rigid_q = q
        xc[n_x].po.rigid = rigid
        # Accumulate neg log-lik
        sll += ll

    return xc, sll


def set_sched(sett):
    """ For unified registration, define a coarse-to-fine scaling of regularisation

    """
    if sett.unified_rigid:
        # Parameters/settings
        max = sett.rigid_sched_max
        scl = sett.reg_scl
        two = torch.tensor(2.0, device=sett.device, dtype=torch.float32)
        # Build scheduler
        sched = two ** torch.arange(0, max, device=sett.device, dtype=torch.float32).flip(dims=(0,))
        ix = torch.min((sched - sett.reg_scl).abs(), dim=0)[1]
        sched = sched[:ix]
        sched = torch.cat((sched, scl.reshape(1)))
        sett.reg_scl = sched

    return sett


def write_data(x, y, sett, jtv=None):
    """ Format algorithm output.

    Args:
        jtv (torch.tensor, optional): Joint-total variation image, defaults to None.

    Returns:
        y (torch.tensor): Reconstructed image data, (dim_y, C).
        mat (torch.tensor): Reconstructed affine matrix, (4, 4).
        pth_y ([str, ...]): Paths to reconstructed images.

    """
    # Output orientation matrix
    mat = y[0].mat
    dir_out = sett.dir_out
    if dir_out is None:
        # No output directory given, use directory of input data
        if x[0][0].direc is None:
            dir_out = 'UniRes-output'
        else:
            dir_out = x[0][0].direc
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out, exist_ok=True)
    # Reconstructed images
    prefix_y = sett.prefix
    pth_y = []
    for c in range(len(x)):
        mn = torch.min(x[c][0].dat)
        dat = y[c].dat
        dat[dat < mn] = 0
        if sett.write_out and sett.mat is None:
            # Write reconstructed images (as separate niftis, because given as separate niftis)
            if x[c][0].nam is None:
                nam = str(c) + '.nii'
            else:
                nam = x[c][0].nam
            fname = os.path.join(dir_out, prefix_y + nam)
            pth_y.append(fname)
            write_image(dat, fname, mat=mat, header=x[c][0].head)
        if c == 0:
            dat_y = dat[:, :, :, None]
        else:
            dat_y = torch.cat((dat_y, dat[:, :, :, None]), dim=3)
    if sett.write_out and sett.mat is not None:
        # Write reconstructed images as 4D volume (because given as 4D volume)
        c = 0
        if x[c][0].nam is None:
            nam = str(c) + '.nii'
        else:
            nam = x[c][0].nam
        fname = os.path.join(dir_out, prefix_y + nam)
        pth_y.append(fname)
        write_image(dat_y, fname, mat=mat, header=x[c][0].head)
    if sett.write_jtv and jtv is not None:
        # Write JTV
        if x[c][0].nam is None:
            nam = str(c) + '.nii'
        else:
            nam = x[c][0].nam
        fname = os.path.join(dir_out, 'jtv_' + prefix_y + nam)
        write_image(jtv, fname, mat=mat)

    return dat_y, mat, pth_y


def apply_scaling(dat, scl, dim):
    """ Apply even/odd slice scaling.

    """
    dat_out = torch.zeros_like(dat)
    if dim == 2:
        dat_out[..., :, :, ::2] = torch.exp(scl) * dat[..., :, :, ::2]
        dat_out[..., :, :, 1::2] = torch.exp(-scl) * dat[..., :, :, 1::2]
    elif dim == 1:
        dat_out[..., :, ::2, :] = torch.exp(scl) * dat[..., :, ::2, :]
        dat_out[..., :, 1::2, :] = torch.exp(-scl) * at[..., :, 1::2, :]
    else:
        dat_out[..., ::2, :, :] = torch.exp(scl) * dat[..., ::2, :, :]
        dat_out[..., 1::2, :, :] = torch.exp(-scl) * dat[..., 1::2, :, :]

    return dat_out


def check_adjoint(po, method, dtype=torch.float32):
    """ Print adjointness of A and At operators:
        <Ay, x> - <Atx, y> \approx 0

    Args:
        po (ProjOp()): Encodes projection operator.
        method (string): Either 'denoising' or 'super-resolution'.
        dtype (torch.dtype, optional)

    """
    dim_x = po.dim_x
    dim_y = po.dim_y
    device = po.smo_ker.device
    torch.manual_seed(0)
    x = torch.rand((1, 1,) + dim_x, dtype=dtype, device=device)
    y = torch.rand((1, 1,) + dim_y, dtype=dtype, device=device)
    po.smo_ker = po.smo_ker.type(dtype)
    po.scl = po.scl.type(dtype)
    # Apply A and At operators
    Ay = proj_apply('A', method, y, po)
    Atx = proj_apply('At', method, x, po)
    # Check okay
    val = torch.sum(Ay * x, dtype=torch.float64) - torch.sum(Atx * y, dtype=torch.float64)
    # Print okay
    print('<Ay, x> - <Atx, y> = {}'.format(val))


def even_odd(dat, which, dim):
    """ TODO

    """
    if dim == 2 and which == 'odd':
        return dat[:, :, ::2]
    elif dim == 2 and which == 'even':
        return dat[:, :, 1::2]
    elif dim == 1 and which == 'odd':
        return dat[:, ::2, :]
    elif dim == 1 and which == 'even':
        return dat[:, 1::2, :]
    elif dim == 0 and which == 'odd':
        return dat[::2, :, :]
    elif dim == 0 and which == 'even':
        return dat[1::2, :, :]


def proj_apply(operator, method, dat, po, bound='dct2', interpolation=1):
    """ Applies operator A, At  or AtA (for denoising or super-resolution).

    Args:
        operator (string): Either 'A', 'At', 'AtA' or 'none'.
        method (string): Either 'denoising' or 'super-resolution'.
        dat (torch.tensor()): Image data (1, 1, X_in, Y_in, Z_in).
        po (ProjOp()): Encodes projection operator, has the following fields:
            po.mat_x: Low-res affine matrix.
            po.mat_y: High-res affine matrix.
            po.mat_yx: Intermediate affine matrix.
            po.dim_x: Low-res image dimensions.
            po.dim_y: High-res image dimensions.
            po.dim_yx: Intermediate image dimensions.
            po.ratio: The ratio (low-res voxsize)/(high-res voxsize).
            po.smo_ker: Smoothing kernel (slice-profile).
        bound (str, optional): Bound for nitorch push/pull, defaults to 'zero'.
        interpolation (int, optional): Interpolation order, defaults to 1 (linear).

    Returns:
        dat (torch.tensor()): Projected image data (1, 1, X_out, Y_out, Z_out).

    """
    # Sanity check
    if operator not in ['A', 'At', 'AtA', 'none']:
        raise ValueError('Undefined operator')
    if method not in ['denoising', 'super-resolution']:
        raise ValueError('Undefined method')
    if operator == 'none':
        # No projection
        return dat
    # Get data type and device
    dtype = dat.dtype
    device = dat.device
    # Parse required projection info
    mat_x = po.mat_x
    mat_y = po.mat_y
    mat_yx = po.mat_yx
    rigid = po.rigid
    dim_x = po.dim_x
    dim_y = po.dim_y
    dim_yx = po.dim_yx
    ratio = po.ratio
    smo_ker = po.smo_ker
    scl = po.scl
    dim_thick = po.dim_thick
    if method == 'super-resolution':
        dim = dim_yx
        mat = rigid.mm(mat_yx).solve(mat_y)[0]  # mat_y\rigid*mat_yx
    elif method == 'denoising':
        dim = dim_x
        mat = rigid.mm(mat_x).solve(mat_y)[0]  # mat_y\rigid*mat_x
    # Get grid
    grid = affine(dim, mat, device=device, dtype=dtype)
    # Apply projection
    if method == 'super-resolution':
        extrapolate = True
        if operator == 'A':
            dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
            dat = F.conv3d(dat, smo_ker, stride=ratio)
            if scl != 0:
                dat = apply_scaling(dat, scl, dim_thick)
        elif operator == 'At':
            if scl != 0:
                dat = apply_scaling(dat, scl, dim_thick)
            dat = F.conv_transpose3d(dat, smo_ker, stride=ratio)
            dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
        elif operator == 'AtA':
            dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
            dat = F.conv3d(dat, smo_ker, stride=ratio)
            if scl != 0:
                dat = apply_scaling(dat, 2 * scl, dim_thick)
            dat = F.conv_transpose3d(dat, smo_ker, stride=ratio)
            dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
    elif method == 'denoising':
        extrapolate = False
        if operator == 'A':
            dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
        elif operator == 'At':
            dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
        elif operator == 'AtA':
            dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
            dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate, interpolation=interpolation)

    return dat


def proj_info(dim_y, mat_y, dim_x, mat_x, rigid=None,
              prof_ip=0, prof_tp=0, gap=0.0, device='cpu', scl=0.0,
              samp=0):
    """ Define projection operator object, to be used with proj_apply.

    Args:
        dim_y ((int, int, int))): High-res image dimensions (3,).
        mat_y (torch.tensor): High-res affine matrix (4, 4).
        dim_x ((int, int, int))): Low-res image dimensions (3,).
        mat_x (torch.tensor): Low-res affine matrix (4, 4).
        rigid (torch.tensor): Rigid transformation aligning x to y (4, 4), defaults to eye(4).
        prof_ip (int, optional): In-plane slice profile (0=rect|1=tri|2=gauss), defaults to 0.
        prof_tp (int, optional): Through-plane slice profile (0=rect|1=tri|2=gauss), defaults to 0.
        gap (float, optional): Slice-gap between 0 and 1, defaults to 0.
        device (torch.device, optional): Device. Defaults to 'cpu'.
        scl (float, optional): Odd/even slice scaling, defaults to 0.

    Returns:
        po (ProjOp()): Projection operator object.

    """
    # Get projection operator object
    po = ProjOp()
    # Data types
    dtype = torch.float64
    dtype_smo_ker = torch.float32
    one = torch.tensor([1, 1, 1], device=device, dtype=torch.float64)
    # Output properties
    po.dim_y = torch.tensor(dim_y, device=device, dtype=dtype)
    po.mat_y = mat_y
    po.vx_y = voxsize(mat_y)
    # Input properties
    po.dim_x = torch.tensor(dim_x, device=device, dtype=dtype)
    po.mat_x = mat_x
    po.vx_x = voxsize(mat_x)
    if rigid is None:
        po.rigid = torch.eye(4, device=device, dtype=dtype)
    else:
        po.rigid = rigid.type(dtype).to(device)
    # Slice-profile
    gap_cn = torch.zeros(3, device=device, dtype=dtype)
    profile_cn = torch.tensor((prof_ip,) * 3, device=device, dtype=dtype)
    dim_thick = torch.max(po.vx_x, dim=0)[1]
    gap_cn[dim_thick] = gap
    profile_cn[dim_thick] = prof_tp
    po.dim_thick = dim_thick
    if samp > 0:
        # Sub-sampling
        samp = torch.tensor((samp,) * 3, device=device, dtype=torch.float64)
        # Intermediate to lowres
        sk = torch.max(one, torch.floor(samp * one / po.vx_x + 0.5))
        D_x = torch.diag(torch.cat((sk, one[0, None])))
        po.D_x = D_x
        # Modulate lowres
        po.mat_x = po.mat_x.mm(D_x)
        po.dim_x = D_x.inverse()[:3, :3].mm(po.dim_x.reshape((3, 1))).floor().squeeze()
        if torch.sum(torch.abs(po.vx_x - po.vx_x)) > 1e-4:
            # Intermediate to highres (only for superres)
            sk = torch.max(one, torch.floor(samp * one / po.vx_y + 0.5))
            D_y = torch.diag(torch.cat((sk, one[0, None])))
            po.D_y = D_y
            # Modulate highres
            po.mat_y = po.mat_y.mm(D_y)
            po.vx_y = voxsize(po.mat_y)
            po.dim_y = D_y.inverse()[:3, :3].mm(po.dim_y.reshape((3, 1))).floor().squeeze()
        po.vx_x = voxsize(po.mat_x)
    # Make intermediate
    ratio = torch.solve(po.mat_x, po.mat_y)[0]  # mat_y\mat_x
    ratio = (ratio[:3, :3] ** 2).sum(0).sqrt()
    ratio = ratio.ceil().clamp(1)  # ratio low/high >= 1
    mat_yx = torch.cat((ratio, torch.ones(1, device=device, dtype=dtype))).diag()
    po.mat_yx = po.mat_x.matmul(mat_yx.inverse())  # mat_x/mat_yx
    po.dim_yx = (po.dim_x - 1) * ratio + 1
    # Make elements with ratio <= 1 use dirac profile
    profile_cn[ratio == 1] = -1
    profile_cn = profile_cn.int().tolist()
    # Make smoothing kernel (slice-profile)
    fwhm = (1. - gap_cn) * ratio
    smo_ker = smooth(profile_cn, fwhm, sep=False, dtype=dtype_smo_ker, device=device)
    po.smo_ker = smo_ker
    # Add offset to intermediate space
    off = torch.tensor(smo_ker.shape[-3:], dtype=dtype, device=device)
    off = -(off - 1) // 2  # set offset
    mat_off = torch.eye(4, dtype=torch.float64, device=device)
    mat_off[:3, -1] = off
    po.dim_yx = po.dim_yx + 2 * torch.abs(off)
    po.mat_yx = torch.matmul(po.mat_yx, mat_off)
    # Odd/even slice scaling
    if isinstance(scl, torch.Tensor):
        po.scl = scl
    else:
        po.scl = torch.tensor(scl, dtype=torch.float32, device=device)
    # To tuple of ints
    po.dim_y = tuple(po.dim_y.int().tolist())
    po.dim_yx = tuple(po.dim_yx.int().tolist())
    po.dim_x = tuple(po.dim_x.int().tolist())
    po.ratio = tuple(ratio.int().tolist())

    return po


def read_image(data, device='cpu', is_ct=False):
    """ Reads image data.

    Args:
        data (string|list): Path to file, or list with image data and affine matrix.
        device (string, optional): PyTorch on CPU or GPU? Defaults to 'cpu'.
        is_ct (bool, optional): Is the image a CT scan?

    Returns:
        dat (torch.tensor()): Image data.
        dim (tuple(int)): Image dimensions.
        mat (torch.tensor(double)): Affine matrix.
        fname (string): File path
        direc (string): File directory path
        nam (string): Filename
        head (nibabel.nifti1.Nifti1Header)
        ct (bool): Is data CT?
        var (torch.tensor(float)): Observation uncertainty.

    """
    var = torch.tensor(0, dtype=torch.float32, device=device)  # Observation uncertainty
    if isinstance(data, str):
        # =================================
        # Load from file
        # =================================
        nii = nib.load(data)
        # Get affine matrix
        mat = nii.affine
        mat = torch.tensor(mat).double().to(device)
        # Get image data
        dat = torch.tensor(nii.get_fdata()).float().to(device)
        # Get header, filename, etc
        head = nii.get_header()
        fname = nii.get_filename()
        # Get input directory and filename
        direc, nam = os.path.split(fname)
        # Get observation uncertainty
        slope = nii.dataobj.slope
        dtype = nii.get_data_dtype()
        dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        if dtype in dtypes:
            var = torch.tensor(slope, dtype=torch.float32, device=device)
            var = var ** 2 / 12
    else:
        # =================================
        # Data and matrix given as list
        # =================================
        # Image data
        dat = data[0]
        if not isinstance(dat, torch.Tensor):
            dat = torch.tensor(dat)
        dat = dat.float()
        dat = dat.to(device)
        # Affine matrix
        mat = data[1]
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)
        mat = mat.double().to(device)
        head = None
        fname = None
        direc = None
        nam = None
    # Get dimensions
    dim = tuple(dat.shape)
    # Remove NaNs
    dat[~torch.isfinite(dat)] = 0
    if is_ct and (torch.min(dat) < 0):
        # Input data is CT
        ct = True
        # Winsorize CT
        dat[dat < -1024] = -1024
        dat[dat > 3071] = 3071
    else:
        ct = False

    return dat, dim, mat, fname, direc, nam, head, ct, var


def write_image(dat, ofname, mat=torch.eye(4), header=None, dtype='float32'):
    """ Writes 3D nifti data using nibabel.

    Args:
        dat (torch.tensor): Image data (W, H, D).
        ofname (str): Output filename.
        mat (torch.tensor, optional): Affine matrix (4, 4), defaults to identity.
        header (nibabel.nifti1.Nifti1Header, optional): nibabel header, defaults to None.
        dtype (str, optional): Output data type, defaults to 'float32', but uses the data type
            in the header (if given).
    """
    # Sanity check
    if dtype not in ['float32', 'uint8', 'int16', 'uint16']:
        raise ValueError('Undefined data type')
    # Get min and max
    mn = torch.min(dat)
    mx = torch.max(dat)
    if header is not None:
        # If input was integer type, make output integer type
        dtype = header.get_data_dtype()
        dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        if dtype in dtypes:
            dat = dat.int()
    # Make nii object
    nii = nib.Nifti1Image(dat.cpu().numpy(), header=header, affine=mat.cpu().numpy())
    if header is None:
        # Set data type
        header = nii.get_header()
        header.set_data_dtype(dtype)
        # Set offset, slope and intercept
        if dtype == 'float32':
            offset = 0
            slope = 1
            inter = 0
        elif dtype == 'uint8':
            offset = 0
            slope = (mx / 255).cpu().numpy()
            inter = 0
        elif dtype == 'int16':
            offset = 0
            slope = torch.max(mx / 32767, -mn / 32768).cpu().numpy()
            inter = 0
        elif dtype == 'uint16':
            offset = 0
            slope = torch.max(mx / 65535, -mn / 65535).cpu().numpy()
            inter = 0
        header.set_data_offset(offset=offset)
        header.set_slope_inter(slope=slope, inter=inter)
    # Write to  disk
    nib.save(nii, ofname)
