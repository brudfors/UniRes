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
    cgs_max_iter: int = 2  # Max conjugate gradient (CG) iterations for solving for y
    cgs_tol: float = 0  # CG tolerance for solving for y
    cgs_verbose: bool = False  # CG verbosity (0, 1)
    device: str = 'cuda'  # PyTorch device name
    gr_diff: str = 'forward'  # Gradient difference operator (forward|backward|central)
    dir_out: str = None  # Directory to write output, if None uses same as input (output is prefixed 'y_')
    gap: float = 0.0  # Slice gap, between 0 and 1
    has_ct: bool = True  # Data could be CT (but data must contain negative values)
    max_iter: int = 256  # Max algorithm iterations
    mod_prct: float = 0.0  # Amount to crop mean space, between 0 and 1 (faster, but could loss out on data)
    prefix: str = 'y_'  # Prefix for reconstructed image(s)
    print_info: int = 1  # Print progress to terminal (0, 1, 2)
    plot_conv: bool = False  # Use matplotlib to plot convergence in real-time
    profile_ip: int = 0  # In-plane slice profile (0=rect|1=tri|2=gauss)
    profile_tp: int = 0  # Through-plane slice profile (0=rect|1=tri|2=gauss)
    reg_ix_fix: int = 0  # Index of fixed image in initial co-reg (if zero, pick image with largest FOV)
    reg_scl: float = 20.0  # Scale regularisation estimate (for coarse-to-fine scaling, give as list of floats)
    rho: float = 1.7  # ADMM step-size, if None -> estimate is made
    rho_scl: float = 1.0  # Scaling of ADMM step-size
    rigid_samp: int = 3  # Level of sub-sampling for estimating rigid registration parameters
    scaling: bool = True  # Optimise even/odd slice scaling
    show_hyperpar: bool = False  # Use matplotlib to visualise hyper-parameter estimates
    show_jtv: bool = False  # Show the joint total variation (JTV)
    tolerance: float = 0  # Algorithm tolerance, if zero, run to max_iter
    unified_rigid: bool = True  # Do unified rigid registration
    vx: float = 1.0  # Reconstruction voxel size (if None, set automatically)
    write_jtv: bool = False  # Write JTV to nifti
    write_out: bool = True  # Write reconstructed output images


class Model:
    """ Model class
    """

    # Constructor
    def __init__(self, data, sett=Settings()):
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
        # Algorithm settings
        self.sett = sett  # self.sett is of class Settings()
        self._rho = None  # Infamous ADMM step-size
        self._do_proj = None  # Use projection matrices (set in _format_output())
        self._method = None  # Method name (super-resolution|denoising)
        self._rigid_basis = None  # Rigid transformation basis
        if not isinstance(self.sett.reg_scl, list):  # For coarse-to-fine scaling of regularisation
            self.sett.reg_scl = [self.sett.reg_scl]

        # Read and format data
        self._x = self._format_input(data)
        del data
        # Defines
        # self._x[c][n] = Input()
        # with fields:
        # self._x[c][n].dat
        # self._x[c][n].dim
        # self._x[c][n].mat
        # self._x[c][n].tau
        # self._x[c][n].sd
        # self._x[c][n].mu
        # self._x[c][n].fname
        # self._x[c][n].nam
        # self._x[c][n].direc
        # self._x[c][n].head

        # Init registration
        self._init_reg()
        # Defines:
        # self._rigid_basis
        # self._x[c][n].rigid_q
        # and modifies:
        # self._x[c][n].mat

        # Format output
        # self._y is of class Output()
        self._y = self._format_output()
        # Defines:
        # self._do_proj
        # self._method
        # self._y[c] = Output()
        # with fields:
        # self._y[c].lam0
        # self._y[c].lam
        # self._y[c].dim
        # self._y[c].mat

        # if self.sett.scaling or self.sett.unified_rigid:
        #     # Coarse-to-fine scaling of lambda
        #     mx = 6  # start scaling is 2^mx end is 2^0
        #     rep = 20  # how many times to update y for each scaling value
        #     # sched = torch.tensor(2.0, device=self.sett.device, dtype=torch.float32)\
        #     #         **torch.arange(0, mx, device=self.sett.device, dtype=torch.float32).flip(dims=(0,))
        #     sched = torch.tensor([50, 1], device=self.sett.device, dtype=torch.float32)
        #     sched = sched.view(-1, 1).repeat(1, rep).view(1, -1).squeeze()
        #     self.sett.reg_scl = self.sett.reg_scl[0]*sched

        # Define projection matrices
        self._proj_info_add()
        # Defines:
        # self._x[c][n].po = ProjOp()
        # with fields:
        # self._x[c][n].po.dim_x
        # self._x[c][n].po.mat_x
        # self._x[c][n].po.vx_x
        # self._x[c][n].po.dim_y
        # self._x[c][n].po.mat_y
        # self._x[c][n].po.vx_y
        # self._x[c][n].po.dim_yx
        # self._x[c][n].po.mat_yx
        # self._x[c][n].po.smo_ker
        # self._x[c][n].po.ratio
        # self._x[c][n].po.rigid
        # self._x[c][n].po.scl
        # self._x[c][n].po.dim_thick

        if False:
            # Check adjointness of A and At operators
            self.check_adjoint(po=self._x[0][0].po, method=self._method, dtype=torch.float64)

    # Class methods
    def fit(self):
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
        # Parse function settings
        device = self.sett.device
        dtype = torch.float32
        unified_rigid = self.sett.unified_rigid
        rigid_samp = self.sett.rigid_samp
        scaling = self.sett.scaling
        gr_diff = self.sett.gr_diff
        max_iter = self.sett.max_iter
        tol = self.sett.tolerance
        # Parameters
        C = len(self._y)  # Number of channels
        N = sum([len(x) for x in self._x])  # Number of observations
        dim_y = self._y[0].dim  # Output dimensions
        vx_y = voxsize(self._y[0].mat).float()  # Output voxel size
        bound_grad = 'constant'
        # Constants
        tiny = torch.tensor(1e-7, dtype=dtype, device=device)
        one = torch.tensor(1, dtype=dtype, device=device)
        # Over/under-relaxation parameter
        alpha = torch.tensor(self.sett.alpha, device=device, dtype=torch.float32)

        # Initial guess of reconstructed images (y)
        self._init_y()
        # Defines:
        # self._y[c].dat

        if max_iter > 0:
            # Get ADMM variables
            z, w = self._alloc_admm_vars()

        # Scale lambda
        for c in range(C):
            self._y[c].lam = self.sett.reg_scl[0] * self._y[c].lam0

        # Get ADMM step-size
        self._rho = self._step_size()

        # For visualisation
        fig_ax_nll = None
        fig_ax_jtv = None

        # Start iterating:
        # Updates y, z, w in alternating fashion, until a convergence threshold is met
        # on the model negative log-likelihood.
        obj = torch.zeros(3*max_iter, 3, dtype=torch.float64, device=device)
        cnt_obj = 0
        jtv = None
        tmp = torch.zeros_like(self._y[0].dat)  # for holding rhs in y-update, and jtv in u-update
        t_iter = timer() if self.sett.print_info else 0
        for n_iter in range(max_iter):

            if n_iter == 0:
                t00 = self._print_info('fit-start', C, N, device,
                                       max_iter, tol)  # PRINT

            # Coarse-to-fine scaling of lambda
            if n_iter < len(self.sett.reg_scl):
                for c in range(C):
                    self._y[c].lam = self.sett.reg_scl[n_iter] * self._y[c].lam0
                # Update ADMM step-size
                self._rho = self._step_size(verbose=False)

            # ----------
            # UPDATE: y
            # ----------
            t0 = self._print_info('fit-update', 'y', n_iter)  # PRINT
            for c in range(C):  # Loop over channels
                # RHS
                num_x = len(self._x[c])
                tmp[:] = 0
                for n in range(num_x):  # Loop over observations of channel 'c'
                    # _ = self._print_info('int', n)  # PRINT
                    tmp += self._x[c][n].tau * self._proj('At', self._x[c][n].dat, c, n)

                # Divergence
                div = w[c, ...] - self._rho * z[c, ...]
                div = im_divergence(div, vx=vx_y, bound=bound_grad, which=gr_diff)
                tmp -= self._y[c].lam * div

                # Invert y = lhs\tmp by conjugate gradients
                lhs = lambda y: self._proj('AtA', y, c, vx_y=vx_y, bound__DtD=bound_grad, gr_diff=gr_diff)
                self._y[c].dat = cg(A=lhs, b=tmp, x=self._y[c].dat,
                                    verbose=self.sett.cgs_verbose,
                                    max_iter=self.sett.cgs_max_iter,
                                    tolerance=self.sett.cgs_tol)

                _ = self._print_info('int', c)  # PRINT

            _ = self._print_info('fit-done', t0)  # PRINT

            # ----------
            # UPDATE: z
            # ----------
            if alpha != 1:  # Use over/under-relaxation
                z_old = z.clone()
            t0 = self._print_info('fit-update', 'z', n_iter)  # PRINT
            tmp[:] = 0
            for c in range(C):
                Dy = self._y[c].lam * im_gradient(self._y[c].dat, vx=vx_y, bound=bound_grad, which=gr_diff)
                if alpha != 1:  # Use over/under-relaxation
                    Dy = alpha * Dy + (one - alpha) * z_old[c, ...]
                tmp += torch.sum((w[c, ...] / self._rho + Dy) ** 2, dim=0)
            tmp.sqrt_()  # in-place
            tmp = ((tmp - one / self._rho).clamp_min(0)) / (tmp + tiny)

            if self.sett.show_jtv:  # Show computed JTV
                fig_ax_jtv = show_slices(img=tmp, fig_ax=fig_ax_jtv, title='JTV',
                                         cmap='coolwarm', fig_num=98)

            for c in range(C):
                Dy = self._y[c].lam * im_gradient(self._y[c].dat, vx=vx_y, bound=bound_grad, which=gr_diff)
                if alpha != 1:  # Use over/under-relaxation
                    Dy = alpha * Dy + (one - alpha) * z_old[c, ...]
                for d in range(Dy.shape[0]):
                    z[c, d, ...] = tmp * (w[c, d, ...] / self._rho + Dy[d, ...])
            _ = self._print_info('fit-done', t0)  # PRINT

            # ----------
            # Compute model objective function
            # ----------
            if tol > 0:
                obj[cnt_obj, 0], obj[cnt_obj, 1], obj[cnt_obj, 2] \
                    = self._compute_nll(vx_y=vx_y, bound=bound_grad, gr_diff=gr_diff)  # nl_pyx, nl_pxy, nl_py
                cnt_obj += 1
            if self.sett.plot_conv:  # Plot algorithm convergence
                fig_ax_nll = plot_convergence(vals=obj[:cnt_obj, :], fig_ax=fig_ax_nll, fig_num=99)
            gain = get_gain(obj[:cnt_obj, 0], monotonicity='decreasing')
            t_iter = self._print_info('fit-ll', 'y', n_iter, obj[cnt_obj - 1, :], gain, t_iter)  # PRINT
            if n_iter > 1 and ((torch.abs(gain) < tol) or (n_iter >= (max_iter - 1))):
                _ = self._print_info('fit-finish', t00, n_iter)  # PRINT
                break  # Finished

            # ----------
            # UPDATE: w
            # ----------
            t0 = self._print_info('fit-update', 'w', n_iter)  # PRINT
            for c in range(C):  # Loop over channels
                Dy = self._y[c].lam * im_gradient(self._y[c].dat, vx=vx_y, bound=bound_grad, which=gr_diff)
                if alpha != 1:  # Use over/under-relaxation
                    Dy = alpha * Dy + (one - alpha) * z_old[c, ...]
                w[c, ...] += self._rho * (Dy - z[c, ...])
                _ = self._print_info('int', c)  # PRINT
            _ = self._print_info('fit-done', t0)  # PRINT

            if scaling and n_iter > 0 and not (n_iter % 2) == 0:
                # ----------
                # UPDATE: even/odd scaling
                # ----------
                t0 = self._print_info('fit-update', 'scaling', n_iter)  # PRINT
                # Do update
                obj1 = self._update_scaling(max_niter_gn=1, num_linesearch=4, verbose=0)
                _ = self._print_info('fit-done', t0)  # PRINT
                # # Compute objective function
                # obj[cnt_obj, 0] = obj1 + obj[cnt_obj - 1, 2]
                # obj[cnt_obj, 1] = obj1
                # obj[cnt_obj, 2] = obj[cnt_obj - 1, 2]
                # cnt_obj += 1
                # if self.sett.plot_conv:  # Plot algorithm convergence
                #     fig_ax_nll = plot_convergence(vals=obj[:cnt_obj, :], fig_ax=fig_ax_nll, fig_num=99)
                # gain = get_gain(obj[:cnt_obj, 0], monotonicity='decreasing')
                # t_iter = self._print_info('fit-ll', 's', n_iter, obj[cnt_obj - 1, :], gain, t_iter)  # PRINT

            if unified_rigid and n_iter > 0 and (n_iter % 2) == 0:
                # ----------
                # UPDATE: rigid_q
                # ----------
                t0 = self._print_info('fit-update', 'rigid', n_iter)  # PRINT
                # Do update
                obj1 = self._update_rigid(mean_correct=True, max_niter_gn=1, num_linesearch=4,
                                          verbose=0, samp=rigid_samp)
                _ = self._print_info('fit-done', t0)  # PRINT
                # # Compute objective function
                # obj[cnt_obj, 0] = obj1 + obj[cnt_obj - 1, 2]
                # obj[cnt_obj, 1] = obj1
                # obj[cnt_obj, 2] = obj[cnt_obj - 1, 2]
                # cnt_obj += 1
                # if self.sett.plot_conv:  # Plot algorithm convergence
                #     fig_ax_nll = plot_convergence(vals=obj[:cnt_obj, :], fig_ax=fig_ax_nll, fig_num=99)
                # gain = get_gain(obj[:cnt_obj, 0], monotonicity='decreasing')
                # t_iter = self._print_info('fit-ll', 'q', n_iter, obj[cnt_obj - 1, :], gain, t_iter)  # PRINT

        # Print parameter estimates
        if scaling and self.sett.print_info >= 1:
            _ = self._print_info('scl-param', t0)  # PRINT
        if unified_rigid and self.sett.print_info >= 1:
            _ = self._print_info('reg-param', t0)  # PRINT

        # Get rigid matrices
        R = torch.zeros((4, 4, N), device=device, dtype=torch.float64)
        n = 0
        for c in range(C):
            num_cn = len(self._x[c])
            for cn in range(num_cn):
                R[..., n] = dexpm(self._x[c][cn].rigid_q, self._rigid_basis)[0]
                n += 1

        # Process reconstruction results
        y, mat, pth_y = self._write_data(jtv=tmp)

        return y, mat, pth_y, R

    def _all_mat_dim_vx(self):
        """ Get all images affine matrices, dimensions and voxel sizes (as numpy arrays).

        Returns:
            all_mat (torch.tensor): Image orientation matrices (4, 4, N).
            Dim (torch.tensor): Image dimensions (3, N).
            all_vx (torch.tensor): Image voxel sizes (3, N).

        """
        # Parse function settings
        device = self.sett.device
        dtype = torch.float64

        N = sum([len(x) for x in self._x])
        all_mat = torch.zeros((4, 4, N), device=device, dtype=dtype)
        all_dim = torch.zeros((3, N), device=device, dtype=dtype)
        all_vx = torch.zeros((3, N), device=device, dtype=dtype)

        cnt = 0
        for c in range(len(self._x)):
            for n in range(len(self._x[c])):
                all_mat[..., cnt] = self._x[c][n].mat
                all_dim[..., cnt] = torch.tensor(self._x[c][n].dim, device=device, dtype=dtype)
                all_vx[..., cnt] = voxsize(self._x[c][n].mat)
                cnt += 1

        return all_mat, all_dim, all_vx

    def _alloc_admm_vars(self):
        """ Get ADMM variables z and w.

        Returns:
            z (torch.tensor()): (C, 3, dim_y)
            w (torch.tensor()): (C, 3, dim_y)

        """
        # Parse function settings/parameters
        device = self.sett.device
        dtype = torch.float32
        C = len(self._y)
        dim_y = self._y[0].dim
        dim = (C, 3) + dim_y
        # Allocate
        z = torch.zeros(dim, dtype=dtype, device=device)
        w = torch.zeros(dim, dtype=dtype, device=device)

        return z, w

    def _compute_nll(self, vx_y, sum_dtype=torch.float64, bound='constant', gr_diff='forward'):
        """ Compute negative model log-likelihood.

        Args:
            vx_y (tuple(float)): Output voxel size.
            sum_dtype (torch.dtype): Defaults to torch.float64.
            bound (str, optional): Bound for gradient/divergence calculation, defaults to
                constant zero.
            gr_diff (str, optional): Gradient difference operator, defaults to 'forward'.

        Returns:
            nll_yx (torch.tensor()): Negative log-posterior
            nll_xy (torch.tensor()): Negative log-likelihood.
            nll_y (torch.tensor()): Negative log-prior.

        """
        device = self.sett.device
        dtype = torch.float32

        C = len(self._y)
        nll_xy = 0.0
        for c in range(C):
            num_x = len(self._x[c])
            # Neg. log-likelihood term
            for n in range(num_x):
                msk = self._x[c][n].dat != 0
                nll_xy += 0.5 * self._x[c][n].tau * torch.sum((self._x[c][n].dat[msk] -
                                                               self._proj('A', self._y[c].dat, c, n)[msk]) ** 2,
                                                              dtype=sum_dtype)
            # Neg. log-prior term
            Dy = self._y[c].lam * im_gradient(self._y[c].dat, vx=vx_y, bound=bound, which=gr_diff)
            if c > 0:
                nll_y += torch.sum(Dy ** 2, dim=0, dtype=dtype)
            else:
                nll_y = torch.sum(Dy ** 2, dim=0, dtype=dtype)

        nll_y = torch.sum(torch.sqrt(nll_y), dtype=sum_dtype)

        return nll_xy + nll_y, nll_xy, nll_y

    def _DtD(self, dat, vx_y, bound='constant', gr_diff='forward'):
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

    def _estimate_hyperpar(self, x):
        """ Estimate noise precision (tau) and mean brain
            intensity (mu) of each observed image.

        Args:
            x (Input()): Input data.

        Returns:
            tau (list): List of C torch.tensor(float) with noise precision of each MR image.
            lam (torch.tensor(float)): The parameter lambda (1, C).

        """
        # Parse function settings
        show_hyperpar = self.sett.show_hyperpar

        # Print info to screen
        t0 = self._print_info('hyper_par')

        # Do estimation
        cnt = 0
        C = len(x)
        for c in range(C):
            num_x = len(x[c])
            for n in range(num_x):
                # Get data
                dat = x[c][n].dat
                if x[c][n].ct:
                    # Estimate noise sd from estimate of FWHM
                    sd_bg = estimate_fwhm(dat, voxsize(x[c][n].mat), mn=20, mx=50)[1]
                    mu_bg = torch.tensor(0.0, device=dat.device, dtype=dat.dtype)
                    mu_fg = torch.tensor(2000.0, device=dat.device, dtype=dat.dtype)
                else:
                    # Get noise and foreground statistics
                    sd_bg, sd_fg, mu_bg, mu_fg = noise_estimate(dat,
                                                                num_class=2, show_fit=show_hyperpar, fig_num=100 + cnt)
                # Set values
                x[c][n].sd = sd_bg.float()
                x[c][n].tau = 1 / sd_bg.float() ** 2
                x[c][n].mu = torch.abs(mu_fg.float() - mu_bg.float())
                cnt += 1

        # Print info to screen
        self._print_info('hyper_par', x, t0)

        return x

    def _format_input(self, data):
        """ Construct algorithm input struct.

        Args:
            data

        Returns:
            x (Input()): Formatted algorithm input struct(s).

        """
        # Parse input data into Input() object, filling the following fields:
        # x[c][n].dat
        # x[c][n].dim
        # x[c][n].mat
        # x[c][n].ct
        # x[c][n].fname
        # x[c][n].direc
        # x[c][n].nam
        # x[c][n].head
        x = self._read_data(data)

        # Estimate input image statistics, filling the following fields of Input():
        # x[c][n].tau
        # x[c][n].mu
        # x[c][n].sd
        x = self._estimate_hyperpar(x)

        return x

    def _format_output(self):
        """ Construct algorithm output struct. See Output() dataclass.

        Returns:
            y (Output()): Algorithm output struct(s).

        """
        # Parse function settings
        C = len(self._x)  # Number of channels
        device = self.sett.device
        mod_prct = self.sett.mod_prct

        one = torch.tensor(1.0, device=device, dtype=torch.float64)
        vx_y = self.sett.vx  # output voxel size
        if vx_y is not None:
            if isinstance(vx_y, int):
                vx_y = float(vx_y)
            if isinstance(vx_y, float):
                vx_y = (vx_y,) * 3
            vx_y = torch.tensor(vx_y, dtype=torch.float64, device=device)

        # Get all orientation matrices and dimensions
        all_mat, all_dim, all_vx = self._all_mat_dim_vx()
        N = all_mat.shape[-1]  # Total number of observations

        # Do unified rigid registration?
        if N == 1:
            self.sett.unified_rigid = False
        unified_rigid = self.sett.unified_rigid

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
        self._do_proj = True
        if vx_y is None and ((N == 1) or vx_same):  # One image, voxel size not given
            vx_y = all_vx[..., 0]

        if vx_same and (torch.abs(all_vx[..., 0] - vx_y) < 1e-3).all():
            # All input images have same voxel size, and output voxel size is the also the same
            do_sr = False
            if mat_same and dim_same and not unified_rigid:
                # All input images have the same FOV
                mat = all_mat[..., 0]
                dim = all_dim[..., 0]
                self._do_proj = False

        if do_sr or self._do_proj:
            # Get FOV of mean space
            if N == 1 and do_sr:
                D = torch.diag(torch.cat((vx_y / all_vx[:, 0], one[..., None])))
                mat = all_mat[..., 0].mm(D)
                dim = D.inverse()[:3, :3].mm(all_dim[:, 0].reshape((3, 1))).floor().squeeze()
            else:
                # Mean space from several images
                dim, mat, _ = mean_space(all_mat, all_dim, vx=vx_y, mod_prct=-mod_prct)
        if do_sr:
            self._method = 'super-resolution'
        else:
            self._method = 'denoising'


        # Optimise even/odd scaling parameter?
        if self._method == 'denoising':
            self.sett.scaling = False

        dim = tuple(dim.int().tolist())
        _ = self._print_info('mean-space', dim, mat)

        # CT lambda fudge factor
        ff_ct = 1.0  # Just a CT (no fudge needed)
        if N > 1:
            # CT and MRIs
            ff_ct = 15.0

        # Assign output
        y = []
        for c in range(C):
            y.append(Output())
            # Regularisation (lambda) for channel c
            num_x = len(self._x[c])
            mu_c = torch.zeros(num_x, dtype=torch.float32, device=device)
            for n in range(num_x):
                mu_c[n] = self._x[c][n].mu
            y[c].lam0 = 1 / torch.mean(mu_c)
            y[c].lam = 1 / torch.mean(mu_c)  # To facilitate rescaling
            if self._x[c][0].ct:
                y[c].lam0 *= ff_ct
                y[c].lam *= ff_ct
            # Output image(s) dimension and orientation matrix
            y[c].dim = dim
            y[c].mat = mat.double().to(device)

        return y

    def _init_reg(self):
        """ Initialise rigid registration.

        """
        ix_fix = self.sett.reg_ix_fix
        device = self.sett.device
        # Init rigid basis
        rigid_basis = affine_basis('SE', 3, device=device, dtype=torch.float64)
        self._rigid_basis = rigid_basis
        # Coreg images
        # TODO
        # Init parameterisation of rigid transformation and apply initial coreg
        C = len(self._x)  # Number of channels
        for c in range(C):  # Loop over channels
            num_x = len(self._x[c])  # Number of observations of channel c
            for n in range(num_x):  # Loop over observations of channel c
                self._x[c][n].rigid_q = torch.zeros(6, device=device, dtype=torch.float64)
                self._x[c][n].armijo = torch.tensor(1.0, device=device, dtype=torch.float64)
                # self._x[c][n].mat = mat.solve(R[c][n])  # TODO: Apply rigid transformation

    def _init_y(self, interpolation=1):
        """ Make initial guesses of reconstucted image(s) using b-spline interpolation,
            with averaging if more than one observation per channel.

        Args:
            interpolation (int, optional): Interpolation order, defaults to 1.

        """
        C = len(self._x)
        dim_y = self._x[0][0].po.dim_y
        mat_y = self._x[0][0].po.mat_y
        for c in range(C):
            y = torch.zeros(dim_y, dtype=torch.float32, device=self.sett.device)
            num_x = 1.0
            if self.sett.max_iter == 0:
                # No superres will be performed, so generate the isotropic images using b-splines
                num_x = len(self._x[c])
                for n in range(num_x):
                    # Get image data
                    dat = self._x[c][n].dat[None, None, ...]
                    # Make output grid
                    mat = mat_y.solve(self._x[c][n].po.mat_x)[0]  # mat_x\mat_y
                    grid = affine(self._x[c][n].po.dim_y, mat, device=dat.device, dtype=dat.dtype)
                    # Do interpolation
                    mn = torch.min(dat)
                    mx = torch.max(dat)
                    dat = grid_pull(dat, grid, bound='zero', extrapolate=False, interpolation=interpolation)
                    dat[dat < mn] = mn
                    dat[dat > mx] = mx
                    y = y + dat[0, 0, ...]
            self._y[c].dat = y / num_x

    def _print_info(self, info, *argv):
        """ Print algorithm info to terminal.

        Args:
            info (string): What to print.

        """
        if not self.sett.print_info:
            return 0

        if self.sett.print_info >= 1:
            if info == 'fit-finish':
                print(' {} finished in {:0.5f} seconds and '
                      '{} iterations\n'.format(self._method, timer() - argv[0], argv[1] + 1))
            elif info in 'fit-ll':
                print('{:3} - Convergence ({} | {:0.1f} s)  | nlyx={:0.4f}, nlxy={:0.4f}, nly={:0.4f} '
                      'gain={:0.7f}'.format(argv[1] + 1, argv[0], timer() - argv[4], argv[2][0], argv[2][1], argv[2][2],
                                            argv[3]))
            elif info == 'fit-start':
                print('\nStarting {} \n{} | C={} | N={} | device={} | '
                      'max_iter={} | tol={}'.format(self._method, datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                    argv[0], argv[1], argv[2], argv[3], argv[4]))
            elif info in 'step_size':
                print('\nADMM step-size={:0.4f}'.format(argv[0]))
            elif info in 'reg-param':
                print('Rigid registration fit:')
                for c in range(len(self._x)):
                    for n in range(len(self._x[c])):
                        print('c={} n={} | q={}'.format(c, n, round(self._x[c][n].rigid_q, 4).cpu().tolist()))
                print('')
            elif info in 'scl-param':
                print('Scale fit:')
                for c in range(len(self._x)):
                    for n in range(len(self._x[c])):
                        print('c={} n={} | exp(s)={}'.format(c, n, round(self._x[c][n].po.scl.exp(), 4)))
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
        if self.sett.print_info >= 2:
            if info == 'fit-done':
                print('(completed in {:0.5f} seconds)'.format(timer() - argv[0]))
            elif info == 'fit-update':
                print('{:3} - Updating {:2}   | '.format(argv[1] + 1, argv[0]), end='')
            elif info == 'int':
                print('{}'.format(argv[0]), end=' ')

        return timer()

    def _proj(self, operator, dat, c=0, n=0, vx_y=None, bound__DtD='constant', gr_diff='forward'):
        """ Projects image data by A, At or AtA.

        Args:
            operator (string): Either 'A', 'At ' or 'AtA'.
            dat (torch.tensor()): Image data (dim_x|dim_y).
            c (int): Channel index, defaults to 0.
            n (int): Observation index, defaults to 0.
            vx_y (tuple(float)): Output voxel size.
            bound__DtD (str, optional): Bound for gradient/divergence calculation, defaults to
                constant zero.
            gr_diff (str, optional): Gradient difference operator, defaults to 'forward'.

        Returns:
            dat (torch.tensor()): Projected image data (dim_y|dim_x).

        """
        # Parse function parameters/settings
        rho = self._rho
        method = self._method
        do_proj = self._do_proj
        # Project
        if operator == 'AtA':
            if not do_proj:  operator = 'none'  # self.proj_apply returns dat
            dat1 = rho * self._y[c].lam ** 2 * self._DtD(dat, vx_y=vx_y, bound=bound__DtD, gr_diff=gr_diff)
            dat = dat[None, None, ...]
            dat = self._x[c][n].tau * self.proj_apply(operator, method, dat, self._x[c][n].po)
            num_x = len(self._x[c])
            for n1 in range(1, num_x):
                dat = dat + self._x[c][n1].tau * self.proj_apply(operator, method, dat, self._x[c][n1].po)
            dat = dat[0, 0, ...]
            dat += dat1
        else:  # A, At
            if not do_proj:  operator = 'none'  # self.proj_apply returns dat
            dat = dat[None, None, ...]
            dat = self.proj_apply(operator, method, dat, self._x[c][n].po)
            dat = dat[0, 0, ...]

        return dat

    def _proj_info_add(self):
        """ Adds a projection matrix encoding to each input (x).

        """
        # Parse function parameters/settings
        device = self.sett.device
        gap = self.sett.gap
        prof_ip = self.sett.profile_ip
        prof_tp = self.sett.profile_tp
        rigid_basis = self._rigid_basis
        C = len(self._x)
        # Build each projection operator
        for c in range(C):
            dim_y = self._y[c].dim
            mat_y = self._y[c].mat
            num_x = len(self._x[c])
            for n in range(num_x):
                # Get rigid matrix
                rigid = dexpm(self._x[c][n].rigid_q, rigid_basis)[0]
                # Define projection operator
                po = self.proj_info(dim_y, mat_y, self._x[c][n].dim, self._x[c][n].mat,
                                    prof_ip=prof_ip, prof_tp=prof_tp, gap=gap, device=device,
                                    rigid=rigid)
                # Assign
                self._x[c][n].po = po

    def _read_data(self, data):
        """ Parse input data into algorithm input struct(s).

        Args:
            data

        Returns:
            x (Input()): Algorithm input struct(s).

        """
        # Parse function settings
        device = self.sett.device
        has_ct = self.sett.has_ct

        if isinstance(data, str):
            data = [data]
        C = len(data)  # Number of channels

        x = []
        for c in range(C):  # Loop over channels
            x.append([])
            x[c] = []
            if isinstance(data[c], list) and (isinstance(data[c][0], str) or isinstance(data[c][0], list)):
                num_x = len(data[c])  # Number of observations of channel c
                for n in range(num_x):  # Loop over observations of channel c
                    x[c].append(Input())
                    # Get data
                    dat, dim, mat, fname, direc, nam, head, ct, _ = \
                        self.read_image(data[c][n], device, is_ct=has_ct)
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
                x[c].append(Input())
                n = 0  # One repeat per channel
                # Get data
                dat, dim, mat, fname, direc, nam, head, ct, _ = \
                    self.read_image(data[c], device, is_ct=has_ct)
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

    def _step_size(self, verbose=True):
        """ ADMM step size (rho) from image statistics.

        Args:
            verbose (bool, optional): Defaults to True.

        Returns:
            rho (torch.tensor()): Step size.

        """
        # Parse function settings
        device = self.sett.device
        rho = self.sett.rho
        scl = self.sett.rho_scl
        dtype = torch.float32

        if rho is not None:
            rho = torch.tensor(rho, device=device, dtype=dtype)
        else:
            C = len(self._y)
            N = sum([len(x) for x in self._x])
            all_lam = torch.zeros(C, dtype=dtype, device=device)
            all_tau = torch.zeros(N, dtype=dtype, device=device)
            cnt = 0
            for c in range(len(self._x)):
                all_lam[c] = self._y[c].lam
                for n in range(len(self._x[c])):
                    all_tau[cnt] = self._x[c][n].tau
                    cnt += 1
            rho = scl * torch.sqrt(torch.mean(all_tau)) / torch.mean(all_lam)
        if verbose:
            _ = self._print_info('step_size', rho)  # PRINT
        return rho

    def _update_scaling(self, max_niter_gn=1, num_linesearch=4, verbose=0):
        """ Updates an even/odd slice scaling parameter using Gauss-Newton
            optimisation.

        Args:
            verbose (bool, optional): Verbose for testing, defaults to False.

        Returns:
            sll (torch.tensor): Log-likelihood.

        """
        # Parameters
        device = self.sett.device
        rigid_basis = self._rigid_basis
        C = len(self._x)

        # Update rigid parameters, for all input images (self._x[c][n])
        sll = torch.tensor(0, device=device, dtype=torch.float64)
        for c in range(C):  # Loop over channels
            num_x = len(self._x[c])
            for n_x in range(num_x):  # Loop over repeats
                if self._x[c][n_x].ct:
                    # Do not optimise scaling for CT data
                    continue
                # Parameters
                dim_thick = self._x[c][n_x].po.dim_thick
                tau = self._x[c][n_x].tau
                scl = self._x[c][n_x].po.scl
                smo_ker = self._x[c][n_x].po.smo_ker
                dim_thick = self._x[c][n_x].po.dim_thick
                ratio = self._x[c][n_x].po.ratio
                dim = self._x[c][n_x].po.dim_yx
                mat_yx = self._x[c][n_x].po.mat_yx
                mat_y = self._x[c][n_x].po.mat_y
                rigid = dexpm(self._x[c][n_x].rigid_q, rigid_basis)[0]
                mat = rigid.mm(mat_yx).solve(mat_y)[0]  # mat_y\rigid*mat_yx
                # Observed data
                dat_x = self._x[c][n_x].dat
                msk = dat_x != 0
                # Get even/odd data
                xo = self.even_odd(dat_x, 'odd', dim_thick)
                mo = self.even_odd(msk, 'odd', dim_thick)
                xo = xo[mo]
                xe = self.even_odd(dat_x, 'even', dim_thick)
                me = self.even_odd(msk, 'even', dim_thick)
                xe = xe[me]
                # Get reconstruction (without scaling)
                grid = affine(dim, mat, device=device, dtype=torch.float32)
                dat_y = grid_pull(self._y[c].dat[None, None, ...], grid, bound='dct2', extrapolate=True)
                dat_y = F.conv3d(dat_y, smo_ker, stride=ratio)[0, 0, ...]
                # Apply scaling
                dat_y = self.apply_scaling(dat_y, scl, dim_thick)

                for n_gn in range(max_niter_gn):  # Loop over Gauss-Newton iterations
                    # Log-likelihood
                    ll = 0.5 * tau * torch.sum((dat_x[msk] - dat_y[msk]) ** 2, dtype=torch.float64)

                    if verbose >= 2:  # Show images
                        show_slices(torch.stack((dat_x, dat_y, (dat_x - dat_y) ** 2), 3),
                                    fig_num=666, colorbar=False, flip=False)

                    # Get even/odd data
                    yo = self.even_odd(dat_y, 'odd', dim_thick)
                    yo = yo[mo]
                    ye = self.even_odd(dat_y, 'even', dim_thick)
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
                    armijo = torch.tensor(1.0, device=device, dtype=old_scl.dtype)
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
                            dat_y = self.apply_scaling(dat_y, scl - old_scl, dim_thick)
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
                self._x[c][n_x].po.scl = scl
                # Accumulate neg log-lik
                sll += ll
        # Return neg log-lik
        return sll

    def _update_rigid(self, mean_correct=True, max_niter_gn=1, num_linesearch=4, verbose=0, samp=3):
        """ Updates each input image's specific registration parameters:
                self._x[c][n].rigid_q
            using a Gauss-Newton optimisation. After the parameters have
            been updated also the rigid matrix:
                self._x[c][n].po.rigid
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
        # Parameters
        device = self.sett.device
        rigid_basis = self._rigid_basis
        C = len(self._y)
        num_q = rigid_basis.shape[2]  # Number of registration parameters

        # Update rigid parameters, for all input images (self._x[c][n])
        sll = torch.tensor(0, device=device, dtype=torch.float64)
        for c in range(C):  # Loop over channels
            # # FOR TESTING
            # from nitorch.spm import matrix
            # mat_re = matrix(torch.tensor([-8, 6, -3, 0, 0, 0], device=device, dtype=torch.float64))
            # mat_y = self._y[c].mat
            # mat_y[:3, 3] = mat_re[:3, 3] + mat_y[:3, 3]
            # mat_y[:3, :3] = mat_re[:3, :3].mm(mat_y[:3, :3])
            # self._y[c].mat = mat_y
            # self._update_rigid_channel(c, rigid_basis, verbose=2, max_niter_gn=16, samp=3)
            sllc = self._update_rigid_channel(c, rigid_basis, max_niter_gn=max_niter_gn,
                                              num_linesearch=num_linesearch, verbose=verbose,
                                              samp=samp)
            sll += sllc

        # Mean correct the rigid-body transforms
        if mean_correct:
            sum_q = torch.zeros(num_q, device=device, dtype=torch.float64)
            num_q = 0.0
            # Sum q parameters
            for c in range(C):  # Loop over channels
                num_x = len(self._x[c])  # Number of observations of channel c
                for n in range(num_x):  # Loop over observations of channel c
                    sum_q += self._x[c][n].rigid_q
                    num_q += 1
            # Compute mean
            mean_q = sum_q / num_q
            # Subtract mean (mean correct)
            for c in range(C):  # Loop over channels
                num_x = len(self._x[c])  # Number of observations of channel c
                for n in range(num_x):  # Loop over observations of channel c
                    # if torch.sum(self._x[c][n].rigid_q) != 0:
                    self._x[c][n].rigid_q -= mean_q

            # Update rigid transformations
            for c in range(C):  # Loop over channels
                num_x = len(self._x[c])  # Number of observations of channel c
                for n in range(num_x):  # Loop over observations of channel c
                    rigid = dexpm(self._x[c][n].rigid_q, rigid_basis)[0]
                    self._x[c][n].po.rigid = rigid
        # Return neg log-lik
        return sll

    def _update_rigid_channel(self, c, rigid_basis, max_niter_gn=1, num_linesearch=4,
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
        device = self._y[c].dat.device
        num_x = len(self._x[c])
        method = self._method
        num_q = rigid_basis.shape[2]
        lkp = [[0, 3, 4], [3, 1, 5], [4, 5, 2]]

        sll = torch.tensor(0, device=device, dtype=torch.float64)
        for n_x in range(num_x):  # Loop over repeats

            # Lowres image data
            dat_x = self._x[c][n_x].dat[None, None, ...]
            # Parameters
            q = self._x[c][n_x].rigid_q
            tau = self._x[c][n_x].tau
            armijo = self._x[c][n_x].armijo
            po = self.proj_info(self._x[c][n_x].po.dim_y, self._x[c][n_x].po.mat_y, self._x[c][n_x].po.dim_x,
                                self._x[c][n_x].po.mat_x, rigid=self._x[c][n_x].po.rigid, prof_ip=self.sett.profile_ip,
                                prof_tp=self.sett.profile_tp, gap=self.sett.gap, device=device,
                                scl=self._x[c][n_x].po.scl, samp=samp)

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
                dat_x = grid_pull(self._x[c][n_x].dat[None, None, ...], grid, bound='zero',
                                  extrapolate=False, interpolation=0)[0, 0, ...]
                if n_x == 0 and po.D_y is not None:
                    # Highres (only for superres)
                    grid = affine(po.dim_y, po.D_y, device=device, dtype=torch.float32)
                    dat_y = grid_pull(self._y[c].dat[None, None, ...], grid, bound='zero',
                                      extrapolate=False, interpolation=0)
                else:
                    dat_y = self._y[c].dat[None, None, ...]
            else:
                dat_x = self._x[c][n_x].dat
                dat_y = self._y[c].dat[None, None, ...]

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
                rigid, d_rigid = dexpm(q, rigid_basis, diff=True)
                d_rigid_q = torch.zeros(4, 4, num_q, device=device, dtype=torch.float64)
                for i in range(num_q):
                    d_rigid_q[:, :, i] = d_rigid[:, :, i].mm(mat).solve(po.mat_y)[0]  # mat_y\d_rigid*mat

                # Compute gradient and Hessian
                gr = torch.zeros(num_q, 1, device=device, dtype=torch.float64)
                Hes = torch.zeros(num_q, num_q, device=device, dtype=torch.float64)

                # Compute matching-term part (log-likelihood)
                ll, gr_m, Hes_m = self._rigid_match(dat_x, dat_y, po, tau, rigid, diff=True,
                                                        verbose=verbose, CtC=CtC)

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
                # armijo = torch.tensor(1.0, device=device, dtype=armijo.dtype)
                if num_linesearch == 0:
                    # ..without a line-search
                    q = old_q - armijo * Update
                    rigid = dexpm(q, rigid_basis)[0]
                    if verbose >= 1:
                        print('c={}, n={}, gn={} | q={}'.format(c, n_x, n_gn, round(q, 7).tolist()))
                else:
                    # ..using a line-search
                    for n_ls in range(num_linesearch):
                        # Take step
                        q = old_q - armijo * Update
                        # Compute matching term
                        rigid = dexpm(q, rigid_basis)[0]
                        ll = self._rigid_match(dat_x, dat_y, po, tau, rigid, verbose=verbose)[0]
                        # Matching improved?
                        if ll < old_ll:
                            # Better fit!
                            armijo = torch.min(1.25 * armijo, torch.tensor(1.0, device=device, dtype=armijo.dtype))
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
                            armijo = torch.max(0.5 * armijo, torch.tensor(1e-6, device=device, dtype=armijo.dtype))
                            if n_ls == num_linesearch - 1 and verbose >= 1:
                                print('c={}, n={}, gn={}, ls={} | :( ll={:0.2f}, ll-oll={:0.2f} | q={} armijo={}'
                                      .format(c, n_x, n_gn, n_ls, ll, ll - old_ll, round(q, 7).tolist(),
                                              round(armijo, 4)))
            # Assign
            self._x[c][n_x].armijo = armijo
            self._x[c][n_x].rigid_q = q
            self._x[c][n_x].po.rigid = rigid
            if not old_ll == ll and samp > 0 and self.sett.tolerance > 0:
                # Recompute neg log-lik
                ll = self._rigid_match(self._x[c][n_x].dat, self._y[c].dat[None, None, ...],
                                       self._x[c][n_x].po, tau, rigid)[0]
            # Accumulate neg log-lik
            sll += ll
        # Return neg log-lik
        return sll

    def _rigid_match(self, dat_x, dat_y, po, tau, rigid, CtC=None, diff=False, verbose=0):
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
        # Parameters
        device = self.sett.device
        method = self._method
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
        grid = affine(dim, mat, device=device, dtype=torch.float32)

        # Warp y and compute spatial derivatives
        dat_yx = grid_pull(dat_y, grid, bound=bound, extrapolate=extrapolate)[0, 0, ...]
        if method == 'super-resolution':
            dat_yx = F.conv3d(dat_yx[None, None, ...], smo_ker, stride=ratio)[0, 0, ...]
            if scl != 0:
                dat_yx = self.apply_scaling(dat_yx, scl, dim_thick)
        if diff:
            gr = grid_grad(dat_y, grid, bound=bound, extrapolate=extrapolate)[0, 0, ...]

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
            Hes = torch.zeros(dim + (6,), device=device, dtype=torch.float32)
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

    def _write_data(self, jtv=None):
        """ Format algorithm output.

        Args:
            jtv (torch.tensor, optional): Joint-total variation image, defaults to None.

        Returns:
            y (torch.tensor): Reconstructed image data, (dim_y, C).
            mat (torch.tensor): Reconstructed affine matrix, (4, 4).
            pth_y ([str, ...]): Paths to reconstructed images.

        """
        # Parse function settings
        write_out = self.sett.write_out
        write_jtv = self.sett.write_jtv
        # Output data
        dim_y = self._y[0].dim + (3,)
        # Output orientation matrix
        mat = self._y[0].mat
        if self.sett.dir_out is None:
            # No output directory given, use directory of input data
            if self._x[0][0].direc is None:
                dir_out = 'UniRes-output'
                if not os.path.isdir(dir_out):
                    os.mkdir(dir_out)
            else:
                dir_out = self._x[0][0].direc
        # Reconstructed images
        C = len(self._y)
        prefix_y = self.sett.prefix
        pth_y = []
        for c in range(C):
            mn = torch.min(self._x[c][0].dat)
            dat = self._y[c].dat
            dat[dat < mn] = 0
            if write_out:
                # Write reconstructed images
                if self._x[c][0].nam is None:
                    nam = str(c) + '.nii'
                else:
                    nam = self._x[c][0].nam
                fname = os.path.join(dir_out, prefix_y + nam)
                pth_y.append(fname)
                self.write_image(dat, fname, mat=mat, header=self._x[c][0].head)
            if c == 0:
                y = dat[:, :, :, None]
            else:
                y = torch.cat((y, dat[:, :, :, None]), dim=3)
        if write_jtv and jtv is not None:
            # Write JTV
            if self._x[c][0].nam is None:
                nam = str(c) + '.nii'
            else:
                nam = self._x[c][0].nam
            fname = os.path.join(dir_out, 'jtv_' + prefix_y + nam)
            self.write_image(jtv, fname, mat=mat)
        return y, mat, pth_y

    # Static methods
    @staticmethod
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

    @staticmethod
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
        Ay = Model.proj_apply('A', method, y, po)
        Atx = Model.proj_apply('At', method, x, po)
        # Check okay
        val = torch.sum(Ay * x, dtype=torch.float64) - torch.sum(Atx * y, dtype=torch.float64)
        # Print okay
        print('<Ay, x> - <Atx, y> = {}'.format(val))

    @staticmethod
    def even_odd(dat, which, dim):
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

    @staticmethod
    def proj_apply(operator, method, dat, po, bound='dct2'):
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
                dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate)
                dat = F.conv3d(dat, smo_ker, stride=ratio)
                if scl != 0:
                    dat = Model.apply_scaling(dat, scl, dim_thick)
            elif operator == 'At':
                if scl != 0:
                    dat = Model.apply_scaling(dat, scl, dim_thick)
                dat = F.conv_transpose3d(dat, smo_ker, stride=ratio)
                dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate)
            elif operator == 'AtA':
                dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate)
                dat = F.conv3d(dat, smo_ker, stride=ratio)
                if scl != 0:
                    dat = Model.apply_scaling(dat, 2 * scl, dim_thick)
                dat = F.conv_transpose3d(dat, smo_ker, stride=ratio)
                dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate)
        elif method == 'denoising':
            extrapolate = False
            if operator == 'A':
                dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate)
            elif operator == 'At':
                dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate)
            elif operator == 'AtA':
                dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate)
                dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate)
        return dat

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
