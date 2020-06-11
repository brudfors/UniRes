# -*- coding: utf-8 -*-
""" A model for processing NIfTI images.

TODO:
    . STEPS:
        1 Rigid to mean space
        2 Coreg
        3 SuperRes w bounding box (vx is set here)
    . Deal with cross-talk? (http://www.mri-q.com/cross-talk.html)
    . Test A and At using the gradcheck function in torch
    . Make A and At layers instead, and import these
    . Remove dependency on numpy
    . Support read/write nifti when large number of observations.

WHY:
    . Artefacts when using central difference?

REFERENCES:
    Brudfors M, Balbastre Y, Nachev P, Ashburner J.
    A Tool for Super-Resolving Multimodal Clinical MRI.
    2019 arXiv preprint arXiv:1909.01140.

    Brudfors M, Balbastre Y, Nachev P, Ashburner J.
    MRI Super-Resolution Using Multi-channel Total Variation.
    In Annual Conference on Medical Image Understanding and Analysis
    2018 Jul 9 (pp. 217-228). Springer, Cham.
"""


""" Imports
"""
from dataclasses import dataclass
from datetime import datetime
import math
import nibabel as nib
from nitorch.kernels import smooth
from nitorch.spatial import grid_pull, grid_push, voxsize
from nitorch.spm import affine, mean_space, noise_estimate
from nitorch.utils import gradient_3d, divergence_3d
from nitorch.optim import cg, get_gain, plot_convergence
import numpy as np
import os
from timeit import default_timer as timer
import torch
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True


""" Data classes
"""
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


@dataclass
class Settings:
    """ Algorithm settings.

    """
    cgs_iter: int = 4  # Conjugate gradient (CG) iterations for solving for y
    cgs_tol: float = 0  # CG tolerance for solving for y
    cgs_verbose: bool = False  # CG verbosity (0, 1)
    device: str = None  # PyTorch device name
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
    reg_scl: float = 20.0  # Scale regularisation estimate
    show_hyperpar: bool = False  # Use matplotlib to visualise hyper-parameter estimates
    show_jtv: bool = False  # Show the joint total variation (JTV)
    tolerance: float = 1e-4  # Algorithm tolerance, if zero, run to max_iter
    vx: float = None  # Reconstruction voxel size (if None, set automatically)
    write_jtv: bool = False  # Write JTV to nifti


class NiiProc:
    """ NIfTI processing class
    """

    """ Constructor
    """
    def __init__(self, pth_nii, sett=Settings()):
        """ Model initialiser.

            This is the entry point to the algorithm, it takes a bunch of nifti files
            as a list of paths (.nii|.nii.gz) and initialises input, output and projection
            operator objects. Settings are changed by editing the Settings() object and
            providing it to this constructor. If not given, default settings are used
            (see Settings() dataclass).

        Args:
            pth_nii (list of strings): Path(s) to nifti(s).
            sett (Settings(), optional): Algorithm settings. Described in Settings() class.

        """
        # Algorithm settings
        self.sett = sett  # self.sett is of class Settings()
        self._rho = None  # Infamous ADMM step-size
        self._do_proj = None  # Use projection matrices (set in _format_output())
        self._method = None  # Method name (super-resolution|denoising)

        # Read and format input
        self._x = self._format_input(pth_nii)
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

        if False:  # Check adjointness of A and At operators
            self.check_adjoint(po=self._x[0][0].po, method=self._method, dtype=torch.float64)

    """ Class methods
    """
    def fit(self):
        """ Fit model.

            This runs the iterative denoising/super-resolution algorithm and,
            at the end, writes the reconstructed images to disk. If the maximum number
            of iterations are set to zero, the initial guesses of the reconstructed
            images will be written to disk (acquired with b-spline interpolation), no
            denoising/super-resolution will be applied.

        Returns:
            fnames_y ([str, ..]): Filenames of reconstructed images.

        """
        # Parse function settings/parameters
        device = self.sett.device
        dtype = torch.float32
        C = len(self._y)  # Number of channels
        N = sum([len(x) for x in self._x])  # Number of observations
        dim_y = self._y[0].dim  # Output dimensions
        vx_y = voxsize(self._y[0].mat).float()  # Output voxel size
        bound_grad = 'constant'
        # Constants
        tiny = torch.tensor(1e-7, dtype=dtype, device=device)
        one = torch.tensor(1, dtype=dtype, device=device)

        # Initial guess of reconstructed images (y)
        self._init_y()
        # Defines:
        # self._y[c].dat

        if self.sett.max_iter > 0:
            # Get ADMM variables
            z, w = self._alloc_admm_vars()

        # Scale lambda
        for c in range(C):
            self._y[c].lam = self.sett.reg_scl * self._y[c].lam0

        # Get ADMM step-size (depends on lam and tau)
        self._rho = self._step_size()
        # self._rho = torch.tensor(1, dtype=dtype, device=device)

        # Init visualisation
        if self.sett.plot_conv:
            fig_ax_nll = plot_convergence(fig_num=98)
        if self.sett.show_jtv:
            fig_ax_jtv = self.show_im(fig_num=99)

        """ Start iterating:
            Updates y, z, w in alternating fashion, until a convergence threshold is met
            on the model negative log-likelihood.
        """
        nll = torch.zeros(self.sett.max_iter, dtype=dtype, device=device)
        t_iter = timer() if self.sett.print_info else 0
        for iter in range(self.sett.max_iter):

            if iter == 0:
                t00 = self._print_info('fit-start', C, N, device,
                    self.sett.max_iter, self.sett.tolerance)  # PRINT

            # """ Scale lambda
            # """
            # for c in range(C):
            #     if type(self.sett.reg_scl) is list:  # coarse-to-fine scaling
            #         if iter >= len(self.sett.reg_scl):
            #             self._y[c].lam = self.sett.reg_scl[-1] * self._y[c].lam0
            #         else:
            #             self._y[c].lam = self.sett.reg_scl[iter] * self._y[c].lam0
            #     else:
            #         self._y[c].lam = self.sett.reg_scl * self._y[c].lam0
            #
            # """ Get ADMM step-size (depends on lam and tau)
            # """
            # self._rho = self._step_size()
            # # self._rho = torch.tensor(1, dtype=dtype, device=device)

            """ UPDATE: z
            """
            t0 = self._print_info('fit-update', 'z', iter)  # PRINT
            jtv = torch.zeros(dim_y, dtype=dtype, device=device)
            for c in range(C):
                Dy = self._y[c].lam * gradient_3d(self._y[c].dat, vx=vx_y, bound=bound_grad)
                jtv = jtv + torch.sum((w[c, ...]/self._rho + Dy)**2, dim=0)
            jtv = torch.sqrt(jtv)
            jtv = ((jtv - one/self._rho).clamp_min(0))/(jtv + tiny)
            if self.sett.show_jtv:  # Show computed JTV
                _ = self.show_im(im=jtv, fig_ax=fig_ax_jtv, fig_title='JTV')
            for c in range(C):
                Dy = self._y[c].lam * gradient_3d(self._y[c].dat, vx=vx_y, bound=bound_grad)
                for d in range(Dy.shape[0]):
                    z[c, d, ...] = jtv*(w[c, d, ...]/self._rho + Dy[d, ...])
            _ = self._print_info('fit-done', t0)  # PRINT

            """ UPDATE: y
            """
            t0 = self._print_info('fit-update', 'y', iter)  # PRINT
            for c in range(C):  # Loop over channels
                Nc = len(self._x[c])

                # RHS
                rhs = torch.zeros(dim_y, device=device, dtype=dtype)
                for n in range(Nc):  # Loop over observations of channel 'c'
                    # _ = self._print_info('int', n)  # PRINT
                    rhs = rhs + self._x[c][n].tau*self._proj('At', self._x[c][n].dat, c, n)

                # Divergence
                div = w[c, ...] - self._rho*z[c, ...]
                div = divergence_3d(div, vx=vx_y, bound=bound_grad)
                rhs = rhs - self._y[c].lam * div

                # Invert y = lhs\rhs by conjugate gradients
                lhs = lambda y: self._proj('AtA', y, c, vx_y=vx_y, bound__DtD=bound_grad)
                self._y[c].dat = cg(A=lhs,
                                    b=rhs, x=self._y[c].dat,
                                    verbose=self.sett.cgs_verbose,
                                    maxiter=self.sett.cgs_iter,
                                    tolerance=self.sett.cgs_tol)

                _ = self._print_info('int', c)  # PRINT

            _ = self._print_info('fit-done', t0)  # PRINT

            """ Objective function and convergence related
            """
            if self.sett.tolerance > 0:
                nll[iter] = self._compute_nll(vx_y=vx_y, bound=bound_grad)
            if self.sett.plot_conv:  # Plot algorithm convergence
                _ = plot_convergence(vals=nll[:iter + 1], fig_ax=fig_ax_nll)
            # Check convergence
            gain = get_gain(nll, iter, monotonicity='decreasing')
            t_iter = self._print_info('fit-ll', iter, nll[iter], gain, t_iter)  # PRINT
            if (gain < self.sett.tolerance) or (iter >= (self.sett.max_iter - 1)):
                _ = self._print_info('fit-finish', t00, iter)  # PRINT
                break  # Finished

            """ UPDATE: w
            """
            t0 = self._print_info('fit-update', 'w', iter)  # PRINT
            for c in range(C):  # Loop over channels
                Dy = self._y[c].lam * gradient_3d(self._y[c].dat, vx=vx_y, bound=bound_grad)
                w[c, ...] = w[c, ...] + self._rho*(Dy - z[c, ...])
                _ = self._print_info('int', c)  # PRINT
            _ = self._print_info('fit-done', t0)  # PRINT

        """ Write results to disk
        """
        if self.sett.dir_out is None:
            # No output directory given, use directory of input data
            dir_out = self._x[0][0].direc
        prefix_y = self.sett.prefix
        mat = self._y[0].mat
        fnames_y = []
        for c in range(C):
            # Reconstructed images
            mn = torch.min(self._x[c][0].dat)
            dat = self._y[c].dat
            dat[dat < mn] = 0
            fname = os.path.join(dir_out, prefix_y + self._x[c][0].nam)
            self.write_nifti_3d(dat, fname, mat=mat, header=self._x[c][0].head)
            fnames_y.append(fname)

        if self.sett.write_jtv and (self.sett.max_iter > 0):
            # JTV
            fname = os.path.join(dir_out, 'jtv_' + self._x[c][0].nam)
            self.write_nifti_3d(jtv, fname, mat=mat)

        return fnames_y

    def _all_mat_dim_vx(self):
        """ Get all images affine matrices, dimensions and voxel sizes (as numpy arrays).

        Returns:
            all_mat (np.array()): Image orientation matrices (4, 4, N).
            Dim (np.array()): Image dimensions (3, N).
            all_vx (np.array()): Image voxel sizes (3, N).

        """
        # Parse function settings
        device = self.sett.device

        N = sum([len(x) for x in self._x])
        all_mat = np.zeros((4, 4, N))
        all_dim = np.zeros((3, N))
        all_vx = np.zeros((3, N))

        cnt = 0
        for c in range(len(self._x)):
            for n in range(len(self._x[c])):
                all_mat[..., cnt] = self._x[c][n].mat.cpu()
                all_dim[..., cnt] = self._x[c][n].dim
                all_vx[..., cnt] = voxsize(self._x[c][n].mat).cpu()
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

    def _compute_nll(self, vx_y, sum_dtype=torch.float64, bound='constant'):
        """ Compute negative model log-likelihood.

        Args:
            vx_y (tuple(float)): Output voxel size.
            sum_dtype (torch.dtype): Defaults to torch.float64.
            bound (str, optional): Bound for gradient/divergence calculation, defaults to
                constant zero.

        Returns:
            nll (torch.tensor()): Negative log-likelihood.

        """
        device = self.sett.device
        dtype = torch.float32

        C = len(self._y)
        nll_xy = torch.tensor(0, dtype=dtype, device=device)
        nll_y = torch.zeros(self._y[0].dim, dtype=dtype, device=device)
        for c in range(C):
            Nc = len(self._x[c])
            for n in range(Nc):
                nll_xy = nll_xy + \
                    self._x[c][n].tau/2*torch.sum((self._proj('A', self._y[c].dat, c, n)
                                                  - self._x[c][n].dat)**2, dtype=sum_dtype)

            Dy = self._y[c].lam * gradient_3d(self._y[c].dat, vx=vx_y, bound=bound)
            nll_y = nll_y + torch.sum(Dy**2, dim=0, dtype=dtype)

        nll_y = torch.sum(torch.sqrt(nll_y), dtype=sum_dtype)

        return nll_xy + nll_y

    def _DtD(self, dat, vx_y, bound='constant'):
        """ Computes the divergence of the gradient.

        Args:
            dat (torch.tensor()): A tensor (dim_y).
            vx_y (tuple(float)): Output voxel size.
            bound (str, optional): Bound for gradient/divergence calculation, defaults to
                constant zero.

        Returns:
              div (torch.tensor()): Dt(D(dat)) (dim_y).

        """
        dat = gradient_3d(dat, vx=vx_y, bound=bound)
        dat = divergence_3d(dat, vx=vx_y, bound=bound)
        return dat

    def _format_input(self, pth_nii):
        """ Construct algorithm input struct.

        Args:
            pth_nii (list): List of path(s) to nifti file.

        Returns:
            x (Input()): Algorithm input struct(s).

        """
        # Parse input niftis into Input() object, filling the following fields:
        # x[c][n].dat
        # x[c][n].dim
        # x[c][n].mat
        # x[c][n].ct
        # x[c][n].fname
        # x[c][n].direc
        # x[c][n].nam
        # x[c][n].head
        x = self._load_data(pth_nii)

        # Estimate input image statistics, filling the following fields of Input():
        # x[c][n].tau
        # x[c][n].mu
        # x[c][n].sd
        x = self._image_statistics(x)

        return x

    def _format_output(self):
        """ Construct algorithm output struct. See Output() dataclass.

        Returns:
            y (Output()): Algorithm output struct(s).

        """
        # Parse function settings
        C = len(self._x)  # Number of channels
        device = self.sett.device
        dtype = torch.float32
        mod_prct = self.sett.mod_prct
        vx_y = self.sett.vx  # output voxel size
        if vx_y is not None:
            # Format voxel size as np.array()
            if type(vx_y) is int:
                vx_y = float(vx_y)
            if type(vx_y) is float:
                vx_y = (vx_y,) * 3
            vx_y = np.asarray(vx_y)

        # Get all orientation matrices and dimensions
        all_mat, all_dim, all_vx = self._all_mat_dim_vx()
        N = all_mat.shape[-1]  # Total number of observations

        # Check if all input images have the same fov/vx
        mat_same = True
        dim_same = True
        vx_same = True
        for n in range(1, all_mat.shape[2]):
            mat_same = mat_same & \
                np.all(np.equal(np.round(all_mat[..., n - 1], 3), np.round(all_mat[..., n], 3)))
            dim_same = dim_same & \
                np.all(np.equal(np.round(all_dim[..., n - 1], 3), np.round(all_dim[..., n], 3)))
            vx_same = vx_same & \
                np.all(np.equal(np.round(all_vx[..., n - 1], 3), np.round(all_vx[..., n], 3)))

        """
        Decide if super-resolving and/or projection is necessary
        """
        do_sr = True
        self._do_proj = True
        if vx_y is None and ((N == 1) or vx_same):  vx_y = all_vx[..., 0]  # One image, voxel size not given

        if vx_same and (np.abs(all_vx[..., 0] - vx_y) < 1e-3).all():
            # All input images have same voxel size, and output voxel size is the also the same
            do_sr = False
            if mat_same and dim_same:
                # All input images have the same FOV
                mat = all_mat[..., 0]
                dim = all_dim[..., 0]
                self._do_proj = False

        if do_sr or self._do_proj:
            # Get FOV of mean space
            if N == 1 and do_sr:
                D = np.diag([vx_y[0]/all_vx[0, 0], vx_y[1]/all_vx[1, 0], vx_y[2]/all_vx[2, 0], 1])
                mat = np.matmul(all_mat[..., 0], D)
                dim = np.squeeze(np.floor(np.matmul(np.linalg.inv(D)[:3, :3], np.reshape(all_dim[:, 0], (3, 1)))))
            else:
                # Mean space from several images
                dim, mat, _ = mean_space(all_mat, all_dim, vx=vx_y, mod_prct=-mod_prct)
        if do_sr:
            self._method = 'super-resolution'
        else:
            self._method = 'denoising'

        mat = torch.from_numpy(mat)
        dim = tuple(dim.astype(np.int).tolist())
        _ = self._print_info('mean-space', dim, mat)

        """ Assign output
        """
        y = []
        for c in range(C):
            y.append(Output())
            # Regularisation (lambda) for channel c
            Nc = len(self._x[c])
            mu_c = torch.zeros(Nc, dtype=dtype, device=device)
            for n in range(Nc):
                mu_c[n] = self._x[c][n].mu
            y[c].lam0 = 1/torch.mean(mu_c)
            y[c].lam = 1/torch.mean(mu_c)  # To facilitate rescaling
            # Output image(s) dimension and orientation matrix
            y[c].dim = dim
            y[c].mat = mat.double().to(device)

        return y

    def _image_statistics(self, x):
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
            Nc = len(x[c])
            for n in range(Nc):

                # Set options for spm.noise_estimate
                dat = x[c][n].dat
                mu_noise = None
                num_class = 2
                max_iter = 10000
                if x[c][n].ct:
                    # Get CT foreground
                    mu_noise = -1000
                    num_class = 10
                    _, sd_fg, _, mu_fg = noise_estimate(dat,
                                                        num_class=num_class, show_fit=show_hyperpar,
                                                        fig_num=100 + cnt,
                                                        mu_noise=mu_noise, max_iter=max_iter)
                    # Get CT noise
                    dat = dat[(dat >= -1020) & (dat < -900)]
                    num_class = 2
                    sd_bg, _, mu_bg, _ = noise_estimate(dat,
                                                        num_class=num_class, show_fit=show_hyperpar,
                                                        fig_num=100 + cnt + C,
                                                        mu_noise=mu_noise, max_iter=max_iter)
                else:
                    # Get noise and foreground statistics
                    sd_bg, sd_fg, mu_bg, mu_fg = noise_estimate(dat,
                        num_class=num_class, show_fit=show_hyperpar, fig_num=100 + cnt,
                        mu_noise=mu_noise, max_iter=max_iter)
                x[c][n].sd = sd_bg.float()
                x[c][n].tau = 1/sd_bg.float()**2
                x[c][n].mu = mu_fg.float() - mu_bg.float()
                cnt += 1

        # Print info to screen
        self._print_info('hyper_par', x, t0)

        return x

    def _init_y(self, interpolation=4):
        """ Make initial guesses of reconstucted image(s) using b-spline interpolation,
            with averaging if more than one observation per channel.

        Args:
            interpolation (int, optional): Interpolation order, defaults to 4.

        """
        C = len(self._x)
        dim_y = self._x[0][0].po.dim_y
        mat_y = self._x[0][0].po.mat_y
        for c in range(C):
            y = torch.zeros(dim_y, dtype=torch.float32, device=self.sett.device)
            Nc = len(self._x[c])
            for n in range(Nc):
                # Make output grid
                mat = mat_y.solve(self._x[c][n].po.mat_x)[0]  # mat_x\mat_y
                grid = affine(self._x[c][n].po.dim_y, mat, device=self.sett.device)
                # Get image data
                dat = self._x[c][n].dat[None, None, ...]
                # Do interpolation
                dat = grid_pull(dat, grid, bound='zero', extrapolate=False, interpolation=interpolation)
                y = y + dat[0, 0, ...]
            self._y[c].dat = y / Nc

    def _load_data(self, pth_nii):
        """ Parse nifti files into algorithm input struct(s).

        Args:
            pth_nii (list): List of path(s) to nifti file.

        Returns:
            x (Input()): Algorithm input struct(s).

        """
        # Parse function settings
        device = self.sett.device
        has_ct = self.sett.has_ct

        if type(pth_nii) is str:
            pth_nii = [pth_nii]

        C = len(pth_nii)  # Number of channels
        x = []
        for c in range(C):  # Loop over channels
            x.append([])
            x[c] = []

            if type(pth_nii[c]) is list:
                Nc = len(pth_nii[c])  # Number of observations of channel c
                for n in range(Nc):  # Loop over observations of channel c
                    x[c].append(Input())
                    # Get data
                    dat, dim, mat, fname, direc, nam, head, ct, _ = \
                        self.read_nifti_3d(pth_nii[c][n], device, is_ct=has_ct)
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
                n = 0
                # Get data
                dat, dim, mat, fname, direc, nam, head, ct, _ = \
                    self.read_nifti_3d(pth_nii[c], device, is_ct=has_ct)
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
                print('{:3} - Convergence ({:0.1f} s)  | nll={:0.4f}, '
                      'gain={:0.7f}'.format(argv[0] + 1, timer() - argv[3], argv[1], argv[2]))
            elif info == 'fit-start':
                print('\nStarting {} \n{} | C={} | N={} | device={} | '
                      'maxiter={} | tol={}'.format(self._method, datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                   argv[0], argv[1], argv[2], argv[3], argv[4]))
            elif info in 'step_size':
                print('\nADMM step-size={:0.4f}'.format(argv[0]))
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

    def _proj(self, operator, dat, c=0, n=0, vx_y=None, bound__DtD='constant'):
        """ Projects image data by A, At or AtA.

        Args:
            operator (string): Either 'A', 'At ' or 'AtA'.
            dat (torch.tensor()): Image data (dim_x|dim_y).
            c (int): Channel index, defaults to 0.
            n (int): Observation index, defaults to 0.
            vx_y (tuple(float)): Output voxel size.
            bound__DtD (str, optional): Bound for gradient/divergence calculation, defaults to
                constant zero.

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
            dat1 = rho * self._y[c].lam ** 2 * self._DtD(dat, vx_y=vx_y, bound=bound__DtD)
            dat = dat[None, None, ...]
            dat = self._x[c][n].tau * self.proj_apply(operator, method, dat, self._x[c][n].po)
            Nc = len(self._x[c])
            for n1 in range(1, Nc):
                dat = dat + self._x[c][n1].tau * self.proj_apply(operator, method, dat, self._x[c][n1].po)
            dat = dat[0, 0, ...]
            dat = dat + dat1
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
        C = len(self._x)
        # Build each projection operator
        for c in range(C):
            dim_y = self._y[c].dim
            mat_y = self._y[c].mat
            Nc = len(self._x[c])
            for n in range(Nc):
                # Define projection operator
                po = self.proj_info(dim_y, mat_y, self._x[c][n].dim, self._x[c][n].mat,
                                    prof_ip=prof_ip, prof_tp=prof_tp, gap=gap, device=device)
                # Assign
                self._x[c][n].po = po

    def _step_size(self):
        """ ADMM step size (rho) from image statistics.

        Returns:
            rho (torch.tensor()): Step size.

        """
        # Parse function settings
        device = self.sett.device
        dtype = torch.float32

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

        rho = torch.sqrt(torch.mean(all_tau))/torch.mean(all_lam)
        _ = self._print_info('step_size', rho)
        return rho

    """ Static methods
    """
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
        x = torch.rand((1, 1, ) + dim_x, dtype=dtype, device=device)
        y = torch.rand((1, 1, ) + dim_y, dtype=dtype, device=device)
        po.smo_ker = po.smo_ker.type(dtype)
        # Apply A and At operators
        Ay = NiiProc.proj_apply('A', method, y, po)
        Atx = NiiProc.proj_apply('At', method, x, po)
        # Check okay
        val = torch.sum(Ay * x, dtype=torch.float64) - torch.sum(Atx * y, dtype=torch.float64)
        # Print okay
        print('<Ay, x> - <Atx, y> = {}'.format(val))

    @staticmethod
    def read_nifti_3d(pth_nii, device='cpu', as_float=True, is_ct=False):
        """ Reads 3D nifti data using nibabel.

        Args:
            pth_nii (string): Path to nifti file.
            device (string, optional): PyTorch on CPU or GPU? Defaults to 'cpu'.
            as_float (bool, optional): Load image data as float (else double), defaults to True.
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
        # Load nifti
        nii = nib.load(pth_nii)
        # Get image data
        if as_float:
            dat = torch.tensor(nii.get_fdata()).float().to(device)
        else:
            dat = torch.tensor(nii.get_fdata()).double().to(device)
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
        # Get affine matrix
        mat = nii.affine
        mat = torch.from_numpy(mat).double().to(device)
        # Get dimensions
        dim = tuple(dat.shape)
        # Get observation uncertainty
        slope = nii.dataobj.slope
        dtype = nii.get_data_dtype()
        dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        if dtype in dtypes:
            var = torch.tensor(slope, dtype=torch.float32, device=device)
            var = var ** 2 / 12
        else:
            var = torch.tensor(0, dtype=torch.float32, device=device)
        # Get header, filename, etc
        head = nii.get_header()
        fname = nii.get_filename()
        direc, nam = os.path.split(fname)

        return dat, dim, mat, fname, direc, nam, head, ct, var

    @staticmethod
    def proj_apply(operator, method, dat, po, bound='zero'):
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
        dim_x = po.dim_x
        dim_y = po.dim_y
        dim_yx = po.dim_yx
        ratio = po.ratio
        smo_ker = po.smo_ker
        # Apply projection
        if method == 'super-resolution':
            """ Super-resolution
            """
            mat = mat_yx.solve(mat_y)[0]  # mat_y\mat_yx
            grid = affine(dim_yx, mat, device=device, dtype=dtype)
            if operator == 'A':
                dat = grid_pull(dat, grid, bound=bound)
                dat = F.conv3d(dat, smo_ker, stride=ratio)
            elif operator == 'At':
                dat = F.conv_transpose3d(dat, smo_ker, stride=ratio)
                dat = grid_push(dat, grid, shape=dim_y, bound=bound)
            elif operator == 'AtA':
                dat = grid_pull(dat, grid, bound=bound)
                dat = F.conv3d(dat, smo_ker, stride=ratio)
                dat = F.conv_transpose3d(dat, smo_ker, stride=ratio)
                dat = grid_push(dat, grid, shape=dim_y, bound=bound)
        elif method == 'denoising':
            """ Denoising
            """
            mat = mat_x.solve(mat_y)[0]  # mat_y\mat_x
            grid = affine(dim_x, mat, device=device)
            if operator == 'A':
                dat = grid_pull(dat, grid, bound=bound, extrapolate=False)
            elif operator == 'At':
                dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=False)
            elif operator == 'AtA':
                dat = grid_pull(dat, grid, bound=bound, extrapolate=False)
                dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=False)
        return dat

    @staticmethod
    def proj_info(dim_y, mat_y, dim_x, mat_x, prof_ip=0, prof_tp=0, gap=0.0, device='cpu'):
        """ Define projection operator object, to be used with proj_apply.

        Args:
            dim_y ((int, int, int))): High-res image dimensions (3,).
            mat_y (torch.tensor): High-res affine matrix (4, 4).
            dim_x ((int, int, int))): Low-res image dimensions (3,).
            mat_x (torch.tensor): Low-res affine matrix (4, 4).
            prof_ip (int, optional): In-plane slice profile (0=rect|1=tri|2=gauss), defaults to 0.
            prof_tp (int, optional): Through-plane slice profile (0=rect|1=tri|2=gauss), defaults to 0.
            gap (float, optional): Slice-gap between 0 and 1, defaults to 0.
            device (torch.device, optional): Device. Defaults to 'cpu'.

        Returns:
            po (ProjOp()): Projection operator object.

        """
        # Get projection operator object
        po = ProjOp()
        # Data types
        dtype = torch.float64
        dtype_smo_ker = torch.float32
        # Output properties
        po.dim_y = dim_y
        po.mat_y = mat_y
        po.vx_y = voxsize(mat_y)
        # Input properties
        po.dim_x = torch.tensor(dim_x, device=device, dtype=dtype)
        po.mat_x = mat_x
        po.vx_x = voxsize(mat_x)
        # Slice-profile
        gap_cn = torch.zeros(3, device=device, dtype=dtype)
        profile_cn = torch.tensor((prof_ip,) * 3, device=device, dtype=dtype)
        ix_thick = torch.max(po.vx_x, dim=0)[1]
        gap_cn[ix_thick] = gap
        profile_cn[ix_thick] = prof_tp
        # Intermediate
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
        # To tuple of ints
        po.dim_yx = tuple(po.dim_yx.int().tolist())
        po.dim_x = tuple(po.dim_x.int().tolist())
        po.ratio = tuple(ratio.int().tolist())
        return po

    @staticmethod
    def write_nifti_3d(dat, ofname, mat=torch.eye(4), header=None, dtype='float32'):
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
        # To CPU -> Numpy (because nibable cannot deal with torch.tensor)
        dat = dat.cpu().numpy()
        # Make nii object
        nii = nib.Nifti1Image(dat, header=header, affine=mat.cpu().numpy())
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
                slope = (mx/255).cpu().numpy()
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

    @staticmethod
    def show_im(im=None, fig_ax=None, fig_num=1, fig_title=''):
        """ Show an image.

        This allows for real-time plotting of an image. It is initialised by
        called with no argument:

        fig_ax = _show_im()

        Subsequent calls are then performed as:

        _ = _show_im(im=im, fig_ax=fig_ax)

        Args:
            im (torch.tensor, optional): Image as tensor (W, H, D).
            fig_ax ([matplotlib.figure, matplotlib.axes])
            fig_num (int, optional): Figure number to plot to, defaults to 1.
            fig_title (str, optional): Figure title, defaults to ''.

        Returns:
            fig_ax ([matplotlib.figure, matplotlib.axes])

        """
        import matplotlib.pyplot as plt

        if fig_ax is None:
            fig, ax = plt.subplots(1, 3, num=fig_num)
            fig_ax = [fig, ax]
            plt.ion()
            fig.show()
        elif im is not None:
            im = im.squeeze()
            dm = torch.tensor(im.shape)
            ix = torch.round(0.5 * dm).int().tolist()

            cmap = 'coolwarm'
            ax = fig_ax[1][0]
            ax.clear()
            im1 = im[:, :, ix[2]].cpu()
            ax.imshow(im1, interpolation='None', cmap=cmap, aspect='auto')
            ax.axis('off')
            ax = fig_ax[1][1]
            ax.clear()
            im1 = im[:, ix[1], :].squeeze().cpu()
            ax.imshow(im1, interpolation='None', cmap=cmap, aspect='auto')
            ax.axis('off')
            ax = fig_ax[1][2]
            ax.clear()
            im1 = im[ix[0], :, :].squeeze().cpu()
            ax.imshow(im1, interpolation='None', cmap=cmap, aspect='auto')
            ax.axis('off')

            fig_ax[0].suptitle(fig_title)
            fig_ax[0].canvas.draw()
            fig_ax[0].canvas.flush_events()

        return fig_ax
