# Python
from timeit import default_timer as timer
# 3rd party
import torch
# NITorch
from nitorch.spatial import (affine_grid, grid_pull)
from nitorch.core.optim import (get_gain, plot_convergence)
from nitorch.plot.volumes import show_slices
from nitorch.core._linalg_expm import _expm
# UniRes
from ._project import _check_adjoint
from .struct import settings
from ._update import (_admm_aux, _update_admm, _update_rigid,
                     _update_scaling, _step_size)
from ._util import _print_info
from._core import (_crop_y, _estimate_hyperpar, _fix_affine,
                   _format_y, _get_sched, _init_reg, _init_y_dat,
                   _init_y_label, _proj_info_add, _read_data, _write_data)

torch.backends.cudnn.benchmark = True


def fit(x, y, sett):
    """ Fit model.

        This runs the iterative denoising/super-resolution algorithm and,
        at the end, writes the reconstructed images to disk. If the maximum number
        of iterations are set to zero, the initial guesses of the reconstructed
        images will be written to disk (acquired with b-spline interpolation), no
        denoising/super-resolution will be applied.

    Returns:
        dat_y (torch.tensor): Reconstructed image data as float32, (dim_y, C).
        mat_y (torch.tensor): Reconstructed affine matrix, (4, 4).
        pth_y ([str, ...]): Paths to reconstructed images.
        R (torch.tensor): Rigid matrices (N, 4, 4).
        label (torch.tensor): Reconstructed label image, (dim_y).
        pth_label str: Path to reconstructed label image.

    """
    with torch.no_grad():
        # Total number of observations
        N = sum([len(xn) for xn in x])

        # Sanity check scaling parameter
        if not isinstance(sett.reg_scl, torch.Tensor):
            sett.reg_scl = torch.tensor(sett.reg_scl, dtype=torch.float32, device=sett.device)
            sett.reg_scl = sett.reg_scl.reshape(1)

        # Defines a coarse-to-fine scaling of regularisation
        sett = _get_sched(N, sett)

        # For visualisation
        fig_ax_nll = None
        fig_ax_jtv = None

        # Scale lambda
        cnt_scl = 0
        for c in range(len(x)):
            y[c].lam = sett.reg_scl[cnt_scl] * y[c].lam0

        # Get ADMM step-size
        rho = _step_size(x, y, sett, verbose=True)

        if sett.max_iter > 0:
            # Get ADMM variables (only if algorithm is run)
            z, w = _admm_aux(y, sett)

        # ----------
        # ITERATE:
        # Updates model in an alternating fashion, until a convergence threshold is met
        # on the model negative log-likelihood.
        # ----------
        obj = torch.zeros(sett.max_iter, 3, dtype=torch.float64, device=sett.device)
        tmp = torch.zeros_like(y[0].dat)  # for holding rhs in y-update, and jtv in u-update
        t_iter = timer() if sett.do_print else 0
        cnt_scl_iter = 0  # To ensure we do, at least, a fixed number of iterations for each scale
        for n_iter in range(sett.max_iter):

            if n_iter == 0:
                t00 = _print_info('fit-start', sett, len(x), N)  # PRINT

            # ----------
            # UPDATE: image
            # ----------
            y, z, w, tmp, obj = _update_admm(x, y, z, w, rho, tmp, obj, n_iter, sett)

            # Show JTV
            if sett.show_jtv:
                fig_ax_jtv = show_slices(img=tmp, fig_ax=fig_ax_jtv, title='JTV',
                                         cmap='coolwarm', fig_num=98)

            # ----------
            # Check convergence
            # ----------
            if sett.plot_conv:  # Plot algorithm convergence
                fig_ax_nll = plot_convergence(vals=obj[:n_iter + 1, :], fig_ax=fig_ax_nll, fig_num=99,
                                              legend=['-ln(p(y|x))', '-ln(p(x|y))', '-ln(p(y))'])
            gain = get_gain(obj[:n_iter + 1, 0], monotonicity='decreasing')
            t_iter = _print_info('fit-ll', sett, 'y', n_iter, obj[n_iter, :], gain, t_iter)
            # Converged?
            if cnt_scl >= (sett.reg_scl.numel() - 1) and cnt_scl_iter > 20 \
                and ((gain.abs() < sett.tolerance) or (n_iter >= (sett.max_iter - 1))):
                countdown0 -= 1
                if countdown0 == 0:
                    _ = _print_info('fit-finish', sett, t00, n_iter)
                    break  # Finished
            else:
                countdown0  = 6

            # ----------
            # UPDATE: even/odd scaling
            # ----------
            if sett.scaling:

                t0 = _print_info('fit-update', sett, 's', n_iter)  # PRINT
                # Do update
                x, _ = _update_scaling(x, y, sett, max_niter_gn=1, num_linesearch=6, verbose=0)
                _ = _print_info('fit-done', sett, t0)  # PRINT
                # Print parameter estimates
                _ = _print_info('scl-param', sett, x, t0)

            # ----------
            # UPDATE: rigid_q
            # ----------
            if sett.unified_rigid and n_iter > 0 \
                and (n_iter % sett.rigid_mod) == 0:

                t0 = _print_info('fit-update', sett, 'q', n_iter)  # PRINT
                x, _ = _update_rigid(x, y, sett,
                    mean_correct=True, max_niter_gn=1, num_linesearch=6, verbose=0, samp=sett.rigid_samp)
                _ = _print_info('fit-done', sett, t0)  # PRINT
                # Print parameter estimates
                _ = _print_info('reg-param', sett, x, t0)

            # ----------
            # Coarse-to-fine scaling of regularisation
            # ----------
            if cnt_scl + 1 < len(sett.reg_scl) and cnt_scl_iter > 16 and\
                    gain.abs() < 1e-3:
                countdown1 -= 1
                if countdown1 == 0:
                    cnt_scl_iter = 0
                    cnt_scl += 1
                    # Coarse-to-fine scaling of lambda
                    for c in range(len(x)):
                        y[c].lam = sett.reg_scl[cnt_scl] * y[c].lam0
                    # Also update ADMM step-size
                    rho = _step_size(x, y, sett)
            else:
                countdown1 = 6

            cnt_scl_iter += 1

        # ----------
        # Some post-processing
        # ----------
        if sett.clean_fov:
            # Zero outside FOV in reconstructed data
            for c in range(len(x)):
                msk_fov = torch.ones(y[c].dim, dtype=torch.bool, device=sett.device)
                for n in range(len(x[c])):
                    # Map to voxels in low-res image
                    M = x[c][n].po.rigid.mm(x[c][n].mat).solve(y[c].mat)[0].inverse()
                    grid = affine_grid(M.type(x[c][n].dat.dtype), y[c].dim)[None, ...]
                    # Mask of low-res image FOV projected into high-res space
                    msk_fov = msk_fov & \
                              (grid[0, ..., 0] >= 1) & (grid[0, ..., 0] <= x[c][n].dim[0]) & \
                              (grid[0, ..., 1] >= 1) & (grid[0, ..., 1] <= x[c][n].dim[1]) & \
                              (grid[0, ..., 2] >= 1) & (grid[0, ..., 2] <= x[c][n].dim[2])
                    # if x[c][n].ct:
                    #     # Resample low-res image into high-res space
                    #     dat_c = grid_pull(x[c][n].dat[None, None, ...],
                    #                       grid, bound=sett.bound,
                    #                       extrapolate=False,
                    #                       interpolation=sett.interpolation)[0, 0, ...]
                    #     # Set voxels inside the FOV that are positive in the
                    #     # low-res data but negative in the high-res, to the
                    #     # their original values
                    #     msk = msk_fov & (dat_c >= 0) & (y[c].dat < 0)
                    #     y[c].dat[msk] = tmp[msk]
                # Zero voxels outside projected FOV
                y[c].dat[~msk_fov] = 0.0

        # Possibly crop reconstructed data
        y = _crop_y(y, sett)

        # ----------
        # Get rigid matrices
        # ----------
        R = torch.zeros((N, 4, 4), device=sett.device, dtype=torch.float64)
        cnt = 0
        for c in range(len(x)):
            for n in range(len(x[c])):
                R[cnt, ...] = _expm(x[c][n].rigid_q, sett.rigid_basis)
                cnt += 1

        # ----------
        # Possibly write reconstruction results to disk
        # ----------
        dat_y, pth_y, label, pth_label = _write_data(x, y, sett, jtv=tmp)

        return dat_y, y[0].mat, pth_y, R, label, pth_label


def init(data, sett=settings()):
    """ Model initialiser.

        This is the entry point to the algorithm, it takes a bunch of nifti files
        as a list of paths (.nii|.nii.gz) and initialises input, output and projection
        operator objects. Settings are changed by editing the settings() object and
        providing it to this constructor. If not given, default settings are used
        (see settings() dataclass).

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

        sett (settings(), optional): Algorithm settings. Described in settings() class.

    """
    with torch.no_grad():
        _ = _print_info('init', sett)

        # Read and format data
        x = _read_data(data, sett)
        del data

        # Estimate model hyper-parameters
        x = _estimate_hyperpar(x, sett)

        # Possibly, fix messed up affine in CT scans
        x = _fix_affine(x, sett)

        # Init registration, possibly:
        # * Co-registers all input images
        # * Aligns to atlas space
        # * Crops to atlas space
        x, sett = _init_reg(x, sett)

        # Format output
        y, sett = _format_y(x, sett)

        # Define projection matrices
        x = _proj_info_add(x, y, sett)

        # Initial guess of reconstructed images (y)
        y = _init_y_dat(x, y, sett)

        # Initial guess of labels (if given)
        y = _init_y_label(x, y, sett)

        # # Check adjointness of A and At operators
        # _check_adjoint(po=x[0][0].po, method=sett.method, dtype=torch.float64)

        return x, y, sett
