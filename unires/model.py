import nibabel as nib
from nitorch.spatial import grid_pull, voxsize
from nitorch.spm import (affine, mean_space, noise_estimate,
                         affine_basis, dexpm, estimate_fwhm)
from nitorch.optim import get_gain, plot_convergence
from nitorch.utils import show_slices, round
import os
from timeit import default_timer as timer
import torch

from .project import check_adjoint, proj_info
from .struct import Input, Output, Settings
from .update import (admm_aux, update_admm, update_rigid, 
                     update_scaling, step_size)
from .util import print_info, read_image, write_image

torch.backends.cudnn.benchmark = True


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
    y = _init_y_dat(x, y, sett)

    # Sanity check scaling parameter
    if not isinstance(sett.reg_scl, torch.Tensor):
        sett.reg_scl = torch.tensor(sett.reg_scl, dtype=torch.float32, device=sett.device)
        sett.reg_scl = sett.reg_scl.reshape(1)

    # For unified registration, defines a coarse-to-fine scaling of regularisation
    sett = _get_sched(sett)

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
        z, w = admm_aux(y, sett)

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
            fig_ax_nll = plot_convergence(vals=obj[:n_iter + 1, :], fig_ax=fig_ax_nll, fig_num=99,
                                          legend=['-ln(p(y|x))', '-ln(p(x|y))', '-ln(p(y))'])
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
            x, _ = update_scaling(x, y, sett, max_niter_gn=3, num_linesearch=6, verbose=0)
            _ = print_info('fit-done', sett, t0)  # PRINT
            # Print parameter estimates
            _ = print_info('scl-param', sett, x, t0)

        # ----------
        # UPDATE: rigid_q (not every iteration)
        # ----------
        if sett.unified_rigid and 0 < n_iter and cnt_scl < (sett.reg_scl.numel() - 1) \
            and ((n_iter + 1) % sett.rigid_mod) == 0:

            t0 = print_info('fit-update', sett, 'q', n_iter)  # PRINT
            x, _ = update_rigid(x, y, sett, mean_correct=False, max_niter_gn=3, num_linesearch=1,
                                verbose=0, samp=sett.rigid_samp)
            _ = print_info('fit-done', sett, t0)  # PRINT
            # Print parameter estimates
            _ = print_info('reg-param', sett, x, t0)
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
    y = _crop_fov(y, bb=sett.bb)
    y, mat, pth_y = _write_data(x, y, sett, jtv=tmp)

    return y, mat, pth_y, R


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
    _ = print_info('init', sett)

    # Read and format data
    x = _read_data(data, sett)
    del data
    
    # Estimate model hyper-parameters
    x = _estimate_hyperpar(x, sett)
    
    # Init registration
    x, sett = _init_reg(x, sett)

    # Format output
    y, sett = _format_y(x, sett)

    # Define projection matrices
    x = _proj_info_add(x, y, sett)

    if False:
        # Check adjointness of A and At operators
        check_adjoint(po=x[0][0].po, method=sett.method, dtype=torch.float64)

    return x, y, sett


def _all_mat_dim_vx(x, sett):
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


def _crop_fov(y, bb='full'):
    """ Crop reconstructed images FOV to a specified bounding-box.

    Args:
        y (Output()): Output data.
        bb (string): Bounding-box ('full'|'mni'), defaults to 'full'.

    Returns:
        y (Output()): Output data.

    """
    if bb == 'full':  # No cropping
        return y
    # y image information
    dim0 = torch.tensor(y[0].dim, device=y[0].dat.device)
    mat0 = y[0].mat
    vx0 = voxsize(mat0)
    # Set cropping
    if bb == 'mni':
        dim_mni = torch.tensor([181, 217, 181], device=y[0].dat.device,)  # Size of SPM MNI
        dim_mni = (dim_mni / vx0).round()  # Modulate with voxel size
        off = - (dim0 - dim_mni) / 2
        dim1 = dim0 + 2 * off
        mat_crop = torch.tensor([[1, 0, 0, - (off[0] + 1)],
                                 [0, 1, 0, - (off[1] + 1)],
                                 [0, 0, 1, - (off[2] + 1)],
                                 [0, 0, 0, 1]], device=y[0].dat.device)
        mat1 = mat0.mm(mat_crop)
        dim1 = dim1.cpu().int().tolist()
    else:
        raise ValueError('Undefined bounding-box (bb)')
    # Make output grid
    grid = affine(dim1, mat_crop, device=y[0].dat.device, dtype=y[0].dat.dtype)
    # Do interpolation
    for c in range(len(y)):
        dat = grid_pull(y[c].dat[None, None, ...] ,
                        grid, bound='zero', extrapolate=False, interpolation=0)
        # Assign
        y[c].dat = dat[0, 0, ...]
        y[c].mat = mat1
        y[c].dim = dim1

    return y


def _estimate_hyperpar(x, sett):
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
                mu_fg = torch.tensor(5000.0, device=dat.device, dtype=dat.dtype)
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


def _format_y(x, sett):
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
    all_mat, all_dim, all_vx = _all_mat_dim_vx(x, sett)
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
            dim, mat, _ = mean_space(all_mat, all_dim, vx=vx_y)

    # Set method
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
    ff_ct = 1.0  # Just a single CT
    if N > 1:
        # CT and MRIs: We need to fudge the CT scaling of lambda because the gradient
        # distribution between CT and MRIs are so different
        ff_ct = 20.0

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


def _get_sched(sett):
    """ For unified registration, define a coarse-to-fine scaling of regularisation

    """
    if sett.unified_rigid:
        # Parameters/settings
        max = sett.rigid_sched_max
        scl = sett.reg_scl
        two = torch.tensor(2.0, device=sett.device, dtype=torch.float32)
        # Build scheduler
        sched = two ** torch.arange(0, max, step=2, device=sett.device, dtype=torch.float32).flip(dims=(0,))
        ix = torch.min((sched - sett.reg_scl).abs(), dim=0)[1]
        sched = sched[:ix]
        sched = torch.cat((sched, scl.reshape(1)))
        sett.reg_scl = sched

    return sett


def _init_reg(x, sett):
    """ Initialise rigid registration.

    """
    # Init rigid basis
    sett.rigid_basis = affine_basis('SE', 3, device=sett.device, dtype=torch.float64)
    for c in range(len(x)):  # Loop over channels
        for n in range(len(x[c])):  # Loop over observations of channel c
            x[c][n].rigid_q = torch.zeros(6, device=sett.device, dtype=torch.float64)

    return x, sett


def _init_y_dat(x, y, sett):
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
                dat = grid_pull(dat, grid, bound=sett.bound, extrapolate=False, interpolation=1)
                dat[dat < mn] = mn
                dat[dat > mx] = mx
                dat_y = dat_y + dat[0, 0, ...]
        y[c].dat = dat_y / num_x

    return y


def _proj_info_add(x, y, sett):
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


def _read_data(data, sett):
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


def _write_data(x, y, sett, jtv=None):
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
