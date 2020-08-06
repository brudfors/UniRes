from nitorch.spatial import grid_pull, voxsize, im_gradient, im_divergence, grid_grad
from nitorch.spm import affine, dexpm, identity
from nitorch.optim import cg
from nitorch.utils import show_slices, round
import torch
from torch.nn import functional as F

from .project import apply_scaling, proj, proj_info
from .util import print_info


def admm_aux(y, sett):
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
            tmp += x[c][n].tau * proj('At', x[c][n].dat, x[c], y[c], method=sett.method, do=sett.do_proj, n=n)

        # Divergence
        div = w[c, ...] - rho * z[c, ...]
        div = im_divergence(div, vx=vx_y, bound=bound_grad, which=sett.gr_diff)
        tmp -= y[c].lam * div

        # Invert y = lhs\tmp by conjugate gradients
        lhs = lambda dat: proj('AtA', dat, x[c], y[c], method=sett.method, do=sett.do_proj, rho=rho, vx_y=vx_y, bound_DtD=bound_grad, gr_diff=sett.gr_diff)
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
            = _compute_nll(x, y, sett, rho, bound=bound_grad, gr_diff=sett.gr_diff)  # nl_pyx, nl_pxy, nl_py

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
        # _update_rigid_channel(c, sett.rigid_basis, verbose=2, max_niter_gn=16, samp=3)
        x[c], sllc = _update_rigid_channel(x[c], y[c], sett, max_niter_gn=max_niter_gn,
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
            xo = _even_odd(dat_x, 'odd', dim_thick)
            mo = _even_odd(msk, 'odd', dim_thick)
            xo = xo[mo]
            xe = _even_odd(dat_x, 'even', dim_thick)
            me = _even_odd(msk, 'even', dim_thick)
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
                yo = _even_odd(dat_y, 'odd', dim_thick)
                yo = yo[mo]
                ye = _even_odd(dat_y, 'even', dim_thick)
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


def _compute_nll(x, y, sett, rho, sum_dtype=torch.float64, bound='constant', gr_diff='forward'):
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
                                                    proj('A', y[c].dat, x[c], y[c], method=sett.method, do=sett.do_proj, n=n)[msk]) ** 2,
                                                    dtype=sum_dtype)
        # Neg. log-prior term
        Dy = y[c].lam * im_gradient(y[c].dat, vx=vx_y, bound=bound, which=gr_diff)
        if c > 0:
            nll_y += torch.sum(Dy ** 2, dim=0)
        else:
            nll_y = torch.sum(Dy ** 2, dim=0)

    nll_y = torch.sum(torch.sqrt(nll_y), dtype=sum_dtype)

    return nll_xy + nll_y, nll_xy, nll_y


def _even_odd(dat, which, dim):
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


def _rigid_match(method, dat_x, dat_y, po, tau, rigid, CtC=None, diff=False, verbose=0):
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


def _update_rigid_channel(xc, yc, sett, max_niter_gn=1, num_linesearch=4,
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
            ll, gr_m, Hes_m = _rigid_match(sett.method, dat_x, dat_y, po, tau, rigid,
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
                    ll = _rigid_match(sett.method, dat_x, dat_y, po, tau, rigid,
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
