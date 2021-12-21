from nitorch.core.kernels import smooth
from nitorch.spatial import (affine_grid, grid_pull, grid_push,
                             voxel_size, im_gradient, im_divergence)
import torch
from torch.nn import functional as F
from .struct import _proj_op


def _apply_scaling(dat, scl, dim):
    """ Apply even/odd slice scaling.

    """
    dat_out = torch.zeros_like(dat)
    if dim == 2:
        dat_out[..., :, :, ::2] = torch.exp(scl) * dat[..., :, :, ::2]
        dat_out[..., :, :, 1::2] = torch.exp(-scl) * dat[..., :, :, 1::2]
    elif dim == 1:
        dat_out[..., :, ::2, :] = torch.exp(scl) * dat[..., :, ::2, :]
        dat_out[..., :, 1::2, :] = torch.exp(-scl) * dat[..., :, 1::2, :]
    else:
        dat_out[..., ::2, :, :] = torch.exp(scl) * dat[..., ::2, :, :]
        dat_out[..., 1::2, :, :] = torch.exp(-scl) * dat[..., 1::2, :, :]

    return dat_out


def _check_adjoint(po, method, bound, interpolation, dtype=torch.float32):
    """ Print adjointness of A and At operators:
        <Ay, x> - <Atx, y> \approx 0

    Args:
        po (_proj_op()): Encodes projection operator.
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
    Ay = _proj_apply('A', y, po, method=method, bound=bound, interpolation=interpolation)
    Atx = _proj_apply('At', x, po, method=method, bound=bound, interpolation=interpolation)
    # Check okay
    val = torch.sum(Ay * x, dtype=torch.float64) - torch.sum(Atx * y, dtype=torch.float64)
    # Print okay
    print('<Ay, x> - <Atx, y> = {}'.format(val))


def _proj(operator, dat, x, y, method='super-resolution', do=True, 
          rho=1, n=0, vx_y=None,
         interpolation='linear', bound='zero', diff='forward'):
    """ Projects image data by A, At or AtA.

    Args:
        operator (string): Either 'A', 'At ' or 'AtA'.
        dat (torch.Rensor): Image data (dim_x|dim_y).
        rho (torch.Tensor): ADMM step size, defaults to 1.
        n (int): Observation index, defaults to 0.
        vx_y (tuple(float)): Output voxel size.
        bound (str, optional): Bound for gradient/divergence calculation, defaults to
            constant zero.
        diff (str, optional): Gradient difference operator, defaults to 'forward'.

    Returns:
        dat_p (torch.tensor()): Projected image data (dim_y|dim_x).

    """
    if operator == 'AtA':
        # AtA
        # (dat = y)
        if not do:  # return dat
            operator = 'none'        
        dat = dat[None, None, ...]
        # sum likelihood terms
        dat_p = x[n].tau * _proj_apply(operator, dat, x[n].po, method=method,
                                    bound=bound, interpolation=interpolation)
        for n1 in range(1, len(x)):
            dat_p += x[n1].tau * _proj_apply(operator, dat, x[n1].po, method=method,
                                             bound=bound, interpolation=interpolation)
        dat_p = dat_p[0, 0, ...]
        # add prior term
        dat_p += rho * y.lam ** 2 * _DtD(dat[0, 0, ...], vx_y=vx_y, bound=bound, diff=diff)
    else:  
        # A, At
        # (dat = x or y)
        if not do:  # return dat
            operator = 'none'
        dat_p = _proj_apply(operator, dat[None, None, ...], x[n].po, method=method,
                         bound=bound, interpolation=interpolation)[0, 0, ...]        

    return dat_p


def _proj_apply(operator, dat, po, method='super-resolution', bound='zero', interpolation='linear'):
    """ Applies operator A, At  or AtA (for denoising or super-resolution).

    Args:
        operator (string): Either 'A', 'At', 'AtA' or 'none'.
        dat (torch.tensor()): Image data (1, 1, X_in, Y_in, Z_in).
        po (_proj_op()): Encodes projection operator, has the following fields:
            po.mat_x: Low-res affine matrix.
            po.mat_y: High-res affine matrix.
            po.mat_yx: Intermediate affine matrix.
            po.dim_x: Low-res image dimensions.
            po.dim_y: High-res image dimensions.
            po.dim_yx: Intermediate image dimensions.
            po.ratio: The ratio (low-res voxel_size)/(high-res voxel_size).
            po.smo_ker: Smoothing kernel (slice-profile).
        method (string): Either 'denoising' or 'super-resolution' (default).
        bound (str, optional): Bound for nitorch push/pull, defaults to 'zero'.
        interpolation (int, optional): Interpolation order, defaults to linear.

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
        mat = torch.linalg.solve(mat_y, rigid.mm(mat_yx))  # mat_y\rigid*mat_yx        
    elif method == 'denoising':
        dim = dim_x
        mat = torch.linalg.solve(mat_y, rigid.mm(mat_x))  # mat_y\rigid*mat_x        
    # Smoothing operator
    if len(ratio) == 3:  # 3D
        conv = lambda x: F.conv3d(x, smo_ker, stride=ratio)
        conv_transpose = lambda x: F.conv_transpose3d(x, smo_ker, stride=ratio)
    else:  # 2D
        conv = lambda x: F.conv2d(x, smo_ker, stride=ratio)
        conv_transpose = lambda x: F.conv_transpose2d(x, smo_ker, stride=ratio)
    # Get grid
    grid = affine_grid(mat.type(dat.dtype), dim, jitter=False)[None, ...]
    # Apply projection
    if method == 'super-resolution':
        extrapolate = False
        if operator == 'A':
            dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
            dat = conv(dat)
            if scl != 0:
                dat = _apply_scaling(dat, scl, dim_thick)
        elif operator == 'At':
            if scl != 0:
                dat = _apply_scaling(dat, scl, dim_thick)
            dat = conv_transpose(dat)
            dat = grid_push(dat, grid, shape=dim_y, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
        elif operator == 'AtA':
            dat = grid_pull(dat, grid, bound=bound, extrapolate=extrapolate, interpolation=interpolation)
            dat = conv(dat)
            if scl != 0:
                dat = _apply_scaling(dat, 2 * scl, dim_thick)
            dat = conv_transpose(dat)
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


def _proj_info(dim_y, mat_y, dim_x, mat_x, rigid=None,
              prof_ip=0, prof_tp=0, gap=0.0, device='cpu', scl=0.0,
              samp=0):
    """ Define projection operator object, to be used with _proj_apply.

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
        po (_proj_op()): Projection operator object.

    """
    # Get projection operator object
    po = _proj_op()
    # Data types
    dtype = torch.float64
    dtype_smo_ker = torch.float32
    # Output properties
    if not isinstance(dim_y, torch.Tensor):
        dim_y = torch.tensor(dim_y, device=device, dtype=dtype)
    po.dim_y = dim_y
    po.mat_y = mat_y
    po.vx_y = voxel_size(mat_y)
    # Input properties
    if not isinstance(dim_x, torch.Tensor):
        dim_x = torch.tensor(dim_x, device=device, dtype=dtype)
    po.dim_x = dim_x
    po.mat_x = mat_x
    po.vx_x = voxel_size(mat_x)
    # Number of dimensions
    ndim = len(dim_y)
    one = torch.tensor((1,) * ndim, device=device, dtype=torch.float64)
    if rigid is None:
        po.rigid = torch.eye(ndim + 1, device=device, dtype=dtype)
    else:
        po.rigid = rigid.type(dtype).to(device)
    # Slice-profile
    gap_cn = torch.zeros(ndim, device=device, dtype=dtype)
    profile = torch.tensor((prof_ip,) * ndim, device=device, dtype=dtype)
    dim_thick = torch.max(po.vx_x, dim=0)[1]
    gap_cn[dim_thick] = gap
    profile[dim_thick] = prof_tp
    po.dim_thick = dim_thick
    if samp > 0:
        # Sub-sampling
        samp = torch.tensor((samp,) * ndim, device=device, dtype=torch.float64)
        # Intermediate to lowres
        sk = torch.max(one, torch.floor(samp * one / po.vx_x + 0.5))
        D_x = torch.diag(torch.cat((sk, one[0, None])))
        po.D_x = D_x
        # Modulate lowres
        po.mat_x = po.mat_x.mm(D_x)
        po.dim_x = D_x.inverse()[:ndim, :ndim].mm(po.dim_x[..., None]).floor().squeeze()
        if torch.sum(torch.abs(po.vx_x - po.vx_x)) > 1e-4:
            # Intermediate to highres (only for superres)
            sk = torch.max(one, torch.floor(samp * one / po.vx_y + 0.5))
            D_y = torch.diag(torch.cat((sk, one[0, None])))
            po.D_y = D_y
            # Modulate highres
            po.mat_y = po.mat_y.mm(D_y)
            po.vx_y = voxel_size(po.mat_y)
            po.dim_y = D_y.inverse()[:ndim, :ndim].mm(po.dim_y[..., None]).floor().squeeze()
        po.vx_x = voxel_size(po.mat_x)
    # Make intermediate
    ratio = torch.linalg.solve(po.mat_y, po.mat_x)  # mat_y\mat_x
    ratio = (ratio[:ndim, :ndim] ** 2).sum(0).sqrt()
    ratio = ratio.ceil().clamp(1)  # ratio low/high >= 1
    mat_yx = torch.cat((ratio, torch.ones(1, device=device, dtype=dtype))).diag()
    po.mat_yx = po.mat_x.matmul(mat_yx.inverse())  # mat_x/mat_yx
    po.dim_yx = (po.dim_x - 1) * ratio + 1
    # Make elements with ratio <= 1 use dirac profile
    profile[ratio == 1] = -1
    profile = profile.int().tolist()
    # Make smoothing kernel (slice-profile)
    fwhm = (1. - gap_cn) * ratio
    smo_ker = smooth(profile, fwhm, sep=False, dtype=dtype_smo_ker, device=device)
    po.smo_ker = smo_ker
    # Add offset to intermediate space
    off = torch.tensor(smo_ker.shape[-ndim:], dtype=dtype, device=device)
    off = torch.div(-(off - 1), 2, rounding_mode='floor')  # set offset
    mat_off = torch.eye(ndim + 1, dtype=torch.float64, device=device)
    mat_off[:ndim, -1] = off
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


def _DtD(dat, vx_y, bound='zero', diff='forward'):
    """ Computes the divergence of the gradient.

    Args:
        dat (torch.tensor()): A tensor (dim_y).
        vx_y (tuple(float)): Output voxel size.
        bound (str, optional): Bound for gradient/divergence calculation, defaults to
            constant zero.
        diff (str, optional): Gradient difference operator, defaults to 'forward'.

    Returns:
          div (torch.tensor()): Dt(D(dat)) (dim_y).

    """
    dat = im_gradient(dat, vx=vx_y, bound=bound, which=diff)
    dat = im_divergence(dat, vx=vx_y, bound=bound, which=diff)
    
    return dat

