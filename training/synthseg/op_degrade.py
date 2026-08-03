"""Operator-consistent degradation for the synth restoration prior.

The trilinear `_thick_slice_degrade` produces an iso volume whose blur/PSF and texture do NOT
match what the RED denoiser actually sees at inference: UniRes feeds it the ADMM reconstruction
estimate `y`, formed by the projector A (real slice profile + sampling) and a data-term solve.
Training on a naive trilinear up/down-sample is a Plug-and-Play distribution gap.

This module closes that gap: it forward-projects the clean iso volume through UniRes's OWN
projector A (matching the acquisition PSF and grid), then reconstructs back to the iso recon grid
with a few SIRT-style (density-normalised) data-term iterations -- an operator-consistent estimate
of what the denoiser sees at inference. The net then learns to map THAT -> clean, so training and
inference distributions match. `recon_iters=0` = pure adjoint/zero-filled back-projection; a few
iterations sharpen toward the (unregularised) least-squares solution, closer to the ADMM iterate.
"""
import torch
from unires._project import _proj_info, _proj_apply

_DT = torch.float64  # projector affines are double (grid math); data stays float
_PO_CACHE = {}       # (shape, axis, factor, device) -> po; the geometry is shared across a batch
                     # (batched_degrade uses ONE axis+factor per step) and reused across steps.


def _po_cached(shape, axis, factor, device):
    key = (tuple(shape), int(axis), int(round(float(factor))), str(device))
    po = _PO_CACHE.get(key)
    if po is None:
        po = _po_for(shape, axis, factor, device)
        _PO_CACHE[key] = po
    return po


def _po_for(shape, axis, factor, device):
    """Build the projector info for a 1mm-iso volume of `shape` acquired thick-by-`factor`
    along `axis` (in-plane 1mm, through-plane `factor`mm) -- mirrors bench_ixi.simulate_one."""
    dy = torch.as_tensor(shape, dtype=_DT)
    vx = torch.ones(3, dtype=_DT); vx[axis] = float(factor)
    mat = torch.eye(4, dtype=_DT)                                   # iso recon reference (1mm)
    ms = torch.diag(torch.cat((vx, torch.ones(1, dtype=_DT))))
    mat_x = mat.mm(ms)                                              # anisotropic acquisition affine
    dx = ms[:3, :3].inverse().mm(dy[:, None]).floor().squeeze()     # low-res dims
    return _proj_info(dy.to(device), mat.to(device), dx.to(device), mat_x.to(device),
                      prof_ip=0, prof_tp=0, gap=0.0, device=device, scl=0.0)


def op_degrade_batch(x, axis, factor, recon_iters=3):
    """x: [B,1,D,H,W] iso (GPU float), all sharing the SAME thick axis+factor. Returns the
    operator-consistent iso estimate (same grid) = SIRT reconstruction of A(x) through UniRes's
    projector, in ONE batched call (the projector broadcasts over B). factor<=1 -> passthrough."""
    if factor <= 1.0:
        return x
    po = _po_cached(tuple(x.shape[2:]), axis, factor, x.device)
    A = lambda v: _proj_apply("A", v, po)
    At = lambda v: _proj_apply("At", v, po)
    x_lo = A(x)                                                    # anisotropic low-res (real PSF)
    dens = At(torch.ones_like(x_lo)).clamp_min(1e-3)              # density (adjoint of ones)
    y = At(x_lo) / dens                                           # zero-filled back-projection
    for _ in range(int(recon_iters)):                             # SIRT data-term iterations
        y = (y + At(x_lo - A(y)) / dens).clamp_min(0.0)
    return torch.nan_to_num(y, nan=0.0, posinf=4.0, neginf=0.0).to(x.dtype)


def op_degrade_volume(clean, axis, factor, recon_iters=3):
    """Single-volume [D,H,W] convenience wrapper around op_degrade_batch."""
    return op_degrade_batch(clean[None, None], axis, factor, recon_iters)[0, 0]
