"""Experiment (beyond issue #12): FFT/circulant preconditioner for the SR Hessian.

Jacobi (issue #12) can't help because the Hessian diagonal is ~constant; the
ill-conditioning is spectral (the membrane D'D is a Laplacian, and A'A is a blur).
Both are (approximately) convolutions, so H is well approximated by a circulant
operator whose eigenvalues are the FFT of its point-spread function h = H(delta):

    M^{-1} r = IFFT( FFT(r) / S(k) ),   S(k) = FFT(ifftshift(H(delta_center))).

This diagonalises the (exact) membrane term and the (approx shift-invariant) blur.
Cost: two FFTs per CG iteration.

For each channel we compare CG iterations to a tight tolerance for:
  none (identity)  vs  Jacobi (unires._precond)  vs  Fourier (this),
at a fine (reg_scl=4) and coarse (reg_scl=32) scale, and confirm every solver
reaches the SAME solution of H y = b (correctness).

Run:
  docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app unires:latest \
    python benchmarks/experiment_fourier.py
"""
import io as _io
import re
import sys
from contextlib import redirect_stdout

import torch

from nitorch.io import map as io_map
from nitorch.spatial import voxel_size
from nitorch.core.optim import cg
from unires.struct import settings
from unires.run import init
from unires._update import _precond, _step_size
from unires._project import _proj

from bench_precond import simulate_lowres

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

RE_CG = re.compile(r'(?m)^\s*\d+\s*\|\s*a\s*=')
DATA = ['data/t1_icbm_normal_1mm_pn0_rf0.nii.gz',
        'data/t2_icbm_normal_1mm_pn0_rf0.nii.gz',
        'data/pd_icbm_normal_1mm_pn0_rf0.nii.gz']
AXES = [2, 1, 0]
FACTOR = 3
TOL = 1e-8
MAX_IT = 300


def count_cg(fn):
    buf = _io.StringIO()
    with redirect_stdout(buf):
        out = fn()
    return out, len(RE_CG.findall(buf.getvalue()))


def fourier_precond(H, dim, device):
    """Circulant preconditioner from the Hessian's point-spread function."""
    delta = torch.zeros(dim, device=device, dtype=torch.float32)
    delta[tuple(d // 2 for d in dim)] = 1.0
    psf = H(delta)                                   # centered PSF of H
    S = torch.fft.fftn(torch.fft.ifftshift(psf)).real
    negfrac = float((S < 0).float().mean())
    S = S.clamp_min(S.abs().max() * 1e-6)            # ensure SPD
    def precond(r):
        return torch.fft.ifftn(torch.fft.fftn(r) / S).real.to(r.dtype)
    return precond, negfrac


def main():
    dev = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    lowres = []
    for c, p in enumerate(DATA):
        f = io_map(p)
        gt = f.fdata().float().to(dev)
        mat = f.affine.float().to(dev)
        lr, lr_mat = simulate_lowres(gt, mat, AXES[c], FACTOR)
        lowres.append([lr, lr_mat])

    s = settings()
    s.device = dev
    s.vx = 1.0
    s.write_out = False
    with redirect_stdout(_io.StringIO()):
        x0, y0, s0 = init(lowres, s)
    rho = _step_size(x0, y0, s0)
    print(f'ADMM rho = {float(rho):.4g}; TOL={TOL}, MAX_IT={MAX_IT}\n')
    print(f'{"chan":>4} {"reg_scl":>8} {"none":>6} {"Jacobi":>8} {"Fourier":>8}   '
          f'{"solutions match (rel.diff vs none)":>36}   S<0%')
    print('-' * 96)

    for c in range(len(x0)):
        xc, yc = x0[c], y0[c]
        dim = yc.dim
        vx = voxel_size(yc.mat).float()
        lam0 = yc.lam0 if getattr(yc, 'lam0', None) is not None else yc.lam
        b = torch.randn(dim, device=dev, dtype=torch.float32)
        for scl in (4.0, 32.0):
            yc.lam = scl * lam0  # mirror fit()'s coarse-to-fine scaling

            def H(d):
                return _proj('AtA', d, xc, yc, method=s0.method, do=s0.do_proj, rho=rho,
                             vx_y=vx, bound=s0.bound, interpolation=s0.interpolation,
                             diff=s0.diff)

            pc_four, negfrac = fourier_precond(H, dim, dev)
            solvers = {
                'none': lambda r: r,
                'jac': _precond('jacobi', xc, yc, rho, s0),
                'four': pc_four,
            }
            res = {}
            sols = {}
            for name, pc in solvers.items():
                y, n = count_cg(lambda pc=pc: cg(A=H, b=b, x=torch.zeros_like(b),
                                                 precond=pc, max_iter=MAX_IT, tolerance=TOL,
                                                 stop='max_gain', verbose=True, inplace=False))
                res[name] = n
                sols[name] = y
            d_jac = float((sols['jac'] - sols['none']).norm() / sols['none'].norm())
            d_four = float((sols['four'] - sols['none']).norm() / sols['none'].norm())
            print(f'{c:>4} {scl:>8g} {res["none"]:>6} {res["jac"]:>8} {res["four"]:>8}   '
                  f'jac={d_jac:.1e}  four={d_four:.1e}{"":>10}   {negfrac*100:.1f}')

    print('\nInterpretation: large "none">"Fourier" gap = Fourier fixes the spectral '
          'ill-conditioning Jacobi cannot. Solutions must match none (valid SPD precond).')


if __name__ == '__main__':
    main()
