"""Rigorous, NON-circular correctness verification for the CG preconditioner (issue #12).

The benchmark's self-check only proves _precond faithfully implements the intended
formula, and "same fixed point" only proves it is *some* valid SPD preconditioner.
Neither proves M approximates the TRUE Hessian diagonal. This script closes that gap:

  (1) Formula faithfulness  : M == independent recompute of Sum_n tau_n A_n'A_n(1)
                              + rho lam^2 2 sum(1/vx^2).
  (2) Diagonal accuracy     : compare M against the EXACT Hessian diagonal
                              diag(H)[p] = H(e_p)[p] probed with unit impulses at
                              random voxels. For A with non-negative interpolation
                              weights, the row-sum A'A(1) >= diag(A'A), so we expect
                              M >= diag(H) (the intentional "more positive definite"
                              over-estimate the issue describes) -- i.e. ratio >= 1.
  (3) Solve correctness     : preconditioned CG must reach the SAME solution of
                              H y = b as a high-accuracy unpreconditioned reference,
                              and we report the iteration counts (efficacy).

Also reports the coefficient of variation of diag(H): if it is small, the diagonal
is ~constant, so a diagonal (Jacobi) preconditioner ~ a scalar and CANNOT help --
which explains a null speedup as a property of the operator, not a bug.

Run (in Docker):
  docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app unires:latest \
    python benchmarks/verify_precond.py
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
from unires._project import _proj, _proj_apply

from bench_precond import simulate_lowres  # reuse the low-res simulation

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

RE_CG = re.compile(r'(?m)^\s*\d+\s*\|\s*a\s*=')

DATA = ['data/t1_icbm_normal_1mm_pn0_rf0.nii.gz',
        'data/t2_icbm_normal_1mm_pn0_rf0.nii.gz',
        'data/pd_icbm_normal_1mm_pn0_rf0.nii.gz']
AXES = [2, 1, 0]
FACTOR = 3


def count_cg(fn):
    buf = _io.StringIO()
    with redirect_stdout(buf):
        out = fn()
    return out, len(RE_CG.findall(buf.getvalue()))


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
    print(f'ADMM rho = {float(rho):.4g}\n')

    all_ok = True
    for c in range(len(x0)):
        xc, yc = x0[c], y0[c]
        dim = yc.dim
        vx = voxel_size(yc.mat).float()
        ones = torch.ones(dim, device=dev, dtype=torch.float32)[None, None, ...]

        # M as produced by _precond (apply to ones, invert element-wise)
        M = 1.0 / _precond('jacobi', xc, yc, rho, s0)(ones)[0, 0, ...]

        # (1) Formula faithfulness -- independent recompute
        op = 'AtA' if s0.do_proj else 'none'
        ref = sum(xc[n].tau * _proj_apply(op, ones, xc[n].po, method=s0.method,
                                          bound=s0.bound, interpolation=s0.interpolation)
                  for n in range(len(xc)))[0, 0, ...].clamp(min=0)
        ref = (ref + 2 * rho * yc.lam ** 2 * vx.square().reciprocal().sum()).clamp(min=1e-7)
        f_ok = torch.allclose(M, ref, rtol=1e-4, atol=1e-4)

        # The channel's TRUE Hessian operator H(d) = sum_n tau_n A_n'A_n d + rho lam^2 D'D d
        def H(d):
            return _proj('AtA', d, xc, yc, method=s0.method, do=s0.do_proj, rho=rho,
                         vx_y=vx, bound=s0.bound, interpolation=s0.interpolation, diff=s0.diff)

        # (2) EXACT diagonal via unit impulses at K random interior voxels
        K = 64
        flat = torch.randint(0, int(torch.tensor(dim).prod()), (K,), device=dev)
        true_d, m_at = [], []
        for idx in flat.tolist():
            e = torch.zeros(dim, device=dev, dtype=torch.float32)
            e.view(-1)[idx] = 1.0
            true_d.append(float(H(e).view(-1)[idx]))
            m_at.append(float(M.view(-1)[idx]))
        true_d = torch.tensor(true_d)
        m_at = torch.tensor(m_at)
        ratio = (m_at / true_d.clamp_min(1e-12))
        # M should be >= true diagonal (row-sum over-estimate); allow tiny FP slack
        ge_ok = bool((ratio >= 1.0 - 1e-3).float().mean() > 0.98)
        # spatial variation of the TRUE diagonal (probe full field once via a dense
        # estimate: apply H to ones gives row-sums, but for CV of the diagonal we use
        # the sampled exact values)
        cv = float(true_d.std() / true_d.mean().clamp_min(1e-12))

        # (3) Solve correctness + iteration count on H y = b (random b)
        b = torch.randn(dim, device=dev, dtype=torch.float32)
        y_ref, n_ref = count_cg(lambda: cg(A=H, b=b, x=torch.zeros_like(b),
                                           precond=lambda r: r, max_iter=400,
                                           tolerance=1e-10, stop='max_gain',
                                           verbose=True, inplace=False))
        y_pc, n_pc = count_cg(lambda: cg(A=H, b=b, x=torch.zeros_like(b),
                                         precond=_precond('jacobi', xc, yc, rho, s0), max_iter=400,
                                         tolerance=1e-10, stop='max_gain',
                                         verbose=True, inplace=False))
        sol_rel = float((y_pc - y_ref).norm() / y_ref.norm().clamp_min(1e-12))
        # residual of the preconditioned solution (does PCG actually solve H y = b?)
        res_pc = float((H(y_pc) - b).norm() / b.norm())
        res_ref = float((H(y_ref) - b).norm() / b.norm())
        sol_ok = sol_rel < 1e-3

        ch_ok = f_ok and ge_ok and sol_ok
        all_ok = all_ok and ch_ok
        print(f'channel {c} (repeats={len(xc)}):')
        print(f'  (1) formula faithful (M == recompute)      : {f_ok}')
        print(f'  (2) M vs EXACT diag(H): ratio median={float(ratio.median()):.3f} '
              f'[{float(ratio.min()):.3f}, {float(ratio.max()):.3f}], M>=diag for '
              f'{float((ratio>=1-1e-3).float().mean()*100):.0f}% of probes -> ok={ge_ok}')
        print(f'      diag(H) coeff. of variation = {cv:.3f}  '
              f'({"~constant -> Jacobi cannot help" if cv < 0.35 else "varies"})')
        print(f'  (3) PCG vs reference solution rel.diff={sol_rel:.2e} -> ok={sol_ok}; '
              f'residuals: ref={res_ref:.1e} pc={res_pc:.1e}')
        print(f'      CG iterations to tol=1e-10:  unprecond={n_ref}  precond={n_pc}\n')

    print('=' * 56)
    print(f'  PRECONDITIONER IMPLEMENTATION: {"VERIFIED CORRECT" if all_ok else "PROBLEM DETECTED"}')
    print('=' * 56)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
