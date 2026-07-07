"""Benchmark: CG diagonal preconditioner (issue #12) — before vs after.

Simulates realistic low-resolution acquisitions from the 1 mm BrainWeb images
(so we have ground truth), then runs UniRes twice on identical data/initialisation
-- once WITHOUT the CG preconditioner (baseline) and once WITH it -- and compares:

  * convergence   : outer iterations, wall-clock, objective-vs-time / -vs-iteration
  * correctness   : objective (nlyx) monotonic within each regularisation scale;
                    same final objective (a valid SPD preconditioner only changes
                    speed, not the solution)
  * image quality : PSNR/SSIM of each reconstruction vs the 1 mm ground truth, plus
                    direct precond-vs-baseline agreement (they should be near-identical)

A valid preconditioner must not regress correctness or quality; if it does, the run
FAILS (nonzero exit). See benchmarks/README.md for how to run inside Docker.
"""
import argparse
import copy
import io as _io
import os
import re
import sys
from contextlib import redirect_stdout
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg')  # headless (no display in Docker)
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

# Seed for reproducibility. We keep cudnn.benchmark=True (as real UniRes does, for
# realistic timing); the A/B is made fair by a shared init + deepcopy, and cudnn
# autotune noise (~1e-6) is far below the agreement/fixed-point thresholds.
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from nitorch.io import map as io_map
from nitorch.spatial import affine_grid, grid_pull, voxel_size
from unires.struct import settings
from unires.run import init, fit


# ----------------------------------------------------------------------------
# Low-res simulation
# ----------------------------------------------------------------------------
def simulate_lowres(gt, mat, axis, factor):
    """Block-mean downsample `gt` along `axis` by `factor` and adjust the affine.

    Averaging `factor` adjacent slices then subsampling models a thick-slice
    acquisition with a rectangular through-plane profile (UniRes' default). The
    low-res voxel j maps to high-res index factor*j + (factor-1)/2, so the new
    voxel-to-world affine is  mat @ T  with T scaling `axis` by `factor` and
    shifting its origin by (factor-1)/2.
    """
    lr = gt.unfold(axis, factor, factor).mean(dim=-1).contiguous()
    T = torch.eye(4, dtype=mat.dtype, device=mat.device)
    T[axis, axis] = factor
    T[axis, 3] = (factor - 1) / 2.0
    return lr, mat @ T


# ----------------------------------------------------------------------------
# Image-quality metrics (self-contained; scikit-image is not a dependency)
# ----------------------------------------------------------------------------
def _gaussian_kernel3d(sigma, radius, device, dtype):
    c = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return k[None, None]


def _bbox(mask):
    idx = torch.nonzero(mask, as_tuple=False)
    lo = idx.min(dim=0).values
    hi = idx.max(dim=0).values + 1
    return tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))


def _norm(a):
    return a / a.max().clamp_min(1e-8)


def psnr(rec, gt, mask):
    """PSNR over the foreground, each image max-normalised to [0, 1]."""
    a = _norm(rec[mask])
    b = _norm(gt[mask])
    mse = ((a - b) ** 2).mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def ssim(rec, gt, mask, sigma=1.5, radius=3):
    """Windowed 3D SSIM over the mask bounding box, each max-normalised to [0, 1]."""
    sl = _bbox(mask)
    a = _norm(rec[sl].clamp_min(0))[None, None]
    b = _norm(gt[sl].clamp_min(0))[None, None]
    k = _gaussian_kernel3d(sigma, radius, a.device, a.dtype)
    p = radius
    mu_a = F.conv3d(a, k, padding=p)
    mu_b = F.conv3d(b, k, padding=p)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = F.conv3d(a * a, k, padding=p) - mu_a2
    vb = F.conv3d(b * b, k, padding=p) - mu_b2
    vab = F.conv3d(a * b, k, padding=p) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_ab + c1) * (2 * vab + c2)) / ((mu_a2 + mu_b2 + c1) * (va + vb + c2))
    return float(smap.mean())


def resample_gt(gt, mat_gt, mat_y, dim_y):
    """Resample a ground-truth channel onto the reconstruction grid."""
    M = torch.linalg.solve(mat_gt.double(), mat_y.double())  # recon-voxel -> gt-voxel
    grid = affine_grid(M.type(gt.dtype), list(dim_y), jitter=False)[None]
    return grid_pull(gt[None, None], grid, bound='zero', extrapolate=False,
                     interpolation=1)[0, 0]


# ----------------------------------------------------------------------------
# fit-ll stdout parsing
# ----------------------------------------------------------------------------
RE_LL = re.compile(
    r'(\d+)\s*-\s*Convergence\s*\(\s*([0-9.]+)\s*s\)\s*\|\s*'
    r'nlyx\s*=\s*(nan|[-+0-9.eE]+),\s*nlxy\s*=\s*(nan|[-+0-9.eE]+),\s*'
    r'nly\s*=\s*(nan|[-+0-9.eE]+),\s*gain\s*=\s*(nan|[-+0-9.eE]+)')
RE_OBS = re.compile(r'OBS: Regularisation changed')
RE_FIN = re.compile(r'finished in ([0-9.]+) seconds and (\d+) iterations')
RE_CG = re.compile(r'^\s*\d+\s*\|\s*a\s*=')  # nitorch cg verbose line (one per inner iteration)


def parse(text):
    rows, seg, fin, n_inner = [], 0, None, 0
    for line in text.splitlines():
        if RE_OBS.search(line):
            seg += 1
            continue
        if RE_CG.search(line):
            n_inner += 1
            continue
        m = RE_LL.search(line)
        if m:
            it, t, nlyx, nlxy, nly, gain = m.groups()
            rows.append(dict(it=int(it), t=float(t), seg=seg,
                             nlyx=float(nlyx), nlxy=float(nlxy),
                             nly=float(nly), gain=float(gain)))
            continue
        m = RE_FIN.search(line)
        if m:
            fin = (float(m.group(1)), int(m.group(2)))
    return rows, fin, n_inner


def check_monotonic(rows, atol=1e-6, rtol=1e-6):
    """nlyx must be non-increasing within each fixed regularisation segment."""
    worst = 0.0
    for i in range(1, len(rows)):
        cur, prev = rows[i]['nlyx'], rows[i - 1]['nlyx']
        if cur != cur:  # NaN
            return False, float('inf')
        if rows[i]['seg'] != rows[i - 1]['seg']:
            continue  # legitimate jump at reg-scale change
        inc = cur - prev
        if inc > atol + rtol * abs(prev):
            worst = max(worst, inc)
    return worst == 0.0, worst


# ----------------------------------------------------------------------------
# A/B run
# ----------------------------------------------------------------------------
def run_config(x0, y0, sett0, ptype):
    x = copy.deepcopy(x0)
    y = copy.deepcopy(y0)
    sett = copy.deepcopy(sett0)
    sett.cgs_precond = ptype  # 'none' | 'jacobi' | 'fourier'
    sett.write_out = False
    sett.do_print = 1
    assert sett.tolerance > 0, 'obj trajectory is only recorded when tolerance > 0'
    buf = _io.StringIO()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = timer()
    with redirect_stdout(buf):
        dat_y, mat_y, _, _, _, _ = fit(x, y, sett)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = timer() - t0
    text = buf.getvalue()
    rows, fin, n_inner = parse(text)
    return dict(rows=rows, fin=fin, n_inner=n_inner, wall=wall,
                dat_y=dat_y, mat_y=mat_y, log=text)


def verify_precond_math(x0, y0, sett0):
    """Independent recompute of the preconditioner diagonal vs unires._precond.

    Catches formula / summation bugs (e.g. not summing over repeats) that a
    monotonic loss cannot: both are SPD so both converge, but a wrong diagonal
    gives the wrong M. Also asserts strict positivity / finiteness.
    """
    from unires._update import _precond
    from unires._project import _proj_apply
    rho = torch.tensor(1.0, dtype=torch.float32, device=sett0.device)
    ok = True
    for c in range(len(x0)):
        xc, yc = x0[c], y0[c]
        vx = voxel_size(yc.mat).float()
        ones = torch.ones(yc.dim, device=sett0.device, dtype=torch.float32)[None, None, ...]
        op = 'AtA' if sett0.do_proj else 'none'
        ref = xc[0].tau * _proj_apply(op, ones, xc[0].po, method=sett0.method,
                                      bound=sett0.bound, interpolation=sett0.interpolation)
        for n in range(1, len(xc)):
            ref += xc[n].tau * _proj_apply(op, ones, xc[n].po, method=sett0.method,
                                           bound=sett0.bound, interpolation=sett0.interpolation)
        data = ref[0, 0, ...].clamp(min=0)
        ref = data + 2 * rho * yc.lam ** 2 * vx.square().reciprocal().sum()
        ref.clamp_(min=1e-7)
        got_inv = _precond('jacobi', xc, yc, rho, sett0)(ones)[0, 0, ...]  # M^{-1} applied to ones
        M_got = 1.0 / got_inv
        close = torch.allclose(M_got, ref, rtol=1e-4, atol=1e-4)
        positive = bool((M_got > 0).all()) and bool(torch.isfinite(M_got).all())
        ok = ok and close and positive
        print(f'  channel {c} (repeats={len(xc)}): matches recompute={close}, '
              f'M>0 & finite={positive}')
    return ok


def report_balance(x0, y0, s0):
    """Print the data-vs-membrane balance of M at the REAL rho / reg-scale range.

    A near-constant M (membrane >> spatially-varying data term) reduces the diagonal
    preconditioner to a scalar, which CG is invariant to -> no speedup in that regime.
    """
    from unires._update import _step_size
    from unires._project import _proj_apply
    from unires._core import _get_sched
    rho = _step_size(copy.deepcopy(x0), copy.deepcopy(y0), copy.deepcopy(s0))
    N = sum(len(xn) for xn in x0)
    ss = copy.deepcopy(s0)
    if not isinstance(ss.reg_scl, torch.Tensor):  # mirror run.py fit() before _get_sched
        ss.reg_scl = torch.tensor(ss.reg_scl, dtype=torch.float32, device=ss.device).reshape(1)
    sched = _get_sched(N, ss).reg_scl
    scls = [float(sched.min()), float(sched.max())]
    print(f'  ADMM rho={float(rho):.4g}; reg_scl schedule spans {scls[0]:g}..{scls[1]:g}')
    for c in range(len(x0)):
        xc, yc = x0[c], y0[c]
        vx = voxel_size(yc.mat).float()
        ones = torch.ones(yc.dim, device=s0.device, dtype=torch.float32)[None, None, ...]
        op = 'AtA' if s0.do_proj else 'none'
        data = xc[0].tau * _proj_apply(op, ones, xc[0].po, method=s0.method,
                                       bound=s0.bound, interpolation=s0.interpolation)
        for n in range(1, len(xc)):
            data += xc[n].tau * _proj_apply(op, ones, xc[n].po, method=s0.method,
                                            bound=s0.bound, interpolation=s0.interpolation)
        data = data[0, 0, ...].clamp(min=0)
        dmed = float(data.median())
        for scl in scls:
            lam = scl * yc.lam
            memb = float(2 * rho * lam ** 2 * vx.square().reciprocal().sum())
            frac = dmed / (dmed + memb) if (dmed + memb) > 0 else 0.0
            print(f'    ch{c} reg_scl={scl:>4g}: data(med)={dmed:.3g}  membrane={memb:.3g}  '
                  f'data-fraction of M ~ {frac:.1%}')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', default=[
        'data/t1_icbm_normal_1mm_pn0_rf0.nii.gz',
        'data/t2_icbm_normal_1mm_pn0_rf0.nii.gz',
        'data/pd_icbm_normal_1mm_pn0_rf0.nii.gz'],
        help='1 mm ground-truth images (used as GT and to simulate low-res inputs).')
    ap.add_argument('--out', default='benchmarks/out', help='Output directory.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vx', type=float, default=1.0, help='Reconstruction voxel size.')
    ap.add_argument('--factor', type=int, default=3, help='Slice downsample factor.')
    ap.add_argument('--sched', type=int, default=None,
                    help='Coarse-to-fine scalings (0 => single scale, strict monotone).')
    ap.add_argument('--reg_scl', type=float, default=None,
                    help='Regularisation scaling (scalar). Use with --sched 0 for the strict run.')
    ap.add_argument('--max_iter', type=int, default=None)
    ap.add_argument('--cgs_max_iter', type=int, default=None, help='Max inner CG iterations.')
    ap.add_argument('--cgs_tol', type=float, default=None, help='Inner CG tolerance.')
    ap.add_argument('--cgs_verbose', action='store_true',
                    help='Print per-inner-iteration CG lines (needed to count inner iterations).')
    ap.add_argument('--rho_scl', type=float, default=None,
                    help='Scale the ADMM step-size rho. Small values make the y-subproblem '
                         'data-dominated (worse-conditioned) -- the regime where the '
                         'preconditioner is expected to help most.')
    # Anisotropy directions per channel: T1 axial(z), T2 coronal(y), PD sagittal(x)
    ap.add_argument('--axes', nargs='+', type=int, default=[2, 1, 0])
    ap.add_argument('--types', nargs='+', default=['none', 'jacobi', 'fourier'],
                    choices=['none', 'jacobi', 'fourier'],
                    help="Preconditioners to compare (baseline 'none' must be included).")
    args = ap.parse_args()
    if 'none' not in args.types:
        args.types = ['none'] + args.types

    dev = torch.device('cpu' if not torch.cuda.is_available() else args.device)
    os.makedirs(args.out, exist_ok=True)

    # --- Load ground truth + simulate low-res inputs ------------------------
    print('== Simulating low-res inputs ==')
    gts, mats, lowres = [], [], []
    for c, pth in enumerate(args.data):
        f = io_map(pth)
        gt = f.fdata().float().to(dev)
        mat = f.affine.float().to(dev)
        axis = args.axes[c % len(args.axes)]
        lr, lr_mat = simulate_lowres(gt, mat, axis, args.factor)
        gts.append(gt)
        mats.append(mat)
        lowres.append([lr, lr_mat])
        print(f'  {os.path.basename(pth)}: GT {tuple(gt.shape)} vx={voxel_size(mat).tolist()}'
              f'  -> low-res {tuple(lr.shape)} vx={[round(v, 2) for v in voxel_size(lr_mat).tolist()]}'
              f' (axis {axis})')

    # --- Base settings + shared initialisation ------------------------------
    s = settings()
    s.device = dev
    s.vx = args.vx
    s.write_out = False
    if args.sched is not None:
        s.sched_num = args.sched
    if args.reg_scl is not None:
        s.reg_scl = args.reg_scl
    if args.max_iter is not None:
        s.max_iter = args.max_iter
    if args.cgs_max_iter is not None:
        s.cgs_max_iter = args.cgs_max_iter
    if args.cgs_tol is not None:
        s.cgs_tol = args.cgs_tol
    s.cgs_verbose = args.cgs_verbose
    if args.rho_scl is not None:
        s.rho_scl = args.rho_scl

    print('\n== Shared init (co-registration, hyper-parameters) ==')
    with redirect_stdout(_io.StringIO()):
        x0, y0, s0 = init(copy.deepcopy(lowres), copy.deepcopy(s))

    print('\n== Preconditioner math self-check ==')
    math_ok = verify_precond_math(x0, y0, s0)

    print('\n== M composition (data vs membrane) at real rho / reg scales ==')
    try:
        report_balance(x0, y0, s0)
    except Exception as e:  # diagnostic only — never abort the benchmark
        print(f'  (balance diagnostic skipped: {type(e).__name__}: {e})')

    # Warm up cudnn autotune (benchmark=True) so every timed run uses cached kernels
    # -- otherwise the first config pays the one-off autotuning cost and looks slow.
    print('\n== Warmup (prime cudnn kernels) ==')
    s_warm = copy.deepcopy(s0)
    s_warm.max_iter = 3
    with redirect_stdout(_io.StringIO()):
        run_config(x0, y0, s_warm, 'none')

    # --- Run each preconditioner (baseline 'none' first) --------------------
    res = {}
    for ptype in args.types:
        print(f'\n== Run: precond = {ptype} ==')
        res[ptype] = run_config(x0, y0, s0, ptype)
        print(res[ptype]['log'])

    base = res['none']
    dim_y = tuple(base['dat_y'].shape[:3])
    it = lambda r: r['fin'][1] if r['fin'] else len(r['rows'])

    # Ground truth resampled onto the recon grid (same grid for all configs)
    gt_on = []
    for c in range(len(gts)):
        gt_r = resample_gt(gts[c], mats[c], base['mat_y'].to(dev), dim_y)
        gt_on.append((gt_r, gt_r > 0))

    def quality(r):
        ps, ss = [], []
        for c in range(len(gts)):
            gt_r, mask = gt_on[c]
            rec = r['dat_y'][..., c].to(dev)
            ps.append(psnr(rec, gt_r, mask))
            ss.append(ssim(rec, gt_r, mask))
        return sum(ps) / len(ps), sum(ss) / len(ss)

    def agreement(r):  # vs baseline reconstruction (same fixed point => tiny)
        worst = 0.0
        for c in range(len(gts)):
            mask = gt_on[c][1]
            d = (r['dat_y'][..., c].to(dev) - base['dat_y'][..., c].to(dev))[mask]
            rng = (base['dat_y'][..., c].to(dev)[mask].max()
                   - base['dat_y'][..., c].to(dev)[mask].min()).clamp_min(1e-8)
            worst = max(worst, float(d.pow(2).mean().sqrt() / rng))
        return worst

    # --- Metrics per config -------------------------------------------------
    print('\n== Image quality vs 1 mm ground truth ==')
    stats = {}
    for ptype in args.types:
        r = res[ptype]
        mono, worst = check_monotonic(r['rows'])
        ps, ss = quality(r)
        fobj = r['rows'][-1]['nlyx']
        stats[ptype] = dict(
            iters=it(r), wall=r['wall'], n_inner=r['n_inner'], fobj=fobj,
            psnr=ps, ssim=ss, mono=mono, worst=worst,
            fobj_rel=abs(fobj - base['rows'][-1]['nlyx']) / max(abs(base['rows'][-1]['nlyx']), 1e-12),
            agree=agreement(r))

    # --- Plot (overlay all configs) -----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'none': 'tab:red', 'jacobi': 'tab:orange', 'fourier': 'tab:blue'}
    for ptype in args.types:
        r = res[ptype]
        xs = [d['it'] for d in r['rows']]
        obj = [d['nlyx'] for d in r['rows']]
        ct = torch.tensor([d['t'] for d in r['rows']]).cumsum(0).tolist()
        ax[0].plot(xs, obj, color=colors.get(ptype), label=ptype)
        ax[1].plot(ct, obj, color=colors.get(ptype), label=ptype)
    for a, xl in ((ax[0], 'outer iteration'), (ax[1], 'cumulative time (s)')):
        a.set_xlabel(xl); a.set_ylabel('objective (nlyx = -log posterior)')
        a.set_yscale('log'); a.legend(); a.grid(True, which='both', alpha=0.3)
    ax[0].set_title('Convergence vs iteration')
    ax[1].set_title('Convergence vs wall-clock')
    fig.tight_layout()
    png = os.path.join(args.out, 'precond_bench.png')
    fig.savefig(png, dpi=110)
    print(f'\nSaved plot -> {png}')

    # --- Report + gates -----------------------------------------------------
    PSNR_EPS, SSIM_EPS, AGREE_TOL, FOBJ_TOL = 0.1, 1e-3, 2e-2, 1e-3
    b = stats['none']
    w = 14
    hdr = '  {:<26}'.format('') + ''.join(f'{p:>{w}}' for p in args.types)
    print('\n' + '=' * len(hdr)); print('  RESULTS (baseline = none)'); print('=' * len(hdr))
    def line(name, fmt):
        print('  {:<26}'.format(name) + ''.join(f'{fmt(stats[p]):>{w}}' for p in args.types))
    line('outer iterations', lambda s_: s_['iters'])
    if any(stats[p]['n_inner'] for p in args.types):
        line('total inner CG iters', lambda s_: s_['n_inner'])
    line('wall-clock (s)', lambda s_: f'{s_["wall"]:.1f}')
    line('wall speedup vs none', lambda s_: f'{b["wall"]/max(s_["wall"],1e-9):.2f}x')
    line('inner-CG speedup vs none', lambda s_: (f'{b["n_inner"]/max(s_["n_inner"],1):.2f}x'
                                                 if s_['n_inner'] else '-'))
    line('final objective', lambda s_: f'{s_["fobj"]:.5g}')
    line('PSNR vs GT (dB)', lambda s_: f'{s_["psnr"]:.2f}')
    line('SSIM vs GT', lambda s_: f'{s_["ssim"]:.4f}')
    line('monotonic loss', lambda s_: str(s_['mono']))
    line('fixed-pt reldiff', lambda s_: f'{s_["fobj_rel"]:.1e}')
    line('recon agree (NRMSE)', lambda s_: f'{s_["agree"]:.1e}')
    print('=' * len(hdr))

    # Gates: every config must be correct + no quality regression vs baseline.
    all_ok = math_ok
    for p in args.types:
        s_ = stats[p]
        ok = (s_['mono'] and s_['fobj_rel'] <= FOBJ_TOL and s_['agree'] <= AGREE_TOL
              and s_['psnr'] >= b['psnr'] - PSNR_EPS and s_['ssim'] >= b['ssim'] - SSIM_EPS)
        all_ok = all_ok and ok
        faster = s_['wall'] < b['wall']
        tag = 'baseline' if p == 'none' else (
            f'{"OK" if ok else "FAIL"}, {"FASTER" if faster else "not faster"} '
            f'({b["wall"]/max(s_["wall"],1e-9):.2f}x)')
        print(f'  {p:<10} {tag}')

    best = min((p for p in args.types if p != 'none'),
               key=lambda p: stats[p]['wall'], default=None)
    print(f'\n  VERDICT: {"PASS" if all_ok else "FAIL"}')
    if best is not None:
        sp = b['wall'] / max(stats[best]['wall'], 1e-9)
        print(f'  Best preconditioner: {best}  ({sp:.2f}x wall-clock vs none, '
              f'correct & quality-preserving={all_ok})')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
