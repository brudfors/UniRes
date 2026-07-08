"""3-way CG preconditioner benchmark on REAL (already low-res) input images.

Runs UniRes on the given images *as-is* at default settings, comparing the CG
preconditioners 'none', 'jacobi' and 'fourier'. Unlike bench_precond.py this does NOT
simulate low-res inputs and assumes NO ground truth, so:
  * correctness  : loss monotonic within each reg-scale segment, identical final
                   objective across preconditioners, and cross-preconditioner
                   reconstruction agreement (a valid preconditioner only changes CG
                   convergence speed, not the solution);
  * efficacy     : wall-clock and deterministic total inner-CG-iteration counts.

Reuses the machinery of bench_precond.py (run_config, check_monotonic).

Run (in Docker):
  docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app unires:latest \
    python benchmarks/bench_precond_real.py \
      --data data/a.nii.gz data/b.nii.gz data/c.nii.gz --out benchmarks/out --cgs_verbose
"""
import argparse
import copy
import io as _io
import os
import sys
from contextlib import redirect_stdout

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import torch

from nitorch.io import map as io_map
from nitorch.spatial import voxel_size
from unires.struct import settings
from unires.run import init

from bench_precond import run_config, check_monotonic  # reuse validated machinery

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def agreement(rec, base, dev):
    """NRMSE and max-abs of a reconstruction vs the baseline recon, over the
    baseline foreground, per channel (worst channel reported). Both are on the same
    (shared-init) grid, so a valid SPD preconditioner should give ~0."""
    worst_nrmse, worst_max = 0.0, 0.0
    for c in range(base.shape[-1]):
        b = base[..., c].to(dev)
        r = rec[..., c].to(dev)
        mask = b != 0
        if int(mask.sum()) == 0:
            mask = torch.ones_like(b, dtype=torch.bool)
        d = (r - b)[mask]
        rng = (b[mask].max() - b[mask].min()).clamp_min(1e-8)
        worst_nrmse = max(worst_nrmse, float(d.pow(2).mean().sqrt() / rng))
        worst_max = max(worst_max, float(d.abs().max() / rng))
    return worst_nrmse, worst_max


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True,
                    help='Real input images (multi-contrast), used as-is.')
    ap.add_argument('--out', default='benchmarks/out')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--types', nargs='+', default=['none', 'jacobi', 'fourier'],
                    choices=['none', 'jacobi', 'fourier'],
                    help="Preconditioners to compare (baseline 'none' is always included).")
    ap.add_argument('--cgs_verbose', action='store_true',
                    help='Count inner CG iterations (deterministic, hardware-independent).')
    ap.add_argument('--max_iter', type=int, default=None,
                    help='Cap outer iterations (for a quick smoke test; omit for default params).')
    args = ap.parse_args()
    if 'none' not in args.types:
        args.types = ['none'] + args.types

    dev = torch.device('cpu' if not torch.cuda.is_available() else args.device)
    os.makedirs(args.out, exist_ok=True)

    # --- Load the real images as-is (multi-contrast input) ------------------
    print('== Input images (used as-is; default UniRes parameters) ==')
    data = []
    for p in args.data:
        f = io_map(p)
        dat = f.fdata().float().to(dev)
        mat = f.affine.float().to(dev)
        data.append([dat, mat])
        print(f'  {os.path.basename(p)}: dim={tuple(dat.shape)}  '
              f'vx={[round(v, 2) for v in voxel_size(mat).tolist()]} mm')

    # --- Default settings + shared init -------------------------------------
    s = settings()                 # all defaults (vx=1.0, reg_scl, max_iter, tol, cgs_*, sched_num)
    s.device = dev
    s.write_out = False
    s.cgs_verbose = args.cgs_verbose
    if args.max_iter is not None:
        s.max_iter = args.max_iter

    print('\n== Shared init (co-registration, hyper-parameters) at default settings ==')
    with redirect_stdout(_io.StringIO()):
        x0, y0, s0 = init(copy.deepcopy(data), copy.deepcopy(s))

    # Warm up cuDNN autotune so every timed run uses cached kernels (fair wall-clock)
    print('== Warmup (prime cuDNN kernels) ==')
    s_warm = copy.deepcopy(s0)
    s_warm.max_iter = 3
    with redirect_stdout(_io.StringIO()):
        run_config(x0, y0, s_warm, 'none')

    # --- Run each preconditioner from the identical shared init -------------
    res = {}
    for ptype in args.types:
        print(f'\n== Run: precond = {ptype} ==')
        res[ptype] = run_config(x0, y0, s0, ptype)
        print(res[ptype]['log'])

    base = res['none']
    it = lambda r: r['fin'][1] if r['fin'] else len(r['rows'])

    stats = {}
    for ptype in args.types:
        r = res[ptype]
        mono, worst = check_monotonic(r['rows'])
        nrmse, mx = agreement(r['dat_y'], base['dat_y'], dev)
        fobj = r['rows'][-1]['nlyx']
        stats[ptype] = dict(
            iters=it(r), wall=r['wall'], n_inner=r['n_inner'], fobj=fobj,
            mono=mono, worst=worst, nrmse=nrmse, maxabs=mx,
            fobj_rel=abs(fobj - base['rows'][-1]['nlyx']) / max(abs(base['rows'][-1]['nlyx']), 1e-12))

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
    ax[0].set_title('Convergence vs iteration (real data)')
    ax[1].set_title('Convergence vs wall-clock (real data)')
    fig.tight_layout()
    png = os.path.join(args.out, 'precond_bench_real.png')
    fig.savefig(png, dpi=110)
    print(f'\nSaved plot -> {png}')

    # --- Report + gates -----------------------------------------------------
    FOBJ_TOL, AGREE_TOL = 1e-3, 2e-2
    b = stats['none']
    w = 14
    hdr = '  {:<26}'.format('') + ''.join(f'{p:>{w}}' for p in args.types)
    print('\n' + '=' * len(hdr)); print('  RESULTS (real images, baseline = none)'); print('=' * len(hdr))
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
    line('monotonic loss', lambda s_: str(s_['mono']))
    line('fixed-pt reldiff', lambda s_: f'{s_["fobj_rel"]:.1e}')
    line('recon agree (NRMSE)', lambda s_: f'{s_["nrmse"]:.1e}')
    print('=' * len(hdr))

    all_ok = True
    for p in args.types:
        s_ = stats[p]
        ok = s_['mono'] and s_['fobj_rel'] <= FOBJ_TOL and s_['nrmse'] <= AGREE_TOL
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
        print(f'  Best preconditioner: {best}  '
              f'({b["wall"]/max(stats[best]["wall"],1e-9):.2f}x wall-clock vs none, '
              f'correct={all_ok})')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
