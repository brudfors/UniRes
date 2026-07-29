"""ADMM solver speed-up benchmark on REAL (already low-res) multi-contrast input images.

Compares named `settings` overrides ("configs") against a baseline, all from ONE shared
init (fair A/B). Each config is gated on QUALITY before its SPEED counts, so a change that
degrades the reconstruction is never reported as a win:

  quality gate (all must hold vs the baseline recon):
    * monotonic loss within each reg-scale segment,
    * final objective not worse:  fobj <= fobj_baseline * (1 + FOBJ_TOL),
    * reconstruction agreement:   NRMSE(recon, baseline_recon) <= AGREE_TOL;
  speed (deterministic first): outer iterations, total inner-CG iterations, then wall-clock.

The default configs compare 'legacy' (all speed-up knobs OFF = pre-speed-up behaviour) against
'default' (the shipped fast defaults), which reproduces the ~1.7x safe-stack speed-up. The
'no_*' configs turn a single knob back off to show its individual contribution.

Run (in Docker):
  docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app unires:latest \
    python benchmarks/bench_speedups_real.py \
      --data data/a.nii data/b.nii data/c.nii --out benchmarks/out --cgs_verbose
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
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import torch

from nitorch.io import map as io_map
from nitorch.spatial import voxel_size
from unires.struct import settings
from unires.run import init, fit

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# ---------------------------------------------------------------------------
# Config registry: name -> dict of `settings` overrides applied on top of the
# (fast) shipped defaults. 'legacy' turns every speed-up knob back off.
# ---------------------------------------------------------------------------
LEGACY = dict(alpha=1.0, rho_adapt=False, converged_countdown=6,
              cgs_tol_adapt=False, cache_grad=False)
CONFIGS = {
    'legacy':      dict(LEGACY),   # pre-speed-up behaviour (baseline)
    'default':     {},             # shipped fast defaults (safe stack)
    # turn ONE knob off, relative to the fast default, to isolate its contribution
    'no_relax':    {'alpha': 1.0},
    'no_rhoadapt': {'rho_adapt': False},
    'no_tailtrim': {'converged_countdown': 6},
    'no_inexact':  {'cgs_tol_adapt': False},
    'no_cache':    {'cache_grad': False},
}

# ---------------------------------------------------------------------------
# stdout parsing of fit()'s verbose 'fit-ll' + inner-CG lines
# ---------------------------------------------------------------------------
RE_LL = re.compile(
    r'(\d+)\s*-\s*Convergence\s*\(\s*([0-9.]+)\s*s\)\s*\|\s*'
    r'nlyx\s*=\s*(nan|[-+0-9.eE]+),\s*nlxy\s*=\s*(nan|[-+0-9.eE]+),\s*'
    r'nly\s*=\s*(nan|[-+0-9.eE]+),\s*gain\s*=\s*(nan|[-+0-9.eE]+)')
RE_OBS = re.compile(r'OBS: Regularisation changed')  # reg-scale change -> new monotonicity segment
RE_FIN = re.compile(r'finished in ([0-9.]+) seconds and (\d+) iterations')
RE_CG = re.compile(r'^\s*\d+\s*\|\s*a\s*=')  # one nitorch cg verbose line per inner iteration


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
            rows.append(dict(it=int(it), t=float(t), seg=seg, nlyx=float(nlyx),
                             nlxy=float(nlxy), nly=float(nly), gain=float(gain)))
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


def agreement(rec, base, dev):
    """Worst-channel NRMSE / max-abs of a reconstruction vs the baseline recon, over the
    baseline foreground. Both live on the same shared-init grid, so a solution-preserving
    speed-up should give ~0."""
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


def run_cfg(x0, y0, sett0, overrides):
    """Run fit() from the shared init with `overrides` applied to a deepcopy of settings."""
    x, y, sett = copy.deepcopy(x0), copy.deepcopy(y0), copy.deepcopy(sett0)
    for k, v in overrides.items():
        if not hasattr(sett, k):
            raise AttributeError(f"settings has no attribute '{k}' (config typo?)")
        setattr(sett, k, v)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True, help='Real input images, used as-is.')
    ap.add_argument('--out', default='benchmarks/out')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--configs', nargs='+', default=['legacy', 'default'],
                    help=f'Named configs from the registry: {list(CONFIGS)}')
    ap.add_argument('--cgs_verbose', action='store_true',
                    help='Count inner CG iterations (deterministic, hardware-independent).')
    ap.add_argument('--max_iter', type=int, default=None, help='Cap outer iterations (smoke test).')
    args = ap.parse_args()

    # First requested config is the baseline everything else is judged against.
    cfgs = args.configs
    unknown = [c for c in cfgs if c not in CONFIGS]
    if unknown:
        ap.error(f'unknown config(s) {unknown}; known: {list(CONFIGS)}')

    dev = torch.device('cpu' if not torch.cuda.is_available() else args.device)
    os.makedirs(args.out, exist_ok=True)

    print('== Input images (used as-is; default UniRes parameters) ==')
    data = []
    for p in args.data:
        f = io_map(p)
        dat = f.fdata().float().to(dev)
        mat = f.affine.float().to(dev)
        data.append([dat, mat])
        print(f'  {os.path.basename(p)}: dim={tuple(dat.shape)}  '
              f'vx={[round(v, 2) for v in voxel_size(mat).tolist()]} mm')

    s = settings()
    s.device = dev
    s.write_out = False
    s.cgs_verbose = args.cgs_verbose
    if args.max_iter is not None:
        s.max_iter = args.max_iter

    print('\n== Shared init (co-registration, hyper-parameters) at shipped defaults ==')
    print(f'   sched_num={s.sched_num}  alpha={s.alpha}  rho_adapt={s.rho_adapt}(mu={s.rho_adapt_mu})'
          f'  cgs_tol_adapt={s.cgs_tol_adapt}  converged_countdown={s.converged_countdown}'
          f'  cache_grad={s.cache_grad}')
    with redirect_stdout(_io.StringIO()):
        x0, y0, s0 = init(copy.deepcopy(data), copy.deepcopy(s))

    print('== Warmup (prime cuDNN kernels) ==')
    s_warm = copy.deepcopy(s0)
    s_warm.max_iter = 3
    with redirect_stdout(_io.StringIO()):
        run_cfg(x0, y0, s_warm, {})

    res = {}
    for name in cfgs:
        print(f'\n== Run: {name}   overrides={CONFIGS[name]} ==')
        res[name] = run_cfg(x0, y0, s0, CONFIGS[name])
        print('\n'.join(res[name]['log'].splitlines()[-3:]))

    base_name = cfgs[0]
    base = res[base_name]
    it = lambda r: r['fin'][1] if r['fin'] else len(r['rows'])
    base_obj = base['rows'][-1]['nlyx']

    stats = {}
    for name in cfgs:
        r = res[name]
        mono, _ = check_monotonic(r['rows'])
        nrmse, mx = agreement(r['dat_y'], base['dat_y'], dev)
        fobj = r['rows'][-1]['nlyx']
        stats[name] = dict(iters=it(r), wall=r['wall'], n_inner=r['n_inner'], fobj=fobj,
                           mono=mono, nrmse=nrmse, maxabs=mx,
                           fobj_rel=(fobj - base_obj) / max(abs(base_obj), 1e-12))

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap('tab10')
    for i, name in enumerate(cfgs):
        r = res[name]
        xs = [d['it'] for d in r['rows']]
        obj = [d['nlyx'] for d in r['rows']]
        ct = torch.tensor([d['t'] for d in r['rows']]).cumsum(0).tolist()
        ax[0].plot(xs, obj, color=cmap(i), label=name)
        ax[1].plot(ct, obj, color=cmap(i), label=name)
    for a, xl in ((ax[0], 'outer iteration'), (ax[1], 'cumulative time (s)')):
        a.set_xlabel(xl); a.set_ylabel('objective (nlyx = -log posterior)')
        a.set_yscale('log'); a.legend(); a.grid(True, which='both', alpha=0.3)
    ax[0].set_title('Convergence vs iteration (real data)')
    ax[1].set_title('Convergence vs wall-clock (real data)')
    fig.tight_layout()
    png = os.path.join(args.out, 'speedups_bench_real.png')
    fig.savefig(png, dpi=110)
    print(f'\nSaved plot -> {png}')

    # --- Report + gates -----------------------------------------------------
    FOBJ_TOL, AGREE_TOL = 5e-3, 2e-2
    bw, bi = stats[base_name]['wall'], stats[base_name]['n_inner']
    w = 14
    hdr = '  {:<24}'.format('') + ''.join(f'{p:>{w}}' for p in cfgs)
    print('\n' + '=' * len(hdr)); print(f'  RESULTS (real images, baseline = {base_name})'); print('=' * len(hdr))

    def line(name, fmt):
        print('  {:<24}'.format(name) + ''.join(f'{fmt(stats[p]):>{w}}' for p in cfgs))
    line('outer iterations', lambda s_: s_['iters'])
    if any(stats[p]['n_inner'] for p in cfgs):
        line('total inner CG iters', lambda s_: s_['n_inner'])
        line('inner-CG speedup', lambda s_: (f'{bi/max(s_["n_inner"],1):.2f}x' if s_['n_inner'] else '-'))
    line('wall-clock (s)', lambda s_: f'{s_["wall"]:.1f}')
    line('wall speedup', lambda s_: f'{bw/max(s_["wall"],1e-9):.2f}x')
    line('final objective', lambda s_: f'{s_["fobj"]:.5g}')
    line('obj reldiff (signed)', lambda s_: f'{s_["fobj_rel"]:+.1e}')
    line('monotonic loss', lambda s_: str(s_['mono']))
    line('recon NRMSE', lambda s_: f'{s_["nrmse"]:.1e}')
    print('=' * len(hdr))

    print('\n  gate: quality OK if  mono AND obj_reldiff <= {:.0e} AND NRMSE <= {:.0e}'
          .format(FOBJ_TOL, AGREE_TOL))
    for p in cfgs:
        s_ = stats[p]
        if p == base_name:
            print(f'  {p:<12} baseline'); continue
        quality_ok = s_['mono'] and (s_['fobj_rel'] <= FOBJ_TOL) and (s_['nrmse'] <= AGREE_TOL)
        faster = s_['wall'] < bw
        if not quality_ok:
            why = ([] + (['non-monotonic'] if not s_['mono'] else [])
                   + ([f'obj worse ({s_["fobj_rel"]:+.1e})'] if s_['fobj_rel'] > FOBJ_TOL else [])
                   + ([f'recon drift ({s_["nrmse"]:.1e})'] if s_['nrmse'] > AGREE_TOL else []))
            print(f'  {p:<12} DEGRADES / REJECT  [{", ".join(why)}]')
        else:
            tag = f'ACCEPT ({bw/max(s_["wall"],1e-9):.2f}x)' if faster else 'neutral (no speedup)'
            print(f'  {p:<12} quality OK, {tag}')
    sys.exit(0)


if __name__ == '__main__':
    main()
