"""MULTI-ORIENTATION super-resolution benchmark — the regime UniRes is actually designed for.

The single-stack benchmarks (bench_ixi/bench_miccai) simulate ONE thick-slice acquisition, which
at 6mm is pathological: most through-plane detail lies in the forward operator's null space, so
NO prior can recover it (all priors sit near 0 or hurt). Real protocols instead acquire 2-3 thick
stacks in DIFFERENT orientations; each stack's in-plane detail fills the others' null space, and
the prior's job becomes sharpening genuinely-constrained structure rather than hallucinating.

Here we simulate `--n_ori` orthogonal thick stacks of the same subject (axial/coronal/sagittal),
feed them to UniRes as multiple observations of ONE channel, and score MTV vs MTV+RED.
"""
import argparse
import contextlib
import glob
import io as _io
import re
import statistics as st
import time
import torch

from unires.run import preproc
from unires.struct import settings
from bench_ixi import load_gt, simulate_one
from bench_dl_prior import resample_gt, score

DEV = 'cuda'


def run_multi(gt, gt_mat, obs, prior, max_iter, **kw):
    """obs = [(x, mat_x), ...] observations of ONE channel (different orientations)."""
    data = [[[x, m] for x, m in obs]]                 # 1 channel, N observations
    s = settings()
    s.device = DEV; s.write_out = False
    s.do_coreg = False; s.unified_rigid = False; s.scaling = False
    s.max_iter = max_iter; s.prior = prior
    s.red_batch = 16
    for k, v in kw.items():
        setattr(s, k, v)
    buf = _io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        y_hat, y_mat, _ = preproc(data, s)
    dt = time.time() - t0
    gt_r = resample_gt(gt, gt_mat.to(DEV), y_hat.shape[:3], y_mat.to(DEV))
    p, ss, _ = score(y_hat[..., 0], gt_r)
    return p, ss, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ixi_ckpt', default='/workspace/data/unires_dl_prior/checkpoints/biasfree3d_restore_thick.pt')
    ap.add_argument('--synth_ckpt', required=True)
    ap.add_argument('--synth_ckpt2', default=None)
    ap.add_argument('--contrasts', nargs='+', default=['T1', 'T2', 'PD'])
    ap.add_argument('--thick', type=float, nargs='+', default=[6])
    ap.add_argument('--n_ori', type=int, default=3, help='number of orthogonal thick stacks')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--mu', type=float, default=0.02)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--max_iter', type=int, default=6)
    ap.add_argument('--noise_frac', type=float, default=0.03)
    ap.add_argument('--root', default='/workspace/data/ixi')
    ap.add_argument('--no_taper', action='store_true')
    ap.add_argument('--red_nullspace', type=float, default=0.0,
                    help='restrict the prior correction to the operator null space (0=off, 1=full)')
    args = ap.parse_args()
    taper_kw = {'red_thick_taper': False} if args.no_taper else {}
    if args.red_nullspace > 0:
        taper_kw['red_nullspace'] = args.red_nullspace

    priors = [('red(IXI)', args.ixi_ckpt), ('red(synth)', args.synth_ckpt)]
    if args.synth_ckpt2:
        priors.append(('red(synth2)', args.synth_ckpt2))

    print(f"MULTI-ORIENTATION SR | n_ori={args.n_ori} n={args.n}/contrast mu={args.mu} "
          f"max_iter={args.max_iter} taper={'off' if args.no_taper else 'on'}")
    hdr = f"{'contrast':>8} {'thick':>5} {'prior':>14} {'dPSNR':>8} {'dSSIM':>8}"
    print(hdr); print('-' * len(hdr))
    for c in args.contrasts:
        paths = sorted(glob.glob(f"{args.root}/{c}/*.nii.gz"))[-args.n:]
        for th in args.thick:
            acc = {name: ([], []) for name, _ in priors}
            for i, p in enumerate(paths):
                vol, mat = load_gt(p)
                obs = []
                for a in range(args.n_ori):                     # orthogonal orientations
                    x, mx = simulate_one(vol, mat, a % 3, vx_ts=th,
                                         noise_frac=args.noise_frac, seed=i * 10 + a)
                    obs.append((x, mx))
                pm, sm, _ = run_multi(vol, mat, obs, 'mtv', args.max_iter)
                for name, ck in priors:
                    pr, sr, _ = run_multi(vol, mat, obs, 'mtv+red', args.max_iter,
                                          red_mu=args.mu, red_sigma=args.sigma,
                                          red_denoiser='biasfree3d', red_weights=ck,
                                          red_linesearch=True, **taper_kw)
                    acc[name][0].append(pr - pm); acc[name][1].append(sr - sm)
            for name, _ in priors:
                dP, dS = acc[name]
                n = len(dP)
                sd = st.pstdev(dP) if n > 1 else 0.0
                print(f"{c:>8} {th:>5g} {name:>14} {sum(dP)/n:>+8.3f} {sum(dS)/n:>+8.4f}"
                      f"   (+/-{sd:.3f})", flush=True)
        print()


if __name__ == '__main__':
    main()
