"""3-way prior comparison on MICCAI2012-TEST (real T1w OASIS volumes, single-channel SR).

A fresh real-data generalization test: 20 T1-weighted brain MRIs (OASIS, ~256^3, 1mm iso)
from a different scanner/cohort than IXI, and unseen by every prior. T1-only, so it directly
probes the one contrast where the real-data IXI prior currently leads on IXI. For each of N
subjects and each slice thickness: simulate a thick-slice acquisition, reconstruct with UniRes,
score dPSNR/dSSIM vs the real GT for MTV | red(IXI) | red(synth-patch) | red(synth-full).

Flat directory of *.nii (no contrast subdirs). Reuses bench_ixi machinery.
"""
import argparse
import os
import glob
import statistics as st
from bench_ixi import load_gt, simulate_one, recon_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ixi_ckpt', default=os.environ.get('UNIRES_REF_PRIOR'))
    ap.add_argument('--synth_patch_ckpt', required=True)
    ap.add_argument('--synth_full_ckpt', default=None)
    ap.add_argument('--thick', type=float, nargs='+', default=[2, 4])
    ap.add_argument('--n', type=int, default=8, help='held-out subjects')
    ap.add_argument('--mu', type=float, default=0.02)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--max_iter', type=int, default=6)
    ap.add_argument('--noise_frac', type=float, default=0.03)
    ap.add_argument('--axis', type=int, default=2)
    ap.add_argument('--root', default=os.environ.get('UNIRES_MICCAI_ROOT', 'data/miccai2012'))
    ap.add_argument('--no_taper', action='store_true',
                    help='disable red_thick_taper — probes whether the prior helps at extreme thickness')
    ap.add_argument('--red_nullspace', type=float, default=0.0,
                    help='restrict the prior correction to the operator null space (0=off, 1=full)')
    args = ap.parse_args()
    taper_kw = {} if not args.no_taper else {'red_thick_taper': False}
    if args.red_nullspace > 0:
        taper_kw['red_nullspace'] = args.red_nullspace

    priors = [('red(IXI)', args.ixi_ckpt), ('red(synth-patch)', args.synth_patch_ckpt)]
    if args.synth_full_ckpt:
        priors.append(('red(synth-full)', args.synth_full_ckpt))

    paths = sorted(glob.glob(f"{args.root}/*.nii") + glob.glob(f"{args.root}/*.nii.gz"))[:args.n]
    print(f"MICCAI2012-TEST real-T1 SR | n={len(paths)} mu={args.mu} sigma={args.sigma} "
          f"noise={args.noise_frac} max_iter={args.max_iter}")
    hdr = f"{'thick':>5} {'prior':>18} {'dPSNR':>8} {'dSSIM':>8}"
    print(hdr); print('-' * len(hdr))
    for th in args.thick:
        acc = {name: ([], []) for name, _ in priors}
        for i, p in enumerate(paths):
            vol, mat = load_gt(p)
            x, mat_x = simulate_one(vol, mat, args.axis, vx_ts=th,
                                    noise_frac=args.noise_frac, seed=i)
            pm, sm = recon_score(vol, mat, x, mat_x, 'mtv', max_iter=args.max_iter)
            for name, ck in priors:
                pr, sr = recon_score(vol, mat, x, mat_x, 'mtv+red', max_iter=args.max_iter,
                                     red_mu=args.mu, red_sigma=args.sigma,
                                     red_denoiser='biasfree3d', red_weights=ck,
                                     red_linesearch=True, **taper_kw)
                acc[name][0].append(pr - pm); acc[name][1].append(sr - sm)
        for name, _ in priors:
            dP, dS = acc[name]
            n = len(dP)
            sd = st.pstdev(dP) if n > 1 else 0.0
            print(f"{th:>5g} {name:>18} {sum(dP)/n:>+8.3f} {sum(dS)/n:>+8.4f}"
                  f"   (+/-{sd:.3f})", flush=True)
        print()


if __name__ == '__main__':
    main()
