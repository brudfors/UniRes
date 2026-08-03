"""4-way prior comparison on REAL IXI data (single-channel super-resolution).

Unlike BrainWeb (a clean phantom), IXI is real acquired MRI (real texture + scanner noise),
so it's the honest test of synthetic-vs-real transfer -- and IXI is the IXI-prior's home turf,
setting a fair bar. For each contrast (T1/T2/PD), N held-out subjects, and slice thickness,
simulate a thick-slice acquisition, reconstruct with UniRes, and score dPSNR vs the real IXI GT
for: MTV | red(IXI) | red(synth-patch) | red(synth-full).

Uses the LAST N subjects (sorted) to reduce IXI-prior training leakage. Reuses bench_ixi machinery.
"""
import argparse
import glob
import statistics as st
from bench_ixi import load_gt, simulate_one, recon_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ixi_ckpt', default='/workspace/data/unires_dl_prior/checkpoints/biasfree3d_restore_thick.pt')
    ap.add_argument('--synth_patch_ckpt', required=True)
    ap.add_argument('--synth_full_ckpt', default=None)
    ap.add_argument('--contrasts', nargs='+', default=['T1', 'T2', 'PD'])
    ap.add_argument('--thick', type=float, nargs='+', default=[2, 4, 6])
    ap.add_argument('--n', type=int, default=2, help='held-out IXI subjects per contrast')
    ap.add_argument('--mu', type=float, default=0.02)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--max_iter', type=int, default=6)
    ap.add_argument('--noise_frac', type=float, default=0.03)
    ap.add_argument('--axis', type=int, default=2)
    ap.add_argument('--root', default='/workspace/data/ixi')
    ap.add_argument('--no_taper', action='store_true',
                    help='disable red_thick_taper (the thickness safety-fade) — probes whether the prior helps at extreme thickness')
    ap.add_argument('--red_nullspace', type=float, default=0.0,
                    help='restrict the prior correction to the operator null space (0=off, 1=full)')
    ap.add_argument('--thick_full', type=float, default=None,
                    help='retune red_thick_full (thickness factor up to which mu is at FULL strength)')
    ap.add_argument('--thick_zero', type=float, default=None,
                    help='retune red_thick_zero (thickness factor at which mu reaches 0)')
    args = ap.parse_args()
    taper_kw = {} if not args.no_taper else {'red_thick_taper': False}
    if args.thick_full is not None or args.thick_zero is not None:
        taper_kw['red_thick_taper'] = True          # retuned taper => resolution-adaptive mu
        if args.thick_full is not None:
            taper_kw['red_thick_full'] = args.thick_full
        if args.thick_zero is not None:
            taper_kw['red_thick_zero'] = args.thick_zero
    if args.red_nullspace > 0:
        taper_kw['red_nullspace'] = args.red_nullspace

    priors = [('red(IXI)', args.ixi_ckpt), ('red(synth-patch)', args.synth_patch_ckpt)]
    if args.synth_full_ckpt:
        priors.append(('red(synth-full)', args.synth_full_ckpt))

    print(f"IXI real-anatomy 4-way SR | n={args.n}/contrast mu={args.mu} sigma={args.sigma} "
          f"noise={args.noise_frac} max_iter={args.max_iter}")
    hdr = f"{'contrast':>8} {'thick':>5} {'prior':>18} {'dPSNR':>8} {'dSSIM':>8}"
    print(hdr); print('-' * len(hdr))
    for c in args.contrasts:
        paths = sorted(glob.glob(f"{args.root}/{c}/*.nii.gz"))[-args.n:]
        for th in args.thick:
            # accumulate per-prior dPSNR/dSSIM over subjects (baseline = MTV per subject)
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
                print(f"{c:>8} {th:>5g} {name:>18} {sum(dP)/n:>+8.3f} {sum(dS)/n:>+8.4f}"
                      f"   (+/-{sd:.3f})", flush=True)
        print()


if __name__ == '__main__':
    main()
