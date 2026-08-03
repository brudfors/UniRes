"""FINAL multi-model validation: every approach we tried, one shared MTV baseline per case.

Pairwise benches recompute MTV for each comparison and only fit two models at a time. This
evaluates an arbitrary LIST of priors against ONE MTV baseline per (contrast, thickness,
subject), so every model sees the IDENTICAL simulated acquisition -> fully paired, and the
per-case deltas support mean +/- sem (not just the pooled sd the pairwise benches print).

  python bench_final.py --dataset ixi --models "best=/path/a.pt,mo=/path/b.pt" \
      --n 8 --thick 2 4 6 --contrasts T1 T2 PD --mu 0.05 --red_nullspace 0.5 --out res.json
"""
import argparse, glob, json, math, os, statistics as st
from bench_ixi import load_gt, simulate_one, recon_score

#: optional reference prior to compare against (set $UNIRES_REF_PRIOR)
IXI_PRIOR = os.environ.get('UNIRES_REF_PRIOR')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['ixi', 'miccai'], default='ixi')
    ap.add_argument('--models', required=True, help='comma list name=path')
    ap.add_argument('--include_ixi_prior', type=int, default=1)
    ap.add_argument('--contrasts', nargs='+', default=['T1', 'T2', 'PD'])
    ap.add_argument('--thick', type=float, nargs='+', default=[2, 4, 6])
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--axis', type=int, default=2)
    ap.add_argument('--mu', type=float, default=0.05)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--max_iter', type=int, default=6)
    ap.add_argument('--noise_frac', type=float, default=0.03)
    ap.add_argument('--red_nullspace', type=float, default=0.5)
    ap.add_argument('--no_taper', type=int, default=1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    models = []
    if args.include_ixi_prior and IXI_PRIOR:
        models.append(('red(reference)', IXI_PRIOR))
    for tok in args.models.split(','):
        tok = tok.strip()
        if tok:
            name, path = tok.split('=', 1)
            models.append((name, path))

    kw = dict(red_mu=args.mu, red_sigma=args.sigma, red_denoiser='biasfree3d',
              red_linesearch=True)
    if args.no_taper:
        kw['red_thick_taper'] = False
    if args.red_nullspace > 0:
        kw['red_nullspace'] = args.red_nullspace

    if args.dataset == 'ixi':
        groups = [(c, sorted(glob.glob(os.path.join(os.environ.get("UNIRES_IXI_ROOT", "data/ixi"), c, "*.nii.gz")))[-args.n:])
                  for c in args.contrasts]
    else:
        paths = sorted(glob.glob(os.path.join(os.environ.get("UNIRES_MICCAI_ROOT", "data/miccai2012"), "*.nii"))
                       + glob.glob(os.path.join(os.environ.get("UNIRES_MICCAI_ROOT", "data/miccai2012"), "*.nii.gz")))[:args.n]
        groups = [('T1', paths)]

    print(f"FINAL VALIDATION | {args.dataset} n={args.n} axis={args.axis} mu={args.mu} "
          f"beta={args.red_nullspace} taper={'off' if args.no_taper else 'on'} models={len(models)}")
    res = {}
    for c, paths in groups:
        for th in args.thick:
            acc = {nm: {'dp': [], 'ds': []} for nm, _ in models}
            for i, p in enumerate(paths):
                vol, mat = load_gt(p)
                x, mat_x = simulate_one(vol, mat, args.axis, vx_ts=th,
                                        noise_frac=args.noise_frac, seed=i)
                pm, sm = recon_score(vol, mat, x, mat_x, 'mtv', max_iter=args.max_iter)
                for nm, ck in models:
                    pr, sr = recon_score(vol, mat, x, mat_x, 'mtv+red',
                                         max_iter=args.max_iter, red_weights=ck, **kw)
                    acc[nm]['dp'].append(pr - pm); acc[nm]['ds'].append(sr - sm)
            for nm, _ in models:
                dp, ds = acc[nm]['dp'], acc[nm]['ds']
                n = len(dp)
                sem = (st.pstdev(dp) / math.sqrt(n)) if n > 1 else 0.0
                key = f"{c}@{th:g}"
                res.setdefault(key, {})[nm] = dict(dPSNR=sum(dp)/n, sem=sem,
                                                   dSSIM=sum(ds)/n, per_case=dp)
                print(f"{key:>8} {nm:>22} dPSNR {sum(dp)/n:+7.3f} +/-{sem:.3f}  "
                      f"dSSIM {sum(ds)/n:+8.4f}", flush=True)
            print()
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)
        print("wrote", args.out)


if __name__ == '__main__':
    main()
