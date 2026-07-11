"""Deliverable comparison: MTV vs MTV+RED(DRUNet 2.5D) vs MTV+RED(bias-free 3D IXI).

Reports, per channel (T1/T2/PD) on the BrainWeb benchmark, all three objectives:
  * QUALITY   -- per-channel PSNR/SSIM vs GT (dPSNR vs the MTV baseline);
  * MONOTONE  -- reruns with the RED line-search + per-iteration objective eval and
                 checks the reported objective never increases;
  * RUNTIME   -- wall-clock per recon (one 3D pass vs 2.5D's 3*N_slices passes).

Usage (after training a 3D denoiser):
  python benchmarks/compare_denoisers.py --weights /dev/shm/denoiser_runs/biasfree3d_full/best.pt \
      --mu 0.1 --sigma 0.03 --channels 3 --max_iter 30
"""
import argparse
from bench_dl_prior import simulate, run

CH = ['T1', 'T2', 'PD']


def _fmt(r, base):
    parts = []
    for i, nm in enumerate(CH[:len(r['psnr_ch'])]):
        dp = r['psnr_ch'][i] - base['psnr_ch'][i] if base else 0.0
        parts.append(f"{nm} {r['psnr_ch'][i]:6.3f}({dp:+.3f})/{r['ssim_ch'][i]:.4f}")
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='bias-free 3D checkpoint (best.pt)')
    ap.add_argument('--mu', type=float, default=0.1)
    ap.add_argument('--sigma', type=float, default=0.03)
    ap.add_argument('--channels', type=int, default=3)
    ap.add_argument('--max_iter', type=int, default=30)
    ap.add_argument('--check_mono', action='store_true',
                    help='also rerun with line-search + objective eval to verify monotonicity')
    args = ap.parse_args()
    gt, gt_mat, x, mat_x = simulate(args.channels)
    fast = {'tolerance': 0.0}

    print(f"COMPARE  channels={args.channels} mu={args.mu:g} sigma={args.sigma} "
          f"max_iter={args.max_iter}\n"
          f"{'method':>22} | per-channel PSNR(dvsMTV)/SSIM {'':>10} | avgPSNR  time")
    print('-' * 108)
    configs = [
        ('MTV', dict(prior='mtv')),
        ('MTV+RED drunet(2.5D)', dict(prior='mtv+red', red_mu=args.mu, red_sigma=args.sigma,
                                      red_denoiser='drunet')),
        ('MTV+RED biasfree3d', dict(prior='mtv+red', red_mu=args.mu, red_sigma=args.sigma,
                                    red_denoiser='biasfree3d', red_weights=args.weights)),
    ]
    base = None
    for name, kw in configs:
        prior = kw.pop('prior')
        r = run(gt, gt_mat, x, mat_x, prior, args.max_iter, **kw, **fast)
        if base is None:
            base = r
        print(f"{name:>22} | {_fmt(r, base)} | {r['psnr']:7.3f}  {r['dt']:.0f}s", flush=True)

    if args.check_mono:
        print("\nMonotonicity (line-search on, per-iteration objective):")
        for name, kw in [('drunet', dict(red_denoiser='drunet')),
                         ('biasfree3d', dict(red_denoiser='biasfree3d', red_weights=args.weights))]:
            r = run(gt, gt_mat, x, mat_x, 'mtv+red', args.max_iter, red_mu=args.mu,
                    red_sigma=args.sigma, red_linesearch=True, tolerance=1e-4, **kw)
            print(f"  {name:>12}: monotone={r['mono']}  (avgPSNR={r['psnr']:.3f})", flush=True)


if __name__ == '__main__':
    main()
