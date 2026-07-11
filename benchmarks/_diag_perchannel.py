"""Per-channel diagnostic for the RED sweep: print PSNR/SSIM for EACH channel
(not just the average), for MTV vs a few mu values. Helps see whether RED helps
some contrasts (T1/T2) while dragging others (PD)."""
import argparse
from bench_dl_prior import simulate, run

CH_NAMES = ['T1', 'T2', 'PD']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', type=float, nargs='+', default=[0.0, 1e-5, 1e-4])
    ap.add_argument('--sigma', type=float, default=0.03)
    ap.add_argument('--max_iter', type=int, default=30)
    ap.add_argument('--channels', type=int, default=3)
    ap.add_argument('--denoiser', default='drunet',
                    help="RED backend: 'drunet' | 'biasfree3d' | 'gaussian'")
    ap.add_argument('--weights', default=None, help='checkpoint for biasfree3d')
    args = ap.parse_args()
    extra = {'tolerance': 0.0}
    if args.weights:
        extra['red_weights'] = args.weights
    gt, gt_mat, x, mat_x = simulate(args.channels)
    names = CH_NAMES[:args.channels]
    print(f"PER-CHANNEL  channels={args.channels} sigma={args.sigma} "
          f"max_iter={args.max_iter} denoiser={args.denoiser}\n")
    base = None
    for mu in args.mu:
        if mu == 0:
            r = run(gt, gt_mat, x, mat_x, 'mtv', args.max_iter, **extra)
            tag = 'mtv'
        else:
            r = run(gt, gt_mat, x, mat_x, 'mtv+red', args.max_iter,
                    red_mu=mu, red_sigma=args.sigma, red_denoiser=args.denoiser, **extra)
            tag = f'red mu={mu:g}'
        if base is None:
            base = r['psnr_ch'][:]
        parts = []
        for i, nm in enumerate(names):
            dp = r['psnr_ch'][i] - base[i]
            parts.append(f"{nm}: PSNR={r['psnr_ch'][i]:6.3f}({dp:+.3f}) "
                         f"SSIM={r['ssim_ch'][i]:.4f}")
        print(f"{tag:>14} | " + " | ".join(parts) + f" | avgP={r['psnr']:.3f}",
              flush=True)


if __name__ == '__main__':
    main()
