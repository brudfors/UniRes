"""Convergence diagnostic: does RED's advantage over MTV shrink as MTV converges?

Scans max_iter for MTV vs MTV+RED at a fixed mild (mu, sigma), reporting dPSNR/
dSSIM at each iteration budget. If dPSNR -> 0 as max_iter grows, RED mainly helps
UNDER-converged MTV (a convergence artifact) rather than the converged optimum."""
import argparse
from bench_dl_prior import simulate, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, nargs='+', default=[8, 12, 20, 30, 45])
    ap.add_argument('--mu', type=float, default=1e-4)
    ap.add_argument('--sigma', type=float, default=0.03)
    ap.add_argument('--channels', type=int, default=2)
    ap.add_argument('--sched_num', type=int, default=None,
                    help='coarse-to-fine levels; None=default, 0=off (sit at target reg)')
    args = ap.parse_args()
    gt, gt_mat, x, mat_x = simulate(args.channels)
    extra = {'tolerance': 0.0}
    if args.sched_num is not None:
        extra['sched_num'] = args.sched_num
    print(f"CONVERGENCE  channels={args.channels} mu={args.mu:g} sigma={args.sigma} "
          f"sched_num={args.sched_num}\n")
    print(f"{'max_iter':>8} | {'MTV PSNR':>9} {'RED PSNR':>9} {'dPSNR':>7} | "
          f"{'MTV SSIM':>9} {'RED SSIM':>9} {'dSSIM':>7}")
    print('-' * 70)
    for it in args.iters:
        m = run(gt, gt_mat, x, mat_x, 'mtv', it, **extra)
        r = run(gt, gt_mat, x, mat_x, 'mtv+red', it,
                red_mu=args.mu, red_sigma=args.sigma, red_denoiser='drunet', **extra)
        print(f"{it:>8} | {m['psnr']:9.3f} {r['psnr']:9.3f} {r['psnr']-m['psnr']:+7.3f} | "
              f"{m['ssim']:9.4f} {r['ssim']:9.4f} {r['ssim']-m['ssim']:+7.4f}", flush=True)


if __name__ == '__main__':
    main()
