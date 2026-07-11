"""Per-channel no-hallucination visual for a RED denoiser (default: bias-free 3D).

Renders, for each channel (T1/T2/PD), a mid axial slice: GT | MTV | MTV+RED |
(RED-MTV) diff. The diff panel is the no-hallucination check -- RED's changes must
be small-amplitude and EDGE-LOCALIZED (no invented structure). Saves benchmarks/
viz_denoiser.png and prints per-channel PSNR/SSIM.

  python benchmarks/viz_denoiser.py --weights /dev/shm/bf3d_best.pt \
      --denoiser biasfree3d --mu 0.03 --sigma 0.05 --channels 3
"""
import argparse
import contextlib
import io as _io
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, 'benchmarks')
from bench_dl_prior import simulate, resample_gt, score
from unires.struct import settings
from unires.run import preproc

DEV = 'cuda'
NAMES = ['T1', 'T2', 'PD']


def recon(x, mat_x, prior, max_iter, **kw):
    data = [[[x[i], mat_x[i]]] for i in range(len(x))]
    s = settings(); s.device = DEV; s.write_out = False; s.do_coreg = False
    s.unified_rigid = False; s.scaling = False; s.max_iter = max_iter
    s.tolerance = 0.0; s.prior = prior
    for k, v in kw.items():
        setattr(s, k, v)
    with contextlib.redirect_stdout(_io.StringIO()):
        y, ymat, _ = preproc(data, s)
    return y, ymat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=None)
    ap.add_argument('--denoiser', default='biasfree3d')
    ap.add_argument('--mu', type=float, default=0.03)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--channels', type=int, default=3)
    ap.add_argument('--max_iter', type=int, default=30)
    ap.add_argument('--vx_ts', type=float, default=4.0)
    ap.add_argument('--linesearch', action='store_true')
    args = ap.parse_args()

    gt, gt_mat, x, mat_x = simulate(args.channels, vx_ts=args.vx_ts)
    y_mtv, ymat = recon(x, mat_x, 'mtv', args.max_iter)
    kw = dict(red_mu=args.mu, red_sigma=args.sigma, red_denoiser=args.denoiser, red_linesearch=args.linesearch)
    if args.weights:
        kw['red_weights'] = args.weights
    y_red, ymat2 = recon(x, mat_x, 'mtv+red', args.max_iter, **kw)

    nrow = args.channels
    fig, ax = plt.subplots(nrow, 4, figsize=(14, 3.4 * nrow))
    if nrow == 1:
        ax = ax[None]
    for c in range(args.channels):
        gr = resample_gt(gt[c], gt_mat[c], y_mtv.shape[:3], ymat.to(DEV))
        pm, sm, _ = score(y_mtv[..., c], gr)
        pr, sr, _ = score(y_red[..., c], gr)
        print(f"{NAMES[c]}: MTV PSNR={pm:.3f}/SSIM={sm:.4f}  "
              f"RED PSNR={pr:.3f}/SSIM={sr:.4f}  dPSNR={pr-pm:+.3f} dSSIM={sr-sm:+.4f}",
              flush=True)
        gr = gr.cpu()

        def match(a):
            a = a.cpu(); m = gr > 0.02 * gr.max()
            s = (a[m] * gr[m]).sum() / (a[m] * a[m]).sum().clamp_min(1e-8)
            return a * s
        mtv, red = match(y_mtv[..., c]), match(y_red[..., c])
        z = gr.shape[2] // 2
        vmax = float(gr[gr > 0].quantile(0.99))
        diff = red[:, :, z] - mtv[:, :, z]
        dv = float(diff.abs().quantile(0.995)) + 1e-6
        cols = [(gr[:, :, z], 'GT', 'gray', 0, vmax),
                (mtv[:, :, z], f'MTV (PSNR {pm:.1f})', 'gray', 0, vmax),
                (red[:, :, z], f'MTV+RED-3D (PSNR {pr:.1f})', 'gray', 0, vmax),
                (diff, f'RED-MTV (dSSIM {sr-sm:+.3f})', 'coolwarm', -dv, dv)]
        for j, (img, title, cmap, lo, hi) in enumerate(cols):
            ax[c, j].imshow(img.T, cmap=cmap, vmin=lo, vmax=hi, origin='lower')
            ax[c, j].axis('off')
            ax[c, j].set_title(f'{NAMES[c]}: {title}' if j == 0 else title, fontsize=10)
    fig.suptitle(f'{args.denoiser}  mu={args.mu} sigma={args.sigma}  '
                 f'(diff = RED changes; edge-localized => no hallucination)')
    fig.tight_layout()
    fig.savefig('benchmarks/viz_denoiser.png', dpi=110)
    print('wrote benchmarks/viz_denoiser.png', flush=True)


if __name__ == '__main__':
    main()
