"""Render a UniRes reconstruction with and without the DL (RED) prior as a PNG.

Two modes:

  REAL input (no ground truth) -- reconstruct the given thick-slice volume(s) twice and show
  MTV | MTV+RED | their difference, so the prior's effect is directly visible:

      python benchmarks/viz_prior.py --input sub_ax.nii.gz [sub_cor.nii.gz ...] --out viz.png

  SIMULATED (has ground truth) -- take an isotropic volume, simulate a thick-slice
  acquisition from it, and show GT | MTV | MTV+RED | error maps with PSNR/SSIM:

      python benchmarks/viz_prior.py --gt iso.nii.gz --thick 4 --out viz.png

Slices are matched across panels and share one intensity window, so panels are comparable;
difference maps use a symmetric window. Writes a PNG (no display needed).
"""
import argparse
import contextlib
import io as _io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nitorch import io as nio                                   # noqa: E402
from unires.run import preproc                                  # noqa: E402
from unires.struct import settings                              # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _recon(data, prior, **kw):
    s = settings()
    s.device = DEV
    s.write_out = False
    s.do_coreg = len(data) > 1
    s.prior = prior
    for k, v in kw.items():
        setattr(s, k, v)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):                        # UniRes is chatty
        y, mat, _ = preproc(data, s)
    return y, mat


def _slices(vol):
    """Three orthogonal mid-slices of a [D,H,W] tensor, as numpy, correctly oriented."""
    v = vol.detach().float().cpu().numpy()
    d, h, w = v.shape
    return [np.rot90(v[d // 2, :, :]), np.rot90(v[:, h // 2, :]), np.rot90(v[:, :, w // 2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', help='real thick-slice volume(s); one per orientation')
    ap.add_argument('--gt', help='isotropic volume to simulate a thick-slice acquisition from')
    ap.add_argument('--thick', type=float, default=4.0, help='simulated slice thickness (--gt mode)')
    ap.add_argument('--axis', type=int, default=2, help='simulated thick axis (--gt mode)')
    ap.add_argument('--noise_frac', type=float, default=0.03)
    ap.add_argument('--mu', type=float, default=0.05, help='red_mu (0.05 = validated)')
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--denoiser', default='biasfree3d')
    ap.add_argument('--weights', default=None, help='checkpoint; default = auto-resolve/download')
    ap.add_argument('--max_iter', type=int, default=6)
    ap.add_argument('--out', default='viz_prior.png')
    args = ap.parse_args()
    if not args.input and not args.gt:
        ap.error('give --input (real data) or --gt (simulate)')

    red_kw = dict(red_mu=args.mu, red_sigma=args.sigma, red_denoiser=args.denoiser,
                  max_iter=args.max_iter)
    if args.weights:
        red_kw['red_weights'] = args.weights

    if args.gt:                                                  # ---- simulated, has GT ----
        from bench_ixi import load_gt, simulate_one
        from bench_dl_prior import resample_gt, score
        vol, mat = load_gt(args.gt)
        x, mat_x = simulate_one(vol, mat, args.axis, vx_ts=args.thick,
                                noise_frac=args.noise_frac, seed=0)
        data = [[[x, mat_x]]]
        y_mtv, m1 = _recon(data, 'mtv', max_iter=args.max_iter)
        y_red, m2 = _recon(data, 'mtv+red', **red_kw)
        gt1 = resample_gt(vol, mat.to(DEV), y_mtv.shape[:3], m1.to(DEV))
        gt2 = resample_gt(vol, mat.to(DEV), y_red.shape[:3], m2.to(DEV))
        p1, s1, _ = score(y_mtv[..., 0], gt1)
        p2, s2, _ = score(y_red[..., 0], gt2)
        panels = [('ground truth', gt1, None),
                  (f'MTV\nPSNR {p1:.2f} / SSIM {s1:.3f}', y_mtv[..., 0], None),
                  (f'MTV+RED (mu={args.mu})\nPSNR {p2:.2f} / SSIM {s2:.3f}', y_red[..., 0], None),
                  ('|MTV - GT|', (y_mtv[..., 0] - gt1).abs(), 'err'),
                  ('|MTV+RED - GT|', (y_red[..., 0] - gt2).abs(), 'err')]
        print(f'MTV      PSNR {p1:.4f}  SSIM {s1:.4f}')
        print(f'MTV+RED  PSNR {p2:.4f}  SSIM {s2:.4f}   dPSNR {p2 - p1:+.3f}  dSSIM {s2 - s1:+.4f}')
    else:                                                        # ---- real input, no GT ----
        data = []
        for p in args.input:
            n = nio.map(p)
            data.append([n.fdata().to(DEV).float(), n.affine.cpu().double()])
        data = [[[d[0], d[1]] for d in data]]                    # one channel, N observations
        y_mtv, _ = _recon(data, 'mtv', max_iter=args.max_iter)
        y_red, _ = _recon(data, 'mtv+red', **red_kw)
        diff = y_red[..., 0] - y_mtv[..., 0]
        panels = [('MTV', y_mtv[..., 0], None),
                  (f'MTV+RED (mu={args.mu})', y_red[..., 0], None),
                  ('RED - MTV', diff, 'diff')]
        print(f'reconstructed {len(args.input)} observation(s); '
              f'RED-MTV abs mean {diff.abs().mean():.5f}, max {diff.abs().max():.5f}')

    # shared window over the image panels so they are visually comparable
    imgs = [p[1] for p in panels if p[2] is None]
    hi = float(torch.quantile(torch.stack([i.flatten() for i in imgs]).flatten()[:2_000_000], 0.99))
    fig, axes = plt.subplots(3, len(panels), figsize=(3.1 * len(panels), 9))
    axes = np.atleast_2d(axes)
    for col, (title, vol_, kind) in enumerate(panels):
        sl = _slices(vol_)
        for row in range(3):
            ax = axes[row, col]
            if kind is None:
                ax.imshow(sl[row], cmap='gray', vmin=0, vmax=hi)
            elif kind == 'err':
                ax.imshow(sl[row], cmap='inferno', vmin=0, vmax=hi * 0.35)
            else:
                m = np.abs(sl[row]).max() or 1.0
                ax.imshow(sl[row], cmap='bwr', vmin=-m, vmax=m)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
