"""Evaluate the SynthSeg (contrast-agnostic) restoration prior vs the IXI-trained one.

Reconstructs thick-slice BrainWeb with UniRes + RED prior, scoring PSNR/SSIM vs the 1mm
GT, for three priors:
  * mtv               -- classic multi-channel TV (baseline, no DL prior)
  * mtv+red (IXI)     -- biasfree3d trained on real IXI patches
  * mtv+red (synth)   -- biasfree3d trained on SYNTHESIZED random-contrast images (this work)
across slice thickness 2/4/6 mm, on:
  A) SEEN contrasts   -- BrainWeb T1/T2/PD (IXI model's home turf).
  B) NOVEL contrast   -- a FLAIR-like remap (CSF suppressed => a tissue-intensity ORDERING
                         neither model saw as REAL data). This is the contrast-agnosticism test.

Usage:
  PYTHONPATH=/workspace/code/UniRes python training/synthseg/eval_synth.py \
     --synth_ckpt /workspace/data/synthseg_restore/runs/synth_thick/best.pt \
     --ixi_ckpt  /workspace/data/unires_dl_prior/checkpoints/biasfree3d_restore_thick.pt \
     --thick 2 4 6 --mu 0.02 --sigma 0.05
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'benchmarks'))

from nitorch import io  # noqa: E402
import bench_dl_prior as bench  # noqa: E402


def make_flair_like(out_path):
    """Build a FLAIR-like NOVEL-contrast volume from BrainWeb T1/T2/PD: T2 weighting with
    CSF (bright in T2 & PD) SUPPRESSED -> the brightest tissue becomes dark, a tissue
    ordering absent from any of T1/T2/PD. Saved on T2's grid."""
    t1 = io.map(bench.PATHS[0]).fdata(); t2 = io.map(bench.PATHS[1]).fdata()
    pd = io.map(bench.PATHS[2]).fdata()
    aff = io.map(bench.PATHS[1]).affine
    n = lambda v: v / v[v > 0].float().quantile(0.99).clamp_min(1e-6)
    t2n, pdn, t1n = n(t2), n(pd), n(t1)
    # CSF: bright in PD and T2, dark in T1
    csf = (pdn.clamp(0, 1) * t2n.clamp(0, 1) * (1 - t1n).clamp(0, 1))
    csf = (csf / csf[csf > 0].float().quantile(0.95).clamp_min(1e-6)).clamp(0, 1)
    flair = (t2n * (1.0 - 0.9 * csf)).clamp_min(0) * 1000.0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    io.save(flair, out_path, affine=aff)
    return out_path


def make_novel_random(out_dir, n_ch=3, nbins=10, seed=0):
    """Build n_ch STRONGLY out-of-distribution random-contrast channels from BrainWeb anatomy:
    quantise the T1 foreground into `nbins` intensity regions, then assign each region a fresh
    RANDOM intensity per channel (+ light smoothing). This is exactly the SynthSeg training
    distribution (per-label random Gaussian intensities) applied to REAL anatomy -> a contrast
    set the IXI prior NEVER saw but the synth prior was designed for. Multi-channel (like the
    seen T1/T2/PD test) so the reconstruction has the SAME restoration headroom -> discriminative,
    unlike the near-zero single-channel FLAIR-like test. Returns list of n_ch file paths."""
    import torch
    import torch.nn.functional as F
    t1 = io.map(bench.PATHS[0]).fdata().float()
    aff = io.map(bench.PATHS[0]).affine
    pos = t1 > 0
    # quantise the foreground into nbins anatomy regions by T1 intensity quantiles
    qs = torch.quantile(t1[pos], torch.linspace(0, 1, nbins + 1))
    lab = torch.bucketize(t1, qs[1:-1]).clamp(0, nbins - 1) + 1     # 1..nbins on fg
    lab[~pos] = 0                                                    # 0 = background

    def _blur(v, r=2):
        k = torch.exp(-(torch.arange(-r, r + 1).float() ** 2) / 2.0); k = k / k.sum()
        x = v[None, None]
        for ax in range(3):
            shp = [1, 1, 1, 1, 1]; shp[2 + ax] = k.numel()
            pad = [0] * 6; pad[2 * (2 - ax)] = r; pad[2 * (2 - ax) + 1] = r
            x = F.conv3d(F.pad(x, pad, mode="reflect"), k.view(shp))
        return x[0, 0]

    g = torch.Generator().manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for c in range(n_ch):
        mean = torch.rand(nbins + 1, generator=g)      # random intensity per region
        mean[0] = 0.0                                    # background dark (realistic air)
        img = _blur(mean[lab]).clamp_min(0)
        img = img / img[img > 0].quantile(0.99).clamp_min(1e-6) * 1000.0
        p = os.path.join(out_dir, f"novel_rand_{c}.nii.gz")
        io.save(img, p, affine=aff)
        paths.append(p)
    return paths


def eval_one(paths, n_ch, thick_list, mu, sigma, max_iter, ckpts, taper=True):
    """For each thickness, run MTV + each RED ckpt; return rows of (thick, name, PSNR, per-ch)."""
    old = bench.PATHS
    bench.PATHS = paths
    rows = []
    try:
        for th in thick_list:
            gt, gt_mat, x, mat_x = bench.simulate(n_ch, vx_ts=th, std_noise=50.0, seed=0)
            r0 = bench.run(gt, gt_mat, x, mat_x, 'mtv', max_iter, tolerance=0.0)
            rows.append((th, 'mtv', r0['psnr'], r0['ssim'], r0['psnr_ch']))
            base = r0['psnr']
            for name, ck in ckpts:
                r = bench.run(gt, gt_mat, x, mat_x, 'mtv+red', max_iter,
                              red_mu=mu, red_sigma=sigma, red_denoiser='biasfree3d',
                              red_weights=ck, red_thick_taper=taper, tolerance=0.0)
                rows.append((th, name, r['psnr'], r['ssim'], r['psnr_ch'],
                             r['psnr'] - base))
    finally:
        bench.PATHS = old
    return rows


def print_rows(title, rows, n_ch):
    print(f"\n=== {title} ===")
    print(f"{'thick':>5} {'prior':>14} {'PSNR':>7} {'SSIM':>7} {'dPSNR':>7}  per-ch")
    for row in rows:
        th, name = row[0], row[1]
        psnr, ssim, pch = row[2], row[3], row[4]
        d = f"{row[5]:+.3f}" if len(row) > 5 else "   -  "
        pcs = " ".join(f"{p:.2f}" for p in pch)
        print(f"{th:>5} {name:>14} {psnr:7.3f} {ssim:7.4f} {d:>7}  [{pcs}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth_ckpt', required=True, help='SynthSeg PATCH-trained restoration prior')
    ap.add_argument('--synth_full_ckpt', default=None, help='SynthSeg FULL-IMAGE-trained restoration prior (optional 4th column)')
    ap.add_argument('--ixi_ckpt',
                    default='/workspace/data/unires_dl_prior/checkpoints/biasfree3d_restore_thick.pt')
    ap.add_argument('--thick', type=float, nargs='+', default=[2, 4, 6])
    ap.add_argument('--mu', type=float, default=0.02)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--max_iter', type=int, default=6)
    ap.add_argument('--seen', type=int, default=1)
    ap.add_argument('--novel', type=int, default=1)
    ap.add_argument('--novel_random', type=int, default=1, help='3-channel STRONG OOD random-contrast test (the discriminative contrast-agnosticism test)')
    ap.add_argument('--no_taper', action='store_true', help='disable red_thick_taper (hallucination stress-test)')
    args = ap.parse_args()

    ckpts = [('red(IXI)', args.ixi_ckpt), ('red(synth-patch)', args.synth_ckpt)]
    if args.synth_full_ckpt:
        ckpts.append(('red(synth-full)', args.synth_full_ckpt))
    taper = not args.no_taper
    print(f"mu={args.mu} sigma={args.sigma} max_iter={args.max_iter} taper={taper}")
    print(f"synth_ckpt={args.synth_ckpt}\nixi_ckpt={args.ixi_ckpt}")

    if args.seen:
        rows = eval_one(bench.PATHS, 3, args.thick, args.mu, args.sigma,
                        args.max_iter, ckpts, taper=taper)
        print_rows("SEEN contrasts (BrainWeb T1/T2/PD, per-ch = T1 T2 PD)", rows, 3)

    if args.novel:
        fl = make_flair_like('/workspace/data/synthseg_restore/novel/flair_like.nii.gz')
        rows = eval_one([fl], 1, args.thick, args.mu, args.sigma,
                        args.max_iter, ckpts, taper=taper)
        print_rows("NOVEL contrast (FLAIR-like, single-channel)", rows, 1)

    if args.novel_random:
        # THE discriminative test: 3 random-contrast channels on real anatomy (IXI never saw
        # these orderings; synth was trained on exactly this distribution). Multi-channel ->
        # same headroom as the seen test.
        nr = make_novel_random('/workspace/data/synthseg_restore/novel', n_ch=3, seed=0)
        rows = eval_one(nr, 3, args.thick, args.mu, args.sigma,
                        args.max_iter, ckpts, taper=taper)
        print_rows("NOVEL-RANDOM (3ch OOD random-contrast on real anatomy)", rows, 3)


if __name__ == '__main__':
    main()
