"""Phase-2 quality benchmark: RED denoiser prior vs classic MTV on BrainWeb.

Ground truth = the clean 1 mm multichannel BrainWeb scans in data/. Low-res
observations are synthesised with UniRes's own forward model (per-channel
orthogonal thick-slice), the rigid disabled so the recon lands on the GT grid.
Reconstructions are scored (PSNR/SSIM/MSE vs GT), for prior='mtv' and
prior='mtv+red' (DRUNet) over a small mu sweep. Also reports wall-clock and
whether the reported objective was monotone.

Usage:
  python benchmarks/bench_dl_prior.py [--mu 0 1e-3 1e-2 1e-1] [--sigma 0.03]
                                      [--max_iter 6] [--channels 3]
"""
import argparse
import contextlib
import io as _io
import re
import time

import torch
from nitorch import io
from nitorch.spatial import affine_grid, grid_pull
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from unires._project import _proj_info, _proj_apply
from unires.struct import settings
from unires.run import preproc

DEV = 'cuda'
PATHS = [
    "data/t1_icbm_normal_1mm_pn0_rf0.nii.gz",
    "data/t2_icbm_normal_1mm_pn0_rf0.nii.gz",
    "data/pd_icbm_normal_1mm_pn0_rf0.nii.gz",
]


def simulate(n_ch, vx_ts=4, std_noise=50.0, seed=0):
    torch.manual_seed(seed)
    nii = [io.map(p) for p in PATHS[:n_ch]]
    gt = [n.fdata().to(DEV) for n in nii]
    gt_mat = [n.affine.to(DEV) for n in nii]
    x, mat_x = [], []
    for i in range(n_ch):
        vx = torch.ones(3); vx[i % 3] = vx_ts
        my = nii[i].affine
        dy = torch.as_tensor(gt[i].shape, dtype=my.dtype)
        ms = torch.diag(torch.cat((vx, torch.ones(1))).to(my.dtype))
        mat_x.append((my.mm(ms)).to(DEV))
        dx = ms[:3, :3].inverse().mm(dy[:, None]).floor().squeeze()
        po = _proj_info(dy.to(DEV), my.to(DEV), dx.to(DEV), mat_x[i],
                        prof_ip=0, prof_tp=0, gap=0.0, device=DEV, scl=0.0)  # rigid = identity
        A = lambda v: _proj_apply("A", v[None, None, ...], po)[0, 0, ...]
        n = std_noise * torch.randn(dx.cpu().int().tolist(), device=DEV)
        x.append(A(gt[i]) + n)
    return gt, gt_mat, x, mat_x


def resample_gt(gt, gt_mat, out_dim, out_mat):
    """Resample GT onto the reconstruction grid (recon voxel -> GT voxel)."""
    M = torch.linalg.solve(gt_mat.double(), out_mat.double()).to(gt.dtype)
    grid = affine_grid(M, list(out_dim))[None]
    return grid_pull(gt[None, None], grid, bound='zero', extrapolate=False,
                     interpolation='linear')[0, 0]


def score(recon, gt_r):
    """LS-match intensity over the GT foreground, then PSNR/SSIM/MSE."""
    mask = gt_r > 0.02 * gt_r.max()
    a, b = recon[mask], gt_r[mask]
    s = (a * b).sum() / (a * a).sum().clamp_min(1e-12)
    rec = recon * s
    mse = ((rec[mask] - gt_r[mask]) ** 2).mean()
    rng = (gt_r[mask].max() - gt_r[mask].min()).clamp_min(1e-8)
    psnr = 10.0 * torch.log10(rng ** 2 / mse)
    ssim = ssim_fn(rec[None, None], gt_r[None, None], data_range=float(rng))
    return psnr.item(), float(ssim), mse.item()


def run(gt, gt_mat, x, mat_x, prior, max_iter, **kw):
    data = [[[x[i], mat_x[i]]] for i in range(len(x))]
    s = settings()
    s.device = DEV; s.write_out = False
    s.do_coreg = False; s.unified_rigid = False; s.scaling = False
    s.max_iter = max_iter; s.prior = prior
    # Sweep defaults tuned for speed: no line search (monotonicity verified separately),
    # large denoiser slice-batch. Callers may override via kw.
    s.red_linesearch = False
    s.red_batch = 16
    for k, v in kw.items():
        setattr(s, k, v)
    t0 = time.time()
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        y_hat, y_mat, _ = preproc(data, s)
    dt = time.time() - t0
    nlyx = [float(m) for m in re.findall(r"nlyx =\s*([-\d.eE+]+)", buf.getvalue())]
    mono = all(nlyx[i + 1] <= nlyx[i] * (1 + 1e-6) + 1e-6 for i in range(len(nlyx) - 1))
    # score every channel vs GT resampled to the recon grid
    out_dim = y_hat.shape[:3]
    ps, ss, ms = [], [], []
    for i in range(len(x)):
        gt_r = resample_gt(gt[i], gt_mat[i], out_dim, y_mat.to(DEV))
        p, s_, m = score(y_hat[..., i], gt_r)
        ps.append(p); ss.append(s_); ms.append(m)
    return dict(dt=dt, mono=mono, psnr=sum(ps) / len(ps), ssim=sum(ss) / len(ss),
                mse=sum(ms) / len(ms), psnr_ch=ps, ssim_ch=ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', type=float, nargs='+', default=[0.0, 1e-3, 1e-2, 1e-1])
    ap.add_argument('--sigma', type=float, default=0.03)
    ap.add_argument('--max_iter', type=int, default=6)
    ap.add_argument('--channels', type=int, default=3)
    ap.add_argument('--denoiser', type=str, default='drunet')
    ap.add_argument('--cgs', type=int, default=None, help='override cgs_max_iter (inner CG)')
    args = ap.parse_args()
    # tolerance=0 skips the per-iteration objective eval (and its extra denoise), so the
    # 'mono' column is not measured here (monotonicity is verified separately in the smoke test).
    extra = {'tolerance': 0.0}
    if args.cgs is not None:
        extra['cgs_max_iter'] = args.cgs

    gt, gt_mat, x, mat_x = simulate(args.channels)
    print(f"channels={args.channels} sigma={args.sigma} max_iter={args.max_iter} "
          f"denoiser={args.denoiser}\n")
    print(f"{'config':>16} | {'PSNR':>7} {'SSIM':>7} {'MSE':>10} | {'mono':>5} {'time':>6}")
    print("-" * 62)
    base = None
    for mu in args.mu:
        if mu == 0:
            r = run(gt, gt_mat, x, mat_x, 'mtv', args.max_iter, **extra)
            name = 'mtv'
        else:
            r = run(gt, gt_mat, x, mat_x, 'mtv+red', args.max_iter,
                    red_mu=mu, red_sigma=args.sigma, red_denoiser=args.denoiser, **extra)
            name = f'mtv+red mu={mu:g}'
        print(f"  [done] {name}: PSNR={r['psnr']:.3f} SSIM={r['ssim']:.4f} "
              f"time={r['dt']:.1f}s", flush=True)
        if base is None:
            base = r['psnr']
        dpsnr = r['psnr'] - base
        print(f"{name:>16} | {r['psnr']:7.3f} {r['ssim']:7.4f} {r['mse']:10.4g} | "
              f"{str(r['mono']):>5} {r['dt']:5.1f}s   dPSNR={dpsnr:+.3f}")
    print("\n(dPSNR is vs the mtv baseline; positive => the DL prior improved quality)")


if __name__ == '__main__':
    main()
