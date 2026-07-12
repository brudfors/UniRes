"""Phase 3 head-to-head: MTV vs image-domain RED (Track-1 restoration) vs merged-
gradient prior (Track-2), on the SAME thick-slice BrainWeb sim, across contrast count
C. Measures QUALITY (per-channel + avg PSNR), RUNTIME (wall-clock, prior net calls/iter),
and PEAK GPU MEMORY, and how they scale with C -- the merged-gradient prior runs its net
ONCE/iter regardless of C, the image-RED denoiser runs C times/iter.

  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_headtohead.py --channels 3 --vx_ts 3
"""
import argparse
import contextlib
import io as _io
import re
import time

import torch
from nitorch import io
from unires._project import _proj_info, _proj_apply
from unires.struct import settings
from unires.run import preproc
from bench_dl_prior import PATHS, resample_gt, score, DEV

RED_W = '/workspace/data/unires_dl_prior/checkpoints/biasfree3d_restore_thick.pt'
GRAD_W = '/workspace/data/unires_dl_prior/checkpoints/gradnet_restore.pt'


def simulate_c(C, vx_ts=3.0, std_noise=50.0, seed=0):
    """Thick-slice BrainWeb sim with C channels (cycles the 3 contrasts for C>3)."""
    torch.manual_seed(seed)
    paths = [PATHS[i % len(PATHS)] for i in range(C)]
    nii = [io.map(p) for p in paths]
    gt = [n.fdata().to(DEV) for n in nii]
    gt_mat = [n.affine.to(DEV) for n in nii]
    x, mat_x = [], []
    for i in range(C):
        vx = torch.ones(3); vx[i % 3] = vx_ts
        my = nii[i].affine
        dy = torch.as_tensor(gt[i].shape, dtype=my.dtype)
        ms = torch.diag(torch.cat((vx, torch.ones(1))).to(my.dtype))
        mat_x.append((my.mm(ms)).to(DEV))
        dx = ms[:3, :3].inverse().mm(dy[:, None]).floor().squeeze()
        po = _proj_info(dy.to(DEV), my.to(DEV), dx.to(DEV), mat_x[i], prof_ip=0, prof_tp=0, gap=0.0, device=DEV, scl=0.0)
        A = lambda v: _proj_apply("A", v[None, None, ...], po)[0, 0, ...]
        clean = A(gt[i])
        x.append(clean + std_noise * torch.randn(clean.shape, device=DEV))
    return gt, gt_mat, x, mat_x


def run_method(gt, gt_mat, x, mat_x, tag, prior, max_iter=30, **kw):
    data = [[[x[i], mat_x[i]]] for i in range(len(x))]
    s = settings(); s.device = DEV; s.write_out = False; s.do_coreg = False
    s.unified_rigid = False; s.scaling = False; s.max_iter = max_iter; s.prior = prior
    s.tolerance = 0.0
    for k, v in kw.items():
        setattr(s, k, v)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with contextlib.redirect_stdout(_io.StringIO()):
        y, ym, _ = preproc(data, s)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    ps = []
    for i in range(len(x)):
        gr = resample_gt(gt[i], gt_mat[i], y.shape[:3], ym.to(DEV))
        ps.append(score(y[..., i], gr)[0])
    avg = sum(ps) / len(ps)
    print(f"{tag:>22} | avgPSNR {avg:6.3f} | per-ch {[round(p,2) for p in ps]} | "
          f"{dt:5.1f}s | peak {peak:4.1f}GB", flush=True)
    return dict(psnr=avg, dt=dt, peak=peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channels', type=int, default=3)
    ap.add_argument('--vx_ts', type=float, default=3.0)
    ap.add_argument('--max_iter', type=int, default=30)
    args = ap.parse_args()
    gt, gm, x, mx = simulate_c(args.channels, vx_ts=args.vx_ts)
    print(f"\n=== HEAD-TO-HEAD  C={args.channels}  vx_ts={args.vx_ts}mm  max_iter={args.max_iter} ===", flush=True)
    print(f"{'method':>22} | quality           | per-channel | runtime | memory  "
          f"(net calls/iter: image-RED=C={args.channels}, merged-grad=1)", flush=True)
    run_method(gt, gm, x, mx, 'MTV', 'mtv', args.max_iter)
    run_method(gt, gm, x, mx, 'image-RED (restore)', 'mtv+red', args.max_iter,
               red_denoiser='biasfree3d', red_weights=RED_W, red_mu=0.01, red_sigma=0.05,
               red_linesearch=True)
    run_method(gt, gm, x, mx, 'merged-gradient', 'mtv+gradnet', args.max_iter,
               grad_net='unet', grad_weights=GRAD_W, grad_linesearch=True)


if __name__ == '__main__':
    main()
