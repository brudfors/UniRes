"""Cache RIGIDLY-ALIGNED multi-contrast IXI triplets for cross-channel prior training.

For each subject with T1+T2+PD, rigidly co-register (nitorch affine_align, NMI) and
resample all onto the T1 grid -> an aligned [3, D, H, W] float16 tensor (channels
T1,T2,PD, each per-volume robust-standardised). Saved to <out>/<subject>.pt.

These aligned triplets let the cross-channel restoration prior learn to fill in detail
in one (thick-sliced) channel using the resolved edges of the others.

  python scripts/cache_aligned_ixi.py --n 150 --out /workspace/data/ixi_aligned
"""
import argparse
import glob
import os

import torch
from nitorch import io
from nitorch.tools.preproc import affine_align
from nitorch.spatial import affine_grid, grid_pull

DEV = 'cuda'


def _robust(v):
    pos = v[v > 0]
    if pos.numel() == 0:
        return v
    return v / torch.quantile(pos.float()[:200000], 0.99).clamp_min(1e-6)


def resample(vol, mat, out_shape, out_mat):
    M = torch.linalg.solve(mat.double(), out_mat.double()).to(vol.dtype)
    grid = affine_grid(M, list(out_shape))[None]
    return grid_pull(vol[None, None], grid, bound='zero', extrapolate=False,
                     interpolation='linear')[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/dev/shm/ixi')
    ap.add_argument('--out', default='/workspace/data/ixi_aligned')
    ap.add_argument('--n', type=int, default=150)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    done = 0
    for t1p in sorted(glob.glob(f"{args.root}/T1/*.nii.gz")):
        sid = os.path.basename(t1p).split('-')[0]
        t2p = glob.glob(f"{args.root}/T2/{sid}-*.nii.gz")
        pdp = glob.glob(f"{args.root}/PD/{sid}-*.nii.gz")
        outp = f"{args.out}/{sid}.pt"
        if not (t2p and pdp) or os.path.exists(outp):
            continue
        try:
            paths = [t1p, t2p[0], pdp[0]]
            vols, mats = [], []
            for p in paths:
                n = io.map(p)
                vols.append(n.fdata().to(DEV).clamp_min(0.0).float())
                mats.append(n.affine.to(DEV).double())
            imgs = [[vols[c], mats[c]] for c in range(3)]
            mat_a = affine_align(imgs, cost_fun='nmi', group='SE', samp=(3, 1.5),
                                 fix=0, device=DEV)[1]
            cmats = [torch.linalg.solve(mat_a[c].double(), mats[c]) for c in range(3)]
            t1, t1_mat = vols[0], cmats[0]
            aligned = [_robust(t1)]
            for c in range(1, 3):
                aligned.append(_robust(resample(vols[c], cmats[c], t1.shape, t1_mat)))
            stack = torch.stack(aligned, 0).half().cpu()  # [3,D,H,W]
            torch.save(stack, outp)
            done += 1
            print(f"[{done}] {sid} -> {tuple(stack.shape)}", flush=True)
        except Exception as e:
            print(f"SKIP {sid}: {e}", flush=True)
        if done >= args.n:
            break
    print(f"cached {done} aligned triplets to {args.out}", flush=True)


if __name__ == '__main__':
    main()
