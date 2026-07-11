"""Train the bias-free 3D MRI denoiser on IXI, validate on BrainWeb (Phase 3).

  * TRAIN on IXI patches (pooled contrasts + augmentation)  -> generalization.
  * VALIDATE on BrainWeb data/*.nii.gz (different dataset, no leakage): denoise-only
    PSNR vs the clean GT at a few sigmas, and a held-out-contrast check.
  * Loss: L1(denoised, target) with target = 2nd noise draw (Noise2Noise) or clean.
  * AdamW + cosine, bf16 autocast on A100, checkpointing, tensorboard.

Single-GPU by default; pass --ddp for torchrun multi-GPU (8xA100). The saved
checkpoint plugs into unires/_denoiser.py as the 'biasfree3d' backend.
"""
import argparse
import os
import time

# Limit intra-op threads BEFORE importing torch: the dataset runs torch ops
# (quantile/interpolate) inside every dataloader worker; with many workers, torch's
# default thread pool oversubscribes CPUs and futex-livelocks the whole loader.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import IXIPatches
from model import BiasFreeUNet3D

torch.set_num_threads(4)  # main process (workers pinned to 1 in _worker_init)


def _worker_init(_):
    torch.set_num_threads(1)  # each worker single-threaded -> no CPU oversubscription

BRAINWEB = [
    "data/t1_icbm_normal_1mm_pn0_rf0.nii.gz",
    "data/t2_icbm_normal_1mm_pn0_rf0.nii.gz",
    "data/pd_icbm_normal_1mm_pn0_rf0.nii.gz",
]


def _load_bw(path, dev):
    import nibabel as nib
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    v = torch.from_numpy(a).clamp_min(0.0).to(dev)
    pos = v[v > 0]
    scale = torch.quantile(pos.float(), 0.99).clamp_min(1e-6)
    return v / scale


@torch.no_grad()
def validate(net, dev, sigmas=(0.03, 0.06, 0.1), crop=128):
    """Denoise-only PSNR on BrainWeb (train->test cross-dataset). Returns dict."""
    net.eval()
    names = ['T1', 'T2', 'PD']
    out = {}
    for name, p in zip(names, BRAINWEB):
        if not os.path.exists(p):
            continue
        vol = _load_bw(p, dev)
        c = [min(crop, s) for s in vol.shape]
        o = [(vol.shape[d] - c[d]) // 2 for d in range(3)]
        clean = vol[o[0]:o[0]+c[0], o[1]:o[1]+c[1], o[2]:o[2]+c[2]]
        g = torch.Generator(device=dev).manual_seed(1234)
        ps = []
        for s in sigmas:
            noisy = clean + s * torch.randn(clean.shape, generator=g, device=dev)
            den = net(noisy[None, None], sigma=s)[0, 0].clamp_min(0.0)
            mse = ((den - clean) ** 2).mean()
            rng = clean.max() - clean.min()
            ps.append(float(10 * torch.log10(rng ** 2 / mse.clamp_min(1e-12))))
        out[name] = sum(ps) / len(ps)
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/ixi')
    ap.add_argument('--out', default='/workspace/data/denoiser_runs/biasfree3d')
    ap.add_argument('--patch', type=int, default=96)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--iters', type=int, default=60000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--sigma_max', type=float, default=0.15)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n2n', type=int, default=1)
    ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--holdout', default=None, help="contrast to EXCLUDE from training (e.g. PD) to test cross-contrast generalization")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'

    contrasts = tuple(c for c in ('T1', 'T2', 'PD') if c != args.holdout)
    ds = IXIPatches(args.data, patch=args.patch, patches_per_epoch=args.batch * args.iters,
                    contrasts=contrasts, sigma_max=args.sigma_max, n2n=bool(args.n2n))
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, shuffle=False,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                    worker_init_fn=_worker_init, prefetch_factor=4 if args.workers else None)
    net = BiasFreeUNet3D(base=args.base, depth=args.depth).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)
    writer = SummaryWriter(args.out)
    print(f"train contrasts={contrasts} holdout={args.holdout} params="
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M batch={args.batch} "
          f"iters={args.iters}", flush=True)

    net.train()
    t0 = time.time()
    run_loss = 0.0
    best = -1e9
    step = 0
    for noisy, target, smap in dl:
        noisy = noisy.to(dev, non_blocking=True)
        target = target.to(dev, non_blocking=True)
        s = smap.to(dev, non_blocking=True)[:, :, :1, :1, :1].reshape(-1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            den = net(noisy, sigma=s)
            loss = torch.nn.functional.l1_loss(den, target)
        # skip non-finite / exploded batches (robustness: a rare pathological
        # augmented patch once blew the loss from 0.06 to 1e6 and wrecked training)
        if not torch.isfinite(loss) or loss.item() > 10.0:
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # stabilize
        opt.step(); sched.step()
        run_loss += loss.item(); step += 1
        if step % 100 == 0:
            avg = run_loss / 100; run_loss = 0.0
            ips = step / (time.time() - t0)
            writer.add_scalar('train/l1', avg, step)
            print(f"step {step}/{args.iters}  l1={avg:.4f}  {ips:.1f} it/s  "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
        if step % args.val_every == 0 or step == args.iters:
            v = validate(net, dev)
            msg = "  ".join(f"{k}={val:.2f}" for k, val in v.items())
            mean = sum(v.values()) / max(len(v), 1)
            for k, val in v.items():
                writer.add_scalar(f'val_psnr/{k}', val, step)
            writer.add_scalar('val_psnr/mean', mean, step)
            print(f"[val step {step}] BrainWeb denoise PSNR  {msg}  mean={mean:.2f}", flush=True)
            torch.save({'model': net.state_dict(), 'args': vars(args), 'step': step},
                       os.path.join(args.out, 'last.pt'))
            if mean > best:
                best = mean
                torch.save({'model': net.state_dict(), 'args': vars(args), 'step': step},
                           os.path.join(args.out, 'best.pt'))
        if step >= args.iters:
            break
    print(f"done in {(time.time()-t0)/60:.1f} min  best val mean PSNR={best:.2f}", flush=True)


if __name__ == '__main__':
    main()
