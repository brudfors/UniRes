"""Train the merged-gradient GAIN net for UniRes's Phase-2 DL prior.

The merged-gradient prior (docs/merged_gradient_prior.md) replaces the JTV z-update's
analytic soft-threshold gain on the channel-pooled gradient MAGNITUDE ``m`` with a
learned per-voxel gain -- ONE 3D network pass per ADMM iteration, C-independent. This
script trains that net.

Paradigm (reuses Track-1's detail-filling, moved to the GRADIENT domain):
  * A clean IXI patch y* is degraded with an anisotropic THICK-SLICE loss (+ blur +
    noise) via ``data.augment_noise_gpu(..., thick_max=...)`` (imported READ-ONLY).
  * Inputs/targets are the POOLED GRADIENT MAGNITUDES:
        m_deg   = ||grad(degraded)||   (what the ADMM z-update sees)
        m_sharp = ||grad(sharp y*)||   (the target the prior should recover)
  * The bias-free 3D U-Net predicts m_hat ~ m_sharp; at inference the gain is
    ``clamp(m_hat/(m+eps), 0, 1)`` (a non-expansive shrinkage -> monotone-safe; the
    net KEEPS coherent edges the analytic JTV gain would over-shrink, and shrinks
    noise-induced gradients).

C-agnostic: trained on single-contrast IXI magnitudes (same statistical family as the
multi-contrast pooled magnitude), transfers to any number of contrasts (spec).

Does NOT modify any training/ file; imports the degradation helpers read-only. Uses
one GPU (pass CUDA_VISIBLE_DEVICES=5|6|7). Checkpoints every --save_every steps and
auto-resumes from <out>/last.pt (the pod livelocks -> run under a watchdog burst loop).
"""
import argparse
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import IXIPatches, augment_noise_gpu, _thick_slice_degrade, _gauss_blur3d
from unires._denoiser_net import BiasFreeUNet3D

torch.set_num_threads(4)

BRAINWEB = [
    "data/t1_icbm_normal_1mm_pn0_rf0.nii.gz",
    "data/t2_icbm_normal_1mm_pn0_rf0.nii.gz",
    "data/pd_icbm_normal_1mm_pn0_rf0.nii.gz",
]


def _dirs(v):
    """Forward-difference gradients of v [B,1,D,H,W] along the 3 axes, zero-padded to
    keep shape -> list of 3 tensors [B,1,D,H,W]."""
    out = []
    for d in range(3):
        gd = torch.diff(v, dim=2 + d)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - d) + 1] = 1                 # pad the far end of this spatial dim
        out.append(F.pad(gd, pad))
    return out


def grad_mag(v, eps=1e-8):
    """K=1 pooled gradient magnitude (over the 3 directions) -> [B,1,D,H,W] (>= 0)."""
    return torch.sqrt(sum(gd ** 2 for gd in _dirs(v)) + eps)


def grad_field(v, perdir, eps=1e-8):
    """Merged-gradient field the z-update sees. perdir=False -> K=1 pooled magnitude
    [B,1,D,H,W]; perdir=True -> K=3 per-direction magnitudes [B,3,D,H,W] (|d/dx|,|d/dy|,
    |d/dz|). For a single-contrast patch (channel pooling trivial) the per-direction
    channel-pooled magnitude is just |forward-diff| along that axis -- the same
    statistical family as the multi-contrast pooled field, so it stays C-agnostic."""
    if not perdir:
        return grad_mag(v, eps)
    return torch.cat([gd.abs() for gd in _dirs(v)], dim=1)   # [B,3,D,H,W] (>= 0)


def _load_bw(path, dev):
    import nibabel as nib
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    v = torch.from_numpy(a).clamp_min(0.0).to(dev)
    pos = v[v > 0]
    scale = torch.quantile(pos.float(), 0.99).clamp_min(1e-6)
    return v / scale


@torch.no_grad()
def validate(net, dev, thick_max, blur_max, sigma_max, perdir=False, crop=128):
    """Cross-dataset (BrainWeb) magnitude-restoration PSNR: recover m_sharp from the
    thick-slice-degraded magnitude. Higher = the net restores the sharp gradient field."""
    net.eval()
    names, out = ['T1', 'T2', 'PD'], {}
    for name, p in zip(names, BRAINWEB):
        if not os.path.exists(p):
            continue
        vol = _load_bw(p, dev)
        c = [min(crop, s) for s in vol.shape]
        o = [(vol.shape[d] - c[d]) // 2 for d in range(3)]
        clean = vol[o[0]:o[0] + c[0], o[1]:o[1] + c[1], o[2]:o[2] + c[2]][None, None]
        g = torch.Generator(device=dev).manual_seed(1234)
        deg = _thick_slice_degrade(clean, thick_max * 0.5, 2)
        if blur_max > 0:
            deg = _gauss_blur3d(deg, blur_max * 0.5)
        deg = deg + (sigma_max * 0.5) * torch.randn(deg.shape, generator=g, device=dev)
        m_deg, m_sharp = grad_field(deg, perdir), grad_field(clean, perdir)
        scale = torch.quantile(m_deg[m_deg > 0].float(), 0.99).clamp_min(1e-8)
        m_hat = (net(m_deg / scale, sigma=0.0).clamp_min(0.0) * scale)
        mse = ((m_hat - m_sharp) ** 2).mean()
        rng = (m_sharp.max() - m_sharp.min()).clamp_min(1e-8)
        out[name] = float(10 * torch.log10(rng ** 2 / mse.clamp_min(1e-12)))
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/dev/shm/ixi')
    ap.add_argument('--out', default='/dev/shm/denoiser_runs/gradnet')
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--iters', type=int, default=12000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--sigma_max', type=float, default=0.05)
    ap.add_argument('--thick_max', type=float, default=8.0)
    ap.add_argument('--blur_max', type=float, default=1.0)
    ap.add_argument('--grad_apply', default='magnitude', choices=['magnitude', 'perdir'],
                    help="'magnitude' (K=1 pooled) or 'perdir' (K=3 per-direction merged field)")
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--save_every', type=int, default=250)
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()
    perdir = args.grad_apply == 'perdir'
    K = 3 if perdir else 1
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'

    ds = IXIPatches(args.data, patch=args.patch,
                    patches_per_epoch=args.batch * args.iters,
                    sigma_max=args.sigma_max, gpu_aug=True)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, shuffle=False,
                    pin_memory=False, drop_last=True)
    # Blind (no sigma channel): the gradient-magnitude "noise level" is not the image
    # sigma; a bias-free blind restorer over the degradation range is cleaner and needs
    # no sigma at inference.
    net = BiasFreeUNet3D(base=args.base, depth=args.depth, sigma_map=False,
                         residual=True, context=False, k=K).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    start_step, best = 0, -1e9
    resume_path = args.resume or os.path.join(args.out, 'last.pt')
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=dev)
        net.load_state_dict(ck['model'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start_step, best = int(ck.get('step', 0)), float(ck.get('best', -1e9))
        print(f"RESUMED from {resume_path} at step {start_step} (best={best:.2f})", flush=True)

    print(f"gradnet params={sum(p.numel() for p in net.parameters())/1e6:.1f}M "
          f"patch={args.patch} batch={args.batch} thick_max={args.thick_max} "
          f"blur_max={args.blur_max} start={start_step}", flush=True)

    def save(name):
        torch.save({'model': net.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'args': vars(args),
                    'step': step, 'best': best}, os.path.join(args.out, name))

    net.train()
    t0, run_loss, step = time.time(), 0.0, start_step
    for clean in dl:
        clean = clean.to(dev, non_blocking=True)
        # thick-slice-degrade + noise on GPU (target = SHARP clean); n2n off (supervised)
        deg, sharp, _ = augment_noise_gpu(
            clean, sigma_min=0.0, sigma_max=args.sigma_max, rician_prob=0.5, n2n=False,
            blur_max=args.blur_max, thick_max=args.thick_max)
        m_deg, m_sharp = grad_field(deg, perdir), grad_field(sharp, perdir)
        # per-sample standardisation by the degraded-field robust scale (shared across the
        # K channels so relative per-direction magnitudes -- the direction info -- are kept)
        scale = torch.quantile(
            m_deg.flatten(1).clamp_min(0).float(), 0.99, dim=1
        ).clamp_min(1e-8).view(-1, 1, 1, 1, 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m_hat = net(m_deg / scale, sigma=0.0)
            # foreground-weighted L1 in standardised units (focus on where there's signal)
            tgt = m_sharp / scale
            w = (tgt > 0.02).float() + 0.1
            loss = (w * (m_hat - tgt).abs()).sum() / w.sum().clamp_min(1.0)
        if not torch.isfinite(loss) or loss.item() > 10.0:
            opt.zero_grad(set_to_none=True); step += 1; continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        run_loss += loss.item(); step += 1
        if step % 25 == 0:
            avg = run_loss / 25; run_loss = 0.0
            ips = (step - start_step) / (time.time() - t0)
            print(f"step {step}/{args.iters}  l1={avg:.4f}  {ips:.1f} it/s  "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
        if step % args.save_every == 0:
            save('last.pt')
        if step % args.val_every == 0 or step == args.iters:
            v = validate(net, dev, args.thick_max, args.blur_max, args.sigma_max, perdir=perdir)
            msg = "  ".join(f"{k}={val:.2f}" for k, val in v.items())
            mean = sum(v.values()) / max(len(v), 1)
            print(f"[val step {step}] BrainWeb grad-mag restore PSNR  {msg}  mean={mean:.2f}",
                  flush=True)
            save('last.pt')
            if mean > best:
                best = mean; save('best.pt')
        if step >= args.iters:
            break
    print(f"done in {(time.time()-t0)/60:.1f} min  best val mean PSNR={best:.2f}", flush=True)


if __name__ == '__main__':
    main()
