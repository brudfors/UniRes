"""Train the CROSS-CHANNEL restoration prior (fills thick-slice detail using other
channels' edges). Aligned IXI triplets; per-target-channel restoration + context.
Hang-proof (num_workers=0, gpu aug), resume + frequent save for the watchdog.

  python training/train_cross.py --out /dev/shm/denoiser_runs/cross --iters 12000
"""
import argparse
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_cross import CrossPatches, augment_noise_cross, edge_map
from data import _thick_slice_degrade
from model import BiasFreeUNet3D

torch.set_num_threads(4)
BRAINWEB = ["data/t1_icbm_normal_1mm_pn0_rf0.nii.gz",
            "data/t2_icbm_normal_1mm_pn0_rf0.nii.gz",
            "data/pd_icbm_normal_1mm_pn0_rf0.nii.gz"]


def _load_bw(path, dev):
    import nibabel as nib
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    v = torch.from_numpy(a).clamp_min(0.0).to(dev)
    return v / torch.quantile(v[v > 0].float(), 0.99).clamp_min(1e-6)


@torch.no_grad()
def validate(net, dev, thick=3.0, sig=0.03):
    """BrainWeb (aligned): degrade each channel thick along a different axis, restore
    the target using the OTHER channels' edges as context. PSNR vs sharp GT."""
    net.eval()
    vols = [_load_bw(p, dev) for p in BRAINWEB if os.path.exists(p)]
    c = [min(128, s) for s in vols[0].shape]; o = [(vols[0].shape[d]-c[d])//2 for d in range(3)]
    clean = [v[o[0]:o[0]+c[0], o[1]:o[1]+c[1], o[2]:o[2]+c[2]] for v in vols]
    g = torch.Generator(device=dev).manual_seed(1)
    deg = [(_thick_slice_degrade(clean[k][None, None], thick, k % 3)[0, 0]
            + sig * torch.randn(clean[k].shape, generator=g, device=dev)) for k in range(len(clean))]
    em = [edge_map(deg[k][None, None])[0, 0] for k in range(len(deg))]
    out = {}
    names = ['T1', 'T2', 'PD']
    for k in range(len(clean)):
        ctx = torch.stack([em[d] for d in range(len(deg)) if d != k]).amax(0)
        den = net(deg[k][None, None], sigma=sig, context=ctx[None, None])[0, 0].clamp_min(0)
        mse = ((den - clean[k]) ** 2).mean(); rng = clean[k].max() - clean[k].min()
        out[names[k]] = float(10 * torch.log10(rng ** 2 / mse.clamp_min(1e-12)))
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/ixi_aligned')
    ap.add_argument('--out', default='/dev/shm/denoiser_runs/cross')
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--batch', type=int, default=6)   # x3 channels -> 18 effective
    ap.add_argument('--iters', type=int, default=12000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--thick_max', type=float, default=8.0)
    ap.add_argument('--ctx_drop', type=float, default=0.4, help='context-dropout prob (classifier-free style)')
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--save_every', type=int, default=250)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'
    ds = CrossPatches(args.data, patch=args.patch, patches_per_epoch=args.batch * args.iters)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=0, drop_last=True)
    net = BiasFreeUNet3D(base=args.base, depth=3, context=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)
    writer = SummaryWriter(args.out)
    start, best = 0, -1e9
    rp = os.path.join(args.out, 'last.pt')
    if os.path.exists(rp):
        ck = torch.load(rp, map_location=dev); net.load_state_dict(ck['model'])
        if 'opt' in ck: opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start, best = int(ck.get('step', 0)), float(ck.get('best', -1e9))
        print(f"RESUMED at {start} (best={best:.2f})", flush=True)
    print(f"cross-channel: triplets={len(ds.paths)} params={sum(p.numel() for p in net.parameters())/1e6:.1f}M "
          f"batch={args.batch}x3 start={start}", flush=True)

    def save(name):
        torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'args': vars(args), 'step': step, 'best': best}, os.path.join(args.out, name))

    net.train(); t0 = time.time(); rl = 0.0; step = start
    for clean in dl:
        clean = clean.to(dev, non_blocking=True)
        inp, tgt, sig, ctx = augment_noise_cross(clean, thick_max=args.thick_max)
        # CONTEXT DROPOUT (classifier-free style): zero the cross-channel context on a
        # fraction of samples so the net learns a strong SELF-channel prior (aggressive
        # filling) AND to exploit context when present -> best of both (gains + no
        # hallucination), instead of over-relying on context (too conservative).
        if args.ctx_drop > 0:
            m = (torch.rand(inp.shape[0], 1, 1, 1, 1, device=dev) < args.ctx_drop).float()
            ctx = ctx * (1.0 - m)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            den = net(inp, sigma=sig, context=ctx)
            loss = torch.nn.functional.l1_loss(den, tgt)
        if not torch.isfinite(loss) or loss.item() > 10:
            opt.zero_grad(set_to_none=True); step += 1; continue
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        rl += loss.item(); step += 1
        if step % 25 == 0:
            print(f"step {step}/{args.iters}  l1={rl/25:.4f}  {(step-start)/(time.time()-t0):.2f} it/s", flush=True)
            rl = 0.0
        if step % args.save_every == 0:
            save('last.pt')
        if step % args.val_every == 0 or step >= args.iters:
            v = validate(net, dev); m = sum(v.values())/len(v)
            print(f"[val {step}] cross-restore@3mm  " + "  ".join(f"{k}={x:.2f}" for k,x in v.items()) + f"  mean={m:.2f}", flush=True)
            save('last.pt')
            if m > best: best = m; save('best.pt')
        if step >= args.iters:
            break
    print(f"done best={best:.2f}", flush=True)


if __name__ == '__main__':
    main()
