"""Train a SynthSeg-style CONTRAST-AGNOSTIC restoration prior.

Same network + degradation as the IXI-trained restoration prior (unires BiasFreeUNet3D,
base=48; training/data.py::augment_noise_gpu thick-slice U(1,8) mm + noise), but the CLEAN
source is SYNTHESIZED from anatomical label maps with fully randomised per-label contrast
(training/synthseg/synth_data.py) instead of real IXI patches. Goal: generalise to UNSEEN
MRI contrasts. Checkpoint is drop-in compatible with unires bench (red_denoiser='biasfree3d').

Single-GPU:
  PYTHONPATH=/workspace/code/UniRes python training/synthseg/train_synth.py --out <dir>
Multi-GPU (DDP):
  CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node=4 training/synthseg/train_synth.py --out <dir>
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# import the existing (read-only) degradation + net
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)                          # synth_data
sys.path.insert(0, _TRAIN)                         # training/data.py
sys.path.insert(0, os.path.join(_TRAIN, ".."))     # repo root -> unires
from data import augment_noise_gpu, _thick_slice_degrade  # noqa: E402
from unires._denoiser_net import BiasFreeUNet3D  # noqa: E402
from synth_data import SynthLabelBank  # noqa: E402


def grad_l1(a, b):
    """L1 on first differences along all 3 axes (edge-preserving term)."""
    loss = 0.0
    for ax in (2, 3, 4):
        da = a.narrow(ax, 1, a.shape[ax] - 1) - a.narrow(ax, 0, a.shape[ax] - 1)
        db = b.narrow(ax, 1, b.shape[ax] - 1) - b.narrow(ax, 0, b.shape[ax] - 1)
        loss = loss + F.l1_loss(da, db)
    return loss / 3.0


@torch.no_grad()
def validate(net, bank, dev, res_cond, sigmas=(0.02, 0.05), thick=(2.0, 4.0, 6.0)):
    """Quick self-consistency PSNR on held-out synth patches across slice thickness."""
    net_eval = net.module if isinstance(net, DDP) else net
    net_eval.eval()
    out = {}
    for th in thick:
        clean = bank.sample(4)
        deg = torch.empty_like(clean)
        rv = torch.zeros(clean.shape[0], device=dev)
        for i in range(clean.shape[0]):
            fac = th
            rv[i] = (fac - 1.0) / 7.0
            deg[i:i + 1] = _thick_slice_degrade(clean[i:i + 1], fac, 2)
        s = clean.new_full((clean.shape[0],), 0.03)
        noisy = deg + s.view(-1, 1, 1, 1, 1) * torch.randn_like(deg)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            den = net_eval(noisy, sigma=s, res=(rv if res_cond else None)).float().clamp_min(0)
        mse = F.mse_loss(den, clean).item()
        out[f"th{int(th)}"] = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-8))).item()
    net_eval.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/workspace/data/synthseg_restore/label_maps")
    ap.add_argument("--out", default="/workspace/data/synthseg_restore/runs/synth_thick")
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8, help="per-GPU batch")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sigma_max", type=float, default=0.05)
    ap.add_argument("--thick_max", type=float, default=8.0)
    ap.add_argument("--blur_max", type=float, default=1.0)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--res_cond", type=int, default=0)
    ap.add_argument("--grad_w", type=float, default=0.5, help="weight of gradient(L1) term")
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    # ---- DDP setup ----
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
        world = dist.get_world_size()
        rank = dist.get_rank()
    else:
        dev = torch.device("cuda")
        world, rank, local_rank = 1, 0, 0
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(1234 + rank)
    bank = SynthLabelBank(args.labels, dev, patch=args.patch)
    net = BiasFreeUNet3D(base=args.base, depth=args.depth, res_map=bool(args.res_cond)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr * (world ** 0.5), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    start_step, best = 0, -1e9
    resume = os.path.join(args.out, "last.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=dev)
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_step = int(ck.get("step", 0)); best = float(ck.get("best", -1e9))
        if is_main:
            print(f"RESUMED from {resume} at step {start_step} (best={best:.2f})", flush=True)

    if ddp:
        net = DDP(net, device_ids=[local_rank], find_unused_parameters=False)

    if is_main:
        n = sum(p.numel() for p in (net.module if ddp else net).parameters()) / 1e6
        print(f"SYNTH restore | params={n:.1f}M world={world} batch/gpu={args.batch} "
              f"thick_max={args.thick_max} res_cond={args.res_cond} iters={args.iters} "
              f"start={start_step}", flush=True)

    def save(name, step):
        if not is_main:
            return
        sd = (net.module if ddp else net).state_dict()
        torch.save({"model": sd, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "args": {**vars(args), "res_cond": args.res_cond}, "step": step,
                    "best": best}, os.path.join(args.out, name))

    (net.module if ddp else net).train()
    t0 = time.time()
    run = 0.0
    step = start_step
    while step < args.iters:
        clean = bank.sample(args.batch)  # [B,1,P,P,P] sharp random-contrast (target source)
        noisy, target, s, res = augment_noise_gpu(
            clean, sigma_min=0.0, sigma_max=args.sigma_max, rician_prob=0.5,
            n2n=False, blur_max=args.blur_max, thick_max=args.thick_max)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            den = net(noisy, sigma=s, res=res if args.res_cond else None)
            loss = F.l1_loss(den, target) + args.grad_w * grad_l1(den, target)
        if not torch.isfinite(loss) or loss.item() > 10.0:
            opt.zero_grad(set_to_none=True); step += 1; continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        run += loss.item(); step += 1

        if is_main and step % args.log_every == 0:
            ips = (step - start_step) / (time.time() - t0) * world
            print(f"step {step}/{args.iters}  loss={run/args.log_every:.4f}  "
                  f"{ips:.1f} it/s  lr={sched.get_last_lr()[0]:.2e}", flush=True)
            run = 0.0
        if step % args.save_every == 0:
            save("last.pt", step)
        if step % args.val_every == 0 or step == args.iters:
            if is_main:
                v = validate(net, bank, dev, bool(args.res_cond))
                msg = "  ".join(f"{k}={val:.2f}" for k, val in v.items())
                mean = sum(v.values()) / len(v)
                print(f"[val step {step}] synth PSNR  {msg}  mean={mean:.2f}", flush=True)
                save("last.pt", step)
                if mean > best:
                    best = mean; save("best.pt", step)
            if ddp:
                dist.barrier()

    save("last.pt", step)
    if is_main:
        print(f"done in {(time.time()-t0)/60:.1f} min  best synth PSNR={best:.2f}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
