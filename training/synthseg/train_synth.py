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
from data import augment_noise_gpu, _thick_slice_degrade, _gauss_blur3d  # noqa: E402,F401
from unires._denoiser_net import BiasFreeUNet3D  # noqa: E402
from synth_data import SynthLabelBank, batched_degrade  # noqa: E402


def grad_l1(a, b):
    """L1 on first differences along all 3 axes (edge-preserving term)."""
    loss = 0.0
    for ax in (2, 3, 4):
        da = a.narrow(ax, 1, a.shape[ax] - 1) - a.narrow(ax, 0, a.shape[ax] - 1)
        db = b.narrow(ax, 1, b.shape[ax] - 1) - b.narrow(ax, 0, b.shape[ax] - 1)
        loss = loss + F.l1_loss(da, db)
    return loss / 3.0


def bf3d_forward_ckpt(net, x, sigma=None, res=None):
    """Gradient-CHECKPOINTED forward that REPLICATES BiasFreeUNet3D.forward using the net's
    own submodules (does NOT modify the core class). Wraps the enc/mid/dec ResBlock stacks
    in torch.utils.checkpoint so activations are recomputed in backward -> lets the 34M net
    train on a WHOLE ~192-256^3 volume that would otherwise OOM. Used only in --full_image
    --grad_ckpt mode; the checkpoint stays byte-identical in structure to the patch model."""
    from torch.utils.checkpoint import checkpoint
    inp = x
    feats = [x]
    ref = x[:, :1] if net.k > 1 else x
    if net.sigma_map:
        if sigma is None:
            sigma = x.new_zeros(())
        if not torch.is_tensor(sigma):
            sigma = x.new_tensor(float(sigma))
        feats.append(sigma.expand_as(ref) if sigma.dim() == 0
                     else sigma.view(-1, 1, 1, 1, 1).expand_as(ref))
    if net.context:
        feats.append(torch.zeros_like(x))
    if net.res_map:
        if res is None:
            res = x.new_zeros(())
        if not torch.is_tensor(res):
            res = x.new_tensor(float(res))
        feats.append(res.expand_as(ref) if res.dim() == 0
                     else res.view(-1, 1, 1, 1, 1).expand_as(ref))
    h = torch.cat(feats, dim=1) if len(feats) > 1 else x
    h = net.head(h)
    skips = []
    for enc, down in zip(net.enc, net.down):
        h = checkpoint(enc, h, use_reentrant=False)
        skips.append(h)
        h = down(h)
    h = checkpoint(net.mid, h, use_reentrant=False)
    for up, reduce, dec, skip in zip(net.up, net.reduce, net.dec, reversed(skips)):
        h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
        h = up(h)
        h = reduce(torch.cat([h, skip], dim=1))
        h = checkpoint(dec, h, use_reentrant=False)
    out = net.tail(h)
    return inp - out if net.residual else out


@torch.no_grad()
def validate(net, bank, dev, res_cond, sigmas=(0.02, 0.05), thick=(2.0, 4.0, 6.0),
             full=False, full_size=None, res_denom=7.0):
    """Quick self-consistency super-resolution PSNR on FRESH (held-out) synth samples across
    slice thickness -- confirms the model is learning to fill in thick-slice detail. full=True
    validates on WHOLE-volume samples (matches the full-image model's training distribution);
    else on 64^3 patches (patch model)."""
    net_eval = net.module if isinstance(net, DDP) else net
    net_eval.eval()
    out = {}
    for th in thick:
        clean = bank.sample_full(full_size) if full else bank.sample(4)
        deg = torch.empty_like(clean)
        rv = torch.zeros(clean.shape[0], 3, device=dev)   # directional per-axis res; axis 2 thick
        for i in range(clean.shape[0]):
            fac = th
            rv[i, 2] = (fac - 1.0) / res_denom
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
    ap.add_argument("--fixed_lr", type=int, default=0,
                    help="SynthSeg-faithful: Adam (wd=0) with a FIXED LR (no scheduler). "
                         "SynthSeg needs a sustained LR over many steps; cosine-annealing to 0 "
                         "starves it. Reference uses lr=sqrt(batch*gpus)*1e-4, no schedule.")
    ap.add_argument("--sigma_max", type=float, default=0.05)
    ap.add_argument("--thick_max", type=float, default=8.0)
    ap.add_argument("--blur_max", type=float, default=1.0)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--res_cond", type=int, default=0)
    ap.add_argument("--inplane_max", type=float, default=0.0,
                    help="max isotropic blur sigma applied to the TARGET (in-plane resolution "
                         "variation, e.g. 0.5x0.5..1x1mm). 0 = fixed iso target (original).")






    ap.add_argument("--realism", type=float, default=0.0,
                    help="coherent MULTI-SCALE intra-tissue structure amplitude (coarse gradients->fine "
                         "texture) for MRI-realistic appearance -> reconstruct recoverable real-T1 detail.")
    ap.add_argument("--net", type=str, default="biasfree3d", choices=["biasfree3d", "basicunet"],
                    help="basicunet = reference SynthSeg net (MONAI BasicUNet 24..384, InstanceNorm, LeakyReLU)")
    ap.add_argument("--grad_w", type=float, default=0.5, help="weight of gradient(L1) term")
    ap.add_argument("--full_image", type=int, default=0,
                    help="1 => train on WHOLE volumes (batch forced 1) instead of patches")
    ap.add_argument("--full_size", type=int, default=192,
                    help="cube side for full-image mode (whole brain, lightly downsampled to fit)")
    ap.add_argument("--grad_ckpt", type=int, default=0,
                    help="gradient checkpointing on enc/mid/dec blocks (memory<->compute; for full-image)")
    ap.add_argument("--zero_bg", type=int, default=0,
                    help="synth background: 0=textured (full-head, e.g. IXI), 1=zeroed (skull-stripped, "
                         "e.g. OASIS/MICCAI), 2=RANDOMISED per sample (robust to BOTH input types)")
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

    full = bool(args.full_image)
    if full:
        args.batch = 1                      # whole-volume input -> batch 1 (memory)
        if args.lr == 2e-4:                 # batch=1 diverges at the patch LR (0.05->7.22 seen)
            args.lr = 1e-4                  # -> halve the default LR for full-image stability
    torch.manual_seed(1234 + rank)
    bank = SynthLabelBank(args.labels, dev, patch=args.patch, zero_bg=int(args.zero_bg),
                          realism=args.realism)

    if args.net == "basicunet":
        from unires._denoiser_net import BasicUNetRestore  # reference SynthSeg net
        net = BasicUNetRestore(in_ch=1, out_ch=1, res_cond=bool(args.res_cond)).to(dev)
    else:
        net = BiasFreeUNet3D(base=args.base, depth=args.depth, res_map=bool(args.res_cond)).to(dev)
    if args.fixed_lr:
        # SynthSeg-faithful: Adam (weight_decay=0) + FIXED lr, NO scheduler.
        opt = torch.optim.Adam(net.parameters(), lr=args.lr * (world ** 0.5))
        sched = None
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr * (world ** 0.5), weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    start_step, best = 0, -1e9
    resume = os.path.join(args.out, "last.pt")
    if os.path.exists(resume):
        # Load to CPU first (NOT map_location=cuda): loading the model + 272MB optimizer
        # state as one direct-to-GPU blob under expandable_segments intermittently wedges
        # the first training kernel on resume. Staging on CPU + syncing avoids it.
        ck = torch.load(resume, map_location="cpu")
        net.load_state_dict(ck["model"])
        # MODEL-ONLY resume (RESUME_OPT=0): skip loading the 272MB AdamW optimizer state.
        # On this pod, loading the full opt-state blob wedges the first CUDA kernel on EVERY
        # resume (<25 steps), which makes watchdog-ratcheting impossible. Skipping it lets a
        # resume actually step; AdamW re-warms its moments in a few dozen steps (negligible
        # quality cost). sched still restored so the LR schedule is continuous.
        if "opt" in ck and os.environ.get("RESUME_OPT", "1") == "1":
            opt.load_state_dict(ck["opt"])
        # RESUME_SCHED=0 => keep the FRESH cosine (T_max=--iters) instead of the loaded one.
        # Used when EXTENDING training (e.g. 40k->100k): a fresh cosine over the new horizon
        # gives a sensible (non-zero) LR for the extra steps instead of the old LR~0 at 40k.
        if sched is not None and "sched" in ck and os.environ.get("RESUME_SCHED", "1") == "1":
            sched.load_state_dict(ck["sched"])
        start_step = int(ck.get("step", 0)); best = float(ck.get("best", -1e9))
        torch.cuda.synchronize()
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
        torch.save({"model": sd, "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "args": {**vars(args), "res_cond": args.res_cond}, "step": step,
                    "best": best}, os.path.join(args.out, name))

    (net.module if ddp else net).train()
    t0 = time.time()
    run = 0.0
    step = start_step
    while step < args.iters:
        # clean random-contrast source: WHOLE volume (full-image mode) or a batch of patches
        clean = bank.sample_full(args.full_size) if full else bank.sample(args.batch)
        torch.cuda.synchronize()   # serialize the random-size synth/degrade ops: leaving
        # sample+degrade+forward concurrently in-flight intermittently WEDGES a CUDA kernel
        # (stream never drains -> .item() blocks forever in cuMemcpyDtoHAsync). Per-phase
        # sync ran 5470 steps clean where the fully-async loop hung at ~700-2000.
        # fully-batched degradation (no per-sample Python loop -> low CUDA-launch count)
        noisy, target, s, res = batched_degrade(
            clean, _thick_slice_degrade, _gauss_blur3d,
            sigma_min=0.0, sigma_max=args.sigma_max, rician_prob=0.5,
            blur_max=args.blur_max, thick_max=args.thick_max, inplane_max=args.inplane_max)
        torch.cuda.synchronize()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if full and args.grad_ckpt:
                den = bf3d_forward_ckpt(net.module if ddp else net, noisy, sigma=s,
                                        res=res if args.res_cond else None)
            else:
                den = net(noisy, sigma=s, res=res if args.res_cond else None)
            loss = F.l1_loss(den, target) + args.grad_w * grad_l1(den, target)
        if not torch.isfinite(loss) or loss.item() > 10.0:
            opt.zero_grad(set_to_none=True); step += 1; continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # NO gradient clipping: SynthSeg's infinite random data can't over-fit, so we don't add
        # protective measures (the reference uses none). The isfinite/loss>10 skip above is only a
        # NaN/explosion guard, not regularisation.
        opt.step()
        if sched is not None:
            sched.step()
        run += loss.item(); step += 1

        if is_main and step % args.log_every == 0:
            ips = (step - start_step) / (time.time() - t0) * world
            print(f"step {step}/{args.iters}  loss={run/args.log_every:.4f}  "
                  f"{ips:.1f} it/s  lr={(sched.get_last_lr()[0] if sched is not None else opt.param_groups[0]['lr']):.2e}", flush=True)
            run = 0.0
        if step % args.save_every == 0:
            save("last.pt", step)
        if step % args.val_every == 0 or step == args.iters:
            if is_main:
                v = validate(net, bank, dev, bool(args.res_cond),
                             full=full, full_size=args.full_size,
                             res_denom=max(args.thick_max - 1.0, 1.0))
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
