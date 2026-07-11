"""Evaluate the RED prior on REAL-anatomy IXI data (single-channel super-resolution).

Uses real IXI scans as GT (real scanner noise/contrast), simulates thick-slice low-res
with UniRes's forward model, reconstructs single-channel (MTV vs MTV+RED), and scores
PSNR/SSIM vs the IXI GT -- averaged over several subjects.

CLEAN (no leakage) config: evaluate the HELD-OUT-PD denoiser (trained on T1+T2 only)
on IXI PD -> the prior has never seen PD nor these exact SR tasks. Other contrasts
with the full model carry mild denoiser leakage (noted), but the SR task differs from
the denoiser's training objective, so it is still informative on real anatomy.

  python benchmarks/bench_ixi.py --contrast PD --weights <holdout_pd.pt> \
      --mu 0.05 --sigma 0.05 --n 5
"""
import argparse
import glob
import torch
from nitorch import io
from unires._project import _proj_info, _proj_apply
from bench_dl_prior import resample_gt, score, run

DEV = 'cuda'


def load_gt(path):
    n = io.map(path)
    return n.fdata().to(DEV).float(), n.affine.cpu().double()  # affine double (projector needs it)


def simulate_one(vol, mat, axis, vx_ts=4, noise_frac=0.03, rician=False, seed=0):
    torch.manual_seed(seed)
    dy = torch.as_tensor(vol.shape, dtype=mat.dtype)  # CPU
    vx = torch.ones(3); vx[axis] = vx_ts
    ms = torch.diag(torch.cat((vx, torch.ones(1))).to(mat.dtype))  # CPU
    mat_x_cpu = mat.mm(ms)
    dx = ms[:3, :3].inverse().mm(dy[:, None]).floor().squeeze()
    po = _proj_info(dy.to(DEV), mat.to(DEV), dx.to(DEV), mat_x_cpu.to(DEV),
                    prof_ip=0, prof_tp=0, gap=0.0, device=DEV, scl=0.0)
    mat_x = mat_x_cpu.to(DEV)
    A = lambda v: _proj_apply("A", v[None, None, ...], po)[0, 0, ...]
    clean = A(vol)
    pos = vol[vol > 0]
    scale = torch.quantile(pos.float()[:200000], 0.99).clamp_min(1e-6)
    std = float(noise_frac) * float(scale)
    if rician:
        x = torch.sqrt((clean + std * torch.randn_like(clean)) ** 2 +
                       (std * torch.randn_like(clean)) ** 2)
    else:
        x = clean + std * torch.randn_like(clean)
    return x, mat_x


def recon_score(vol, mat, x, mat_x, prior, **kw):
    # single channel, single observation; GT affine on DEV for scoring
    r = run([vol], [mat.to(DEV)], [x], [mat_x], prior, kw.pop('max_iter', 30),
            tolerance=0.0, **kw)
    return r['psnr_ch'][0], r['ssim_ch'][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrast', default='PD', choices=['T1', 'T2', 'PD'])
    ap.add_argument('--weights', required=True)
    ap.add_argument('--mu', type=float, default=0.05)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--n', type=int, default=5, help='number of IXI subjects')
    ap.add_argument('--axis', type=int, default=2, help='thick-slice axis')
    ap.add_argument('--noise_frac', type=float, default=0.03)
    ap.add_argument('--rician', action='store_true')
    ap.add_argument('--root', default='/dev/shm/ixi')
    args = ap.parse_args()
    paths = sorted(glob.glob(f"{args.root}/{args.contrast}/*.nii.gz"))[:args.n]
    print(f"IXI real-anatomy SR: contrast={args.contrast} n={len(paths)} "
          f"mu={args.mu} sigma={args.sigma} noise={'Rician' if args.rician else 'Gaussian'}"
          f"({args.noise_frac}) thick-axis={args.axis}\n")
    dmtv_p, dred_p, dmtv_s, dred_s = [], [], [], []
    b = dict(red_mu=args.mu, red_sigma=args.sigma, red_denoiser='biasfree3d',
             red_weights=args.weights, red_linesearch=True)
    for i, p in enumerate(paths):
        vol, mat = load_gt(p)
        x, mat_x = simulate_one(vol, mat, args.axis, noise_frac=args.noise_frac,
                                rician=args.rician, seed=i)
        pm, sm = recon_score(vol, mat, x, mat_x, 'mtv')
        pr, sr = recon_score(vol, mat, x, mat_x, 'mtv+red', **b)
        dmtv_p.append(pm); dred_p.append(pr); dmtv_s.append(sm); dred_s.append(sr)
        subj = p.split('/')[-1].split('-')[0]
        print(f"  {subj}: MTV {pm:.3f}/{sm:.4f}  RED {pr:.3f}/{sr:.4f}  "
              f"dPSNR={pr-pm:+.3f} dSSIM={sr-sm:+.4f}", flush=True)
    import statistics as st
    n = len(dmtv_p)
    dP = [dred_p[i]-dmtv_p[i] for i in range(n)]
    dS = [dred_s[i]-dmtv_s[i] for i in range(n)]
    print(f"\nMEAN over {n} subjects: MTV PSNR={sum(dmtv_p)/n:.3f}  RED PSNR={sum(dred_p)/n:.3f}"
          f"  dPSNR={sum(dP)/n:+.3f} (+/-{(st.pstdev(dP) if n>1 else 0):.3f})"
          f"  dSSIM={sum(dS)/n:+.4f}", flush=True)


if __name__ == '__main__':
    main()
