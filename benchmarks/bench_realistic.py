"""'No inverse crime' validation: simulate low-res with a forward model that DOES
NOT match the recon, to check the RED gains are not an artifact of reconstructing
with the exact model used to simulate.

Mismatches vs the recon (which assumes profile_tp=0 rect, gap=0, Gaussian noise):
  * through-plane slice profile: tri/gauss instead of rect (the recon can't model it),
  * slice gap > 0,
  * RICIAN noise (magnitude MRI) instead of Gaussian.

Runs MTV vs MTV+RED(bias-free 3D) per channel, incl. per-channel mu weights, and
prints the data-consistency residual so we can see RED does not worsen data fit.
"""
import argparse
import torch
from nitorch import io
from unires._project import _proj_info, _proj_apply
from bench_dl_prior import resample_gt, score, run, PATHS, DEV


def simulate_realistic(n_ch, vx_ts=4, std_noise=50.0, seed=0, prof_tp=2, gap=0.2,
                       rician=True):
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
        # MISMATCHED forward model (through-plane tri/gauss profile + gap)
        po = _proj_info(dy.to(DEV), my.to(DEV), dx.to(DEV), mat_x[i],
                        prof_ip=0, prof_tp=prof_tp, gap=gap, device=DEV, scl=0.0)
        A = lambda v: _proj_apply("A", v[None, None, ...], po)[0, 0, ...]
        clean = A(gt[i])
        if rician:  # magnitude MRI noise
            n1 = std_noise * torch.randn(clean.shape, device=DEV)
            n2 = std_noise * torch.randn(clean.shape, device=DEV)
            x.append(torch.sqrt((clean + n1) ** 2 + n2 ** 2))
        else:
            x.append(clean + std_noise * torch.randn(clean.shape, device=DEV))
    return gt, gt_mat, x, mat_x


def per_ch(gt, gt_mat, x, mat_x, tag, prior, **kw):
    r = run(gt, gt_mat, x, mat_x, prior, 30, tolerance=0.0, **kw)
    p, s = r['psnr_ch'], r['ssim_ch']
    names = ['T1', 'T2', 'PD'][:len(x)]
    parts = [f"{names[i]} {p[i]:.3f}/{s[i]:.4f}" for i in range(len(x))]
    print(f"{tag:>26} | " + "  ".join(parts) + f" | avg {r['psnr']:.3f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--mu', type=float, default=0.02)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--channels', type=int, default=3)
    ap.add_argument('--prof_tp', type=int, default=2, help='0 rect|1 tri|2 gauss (recon assumes 0)')
    ap.add_argument('--gap', type=float, default=0.2)
    ap.add_argument('--gaussian', action='store_true', help='Gaussian noise instead of Rician')
    args = ap.parse_args()
    gt, gt_mat, x, mat_x = simulate_realistic(
        args.channels, prof_tp=args.prof_tp, gap=args.gap, rician=not args.gaussian)
    print(f"REALISTIC sim: prof_tp={args.prof_tp}(recon assumes 0/rect) gap={args.gap} "
          f"noise={'Gaussian' if args.gaussian else 'Rician'}  mu={args.mu} sigma={args.sigma}\n")
    b = dict(red_mu=args.mu, red_sigma=args.sigma, red_denoiser='biasfree3d',
             red_weights=args.weights, red_linesearch=True)
    per_ch(gt, gt_mat, x, mat_x, 'MTV', 'mtv')
    per_ch(gt, gt_mat, x, mat_x, 'MTV+RED-3D uniform', 'mtv+red', **b)
    if args.channels == 3:
        per_ch(gt, gt_mat, x, mat_x, 'MTV+RED-3D [0,1,1]', 'mtv+red',
               red_mu_weights=[0.0, 1.0, 1.0], **b)


if __name__ == '__main__':
    main()
