"""Multi-channel real-anatomy validation on IXI (the actual UniRes use case).

For each subject: load real T1/T2/PD, resample T2/PD onto the T1 grid to get an
ALIGNED 3-channel real GT, simulate per-channel thick-slice low-res, reconstruct
JOINTLY (multi-channel MTV, +/- RED), and score each channel vs its real GT.

Tests whether, on real multi-channel data, RED still improves T2/PD, whether T1
regresses (the joint-TV coupling effect seen on BrainWeb), and whether per-channel
mu weights (red_mu_weights) recover T1 -> all channels improve.
"""
import argparse
import glob
import torch
from nitorch import io
from unires._project import _proj_info, _proj_apply
from bench_dl_prior import resample_gt, score, run

DEV = 'cuda'
CH = ['T1', 'T2', 'PD']


def load(path):
    n = io.map(path)
    return n.fdata().to(DEV).float(), n.affine.cpu().double()


def build_aligned(subj_paths):
    """Resample all contrasts onto the T1 grid -> aligned 3ch GT on one grid."""
    vols, mats = zip(*[load(p) for p in subj_paths])
    t1, t1_mat = vols[0], mats[0]
    gt = [t1]
    for c in range(1, len(vols)):
        gt.append(resample_gt(vols[c], mats[c].to(DEV), t1.shape, t1_mat.to(DEV)))
    return gt, t1_mat  # all on t1 grid


def simulate_mc(gt, gt_mat, noise_frac=0.10, seed=0):
    torch.manual_seed(seed)
    x, mat_x = [], []
    for i in range(len(gt)):
        vol = gt[i]
        dy = torch.as_tensor(vol.shape, dtype=gt_mat.dtype)
        vx = torch.ones(3); vx[i % 3] = 4.0
        ms = torch.diag(torch.cat((vx, torch.ones(1))).to(gt_mat.dtype))
        mx = gt_mat.mm(ms)
        dx = ms[:3, :3].inverse().mm(dy[:, None]).floor().squeeze()
        po = _proj_info(dy.to(DEV), gt_mat.to(DEV), dx.to(DEV), mx.to(DEV),
                        prof_ip=0, prof_tp=0, gap=0.0, device=DEV, scl=0.0)
        A = lambda v: _proj_apply("A", v[None, None, ...], po)[0, 0, ...]
        clean = A(vol)
        pos = vol[vol > 0]
        std = noise_frac * float(torch.quantile(pos.float()[:200000], 0.99).clamp_min(1e-6))
        x.append(clean + std * torch.randn_like(clean))
        mat_x.append(mx.to(DEV))
    return x, mat_x


def run_cfg(gt, gt_mat, x, mat_x, tag, prior, base=None, **kw):
    gm = [gt_mat.to(DEV)] * len(gt)
    r = run(gt, gm, x, mat_x, prior, 30, tolerance=0.0, **kw)
    p = r['psnr_ch']
    d = [p[i] - base['psnr_ch'][i] for i in range(len(p))] if base else [0]*len(p)
    print(f"{tag:>22} | " + "  ".join(f"{CH[i]} {p[i]:.3f}({d[i]:+.3f})" for i in range(len(p)))
          + f" | avg {r['psnr']:.3f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--mu', type=float, default=0.03)
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--noise_frac', type=float, default=0.10)
    ap.add_argument('--root', default='/dev/shm/ixi')
    args = ap.parse_args()
    subjects = []
    for t1 in sorted(glob.glob(f"{args.root}/T1/*.nii.gz")):
        sid = t1.split('/')[-1].split('-')[0]
        t2 = glob.glob(f"{args.root}/T2/{sid}-*.nii.gz")
        pd = glob.glob(f"{args.root}/PD/{sid}-*.nii.gz")
        if t2 and pd:
            subjects.append((sid, [t1, t2[0], pd[0]]))
        if len(subjects) >= args.n:
            break
    b = dict(red_mu=args.mu, red_sigma=args.sigma, red_denoiser='biasfree3d',
             red_weights=args.weights, red_linesearch=True)
    for sid, paths in subjects:
        print(f"\n--- subject {sid} (multi-channel, real IXI) ---")
        gt, gt_mat = build_aligned(paths)
        x, mat_x = simulate_mc(gt, gt_mat, noise_frac=args.noise_frac)
        mtv = run_cfg(gt, gt_mat, x, mat_x, 'MTV', 'mtv')
        run_cfg(gt, gt_mat, x, mat_x, 'RED uniform', 'mtv+red', base=mtv, **b)
        run_cfg(gt, gt_mat, x, mat_x, 'RED [0,1,1]', 'mtv+red', base=mtv,
                red_mu_weights=[0.0, 1.0, 1.0], **b)


if __name__ == '__main__':
    main()
