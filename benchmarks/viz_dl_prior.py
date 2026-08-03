"""Quick MTV vs MTV+RED comparison with a visual (GT | MTV | MTV+RED | diff).

Reuses the benchmark's simulation, GT resampling and scoring. Saves benchmarks/viz.png
and prints PSNR/SSIM for both priors. Run from the repo root.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, contextlib, io as _io, sys, time
sys.path.insert(0, 'benchmarks')
from bench_dl_prior import simulate, resample_gt, score
from unires.struct import settings
from unires.run import preproc

DEV = 'cuda'
gt, gt_mat, x, mat_x = simulate(2)   # T1, T2 (2 channels for speed on the 6 GB laptop)


def recon(prior, **kw):
    data = [[[x[i], mat_x[i]]] for i in range(len(x))]
    s = settings(); s.device = DEV; s.write_out = False; s.do_coreg = False
    s.unified_rigid = False; s.scaling = False
    s.sched_num = 0            # no coarse-to-fine: sit at the target reg from the start
    s.max_iter = 12; s.cgs_max_iter = 8; s.tolerance = 0.0; s.prior = prior
    for k, v in kw.items():
        setattr(s, k, v)
    t = time.time()
    with contextlib.redirect_stdout(_io.StringIO()):
        y, ymat, _ = preproc(data, s)
    return y, ymat, time.time() - t


def metrics(y, ymat):
    ps, ss = [], []
    for c in range(len(x)):
        gr = resample_gt(gt[c], gt_mat[c], y.shape[:3], ymat.to(DEV))
        p, s_, _ = score(y[..., c], gr); ps.append(p); ss.append(s_)
    return sum(ps) / len(ps), sum(ss) / len(ss)


print('running mtv...', flush=True)
y_mtv, ymat, t0 = recon('mtv')
p0, s0 = metrics(y_mtv, ymat)
print(f'mtv:      PSNR={p0:.3f} SSIM={s0:.4f}  ({t0:.0f}s)', flush=True)

print('running mtv+red (DRUNet, mu=1e-4, sigma=0.03)...', flush=True)   # mild mu (1e-2 over-regularizes)
y_red, ymat2, t1 = recon('mtv+red', red_mu=1e-4, red_sigma=0.03, red_denoiser='drunet', red_batch=16)
p1, s1 = metrics(y_red, ymat2)
print(f'mtv+red:  PSNR={p1:.3f} SSIM={s1:.4f}  ({t1:.0f}s)   '
      f'dPSNR={p1 - p0:+.3f} dSSIM={s1 - s0:+.4f}', flush=True)

# --- visualise channel 0 (T1): GT | MTV | MTV+RED | (RED - MTV) diff ---
ch = 0
gr = resample_gt(gt[ch], gt_mat[ch], y_mtv.shape[:3], ymat.to(DEV)).cpu()
mtv, red = y_mtv[..., ch].cpu(), y_red[..., ch].cpu()
def match(a):
    m = gr > 0.02 * gr.max()
    sc = (a[m] * gr[m]).sum() / (a[m] * a[m]).sum().clamp_min(1e-8)
    return a * sc
mtv, red = match(mtv), match(red)

mid = [d // 2 for d in gr.shape]
planes = [(gr[mid[0]], mtv[mid[0]], red[mid[0]]),
          (gr[:, mid[1]], mtv[:, mid[1]], red[:, mid[1]]),
          (gr[:, :, mid[2]], mtv[:, :, mid[2]], red[:, :, mid[2]])]
vmax = float(gr[gr > 0].quantile(0.99))
fig, ax = plt.subplots(3, 4, figsize=(14, 10))
for r, (g, a, b) in enumerate(planes):
    cols = [(g, 'GT', 'gray'), (a, f'MTV  (PSNR {p0:.1f})', 'gray'),
            (b, f'MTV+RED  (PSNR {p1:.1f})', 'gray'), (b - a, 'RED - MTV', 'coolwarm')]
    for c, (img, title, cmap) in enumerate(cols):
        vm = vmax if c < 3 else float((b - a).abs().quantile(0.99)) + 1e-6
        ax[r, c].imshow(img.T, cmap=cmap, vmin=(0 if c < 3 else -vm), vmax=vm, origin='lower')
        if r == 0:
            ax[r, c].set_title(title)
        ax[r, c].axis('off')
fig.suptitle(f'T1 mid-slices (axial/coronal/sagittal)  |  mu=1e-2 sigma=0.03 DRUNet  |  '
             f'dPSNR={p1 - p0:+.2f}  dSSIM={s1 - s0:+.3f}')
fig.tight_layout()
fig.savefig('benchmarks/viz.png', dpi=110)
print('wrote benchmarks/viz.png', flush=True)
