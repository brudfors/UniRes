"""Verify the denoiser-cache aliasing fix restores RED line-search monotonicity.

The cache was keyed on vol.data_ptr(); CUDA reuses freed addresses, so a trial tensor from the
line search could collide with a stale entry and return a denoised result for DIFFERENT content.
This checks (a) the reported objective is monotone, and (b) repeated identical runs agree.
"""
import contextlib, io as _io, re, sys, glob, torch
from bench_ixi import load_gt, simulate_one
from bench_dl_prior import run

CK = '/dev/shm/synthseg/full_real_zbg_400k.pt'
DEV = 'cuda'

def one(seed, tol=1e-4, max_iter=12):
    p = sorted(glob.glob('/workspace/data/ixi/T1/*.nii.gz'))[-1]
    vol, mat = load_gt(p)
    x, mat_x = simulate_one(vol, mat, 2, vx_ts=4, noise_frac=0.03, seed=seed)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run([vol], [mat.to(DEV)], [x], [mat_x], 'mtv+red', max_iter,
                tolerance=tol, red_mu=0.05, red_sigma=0.05, red_denoiser='biasfree3d',
                red_weights=CK, red_linesearch=True, red_nullspace=0.5, red_thick_taper=False)
    nlyx = [float(m) for m in re.findall(r"nlyx =\s*([-\d.eE+]+)", buf.getvalue())]
    return r, nlyx

print("=== monotonicity + reproducibility of the RED line search ===")
r1, n1 = one(0)
r2, n2 = one(0)
viol = [(i, n1[i], n1[i+1]) for i in range(len(n1)-1) if n1[i+1] > n1[i] * (1 + 1e-9) + 1e-9]
print(f"objective trace ({len(n1)} pts): {[f'{v:.6e}' for v in n1]}")
print(f"monotone: {not viol}   violations: {len(viol)}")
for i, a, b in viol[:5]:
    print(f"   iter {i}->{i+1}: {a:.6e} -> {b:.6e}   (+{100*(b-a)/a:.3f}%)")
print(f"run-to-run PSNR: {r1['psnr']:.6f} vs {r2['psnr']:.6f}  |diff|={abs(r1['psnr']-r2['psnr']):.2e}")
print(f"run-to-run mono flags: {r1['mono']} / {r2['mono']}")
