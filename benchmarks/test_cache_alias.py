"""Does the denoiser cache actually ALIAS (data_ptr reuse) during a real DL-prior recon?

Counts entries where the (data_ptr, _version, sigma, res) key matched but the cached input
tensor is NOT the tensor being denoised -- i.e. the CUDA allocator handed the same address to
different content. Non-zero => the old cache (which returned `cached` unconditionally) was
returning a denoised result for the WRONG image.
"""
import contextlib, io as _io, glob, torch
from bench_ixi import load_gt, simulate_one
from bench_dl_prior import run
import unires._denoiser as _dn

CK = '/dev/shm/synthseg/full_real_zbg_400k.pt'
DEV = 'cuda'

# track every denoiser instance created during the run
made = []
_orig_init = _dn.Denoiser.__init__ if hasattr(_dn, 'Denoiser') else None
cls = None
for name in dir(_dn):
    o = getattr(_dn, name)
    if isinstance(o, type) and hasattr(o, 'denoise'):
        cls = o
        break
print("denoiser class:", cls.__name__)
_orig = cls.__init__
def patched(self, *a, **k):
    _orig(self, *a, **k); made.append(self)
cls.__init__ = patched

p = sorted(glob.glob('/workspace/data/ixi/T1/*.nii.gz'))[-1]
vol, mat = load_gt(p)
x, mat_x = simulate_one(vol, mat, 2, vx_ts=4, noise_frac=0.03, seed=0)
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    r = run([vol], [mat.to(DEV)], [x], [mat_x], 'mtv+red', 12, tolerance=1e-4,
            red_mu=0.05, red_sigma=0.05, red_denoiser='biasfree3d', red_weights=CK,
            red_linesearch=True, red_nullspace=0.5, red_thick_taper=False)
tot = sum(getattr(d, '_alias_hits', 0) for d in made)
print(f"denoiser instances: {len(made)}")
print(f"ALIASED cache keys detected: {tot}")
print(f"PSNR {r['psnr']:.6f}  mono {r['mono']}")
print("VERDICT:", "ALIASING REAL -> old cache returned wrong-content results" if tot > 0
      else "no aliasing observed in this run -> hypothesis NOT confirmed here")
