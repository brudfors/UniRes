"""DECISIVE A/B: legacy (data_ptr-keyed, unpinned) denoiser cache vs the fixed (pinned) cache.

Post-fix runs show 0 aliases -- but the fix PINS the input tensor, which itself prevents address
reuse, so 0 is what we'd see either way. This restores the LEGACY semantics (return cached
unconditionally, store only the output => address free to be reused) and re-runs the identical
case, counting aliases with an independent content check and reporting monotonicity/PSNR.
"""
import contextlib, io as _io, glob, sys, torch
from bench_ixi import load_gt, simulate_one
from bench_dl_prior import run
import unires._denoiser as _dn
from unires._denoiser import _standardize, _unstandardize

MAXIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CK = '/dev/shm/synthseg/full_real_zbg_400k.pt'
DEV = 'cuda'

cls = None
for n in dir(_dn):
    o = getattr(_dn, n)
    if isinstance(o, type) and hasattr(o, 'denoise'):
        cls = o; break
FIXED = cls.denoise
STATS = {'alias': 0, 'hits': 0, 'miss': 0}

@torch.no_grad()
def legacy_denoise(self, vol, sigma=None, res=None):
    """Pre-fix behaviour: key on data_ptr/_version, store ONLY the output, return it blindly.
    We additionally stash a cheap content signature so we can DETECT (not prevent) aliasing."""
    if sigma is None:
        sigma = self.sigma
    if res is None:
        res_key = None
    elif isinstance(res, (list, tuple)):
        res_key = tuple(round(float(r), 4) for r in res)
    else:
        res_key = round(float(res), 4)
    key = (vol.data_ptr(), int(vol._version), float(sigma), res_key)
    sig_now = (round(float(vol.double().sum()), 4), tuple(vol.shape))
    ent = self._cache.get(key)
    if ent is not None:
        sig_old, out = ent
        if sig_old != sig_now:
            STATS['alias'] += 1          # SAME key, DIFFERENT content -> wrong result returned
        else:
            STATS['hits'] += 1
        return out                        # legacy: returned unconditionally
    STATS['miss'] += 1
    orig_dtype = vol.dtype
    v, scale = _standardize(vol.float())
    if getattr(self.backend, 'is_3d', False):
        out = _unstandardize(self.backend(v, sigma, res=res) if getattr(self.backend, 'is_res', False)
                             else self.backend(v, sigma), scale).to(orig_dtype)
    else:
        acc = None
        for axis in range(3):
            rr = self._apply_axis(v, sigma, axis)
            acc = rr if acc is None else acc + rr
        out = _unstandardize(acc / 3.0, scale).to(orig_dtype)
    self._cache[key] = (sig_now, out)     # NOT pinning vol -> its address may be reused
    self._cache_order.append(key)
    if len(self._cache_order) > self._cache_size:
        self._cache.pop(self._cache_order.pop(0), None)
    return out

p = sorted(glob.glob('/workspace/data/ixi/T1/*.nii.gz'))[-1]
vol, mat = load_gt(p)
x, mat_x = simulate_one(vol, mat, 2, vx_ts=4, noise_frac=0.03, seed=0)

def go(tag):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run([vol], [mat.to(DEV)], [x], [mat_x], 'mtv+red', MAXIT, tolerance=1e-4,
                red_mu=0.05, red_sigma=0.05, red_denoiser='biasfree3d', red_weights=CK,
                red_linesearch=True, red_nullspace=0.5, red_thick_taper=False)
    print(f"{tag:8s} PSNR={r['psnr']:.6f}  mono={r['mono']}  stats={dict(STATS)}")
    return r

print(f"max_iter={MAXIT}")
cls.denoise = legacy_denoise
STATS.update(alias=0, hits=0, miss=0); go("LEGACY")
STATS.update(alias=0, hits=0, miss=0); go("LEGACY")
cls.denoise = FIXED
go("FIXED")
go("FIXED")
