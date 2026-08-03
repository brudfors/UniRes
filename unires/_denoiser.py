"""Deep-learning denoiser priors for UniRes (RED / Plug-and-Play).

A denoiser ``D_sigma`` maps a noisy 3D volume to a cleaner one; the solver uses
the residual ``y - D_sigma(y)`` as a prior gradient in the RED (Regularization by
Denoising) augmentation of the UniRes objective (see ``unires/_update.py``).

Design goals (see the design spec):
  * general -- per-volume intensity standardisation + per-channel application, so
    the same denoiser works on any MRI contrast / intensity range;
  * fast -- a 2D denoiser is applied 2.5D (slice-wise over the 3 axes, averaged)
    with slice mini-batching to bound memory;
  * pluggable -- an analytic Gaussian backend (no weights, for tests / a weak
    baseline) and a pretrained DRUNet backend (lazy ``deepinv`` import).

This module contains no algorithm changes on its own; nothing here runs unless a
caller constructs a denoiser (i.e. ``prior`` includes ``red``).
"""
import math
import os
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Intensity standardisation (MR intensities are arbitrary / un-calibrated)
# ----------------------------------------------------------------------------
def _robust_scale(vol, q=0.99, max_samples=1_000_000):
    """A robust positive scale (~99th percentile of positive voxels)."""
    finite = vol[torch.isfinite(vol)]
    pos = finite[finite > 0]
    ref = pos if pos.numel() > 0 else finite.abs()
    if ref.numel() == 0:
        return vol.new_tensor(1.0)
    if ref.numel() > max_samples:
        # NB: clamp — for volumes > ~16.7M voxels, ref.numel()-1 is not exactly representable in
        # float32 and linspace's endpoint can round UP past numel-1, so .long() would index OOB
        # (device-side assert). Seen on large MICCAI recons at 6mm; clamp makes it safe.
        idx = torch.linspace(0, ref.numel() - 1, max_samples, device=ref.device).long()
        idx = idx.clamp_(max=ref.numel() - 1)
        ref = ref.flatten()[idx]
    scale = torch.quantile(ref.float(), q)
    return torch.clamp(scale, min=1e-8)


def _standardize(vol):
    """Scale a (non-negative MR) volume to roughly [0, 1]; return (vol, scale)."""
    scale = _robust_scale(vol)
    return vol / scale, scale


def _unstandardize(vol, scale):
    return vol * scale


# ----------------------------------------------------------------------------
# Backends: callable (x2d [B,1,H,W], sigma) -> [B,1,H,W]
# ----------------------------------------------------------------------------
def _gaussian_blur2d(x, std):
    """Separable 2D Gaussian blur with reflect padding."""
    radius = max(1, int(math.ceil(3.0 * std)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k1 = torch.exp(-(coords ** 2) / (2.0 * std * std))
    k1 = k1 / k1.sum()
    kw = k1.view(1, 1, 1, -1)
    kh = k1.view(1, 1, -1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode='reflect')
    x = F.conv2d(x, kw)
    x = F.pad(x, (0, 0, radius, radius), mode='reflect')
    x = F.conv2d(x, kh)
    return x


class GaussianBackend:
    """Analytic Gaussian-blur 'denoiser' (no learned weights).

    A weak but genuine denoiser: larger ``sigma`` -> more smoothing. Used for
    plumbing/tests and as a trivial baseline; not intended to improve quality.
    """
    def __init__(self, device=None):
        self.device = device

    def __call__(self, x, sigma):
        std = max(0.3, float(sigma) * 6.0)
        return _gaussian_blur2d(x, std)


class DRUNetBackend:
    """Pretrained DRUNet denoiser (grayscale), via a lazy ``deepinv`` import."""
    def __init__(self, weights=None, device=None):
        try:
            import deepinv  # lazy: only required for this backend
        except ImportError as e:  # pragma: no cover - exercised only when selected
            raise ImportError(
                "The 'drunet' denoiser backend requires deepinv "
                "(`pip install deepinv`)."
            ) from e
        model = deepinv.models.DRUNet(
            in_channels=1, out_channels=1, pretrained=weights or 'download')
        if device is not None:
            model = model.to(device)
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, x, sigma):
        return self.model(x, float(sigma))


class BiasFree3DBackend:
    """Pretrained bias-free 3D denoiser (Phase 3): a native 3D volume->volume model.

    Unlike the 2D backends this is TRUE 3D -- one forward pass over the whole
    (standardised) volume gives through-plane coherence that 2.5D only approximates
    by averaging orthogonal passes, and one pass is cheaper than 2.5D's 3*N_slices.
    Marked ``is_3d`` so :class:`Denoiser` skips the 2.5D axis loop. Fully
    convolutional -> handles arbitrary volume sizes; optional overlap-tiling bounds
    memory for very large volumes."""
    is_3d = True

    def __init__(self, weights, device=None, base=48, depth=3, tile=0, overlap=16):
        ckpt = torch.load(weights, map_location=device or 'cpu')
        a = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        if a.get('net') == 'basicunet':          # reference SynthSeg net (MONAI BasicUNet)
            from ._denoiser_net import BasicUNetRestore
            res_map = bool(a.get('res_cond', 0))  # thickness-conditioned restoration model?
            # infer #conditioning channels from the saved first conv (in_ch = 1 + res_ch):
            # supports both the legacy isotropic scalar (res_ch=1) and DIRECTIONAL per-axis
            # (res_ch=3) conditioned models without a separate flag.
            w = state.get('net.conv_0.conv_0.conv.weight')
            res_ch = max(int(w.shape[1]) - 1, 0) if (res_map and w is not None) else 0
            net = BasicUNetRestore(in_ch=1, out_ch=1, res_ch=res_ch)
            self.is_cross = False
            self.is_res = res_map
            self.res_ch = res_ch
            # res normalisation denominator = trained (thick_max - 1); matches synth_data.py so
            # the inference thickness factor maps into [0,1] the SAME way it did in training.
            self.res_denom = max(float(a.get('thick_max', 8)) - 1.0, 1.0)
        else:
            from ._denoiser_net import BiasFreeUNet3D  # lazy: only for this backend
            # infer input config from the head conv: in_ch = 1 + sigma_map + context + res_map
            in_ch = state['head.c1.weight' if 'head.c1.weight' in state else 'head.weight'].shape[1]
            res_map = bool(a.get('res_cond', 0))   # thickness-conditioned restoration model?
            if res_map:                             # img + sigma + res (no context)
                net = BiasFreeUNet3D(base=a.get('base', base), depth=a.get('depth', depth),
                                     sigma_map=True, context=False, res_map=True)
            else:                                   # legacy: sigma / cross-channel-context inference
                net = BiasFreeUNet3D(base=a.get('base', base), depth=a.get('depth', depth),
                                     sigma_map=(in_ch >= 2), context=(in_ch >= 3))
            self.is_cross = bool(getattr(net, 'context', False))  # cross-channel model?
            self.is_res = res_map                                 # thickness-conditioned model?
            self.res_ch = 1 if res_map else 0                     # BiasFree res = isotropic scalar
            self.res_denom = max(float(a.get('thick_max', 8)) - 1.0, 1.0)
        net.load_state_dict(state)
        self.model = net.to(device).eval() if device is not None else net.eval()
        self.tile = int(tile)
        self.overlap = int(overlap)

    @torch.no_grad()
    def __call__(self, vol, sigma, context=None, res=None):
        """vol: standardised 3D tensor (D,H,W) -> denoised (D,H,W). context: optional
        cross-channel edge map; res: optional scalar normalised slice-thickness."""
        x = vol[None, None]
        ctx = context[None, None] if context is not None else None
        if self.tile <= 0 or max(vol.shape) <= self.tile:
            return self.model(x, float(sigma), context=ctx, res=res)[0, 0]
        # overlap-tiled inference (memory bound for large volumes)
        out = torch.zeros_like(vol)
        wsum = torch.zeros_like(vol)
        t, ov = self.tile, self.overlap
        step = t - ov
        rngs = [list(range(0, max(1, s - ov), step)) for s in vol.shape]
        for oi in rngs[0]:
            for oj in rngs[1]:
                for ok in rngs[2]:
                    i, j, k = (min(oi, vol.shape[0]-t) if vol.shape[0] > t else 0,
                               min(oj, vol.shape[1]-t) if vol.shape[1] > t else 0,
                               min(ok, vol.shape[2]-t) if vol.shape[2] > t else 0)
                    sl = (slice(i, i+t), slice(j, j+t), slice(k, k+t))
                    den = self.model(vol[sl][None, None], float(sigma))[0, 0]
                    out[sl] += den; wsum[sl] += 1.0
        return out / wsum.clamp_min(1.0)


# ----------------------------------------------------------------------------
# 2.5D wrapper (and 3D pass-through for native-3D backends)
# ----------------------------------------------------------------------------
class Denoiser:
    """Apply a 2D ``backend`` to a 3D volume, 2.5D, with standardisation.

    ``denoise(vol, sigma)`` standardises ``vol``, denoises the 2D slices along
    each of the three axes (mini-batched), averages the three results, and undoes
    the standardisation. Returns a tensor of the same shape/dtype/device.
    """
    def __init__(self, backend, sigma=0.05, batch=16, cache_size=8):
        self.backend = backend
        self.sigma = float(sigma)
        self.batch = int(batch)
        # Small cache keyed on (data_ptr, version, sigma). Because CG updates y in
        # place, the next iteration's RHS asks for D(y) of exactly the iterate the
        # previous objective already denoised -> a cache halves the denoise passes.
        self._cache = {}
        self._cache_order = []
        self._cache_size = int(cache_size)

    def _apply_axis(self, vol, sigma, axis):
        v = vol.movedim(axis, 0)          # (N, H, W)
        n = v.shape[0]
        out = torch.empty_like(v)
        x = v.unsqueeze(1)                # (N, 1, H, W)
        for i in range(0, n, self.batch):
            chunk = x[i:i + self.batch].contiguous()
            out[i:i + self.batch] = self.backend(chunk, sigma).squeeze(1)
        return out.movedim(0, axis)

    @torch.no_grad()
    def denoise(self, vol, sigma=None, res=None):
        if sigma is None:
            sigma = self.sigma
        # res may be None, a scalar (legacy isotropic), or a list/tuple (directional per-axis)
        if res is None:
            res_key = None
        elif isinstance(res, (list, tuple)):
            res_key = tuple(round(float(r), 4) for r in res)
        else:
            res_key = round(float(res), 4)
        # CORRECTNESS: the key includes data_ptr, and CUDA's caching allocator REUSES freed
        # addresses -- a freed tensor's address can be handed to a DIFFERENT tensor with
        # _version==0, colliding on this key and returning a denoised result for the WRONG
        # content. The RED line search triggers this constantly (each backtrack allocates and
        # frees a trial (1-t)*y_old + t*y_new). Symptoms: the monotonicity guard compares stale
        # objectives (non-monotone despite the guard), the post-revert objective differs from
        # nll_old, and run-to-run output jitters by 0.1-0.3 dB. FIX: keep a REFERENCE to the
        # input tensor in the cache entry, which pins its address for as long as the entry
        # lives, so no other tensor can occupy it (in-place edits still bump _version -> miss).
        key = (vol.data_ptr(), int(vol._version), float(sigma), res_key)
        hit = self._cache.get(key)
        if hit is not None:
            ref, cached = hit
            if ref is vol:                      # same tensor object -> genuine hit
                return cached
            # key matched but a DIFFERENT tensor now occupies that address: this is the
            # aliasing bug. Count it so the hazard is observable rather than silent.
            self._alias_hits = getattr(self, '_alias_hits', 0) + 1
        orig_dtype = vol.dtype
        v, scale = _standardize(vol.float())
        if getattr(self.backend, 'is_3d', False):
            # native 3D backend: one volume pass (no 2.5D axis averaging). Pass the
            # slice-thickness conditioning to a thickness-conditioned (res) backend.
            out = _unstandardize(self.backend(v, sigma, res=res) if getattr(self.backend, 'is_res', False)
                                 else self.backend(v, sigma), scale).to(orig_dtype)
        else:
            acc = None
            for axis in range(3):
                r = self._apply_axis(v, sigma, axis)
                acc = r if acc is None else acc + r
            out = _unstandardize(acc / 3.0, scale).to(orig_dtype)
        self._cache[key] = (vol, out)           # keep vol alive -> its address cannot be reused
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return out

    @torch.no_grad()
    def denoise_multi(self, vols, sigmas):
        """Cross-channel denoise: each channel is denoised using a CONTEXT edge map
        aggregated (max) over the OTHER channels' standardised edges -- so the prior
        fills in detail guided by where other contrasts have edges. Falls back to
        independent per-channel denoise if the backend is not cross-channel.
        ``vols``/``sigmas``: lists over channels; returns a list of denoised volumes."""
        if not getattr(self.backend, 'is_cross', False):
            return [self.denoise(vols[c], sigmas[c]) for c in range(len(vols))]
        std = [_standardize(v.float()) for v in vols]              # (v_std, scale) per ch

        def _edge(vs):
            g = torch.gradient(vs, dim=(0, 1, 2))
            return torch.sqrt(sum(gi ** 2 for gi in g) + 1e-8)
        edges = [_edge(vs) for vs, _ in std]
        outs = []
        for c in range(len(vols)):
            ctx = torch.stack([edges[d] for d in range(len(vols)) if d != c]).amax(0)
            den = self.backend(std[c][0], sigmas[c], context=ctx)
            outs.append(_unstandardize(den, std[c][1]).to(vols[c].dtype))
        return outs


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
#: filename of the released SynthSeg-trained restoration prior
RED_WEIGHTS_NAME = 'unires_red_synth_full.pt'
#: GitHub release the weights are published in (see docs/dl_prior.md)
RED_WEIGHTS_TAG = 'red-prior-v1'
RED_WEIGHTS_URL = (f'https://github.com/brudfors/UniRes/releases/download/'
                   f'{RED_WEIGHTS_TAG}/{RED_WEIGHTS_NAME}')
#: sha256 of the released file; the download is rejected if it does not match
RED_WEIGHTS_SHA256 = 'c356f6a749bf25406952a2892e799bc91b25f0463f3677039f62b6df1449f06b'
#: where an auto-download is cached
RED_WEIGHTS_CACHE = os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache'),
    'unires', RED_WEIGHTS_NAME)
#: search order when ``red_weights`` is not given explicitly
RED_WEIGHTS_SEARCH = (
    os.environ.get('UNIRES_RED_WEIGHTS'),                                  # 1. explicit env var
    os.path.join(os.path.dirname(__file__), 'weights', RED_WEIGHTS_NAME),  # 2. alongside package
    RED_WEIGHTS_CACHE,                                                     # 3. user cache
)


def _sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def _download_red_weights(dst=None, url=None, verbose=True):
    """Fetch the released weights into the cache (stdlib only -- no new dependency).

    Downloads to a temporary file and renames on success, so an interrupted download can
    never leave a truncated checkpoint behind. The sha256 is verified when
    ``RED_WEIGHTS_SHA256`` is set to a real digest.
    """
    import shutil
    import urllib.request
    dst = dst or RED_WEIGHTS_CACHE
    url = url or RED_WEIGHTS_URL
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.part'
    if verbose:
        print(f'UniRes: downloading RED prior weights\n  from {url}\n  to   {dst}', flush=True)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f'Could not download the RED prior weights from {url} ({e}). Download it manually '
            f'and place it at {dst}, or set settings.red_weights / $UNIRES_RED_WEIGHTS.') from e
    if RED_WEIGHTS_SHA256 and not RED_WEIGHTS_SHA256.endswith('0' * 16):
        got = _sha256(tmp)
        if got != RED_WEIGHTS_SHA256:
            os.remove(tmp)
            raise RuntimeError(f'Checksum mismatch for {url}: expected {RED_WEIGHTS_SHA256}, '
                               f'got {got}. Refusing to use the file.')
    os.replace(tmp, dst)
    return dst


def _resolve_red_weights(weights, download=True):
    """Locate the biasfree3d checkpoint.

    An explicit path always wins. Otherwise search ``RED_WEIGHTS_SEARCH`` (env var, next to
    the package, user cache) and finally fetch the published release into the cache. Set
    ``$UNIRES_NO_DOWNLOAD=1`` for offline use: the error then names every location tried.
    """
    if weights:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"red_weights not found: {weights}")
        return weights
    tried = []
    for cand in RED_WEIGHTS_SEARCH:
        if not cand:
            continue
        tried.append(cand)
        if os.path.isfile(cand):
            return cand
    if download and os.environ.get('UNIRES_NO_DOWNLOAD', '') not in ('1', 'true', 'True'):
        return _download_red_weights()
    raise FileNotFoundError(
        "red_denoiser='biasfree3d' needs a checkpoint. Pass settings.red_weights=<path>, set "
        "$UNIRES_RED_WEIGHTS, or place the released weights at one of:\n  "
        + "\n  ".join(tried) + f"\nRelease: {RED_WEIGHTS_URL}")


def make_denoiser(sett):
    """Build a :class:`Denoiser` from UniRes ``settings``."""
    device = getattr(sett, 'device', None)
    name = getattr(sett, 'red_denoiser', 'gaussian')
    if name == 'gaussian':
        backend = GaussianBackend(device=device)
    elif name == 'drunet':  # noqa: E301 (see _resolve_red_weights above for biasfree3d)
        backend = DRUNetBackend(weights=getattr(sett, 'red_weights', None), device=device)
    elif name == 'biasfree3d':
        weights = _resolve_red_weights(getattr(sett, 'red_weights', None))
        backend = BiasFree3DBackend(weights, device=device,
                                    tile=getattr(sett, 'red_tile', 0))
    else:
        raise ValueError(
            f"Unknown red_denoiser '{name}' (expected 'gaussian', 'drunet' or 'biasfree3d').")
    return Denoiser(backend, sigma=getattr(sett, 'red_sigma', 0.05),
                    batch=getattr(sett, 'red_batch', 16))
