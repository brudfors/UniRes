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
        idx = torch.linspace(0, ref.numel() - 1, max_samples, device=ref.device).long()
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


# ----------------------------------------------------------------------------
# 2.5D wrapper
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
    def denoise(self, vol, sigma=None):
        if sigma is None:
            sigma = self.sigma
        key = (vol.data_ptr(), int(vol._version), float(sigma))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        orig_dtype = vol.dtype
        v, scale = _standardize(vol.float())
        acc = None
        for axis in range(3):
            r = self._apply_axis(v, sigma, axis)
            acc = r if acc is None else acc + r
        out = _unstandardize(acc / 3.0, scale).to(orig_dtype)
        self._cache[key] = out
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return out


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def make_denoiser(sett):
    """Build a :class:`Denoiser` from UniRes ``settings``."""
    device = getattr(sett, 'device', None)
    name = getattr(sett, 'red_denoiser', 'gaussian')
    if name == 'gaussian':
        backend = GaussianBackend(device=device)
    elif name == 'drunet':
        backend = DRUNetBackend(weights=getattr(sett, 'red_weights', None), device=device)
    else:
        raise ValueError(f"Unknown red_denoiser '{name}' (expected 'gaussian' or 'drunet').")
    return Denoiser(backend, sigma=getattr(sett, 'red_sigma', 0.05),
                    batch=getattr(sett, 'red_batch', 16))
