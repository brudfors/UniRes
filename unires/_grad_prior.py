"""Merged-gradient DL prior for UniRes (learned / analytic TV in the gradient domain).

Where the RED prior (``unires/_denoiser.py``) denoises the reconstruction *per
channel* in the *image* domain (C network calls per ADMM iteration, 2.5D), this
module acts on the *channel-pooled gradient magnitude* inside UniRes's joint-TV
z-update. UniRes already:
  1. merges all C channels x 3 directions into one per-voxel scalar magnitude
     ``m`` (``_update.py``),
  2. applies ONE scalar nonlinearity -- the vectorial soft-threshold gain
     ``s = (m - 1/rho)_+ / (m + eps)`` -- to it,
  3. broadcasts that gain back to every channel/direction (direction preserved,
     only the magnitude reshaped).

So the learned prior is a ~1-line swap of step 2: replace the analytic gain with a
learned/generalized gain on the merged magnitude ``m`` and leave the broadcast
verbatim. The network therefore runs ONCE per ADMM iteration and is C-independent
(see ``docs/merged_gradient_prior.md``).

Design (mirrors ``_denoiser.py``'s settings / factory / cache pattern, but NOT its
2.5D wrapper nor its positive-quantile ``_standardize`` -- gradient fields are
*signed*):
  * pluggable backends exposing ``gain(m, rho, eps) -> per-voxel gain`` (analytic
    ``softthr`` == exact JTV, ``charbonnier`` in Phase 1) and, later, native-3D
    ``deepinv`` UNet/DnCNN magnitude denoisers;
  * a thin :class:`GradPrior` wrapper (cache + optional gain clamp for the learned,
    non-hallucinating case);
  * :func:`make_grad_prior` factory from UniRes ``settings``.

Phase 0 provides the ``softthr`` analytic gain (reproduces classic MTV bit-for-bit).
Phase 1 adds ``charbonnier``: a generalized analytic edge-preserving gain with an
EXPLICIT energy (an exact radial prox ⇒ monotone objective; grad_scale->0 == JTV).
Nothing here runs unless a caller constructs a grad-prior (``prior`` includes
``gradnet``).
"""
import torch


def _robust_pos_scale(m, q=0.99, max_samples=1_000_000):
    """Robust positive scale (~99th pct) of the NON-NEGATIVE magnitude field m.

    The pooled gradient *magnitude* m >= 0, so a positive-quantile scale is valid
    here (unlike the signed gradient field -- that is why the spec forbids reusing
    ``_denoiser._standardize`` on the field, not on this magnitude). Standardising m
    to ~[0,1] matches the learned net's training distribution; the derived gain
    ``m_hat/m`` is scale-invariant so this does not change the analytic limit."""
    finite = m[torch.isfinite(m)]
    pos = finite[finite > 0]
    ref = pos if pos.numel() > 0 else finite.abs()
    if ref.numel() == 0:
        return m.new_tensor(1.0)
    if ref.numel() > max_samples:
        # float64 + clamp: for >~16.7M elems a float32 linspace endpoint rounds past
        # numel-1 -> out-of-bounds gather (hits the K=3 per-direction field, ~3x larger).
        idx = torch.linspace(0, ref.numel() - 1, max_samples, device=ref.device,
                             dtype=torch.float64).long().clamp_(max=ref.numel() - 1)
        ref = ref.flatten()[idx]
    return torch.quantile(ref.float(), q).clamp_min(1e-8)


# ----------------------------------------------------------------------------
# Backends: expose gain(m, rho, eps) -> per-voxel scalar gain (same shape as m).
# Analytic backends have is_learned = False and ignore standardisation/clamp.
# ----------------------------------------------------------------------------
class SoftThreshGain:
    """Analytic vectorial soft-threshold gain -- the EXACT classic-JTV z-update.

    ``gain(m, rho, eps) = (m - 1/rho)_+ / (m + eps)``. This is byte-identical to
    the analytic gain in ``_update.py``'s z-update; with ``grad_net='softthr'`` the
    merged-gradient prior reproduces ``prior='mtv'`` bit-for-bit (Phase 0 check).
    """
    is_learned = False
    name = 'softthr'

    def gain(self, m, rho, eps):
        # Bit-identical to the analytic expression in _update.py's z-update:
        #   ((m - one/rho).clamp_min(0)) / (m + tiny),  one = float32 tensor(1.0).
        # Build the numerator constant in m's dtype (float32) so 1/rho is computed in
        # float32 exactly as the original does -- this stays bit-exact even if rho is a
        # Python float (a bare python 1.0/rho would promote to float64 -> 1-ULP drift).
        one = m.new_tensor(1.0)
        return ((m - one / rho).clamp_min(0)) / (m + eps)


class CharbonnierGain:
    """Generalized analytic edge-preserving gain with an EXPLICIT energy (Phase 1).

    A Huber / pseudo-Charbonnier total-variation potential on the pooled gradient
    magnitude ``m`` (variant (a) of the spec -- a genuine *radial prox*, so ADMM
    keeps a monotone augmented-Lagrangian descent to a stationary point):

        Phi_a(m) = m - a/2        (m >= a)   -- TV-like (edge-preserving) above the knee
                 = m^2 / (2a)     (m <  a)   -- quadratic (gentle smoothing) below the knee

    The knee ``a = grad_scale / rho`` is tied to the ADMM knee 1/rho (dimensionless,
    so one setting works across the coarse-to-fine reg schedule and rho updates).
    ``Phi_a`` is convex and C^1, so the z-update is the EXACT prox
    ``prox_{Phi_a/rho}(m)/m`` -- a scalar gain (beta = 1/rho):

        s(m) = (m - beta) / m     (m >= a + beta)   -- == the classic JTV gain
             = a / (a + beta)      (m <  a + beta)   -- a flat sub-knee gain (no hard cut)

    As ``grad_scale -> 0`` the knee a -> 0 and this reduces to the classic JTV
    soft-threshold gain (and Phi -> m, the L1/TV energy): the param->default is JTV.
    Larger ``grad_scale`` keeps strong edges while smoothing weak gradients more
    gently than the hard L1 threshold (edge-preserving), with an explicit energy so
    the reported objective is a genuine, monotone-descending Lyapunov function.
    """
    is_learned = False
    has_energy = True
    name = 'charbonnier'

    def __init__(self, scale=1.0):
        self.scale = float(scale)  # grad_scale: knee a = scale / rho

    def gain(self, m, rho, eps):
        beta = m.new_tensor(1.0) / rho                 # 1/rho in m's dtype
        alpha = self.scale * beta                      # knee a
        thr = alpha + beta                             # magnitude where the two branches meet
        s_large = ((m - beta).clamp_min(0)) / (m + eps)    # == exact JTV gain above the knee
        s_small = alpha / (alpha + beta + eps)             # flat sub-knee gain (0 when scale=0)
        return torch.where(m >= thr, s_large, s_small * torch.ones_like(m))

    def energy(self, m, rho):
        """Per-voxel potential Phi_a(m): the explicit prior energy for the objective."""
        alpha = self.scale * (m.new_tensor(1.0) / rho)
        alpha_safe = alpha + (self.scale == 0.0)       # avoid 0-division in the unused branch
        lin = m - 0.5 * alpha
        quad = (m * m) / (2.0 * alpha_safe)
        return torch.where(m >= alpha, lin, quad)


class GradUNet3D:
    """Learned native-3D gain on the pooled gradient MAGNITUDE (variant (b), K=1).

    A bias-free 3D residual U-Net (``BiasFreeUNet3D``, the same architecture as the
    image-RED denoiser, but here fed the *single* pooled-magnitude channel) predicts
    the RESTORED magnitude ``m_hat`` from the merged, channel-pooled magnitude ``m``;
    the :class:`GradPrior` wrapper turns it into a per-voxel gain
    ``clamp(m_hat / (m + eps), *gain_clamp)`` and broadcasts it to every channel/
    direction. ONE 3D pass per ADMM iteration, C-INDEPENDENT (the net never sees the
    channel count -- only the pooled magnitude), no 2.5D.

    Trained on thick-slice-degraded -> SHARP IXI gradient magnitudes (the Track-1
    detail-filling paradigm moved to the gradient domain): the net learns to KEEP/
    restore coherent edge magnitudes and shrink noise-induced ones. Bias-free ⇒
    scale-equivariant, so with the ``m``-standardisation the derived gain is
    scale-invariant. This is a PnP gain (not a prox); clamp to [0,1] keeps it a
    non-expansive shrinkage, and the z-update line-search guard keeps the reported
    objective monotone (as the image-RED path does).
    """
    is_learned = True
    name = 'unet'

    def __init__(self, weights, device=None, base=48, depth=3, sigma=0.0,
                 patch=0, overlap=16):
        from ._denoiser_net import BiasFreeUNet3D  # lazy: only for this backend
        if not weights:
            raise ValueError("grad_net='unet' requires grad_weights (checkpoint path).")
        ckpt = torch.load(weights, map_location=device or 'cpu')
        a = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        # K = out channels (tail); in_ch = K + sigma_map -> recover sigma_map
        self.k = int(state['tail.weight'].shape[0])     # 1 (pooled magnitude) or 3 (per-direction)
        in_ch = int(state['head.weight'].shape[1])
        net = BiasFreeUNet3D(base=a.get('base', base), depth=a.get('depth', depth),
                             sigma_map=(in_ch > self.k), residual=True, context=False, k=self.k)
        net.load_state_dict(state)
        self.model = net.to(device).eval() if device is not None else net.eval()
        self.sigma = float(a.get('grad_sigma', sigma))  # standardised-units hint (0 = blind)
        self.patch = int(patch)
        self.overlap = int(overlap)

    @torch.no_grad()
    def _restore(self, m):
        """Predict the restored (sharp) magnitude field m_hat from m (>= 0).

        m is [D,H,W] (K=1 pooled magnitude, grad_apply='magnitude') or [K,D,H,W]
        (K=3 per-direction merged field, grad_apply='perdir'). One net pass either way."""
        scale = _robust_pos_scale(m)
        # -> [1, K, D, H, W]; a scalar field gets an explicit channel axis.
        x = (m / scale)[None, None] if m.dim() == 3 else (m / scale)[None]
        spatial = m.shape if m.dim() == 3 else m.shape[1:]
        if self.patch <= 0 or max(spatial) <= self.patch:
            out = self.model(x, self.sigma)[0]           # [K,D,H,W]
            out = out[0] if m.dim() == 3 else out        # scalar -> [D,H,W]
        else:                                            # overlap-tiled (scalar K=1 only)
            out = torch.zeros_like(m); wsum = torch.zeros_like(m)
            t, ov = self.patch, self.overlap
            step = t - ov
            rngs = [list(range(0, max(1, s - ov), step)) for s in m.shape]
            for oi in rngs[0]:
                for oj in rngs[1]:
                    for ok in rngs[2]:
                        i = min(oi, m.shape[0] - t) if m.shape[0] > t else 0
                        j = min(oj, m.shape[1] - t) if m.shape[1] > t else 0
                        k = min(ok, m.shape[2] - t) if m.shape[2] > t else 0
                        sl = (slice(i, i + t), slice(j, j + t), slice(k, k + t))
                        out[sl] += self.model((m[sl] / scale)[None, None], self.sigma)[0, 0]
                        wsum[sl] += 1.0
            out = out / wsum.clamp_min(1.0)
        return (out * scale).clamp_min(0.0)

    def gain(self, m, rho, eps):
        m_hat = self._restore(m)
        return m_hat / (m + eps)                          # wrapper clamps to gain_clamp


# ----------------------------------------------------------------------------
# Wrapper (cache + optional gain clamp)
# ----------------------------------------------------------------------------
class GradPrior:
    """Apply a merged-gradient ``backend`` gain to the pooled magnitude field.

    ``gain(m, rho, eps)`` returns a per-voxel gain of the same shape as ``m``,
    which the caller broadcasts to every channel/direction (the JTV broadcast,
    left verbatim in ``_update.py``).

    For analytic backends (``softthr``/``charbonnier``) the gain is returned
    untouched -- no standardisation, no clamp -- so identity gain stays == JTV.
    For learned backends the gain is clamped to ``gain_clamp`` (default [0, 1]) so
    the prior can only shrink/keep gradients (data-anchored, no hallucination).
    """
    def __init__(self, backend, sigma=0.05, gain_clamp=(0.0, 1.0), cache_size=8):
        self.backend = backend
        self.sigma = float(sigma)
        self.gain_clamp = tuple(gain_clamp) if gain_clamp is not None else None
        # True if the backend carries an explicit prior energy Phi (variant (a)/(b'));
        # then the reported objective uses Phi (monotone Lyapunov), else classic JTV.
        self.has_energy = bool(getattr(backend, 'has_energy', False))
        # Small cache keyed on (data_ptr, version, rho) -- mirrors the RED denoiser
        # cache. The z-update magnitude ``m`` (``tmp``) is mutated in place across
        # updates, so the tensor ``_version`` disambiguates stale entries.
        self._cache = {}
        self._cache_order = []
        self._cache_size = int(cache_size)

    @torch.no_grad()
    def gain(self, m, rho, eps=1e-7):
        if not getattr(self.backend, 'is_learned', False):
            # Analytic gain: pass through verbatim (bit-identical to classic JTV).
            return self.backend.gain(m, rho, eps)
        key = (m.data_ptr(), int(m._version), float(rho))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        g = self.backend.gain(m, rho, eps)
        if self.gain_clamp is not None:
            g = g.clamp(self.gain_clamp[0], self.gain_clamp[1])
        self._cache[key] = g
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return g

    def energy(self, m, rho):
        """Per-voxel prior potential Phi(m) for the reported objective.

        Only defined for explicit-energy backends (``has_energy``); the caller
        checks ``has_energy`` before summing this into the objective."""
        return self.backend.energy(m, rho)


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def make_grad_prior(sett):
    """Build a :class:`GradPrior` from UniRes ``settings`` (or None if inactive)."""
    name = getattr(sett, 'grad_net', 'softthr')
    if name == 'softthr':
        backend = SoftThreshGain()
    elif name == 'charbonnier':
        # Phase 1: generalized analytic (explicit-energy) edge-preserving gain.
        backend = CharbonnierGain(scale=getattr(sett, 'grad_scale', 1.0))
    elif name == 'unet':
        # Phase 2: learned native-3D magnitude restorer (bias-free 3D U-Net).
        backend = GradUNet3D(getattr(sett, 'grad_weights', None),
                             device=getattr(sett, 'device', None),
                             sigma=getattr(sett, 'grad_sigma', 0.0),
                             patch=getattr(sett, 'grad_patch', 0))
    elif name == 'dncnn':
        raise NotImplementedError(
            f"grad_net='{name}' is a Phase 2 learned backend (not yet implemented).")
    else:
        raise ValueError(
            f"Unknown grad_net '{name}' (expected 'softthr', 'charbonnier', "
            f"'unet' or 'dncnn').")
    return GradPrior(backend,
                     sigma=getattr(sett, 'grad_sigma', 0.05),
                     gain_clamp=getattr(sett, 'grad_gain_clamp', (0.0, 1.0)))
