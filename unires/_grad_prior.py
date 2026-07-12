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

Phase 0 implements only the ``softthr`` analytic gain, which reproduces classic
MTV bit-for-bit. Nothing here runs unless a caller constructs a grad-prior (i.e.
``prior`` includes ``gradnet``).
"""
import torch


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


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def make_grad_prior(sett):
    """Build a :class:`GradPrior` from UniRes ``settings`` (or None if inactive)."""
    name = getattr(sett, 'grad_net', 'softthr')
    if name == 'softthr':
        backend = SoftThreshGain()
    elif name in ('charbonnier',):
        # Phase 1: generalized analytic (explicit-energy) gain.
        raise NotImplementedError(
            f"grad_net='{name}' is a Phase 1 backend (not yet implemented).")
    elif name in ('unet', 'dncnn'):
        # Phase 2: learned native-3D magnitude denoiser (deepinv backend).
        raise NotImplementedError(
            f"grad_net='{name}' is a Phase 2 learned backend (not yet implemented).")
    else:
        raise ValueError(
            f"Unknown grad_net '{name}' (expected 'softthr', 'charbonnier', "
            f"'unet' or 'dncnn').")
    return GradPrior(backend,
                     sigma=getattr(sett, 'grad_sigma', 0.05),
                     gain_clamp=getattr(sett, 'grad_gain_clamp', (0.0, 1.0)))
