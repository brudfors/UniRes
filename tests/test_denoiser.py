"""Phase-0 tests for the denoiser-as-prior plumbing (unires/_denoiser.py).

Uses the analytic Gaussian backend on CPU, so it needs no GPU and no learned
weights. Run with pytest, or standalone: `python tests/test_denoiser.py`.
"""
import torch

from unires._denoiser import (
    Denoiser, GaussianBackend, make_denoiser, _standardize, _unstandardize)
from unires.struct import settings


def _phantom(dim=(24, 28, 26), seed=0):
    """A smooth low-frequency volume + noise (non-negative, MR-like)."""
    g = torch.Generator().manual_seed(seed)
    x, y, z = torch.meshgrid(
        [torch.linspace(0, 3.14159, d) for d in dim], indexing='ij')
    base = (1.0 + torch.sin(x) * torch.cos(y) * torch.sin(z)).clamp_min(0.0)
    noise = 0.15 * torch.randn(dim, generator=g)
    return base, (base + noise).clamp_min(0.0)


def _roughness(vol):
    """Mean abs discrete Laplacian -- a simple high-frequency energy measure."""
    lap = (vol[2:, 1:-1, 1:-1] + vol[:-2, 1:-1, 1:-1]
           + vol[1:-1, 2:, 1:-1] + vol[1:-1, :-2, 1:-1]
           + vol[1:-1, 1:-1, 2:] + vol[1:-1, 1:-1, :-2]
           - 6.0 * vol[1:-1, 1:-1, 1:-1])
    return lap.abs().mean().item()


def test_standardize_roundtrip():
    vol = _phantom()[1]
    v, scale = _standardize(vol)
    assert torch.allclose(_unstandardize(v, scale), vol, atol=1e-5)
    # standardised volume is ~[0,1]
    assert v.max() <= 5.0 and v.min() >= 0.0


def test_denoise_preserves_shape_dtype_device():
    vol = _phantom()[1]
    den = Denoiser(GaussianBackend(), sigma=0.05, batch=8)
    out = den.denoise(vol)
    assert out.shape == vol.shape
    assert out.dtype == vol.dtype
    assert out.device == vol.device


def test_denoise_reduces_noise():
    base, noisy = _phantom()
    den = Denoiser(GaussianBackend(), sigma=0.05, batch=8)
    out = den.denoise(noisy)
    mse_before = torch.mean((noisy - base) ** 2).item()
    mse_after = torch.mean((out - base) ** 2).item()
    assert mse_after < mse_before, (mse_after, mse_before)


def test_roughness_monotone_in_sigma():
    noisy = _phantom()[1]
    den = Denoiser(GaussianBackend(), batch=8)
    rough = [_roughness(den.denoise(noisy, sigma=s)) for s in (0.02, 0.1, 0.3)]
    # larger sigma -> more smoothing -> less high-frequency energy
    assert rough[0] > rough[1] > rough[2], rough


def test_intensity_scale_equivariance():
    """Standardisation makes the denoiser ~equivariant to arbitrary MR scaling."""
    _, noisy = _phantom()
    den = Denoiser(GaussianBackend(), sigma=0.08, batch=8)
    out1 = den.denoise(noisy)
    out2 = den.denoise(noisy * 1000.0)
    assert torch.allclose(out2, out1 * 1000.0, rtol=1e-3, atol=1e-2)


def test_factory_and_defaults_preserve_master():
    s = settings()
    # Defaults must reproduce classic behaviour: no DL prior active.
    assert s.prior == 'mtv'
    assert s.red_mu == 0.0
    s.device = 'cpu'
    den = make_denoiser(s)
    out = den.denoise(_phantom()[1])
    assert out.shape == (24, 28, 26)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
