"""Unit tests for the bias-free 3D denoiser (correctness of the generalization
properties). Run: python training/test_model.py"""
import torch
from model import BiasFreeUNet3D


def test_shapes():
    net = BiasFreeUNet3D(base=16, depth=3, n_res=1).eval()
    for shp in [(1, 1, 32, 32, 32), (2, 1, 40, 48, 56), (1, 1, 31, 33, 29)]:
        x = torch.randn(shp)
        y = net(x, sigma=0.05)
        assert y.shape == x.shape, (shp, y.shape)
    print("A. shapes (incl. non-power-of-2): PASS")


def test_bias_free_zero():
    # A bias-free net maps zero input to EXACTLY zero (no additive constants anywhere).
    net = BiasFreeUNet3D(base=16, depth=2, n_res=2, residual=False).eval()
    with torch.no_grad():
        z = net(torch.zeros(1, 1, 32, 32, 32), sigma=0.0)
    m = z.abs().max().item()
    assert m < 1e-6, f"bias-free zero test failed: max|net(0)|={m:.2e}"
    print(f"B. bias-free zero (max|net(0)|={m:.1e}): PASS")


def test_scale_equivariance():
    # Bias-free + ReLU => positively homogeneous: net(a*x, a*sigma) == a*net(x, sigma).
    net = BiasFreeUNet3D(base=16, depth=2, n_res=2, residual=False).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)
    sig = torch.rand(1) * 0.1
    a = 3.7
    with torch.no_grad():
        y1 = net(a * x, sigma=a * sig)
        y2 = a * net(x, sigma=sig)
    rel = (y1 - y2).abs().max() / y2.abs().max().clamp_min(1e-8)
    assert rel < 1e-4, f"scale-equivariance rel err {rel:.2e}"
    print(f"C. scale-equivariance D(a*x,a*s)=a*D(x,s) (rel={rel:.1e}): PASS")


def test_paramcount_and_biasfree_flag():
    net = BiasFreeUNet3D()
    n = sum(p.numel() for p in net.parameters())
    n_bias = sum(1 for name, _ in net.named_parameters() if 'bias' in name)
    assert n_bias == 0, f"found {n_bias} bias params -> not bias-free"
    print(f"D. no bias params (params={n/1e6:.2f}M): PASS")


if __name__ == '__main__':
    test_shapes()
    test_bias_free_zero()
    test_scale_equivariance()
    test_paramcount_and_biasfree_flag()
    print("all model tests passed")
