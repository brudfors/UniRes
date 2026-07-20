"""Bias-free 3D denoiser for the UniRes RED prior (Phase 3).

Design goals (see the RED next-steps notes):
  * GENERAL across any MRI -- BIAS-FREE network (no additive bias / no BatchNorm),
    which is exactly scale-equivariant: D(a*x) = a*D(x) for a > 0. Combined with the
    per-volume intensity standardisation in ``unires/_denoiser.py`` this gives
    intensity-scale invariance AND graceful extrapolation to noise levels outside
    the training range (Mohan et al. 2020, "Robust and Interpretable Blind Image
    Denoising via Bias-Free Networks").
  * BLIND but CONTROLLABLE -- an optional noise-level map channel (FFDNet/DRUNet
    style) lets the RED solver dial denoiser strength via ``sigma`` while the image
    path stays bias-free.
  * SINGLE-CHANNEL (1 volume in -> 1 volume out), applied per channel by the solver,
    so it works for ANY number of input channels / contrasts (no fixed C).
  * RESIDUAL -- predicts the noise; the denoised volume is ``x - net(x)``.

A compact 3D residual U-Net (encoder/decoder with strided-conv down / trilinear up),
all convolutions bias-free, ReLU activations (scale-equivariant for a > 0), no norm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(ci, co, k=3, s=1):
    # bias-free 3x3x3 conv (no additive bias -> preserves scale-equivariance)
    return nn.Conv3d(ci, co, k, stride=s, padding=k // 2, bias=False)


class _ResBlock(nn.Module):
    """Bias-free residual block: conv-relu-conv + skip (all linear/ReLU, no bias)."""
    def __init__(self, ch):
        super().__init__()
        self.c1 = _conv(ch, ch)
        self.c2 = _conv(ch, ch)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class BiasFreeUNet3D(nn.Module):
    """Bias-free 3D residual U-Net denoiser (single-channel, sigma-conditioned).

    Args:
        base (int): channel width at the finest scale.
        depth (int): number of down/up levels.
        n_res (int): residual blocks per level.
        sigma_map (bool): if True, take an extra noise-level-map input channel so
            the RED solver can control strength via sigma. Kept bias-free.
        residual (bool): predict noise (denoised = x - net(x)) if True.
    """
    def __init__(self, base=48, depth=3, n_res=2, sigma_map=True, residual=True,
                 context=False, k=1, res_map=False):
        super().__init__()
        self.sigma_map = sigma_map
        self.residual = residual
        self.context = context  # extra input channel: aggregate of OTHER channels' edges
        self.res_map = res_map   # extra input channel: normalised SLICE-THICKNESS (per-direction
        #                          resolution) so the net restores less aggressively where the input
        #                          is severely under-resolved (degradation-conditioned / no hallucination).
        self.k = int(k)          # signal channels in/out (1 = image/scalar-magnitude; K>1 = per-direction merged field)
        in_ch = self.k + int(sigma_map) + int(context) + int(res_map)
        self.head = _conv(in_ch, base)
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(depth):
            self.enc.append(nn.Sequential(*[_ResBlock(chs[i]) for _ in range(n_res)]))
            self.down.append(_conv(chs[i], chs[i + 1], s=2))     # strided-conv down
        self.mid = nn.Sequential(*[_ResBlock(chs[depth]) for _ in range(n_res)])
        self.up = nn.ModuleList()
        self.reduce = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(_conv(chs[i + 1], chs[i]))            # 1-conv after trilinear up
            self.reduce.append(_conv(2 * chs[i], chs[i]))        # fuse skip
            self.dec.append(nn.Sequential(*[_ResBlock(chs[i]) for _ in range(n_res)]))
        self.tail = _conv(base, self.k)

    def forward(self, x, sigma=None, context=None, res=None):
        inp = x
        feats = [x]
        ref = x[:, :1] if self.k > 1 else x
        if self.sigma_map:
            if sigma is None:
                sigma = x.new_zeros(())
            if not torch.is_tensor(sigma):
                sigma = x.new_tensor(float(sigma))
            # one sigma-map channel (broadcast to the k signal channels if k>1)
            smap = sigma.expand_as(ref) if sigma.dim() == 0 else sigma.view(-1, 1, 1, 1, 1).expand_as(ref)
            feats.append(smap)
        if self.context:
            # cross-channel guidance: aggregate edge map of OTHER channels (bias-free-
            # safe: it's a linear function of the inputs). Zeros if not provided.
            feats.append(context if context is not None else torch.zeros_like(x))
        if self.res_map:
            # slice-thickness / resolution conditioning (normalised in [0,1]): 0 = isotropic,
            # higher = thicker. Lets the net dial back its restoration where detail is
            # unrecoverable. Zeros (isotropic) if not provided -> backward compatible.
            if res is None:
                res = x.new_zeros(())
            if not torch.is_tensor(res):
                res = x.new_tensor(float(res))
            rmap = res.expand_as(ref) if res.dim() == 0 else res.view(-1, 1, 1, 1, 1).expand_as(ref)
            feats.append(rmap)
        h = torch.cat(feats, dim=1) if len(feats) > 1 else x
        h = self.head(h)
        skips = []
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, reduce, dec, skip in zip(self.up, self.reduce, self.dec, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
            h = up(h)
            h = reduce(torch.cat([h, skip], dim=1))
            h = dec(h)
        out = self.tail(h)
        return inp - out if self.residual else out
