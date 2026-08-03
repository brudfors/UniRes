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
                res = torch.as_tensor(res, dtype=x.dtype, device=x.device)
            # res may be a scalar (legacy isotropic) or a per-recon-axis vector -- either [3]
            # (shared across the batch) or [B,3] (directional conditioning, see
            # _thick_factor_axes). This net takes ONE conditioning channel, so reduce a
            # directional vector to its max (the worst-sampled axis); scalars pass through.
            if res.dim() == 2 and res.shape[-1] == 3:
                res = res.amax(dim=-1)                     # [B,3] -> [B]
            elif res.dim() == 1 and res.numel() == 3 and ref.shape[0] != 3:
                res = res.amax()                           # [3]   -> scalar
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


class BasicUNetRestore(nn.Module):
    """MONAI BasicUNet as a BLIND restoration net — faithful to the reference SynthSeg config
    (github.com/brudfors/synthseg-monai utils_synthseg.py): features=(24,48,96,192,384,24)
    [5 down-levels, 384 bottleneck], LeakyReLU(0.2), InstanceNorm (BasicUNet default), deconv
    upsampling. NO regularisation (no dropout/weight-decay) — SynthSeg's infinite random-contrast
    data makes over-fitting impossible by construction, so protecting against it only wastes
    capacity. Predicts the clean image directly from the degraded input.

    With ``res_ch>0`` the net is THICKNESS-CONDITIONED: `res_ch` extra input channels carry the
    (constant, per-volume) normalised slice-thickness so the model restores CONSERVATIVELY on very
    thick slices instead of over-restoring — a blind net can't tell 2mm from 8mm and hallucinates
    in the forward operator's null space at extreme thickness. `res_ch=3` is DIRECTIONAL: one
    channel per recon voxel axis (the per-axis low/high resolution ratio), so the model is TOLD
    which direction is through-plane (e.g. 0.5x0.5x6.5mm -> axes [0,0,0.8]) rather than inferring
    it from the anisotropic-blur signature; it also handles per-axis / crossed anisotropy. `res_ch=1`
    is the legacy isotropic scalar (max ratio). This keeps the reference BasicUNet architecture
    identical (same features/act/norm); only in_channels 1->1+res_ch. `forward` accepts+ignores
    sigma/context so it drops into the same training loop / denoiser backend as BiasFreeUNet3D."""
    def __init__(self, in_ch=1, out_ch=1, features=(24, 48, 96, 192, 384, 24), res_cond=False, res_ch=None):
        super().__init__()
        from monai.networks.nets import BasicUNet  # lazy: only for this net
        # res_ch: number of thickness-conditioning channels. Default: 3 (directional) when
        # res_cond, else 0. An explicit res_ch (e.g. inferred from a checkpoint) overrides.
        self.res_ch = int(res_ch) if res_ch is not None else (3 if res_cond else 0)
        eff_in = in_ch + self.res_ch
        self.net = BasicUNet(
            spatial_dims=3, in_channels=eff_in, out_channels=out_ch, features=tuple(features),
            act=("LeakyReLU", {"negative_slope": 0.2, "inplace": True}), upsample="deconv")
        self.k = out_ch

    def forward(self, x, sigma=None, context=None, res=None):
        if self.res_ch > 0:
            B, sp = x.shape[0], x.shape[2:]
            if res is None:
                rv = x.new_zeros(B, self.res_ch)
            else:
                rv = torch.as_tensor(res, dtype=x.dtype, device=x.device)
                if rv.dim() == 0:                       # scalar -> all axes/samples
                    rv = rv.reshape(1, 1).expand(B, self.res_ch)
                elif rv.dim() == 1:
                    if rv.numel() == self.res_ch:        # per-axis vector shared across batch
                        rv = rv.reshape(1, self.res_ch).expand(B, self.res_ch)
                    else:                                # [B] scalar-per-sample -> broadcast axes
                        rv = rv.reshape(B, 1).expand(B, self.res_ch)
                else:                                    # [B, res_ch]
                    rv = rv.reshape(B, self.res_ch)
            r = rv.reshape(B, self.res_ch, 1, 1, 1).expand(B, self.res_ch, *sp)
            x = torch.cat([x, r], dim=1)
        return self.net(x)
