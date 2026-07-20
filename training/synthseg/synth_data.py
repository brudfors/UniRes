"""SynthSeg-style GPU generative model for a CONTRAST-AGNOSTIC restoration prior.

Instead of training on real MRI, we SYNTHESIZE sharp random-contrast images from a
small set of anatomical LABEL MAPS (the 20 FreeSurfer-style SynthSeg training maps).
Each step draws a fresh per-label Gaussian intensity model => a brand-new "contrast",
so a net trained on this vast synthetic contrast distribution becomes contrast-agnostic.

This is a fast, vectorised GPU reimplementation of the SynthSeg GMM generator (the
cornucopia SynthFromLabelTransform is ~4 s/sample on GPU -- 2000x too slow to run per
training step; this is ~2 ms/sample and identical in spirit).

Pipeline per step:
  label patch (random affine + flip + crop of a label map)
    -> per-label random Gaussian intensities (random contrast) + random smoothing  [GMM]
    -> normalise to a non-negative [~0,1] scale (like an IXI patch)
  => the SHARP random-contrast image (TARGET).
The INPUT is produced downstream by training/data.py::augment_noise_gpu (thick-slice
degrade U(1,8) mm + bias/gamma/noise), exactly the degradation the IXI-trained prior saw.
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F


def _load_label_maps(dir_labels, device):
    """Load all label maps, relabel each to contiguous 0..K-1, cache on `device`.
    Returns list of (label[D,H,W] uint8, K) and per-map foreground bbox for cropping."""
    paths = sorted(glob.glob(os.path.join(dir_labels, "*.nii.gz")))
    if not paths:
        raise FileNotFoundError(f"No label maps (*.nii.gz) in {dir_labels}")
    import nibabel as nib
    maps = []
    for p in paths:
        arr = np.asarray(nib.load(p).dataobj)
        t = torch.as_tensor(arr, device=device).long()
        uniq = torch.unique(t)
        K = int(len(uniq))
        remap = torch.zeros(int(t.max()) + 1, dtype=torch.long, device=device)
        remap[uniq] = torch.arange(K, device=device)
        t = remap[t].to(torch.uint8)  # 0..K-1, small
        fg = (t > 0).nonzero(as_tuple=False)
        lo = fg.min(0).values
        hi = fg.max(0).values
        maps.append((t, K, lo, hi))
    return maps


def _rand_affine_theta(batch, device, max_rot=15.0, max_shear=0.1, zoom=0.15, do_prob=0.8):
    """BATCHED random 3D affine matrices [B,3,4] (rotation+zoom+shear+random flips folded
    in as sign of the diagonal). Vectorised across the batch -> feeds a SINGLE affine_grid
    + grid_sample instead of one per sample (fewer CUDA launches)."""
    B = batch
    rot = (torch.rand(B, 3, device=device) * 2 - 1) * (max_rot * np.pi / 180.0)
    cs, sn = torch.cos(rot), torch.sin(rot)
    z = torch.zeros(B, device=device)
    o = torch.ones(B, device=device)
    Rx = torch.stack([o, z, z, z, cs[:, 0], -sn[:, 0], z, sn[:, 0], cs[:, 0]], 1).view(B, 3, 3)
    Ry = torch.stack([cs[:, 1], z, sn[:, 1], z, o, z, -sn[:, 1], z, cs[:, 1]], 1).view(B, 3, 3)
    Rz = torch.stack([cs[:, 2], -sn[:, 2], z, sn[:, 2], cs[:, 2], z, z, z, o], 1).view(B, 3, 3)
    A = Rz @ Ry @ Rx
    # anisotropic zoom + random flips (sign)
    zm = torch.exp((torch.rand(B, 3, device=device) * 2 - 1) * zoom)
    flip = torch.where(torch.rand(B, 3, device=device) < 0.5, -o[:, None].expand(B, 3), o[:, None].expand(B, 3))
    A = A * (zm * flip)[:, None, :]
    # shear
    sh = (torch.rand(B, 3, device=device) * 2 - 1) * max_shear
    S = torch.eye(3, device=device).expand(B, 3, 3).clone()
    S[:, 0, 1] = sh[:, 0]; S[:, 0, 2] = sh[:, 1]; S[:, 1, 2] = sh[:, 2]
    A = A @ S
    # apply the augmentation only to a random subset; others get identity
    keep = (torch.rand(B, 1, 1, device=device) < do_prob).float()
    eye = torch.eye(3, device=device).expand(B, 3, 3)
    A = keep * A + (1 - keep) * eye
    theta = torch.zeros(B, 3, 4, device=device)
    theta[:, :, :3] = A
    return theta


class SynthLabelBank:
    """Draws batches of SHARP random-contrast image patches on the GPU."""

    def __init__(self, dir_labels, device, patch=96, gmm_fwhm=(0.3, 1.6),
                 affine=True, deform_prob=0.8):
        self.device = device
        self.patch = int(patch)
        self.gmm_fwhm = gmm_fwhm
        self.affine = affine
        self.deform_prob = deform_prob
        self.maps = _load_label_maps(dir_labels, device)

    # --- geometry: crop a foreground-centred raw patch (cheap indexing, no kernels) ---
    def _raw_crop(self):
        t, K, lo, hi = self.maps[int(torch.randint(len(self.maps), (1,)).item())]
        P = self.patch
        c = [int(torch.randint(int(lo[i]), int(hi[i]) + 1, (1,), device=self.device)[0])
             for i in range(3)]
        s = [min(max(0, c[i] - P // 2), t.shape[i] - P) for i in range(3)]
        return t[s[0]:s[0] + P, s[1]:s[1] + P, s[2]:s[2] + P].long()

    @staticmethod
    def _gauss3d(x, sigma):
        if sigma <= 0:
            return x
        r = max(1, int(round(3 * sigma)))
        k = torch.exp(-(torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2)
                      / (2 * sigma * sigma))
        k = k / k.sum()
        for ax in range(3):
            shp = [1, 1, 1, 1, 1]; shp[2 + ax] = k.numel()
            pad = [0] * 6; pad[2 * (2 - ax)] = r; pad[2 * (2 - ax) + 1] = r
            x = F.conv3d(F.pad(x, pad, mode="reflect"), k.view(shp))
        return x

    def sample(self, batch):
        """Return a batch of SHARP random-contrast image patches [B,1,P,P,P] on GPU,
        non-negative, roughly [0,1] scaled (like an IXI clean patch).

        Fully batched: the only per-sample Python work is the cheap crop-index (a view,
        no heavy kernels); geometry (affine+flip), GMM, smoothing and normalisation are
        each a SINGLE batched op -> a large cut in per-step CUDA-launch count vs a
        per-sample loop, so the process keeps progressing under CPU starvation."""
        P = self.patch
        # cheap per-sample raw crops -> one stacked tensor
        raw = torch.stack([self._raw_crop() for _ in range(batch)])[:, None].float()  # [B,1,P,P,P]
        # ONE batched affine warp (rotation+zoom+shear+flips), nearest -> integer labels
        if self.affine:
            theta = _rand_affine_theta(batch, self.device, do_prob=self.deform_prob)
            grid = F.affine_grid(theta, (batch, 1, P, P, P), align_corners=False)
            raw = F.grid_sample(raw, grid, mode="nearest", padding_mode="zeros",
                                align_corners=False)
        labs = raw.long()                                  # [B,1,P,P,P]
        Kmax = int(labs.max().item()) + 1
        # per-label random Gaussian intensities (fresh random CONTRAST per sample) -- one gather
        mean = torch.rand(batch, Kmax, device=self.device)          # U(0,1)
        std = torch.rand(batch, Kmax, device=self.device) * 0.25     # within-tissue variance
        mean[:, 0] *= 0.15                                           # background mostly dark
        flat = labs.reshape(batch, -1)
        m = torch.gather(mean, 1, flat).reshape(batch, 1, P, P, P)
        s = torch.gather(std, 1, flat).reshape(batch, 1, P, P, P)
        img = (m + s * torch.randn_like(m)).float()
        # single batched smoothing (partial-volume / PSF of the "acquisition")
        lo, hi = self.gmm_fwhm
        img = self._gauss3d(img, float(torch.empty(1).uniform_(lo, hi)))
        img = img.clamp_min(0.0)
        # normalise each sample to ~[0,1] via a robust max (like IXI patch scaling)
        hi = torch.quantile(img.reshape(batch, -1), 0.99, dim=1).clamp_min(1e-5)
        img = (img / hi.view(batch, 1, 1, 1, 1)).clamp(0.0, 2.0)
        return img


_LG06, _LG17 = float(np.log(0.6)), float(np.log(1.7))
_LG07, _LG14 = float(np.log(0.7)), float(np.log(1.4))


def batched_degrade(x, _thick_slice_degrade, _gauss_blur3d, sigma_min=0.0, sigma_max=0.05,
                    rician_prob=0.5, blur_max=1.0, thick_max=8.0):
    """FULLY-BATCHED clean->(input,target) degradation for the synth restoration prior.

    Behaviourally mirrors training/data.py::augment_noise_gpu (inversion/gamma/bias/scale
    define the shared clean appearance; thick-slice + iso-blur + Gaussian/Rician noise make
    the input), but WITHOUT the per-sample Python loop: it draws ONE random axis+factor for
    the whole batch and calls the already-batched `_thick_slice_degrade` once, so per-step
    CUDA-launch count stays low and the process progresses even under CPU starvation. Fresh
    axis/factor each STEP still spans the full 1..thick_max mm range over training.

    `_thick_slice_degrade`/`_gauss_blur3d` are passed in (imported read-only from data.py).
    Returns (noisy, target, sigma[B], res[B])."""
    B, dev = x.shape[0], x.device
    x = x.clamp_min(0.0)
    # contrast inversion (per-sample)
    inv = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    x = inv * (1.0 - x).clamp_min(0.0) + (1.0 - inv) * x
    # gamma (per-sample, prob 0.7)
    do_g = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.7).float()
    gamma = torch.exp(torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(_LG06, _LG17))
    x = x.clamp_min(0.0) ** (do_g * gamma + (1.0 - do_g))
    # smooth multiplicative bias field (per-sample, prob 0.5) -- already batched interpolate
    do_b = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    field = F.interpolate(torch.randn(B, 1, 4, 4, 4, device=dev), size=x.shape[2:],
                          mode="trilinear", align_corners=False)
    field = field / field.abs().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    x = x * (1.0 + do_b * 0.3 * field)
    # global scale (per-sample, prob 0.5)
    do_s = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    scl = torch.exp(torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(_LG07, _LG14))
    x = (x * (do_s * scl + (1.0 - do_s))).clamp_min(0.0)
    target = x
    # thick-slice degradation: ONE axis + factor for the whole batch (single batched call)
    sigma = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(sigma_min, sigma_max)
    if thick_max > 0:
        axis = int(torch.randint(0, 3, (1,)).item())
        factor = float(torch.empty(1).uniform_(1.0, thick_max).item())
        res = x.new_full((B,), (factor - 1.0) / 7.0)
        deg = _thick_slice_degrade(x, factor, axis)
        if blur_max > 0:
            deg = _gauss_blur3d(deg, float(torch.empty(1).uniform_(0.0, blur_max).item()))
    else:
        deg, res = x, torch.zeros(B, device=dev)
    # Gaussian/Rician noise (per-sample choice), batched
    ric = (torch.rand(B, 1, 1, 1, 1, device=dev) < rician_prob).float()
    n1, n2 = torch.randn_like(deg), torch.randn_like(deg)
    gauss = deg + sigma * n1
    rician = torch.sqrt((deg + sigma * n1) ** 2 + (sigma * n2) ** 2)
    noisy = ric * rician + (1.0 - ric) * gauss
    return noisy, target, sigma.reshape(-1), res
