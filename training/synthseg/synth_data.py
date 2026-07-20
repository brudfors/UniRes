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


def _rand_affine_grid(patch, device, max_rot=15.0, max_shear=0.1, zoom=0.15):
    """Random 3D affine sampling grid for a cube of side `patch` (for grid_sample)."""
    rot = (torch.rand(3, device=device) * 2 - 1) * (max_rot * np.pi / 180.0)
    cs = torch.cos(rot)
    sn = torch.sin(rot)
    Rx = torch.tensor([[1, 0, 0], [0, cs[0], -sn[0]], [0, sn[0], cs[0]]], device=device)
    Ry = torch.tensor([[cs[1], 0, sn[1]], [0, 1, 0], [-sn[1], 0, cs[1]]], device=device)
    Rz = torch.tensor([[cs[2], -sn[2], 0], [sn[2], cs[2], 0], [0, 0, 1]], device=device)
    A = Rz @ Ry @ Rx
    # anisotropic zoom + shear
    z = torch.exp((torch.rand(3, device=device) * 2 - 1) * zoom)
    A = A * z[None, :]
    sh = (torch.rand(3, device=device) * 2 - 1) * max_shear
    S = torch.eye(3, device=device)
    S[0, 1] = sh[0]; S[0, 2] = sh[1]; S[1, 2] = sh[2]
    A = A @ S
    theta = torch.zeros(1, 3, 4, device=device)
    theta[0, :, :3] = A
    grid = F.affine_grid(theta, (1, 1, patch, patch, patch), align_corners=False)
    return grid


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

    # --- geometry: crop a foreground-centred patch, optional affine warp -----------
    def _label_patch(self):
        t, K, lo, hi = self.maps[int(torch.randint(len(self.maps), (1,)).item())]
        P = self.patch
        # sample a crop centre inside the foreground bbox (so the patch has anatomy)
        c = torch.stack([torch.randint(int(lo[i]), int(hi[i]) + 1, (1,), device=self.device)[0]
                         for i in range(3)])
        s = [int((c[i] - P // 2).clamp(0, t.shape[i] - P)) for i in range(3)]
        lp = t[s[0]:s[0] + P, s[1]:s[1] + P, s[2]:s[2] + P].long()
        # random flips
        for ax in range(3):
            if torch.rand(1).item() < 0.5:
                lp = lp.flip(ax)
        # random affine warp (nearest, so labels stay integer)
        if self.affine and torch.rand(1).item() < self.deform_prob:
            grid = _rand_affine_grid(P, self.device)
            lp = F.grid_sample(lp[None, None].float(), grid, mode="nearest",
                               padding_mode="zeros", align_corners=False)[0, 0].long()
        return lp, K

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
        non-negative, roughly [0,1] scaled (like an IXI clean patch)."""
        P = self.patch
        labs = torch.empty(batch, 1, P, P, P, dtype=torch.long, device=self.device)
        Ks = []
        for b in range(batch):
            lp, K = self._label_patch()
            labs[b, 0] = lp
            Ks.append(K)
        Kmax = max(Ks)
        # per-label random Gaussian intensities (fresh random CONTRAST per sample)
        mean = torch.rand(batch, Kmax, device=self.device)          # U(0,1)
        std = torch.rand(batch, Kmax, device=self.device) * 0.25     # within-tissue variance
        # background label 0: mostly dark
        mean[:, 0] *= 0.15
        flat = labs.reshape(batch, -1)
        m = torch.gather(mean, 1, flat).reshape(batch, 1, P, P, P)
        s = torch.gather(std, 1, flat).reshape(batch, 1, P, P, P)
        img = (m + s * torch.randn_like(m)).float()
        # random per-sample smoothing (models partial-volume / PSF of the "acquisition")
        lo, hi = self.gmm_fwhm
        img = self._gauss3d(img, float(torch.empty(1).uniform_(lo, hi)))
        img = img.clamp_min(0.0)
        # normalise each sample to ~[0,1] via a robust max (like IXI patch scaling)
        hi = torch.quantile(img.reshape(batch, -1), 0.99, dim=1).clamp_min(1e-5)
        img = (img / hi.view(batch, 1, 1, 1, 1)).clamp(0.0, 2.0)
        return img
