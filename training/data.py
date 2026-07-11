"""IXI patch dataset for training the bias-free 3D MRI denoiser (Phase 3).

Pools all contrasts (T1/T2/PD) and applies aggressive intensity/contrast
augmentation so the learned prior is CONTRAST-AGNOSTIC (generalizes to any MRI,
not just the training contrasts) -- this is the fix for DRUNet's PD collapse.

Per sample it returns ``(noisy, target, sigma_map)`` single-channel 3D patches:
  * per-volume robust intensity standardization (~[0,1]), matching the inference
    standardization in ``unires/_denoiser.py``;
  * foreground-weighted random 3D patch (avoid training mostly on air);
  * augmentation: flips, 90deg rotations, gamma, contrast inversion, smooth bias
    field, mild global scale;
  * noise: Gaussian and/or Rician (magnitude MRI) at sigma ~ U(sigma_min, sigma_max);
  * target: the clean patch (supervised) OR a SECOND independent noise draw
    (simulated Noise2Noise -- no clean-target assumption, honest about IXI's own noise).

TRAIN on IXI, TEST on BrainWeb (benchmarks/) -> different datasets, no leakage.
"""
import os
import glob
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CONTRASTS = ('T1', 'T2', 'PD')


def list_ixi(root, contrasts=CONTRASTS):
    """All .nii.gz volume paths under root/{contrast}/ , pooled across contrasts."""
    paths = []
    for c in contrasts:
        paths += sorted(glob.glob(os.path.join(root, c, '*.nii.gz')))
    if not paths:
        raise FileNotFoundError(f"No IXI volumes under {root}/{{{','.join(contrasts)}}}")
    return paths


def _robust_scale(vol, q=0.99):
    """~99th percentile of positive voxels (matches unires/_denoiser._robust_scale)."""
    pos = vol[vol > 0]
    if pos.numel() == 0:
        return vol.new_tensor(1.0)
    return torch.clamp(torch.quantile(pos.float(), q), min=1e-6)


def _load_standardized(path):
    """Load a volume as a standardized float32 torch tensor (~[0,1])."""
    import nibabel as nib
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    vol = torch.from_numpy(arr)
    vol = torch.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = vol.clamp_min(0.0)                       # MR magnitude is non-negative
    return vol / _robust_scale(vol)


def _smooth_bias_field(shape, device, strength=0.3, ctrl=4):
    """A smooth positive multiplicative field ~1+/-strength (coil inhomogeneity)."""
    low = torch.randn(1, 1, ctrl, ctrl, ctrl, device=device)
    field = torch.nn.functional.interpolate(
        low, size=tuple(shape), mode='trilinear', align_corners=False)[0, 0]
    field = field / field.abs().max().clamp_min(1e-6)
    return 1.0 + strength * field


class IXIPatches(Dataset):
    def __init__(self, root, patch=96, patches_per_epoch=20000, contrasts=CONTRASTS,
                 sigma_min=0.0, sigma_max=0.15, rician_prob=0.5, n2n=True,
                 augment=True, cache_volumes=64, fg_frac=0.9, p_new=0.12, seed=0,
                 gpu_aug=False):
        # gpu_aug: if True, __getitem__ returns ONLY a cheap clean patch (crop + flips/
        # rot90) and the heavy augmentation + noise is done batched on the GPU in the
        # training loop (see augment_noise_gpu). This keeps per-sample CPU cost ~0 so
        # num_workers=0 is fast -> no multiprocessing DataLoader (which livelocked on
        # this shared, heavily-loaded pod).
        self.gpu_aug = bool(gpu_aug)
        self.paths = list_ixi(root, contrasts)
        self.patch = int(patch)
        self.length = int(patches_per_epoch)
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.rician_prob = float(rician_prob)
        self.n2n = bool(n2n)
        self.augment = bool(augment)
        self.fg_frac = float(fg_frac)
        # p_new: probability of loading a NEW (uncached) volume; otherwise reuse a
        # cached one. Reuse amortizes the (expensive) gzip .nii.gz decompression over
        # many patches -> the loader is cache-bound, not decompress-bound.
        self.p_new = float(p_new)
        self._cache = OrderedDict()
        self._cache_size = int(cache_volumes)
        self._base_seed = int(seed)
        self._rng = None  # lazily created per worker

    def __len__(self):
        return self.length

    @staticmethod
    def _foreground_bbox(vol, thr=0.05):
        """Cheap per-axis foreground bounding box (min/max index where any signal),
        via reductions -- computed ONCE per volume (vs an O(vol) nonzero per patch)."""
        m = vol > thr
        bb = []
        for d in range(3):
            proj = m.any(dim=tuple(a for a in range(3) if a != d))
            nz = torch.nonzero(proj, as_tuple=False)
            if nz.numel():
                bb.append((int(nz[0]), int(nz[-1]) + 1))
            else:
                bb.append((0, vol.shape[d]))
        return bb

    def _get_volume(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        vol = _load_standardized(self.paths[idx])
        entry = (vol, self._foreground_bbox(vol))
        self._cache[idx] = entry
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return entry

    def _sample_patch(self, vol, bbox, rng):
        p = self.patch
        shp = vol.shape
        # pad small volumes up to the patch size (rare for IXI 256^2 x ~150)
        if any(s < p for s in shp):
            pad = []
            for d in reversed(range(3)):
                deficit = max(0, p - shp[d])
                pad += [deficit // 2, deficit - deficit // 2]
            vol = torch.nn.functional.pad(vol[None, None], pad)[0, 0]
            shp = vol.shape
            bbox = [(0, shp[d]) for d in range(3)]
        # sample origin so the patch overlaps the cached foreground bbox (with prob
        # fg_frac); no per-patch nonzero. A few tries to ensure enough signal.
        for _ in range(6):
            o = []
            for d in range(3):
                if rng.random() < self.fg_frac:
                    lo = max(0, bbox[d][0] - p // 2)
                    hi = min(shp[d] - p, bbox[d][1] - p // 2)
                    o.append(rng.randint(lo, hi) if hi > lo else max(0, min(lo, shp[d] - p)))
                else:
                    o.append(rng.randint(0, shp[d] - p) if shp[d] > p else 0)
            patch = vol[o[0]:o[0] + p, o[1]:o[1] + p, o[2]:o[2] + p]
            if (patch > 0.05).float().mean() > 0.05:      # enough signal
                return patch.clone()
        return patch.clone()

    def _augment(self, x, rng):
        # flips + 90deg rotations (orientation-agnostic anatomy)
        for d in range(3):
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[d])
        if rng.random() < 0.5:
            k = rng.randint(1, 3)
            x = torch.rot90(x, k, dims=[0, 1])
        # contrast inversion (break polarity: bright-CSF vs dark-CSF etc.)
        if rng.random() < 0.5:
            x = 1.0 - x
            x = x.clamp_min(0.0)
        # gamma (nonlinear contrast remap)
        if rng.random() < 0.7:
            gamma = float(np.exp(rng.uniform(np.log(0.6), np.log(1.7))))
            x = x.clamp_min(0.0) ** gamma
        # smooth multiplicative bias field
        if rng.random() < 0.5:
            x = x * _smooth_bias_field(x.shape, x.device)
        # mild global scale (bias-free net should be invariant; keeps ranges varied)
        if rng.random() < 0.5:
            x = x * float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
        return x.clamp_min(0.0)

    def _add_noise(self, clean, sigma, rng):
        if rng.random() < self.rician_prob:
            n1 = torch.randn_like(clean); n2 = torch.randn_like(clean)
            return torch.sqrt((clean + sigma * n1) ** 2 + (sigma * n2) ** 2)
        return clean + sigma * torch.randn_like(clean)

    def _worker_rng(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.RandomState(self._base_seed + 100003 * (wid + 1))
        return self._rng

    def __getitem__(self, i):
        rng = self._worker_rng()
        # reuse a cached volume most of the time (cheap); occasionally load a new one
        if self._cache and rng.random() > self.p_new:
            vidx = rng.choice(list(self._cache.keys()))
        else:
            vidx = rng.randint(0, len(self.paths))
        vol, bbox = self._get_volume(vidx)
        clean = self._sample_patch(vol, bbox, rng)
        if self.gpu_aug:
            # cheap CPU part only: flips + 90deg rotations (views/copies); heavy aug +
            # noise happen on the GPU in the training loop.
            if self.augment:
                for d in range(3):
                    if rng.random() < 0.5:
                        clean = torch.flip(clean, dims=[d])
                if rng.random() < 0.5:
                    clean = torch.rot90(clean, rng.randint(1, 3), dims=[0, 1])
            return clean[None].float().contiguous()
        if self.augment:
            clean = self._augment(clean, rng)
        sigma = rng.uniform(self.sigma_min, self.sigma_max)
        noisy = self._add_noise(clean, sigma, rng)
        target = self._add_noise(clean, sigma, rng) if self.n2n else clean
        smap = torch.full_like(noisy, sigma)
        return (noisy[None].float(), target[None].float(), smap[None].float())


def _gauss_blur3d(x, fwhm):
    """Separable 3D Gaussian blur (single scalar fwhm in voxels) with reflect pad."""
    if fwhm <= 0:
        return x
    std = float(fwhm) / 2.355
    r = max(1, int(round(3 * std)))
    c = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(c ** 2) / (2 * std * std)); k = k / k.sum()
    for d in range(3):
        shp = [1, 1, 1, 1, 1]; shp[2 + d] = k.numel()
        ker = k.view(shp)
        pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - d)] = r; pad[2 * (2 - d) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode='reflect'), ker)
    return x


def augment_noise_gpu(clean, sigma_min=0.0, sigma_max=0.15, rician_prob=0.5, n2n=True,
                      blur_max=0.0):
    """Batched GPU augmentation + degradation for gpu_aug mode. Input/return on the
    SAME device as `clean` ([B,1,D,H,W]). Applies per-sample gamma, contrast inversion,
    smooth bias field, global scale, then degradation -> returns (input, target, sigma).

    * blur_max == 0 (DENOISE task): input = clean + Gaussian/Rician noise; target =
      clean (or a 2nd noise draw if n2n). The prior learns to remove noise.
    * blur_max > 0 (RESTORE / super-resolution task): input = blur(clean, fwhm) + noise
      with fwhm ~ U(0, blur_max) voxels; target = the SHARP clean patch. The prior
      learns to remove blur AND noise -> in RED it can ADD high-frequency detail
      (improves sharp channels like T1, not just denoise). Supervised (target sharp)."""
    B = clean.shape[0]
    dev = clean.device
    x = clean.clamp_min(0.0)
    # contrast inversion (per-sample, prob 0.5)
    inv = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    x = inv * (1.0 - x).clamp_min(0.0) + (1.0 - inv) * x
    # gamma (prob 0.7): x**gamma, gamma ~ logU(0.6,1.7)
    do_g = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.7).float()
    gamma = torch.exp(torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(
        float(torch.log(torch.tensor(0.6))), float(torch.log(torch.tensor(1.7)))))
    gamma = do_g * gamma + (1.0 - do_g) * 1.0
    x = x.clamp_min(0.0) ** gamma
    # smooth multiplicative bias field (prob 0.5): upsample a 4^3 low-freq field
    do_b = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    low = torch.randn(B, 1, 4, 4, 4, device=dev)
    field = F.interpolate(low, size=x.shape[2:], mode='trilinear', align_corners=False)
    field = field / field.abs().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    x = x * (1.0 + do_b * 0.3 * field)
    # mild global scale (prob 0.5)
    do_s = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    scl = torch.exp(torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(
        float(torch.log(torch.tensor(0.7))), float(torch.log(torch.tensor(1.4)))))
    x = x * (do_s * scl + (1.0 - do_s))
    x = x.clamp_min(0.0)
    # noise: per-sample sigma; Rician for a fraction, else Gaussian
    sigma = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(sigma_min, sigma_max)

    def add_noise(c):
        ric = (torch.rand(B, 1, 1, 1, 1, device=dev) < rician_prob).float()
        n1 = torch.randn_like(c); n2 = torch.randn_like(c)
        gauss = c + sigma * n1
        rician = torch.sqrt((c + sigma * n1) ** 2 + (sigma * n2) ** 2)
        return ric * rician + (1.0 - ric) * gauss

    if blur_max > 0:
        # RESTORE: blur the input (one fwhm per batch), keep the target sharp.
        fwhm = float(torch.empty(1).uniform_(0.0, blur_max).item())
        noisy = add_noise(_gauss_blur3d(x, fwhm))
        target = x
    else:
        noisy = add_noise(x)
        target = add_noise(x) if n2n else x
    return noisy, target, sigma.reshape(-1)
