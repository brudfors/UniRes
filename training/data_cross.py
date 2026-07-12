"""Cross-channel restoration data: aligned multi-contrast IXI triplets -> per-channel
thick-slice restoration with CROSS-CHANNEL context.

Each sample degrades every channel thickly along a DIFFERENT random axis (so where one
channel is under-resolved another is resolved there), and gives the network, for each
target channel, an edge map aggregated over the OTHER channels as guidance. This lets
the prior fill in detail using cross-channel edge agreement -> less hallucination at
severe thickness, while staying variable-channel-count (context is a symmetric aggregate).
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data import _thick_slice_degrade  # reuse anisotropic thick-slice degradation


def edge_map(v):
    """Scale-equivariant edge magnitude |grad v| (per [B,1,D,H,W])."""
    g = torch.gradient(v, dim=(2, 3, 4))
    return torch.sqrt(sum(gi ** 2 for gi in g) + 1e-8)


class CrossPatches(Dataset):
    """Aligned [3,D,H,W] triplets -> random [3,p,p,p] patches (all channels aligned)."""
    def __init__(self, root, patch=64, patches_per_epoch=20000, cache=64, seed=0):
        self.paths = sorted(glob.glob(os.path.join(root, '*.pt')))
        if not self.paths:
            raise FileNotFoundError(f"No aligned triplets in {root} (run cache_aligned_ixi.py)")
        self.patch = patch
        self.length = patches_per_epoch
        self._cache = {}
        self._order = []
        self._cache_n = cache
        self._rng = None
        self._seed = seed

    def __len__(self):
        return self.length

    def _rng_(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.RandomState(self._seed + 7919 * ((info.id if info else 0) + 1))
        return self._rng

    def _vol(self, i):
        if i in self._cache:
            return self._cache[i]
        v = torch.load(self.paths[i], map_location='cpu').float()  # [3,D,H,W]
        self._cache[i] = v; self._order.append(i)
        if len(self._order) > self._cache_n:
            self._cache.pop(self._order.pop(0), None)
        return v

    def __getitem__(self, idx):
        rng = self._rng_()
        p = self.patch
        # reuse a cached volume most of the time (amortize load)
        if self._cache and rng.random() > 0.15:
            i = rng.choice(list(self._cache.keys()))
        else:
            i = rng.randint(0, len(self.paths))
        v = self._vol(i)  # [3,D,H,W]
        shp = v.shape[1:]
        o = [rng.randint(0, max(1, shp[d] - p)) for d in range(3)]
        patch = v[:, o[0]:o[0]+p, o[1]:o[1]+p, o[2]:o[2]+p].clone()
        if any(patch.shape[1+d] < p for d in range(3)):
            pad = []
            for d in reversed(range(3)):
                deficit = max(0, p - patch.shape[1+d]); pad += [0, deficit]
            patch = F.pad(patch[None], pad)[0]
        return patch.contiguous()  # [3,p,p,p]


def augment_noise_cross(clean, sigma_max=0.05, thick_max=8.0, blur_noise=0.02):
    """clean: [B,3,D,H,W] aligned triplet. Returns per-target-channel restoration with
    cross-channel context: (input, target, sigma, context) each [B*3,1,D,H,W].

    - per-sample intensity aug per channel (gamma, inversion, scale);
    - each channel thick-degraded along a DIFFERENT random axis + noise;
    - target = the SHARP channel; context_c = max over OTHER degraded channels of edge map."""
    B, C = clean.shape[0], clean.shape[1]
    dev = clean.device
    x = clean.clamp_min(0.0)
    # per (sample,channel) intensity aug
    inv = (torch.rand(B, C, 1, 1, 1, device=dev) < 0.5).float()
    x = inv * (1 - x).clamp_min(0) + (1 - inv) * x
    g = torch.exp(torch.empty(B, C, 1, 1, 1, device=dev).uniform_(-0.5, 0.5))
    x = x.clamp_min(0.0) ** g
    x = x * torch.exp(torch.empty(B, C, 1, 1, 1, device=dev).uniform_(-0.3, 0.3))
    x = x.clamp_min(0.0)
    # per-channel thick degradation along DIFFERENT random axes
    deg = torch.empty_like(x)
    for b in range(B):
        axes = list(np.random.permutation(3))  # different axis per channel
        for c in range(C):
            factor = float(torch.empty(1).uniform_(1.0, thick_max).item())
            deg[b, c:c+1] = _thick_slice_degrade(x[b, c:c+1][None], factor, axes[c % 3])[0]
    sigma = torch.empty(B, C, 1, 1, 1, device=dev).uniform_(0.0, sigma_max)
    deg = deg + sigma * torch.randn_like(deg)
    # edge maps of degraded channels
    em = edge_map(deg.reshape(B * C, 1, *deg.shape[2:])).reshape(B, C, *deg.shape[2:])
    inputs, targets, contexts, sigs = [], [], [], []
    for c in range(C):
        inputs.append(deg[:, c:c+1])
        targets.append(x[:, c:c+1])                    # sharp target
        others = [d for d in range(C) if d != c]
        ctx = em[:, others].amax(dim=1, keepdim=True)  # max edge over other channels
        contexts.append(ctx)
        sigs.append(sigma[:, c, 0, 0, 0])
    return (torch.cat(inputs, 0), torch.cat(targets, 0),
            torch.cat(sigs, 0), torch.cat(contexts, 0))
