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
                 affine=True, deform_prob=0.8, zero_bg=False, texture=0.0, realism=0.0):
        self.device = device
        self.patch = int(patch)
        self.gmm_fwhm = gmm_fwhm
        self.affine = affine
        self.deform_prob = deform_prob
        # 0 = textured background (full-head, e.g. IXI); 1 = zeroed background (skull-stripped,
        # e.g. OASIS/MICCAI); 2 = RANDOMISED per sample -> one model robust to BOTH input types.
        self.zero_bg = int(zero_bg)
        # WITHIN-TISSUE TEXTURE: piecewise-constant GMM synth is TOO recoverable under thick-slice
        # degradation -> the restorer learns to confidently reconstruct sharp edges and then
        # HALLUCINATES/over-sharpens on real (textured) T1. Adding coherent multi-scale texture
        # inside tissues (destroyed by thick-slicing, so not fully recoverable) keeps the model
        # calibrated. texture = max multiplicative amplitude (0 = classic piecewise-constant).
        self.texture = float(texture)
        # REALISM: richer generative appearance to close the recoverable-real-T1 gap (the real-T1
        # prior reconstructs fine anatomy synth can't). Adds COHERENT MULTI-SCALE structure
        # (coarse intra-tissue intensity gradients -> fine texture) + soft partial-volume
        # boundaries, so the restorer learns to reconstruct realistic fine T1 detail rather than
        # flat piecewise-constant regions. realism = max amplitude (0 = off).
        self.realism = float(realism)
        maps = _load_label_maps(dir_labels, device)
        # Pre-stack all maps into ONE tensor on a common grid so a whole BATCH of crops can
        # be pulled with a single index_select + grid_sample (no per-sample Python/crop loop
        # -> per-step CUDA-launch count drops to ~Part-B level, survives the loaded box).
        S = int(max(max(t.shape) for t, _, _, _ in maps))
        vols, lo_n, hi_n = [], [], []
        for t, _K, lo, hi in maps:
            if tuple(t.shape) != (S, S, S):
                t = F.interpolate(t[None, None].float(), size=(S, S, S),
                                  mode="nearest")[0, 0].to(torch.uint8)
                fg = (t > 0).nonzero(as_tuple=False)
                lo, hi = fg.min(0).values, fg.max(0).values
            vols.append(t)
            lo_n.append(lo.float()); hi_n.append(hi.float())
        self.S = S
        self.vol = torch.stack(vols)[:, None]           # [N,1,S,S,S] uint8
        self.lo = torch.stack(lo_n)                     # [N,3] fg bbox (voxels)
        self.hi = torch.stack(hi_n)

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
        P, S, dev = self.patch, self.S, self.device
        # --- crop + affine for the WHOLE batch in ONE grid_sample (no per-sample loop) ---
        idx = torch.randint(len(self.vol), (batch,), device=dev)
        vol_b = self.vol[idx].float()                       # [B,1,S,S,S]
        lo, hi = self.lo[idx], self.hi[idx]                 # [B,3] (D,H,W voxel bbox)
        # per-sample crop centre inside the foreground bbox, patch kept in-bounds
        half = P / 2.0
        cmin = torch.clamp(lo, min=half)
        cmax = torch.maximum(torch.clamp(hi, max=S - half), cmin)
        c = cmin + (cmax - cmin) * torch.rand(batch, 3, device=dev)     # voxel coords (D,H,W)
        center_norm = (2.0 * c / (S - 1) - 1.0).flip(1)                 # -> (x,y,z)=(W,H,D)
        # linear part: random rotation/zoom/shear/flips, SCALED by P/S so the P^3 output
        # samples a patch-sized region (identity-kept samples become a pure axis-aligned crop)
        theta = _rand_affine_theta(batch, dev,
                                   do_prob=self.deform_prob if self.affine else 0.0)  # [B,3,4]
        theta[:, :, :3] = theta[:, :, :3] * (P / S)
        theta[:, :, 3] = center_norm
        grid = F.affine_grid(theta, (batch, 1, P, P, P), align_corners=False)
        # GUARD grid coords: NaN -> OOB (sampled as background via zeros pad); clamp finite range
        grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0).clamp_(-2.0, 2.0)
        labs = F.grid_sample(vol_b, grid, mode="nearest", padding_mode="zeros",
                             align_corners=False).long()               # [B,1,P,P,P]
        return self._synthesize(labs)

    def _synthesize(self, labs):
        """labs [B,1,...] long -> SHARP random-contrast image [B,1,...] via a per-label
        random Gaussian intensity model (fresh CONTRAST), one smoothing, robust [0,1] norm.
        Shared by patch (`sample`) and whole-volume (`sample_full`) generation."""
        B = labs.shape[0]
        spat = labs.shape[2:]
        Kmax = int(labs.max().item()) + 1
        mean = torch.rand(B, Kmax, device=self.device)              # U(0,1) per-label intensity
        std = torch.rand(B, Kmax, device=self.device) * 0.25         # within-tissue variance
        mean[:, 0] *= 0.15                                           # background mostly dark
        flat = labs.reshape(B, -1)
        m = torch.gather(mean, 1, flat).reshape(B, 1, *spat)
        s = torch.gather(std, 1, flat).reshape(B, 1, *spat)
        img = (m + s * torch.randn_like(m)).float()
        if self.realism > 0:
            # COHERENT MULTI-SCALE intra-tissue structure (coarse gradients -> fine texture),
            # multiplicative so it scales with tissue intensity like real MRI. Spans sigma 8..0.75
            # so tissues get both smooth internal gradients AND recoverable fine detail. Per-sample
            # amplitude U(0, realism). Applied to the piecewise-constant mean BEFORE the PV smooth.
            fld = torch.zeros_like(img)
            for sig, w in ((8.0, 1.0), (4.0, 0.8), (2.0, 0.6), (1.0, 0.45), (0.75, 0.35)):
                fld = fld + w * self._gauss3d(torch.randn_like(img), sig)
            fld = fld / fld.reshape(B, -1).std(dim=1).clamp_min(1e-6).view(B, 1, 1, 1, 1)
            amp = torch.rand(B, 1, 1, 1, 1, device=self.device) * self.realism
            img = (img * (1.0 + amp * fld)).clamp_min(0.0)
        lo, hi = self.gmm_fwhm
        img = self._gauss3d(img, float(torch.empty(1).uniform_(lo, hi))).clamp_min(0.0)
        if self.texture > 0:
            # coherent multi-scale texture field (sum of noise smoothed at a few scales),
            # applied MULTIPLICATIVELY AFTER the GMM smooth (so the fine scales SURVIVE into the
            # image and are then destroyed by thick-slicing downstream -> the model learns fine
            # detail isn't always recoverable). Per-sample amplitude U(0, texture).
            tex = torch.zeros_like(img)
            for sig in (0.75, 1.5, 3.0):
                tex = tex + self._gauss3d(torch.randn_like(img), sig)
            tex = tex / tex.reshape(B, -1).std(dim=1).clamp_min(1e-6).view(B, 1, 1, 1, 1)
            amp = torch.rand(B, 1, 1, 1, 1, device=self.device) * self.texture
            img = (img * (1.0 + amp * tex)).clamp_min(0.0)
        if self.zero_bg == 1:
            img = img * (labs != 0).float()                # background -> 0 (realistic air)
        elif self.zero_bg == 2:
            # RANDOMISED per-sample: half the batch gets a zeroed background (skull-stripped /
            # brain-extracted inputs like OASIS-MICCAI), half keeps the textured background
            # (full-head like IXI). A FIXED choice is a bad trade (zero_bg=1 helped MICCAI but
            # cost the IXI win); randomising makes ONE model robust to both input types.
            keep = (torch.rand(B, 1, 1, 1, 1, device=self.device) < 0.5).float()
            img = img * (keep + (1.0 - keep) * (labs != 0).float())
        # robust [0,1] scale; use a quantile of the FOREGROUND so an all-bg patch can't blow up
        hi = torch.quantile(img.reshape(B, -1), 0.99, dim=1).clamp_min(1e-3)
        img = (img / hi.view(B, 1, 1, 1, 1))
        # HARD finite-guard: no NaN/Inf may ever leave the synth (would hang a downstream kernel)
        return torch.nan_to_num(img, nan=0.0, posinf=2.0, neginf=0.0).clamp_(0.0, 2.0)

    def sample_full(self, size=None):
        """WHOLE-VOLUME synth: one label map (optionally resized to `size`), a single
        random affine (rotation/zoom/shear/flips, NO crop/translation -> whole brain kept),
        then a fresh random contrast. Returns [1,1,D,D,D] for full-image training."""
        dev, S = self.device, self.S
        D = int(size) if size else S
        idx = int(torch.randint(len(self.vol), (1,), device=dev).item())
        vol = self.vol[idx:idx + 1].float()                          # [1,1,S,S,S]
        if D != S:
            vol = F.interpolate(vol, size=(D, D, D), mode="nearest")
        if self.affine:
            theta = _rand_affine_theta(1, dev, do_prob=1.0)          # whole-volume warp (scale=1)
            theta[:, :, 3] = 0.0                                     # centred (no crop shift)
            grid = F.affine_grid(theta, (1, 1, D, D, D), align_corners=False)
            grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0).clamp_(-2.0, 2.0)
            vol = F.grid_sample(vol, grid, mode="nearest", padding_mode="zeros",
                                align_corners=False)
        return self._synthesize(vol.long())


_LG06, _LG17 = float(np.log(0.6)), float(np.log(1.7))
_LG07, _LG14 = float(np.log(0.7)), float(np.log(1.4))


def batched_degrade(x, _thick_slice_degrade, _gauss_blur3d, sigma_min=0.0, sigma_max=0.05,
                    rician_prob=0.5, blur_max=1.0, thick_max=8.0, inplane_max=0.0, op_degrade=None,
                    multi_ori=0.0):
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
        # IN-PLANE RESOLUTION VARIATION: real acquisitions have varying in-plane resolution
        # (0.5x0.5 .. 1x1mm), i.e. the recon grid the prior restores TO is not fixed. Blur the
        # TARGET isotropically by a random amount so the model learns to restore to a range of
        # in-plane resolutions; the thick-slice is applied to THIS target, so `res` encodes ONLY
        # the through/in-plane anisotropy (matching po.ratio at inference). Discrete sigma =
        # allocator-friendly. inplane_max=0 -> off (original iso-target behaviour).
        if inplane_max > 0:
            ipsig = float(torch.randint(0, int(round(inplane_max * 4)) + 1, (1,)).item()) / 4.0
            if ipsig > 0:
                target = _gauss_blur3d(x, ipsig)
        # DISCRETE integer through/in-plane ANISOTROPY factor (1..thick_max): continuous factors
        # make _thick_slice_degrade produce a differently-sized interpolate EVERY step, so the
        # CUDA caching allocator accumulates ever-more distinct block sizes and eventually stalls
        # a kernel (intermittent stream WEDGE). A small fixed set of sizes lets it reuse blocks.
        # Range 1..thick_max covers realistic extreme anisotropy (e.g. 0.5x0.5x6.5mm -> ~13x).
        else:
            factor = float(torch.randint(1, int(thick_max) + 1, (1,)).item())
        # DIRECTIONAL per-axis res [B,3]: only the degraded (thick) axis is set; the other two
        # recon axes stay at 0 (ratio 1). Matches _thick_factor_axes at inference, so the model
        # is told WHICH recon axis is through-plane.
        res = x.new_zeros(B, 3)
        res[:, axis] = (factor - 1.0) / max(thick_max - 1.0, 1.0)
        if multi_ori > 0 and float(torch.rand(1).item()) < multi_ori:
            # MULTI-ORIENTATION: real protocols acquire 2-3 CROSSED thick stacks; UniRes fuses
            # them, so the denoiser sees an estimate that is MUCH better than any single stack
            # (each stack's in-plane detail fills the others' null space). Trained single-stack
            # only, a thickness-conditioned prior restores too CONSERVATIVELY there (measured:
            # the blind IXI prior beats it at 2 orientations). Simulate the fused estimate by
            # degrading along 2-3 distinct axes and averaging, and set `res` by the SAME
            # elementwise-MIN aggregation _thick_factor_axes uses at inference -> train and
            # inference agree on what the conditioning means.
            n_ori = int(torch.randint(2, 4, (1,)).item())
            axes = torch.randperm(3)[:n_ori].tolist()
            acc, ratios = None, [None, None, None]
            for a in axes:
                f = float(torch.randint(1, int(thick_max) + 1, (1,)).item())
                d = _thick_slice_degrade(target, f, a)
                acc = d if acc is None else acc + d
                for k in range(3):                      # this stack's per-axis ratio vector
                    rv = f if k == a else 1.0
                    ratios[k] = rv if ratios[k] is None else min(ratios[k], rv)
            deg = acc / float(n_ori)
            for k in range(3):
                res[:, k] = (float(ratios[k]) - 1.0) / max(thick_max - 1.0, 1.0)
            if blur_max > 0:
                bsig = float(torch.randint(0, int(round(blur_max * 4)) + 1, (1,)).item()) / 4.0
                deg = _gauss_blur3d(deg, bsig)
        elif op_degrade is not None:
            # OPERATOR-CONSISTENT: forward-project + reconstruct the whole batch through UniRes's
            # projector in one call (matches the ADMM iterate the denoiser sees at inference; the
            # PSF is modelled so no extra iso blur is applied).
            deg = op_degrade(target, axis, factor)
        else:
            deg = _thick_slice_degrade(target, factor, axis)
            if blur_max > 0:
                # discretise blur sigma too (fixed kernel sizes -> allocator-friendly)
                bsig = float(torch.randint(0, int(round(blur_max * 4)) + 1, (1,)).item()) / 4.0
                deg = _gauss_blur3d(deg, bsig)
    else:
        deg, res = x, torch.zeros(B, 3, device=dev)
    # Gaussian/Rician noise (per-sample choice), batched
    ric = (torch.rand(B, 1, 1, 1, 1, device=dev) < rician_prob).float()
    n1, n2 = torch.randn_like(deg), torch.randn_like(deg)
    gauss = deg + sigma * n1
    rician = torch.sqrt((deg + sigma * n1) ** 2 + (sigma * n2) ** 2)
    noisy = ric * rician + (1.0 - ric) * gauss
    # HARD finite-guard on BOTH tensors before they hit the net (belt-and-braces vs kernel hangs)
    noisy = torch.nan_to_num(noisy, nan=0.0, posinf=4.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=4.0, neginf=0.0)
    return noisy, target, sigma.reshape(-1), res
