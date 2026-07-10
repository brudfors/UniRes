# Design Spec: Merged-gradient DL prior for UniRes (one 3D network call, C-independent)

> **Scope:** design / feasibility exploration answering "*is it possible to put the DL prior on a
> merged gradient representation so the network runs once instead of per channel, saving compute?*"
> — plus a phased prototype roadmap. Native-3D network (A100-80GB target). Branch: `dl-prior-red`.

## The question, answered
Today's DL prior (RED, `prior='mtv+red'`) denoises the reconstruction **per channel** — **C network
calls per ADMM iteration** (and 2.5D). Can we reformulate so the prior acts on a **merged,
channel-pooled gradient** representation and run the network **once**? **Yes — cleanly, and it is
*more faithful* to UniRes's multi-channel design than per-channel image-RED.**

## Why it's clean: the joint-TV z-update already does exactly this
UniRes's JTV z-update (`unires/_update.py:223-237`) is the group-soft-threshold prox, and it already:
1. **merges** all C channels × 3 directions into one per-voxel scalar magnitude `m` (`:228-229`),
2. applies **one** scalar nonlinearity `s = (m − 1/ρ)₊ / (m + ε)` (`:230`),
3. **broadcasts** that per-voxel gain to every channel/direction (`:237`) — preserving each gradient's
   direction, only reshaping its magnitude.

So the learned prior is a **~1-line swap at `:230`**: replace the analytic gain with a learned gain on
the merged representation; **the broadcast at `:237` is left verbatim**. The y-update (CG) and w-update
are untouched; the image-RED `+μ` terms are not used.

## Formulation
Feed the network the **dimensionless** `u = ρ·m` (ties the gain to the ADMM knee `1/ρ`, so **one net
works across the coarse-to-fine λ schedule, ρ updates, and any number of contrasts**). Net → per-voxel
gain → broadcast (`:237`). The identity/soft-threshold gain reproduces JTV **bit-for-bit**.

Three variants (increasing power / decreasing structure guarantee):
- **(a) Pointwise learned potential** `Φ_θ(m)` → gain = a learned 1-D scalar shrinkage curve
  (`prox_{Φ_θ/ρ}(m)/m`). **Explicit energy** `Σ_voxel Φ_θ(m)` ⇒ monotone objective preserved.
  Cheapest, C-independent, but **no spatial context** (learned edge-preserving / nonconvex TV; cf.
  TNRD, Fields-of-Experts). **Safe baseline.**
- **(b) Spatial 3D-CNN gain** — **RECOMMENDED prototype.** A native 3D CNN on the merged magnitude
  field (K=1) or per-direction magnitudes (K=3) → per-voxel gain ∈ [0,1] (sigmoid) → broadcast.
  Spatially aware, C-independent, **one 3D pass**. PnP (no explicit energy) → reuse the image-RED
  **line-search guard** for a monotone *reported* objective; or a proximal/gradient-step-denoiser
  parametrization for an explicit energy.
- **(c) Structure-tensor anisotropic gain** — stretch. 3D net on the per-voxel structure tensor
  `S = Σ_c ∇y_c ∇y_cᵀ` (6 unique, C-independent) → SPD 3×3 gain `G`; `z_{c,·} = G·v_{c,·}`.
  Anisotropic (coherence-enhancing), most expressive, hardest map-back/training, no simple energy.

## Mathematical formalization & monotone convergence
UniRes solves `min_y f(y) + R(Dy)` (f = data term, D = gradient). ADMM splits `z=Dy`, dual `w`,
augmented Lagrangian `L_ρ = f(y) + R(z) + ⟨w,Dy−z⟩ + (ρ/2)‖Dy−z‖²`; updates: y (CG), `z ←
prox_{R/ρ}(Dy+w/ρ)`, `w ← w+ρ(Dy−z)`. Classic JTV = `R(g)=Σ_v‖g_v‖₂`, whose prox is the current
vectorial soft-threshold. The three variants differ in whether the learned z-update is still a prox:

- **(a) explicit variational — full guarantees.** `R_θ(g)=Σ_v Φ_θ(‖g_v‖)`; the z-update is *exactly*
  `prox_{R_θ/ρ}` (radial ⇒ scalar gain `s=prox_{Φ_θ/ρ}(m)/m`). Constrain the learned 1-D shrinkage to
  be **monotone + non-expansive** ⇒ it is a genuine prox (Moreau; Gribonval–Nikolova). Convex `Φ_θ`
  ⇒ ADMM → global optimum; nonconvex edge-preserving `Φ_θ` ⇒ → stationary point with **monotone
  augmented-Lagrangian descent** (Wang–Yin–Zeng 2019). **Same monotone-objective oracle as JTV.**
- **(b)/(c) plug-and-play.** A spatial CNN gain is **not** a prox ⇒ PnP-ADMM: **no underlying energy**;
  monotonicity only of the *reported* objective via the line-search guard (as image-RED). Fixed-point
  **convergence** holds if the gain-operator is **nonexpansive** (Lipschitz/spectral-norm training,
  Ryu 2019) or **bounded+continuation** (Chan 2017).
- **(b′) proximal / gradient-step denoiser — the rigorous sweet spot.** For ADMM's *prox* z-step use a
  **proximal denoiser** (Hurault et al., ICML 2022): an *exact prox* of an explicit nonconvex potential
  `g_θ` on the merged field ⇒ **explicit energy** `R_θ=g_θ`, exact z-update, provable **convergence to
  stationary points with monotone descent**. (The ICLR-2022 *gradient-step* denoiser `z=v−∇g_θ(v)` is
  the analogous form for an HQS/PGD solver.) Spatial-CNN power + explicit energy + monotone convergence,
  still one call / C-independent.

**Verdict on monotonicity:** preserved for the **explicit-energy** formulations (a) and (b′) — same or
better footing than classic JTV; a *bare* PnP CNN gain keeps only the guarded (non-increasing reported
objective) property, identical to image-RED. **If monotone convergence is a hard requirement, use (a)
or (b′), not a plain PnP gain.**

## Compute & memory (native 3D — the core answer)
- **Current image-RED:** C denoiser calls/iter, each 2.5D = 3 axis sweeps ⇒ ~**3C volume-sweeps/iter**.
- **Proposed:** **ONE 3D pass/iter** on a **K ≤ 6-channel** field, **C-independent**, no 2.5D
  averaging. CNN FLOPs barely depend on the 1–6 input channels ⇒ **~factor-C fewer prior passes**,
  growing with C.
- **Memory:** the merged input is small (K=1→33 MB, K=3→99 MB, K=6→198 MB — never the full 9-channel
  field). 3D-conv **activations** dominate: ~**6–9 GB** for a width-48–64 3D UNet at 192×224×192,
  ~15–18 GB at 256³ — **comfortably within A100-80GB**. The low-channel merged field is exactly what
  makes native 3D feasible (a full 3D image/field denoiser would be far heavier). Safety valves:
  narrow/shallow net; `monai.inferers.SlidingWindowInferer` patch inference (baked into the Lepton image).

## Beyond compute — why it's also *better*
- **Restores the cross-channel coupling** that per-channel image-RED dropped (pools channels *before*
  the nonlinearity ⇒ genuinely multi-contrast, like JTV).
- Acts in the **gradient domain**, where the ADMM already splits and where `p(gradient) ≠ p(image)`
  (so an image denoiser like DRUNet can't be reused here — the `_update.py:208-219` TODO).
- **One model for any number of contrasts.**

## Honest risks
- Quality is **not guaranteed** to beat image-RED (~+0.4 PSNR at mild strength) — must be benchmarked.
  Scalar gain (a/b) is expressively limited; (c) lifts this but is harder.
- (b)/(c) are **PnP** ⇒ monotonicity only for the *reported* objective via the guard; no
  global-energy convergence guarantee. (a) is a genuine prox with an explicit energy.

## Training the 3D grad-net (Phase 3 — Lepton A100 / IXI)
- Train on **IXI** (`scripts/download_ixi.py`), **test on BrainWeb** (clean train/test split).
- From a clean volume `y*`: clean magnitude `m* = ‖λ·∇y*‖`; add synthetic noise ⇒ `m_noisy = ‖∇(y*+n)‖`;
  train `D_θ: m_noisy → m*` (supervised `deepinv.loss.SupLoss`; or self-supervised
  `Neighbor2Neighbor`/`R2RLoss`). Gain = `m̂/(m+ε)` clamped to [0,1].
- **C-agnostic:** the pooled magnitude of 1 contrast and of C contrasts share the same statistical
  family ⇒ train on abundant **single-contrast** IXI volumes, transfers to any C (no co-registration).
- **Signed handling:** do NOT reuse `_denoiser._standardize` (its positive-quantile scale assumes
  non-negative intensities); normalize by `u=ρ·m` or a robust `|·|` scale.
- **Bias-free** 3D UNet/DnCNN (`deepinv.models.UNet/DnCNN`, `dim=3`) ⇒ scale-equivariance; σ-conditioned
  or blind over a σ range.
- (Related) the image-RED denoiser should likewise go **native 3D** (drop 2.5D), reusing this 3D infra —
  secondary to the merged-gradient prior.

## Integration points (keep `prior='mtv'` byte-identical)
- `unires/struct.py`: add `prior='mtv+gradnet'` (reserve `'gradnet'`) + `grad_form`
  (`'pointwise'|'cnn'|'structure'`), `grad_net` (`'softthr'|'charbonnier'|'unet'|'dncnn'`),
  `grad_weights/grad_sigma/grad_scale/grad_gain_clamp/grad_apply/grad_patch/grad_linesearch`.
- `unires/_cli.py`: `--prior mtv+gradnet` + `--grad_*` flags (mirror `--red_*`).
- `unires/run.py::fit`: build `grad_prior = make_grad_prior(sett)` once (sibling to the RED branch),
  thread into `_update_admm`.
- `unires/_update.py::_update_admm`: the z-update swap at `~:230` (`grad_prior.gain(tmp, rho)` if
  active, else the **verbatim** analytic gain); broadcast `:232-237` unchanged (a/b); the PnP guard
  (pattern from the RED guard `:180-197`); `_compute_nll` (`:476-494`) energy term for (a).
- **NEW `unires/_grad_prior.py`** (parallel to `_denoiser.py`): K-channel **signed-field, native-3D**
  forward — analytic gains (`SoftThresh` = exact JTV, `Charbonnier`), deepinv 3D backends
  (UNet/DnCNN), optional `SlidingWindowInferer`, `.gain(m,ρ)` / `.apply_structure(...)` / optional
  `.energy(...)`, cache, `make_grad_prior(sett)`. Reuse `_denoiser.py`'s settings/factory/cache
  pattern — NOT its 2.5D wrapper or positive-quantile standardization.

## Phased roadmap
- **Phase 0 — plumbing + equivalence (no training, CPU-verifiable).** Settings/CLI, `_grad_prior.py`
  with the analytic `softthr` gain, the z-update swap. **Verify `grad_net='softthr'` reproduces
  `prior='mtv'` bit-for-bit (≤1e-6)** — proves the swap + broadcast add zero drift and identity gain = JTV.
- **Phase 1 — generalized analytic gain.** `charbonnier` (explicit energy): param→default = JTV;
  verify monotone objective; show edge-preserving behavior.
- **Phase 2 — train the 3D CNN gain on IXI** (`unet`/`dncnn`, bias-free magnitude denoiser). To keep
  **monotone convergence**, prefer the **proximal-denoiser (b′)** parametrization (Hurault ICML 2022 —
  exact prox of an explicit energy `g_θ` ⇒ provable monotone descent); a plain PnP gain falls back to
  the line-search guard. Verify the reported objective is monotone per reg-scale.
- **Phase 3 — benchmark (the key head-to-head).** Extend `benchmarks/bench_dl_prior.py` with a
  `mtv+gradnet` path and compare it against **(a) classic MTV** (`prior='mtv'`) and **(b) the true-3D
  image-domain RED prior on the reconstruction** (`prior='mtv+red'` with the native-3D recon denoiser —
  built in parallel; use its trained model), on the **same** simulated BrainWeb inputs/settings.
  Report **quality** (PSNR/SSIM vs GT, per channel and averaged, at each prior's best mild strength)
  **and compute** (prior passes/iter, wall-clock, `torch.cuda.max_memory_allocated`). Central question:
  does the merged-gradient prior **match/beat the 3D-recon RED's quality at *lower* prior compute**
  (one 3D pass on a low-channel gradient field, C-independent, vs the image-RED's per-channel passes)?
  **Sweep C=2,3,4** so the compute gap widens with contrast count. (Interim baseline: the 2.5D image-RED
  if the 3D one isn't ready.)

## Verification / success criteria
- **Equivalence:** `softthr` gain == classic MTV bit-for-bit.
- **Monotonicity:** explicit energy (a)/(b′) or the line-search guard (b/c) ⇒ reported objective
  non-increasing per reg-scale.
- **Quality vs compute:** BrainWeb PSNR/SSIM ≥ MTV, and **≈ or > the true-3D image-domain RED at
  *lower* prior compute** (the central head-to-head); no hallucination (data-anchored, gain ∈ [0,1]).
- **Compute:** **1 network pass/iter, C-independent**; single-digit-GB 3D forward on the A100.

## Critical files
- `unires/_update.py` — z-update swap `~:229-230`; broadcast `:232-237` unchanged; guard `:180-197`;
  energy `:476-494`.
- **NEW `unires/_grad_prior.py`** — the K-channel signed-field native-3D prior.
- `unires/struct.py`, `unires/_cli.py`, `unires/run.py` — settings/CLI/wiring.
- `benchmarks/bench_dl_prior.py` (add the `mtv+gradnet` path), `scripts/download_ixi.py` (training
  data), `unires/_denoiser.py` (pattern to parallel; later move image-RED to native 3D).
