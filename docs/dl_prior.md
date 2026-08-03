# Deep-learning (RED) prior for UniRes

UniRes can optionally replace/augment its multi-channel total-variation (MTV) prior with a
learned denoiser, plugged in via **RED** (Regularization by Denoising). The shipped prior is
**contrast-agnostic**: it is trained purely on *synthetic* random-contrast data (SynthSeg-style)
and never sees real MRI, so it is not tied to any acquisition, contrast or scanner.

## Quick start

```python
from unires.struct import settings
from unires.run import preproc

s = settings()
s.prior        = 'mtv+red'      # MTV data-consistency + learned prior
s.red_denoiser = 'biasfree3d'
s.red_mu       = 0.05           # recommended (see "Tuning")
# s.red_weights = '/path/to/unires_red_synth_full.pt'   # optional, see "Weights"
y, mat, _ = preproc(data, s)
```

`prior='mtv'` (the default) is byte-for-byte the classic behaviour — the DL prior is opt-in.

## Weights

The checkpoint is ~150 MB and is **not** stored in the repository. `red_weights` is resolved as:

1. `settings.red_weights` — an explicit path (always wins)
2. `$UNIRES_RED_WEIGHTS`
3. `unires/weights/unires_red_synth_full.pt` (next to the package)
4. `~/.cache/unires/unires_red_synth_full.pt`

Copy the released file into (3) or (4) and it is picked up automatically. If none is found the
error names every location tried.

## Validated defaults

| setting | default | why |
|---|---|---|
| `red_mu` | `0.0` (off); **use 0.05** | the prior's response to `mu` is steep; 0.05 is the validated operating point |
| `red_nullspace` | **0.5** | restricts the prior's correction to frequency bands the data does *not* measure |
| `red_thick_taper` | **False** | superseded by `red_nullspace`; the taper only slides `mu` down its response curve |
| `red_linesearch` | `True` | backtracking guard: the reported objective is non-increasing |

`red_nullspace` is a **solver** setting, not a model one: it splits the denoiser's correction into
"bands the data already measures" and "null space", and attenuates the former. Rewriting measured
structure is what produced the classic `dPSNR>0` but `dSSIM<0` signature. At 0.5 *both* metrics are
positive on every 2/4 mm cell, and it improves **every** prior (a real-data-trained prior's 6 mm T1
goes −0.05 → +1.03 dB).

## What to expect

Measured on real MRI (IXI, n=8 subjects/contrast, paired, `dPSNR` vs MTV, `mu=0.05`,
`red_nullspace=0.5`). `red(real)` is a prior trained on real IXI, for reference:

| | T1@2 | T1@4 | T2@2 | T2@4 | PD@2 | PD@4 | T1 fine-axis |
|---|---|---|---|---|---|---|---|
| **this prior (synthetic)** | +0.75 | +0.18 | **+3.25** | **+1.68** | **+7.24** | **+4.03** | **+0.95** |
| `red(real-data)` | +0.77 | **+1.00** | +1.28 | +0.05 | +2.46 | +0.67 | +0.94 |

* Large, consistent gains on **T2/PD** (2–3× a real-data-trained prior), and it stays positive at
  6 mm where the real-data prior goes negative.
* **T1** is on par in the well-posed and recoverable-detail cases.
* **Known limitation:** on strongly under-sampled T1 (4–6 mm along an already-coarse axis) and on
  skull-stripped data (e.g. OASIS/MICCAI), a prior trained on real T1 is still clearly better.
  Training our own model on 100% real data did **not** reproduce that advantage, so it stems from
  that model's architecture/training regime rather than from real data per se.

## Multiple orientations

`red_nullspace` and the prior's thickness conditioning are computed **per recon axis** and
aggregated with a **min** over observations: if any stack samples an axis finely, that axis is
well constrained in the fused reconstruction. (Using `max` tells the model everything is maximally
degraded and costs up to ~1.15 dB on crossed stacks.)

## Training your own

See `training/synthseg/`. The validated recipe:

```
--net basicunet --res_cond 1 --thick_max 16 --inplane_max 0 \
--realism 1.0 --zero_bg 2 --fixed_lr 1 --lr 1e-4 --full_image 1 --full_size 192
```

Ablations that did **not** pay off: operator-consistent degradation, within-tissue texture alone,
multi-orientation training (helps crossed stacks, costs single-stack), and mixing in real data
(`--real_frac`, neutral at best, harmful at high fractions).
