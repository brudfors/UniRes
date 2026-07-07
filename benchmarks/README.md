# Benchmarks

## `bench_precond.py` — CG preconditioner (issue #12)

Compares UniRes convergence **without** vs **with** the CG diagonal (Jacobi)
preconditioner. It simulates orthogonal thick-slice low-resolution acquisitions from
the 1 mm BrainWeb images in `data/` (so the originals serve as ground truth), runs the
algorithm twice from an identical shared initialisation, and reports:

- **convergence** — outer iterations, wall-clock, and objective-vs-time / -vs-iteration curves;
- **correctness** — objective monotonic within each regularisation scale, matching final
  objective (same fixed point), and an independent recompute of the preconditioner diagonal;
- **image quality** — PSNR/SSIM of each reconstruction vs the 1 mm ground truth, plus a
  direct precond-vs-baseline agreement check.

It prints a table + `PASS/FAIL` and saves `benchmarks/out/precond_bench.png`. Exit code is
nonzero if correctness or quality regresses (CI-friendly).

### Run in Docker (GPU)

The image installs `unires` **non-editable**, so bind-mounting the source is not enough on
its own — set `PYTHONPATH=/app` so Python imports the mounted (edited) package:

```bash
docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app unires:latest \
  python benchmarks/bench_precond.py --out benchmarks/out --vx 1.0
```

Stricter, single-scale monotonicity run (every-iteration decrease expected):

```bash
docker run --rm --gpus all -v "$PWD:/app" -w /app -e PYTHONPATH=/app unires:latest \
  python benchmarks/bench_precond.py --out benchmarks/out --sched 0 --reg_scl 1.0
```

To exercise the CLI flag (`--no-precond`) instead, rebuild the image (or `pip install -e .`)
so the `unires` console script picks up the changes.

### Useful options
- `--factor N` slice downsample factor (default 3)
- `--axes 2 1 0` per-channel thick-slice axis (default: T1 axial, T2 coronal, PD sagittal)
- `--sched N` coarse-to-fine scalings (`0` = single fixed scale)
- `--reg_scl X` regularisation scaling
- `--max_iter N` cap outer iterations
