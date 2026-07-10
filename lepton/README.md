# UniRes on NVIDIA Lepton (8×A100-80GB)

Builds a dev-pod Docker image with the **UniRes environment** — torch 2.6.0/cu126, the **compiled
nitorch backend** (arch 8.0 for A100, 8.9 for local Ada verify), `deepinv` (DRUNet prior), a baked
training stack (monai/h5py/tensorboard), and **Claude Code** + dev tooling. The `unires` package is
*not* baked; pull branch `dl-prior-red` on the pod and install it editable (below).

`Dockerfile` here is derived from the repo-root `Dockerfile` (the validated compiled-nitorch recipe)
— no Docker-in-Docker.

## 1. Log in to NGC (do NOT commit the key)

Export your NGC API key in the shell (or pull it from a secret store), then log in via stdin:

```sh
export NGC_API_KEY=<your NGC key>            # never commit this
printf '%s' "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

## 2. Build (arch 8.0 for A100 + 8.9 so a local Ada laptop can execute-verify)

The image bakes no repo source, so the build context can be this `lepton/` folder:

```sh
docker build -f lepton/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.9" \
  -t nvcr.io/r2kuatviomfd/internal-sandbox/mbrudfors-unires:latest \
  lepton/
```

For an A100-only image use `--build-arg TORCH_CUDA_ARCH_LIST=8.0`; for H100 use `9.0`.

## 3. Verify the COMPILED backend before pushing (the whole point)

```sh
# CPU check: compiled backend present, no TorchScript fallback, torch not bumped
docker run --rm nvcr.io/r2kuatviomfd/internal-sandbox/mbrudfors-unires:latest \
  python -W error::UserWarning -c \
  "import torch, nitorch, deepinv, monai, h5py; \
   print('torch', torch.__version__); \
   from nitorch.spatial import im_gradient; print('nitorch import OK')"
#  -> must NOT print 'nitorch uses its non-compiled backend (TS)'

# GPU check (executes the compiled kernels; laptop has the 8.9 build):
docker run --rm --gpus all nvcr.io/r2kuatviomfd/internal-sandbox/mbrudfors-unires:latest \
  python -c "import torch; from nitorch.spatial import im_gradient; \
             g=im_gradient(torch.rand(8,8,8,device='cuda')); print('compiled GPU op OK', tuple(g.shape))"
```

## 4. Push

```sh
docker push nvcr.io/r2kuatviomfd/internal-sandbox/mbrudfors-unires:latest
```

## 5. Launch a Lepton dev pod + connect

- Create a long-lived Lepton dev pod: image
  `nvcr.io/r2kuatviomfd/internal-sandbox/mbrudfors-unires:latest`, resource = 8×A100-80GB node,
  with the persistent `/workspace` NFS mount and an NGC pull secret configured.
- Connect via Teleport SSH (staging `nv-stg-dgxc`), then `tmux` and run `claude`:

```sh
tsh login --proxy=nv-stg-dgxc.teleport.sh
tsh config >> ~/.ssh/config
ssh root@<your-dev-pod-id>.nv-stg-dgxc.teleport.sh
```

## 6. On the pod: pull the branch + editable install

```sh
cd /workspace
git clone -b dl-prior-red https://github.com/brudfors/UniRes.git   # or: cd UniRes && git pull
cd UniRes
# fast: nitorch (the only compiled dep) is already baked, so skip deps + build isolation
NI_COMPILED_BACKEND=C pip install -e . --no-deps --no-build-isolation
```

## 7. Smoke test on the A100 (re-verify the compiled backend + a real recon)

```sh
python -c "import torch, nitorch, deepinv; print(torch.__version__, torch.cuda.is_available())"
from_ts_warning_should_be_absent=1   # watch stderr on the next import/recon
unires --help
# tiny GPU super-resolution on the bundled BrainWeb data:
unires --vx 1.0 data/t1_icbm_normal_1mm_pn0_rf0.nii.gz \
                data/t2_icbm_normal_1mm_pn0_rf0.nii.gz \
                data/pd_icbm_normal_1mm_pn0_rf0.nii.gz
# DL-prior path (downloads DRUNet weights via deepinv; use a mild mu):
unires --prior mtv+red --red_denoiser drunet --red_mu 1e-4 --red_sigma 0.03 \
       data/t1_icbm_normal_1mm_pn0_rf0.nii.gz data/t2_icbm_normal_1mm_pn0_rf0.nii.gz
```

If any run prints `nitorch uses its non-compiled backend (TS)`, the compiled backend is not active —
that is a failure to fix (it is the reason this image exists), not an acceptable slow fallback.

## Notes

- Training deps (monai/h5py/tensorboard) are baked; add anything else with `pip install ... `
  (torch stays pinned at 2.6.0 — pass `-c` with `torch==2.6.0 torchvision==0.21.0` if a package
  tries to move it).
- The image bakes the environment only; edit/iterate on the `dl-prior-red` branch live with Claude
  Code — the editable install means changes take effect with no reinstall.
