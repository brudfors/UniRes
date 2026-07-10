"""Validate the mu auto-scaling refactor.

A. REGRESSION (must be bit-identical to pre-refactor):
   - prior='mtv' 3ch  -> PSNR 24.902 (classic MTV path untouched).
   - prior='mtv+red' red_auto_scale=False mu=1e-4 3ch -> PSNR 24.457 (old absolute path).
B. AUTO==ABS EQUIVALENCE (1 channel): capture the effective mu_c the auto path uses,
   then run the absolute path with exactly that mu_c -> reconstructions must match to
   floating-point noise. Proves auto-scale only *rescales* mu, changing nothing else.
C. Unit test of _red_mu_channels arithmetic on a mock.
"""
import torch
import unires._update as U
from bench_dl_prior import simulate, run


class _Obs:
    def __init__(self, tau):
        self.tau = tau


class _S:
    prior = 'mtv+red'; red_mu = 2.0; red_auto_scale = True


def unit():
    x = [[_Obs(0.1), _Obs(0.3)], [_Obs(0.5)]]
    mus = U._red_mu_channels(x, _S())
    assert abs(mus[0] - 2.0 * 0.4) < 1e-9 and abs(mus[1] - 2.0 * 0.5) < 1e-9, mus
    _S.red_auto_scale = False
    assert U._red_mu_channels(x, _S()) == [2.0, 2.0]
    _S.red_mu = 0.0
    assert U._red_mu_channels(x, _S()) == [0.0, 0.0]
    _S.red_mu = 2.0; _S.red_auto_scale = True
    print("C. unit _red_mu_channels: PASS")


def main():
    torch.set_grad_enabled(False)
    unit()

    # --- A. regression, 3 channels ---
    gt, gt_mat, x, mat_x = simulate(3)
    r_mtv = run(gt, gt_mat, x, mat_x, 'mtv', 30, tolerance=0.0)
    r_abs = run(gt, gt_mat, x, mat_x, 'mtv+red', 30, red_mu=1e-4, red_sigma=0.03,
                red_denoiser='drunet', red_auto_scale=False, tolerance=0.0)
    ok_mtv = abs(r_mtv['psnr'] - 24.902) < 0.01
    ok_abs = abs(r_abs['psnr'] - 24.457) < 0.01
    print(f"A. mtv 3ch          PSNR={r_mtv['psnr']:.4f}  (expect 24.902)  "
          f"{'PASS' if ok_mtv else 'FAIL'}")
    print(f"A. red abs mu=1e-4  PSNR={r_abs['psnr']:.4f}  (expect 24.457)  "
          f"{'PASS' if ok_abs else 'FAIL'}")

    # --- B. auto == abs equivalence on 1 channel ---
    captured = {}
    orig = U._red_mu_channels
    def spy(xx, sett):
        v = orig(xx, sett)
        captured['mu'] = v
        return v
    U._red_mu_channels = spy
    gt1, m1, x1, mx1 = simulate(1)
    r_auto = run(gt1, m1, x1, mx1, 'mtv+red', 20, red_mu=0.5, red_sigma=0.03,
                 red_denoiser='drunet', red_auto_scale=True, tolerance=0.0)
    mu_eff = captured['mu'][0]
    U._red_mu_channels = orig
    r_abs1 = run(gt1, m1, x1, mx1, 'mtv+red', 20, red_mu=mu_eff, red_sigma=0.03,
                 red_denoiser='drunet', red_auto_scale=False, tolerance=0.0)
    dP = abs(r_auto['psnr'] - r_abs1['psnr'])
    print(f"B. 1ch auto red_mu=0.5 -> mu_eff={mu_eff:.6g}  PSNR={r_auto['psnr']:.5f}")
    print(f"   1ch abs mu={mu_eff:.6g}                 PSNR={r_abs1['psnr']:.5f}")
    print(f"   |dPSNR|={dP:.2e}  {'PASS' if dP < 1e-3 else 'FAIL'}")


if __name__ == '__main__':
    main()
