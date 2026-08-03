import torch


class _input:
    """ Algorithm input.
    """

    def __init__(self):
        self.dat = None
        self.dim = None
        self.ct = None
        self.mat = None
        self.mu = 1.0
        self.po = None
        self.sd = 1.0
        self.tau = 1.0
        self.file = None
        self.fname = None
        self.direc = None
        self.nam = None
        self.rigid_q = None
        self.label = None


class _output:
    """ Algorithm output.
    """
    def __init__(self):
        self.dat = None
        self.dim = None
        self.lam = None
        self.mat = None
        self.label = None


class _proj_op:
    """ Encodes a projection operator.
    """
    def __init__(self):
        self.dim_x = None
        self.mat_x = None
        self.vx_x = None
        self.dim_y = None
        self.mat_y = None
        self.vx_y = None
        self.dim_yx = None
        self.mat_yx = None
        self.ratio = None
        self.smo_ker = None
        self.rigid = None
        self.scl = None
        self.dim_thick = None
        self.D_x = None
        self.D_y = None


class settings:
    """ Algorithm settings.
    """
    def __init__(self):
        self.alpha: float = 1.0  # Relaxation parameter 0 < alpha < 2, alpha < 1: under-relaxation, alpha > 1: over-relaxation
        self.atlas_rigid: bool = False  # Rigid or rigid+isotropic scaling alignment to atlas
        self.bids: bool = False  # For adding a BIDS compatible space tag ('_space-unires_')
        self.bound: str = 'zero'  # Boundary conditions (see nitorch.spatial)
        self.cgs_max_iter: int = 20  # Max conjugate gradient (CG) iterations for solving for y
        self.cgs_tol: float = 1e-3  # CG tolerance for solving for y
        self.cgs_verbose: bool = False  # CG verbosity (0, 1)
        # Deep-learning prior (RED / denoiser-as-prior, issue: DL prior). Defaults reproduce
        # the classic multi-channel TV behaviour exactly (prior='mtv', red_mu=0).
        self.prior: str = 'mtv'  # Image prior: 'mtv' | 'red' | 'mtv+red' | 'mtv+gradnet'
        self.red_mu: float = 0.0  # RED denoiser-prior weight (mu); 0 disables the DL term
        self.red_auto_scale: bool = True  # mu is dimensionless: scale it to the per-channel data term (tau) so a global mu can't be catastrophically mis-set. False = old absolute mu.
        self.red_sigma: float = 0.05  # RED denoiser strength (noise level, standardised [0,1] units)
        self.red_sigma_end: float = None  # if set, anneal red_sigma -> this over max_iter (strong denoise early, gentle late); None = constant
        self.red_sigma_auto: bool = False  # NOT recommended: derive red_sigma from the noise estimate. Tested (IXI PD, 3-20% noise) and it HURTS -- noise adaptation is ALREADY handled by the dimensionless mu (mu_c=red_mu*tau_c ~ 1/noise^2), so this double-adapts, and it over-smooths (observation noise > residual noise in y). Keep red_sigma fixed.
        self.red_nullspace: float = 0.5  # NULL-SPACE-RESTRICTED RED: fraction of the denoiser's correction to REMOVE in frequency bands the data already measures (per-axis cutoff from po.ratio). 0 = classic RED; 1 = the prior may only synthesise detail the data cannot see. Rationale: letting the prior rewrite MEASURED bands blurs real structure -> dPSNR>0 but dSSIM<0. VALIDATED DEFAULT 0.5: both dPSNR AND dSSIM positive on every 2/4mm IXI cell while keeping ~80% of the PSNR gain; it is a SOLVER setting and helps every prior (the real-data prior's 6mm T1 goes -0.05 -> +1.03). No effect unless RED is active.
        self.red_thick_taper: bool = False  # resolution-adaptive strength: taper the DL prior toward 0 as slice thickness (po.ratio) exceeds red_thick_full, reaching 0 by red_thick_zero. VALIDATED DEFAULT False: with red_nullspace>0 the prior no longer degrades thick slices, and the taper then only slides mu down a steep response curve (measured ~0.3-0.5 dB loss on well-posed cells, incl. the recoverable-detail case). Set True if you need a hard never-hurt guarantee on very thick input.
        self.red_thick_full: float = 4.0  # thickness factor (low/high-res voxel ratio) up to which the DL prior is applied at FULL strength
        self.red_thick_zero: float = 8.0  # thickness factor at/above which the DL prior is fully OFF (linear taper between full and zero)
        self.red_sigma_scale: float = 1.0  # multiplier on the auto-estimated noise sigma (red_sigma_auto only)
        self.red_denoiser: str = 'gaussian'  # Denoiser backend: 'gaussian' (analytic) | 'drunet' (pretrained 2.5D CNN) | 'biasfree3d' (native-3D bias-free CNN, needs red_weights)
        self.red_weights: str = None  # Path to denoiser weights (None -> backend default/download)
        self.red_tile: int = 0  # biasfree3d: overlap-tile inference at this size (0 = whole-volume pass)
        self.red_linesearch: bool = True  # Backtracking line search to keep the objective monotone (also improves quality)
        self.red_ls_perchannel: bool = False  # Per-channel (coordinate) line search: each channel steps only if it helps -> all channels improve-or-hold (protects high-contrast channels the joint step would drag)
        self.red_batch: int = 16  # Slice mini-batch for 2.5D denoiser application (memory control)
        # Merged-gradient DL prior (issue: DL prior; docs/merged_gradient_prior.md). Acts on the
        # channel-pooled gradient magnitude m in the JTV z-update: ONE network call per ADMM
        # iteration, C-independent, replacing the analytic soft-threshold gain with a learned/
        # generalized one and broadcasting it back to every channel/direction verbatim. Defaults
        # (grad_net='softthr') reproduce classic MTV bit-for-bit; nothing runs unless prior
        # includes 'gradnet'.
        self.grad_form: str = 'pointwise'  # Gain support: 'pointwise' (scalar shrinkage curve) | 'cnn' (spatial 3D CNN) | 'structure' (structure-tensor anisotropic)
        self.grad_net: str = 'softthr'  # Gain backend: 'softthr' (== exact JTV) | 'charbonnier' (analytic edge-preserving) | 'unet' | 'dncnn' (learned 3D magnitude denoisers)
        self.grad_weights: str = None  # Path to grad-net weights (None -> backend default/untrained)
        self.grad_sigma: float = 0.05  # Learned-gain noise level (standardised units); analytic gains ignore it
        self.grad_scale: float = 1.0  # Scale on the Charbonnier knee (grad_net='charbonnier'); 1.0 -> JTV at the sharp limit
        self.grad_gain_clamp = (0.0, 1.0)  # Clamp the (learned) per-voxel gain to this range (no hallucination); analytic softthr is untouched
        self.grad_apply: str = 'magnitude'  # Merged field the net sees: 'magnitude' (K=1 pooled) | 'perdir' (K=3) | 'structure' (K=6)
        self.grad_patch: int = 0  # Sliding-window patch size for the 3D grad-net (0 = whole-volume pass)
        self.grad_linesearch: bool = True  # Backtracking line search on the reported objective for PnP (non-prox) grad-nets; keeps the reported objective monotone
        self.clean_fov: bool = False  # Set voxels outside of low-res FOV, projected in high-res space, to zero
        self.coreg_params = {'cost_fun': 'nmi', 'group': 'SE', 'samp': (1), 'fwhm': 7, 'mean_space': False}  # parameters for coregistration
        self.crop: bool = False  # Crop input images' FOV to brain in the NITorch atlas
        self.common_output: bool = False  # Makes recons aligned with same grid, across subjects
        self.ct: bool = False  # Data could be CT (if contain negative values)
        self.device: str = 'cuda'  # PyTorch device name
        self.diff: str = 'forward'  # Gradient difference operator (forward|backward|central)
        self.dir_out: str = None  # Directory to write output, if None uses same as input (output prefix set with 'self.prefix')
        self.do_coreg: bool = True  # Coregistration of input images
        self.do_atlas_align: bool = False  # Align images to an atlas space
        self.do_print: int = 1  # Print progress to terminal (0, 1, 2, 3)
        self.do_proj: bool = None  # Use projection matrices, defined in format_output()
        self.do_res_origin: bool = False  # Resets origin, if CT data
        self.fix: int = 0  # Fixed image index for atlas alignment
        self.force_inplane_res: bool = False  # Force in-plane resolution of observed data to be greater or equal to recon vx
        self.fov: str = 'brain'  # If crop=True, uses this field-of-view ('brain'|'head').
        self.gap: float = 0.0  # Slice gap, between 0 and 1
        self.interpolation: str = 'linear'  # Interpolation order (see nitorch.spatial)
        self.label: tuple = None  # Manual labels, given as (str, (int, int))),
        # where the first element is the path and the second element are the
        # channel and repeat indices, respectively.
        self.mat: torch.Tensor = None  # Observed image(s) affine matrix. OBS: Data needs to be given as 4D array
        self.max_iter: int = 512  # Max algorithm iterations
        self.method = None  # Method name (super-resolution|denoising), defined in format_output()
        self.plot_conv: bool = False  # Use matplotlib to plot convergence in real-time
        self.pow: int = 0  # Ensure output image dimensions are a power of two or three, with max dimensions pow
        self.prefix: str = 'u_'  # Prefix for reconstructed image(s)
        self.profile_ip: int = 2  # In-plane slice profile (0=rect|1=tri|2=gauss)
        self.profile_tp: int = 0  # Through-plane slice profile (0=rect|1=tri|2=gauss)
        self.reg_scl: float = 4.0  # Scale regularisation estimate (for coarse-to-fine scaling, give as list of floats)
        self.rho: float = None  # ADMM step-size, if None -> estimate is made
        self.rho_scl: float = 1.0  # Scaling of ADMM step-size
        self.rigid_basis = None  # Rigid transformation basis, defined in init_reg()
        self.rigid_mod: int = 1  # Update rigid every rigid_mod iteration
        self.rigid_samp: int = 1  # Level of sub-sampling for estimating rigid registration parameters
        self.scaling: bool = False  # Optimise even/odd slice scaling
        self.sched_num: int = 3  # Number of coarse-to-fine scalings
        self.show_hyperpar: bool = False  # Use matplotlib to visualise hyper-parameter estimates
        self.show_jtv: bool = False  # Show the joint total variation (JTV)
        self.tolerance: float = 1e-4  # Algorithm tolerance, if zero, run to max_iter
        self.unified_rigid: bool = False  # Do unified rigid registration
        self.vx: float = 1.0  # Reconstruction voxel size (use 0 or None to just denoise)
        self.write_jtv: bool = False  # Write JTV to nifti
        self.write_out: bool = True  # Write reconstructed output images
