from dataclasses import dataclass
import torch


@dataclass
class Input:
    """ Algorithm input.

    """
    dat = None
    dim = None
    ct = None
    mat = None
    mu = None
    po = None
    sd = None
    tau = None
    head = None
    fname = None
    direc = None
    nam = None
    rigid_q = None


@dataclass
class Output:
    """ Algorithm output.

    """
    dat = None
    dim = None
    lam = None
    mat = None


@dataclass
class ProjOp:
    """ Encodes a projection operator.

    """
    dim_x = None
    mat_x = None
    vx_x = None
    dim_y = None
    mat_y = None
    vx_y = None
    dim_yx = None
    mat_yx = None
    ratio = None
    smo_ker = None
    rigid = None
    scl = None
    dim_thick = None
    D_x = None
    D_y = None


@dataclass
class Settings:
    """ Algorithm settings.

    """
    alpha: float = 1.0  # Relaxation parameter 0 < alpha < 2, alpha < 1: under-relaxation, alpha > 1: over-relaxation
    cgs_max_iter: int = 128  # Max conjugate gradient (CG) iterations for solving for y
    cgs_tol: float = 1e-2  # CG tolerance for solving for y
    cgs_verbose: bool = False  # CG verbosity (0, 1)
    device: str = 'cuda'  # PyTorch device name
    gr_diff: str = 'forward'  # Gradient difference operator (forward|backward|central)
    dir_out: str = None  # Directory to write output, if None uses same as input (output is prefixed 'y_')
    do_proj = None  # Use projection matrices, defined in format_output()
    gap: float = 0.0  # Slice gap, between 0 and 1
    has_ct: bool = False  # Data could be CT (but data must contain negative values)
    mat: torch.Tensor = None  # Observed image(s) affine matrix. OBS: Data needs to be given as 4D array
    max_iter: int = 512  # Max algorithm iterations
    method = None  # Method name (super-resolution|denoising), defined in format_output()
    mod_prct: float = 0.0  # Amount to crop mean space, between 0 and 1 (faster, but could loss out on data)
    prefix: str = 'y_'  # Prefix for reconstructed image(s)
    do_print: int = 1  # Print progress to terminal (0, 1, 2)
    plot_conv: bool = False  # Use matplotlib to plot convergence in real-time
    profile_ip: int = 0  # In-plane slice profile (0=rect|1=tri|2=gauss)
    profile_tp: int = 0  # Through-plane slice profile (0=rect|1=tri|2=gauss)
    reg_scl: float = 20.0  # Scale regularisation estimate (for coarse-to-fine scaling, give as list of floats)
    rho: float = None  # ADMM step-size, if None -> estimate is made
    rho_scl: float = 1.0  # Scaling of ADMM step-size
    rigid_basis = None  # Rigid transformation basis, defined in init_reg()
    rigid_mod: int = 8  # Update rigidt every rigid_mod iteration
    rigid_sched_max: int = 13  # Start scaling at 2^rigid_sched_max
    rigid_samp: int = 1  # Level of sub-sampling for estimating rigid registration parameters
    scaling: bool = True  # Optimise even/odd slice scaling
    show_hyperpar: bool = False  # Use matplotlib to visualise hyper-parameter estimates
    show_jtv: bool = False  # Show the joint total variation (JTV)
    tolerance: float = 0.5*1e-4  # Algorithm tolerance, if zero, run to max_iter
    unified_rigid: bool = True  # Do unified rigid registration
    vx: float = 1.0  # Reconstruction voxel size (if None, set automatically)
    write_jtv: bool = False  # Write JTV to nifti
    write_out: bool = True  # Write reconstructed output images
