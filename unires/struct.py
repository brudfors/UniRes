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
        self.clean_fov: bool = False  # Set voxels outside of low-res FOV, projected in high-res space, to zero
        self.coreg_params = {'cost_fun': 'nmi', 'group': 'SE', 'samp': (1), 'fwhm': 7, 'mean_space': False}  # parameters for coregistration
        self.crop: bool = False  # Crop input images' FOV to brain in the NITorch atlas
        self.common_output: bool = False  # Makes recons aligned with same grid, across subjects
        self.ct: bool = False  # Data could be CT (if contain negative values)
        self.device: str = 'cuda'  # PyTorch device name
        self.diff: str = 'forward'  # Gradient difference operator (forward|backward|central)
        self.dir_out: str = None  # Directory to write output, if None uses same as input (output is prefixed 'ur_')
        self.do_coreg: bool = True  # Coregistration of input images
        self.do_atlas_align: bool = False  # Align images to an atlas space
        self.do_print: int = 1  # Print progress to terminal (0, 1, 2, 3)
        self.do_proj: bool = None  # Use projection matrices, defined in format_output()
        self.do_res_origin: bool = False  # Resets origin, if CT data
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
        self.prefix: str = 'ur_'  # Prefix for reconstructed image(s)
        self.profile_ip: int = 2  # In-plane slice profile (0=rect|1=tri|2=gauss)
        self.profile_tp: int = 0  # Through-plane slice profile (0=rect|1=tri|2=gauss)
        self.reg_scl: float = 10.0  # Scale regularisation estimate (for coarse-to-fine scaling, give as list of floats)
        self.rho: float = None  # ADMM step-size, if None -> estimate is made
        self.rho_scl: float = 1.0  # Scaling of ADMM step-size
        self.rigid_basis = None  # Rigid transformation basis, defined in init_reg()
        self.rigid_mod: int = 1  # Update rigid every rigid_mod iteration
        self.rigid_samp: int = 1  # Level of sub-sampling for estimating rigid registration parameters
        self.scaling: bool = True  # Optimise even/odd slice scaling
        self.sched_num: int = 3  # Number of coarse-to-fine scalings
        self.show_hyperpar: bool = False  # Use matplotlib to visualise hyper-parameter estimates
        self.show_jtv: bool = False  # Show the joint total variation (JTV)
        self.tolerance: float = 1e-4  # Algorithm tolerance, if zero, run to max_iter
        self.unified_rigid: bool = True  # Do unified rigid registration
        self.vx: float = 1.0  # Reconstruction voxel size (use 0 or None to just denoise)
        self.write_jtv: bool = False  # Write JTV to nifti
        self.write_out: bool = True  # Write reconstructed output images
