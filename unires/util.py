import contextlib
from datetime import datetime
import nibabel as nib
from nitorch.spatial import voxel_size
from nitorch.core.math import round
import os
from timeit import default_timer as timer
import torch


_unires_title = r"""
  _   _       _ ____           
 | | | |_ __ (_)  _ \ ___  ___ 
 | | | | '_ \| | |_) / _ \/ __|
 | |_| | | | | |  _ <  __/\__ \
  \___/|_| |_|_|_| \_\___||___/
"""


def print_info(info, sett, *argv):
    """ Print algorithm info to terminal.

    Args:
        info (string): What to print.

    """
    if not sett.do_print:
        return 0

    if sett.do_print >= 1:
        if info == 'init':
            print(_unires_title)
            if type(sett.device) is torch.device:
                print('GPU: ' + torch.cuda.get_device_name(0) + ', CUDA: ' + str(torch.cuda.is_available()))
            else:
                print('CPU')
        elif info == 'fit-finish':
            print(' {} finished in {:0.5f} seconds and '
                  '{} iterations\n'.format(sett.method, timer() - argv[0], argv[1] + 1))
        elif info in 'fit-ll':
            print('{:3} - Convergence ({} | {:0.1f} s)  | nlyx={:0.4f}, nlxy={:0.4f}, nly={:0.4f} '
                  'gain={:0.7f}'.format(argv[1], argv[0], timer() - argv[4], argv[2][0], argv[2][1], argv[2][2],
                                        argv[3]))
        elif info == 'fit-start':
            print('\nStarting {} (update_rigid={}, update_scaling={}) \n{} | C={} | N={} | device={} | '
                  'max_iter={} | tol={}'.format(sett.method, sett.unified_rigid, sett.scaling,
                                                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                argv[0], argv[1], argv[2], argv[3], argv[4]))
        elif info in 'step_size':
            print('\nADMM step-size={:0.4f}'.format(argv[0]))
        elif info == 'hyper_par':
            if len(argv) == 2:
                print('completed in {:0.5f} seconds:'.format(timer() - argv[1]))
                for c in range(len(argv[0])):
                    print('c={} | tau='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:0.7f}'.format(argv[0][c][n].tau), end=' ')
                    print('| sd='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:0.7f}'.format(argv[0][c][n].sd), end=' ')
                    print('| mu='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:0.7f}'.format(argv[0][c][n].mu), end=' ')
                    print('| ct='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{}'.format(argv[0][c][n].ct), end=' ')
                    print()
            else:
                print('\nEstimating model hyper-parameters...', end='')
        elif info == 'mean-space':
            vx_y = voxel_size(argv[1])
            vx_y = tuple(vx_y.tolist())
            print('\nMean space | dim={}, vx_y={}'.format(argv[0], vx_y))
    if sett.do_print >= 2:
        if info in 'reg-param':
            print('Rigid registration fit:')
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print('c={} n={} | q={}'.format(c, n, round(argv[0][c][n].rigid_q, 4).cpu().tolist()))
        elif info in 'scl-param':
            print('Scale fit:')
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print('c={} n={} | exp(s)={}'.format(c, n, round(argv[0][c][n].po.scl.exp(), 4)))
    if sett.do_print >= 3:
        if info == 'fit-done':
            print('(completed in {:0.5f} seconds)'.format(timer() - argv[0]))
        elif info == 'fit-update':
            print('{:3} - Updating {:2}   | '.format(argv[1] + 1, argv[0]), end='')
        elif info == 'int':
            print('{}'.format(argv[0]), end=' ')

    return timer()


def read_image(data, device='cpu', is_ct=False):
    """ Reads image data.

    Args:
        data (string|list): Path to file, or list with image data and affine matrix.
        device (string, optional): PyTorch on CPU or GPU? Defaults to 'cpu'.
        is_ct (bool, optional): Is the image a CT scan?

    Returns:
        dat (torch.tensor()): Image data.
        dim (tuple(int)): Image dimensions.
        mat (torch.tensor(double)): Affine matrix.
        fname (string): File path
        direc (string): File directory path
        nam (string): Filename
        head (nibabel.nifti1.Nifti1Header)
        ct (bool): Is data CT?
        var (torch.tensor(float)): Observation uncertainty.

    """
    var = torch.tensor(0, dtype=torch.float32, device=device)  # Observation uncertainty
    if isinstance(data, str):
        # =================================
        # Load from file
        # =================================
        nii = nib.load(data)
        # Get affine matrix
        mat = nii.affine
        mat = torch.tensor(mat).double().to(device)
        # Get image data
        dat = torch.tensor(nii.get_fdata()).float().to(device)
        # Get header, filename, etc
        head = nii.get_header()
        fname = nii.get_filename()
        # Get input directory and filename
        direc, nam = os.path.split(fname)
        # Get observation uncertainty
        slope = nii.dataobj.slope
        dtype = nii.get_data_dtype()
        dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        if dtype in dtypes:
            var = torch.tensor(slope, dtype=torch.float32, device=device)
            var = var ** 2 / 12
    else:
        # =================================
        # Data and matrix given as list
        # =================================
        # Image data
        dat = data[0]
        if not isinstance(dat, torch.Tensor):
            dat = torch.tensor(dat)
        dat = dat.float()
        dat = dat.to(device)
        # Affine matrix
        mat = data[1]
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)
        mat = mat.double().to(device)
        head = None
        fname = None
        direc = None
        nam = None
    # Get dimensions
    dim = tuple(dat.shape)
    # Remove NaNs
    dat[~torch.isfinite(dat)] = 0
    if is_ct and (torch.min(dat) < 0):
        # Input data is CT
        ct = True
        # Winsorize CT
        dat[dat < -1024] = -1024
        dat[dat > 3071] = 3071
    else:
        ct = False

    return dat, dim, mat, fname, direc, nam, head, ct, var


def write_image(dat, ofname, mat=torch.eye(4), header=None, dtype='float32'):
    """ Writes 3D nifti data using nibabel.

    Args:
        dat (torch.tensor): Image data (W, H, D).
        ofname (str): Output filename.
        mat (torch.tensor, optional): Affine matrix (4, 4), defaults to identity.
        header (nibabel.nifti1.Nifti1Header, optional): nibabel header, defaults to None.
        dtype (str, optional): Output data type, defaults to 'float32', but uses the data type
            in the header (if given).
    """
    # Sanity check
    if dtype not in ['float32', 'uint8', 'int16', 'uint16']:
        raise ValueError('Undefined data type')
    # Get min and max
    mn = torch.min(dat)
    mx = torch.max(dat)
    if header is not None:
        # If input was integer type, make output integer type
        dtype = header.get_data_dtype()
        dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        if dtype in dtypes:
            dat = dat.int()
    # Make nii object
    nii = nib.Nifti1Image(dat.cpu().numpy(), header=header, affine=mat.cpu().numpy())
    if header is None:
        # Set data type
        header = nii.get_header()
        header.set_data_dtype(dtype)
        # Set offset, slope and intercept
        if dtype == 'float32':
            offset = 0
            slope = 1
            inter = 0
        elif dtype == 'uint8':
            offset = 0
            slope = (mx / 255).cpu().numpy()
            inter = 0
        elif dtype == 'int16':
            offset = 0
            slope = torch.max(mx / 32767, -mn / 32768).cpu().numpy()
            inter = 0
        elif dtype == 'uint16':
            offset = 0
            slope = torch.max(mx / 65535, -mn / 65535).cpu().numpy()
            inter = 0
        header.set_data_offset(offset=offset)
        header.set_slope_inter(slope=slope, inter=inter)
    # Write to  disk
    with contextlib.suppress(FileNotFoundError):
        os.remove(ofname)
    nib.save(nii, ofname)
