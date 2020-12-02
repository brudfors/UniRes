from datetime import datetime
from nitorch.spatial import voxel_size
from nitorch.core.math import round
from nitorch.io import (map, savef)
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


def _print_info(info, sett, *argv):
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
        elif info == 'init-reg':
            if argv[1] == 'begin':
                print('\nPerforming ', end='')
                if argv[0] == 'atlas':
                    print('atlas ', end='')
                elif argv[0] == 'co':
                    print('multi-channel ', end='')
                print('alignment...', end='')
            elif argv[1] == 'finished':
                print('completed in {:0.5f} seconds.'.format(timer() - argv[2]))
        elif info == 'fix-affine':
            if argv[0] > 0:
                print('\nFixed affine of {} CT image(s).'.format(argv[0]))
        elif info == 'crop':
            if argv[0] > 0 and sett.bound != 'full':
                print('\nCropped FOV of {} input images.'.format(argv[0]))
    if sett.do_print >= 2:
        if info in 'reg-param':
            print('Rigid registration fit:')
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print('c={} n={} | q={}'.format(c, n, round(argv[0][c][n].rigid_q, 4).tolist()))
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


def _read_image(data, device='cpu', is_ct=False):
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
        file (io.BabelArray)
        ct (bool): Is data CT

    """
    if isinstance(data, str):
        # =================================
        # Load from file
        # =================================
        file = map(data)
        dat = file.fdata(dtype=torch.float32, device=device,
                         rand=True, cutoff=(0.0005, 0.9995))
        mat = file.affine.to(device).type(torch.float64)
        fname = file.filename()
        direc, nam = os.path.split(fname)
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
        dat[~torch.isfinite(dat)] = 0
        # Add some random noise
        torch.manual_seed(0)
        dat[dat > 0] += torch.rand_like(dat[dat > 0]) - 1 / 2
        # Affine matrix
        mat = data[1]
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)
        mat = mat.double().to(device)
        file = None
        fname = None
        direc = None
        nam = None
    # Get dimensions
    dim = tuple(dat.shape)
    # CT?
    if _is_ct(dat):
        ct = True
    else:
        ct = False
    # Mask
    dat[~dat.isfinite()] = 0.0

    return dat, dim, mat, fname, direc, nam, file, ct


def _read_label(x, pth, sett):
    """Read labels and add to input struct.
    """
    # Load labels
    file = map(pth)
    dat = file.fdata(dtype=torch.float32, device=sett.device)
    mat = file.affine.type(torch.float64).to(sett.device)
    # Sanity check
    if not torch.equal(torch.as_tensor(x.dim), torch.as_tensor(dat.shape)):
        raise ValueError('Incorrect label dimensions.')
    if torch.any(x.mat - mat > 1e-4):
        raise ValueError('Incorrect label affine matrix.')
    # Append labels
    x.label = [dat, file]

    return x


def _write_image(dat, ofname, mat=torch.eye(4), file=None,
                 dtype='float32'):
    """ Write data to nifti.
    """
    savef(dat, ofname, like=file, affine=mat)


def _is_ct(dat):
    """Is image a CT scan?
    """
    ct = False
    nm = dat.numel()
    if torch.sum(dat < -990) > 0.01*nm:
        ct = True

    return ct