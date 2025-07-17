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
            device = torch.device(sett.device)
            if device.type == 'cuda':
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' | GPU: ' + torch.cuda.get_device_name(device.index) + ', CUDA: ' + str(torch.cuda.is_available())
                       + ', PyTorch: ' + str(torch.__version__))
            else:
                assert device.type == 'cpu'
                print('CPU')
        elif info == 'fit-finish':
            print(' {} finished in {:0.5f} seconds and '
                  '{} iterations\n'.format(sett.method, timer() - argv[0], argv[1] + 1))
        elif info in 'fit-ll':
            nit = str(len(str(sett.max_iter)))
            print(('{:' + nit + 'd} - Convergence ({:4.1f} s)  | nlyx = {:10.4g}, nlxy = {:10.4g}, nly = {:10.4g}, '
                  'gain = {:10.7f}').format(argv[0], timer() - argv[3], argv[1][0], argv[1][1], argv[1][2],
                                        argv[2]))
        elif info == 'fit-start':
            print('\nStarting {} (update_rigid={}, update_scaling={}) \n | C={} | N={} | device={} | '
                  'max_iter={} | tol={} | sched_num={}'.format(
                sett.method, sett.unified_rigid, sett.scaling,
                                                argv[0], argv[1],
                                                sett.device,
                                                sett.max_iter,
                                                sett.tolerance,
                                                sett.sched_num))
        elif info in 'step_size':
            print('\nADMM step-size={:0.4f} | Regularisation scaling={}'.format(argv[0], sett.reg_scl.cpu()))
        elif info == 'filenames':
            print('')
            print('Input')
            nch = str(len(str(len(argv[0]))))
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print('c={:}, n={:} | fname={:}'.format(c, n, argv[0][c][n].fname))
        elif info == 'hyper_par':
            if len(argv) == 2:
                print('completed in {:0.5f} seconds:'.format(timer() - argv[1]))
                nch = str(len(str(len(argv[0]))))
                for c in range(len(argv[0])):
                    print(('c={:' + nch + 'd} | tau=').format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:10.4g}'.format(argv[0][c][n].tau), end=' ')
                    print('| sd='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:10.4g}'.format(argv[0][c][n].sd), end=' ')
                    print('| mu='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{:10.4g}'.format(argv[0][c][n].mu), end=' ')
                    print('| ct='.format(c, argv[0][c]), end='')
                    for n in range(len(argv[0][c])):
                        print('{}'.format(argv[0][c][n].ct), end=' ')
                    print()
            else:
                print('\nEstimating model hyper-parameters... ', end='')
        elif info == 'mean-space':
            vx_y = voxel_size(argv[1])
            vx_y = tuple(vx_y.tolist())
            vx_y = tuple([float('%4.2f' % val) for val in vx_y])
            print('\nMean space | dim={}, vx={}'.format(argv[0], vx_y))
        elif info == 'init-reg':
            if argv[1] == 'begin':
                print('\nPerforming ', end='')
                if argv[0] == 'atlas':
                    method = 'rigid' if sett.atlas_rigid else 'rigid+scale'
                    print(method + ' atlas ', end='')
                elif argv[0] == 'co':
                    print('multi-channel (N=' + str(argv[2]) + ') ', end='')
                print('alignment...', end='')
            elif argv[1] == 'finished':
                print('completed in {:0.5f} seconds.'.format(timer() - argv[3]))
        elif info == 'fix-affine':
            if argv[0] > 0:
                print('\nFixed affine of {} CT image(s).'.format(argv[0]))
        elif info == 'crop':
            if argv[0] > 0 and sett.bound != 'full':
                print('\nCropped FOV of {} input images.'.format(argv[0]))
    if sett.do_print >= 2:
        if info in 'reg-param':
            print('Rigid registration fit:')
            nch = str(len(str(len(argv[0]))))
            nrp = str(len(str(max(len(ch) for ch in argv[0]))))
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print(('c={:' + nch + 'd} n={:' + nrp + 'd} | q={}').format(c, n, round(argv[0][c][n].rigid_q, 4).tolist()))
        elif info in 'scl-param':
            print('Scale fit:')
            nch = str(len(str(len(argv[0]))))
            nrp = str(len(str(max(len(ch) for ch in argv[0]))))
            for c in range(len(argv[0])):
                for n in range(len(argv[0][c])):
                    print(('c={:' + nch + 'd} n={:' + nrp + 'd} | exp(s)={}').format(c, n, round(argv[0][c][n].po.scl.exp(), 4)))
    if sett.do_print >= 3:
        if info == 'fit-done':
            print('(completed in {:0.5f} seconds)'.format(timer() - argv[0]))
        elif info == 'fit-update':
            nit = str(len(str(sett.max_iter)))
            print(('{:' + nit + 'd} - Updating {:2}   | ').format(argv[1] + 1, argv[0]), end='')
        elif info == 'int':
            print('{}'.format(argv[0]), end=' ')

    return timer()


def _read_image(data, device='cpu', could_be_ct=False):
    """ Reads image data.

    Args:
        data (string|list): Path to file, or list with image data and affine matrix.
        device (string, optional): PyTorch on CPU or GPU? Defaults to 'cpu'.
        could_be_ct (bool, optional): Could the image be a CT scan?

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
                         rand=False, cutoff=None)
        mat = file.affine.to(device).type(torch.float64)
        fname = file.filename()
        direc, nam = os.path.split(os.path.abspath(fname))
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
    dat = dat.squeeze()
    dim = tuple(dat.shape)
    if len(dim) != 3:
        raise ValueError("Input image dimension required to be 3D, recieved {:}D!". \
            format(len(dim)))
    # CT?
    if could_be_ct and _is_ct(dat):
        ct = True
    else:
        ct = False
    # Mask
    dat[~torch.isfinite(dat)] = 0.0

    return dat, dim, mat, fname, direc, nam, file, ct


def _read_label(x, pth, sett):
    """Read labels and add to input struct.
    """
    # Load labels
    file = map(pth)
    dat = file.fdata(dtype=torch.float32, device=sett.device)
    # Sanity check
    if not torch.equal(torch.as_tensor(x.dim), torch.as_tensor(dat.shape)):
        raise ValueError('Incorrect label dimensions.')
    # Append labels
    x.label = [dat, file]

    return x


def _write_image(dat, fname, bids=False, mat=torch.eye(4), file=None,
                 dtype='float32'):
    """ Write data to nifti.
    """
    if bids:
        p, n = os.path.split(fname)
        s = n.split('_')
        fname = os.path.join(p, '_'.join(s[:-1] + ['space-unires'] + [s[-1]]))

    savef(dat, fname, like=file, affine=mat)


def _is_ct(dat):
    """Is image a CT scan?
    """
    ct = False
    nm = dat.numel()
    if torch.sum(dat < -990) > 0.01*nm:
        ct = True

    return ct
