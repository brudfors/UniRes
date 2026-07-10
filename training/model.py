"""Re-export the bias-free 3D denoiser network.

The network lives in the ``unires`` package (``unires/_denoiser_net.py``) so that
inference (the RED 'biasfree3d' backend in ``unires/_denoiser.py``) is
self-contained, and training reuses the exact same definition.
"""
from unires._denoiser_net import BiasFreeUNet3D, _ResBlock, _conv  # noqa: F401
