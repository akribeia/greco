"""Explicit per-call CPU/GPU registration kernels, isolated between threads."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from scipy.ndimage import shift, gaussian_filter
from scipy.signal import fftconvolve
from skimage.registration import phase_cross_correlation

_CPU = SimpleNamespace(shift=shift, gaussian_filter=gaussian_filter,
    fftconvolve=fftconvolve, phase_cross_correlation=phase_cross_correlation,
    overlap_score=None)
_CURRENT = ContextVar('greco_registration_backend', default=_CPU)


def kernels():
    return _CURRENT.get()


@contextmanager
def registration_backend(gpu=False):
    if gpu:
        from . import align2d_gpu as module
        backend = SimpleNamespace(shift=module.gpu_ndi_shift,
            gaussian_filter=module.gpu_gaussian_filter,
            fftconvolve=module.gpu_fftconvolve,
            phase_cross_correlation=module.gpu_phase_cross_correlation,
            overlap_score=module.gpu_overlap_score)
    else:
        backend = _CPU
    token = _CURRENT.set(backend)
    try:
        yield
    finally:
        _CURRENT.reset(token)
