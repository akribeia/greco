"""Checked direct FFT dispatch through Greco's shared numerical core."""
import numpy as np
from ..native.loader import load


def transform(source, destination, shape, *, inverse=False):
    shape = tuple(int(n) for n in shape)
    if len(shape) not in (2, 3) or any(n <= 0 for n in shape):
        raise ValueError('FFT requires a nonempty 2D or 3D grid.')
    single = source.dtype in (np.dtype('float32'), np.dtype('complex64'))
    real_dtype = np.dtype('float32' if single else 'float64')
    complex_dtype = np.dtype('complex64' if single else 'complex128')
    complex_shape = (*shape[:-1], shape[-1] // 2 + 1)
    if source.shape != (complex_shape if inverse else shape) or destination.shape != (shape if inverse else complex_shape):
        raise ValueError('Incompatible FFT buffers.')
    if source.dtype != (complex_dtype if inverse else real_dtype) or destination.dtype != (real_dtype if inverse else complex_dtype):
        raise ValueError('Incompatible FFT buffer precision.')
    if not destination.flags.writeable:
        raise ValueError('FFT output buffer is read-only.')
    lib = load()
    if lib is None:
        raise RuntimeError('Build the Greco numerical core: python -m greco._numerics.native.build')
    source = np.ascontiguousarray(source)
    output = destination if destination.flags.c_contiguous else np.empty_like(destination, order='C')
    d, h, w = (1, *shape) if len(shape) == 2 else shape
    lib.greco_fft_raw(source.ctypes.data, output.ctypes.data, w, h, d, len(shape), int(single), int(inverse))
    if output is not destination:
        np.copyto(destination, output)
    return destination
