"""Radial template weighting derived from a forward reconstructed point.

The existing angular wedge is retained. This changes its radial Fourier RMS,
not the tomogram, the available views, or the template's angular search.
"""
from collections import OrderedDict
import numpy as np

_RESPONSE_CACHE = OrderedDict()


def _point_response(shape, metadata, pixel):
    from ..helpers import _impulse_projections
    from ..tomography import gpu_reconstruct_wbp
    from .spectral import ctf_transfer, radial_grid

    if len(shape) != 3 or len(set(shape)) != 1:
        raise ValueError('Impulse spectral weighting requires a cubic template.')
    angles = np.asarray(metadata.tilt_angles, float)
    if not metadata.angles_in_degrees:
        angles = np.rad2deg(angles)
    key = (tuple(shape), tuple(angles), float(pixel),
           str(metadata.ctf_data), str(metadata.dose_accumulation))
    if key in _RESPONSE_CACHE:
        _RESPONSE_CACHE.move_to_end(key)
        return _RESPONSE_CACHE[key]
    n = shape[0]
    size = 2 * n + 1
    projections = _impulse_projections(angles, size).copy()
    for i in range(len(angles)):
        transfer = np.ones((size, size // 2 + 1), np.float64)
        if metadata.ctf_data is not None:
            transfer *= ctf_transfer((size, size), pixel * 1e-10, metadata.ctf_data[i])
        if metadata.dose_accumulation is not None:
            frequency_squared = (radial_grid((size, size)) / (2 * pixel)) ** 2
            displacement_squared = metadata.dose_accumulation[i] * 4 / (8 * np.pi ** 2)
            transfer *= np.exp(-2 * np.pi ** 2 * displacement_squared * frequency_squared)
        projections[i] = np.fft.irfftn(
            np.fft.rfftn(projections[i]) * transfer,
            s=(size, size), axes=(0, 1)).real
    # Standard Ram-Lak WBP, canonical Z and a template-sized central crop.
    response = gpu_reconstruct_wbp(projections, angles)[::-1]
    first = size // 2 - n // 2
    crop = np.ascontiguousarray(response[first:first+n, first:first+n, first:first+n].T)
    # A real multiplier symmetrizes the finite crop about its periodic origin.
    result = np.fft.rfftn(np.fft.ifftshift(crop)).real
    result /= max(float(np.max(np.abs(result))), 1e-12)
    result[radial_grid(shape) > 1.] = 0
    result = result.astype(np.float32)
    _RESPONSE_CACHE[key] = result
    while len(_RESPONSE_CACHE) > 4:
        _RESPONSE_CACHE.popitem(last=False)
    return result


def radial_weighting(base, shape, metadata, pixel):
    """Match point-response shell power while preserving the original wedge.

    Global volume leveling does not change an ideal radial power spectrum;
    its directional geometry remains represented by the existing base wedge.
    """
    response = _point_response(shape, metadata, pixel)
    n = shape[0]
    frequency = np.fft.fftfreq(n) * n
    radius = np.sqrt(frequency[:, None, None] ** 2 + frequency[None, :, None] ** 2
                     + np.arange(n // 2 + 1)[None, None, :] ** 2)
    shell = np.rint(radius).astype(int)
    multiplicity = np.full(n // 2 + 1, 2.)
    multiplicity[0] = 1.
    if n % 2 == 0:
        multiplicity[-1] = 1.
    multiplicity = np.broadcast_to(multiplicity, response.shape)
    desired = np.bincount(shell.ravel(), weights=(response * response * multiplicity).ravel())
    current = np.bincount(shell.ravel(), weights=(base * base * multiplicity).ravel())
    scale = np.sqrt(desired / np.maximum(current, 1e-20))
    scale[current <= 1e-20] = 0
    return (base * scale[shell]).astype(np.float32)
