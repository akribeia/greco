# Third-party references and adapted components
# Greco GPU adaptations of SciPy FFT convolution and scikit-image phase_cross_correlation/upsampled DFT routines, with Greco registration and device integration.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/scipy/LICENSE.txt
# licenses/scikit-image/LICENSE.txt
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# Copyright: 2009-2022 the scikit-image team
# Copyright: 2007-2010 the Sphinx team
# Copyright: 2003-2009 Massachusetts Institute of Technology
# Copyright: 2009-2015 Board of Regents of the University of
# Copyright: 2003-2009 Massachusetts Institute of Technology
# Copyright: 2003-2009 Massachusetts Institute of Technology
# Copyright: 2020 Broad Institute
# Copyright: 1994 Karel Zuiderveld
# Copyright: 2003-2009 Massachusetts Institute of Technology
# Copyright: 2013 The IPython Development Team
# Copyright: 2009-2011 Pierre Raybaut
# Copyright: 2014 2014 Dan Oneata
# Copyright: 2005-2023, NumPy Developers
# Copyright: 2007 Zachary Pincus
# Full license conditions and disclaimers are retained at the paths above.
#

"""GPU drop-in replacements for the hot CPU kernels of greco.align2d_robust.

Reimplemented (same algorithm, same call contract, numpy in / numpy out):
  - scipy.ndimage.shift          (order=1, mode="constant", prefilter=False)
  - scipy.ndimage.gaussian_filter(mode="reflect")
  - scipy.signal.fftconvolve     (real float, 2-D, mode="full")
  - skimage.registration.phase_cross_correlation (real space, no masks,
    upsampled DFT refinement, normalization "phase" or None)
  - align2d_robust._overlap_score

Only cupy / cupyx / numpy / scipy.fft.next_fast_len are required.
SciPy and scikit-image reference notices are retained under
licenses/scipy/ and licenses/scikit-image/.
"""
from __future__ import annotations

import numpy as np
import cupy as cp
from scipy.fft import next_fast_len
from cupyx.scipy.ndimage import gaussian_filter as _cux_gaussian


def _as_gpu(image, dtype=np.float32):
    if isinstance(image, cp.ndarray):
        return image if image.dtype == dtype else image.astype(dtype)
    return cp.asarray(image, dtype=dtype)


def gpu_ndi_shift(image, shift, order=1, mode="constant", cval=0.0,
                  prefilter=True, output=None):
    """Bilinear shift with constant fill; mirrors scipy.ndimage.shift.

    Two sequential 1-D passes (y then x) with 2-tap weights, exactly like
    the scipy constant-mode spline order-1 path.
    """
    if order != 1 or mode != "constant" or output is not None:
        raise NotImplementedError("gpu shift supports only order=1 mode=constant")
    was_cupy = isinstance(image, cp.ndarray)
    data = _as_gpu(image)
    dy = float(shift[0])
    dx = float(shift[1])
    height, width = data.shape
    # coordinates in float64 (like scipy), weights uploaded as float32
    fy64 = np.arange(height, dtype=np.float64) - dy
    fx64 = np.arange(width, dtype=np.float64) - dx
    y0 = np.floor(fy64).astype(np.int32)
    x0 = np.floor(fx64).astype(np.int32)
    wy1 = cp.asarray((fy64 - y0).astype(np.float32))
    wx1 = cp.asarray((fx64 - x0).astype(np.float32))
    y0 = cp.asarray(y0)
    x0 = cp.asarray(x0)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1
    fill = cp.float32(cval)

    # scipy constant mode tests the float64 source centre, before converting
    # interpolation weights to float32; out-of-image centres are pure cval.
    y_valid = cp.asarray((fy64 >= 0) & (fy64 <= height-1))
    x_valid = cp.asarray((fx64 >= 0) & (fx64 <= width-1))
    y0c = cp.clip(y0, 0, height - 1)
    y1c = cp.clip(y0 + 1, 0, height - 1)
    x0c = cp.clip(x0, 0, width - 1)
    x1c = cp.clip(x0 + 1, 0, width - 1)
    tmp = wy0[:, None] * data[y0c, :] + wy1[:, None] * data[y1c, :]
    tmp = cp.where(y_valid[:, None], tmp, fill)
    out = wx0[None, :] * tmp[:, x0c] + wx1[None, :] * tmp[:, x1c]
    out = cp.where(x_valid[None, :], out, fill)
    return out if was_cupy else out.get()


def gpu_gaussian_filter(input, sigma, output=None, mode="reflect",
                        cval=0.0, truncate=4.0):
    if output is not None:
        raise NotImplementedError('GPU Gaussian output buffers are not supported')
    was_cupy = isinstance(input, cp.ndarray)
    out = _cux_gaussian(_as_gpu(input), sigma, mode=mode, cval=cval,
                        truncate=truncate)
    return out if was_cupy else out.get()


def gpu_fftconvolve(in1, in2, mode="full", axes=None):
    """Real 2-D full FFT convolution, mirroring scipy.signal.fftconvolve:
    pad each axis to next_fast_len(n1+n2-1, real=True), rfftn, multiply,
    irfftn, crop to the full size.
    """
    if mode != "full" or axes is not None:
        raise NotImplementedError("gpu fftconvolve supports only mode='full'")
    a = _as_gpu(in1)
    b = _as_gpu(in2)
    if a.ndim != 2 or b.ndim != 2:
        raise NotImplementedError("gpu fftconvolve supports 2-D inputs")
    out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    fshape = (next_fast_len(out_shape[0], real=True),
              next_fast_len(out_shape[1], real=True))
    sp1 = cp.fft.rfftn(a, fshape, axes=(0, 1))
    sp2 = cp.fft.rfftn(b, fshape, axes=(0, 1))
    ret = cp.fft.irfftn(sp1 * sp2, fshape, axes=(0, 1))
    crop = ret[:out_shape[0], :out_shape[1]]
    return crop if isinstance(in1, cp.ndarray) else crop.get()


def _upsampled_dft_gpu(data, upsampled_region_size, upsample_factor, axis_offsets):
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must match data.ndim")
    if axis_offsets is None:
        axis_offsets = [0] * data.ndim
    elif len(axis_offsets) != data.ndim:
        raise ValueError("number of axis offsets must match data.ndim")
    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))
    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        freqs = np.fft.fftfreq(n_items, d=upsample_factor)
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * freqs[None, :]
        kernel = np.exp(-1j * 2.0 * np.pi * kernel).astype(data.dtype, copy=False)
        data = cp.tensordot(cp.asarray(kernel), data, axes=(1, -1))
    return data


def _compute_error(cross_correlation_max, src_amp, target_amp):
    amp = src_amp * target_amp
    with np.errstate(invalid="ignore"):
        error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / amp
    return np.sqrt(np.abs(error))


def _compute_phasediff(cross_correlation_max):
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def gpu_phase_cross_correlation(reference_image, moving_image, *, upsample_factor=1,
                                space="real", disambiguate=False,
                                reference_mask=None, moving_mask=None,
                                overlap_ratio=0.3, normalization="phase"):
    """Port of skimage.registration.phase_cross_correlation (real space,
    no masks, FFT + local matrix-multiply DFT refinement).
    """
    if reference_mask is not None or moving_mask is not None or disambiguate:
        raise NotImplementedError("masked pcc is not supported on GPU")
    ref = _as_gpu(reference_image)
    mov = _as_gpu(moving_image)
    if ref.shape != mov.shape:
        raise ValueError("images must be same shape")
    if space.lower() != "real":
        raise NotImplementedError("only space='real' supported")
    src_freq = cp.fft.fftn(ref)
    target_freq = cp.fft.fftn(mov)
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product = image_product / cp.maximum(cp.abs(image_product), 100.0 * eps)
    elif normalization is None:
        pass
    else:
        raise ValueError("normalization must be either phase or None")
    cross_correlation = cp.fft.ifftn(image_product)
    maxima = np.unravel_index(int(cp.argmax(cp.abs(cross_correlation)).get()),
                              cross_correlation.shape)
    midpoint = np.array([np.trunc(axis_size / 2) for axis_size in shape])
    float_dtype = np.float32
    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(shape)[shift > midpoint]
    if upsample_factor == 1:
        src_amp = float(cp.sum(cp.real(src_freq * src_freq.conj())).get()) / src_freq.size
        target_amp = float(cp.sum(cp.real(target_freq * target_freq.conj())).get()) / target_freq.size
        CCmax = cross_correlation[maxima].get()
    else:
        upsample_factor = np.array(upsample_factor, dtype=float_dtype)
        shift = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dftshift = np.trunc(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor
        cc_up = _upsampled_dft_gpu(image_product.conj(), upsampled_region_size,
                                   upsample_factor, sample_region_offset).conj()
        cc_up_np = cc_up.get()
        idx = np.unravel_index(int(np.argmax(np.abs(cc_up_np))), cc_up_np.shape)
        CCmax = cc_up_np[idx]
        maxima = np.stack(idx).astype(float_dtype, copy=False)
        maxima -= dftshift
        shift = shift + maxima / upsample_factor
        # The local DFT is an unnormalised sum; unlike ifftn above it has
        # no 1/N factor. Its error estimate must use unnormalised energies.
        src_amp = float(cp.sum(cp.real(src_freq * src_freq.conj())).get())
        target_amp = float(cp.sum(cp.real(target_freq * target_freq.conj())).get())
    for dim in range(len(shape)):
        if shape[dim] == 1:
            shift[dim] = 0
    return shift, _compute_error(CCmax, src_amp, target_amp), _compute_phasediff(CCmax)


def gpu_overlap_score(reference, moving, shift_yx, *, prepared=False):
    """align2d_robust._overlap_score with GPU shift + support."""
    ref = _as_gpu(reference)
    mov = _as_gpu(moving)
    moved = gpu_ndi_shift(mov, shift_yx, order=1, mode="constant",
                          cval=0.0, prefilter=False)
    support = gpu_ndi_shift(cp.ones(mov.shape, dtype=cp.float32), shift_yx,
                            order=1, mode="constant", cval=0.0,
                            prefilter=False) > 0.999
    border = max(2, min(mov.shape) // 100)
    support[:border] = False
    support[-border:] = False
    support[:, :border] = False
    support[:, -border:] = False
    if float(support.mean()) < 0.25:
        return -1.0
    left = ref[support].get().astype(np.float64)
    right = moved[support].get().astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else -1.0
