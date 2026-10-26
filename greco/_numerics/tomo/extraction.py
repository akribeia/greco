# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/jaz/tomography/extraction.h
#
# The upstream header has no individual author notice.
# Upstream attribution: RELION contributors.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# Greco port and adaptations:
# Changes: Python translation and integration of particle extraction routines.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""Particle extraction from a tilt stack -- port of
``jaz/tomography/extraction.h`` (the Fourier path used by
reconstruct_particle).

Conventions:

* ``centers`` are the projected 2-D positions (pixels, image coords)
  of the particle for each frame;
* ``integralShift = (round(cx) - s/2, round(cy) - s/2)``;
* the small stack is a per-frame integer-offset window (edge voxels
  clamped, invisible frames zero);
* ``circle_crop`` subtracts the outside-mean and applies a cosine
  edge falloff (``EDGE_FALLOFF = 5``);
* the 2-D FFT of the centered stack is a roll by (w/2, h/2) followed by
  a plain FFTW half transform with ``Both`` normalisation
  (== numpy ``rfft2(..., norm="ortho")``);
* the final phase shift implements the sub-pixel centering so the
  particle centre lands exactly at the box centre.
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations


import numpy as np
from .fft_f32 import rfft2_both
from ..libm_compat import cos_array, sincos_array

PI = np.pi

__all__ = ['EDGE_FALLOFF', 'crop_circle', 'fourier_transform_stack_fast_centered', 'shift_stack']

EDGE_FALLOFF = 5.0


def _seq_outside_mean(sl: np.ndarray, outside: np.ndarray) -> float:
    """C++ ``cropCircle`` outside-mean: plain sequential ``sum += c * v``
    in the (y outer, x inner) loop order, float64 accumulation (np.sum's
    pairwise blocking is NOT bit-identical)."""
    a = sl.astype(np.float64) * outside
    return float(np.add.accumulate(a.ravel(), dtype=np.float64)[-1])


def crop_circle(stack: np.ndarray, boundary: float = 0.0,
                falloff: float = EDGE_FALLOFF) -> None:
    """reference ``TomoExtraction::cropCircle`` (in place, (fc, h, w))."""
    fc, h, w = stack.shape
    mx = w / 2.0
    my = h / 2.0
    crop_rad = mx - boundary
    x = np.arange(w, dtype=np.float64) - mx
    y = np.arange(h, dtype=np.float64) - my
    r = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2)
    outside = r > crop_rad
    n_out = int(outside.sum())
    # GT ``0.5 - 0.5*cos(PI*(crop_rad-r)/falloff)``: glibc f64 cos
    # (numpy cos is not bit-identical).
    with np.errstate(invalid="ignore"):
        band = 0.5 - 0.5 * cos_array(PI * (crop_rad - r) / falloff)
    c = np.where(r < crop_rad - falloff, 1.0,
                 np.where(r < crop_rad, band, 0.0))
    for f in range(fc):
        sl = stack[f]
        mean_out = _seq_outside_mean(sl, outside) / n_out if n_out > 0 else 0.0
        sl[...] = (c * (sl.astype(np.float64) - mean_out)).astype(
            sl.dtype, copy=False)


def fourier_transform_stack_fast_centered(stack: np.ndarray) -> np.ndarray:
    """reference ``NewStackHelper::FourierTransformStack_fast`` (center=True).

    stack: (fc, h, w) f32 -> (fc, wh, h) complex64 (FFTW half, x first).
    Per frame: roll by (h/2, w/2) then the f32 r2c with reference ``Both``
    normalisation -- the native engine calls the GT plan (bit-exact).
    """
    fc, h, w = stack.shape
    wh = w // 2 + 1
    out = np.empty((fc, wh, h), dtype=np.complex64)
    for f in range(fc):
        rolled = np.roll(stack[f], (-h // 2, -w // 2), axis=(0, 1))
        out[f] = rfft2_both(rolled).T
    return out


def shift_stack(stack_fs: np.ndarray, pos: np.ndarray,
                centered: bool = True) -> np.ndarray:
    """reference ``NewStackHelper::shiftStack`` (f32, bit-exact).

    stack_fs: (fc, wh, h) complex64 half.  pos: (fc, 2) f64.

    GT: ``phi = mx*x + my*y`` (f64); ``z = tComplex<float>(cos(phi),
    sin(phi))`` -- GCC fuses the adjacent cos/sin into one glibc
    ``sincos`` (verified by objdump on the GT binary); the f32 store
    rounds each component; ``stackOut = z * stack`` is the tComplex
    f32 multiply (each product f32, sum/diff f32).
    """
    fc, wh, h = stack_fs.shape
    w = 2 * (wh - 1)
    xx = np.arange(wh, dtype=np.float64)                   # (wh,)
    y = np.arange(h, dtype=np.float64)
    yy = np.where(y < h / 2.0, y, y - h)[None, :]          # (1, h)
    tx = (pos[:, 0] - (w / 2.0 if centered else 0.0))
    ty = (pos[:, 1] - (h / 2.0 if centered else 0.0))
    mx = 2.0 * PI * tx / w                                  # (fc,)
    my = 2.0 * PI * ty / h                                  # (fc,)
    phi = (mx[:, None, None] * xx[None, :, None]
           + my[:, None, None] * yy[None, :, :])
    st, ct = sincos_array(phi)
    zre = ct.reshape(fc, wh, h).astype(np.float32)
    zim = st.reshape(fc, wh, h).astype(np.float32)
    sre = np.ascontiguousarray(stack_fs.real, dtype=np.float32)
    sim = np.ascontiguousarray(stack_fs.imag, dtype=np.float32)
    re = (zre * sre - zim * sim).astype(np.float32)
    im = (zre * sim + zim * sre).astype(np.float32)
    return re + 1j * im


# ------------------------------------------------------------------ f32 path
