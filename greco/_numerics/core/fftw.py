# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/fftw.cpp
#
# Original upstream notices (retained verbatim as comments):
# /***************************************************************************
#  *
#  * Author: "Sjors H.W. Scheres"
#  * MRC Laboratory of Molecular Biology
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * This complete copyright notice must be included in any revised version of the
#  * source code. Additional authorship citations may be added, but existing
#  * author citations must be preserved.
#  ***************************************************************************/
#
# /***************************************************************************
#  *
#  * Authors:    Roberto Marabini					(roberto@cnb.csic.es)
#  *			   Carlos Oscar S. Sorzano			(coss@cnb.csic.es)
#  *
#  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  *	All comments concerning this program package may be sent to the
#  *	e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
#
# Greco port and adaptations:
# Changes: Python adaptation of FFT normalization and FourierTransformer conventions.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""FFTW-backed CPU Fourier transforms — bit-exact stand-in for the GT
``FourierTransformer`` CPU path (src/fftw.cpp).

The GT binary builds its plans with ``FFTW_ESTIMATE`` on 16-byte
aligned buffers and executes with the same ``libfftw3`` this module
links to, so for a given (ndim, N) the transform is bit-identical.

Conventions (GT ``setReal`` / ``Transform``):
* forward 2D: ``r2c`` of (N, N) real -> (N, N//2+1) complex, followed
  by the GT tComplex ``/= N*N`` normalisation, which is a complex
  division by ``(N*N + 0i)`` -- see :func:`gt_forward_normalize`;
* backward 2D: unnormalised ``c2r`` back into the real buffer.
"""

from __future__ import annotations

import numpy as _np

from ._fft_dispatch import transform as _transform


def gt_forward_normalize(out: _np.ndarray, size: int) -> None:
    """Normalise a raw FFTW forward spectrum exactly like the GT
    ``FourierTransformer::Transform(FFTW_FORWARD)`` (src/fftw.cpp)::

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fFourier)
            DIRECT_MULTIDIM_ELEM(fFourier, n) /= size;

    The element type is ``tComplex<RFLOAT>`` which defines no
    ``operator/=(scalar)`` member, so the scalar is implicitly converted
    to ``tComplex(size, 0)`` and the complex-division member
    (src/jaz/single_particle/t_complex.h) applies::

        cd = size*size + 0*0
        re = real*size + imag*0
        im = imag*size - real*0
        real = re/cd
        imag = im/cd

    This is NOT element-wise ``real/size``: for non-power-of-two sizes
    the two differ by 1 ulp on roughly a quarter of the elements
    (verified 664704/664704 bit-exact against the GT binary for a
    1152x1152 r2c).
    """
    s = _np.float64(size)
    cd = s * s + _np.float64(0.0)
    re = out.real * s + out.imag * _np.float64(0.0)
    im = out.imag * s - out.real * _np.float64(0.0)
    out.real = re / cd
    out.imag = im / cd


def fourier_transform_2d(real2d: _np.ndarray,
                         out: _np.ndarray | None = None) -> _np.ndarray:
    """GT ``FourierTransformer::FourierTransform(img(), F, false)`` for
    a 2D image: r2c followed by the GT tComplex ``/= N*N``
    normalisation (see :func:`gt_forward_normalize`)."""
    n = real2d.shape[0]
    if real2d.shape[1] != n:
        raise ValueError("fourier_transform_2d: input must be square")
    flat = _np.ascontiguousarray(real2d, dtype=_np.float64)
    if out is None:
        out = _np.empty((n, n // 2 + 1), dtype=_np.complex128)
    else:
        out = _np.ascontiguousarray(out, dtype=_np.complex128)
    _transform(flat, out, (n, n), inverse=False)
    gt_forward_normalize(out, n * n)
    return out


def fourier_transform_3d(real3d: _np.ndarray,
                         out: _np.ndarray | None = None) -> _np.ndarray:
    """GT ``FourierTransformer::Transform(FFTW_FORWARD)`` for a cubic 3D
    volume (src/fftw.cpp:333): r3c followed by the GT tComplex
    ``/= N*N*N`` normalisation (see :func:`gt_forward_normalize`)."""
    if len(real3d.shape) != 3 or real3d.shape[0] != real3d.shape[1]\
            or real3d.shape[1] != real3d.shape[2]:
        raise ValueError("fourier_transform_3d: input must be cubic (N, N, N)")
    n = real3d.shape[0]
    flat = _np.ascontiguousarray(real3d, dtype=_np.float64)
    if out is None:
        out = _np.empty((n, n, n // 2 + 1), dtype=_np.complex128)
    else:
        out = _np.ascontiguousarray(out, dtype=_np.complex128)
    _transform(flat, out, (n, n, n), inverse=False)
    gt_forward_normalize(out, n * n * n)
    return out


def inverse_fourier_transform_3d(comp3d: _np.ndarray,
                                 out: _np.ndarray) -> _np.ndarray:
    """GT ``inverseFourierTransform()`` (3D): unnormalised c3r into the
    real buffer."""
    n = comp3d.shape[0]
    flat = _np.ascontiguousarray(comp3d, dtype=_np.complex128)
    out = _np.ascontiguousarray(out, dtype=_np.float64)
    _transform(flat, out, (n, n, n), inverse=True)
    return out


def inverse_fourier_transform_2d(comp2d: _np.ndarray,
                                 out: _np.ndarray) -> _np.ndarray:
    """GT ``inverseFourierTransform()`` (2D): unnormalised c2r into the
    real buffer."""
    n = comp2d.shape[0]
    flat = _np.ascontiguousarray(comp2d, dtype=_np.complex128)
    out = _np.ascontiguousarray(out, dtype=_np.float64)
    _transform(flat, out, (n, n), inverse=True)
    return out


def gt_forward_normalize_f32(out: _np.ndarray, size: int) -> None:
    """F32 twin of :func:`gt_forward_normalize` for the GT
    ``NewFFT::_FourierTransform<float>`` FwdOnly branch (new_ft.cpp)::

        const float scale = MULTIDIM_SIZE(src);
        for (i) dest.data[i] /= scale;

    ``fComplex operator/=(tComplex<float>(scale, 0))`` keeps the zero
    terms; every step rounds in single precision.
    """
    s = _np.float32(size)
    cd = s * s + _np.float32(0.0)
    re = out.real * s + out.imag * _np.float32(0.0)
    im = out.imag * s - out.real * _np.float32(0.0)
    out.real = re / cd
    out.imag = im / cd


def fourier_transform_2d_f32(real2d: _np.ndarray,
                             out: _np.ndarray | None = None) -> _np.ndarray:
    """GT ``NewFFT::FourierTransform(Iwork(), Fref)`` (float overload,
    ad-hoc FFTW_ESTIMATE plan, FwdOnly) for a rectangular (ny, nx) f32
    image: fftwf r2c followed by the tComplex f32 ``/= ny*nx``
    normalisation (see :func:`gt_forward_normalize_f32`)."""
    ny, nx = real2d.shape
    flat = _np.ascontiguousarray(real2d, dtype=_np.float32)
    if out is None:
        out = _np.empty((ny, nx // 2 + 1), dtype=_np.complex64)
    else:
        out = _np.ascontiguousarray(out, dtype=_np.complex64)
    _transform(flat, out, (ny, nx), inverse=False)
    gt_forward_normalize_f32(out, ny * nx)
    return out


def inverse_fourier_transform_2d_f32(comp2d: _np.ndarray,
                                     out: _np.ndarray) -> _np.ndarray:
    """GT ``NewFFT::inverseFourierTransform(Fbinned, Iwork())`` (float
    overload, FwdOnly): unnormalised fftwf c2r into the f32 real
    buffer.  ``comp2d`` has shape (ny, nx//2+1), ``out`` shape (ny, nx).
    """
    ny, nxc2 = comp2d.shape
    flat = _np.ascontiguousarray(comp2d, dtype=_np.complex64)
    out = _np.ascontiguousarray(out, dtype=_np.float32)
    if out.ndim != 2 or out.shape[0] != ny or out.shape[1] // 2 + 1 != nxc2 or not out.size:
        raise ValueError("inverse_fourier_transform_2d_f32: incompatible output grid")
    # The half-spectrum cannot distinguish odd/even real widths. The supplied
    # real output buffer determines the width without guessing or underfilling.
    nx = out.shape[1]
    _transform(flat, out, (ny, nx), inverse=True)
    return out
