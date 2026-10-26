# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/fftw.h, RELION src/fftw.cpp
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
#  * Authors:    Roberto Marabini	(roberto@cnb.csic.es)
#  *             Carlos Oscar S. Sorzano	(coss@cnb.csic.es)
#  *
#  * Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
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
#  * 02111-1307 USA
#  *
#  * All comments concerning this program package may be sent to the
#  * e-mail address 'xmipp@cnb.csic.es'
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
# Changes: Python translation and integration of Fourier resizing routines.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""Fourier resizing used by the numerical CTF implementation.

Retained RELION numerical port; see licenses/THIRD_PARTY_NOTICES.md.
"""
import numpy as np
from ..tomo.fft_f64 import rfft2_raw_f64, irfft2_raw_f64
from .fftw import gt_forward_normalize


def window_fourier_transform_2d(in_fs: np.ndarray, newdim: int) -> np.ndarray:
    """GT ``windowFourierTransform<T>`` (fftw.h:809), 2-D branch: crop
    the FFTW half-spectrum to ``newdim`` (zero-pad on upscale, dropping
    the corners outside the inscribed circle)."""
    h, wh = in_fs.shape
    newhdim = newdim // 2 + 1
    if newdim == h and newhdim == wh:
        return in_fs.copy()
    out = np.zeros((newdim, newhdim), dtype=np.complex128)
    if newhdim > wh:
        max_r2 = (wh - 1) * (wh - 1)
        ip = np.arange(h, dtype=np.int64)
        ip = np.where(ip < wh, ip, ip - h)
        oi = np.where(ip >= 0, ip, ip + newdim)
        jp = np.arange(wh, dtype=np.int64)
        mask = ip[:, None] ** 2 + jp[None, :] ** 2 <= max_r2
        out[oi, :wh] = np.where(mask, in_fs, 0.0)
    else:
        ip = np.arange(newdim, dtype=np.int64)
        ip = np.where(ip < newhdim, ip, ip - newdim)
        ii = np.where(ip >= 0, ip, ip + h)
        out[:] = in_fs[ii, :newhdim]
    return out


def resize_map_2d_f64(img: np.ndarray, newsize: int) -> np.ndarray:
    """GT ``resizeMap`` (fftw.cpp:1278) on a 2-D f64 image: raw forward
    FFTW, windowed crop, raw inverse. The forward normalisation is the
    GT-compiled ``fFourier[n] /= size`` (see
    :func:`greco._numerics.core.fftw.gt_forward_normalize`)."""
    s1 = img.shape[0]
    ft = rfft2_raw_f64(np.ascontiguousarray(img, dtype=np.float64))
    gt_forward_normalize(ft, s1 * s1)
    ft2 = window_fourier_transform_2d(ft, newsize)
    return irfft2_raw_f64(ft2, (newsize, newsize))
