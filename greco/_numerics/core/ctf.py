# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/ctf.h
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
#  * Authors: Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#  *
#  * Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  * GNU General Public License for more details.
#  *
#  e You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  * All comments concerning this program package may be sent to the
#  * e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
#
# Greco port and adaptations:
# Changes: Python translation and numerical integration of CTF routines.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""reference CTF -- port of src/ctf.h (getCTF / draw / initialise).

Frequencies are in 1/A, unshifted layout (DC at [0,0]), matching reference's
half-spectrum convention used throughout the tomo code.
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import numerical_error
from ..libm_compat import sincos, sin_array, exp_array
from typing import Optional

import numpy as np

__all__ = ["CTF", "lambda_from_kv"]

PI = np.pi
_CTF_CPP = "greco-source/numerical-reference/src/ctf.cpp"


def lambda_from_kv(kV: float) -> float:
    """Electron wavelength in Angstrom (non-relativistic correction as in
    CTF::initialise)."""
    local_kV = kV * 1e3
    return 12.2643247 / np.sqrt(local_kV * (1.0 + local_kV * 0.978466e-6))


@dataclass
class CTF:
    """reference CTF with astigmatism expressed as a bilinear form."""

    defocus_u: float = 0.0
    defocus_v: float = 0.0
    defocus_angle: float = 0.0
    kv: float = 300.0
    cs: float = 2.7
    q0: float = 0.1
    bfac: float = 0.0
    scale: float = 1.0
    phase_shift: float = 0.0
    dose: float = -1.0

    _cache: Optional[dict] = None

    def initialise(self) -> "CTF":
        # Exact op-order of CTF::initialise (src/ctf.cpp:211)
        local_cs = self.cs * 1e7
        lam = lambda_from_kv(self.kv)
        rad_az = (self.defocus_angle * PI) / 180.0   # DEG2RAD(azimuthal_angle)
        self._lam = lam
        self._k1 = (PI / 2) * 2 * lam
        self._k2 = ((PI / 2) * local_cs) * lam * lam * lam
        self._k3 = np.arctan(self.q0 / np.sqrt(1.0 - self.q0 * self.q0))\
            if self.q0 < 1.0 else 0.0
        self._k4 = -self.bfac / 4.0
        self._k5 = (self.phase_shift * PI) / 180.0   # DEG2RAD(phase_shift)
        # GT CTF::initialise (src/ctf.cpp:241/244): the two REPORT_ERRORs
        # are checked whenever the numerical CTF is initialized.
        if self.q0 < 0.0 or self.q0 > 1.0:
            numerical_error(
                _CTF_CPP, 242,
                "CTF::initialise ERROR: AmplitudeContrast Q0 cannot be "
                "smaller than zero or larger than one!")
        if (abs(self.defocus_u) < 1e-6 and abs(self.defocus_v) < 1e-6
                and abs(self.q0) < 1e-6 and abs(self.cs) < 1e-6):
            numerical_error(
                _CTF_CPP, 245,
                "CTF::initialise: ERROR: CTF initialises to all-zero "
                "values. Was a correct STAR file provided?")
        # C++ CTF::initialise (src/ctf.cpp:249): sin/cos pair fused
        # into glibc sincos (GT binary: SINCOS=1).
        sa, ca = sincos(rad_az)
        # A = Qt * D * Q, elementwise in the C++ d2Matrix operator* order
        # Qt = [[ca, -sa], [sa, ca]], D = diag(-du, -dv), Q = [[ca, sa], [-sa, ca]]
        du, dv = self.defocus_u, self.defocus_v
        q00 = ca * (-du) + (-sa) * 0.0
        q01 = ca * 0.0 + (-sa) * (-dv)
        q10 = sa * (-du) + ca * 0.0
        q11 = sa * 0.0 + ca * (-dv)
        self._axx = q00 * ca + q01 * (-sa)
        self._axy = q00 * sa + q01 * ca
        self._ayy = q10 * sa + q11 * ca
        return self

    # ------------------------------------------------------- astigmatism API

    @property
    def lambda_(self) -> float:
        """reference ``CTF::lambda`` (electron wavelength, Angstrom)."""
        if getattr(self, "_lam", None) is None:
            self.initialise()
        return self._lam


    def get_ctf(self, x: np.ndarray, y: np.ndarray,
                do_damping: bool = True,
                intact_until_first_peak: bool = False,
                intact_after_first_peak: bool = False,
                gamma_offset: float = 0.0,
                do_abs: bool = False,
                do_only_flip_phases: bool = False) -> np.ndarray:
        if self._cache is None:
            self.initialise()
        u2 = x * x + y * y
        u4 = u2 * u2
        # C++ getCTF: gamma = K1*(...) + K2*u4 - K5 - K3 + gammaOffset
        gamma = (self._k1 * (self._axx * x * x + 2.0 * self._axy * x * y
                             + self._ayy * y * y)
                 + self._k2 * u4 - self._k5 - self._k3
                 + gamma_offset)
        # GT build: RFLOAT=double -> glibc double sin/exp/pow per element
        if isinstance(gamma, np.ndarray):
            _sin = sin_array
            _exp = exp_array
        else:
            _sin = math.sin
            _exp = math.exp
        if intact_until_first_peak or intact_after_first_peak:
            cond = (intact_until_first_peak & (np.abs(gamma) < PI / 2))\
                | (intact_after_first_peak & (np.abs(gamma) > PI / 2))
            out = np.where(cond, 1.0, -_sin(gamma))
        else:
            out = -_sin(gamma)
        if do_damping:
            if self.dose >= 0.0:
                # Niko Grigorieff's formulae, as in src/ctf.h
                # C++ pow(0, -0.8325) = inf -> d0 = inf -> E = 1 (reproduce exactly)
                if isinstance(u2, np.ndarray):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        d0 = 0.245 * np.power(u2, -0.8325) + 2.81
                elif u2 == 0.0:
                    d0 = float("inf")
                else:
                    d0 = 0.245 * math.pow(u2, -0.8325) + 2.81
                e = _exp(-0.5 * self.dose / d0)
            else:
                e = _exp(self._k4 * u2)
            out = out * e
        if do_abs:
            # GT: abs applied after damping, before scale/clamp
            out = np.abs(out)
        elif do_only_flip_phases:
            out = np.where(out < 0.0, -1.0, 1.0)
        out = out * self.scale
        # GT SGN(x) = ((x) >= 0) ? 1 : -1 (src/macros.h:143): an exactly
        # zero (or -0.0) CTF value is pushed to +1e-8, never -1e-8.
        tiny = np.abs(out) < 1e-8
        sgn = np.where(out >= 0.0, 1.0, -1.0)
        out = np.where(tiny, sgn * 1e-8, out)
        if isinstance(out, np.ndarray) and out.ndim == 0:
            out = float(out)
        return out


    # ------------------------------------------------- Ewald-sphere helpers


    def get_fftwhimage(self, result: np.ndarray, orixdim: int,
                       oriydim: int, angpix: float,
                       do_abs: bool = False,
                       do_only_flip_phases: bool = False,
                       do_intact_until_first_peak: bool = False,
                       do_damping: bool = True,
                       do_ctf_padding: bool = False,
                       do_intact_after_first_peak: bool = False) -> None:
        """GT ``CTF::getFftwImage`` (src/ctf.cpp:314) into the FFTW
        half grid of ``result`` (shape ``(oriydim, orixdim//2+1)``,
        float64), in place.

        Non-padded branch: elementwise ``getCTF`` on the FFTW grid
        frequencies of ``FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D``
        (fftw.h:107): ``x = jp/xs`` with ``jp = j`` and
        ``y = ip/ys`` with ``ip = i if i < sh else i - oriydim``
        (``sh = orixdim//2+1``).  The magnification matrix of the
        observation model is applied first, exactly like GT's
        ``getCTF``.
        """
        if do_ctf_padding:
            self._get_fftwhimage_padded(
                result, orixdim, oriydim, angpix, do_abs,
                do_only_flip_phases, do_intact_until_first_peak,
                do_damping, do_intact_after_first_peak)
            return
        if self._cache is None:
            self.initialise()
        xs = float(orixdim) * angpix
        ys = float(oriydim) * angpix
        sh = orixdim // 2 + 1
        jp = np.arange(sh, dtype=np.float64)
        # GT FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D: ip = (i < xdim) ? i
        # : i - ydim, where xdim is the HALF width (sh).
        iarr = np.arange(oriydim, dtype=np.float64)
        ip = np.where(iarr < sh, iarr, iarr - oriydim)
        x = jp[None, :] / xs
        y = ip[:, None] / ys
        obs = getattr(self, "obs", None)
        if obs is not None and obs.has_mag_matrices:
            M = obs.mag_matrices[getattr(self, "optics_group", 0)]
            xd = M[0][0] * x + M[0][1] * y
            yd = M[1][0] * x + M[1][1] * y
            x, y = xd, yd
        result[...] = self.get_ctf(
            x, y, do_damping=do_damping,
            intact_until_first_peak=do_intact_until_first_peak,
            intact_after_first_peak=do_intact_after_first_peak,
            gamma_offset=0.0, do_abs=do_abs,
            do_only_flip_phases=do_only_flip_phases)

    def _get_fftwhimage_padded(self, result, orixdim, oriydim, angpix,
                               do_abs, do_only_flip_phases,
                               do_intact_until, do_damping,
                               do_intact_after) -> None:
        """GT ``CTF::getFftwImage`` padded branch (src/ctf.cpp:321).

        ``Mctf`` is an xmipp-origin array: ``A2D_ELEM(Mctf, ip, jp)``
        maps to the direct cell ``(ip + oy/2, jp + ox/2)``; the half
        spectrum (``jp >= 0``) therefore lives in the right half and
        the mirrored part in the left half.
        """
        from .fourier_resize import resize_map_2d_f64
        obs = getattr(self, "obs", None)
        ctf_premultiplied = (
            obs is not None
            and obs.get_ctf_premultiplied(getattr(self, "optics_group", 0)))
        orixdim_pad = 2 * orixdim
        oriydim_pad = 2 * oriydim
        if ctf_premultiplied:
            orixdim_pad *= 2
            oriydim_pad *= 2

        shp = orixdim_pad // 2 + 1
        Fctf = np.empty((oriydim_pad, shp), dtype=np.float64)
        self.get_fftwhimage(
            Fctf, orixdim_pad, oriydim_pad, angpix, do_abs,
            do_only_flip_phases, do_intact_until, do_damping, False,
            do_intact_after)

        # From half to whole (xmipp-origin full image)
        Mctf = np.zeros((oriydim_pad, orixdim_pad), dtype=np.float64)
        oyh = oriydim_pad // 2
        oxh = orixdim_pad // 2
        for i in range(oriydim_pad):
            if i == oyh:
                continue
            ip = i if i < shp else i - oriydim_pad
            row_a = oyh + ip
            row_b = oyh - ip
            for j in range(shp - 1):
                v = Fctf[i, j]
                val = v * v if ctf_premultiplied else v
                Mctf[row_a, oxh + j] = val
                Mctf[row_b, oxh - j] = val

        Mctf = resize_map_2d_f64(Mctf, orixdim)

        # From whole to half (Mctf is orixdim x orixdim, xmipp origin)
        shr = orixdim // 2 + 1
        oy_r, ox_r = Mctf.shape
        oyh_r = oy_r // 2
        oxh_r = ox_r // 2
        for i in range(oriydim):
            if i == oriydim // 2:
                continue
            ip = i if i < shr else i - oriydim
            for j in range(shr - 1):
                val = Mctf[oyh_r + ip, oxh_r + j]
                if ctf_premultiplied:
                    if val < 0.0:
                        val = 0.0
                    elif val > 1.0:
                        val = 1.0
                    else:
                        val = math.sqrt(val)
                result[i, j] = val

    def draw(self, w0: int, h0: int, angpix: float) -> np.ndarray:
        """Draw the CTF on an (h0, w0/2+1) half-spectrum, reference draw() layout."""
        if self._cache is None:
            self.initialise()
        xs = w0 * angpix
        ys = h0 * angpix
        x = np.arange(w0 // 2 + 1, dtype=float) / xs
        # reference folds the PIXEL INDEX first, then divides:
        #   yy = (y < h0/2 ? y : y - h0) / ys      (ctf.h CTF::draw)
        yi = np.arange(h0, dtype=float)
        yi = np.where(yi < h0 // 2, yi, yi - h0)
        y = yi / ys
        xx, yy = np.meshgrid(x, y)
        return self.get_ctf(xx, yy, do_damping=True)
