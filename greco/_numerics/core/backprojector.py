# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/backprojector.cpp
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
# /*
#  * backprojector.cpp
#  *
#  *  Created on: 24 Aug 2010
#  *      Author: scheres
#  */
#
# Greco port and adaptations:
# Changes: Python translation and Greco integration of backprojection routines.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""BackProjector: exact port of BackProjector::backproject2Dto3D
(src/backprojector.cpp) with the Projector half-complex data layout
(z full, y full, x half; all centered as in MultidimArray + setXmippOrigin).

data/weight arrays: numpy (pad, pad, pad//2+1), index order (z, y, x) where
x is the half axis (x_phys = x, unsigned) and z/y are centered at pad//2.
This matches the layout used by core.projector.Projector.data, i.e.
my_data[i, j, k] == C++ DIRECT_A3D_ELEM(data, k, j, i).
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.
import math

import sys
import numpy as np

from .projector import _inv3, TRILINEAR, NEAREST_NEIGHBOUR
GT_SRC = "greco-source/numerical-reference/src"

PI = math.pi

class BackProjector:
    def __init__(self, ori_size: int, interpolator: int = TRILINEAR,
                 padding_factor: float = 2.0, r_min_nn: int = 10,
                 ref_dim: int = 3, data_dim: int = 2,
                 skip_gridding: bool = True,
                 blob_order: int = 0, blob_radius: float = 1.9,
                 blob_alpha: float = 15.0):
        self.ori_size = ori_size
        self.interpolator = interpolator
        self.padding_factor = float(np.float32(padding_factor))
        self.r_min_nn = r_min_nn
        self.ref_dim = ref_dim
        # Dimension of the DATA (2: 2D frames, 3: 3D subtomograms).
        # C++ BackProjector ctor stores _data_dim; used e.g. by the
        # windowToOridimRealSpace normfft factor.
        self.data_dim = data_dim
        # GT ctor (backprojector.h:78): store blob params; tab_ftblob is
        # initialised lazily with radius*2 (see _lazy_tab_ftblob).
        self.blob_order = blob_order
        self.blob_radius = blob_radius
        self.blob_alpha = blob_alpha
        # C++ default: skip_gridding = !--dont_skip_gridding (ml_optimiser.cpp:591)
        self.skip_gridding = skip_gridding
        self.r_max = ori_size // 2
        self.pad_size = 0
        self.data = None
        self.weight = None
        # Validate the reconstruction padding factor.
        from .errors import numerical_error
        if float(padding_factor) < 1.0:
            numerical_error(
                f"{GT_SRC}/backprojector.h", 102,
                "Padding factor cannot be less than 1.")

    # ------------------------------------------------------------------ #
    def initialise_data_and_weight(self, current_size: int = -1):
        if current_size < 0:
            self.r_max = self.ori_size // 2
        else:
            self.r_max = current_size // 2
        self.r_max = min(self.r_max, self.ori_size // 2)
        # GT Projector::initialiseData switch on ref_dim (projector.cpp:83)
        if self.ref_dim not in (2, 3):
            from .errors import numerical_error
            numerical_error(
                f"{GT_SRC}/projector.cpp", 83,
                "Projector::resizeData%%ERROR: Dimension of the data "
                "array should be 2 or 3")
        # pad_size = 2*(ROUND(padding_factor*r_max)+1)+1 with float padding_factor
        # (C++ ROUND: half away from zero, double product int*float)
        pf32 = np.float32(self.padding_factor)
        P = 2 * (int(_c_round_d(float(np.float32(self.r_max) * pf32))) + 1) + 1
        self.pad_size = P
        self.data = np.zeros((P, P, P // 2 + 1), dtype=np.complex128)
        self.weight = np.zeros((P, P, P // 2 + 1), dtype=np.float64)

    def init_zeros(self, current_size: int = -1):
        self.initialise_data_and_weight(current_size)

    # ------------------------------------------------------------------ #
    def backproject2Dto3D(self, f2d: np.ndarray, A: np.ndarray,
                          Mweight: np.ndarray | None = None,
                          r_ewald_sphere: float = -1.0,
                          is_positive_curvature: bool = True,
                          magMatrix: np.ndarray | None = None):
        """Exact port of BackProjector::backproject2Dto3D (ref_dim=3).

        f2d: (s, s//2+1) complex FFTW-layout 2D array.
        A: 3x3 rotation (incl. anisotropy/scale transforms) as passed by
           storeWeightedSums (already multiplied by the tomo rotation).
        """
        if magMatrix is not None:
            m00, m01 = float(magMatrix[0, 0]), float(magMatrix[0, 1])
            m10, m11 = float(magMatrix[1, 0]), float(magMatrix[1, 1])
        else:
            m00, m01, m10, m11 = 1.0, 0.0, 0.0, 1.0

        pf = float(self.padding_factor)
        Ainv = _inv3(np.asarray(A, dtype=np.float64)) * pf

        # const int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor)
        rr = _c_round_d(float(np.float32(self.r_max) * np.float32(pf)))
        max_r2 = rr * rr
        min_r2_nn = _c_round_d(float(np.float32(self.r_min_nn) * np.float32(pf)))
        min_r2_nn = min_r2_nn * min_r2_nn

        Am_Xx = Ainv[0, 0] * m00 + Ainv[0, 1] * m10
        Am_Xy = Ainv[0, 0] * m01 + Ainv[0, 1] * m11
        Am_Yx = Ainv[1, 0] * m00 + Ainv[1, 1] * m10
        Am_Yy = Ainv[1, 0] * m01 + Ainv[1, 1] * m11
        AtA_xx = Am_Xx * Am_Xx + Am_Yx * Am_Yx + (Ainv[2, 0] * m00 + Ainv[2, 1] * m10) ** 2
        Am_Zx = Ainv[2, 0] * m00 + Ainv[2, 1] * m10
        Am_Zy = Ainv[2, 0] * m01 + Ainv[2, 1] * m11
        AtA_xy = Am_Xx * Am_Xy + Am_Yx * Am_Yy + Am_Zx * Am_Zy
        AtA_yy = Am_Xy * Am_Xy + Am_Yy * Am_Yy + Am_Zy * Am_Zy
        AtA_xy2 = AtA_xy * AtA_xy

        inv_diam_ewald = (1.0 / (2.0 * r_ewald_sphere)) if r_ewald_sphere > 0.0 else 0.0
        if not is_positive_curvature:
            inv_diam_ewald *= -1.0

        s = f2d.shape[0]
        sh = f2d.shape[1]
        data = self.data
        weight = self.weight
        P = data.shape[0]
        half = P // 2

        for i in range(s):
            y = i if i < sh else i - s
            first_allowed_x = 0 if i < sh else 1

            discr = AtA_xy2 * y * y - AtA_xx * (AtA_yy * y * y - max_r2)
            if discr < 0.0:
                continue
            d = np.sqrt(discr) / AtA_xx
            q = -AtA_xy * y / AtA_xx
            first_x = int(np.ceil(q - d))
            last_x = int(np.floor(q + d))
            if first_x < first_allowed_x:
                first_x = first_allowed_x
            if last_x > sh - 1:
                last_x = sh - 1
            if first_x > last_x:
                continue

            xv = np.arange(first_x, last_x + 1, dtype=np.float64)
            xv2 = xv.astype(np.int64)
            vals = f2d[i, xv2]
            wts = Mweight[i, xv2] if Mweight is not None else None
            if wts is None:
                wts = np.ones_like(xv)

            xu = m00 * xv + m01 * float(y)
            yu = m10 * xv + m11 * float(y)
            zew = inv_diam_ewald * (xu * xu + yu * yu)

            xp = Ainv[0, 0] * xu + Ainv[0, 1] * yu + Ainv[0, 2] * zew
            yp = Ainv[1, 0] * xu + Ainv[1, 1] * yu + Ainv[1, 2] * zew
            zp = Ainv[2, 0] * xu + Ainv[2, 1] * yu + Ainv[2, 2] * zew
            r2 = xp * xp + yp * yp + zp * zp

            keep = r2 <= max_r2
            if not keep.any():
                continue
            xp, yp, zp = xp[keep], yp[keep], zp[keep]
            r2 = r2[keep]
            vals = vals[keep]
            wts = wts[keep]
            okw = wts > 0.0
            if not okw.any():
                continue
            xp, yp, zp = xp[okw], yp[okw], zp[okw]
            r2 = r2[okw]
            vals, wts = vals[okw], wts[okw]

            is_neg_x = xp < 0.0
            if is_neg_x.any():
                xs = np.where(is_neg_x, -xp, xp)
                ys = np.where(is_neg_x, -yp, yp)
                zs = np.where(is_neg_x, -zp, zp)
                vv = np.where(is_neg_x, np.conj(vals), vals)
            else:
                xs, ys, zs, vv = xp, yp, zp, vals

            do_nn = (self.interpolator == NEAREST_NEIGHBOUR) & (r2 >= min_r2_nn)
            do_tl = ~do_nn

            if do_tl.any():
                self._scatter_trilinear(xs[do_tl], ys[do_tl], zs[do_tl],
                                        vv[do_tl], wts[do_tl], half, P)
            if do_nn.any():
                self._scatter_nn(xs[do_nn], ys[do_nn], zs[do_nn],
                                 vv[do_nn], wts[do_nn], half, P)

    # ------------------------------------------------------------------ #
    def _scatter_trilinear(self, xs, ys, zs, vals, wts, half, P):
        x0 = np.floor(xs).astype(np.int64)
        fx = xs - x0
        y0 = np.floor(ys).astype(np.int64)
        fy = ys - y0
        z0 = np.floor(zs).astype(np.int64)
        fz = zs - z0
        ix = x0                    # x half axis starts at 0
        iy = y0 + half
        iz = z0 + half
        valid = ((ix >= 0) & (ix + 1 < P // 2 + 1) &
                 (iy >= 0) & (iy + 1 < P) &
                 (iz >= 0) & (iz + 1 < P))
        if not valid.any():
            return
        ix, iy, iz = ix[valid], iy[valid], iz[valid]
        fx, fy, fz = fx[valid], fy[valid], fz[valid]
        vals, wts = vals[valid], wts[valid]
        mfx, mfy, mfz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        # Corner layout must match the GT add order per point:
        # (z0,y0,x0), (z0,y0,x1), (z0,y1,x0), (z0,y1,x1),
        # (z1,y0,x0), (z1,y0,x1), (z1,y1,x0), (z1,y1,x1)
        dz = np.stack((iz, iz, iz, iz, iz + 1, iz + 1, iz + 1, iz + 1),
                      axis=1)
        dy = np.stack((iy, iy, iy + 1, iy + 1, iy, iy, iy + 1, iy + 1),
                      axis=1)
        dx = np.stack((ix, ix + 1, ix, ix + 1, ix, ix + 1, ix, ix + 1),
                      axis=1)
        dd = np.stack((mfz * mfy * mfx, mfz * mfy * fx,
                       mfz * fy * mfx, mfz * fy * fx,
                       fz * mfy * mfx, fz * mfy * fx,
                       fz * fy * mfx, fz * fy * fx), axis=1)
        data, weight = self.data, self.weight
        # np.add.at applies repeated indices in array order; a C-order
        # ravel of the (n_points, 8) stacks reproduces the GT
        # per-point accumulation order exactly (bit-exact rounding).
        idx = (dz.ravel(), dy.ravel(), dx.ravel())
        np.add.at(data, idx, (dd * vals[:, None]).ravel())
        np.add.at(weight, idx, (dd * wts[:, None]).ravel())

    def _scatter_nn(self, xs, ys, zs, vals, wts, half, P):
        # C ROUND(xp): half away from zero (inputs are >= 0 here)
        x0 = np.floor(xs + 0.5).astype(np.int64)
        y0 = np.floor(ys + 0.5).astype(np.int64)
        z0 = np.floor(zs + 0.5).astype(np.int64)
        neg = x0 < 0
        if neg.any():
            x0 = np.where(neg, -x0, x0)
            y0 = np.where(neg, -y0, y0)
            z0 = np.where(neg, -z0, z0)
        xx = x0
        yy = y0 + half
        zz = z0 + half
        vv = np.where(neg, np.conj(vals), vals)
        valid = ((xx >= 0) & (xx < P // 2 + 1) &
                 (yy >= 0) & (yy < P) & (zz >= 0) & (zz < P))
        if not valid.any():
            return
        np.add.at(self.data, (zz[valid], yy[valid], xx[valid]), vv[valid])
        np.add.at(self.weight, (zz[valid], yy[valid], xx[valid]), wts[valid])

    # Fourier conjugacy required for a real-valued reconstruction.
    def enforce_hermitian_symmetry(self) -> None:
        """GT ``enforceHermitianSymmetry`` (backprojector.cpp:2189).

        x=0 plane only; pairs (iz,iy) <-> (-iz,-iy) with the GT visit
        rule (iz<0: iy>=0; iz>=0: iy>=1) so (0,0) is left untouched and
        every unordered pair is visited exactly once (no read/write
        conflicts -> vectorisable)."""
        P = self.data.shape[0]
        m = P // 2
        plane = self.data[:, :, 0]
        wplane = self.weight[:, :, 0]
        z = np.arange(-m, m + 1, dtype=np.int64)
        y = np.arange(-m, m + 1, dtype=np.int64)
        Z, Y = np.meshgrid(z, y, indexing="ij")
        visited = ((Z < 0) & (Y >= 0)) | ((Z >= 0) & (Y >= 1))
        kv, iv = np.where(visited)
        mir_k = P - 1 - kv
        mir_i = P - 1 - iv
        fsum = plane[kv, iv] + np.conj(plane[mir_k, mir_i])
        plane[kv, iv] = fsum
        plane[mir_k, mir_i] = np.conj(fsum)
        wsum = wplane[kv, iv] + wplane[mir_k, mir_i]
        wplane[kv, iv] = wsum
        wplane[mir_k, mir_i] = wsum



# =========================================================================== #
# Reconstruction-side port (BackProjector::reconstruct & friends)
# =========================================================================== #

import math as _math
from numba import njit


def _c_round_d_arr(v: np.ndarray) -> np.ndarray:
    """Vectorized C ROUND macro (half away from zero) for signed f64."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v >= 0.0, np.floor(v + 0.5),
                    np.ceil(v - 0.5)).astype(np.int64)


def _c_round_d(v: float) -> int:
    """C ROUND macro (macros.h:197) for doubles: half away from zero.
    (v >= 0 in every current call site; negatives handled for safety.)"""
    if v >= 0.0:
        return int(np.floor(v + 0.5))
    return int(np.ceil(v - 0.5))


# Numerical-Recipes constants (numerical_recipes.cpp)
_EPS_BESS = 1.0e-16
_FPMIN = 1.0e-30
_MAXIT = 10000
_XMIN = 2.0


def _nr_bessi0(x):
    """bessi0 (src/numerical_recipes.cpp), exact polynomial port."""
    ax = abs(x)
    if ax < 3.75:
        y = x / 3.75
        y *= y
        ans = (1.0 + y * (3.5156229 + y * (3.0899424 + y * (
            1.2067492 + y * (0.2659732 + y * (
                0.360768e-1 + y * 0.45813e-2))))))
    else:
        y = 3.75 / ax
        ans = (_math.exp(ax) / _math.sqrt(ax)) * (
            0.39894228 + y * (0.1328592e-1 + y * (
                0.225319e-2 + y * (-0.157565e-2 + y * (
                    0.916281e-2 + y * (-0.2057706e-1 + y * (
                        0.2635537e-1 + y * (-0.1647633e-1
                                            + y * 0.392377e-2))))))))
    return ans


def _bessi1_5(x):
    """bessi1_5 (src/numerical_recipes.cpp): not I_{1/2}."""
    if x == 0.0:
        return 0.0
    return _math.sqrt(2.0 / (_math.pi * x)) * (_math.cosh(x) - _math.sinh(x) / x)


def _chebev(c1, m, xx):
    """chebev(-1, 1, c, m, x) (src/numerical_recipes.cpp)."""
    d = 0.0
    dd = 0.0
    y = 2.0 * (xx) / 2.0  # (2x - a - b)/(b - a) with a=-1, b=1
    y2 = 2.0 * y
    for j in range(m - 1, 0, -1):
        sv = d
        d = y2 * d - dd + c1[j]
        dd = sv
    return y * d - dd + 0.5 * c1[0]


_C1_BESCHB = (-1.142022680371172e0, 6.516511267076e-3, 3.08709017308e-4,
              -3.470626964e-6, 6.943764e-9, 3.6780e-11, -1.36e-13)
_C2_BESCHB = (1.843740587300906e0, -0.076852840844786e0,
              1.271927136655e-3, -4.971736704e-6, -3.3126120e-8,
              2.42310e-10, -1.70e-13, -1.0e-15)


def _beschb(x):
    """beschb (src/numerical_recipes.cpp), NUSE1 = NUSE2 = 5."""
    xx = 8.0 * x * x - 1.0
    gam1 = _chebev(_C1_BESCHB, 5, xx)
    gam2 = _chebev(_C2_BESCHB, 5, xx)
    gampl = gam2 - x * gam1
    gammi = gam2 + x * gam1
    return gam1, gam2, gampl, gammi


def _bessjy(x, xnu):
    """bessjy (src/numerical_recipes.cpp), full NR port. Returns (rj, ry)."""
    if x <= 0.0 or xnu < 0.0:
        raise ValueError("bad arguments in bessjy")
    nl = int(xnu + 0.5) if x < _XMIN else max(
        0, int(xnu - x + 1.5))
    xmu = xnu - nl
    xmu2 = xmu * xmu
    xi = 1.0 / x
    xi2 = 2.0 * xi
    w = xi2 / _math.pi
    isign = 1
    h = xnu * xi
    if h < _FPMIN:
        h = _FPMIN
    b = xi2 * xnu
    d = 0.0
    c = h
    i = 1
    for i in range(1, _MAXIT + 1):
        b += xi2
        d = b - d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b - 1.0 / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        del_ = c * d
        h = del_ * h
        if d < 0.0:
            isign = -isign
        if abs(del_ - 1.0) < _EPS_BESS:
            break
    if i > _MAXIT:
        raise RuntimeError("x too large in bessjy; try asymptotic expansion")
    rjl = isign * _FPMIN
    rjpl = h * rjl
    rjl1 = rjl
    rjp1 = rjpl
    fact = xnu * xi
    for l in range(nl, 0, -1):
        rjtemp = fact * rjl + rjpl
        fact -= xi
        rjpl = fact * rjtemp - rjl
        rjl = rjtemp
    if rjl == 0.0:
        rjl = _EPS_BESS
    f = rjpl / rjl
    if x < _XMIN:
        x2 = 0.5 * x
        pimu = _math.pi * xmu
        fact = 1.0 if abs(pimu) < _EPS_BESS else pimu / _math.sin(pimu)
        d = -_math.log(x2)
        e = xmu * d
        fact2 = 1.0 if abs(e) < _EPS_BESS else _math.sinh(e) / e
        gam1, gam2, gampl, gammi = _beschb(xmu)
        ff = 2.0 / _math.pi * fact * (gam1 * _math.cosh(e) + gam2 * fact2 * d)
        e = _math.exp(e)
        p = e / (gampl * _math.pi)
        q = 1.0 / (e * _math.pi * gammi)
        pimu2 = 0.5 * pimu
        fact3 = 1.0 if abs(pimu2) < _EPS_BESS else _math.sin(pimu2) / pimu2
        r = _math.pi * pimu2 * fact3 * fact3
        c = 1.0
        d = -x2 * x2
        sum_ = ff + r * q
        sum1 = p
        for i in range(1, _MAXIT + 1):
            ff = (i * ff + p + q) / (i * i - xmu2)
            c *= (d / i)
            p /= (i - xmu)
            q /= (i + xmu)
            del_ = c * (ff + r * q)
            sum_ += del_
            del1 = c * p - i * del_
            sum1 += del1
            if abs(del_) < (1.0 + abs(sum_)) * _EPS_BESS:
                break
        if i > _MAXIT:
            raise RuntimeError("bessy series failed to converge")
        rymu = -sum_
        ry1 = -sum1 * xi2
        rymup = xmu * xi * rymu - ry1
        rjmu = w / (rymup - f * rymu)
    else:
        a = 0.25 - xmu2
        p = -0.5 * xi
        q = 1.0
        br = 2.0 * x
        bi = 2.0
        fact = a * xi / (p * p + q * q)
        cr = br + q * fact
        ci = bi + p * fact
        den = br * br + bi * bi
        dr = br / den
        di = -bi / den
        dlr = cr * dr - ci * di
        dli = cr * di + ci * dr
        temp = p * dlr - q * dli
        q = p * dli + q * dlr
        p = temp
        for i in range(2, _MAXIT + 1):
            a += 2 * (i - 1)
            bi += 2.0
            dr = a * dr + br
            di = a * di + bi
            if abs(dr) + abs(di) < _FPMIN:
                dr = _FPMIN
            fact = a / (cr * cr + ci * ci)
            cr = br + cr * fact
            ci = bi - ci * fact
            if abs(cr) + abs(ci) < _FPMIN:
                cr = _FPMIN
            den = dr * dr + di * di
            dr /= den
            di /= -den
            dlr = cr * dr - ci * di
            dli = cr * di + ci * dr
            temp = p * dlr - q * dli
            q = p * dli + q * dlr
            p = temp
            if abs(dlr - 1.0) + abs(dli) < _EPS_BESS:
                break
        if i > _MAXIT:
            raise RuntimeError("cf2 failed in bessjy")
        gam = (p - f) / q
        rjmu = _math.sqrt(w / ((p - f) * gam + q))
        rjmu = abs(rjmu) * (-1.0 if rjl < 0.0 else 1.0)
        rymu = rjmu * gam
        rymup = rymu * (p + q / gam)
        ry1 = xmu * xi * rymu - rymup
    fact = rjmu / rjl
    rj = rjl1 * fact
    rjp = rjp1 * fact
    for i in range(1, nl + 1):
        rytemp = (xmu + i) * xi2 * ry1 - rymu
        rymu = ry1
        ry1 = rytemp
    ry = rymu
    return rj, ry


def _bessj1_5(x):
    """bessj1_5 (src/numerical_recipes.cpp)."""
    rj, _ = _bessjy(x, 1.5)
    return rj


def _kaiser_fourier_value(w, a, alpha, m=0):
    """Exact port of kaiser_Fourier_value (funcs.cpp:233)."""
    w = np.asarray(w, dtype=np.float64)
    sigma = np.sqrt(np.abs(alpha * alpha - (2.0 * np.pi * a * w) ** 2))
    i0 = _nr_bessi0(alpha)
    if m == 0:
        jbranch = (2.0 * np.pi * a * w) > alpha
        if not np.any(jbranch):
            val = np.empty_like(w)
            for idx in np.ndindex(w.shape):
                val[idx] = _bessi1_5(float(sigma[idx]))
            return ((2.0 * np.pi) ** 1.5 * a ** 3 * val /
                    (i0 * sigma ** 1.5))
        out = np.empty_like(w)
        for idx in np.ndindex(w.shape):
            if jbranch[idx]:
                b = _bessj1_5(float(sigma[idx]))
            else:
                b = _bessi1_5(float(sigma[idx]))
            out[idx] = b
        return ((2.0 * np.pi) ** 1.5 * a ** 3 * out /
                (i0 * sigma ** 1.5))
    raise ValueError("unsupported kaiser order")


class _TabFtBlob:
    """Port of TabFtBlob (tabfuncs.cpp)."""

    def __init__(self, radius, alpha, order, nr_elem=10000):
        self.sampling = 0.5 / float(nr_elem)
        w = (np.arange(nr_elem) * self.sampling).astype(np.float64)
        self.table = _kaiser_fourier_value(w, radius, alpha, order)

    def __call__(self, val):
        idx = (np.abs(np.asarray(val, dtype=np.float64)) /
               self.sampling).astype(np.int64)
        out = np.zeros_like(val, dtype=np.float64)
        ok = idx < self.table.shape[0]
        out[ok] = self.table[idx[ok]]
        return out


def _fftw_signed_grid(size, threshold):
    """FFTW-macro signed index for a full axis: idx < threshold ? idx : idx-size."""
    k = np.arange(size)
    return np.where(k < threshold, k, k - size).astype(np.int64)


def _bp_backrotate_3d(self, f3d: np.ndarray, A: np.ndarray,
                      Mweight: np.ndarray | None = None) -> None:
    """GT ``BackProjector::backrotate3D`` (backprojector.cpp:254), the
    dispatch target of ``set2DFourierTransform`` for 3D input
    (reference_reconstruct ``--3d_rot``).

    f3d: (B, B, B//2+1) complex FFTW half-spectrum; points beyond
    r_max_src = B//2 must already be zero (GT contract).  A: 3x3
    rotation.  Mweight: (B, B, B//2+1) f64 or None (-> 1.0).

    Accumulates in the GT (k, i, x) order, row by row, reusing the
    vectorized trilinear/NN scatters of backproject2Dto3D.  The
    trilinear/NN split per row (GT branches per point) may reorder
    additions of points that land in the same target cell; same
    convention as the backproject2Dto3D port.
    """
    import math as _math

    pf = float(self.padding_factor)
    Ainv = _inv3(np.asarray(A, dtype=np.float64)) * pf

    B = f3d.shape[0]
    half = B // 2            # == r_max_src (XSIZE(f3d) - 1)
    r_max_src_2 = half * half

    # C++: const int r_max_ref = r_max * padding_factor;  (float
    # product, truncating int cast -- NO rounding, unlike
    # backproject2Dto3D)
    r_max_ref = int(np.float32(float(self.r_max)) * np.float32(pf))
    r_max_ref_2 = r_max_ref * r_max_ref

    # C++: const int r_min_NN_ref_2 = r_min_nn * r_min_nn *
    #      padding_factor * padding_factor;
    r_min_NN_ref_2 = int(np.float32(self.r_min_nn * self.r_min_nn)
                         * np.float32(pf) * np.float32(pf))

    P = self.pad_size
    halfP = P // 2
    for k in range(B):
        z = k if k <= half else k - B
        x_min = 0 if k <= half else 1
        z2 = z * z
        for i in range(B):
            y = i if i <= half else i - B
            y2 = y * y
            yz2 = y2 + z2
            if yz2 > r_max_src_2:
                continue
            # const int x_max = FLOOR(sqrt(r_max_src_2 - yz2));
            x_max = int(_math.floor(_math.sqrt(r_max_src_2 - yz2)))
            if x_max < x_min:
                continue
            xv = np.arange(x_min, x_max + 1, dtype=np.float64)
            xxi = xv.astype(np.int64)
            xp = Ainv[0, 0] * xv + Ainv[0, 1] * y + Ainv[0, 2] * z
            yp = Ainv[1, 0] * xv + Ainv[1, 1] * y + Ainv[1, 2] * z
            zp = Ainv[2, 0] * xv + Ainv[2, 1] * y + Ainv[2, 2] * z
            r2 = xp * xp + yp * yp + zp * zp
            # GT: ``const int r_ref_2 = xp*xp + yp*yp + zp*zp;`` --
            # the double is truncated toward zero by the int conversion,
            # and BOTH the r_max_ref_2 keep-test and the NN threshold
            # compare that int (double-precision build: f64 products,
            # left-to-right sum, then trunc).
            r2i = np.trunc(r2).astype(np.int64)
            keep = r2i <= r_max_ref_2
            if not keep.any():
                continue
            xp, yp, zp, r2i = xp[keep], yp[keep], zp[keep], r2i[keep]
            xxi = xxi[keep]
            vals = f3d[k, i, xxi]
            if Mweight is not None:
                wts = np.asarray(Mweight[k, i, xxi], dtype=np.float64)
                okw = wts > 0.0
                if not okw.any():
                    continue
                xp, yp, zp = xp[okw], yp[okw], zp[okw]
                r2, r2i = r2[okw], r2i[okw]
                xxi, vals = xxi[okw], vals[okw]
                wts = wts[okw]
            else:
                wts = np.ones(xp.shape, dtype=np.float64)

            # Trilinear branch: hermitian sign flip on the PRE-rounded
            # xp (GT ``bool is_neg_x = xp < 0;``).
            is_neg_x = xp < 0.0
            if is_neg_x.any():
                xs = np.where(is_neg_x, -xp, xp)
                ys = np.where(is_neg_x, -yp, yp)
                zs = np.where(is_neg_x, -zp, zp)
                vv = np.where(is_neg_x, np.conj(vals), vals)
            else:
                xs, ys, zs, vv = xp, yp, zp, vals

            do_nn = ((self.interpolator == NEAREST_NEIGHBOUR)
                     & (r2i >= r_min_NN_ref_2))
            if (~do_nn).any():
                self._scatter_trilinear(xs[~do_nn], ys[~do_nn],
                                        zs[~do_nn], vv[~do_nn],
                                        wts[~do_nn], halfP, P)
            if do_nn.any():
                # NN branch: GT branches on the ROUNDED x0 sign
                # (A3D_ELEM logical coords; data yinit=zinit=-P/2,
                # xinit=0).  Defensive bounds guard: GT has none (UB).
                x0 = _c_round_d_arr(xp[do_nn])
                y0 = _c_round_d_arr(yp[do_nn])
                z0 = _c_round_d_arr(zp[do_nn])
                neg = x0 < 0
                nx = np.where(neg, -x0, x0)
                ny = np.where(neg, -y0, y0)
                nz = np.where(neg, -z0, z0)
                sx = nx
                sy = ny + halfP
                sz = nz + halfP
                valid = ((sx >= 0) & (sx <= halfP) &
                         (sy >= 0) & (sy < P) & (sz >= 0) & (sz < P))
                if valid.any():
                    vvn = np.where(neg, np.conj(vals[do_nn]), vals[do_nn])
                    np.add.at(self.data, (sz[valid], sy[valid], sx[valid]),
                              vvn[valid])
                    np.add.at(self.weight, (sz[valid], sy[valid], sx[valid]),
                              wts[do_nn][valid])


# add the reconstruction methods to the class above

def _bp_decenter(self, min_arr, max_r2):
    """Projector::decenter (projector.h): projector-centered -> FFTW layout.

    Both arrays have shape (P, P, P//2+1) = (z, y, x); z/y centered,
    x unsigned.  Uses the '==' <= max_r2' sphere and the FFTW-macro
    signed thresholds (XSIZE for k and i).
    """
    P = self.pad_size
    H = P // 2 + 1
    half = P // 2
    ks = _fftw_signed_grid(P, H)
    jp = np.arange(H)
    r2 = ks[:, None, None] ** 2 + ks[None, :, None] ** 2 +\
        jp[None, None, :] ** 2
    src = min_arr[np.ix_(ks + half, ks + half, jp)]
    # C++ decenter preserves the element type (weight stays RFLOAT,
    # data stays Complex); keep it here for bit-exact division paths.
    out = np.zeros((P, P, H), dtype=min_arr.dtype)
    out[r2 <= max_r2] = src[r2 <= max_r2]
    return out


def _bp_update_ssnr_arrays(self, tau2_fudge, tau2, sigma2, data_vs_prior,
                           fourier_coverage, fsc, avgctf2,
                           update_tau2_with_fsc=False,
                           is_whole_instead_of_half=False,
                           correct_tau2_by_avgctf2=False):
    """BackProjector::updateSSNRarrays (backprojector.cpp:1041).

    All arrays length ori_size//2+1.  Returns the (possibly updated)
    tau2 array; sigma2/data_vs_prior/fourier_coverage updated in place.

    C++-exact: padding_factor is a C++ float, so every expression that
    is float-typed in C++ (r_max*pf, pf*pf*pf) is reproduced with an
    f32 sub-chain promoted to f64 exactly once.
    """
    pf32 = float(np.float32(self.padding_factor))
    pf32_ = np.float32(pf32)
    # GT: ROUND(r_max * padding_factor), the product is float-typed
    rr = int(np.float32(float(np.float32(self.r_max)) * pf32_)
             + np.float32(0.5))
    max_r2 = rr * rr
    if self.ref_dim == 3:
        occ = float(np.float32(np.float32(pf32_ * pf32_) * pf32_))
    else:
        occ = float(pf32_ * pf32_)
    P = self.pad_size
    H = P // 2 + 1
    # C++ FOR_ALL_ELEMENTS_IN_ARRAY3D(weight): projector-centered
    # logical coords k,i in -P/2..P/2-1, j in 0..P/2 (NOT the FFTW
    # signed grid).
    klog = np.arange(P, dtype=np.int64) - P // 2
    jp = np.arange(H)
    r2 = klog[:, None, None] ** 2 + klog[None, :, None] ** 2 +\
        jp[None, None, :] ** 2
    m = r2 < max_r2
    # GT: ROUND(sqrt((RFLOAT)r2) / padding_factor) -- whole chain f32
    q32 = np.sqrt(r2.astype(np.float32)) / pf32_
    pos32 = q32 > 0
    ires = np.where(pos32, q32 + np.float32(0.5),
                    q32 - np.float32(0.5)).astype(np.int64)

    # loop 1: radial average of (inverse) power -> sigma2
    w = self.weight
    nsh = self.ori_size // 2 + 1
    # GT: local sigma2 / data_vs_prior / fourier_coverage via initZeros;
    # the caller references are only assigned at the very end
    sig2 = np.zeros(nsh)
    counter = np.zeros(nsh)
    np.add.at(sig2, ires[m], occ * w[m])
    np.add.at(counter, ires[m], 1.0)
    big = sig2 > 1e-20
    zero = sig2 == 0.0
    small = ~(big | zero)
    if small.any():
        raise RuntimeError(
            "BackProjector::reconstruct: ERROR: unexpectedly small, "
            "yet non-zero sigma2 value, this should not happen...")
    sig2[big] = counter[big] / sig2[big]
    sig2[zero] = 0.0

    tau2 = np.array(tau2, dtype=np.float64, copy=True)
    if tau2.shape[0] != nsh:
        # GT: tau2.reshape(ori_size/2 + 1) -- coreAllocate uses
        # fftw_malloc, so the fresh (empty) array is zero-filled
        tau2 = np.zeros(nsh, dtype=np.float64)
    dvp = np.zeros(nsh)
    fc = np.zeros(nsh)
    counter[:] = 0.0
    if update_tau2_with_fsc:
        fsc_a = np.asarray(fsc, dtype=np.float64)
        if fsc_a.shape[0] != nsh or tau2.shape[0] != nsh:
            # GT: printShape of each array to stderr, then abort
            sys.stderr.write(
                " Size(X): %d j=[0..%d]\n" % (fsc_a.shape[0],
                                               fsc_a.shape[0] - 1))
            sys.stderr.write(
                " Size(X): %d j=[0..%d]\n" % (tau2.shape[0],
                                               tau2.shape[0] - 1))
            sys.stderr.write(
                " Size(X): %d j=[0..%d]\n" % (nsh, nsh - 1))
            from .errors import numerical_error
            numerical_error(
                "greco-source/"
                "greco-source/"
                "numerical-reference/src/backprojector.cpp", 1104,
                "ERROR BackProjector::reconstruct: sigma2, tau2 and fsc "
                "have different sizes")
        myfsc = np.maximum(0.001, fsc_a)
        if is_whole_instead_of_half:
            myfsc = np.sqrt(2.0 * myfsc / (myfsc + 1.0))
        myfsc = np.minimum(0.999, myfsc)
        myssnr = myfsc / (1.0 - myfsc)
        myssnr = myssnr * tau2_fudge
        tau2[:] = myssnr * sig2
        dvp[:] = myssnr

    # loop 2: data_vs_prior + fourier_coverage.  GT branches per voxel:
    # tau2>0: invtau2 = 1/(occ*fudge*tau2) [optional avgctf2 correction];
    # tau2==0: invtau2 = 1/(0.001*invw) (inf for invw==0, IEEE);
    # anything else (NaN or <0) aborts.
    invw = w[m]
    ii = ires[m]
    tt = tau2[ii]
    bad_t = np.isnan(tt) | (tt < 0.0)
    if bad_t.any():
        raise RuntimeError(
            "BackProjector::reconstruct: Negative or zero values "
            "encountered for tau2 spectrum!")
    invtau2 = np.empty_like(invw)
    pos = tt > 0.0
    invtau2[pos] = 1.0 / (occ * tau2_fudge * tt[pos])
    if correct_tau2_by_avgctf2:
        ac = np.asarray(avgctf2, dtype=np.float64)[ii[pos]]
        sel = ac > 0.0
        if sel.any():
            j = np.nonzero(pos)[0][sel]
            invtau2[j] = invtau2[j] * (1.0 / ac[sel])
    zer = tt == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        invtau2[zer] = 1.0 / (0.001 * invw[zer])
    ratio = np.empty_like(invw)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[pos] = invw[pos] / invtau2[pos]
        ratio[zer] = invw[zer] / invtau2[zer]
    if not update_tau2_with_fsc:
        np.add.at(dvp, ii, ratio)
    np.add.at(fc, ii, (ratio >= 1.0).astype(np.float64))
    np.add.at(counter, ii, 1.0)

    # finalise (GT if/else-if/else on i): i>r_max -> 0; counter<0.001
    # -> 999; else /= counter
    if not update_tau2_with_fsc:
        over = np.arange(nsh) > self.r_max
        low = (~over) & (counter < 0.001)
        rest = ~(over | low)
        dvp[over] = 0.0
        dvp[low] = 999.0
        dvp[rest] = dvp[rest] / counter[rest]
    pos = counter > 0.0
    fc[pos] = fc[pos] / counter[pos]
    # C++: tau2_io = tau2; sigma2_out = sigma2; data_vs_prior_out =
    # data_vs_prior; fourier_coverage_out = fourier_coverage.  Every
    # assignment RESIZES the caller array and copies.  Py cannot
    # resize through a reference, so all four results are returned and
    # the callers rebind (when the C++ code passes the same array
    # object several times, the last assignment wins -- callers
    # reproduce that by re-binding the same name).
    return tau2, sig2, dvp, fc

def _bp_convolute_blob_real_space(self, F, do_mask):
    """BackProjector::convoluteBlobRealSpace (backprojector.cpp:2515).

    F: (P, P, P//2+1) complex FFTW-layout array (the transformer's
    Fconv).  Returns the half spectrum left by
    transformer.FourierTransform() = raw r2c / P**3, as in C++.
    """
    P = self.pad_size
    from ..tomo.fft_f64 import rfftn3_raw_f64, irfftn3_raw_f64
    # transformer.setReal(Mconv); inverseFourierTransform(): raw c2r,
    # FFTW_ESTIMATE plans (legacy FourierTransformer, bit-exact engine).
    Mconv = irfftn3_raw_f64(F, (P, P, P))
    # rval = sqrt(int r2) / (ori_size * padding_factor): the denominator
    # is float-typed in C++ (int * float), then promoted to double.
    pf32 = float(np.float32(self.padding_factor))
    denom = float(np.float32(self.ori_size) * np.float32(pf32))
    padhdim = P // 2
    k = np.arange(P)
    kp = np.where(k < padhdim, k, k - P).astype(np.float64)
    r = np.sqrt(kp[:, None, None] ** 2 + kp[None, :, None] ** 2 +
                kp[None, None, :] ** 2)
    rval = r / denom
    normftblob = self._tab_ftblob(0.0)
    factor = self._tab_ftblob(rval) / normftblob
    if do_mask:
        thr = 1.0 / (2.0 * pf32)
        Mconv = np.where(rval > thr, 0.0, Mconv * factor)
    else:
        Mconv = Mconv * factor
    # FourierTransform(): the GT normalisation loop is
    #   DIRECT_MULTIDIM_ELEM(fFourier, n) /= size   (size = P**3,
    #   unsigned long -> tComplex(size, 0)), i.e. the tComplex
    #   operator/= path: cd = size^2; re' = (re*size + im*0.0)/cd;
    #   im' = (im*size - re*0.0)/cd.
    raw = rfftn3_raw_f64(Mconv)
    s = float(P ** 3)
    cd = s * s
    re_ = raw.real
    im_ = raw.imag
    return (re_ * s + im_ * 0.0) / cd + (1j * ((im_ * s - re_ * 0.0) / cd))

@njit(fastmath=False, cache=True)
def _soft_mask_sum_core(vol, radius, radius_p, cw):
    """softMaskOutsideMap (mask.cpp) background sum: sequential k->i->j
    over the centered (N, N, N) f64 volume (C++ loop order, bit-exact).
    """
    N = vol.shape[0]
    half = N // 2
    sum_bg = 0.0
    sum_w = 0.0
    for k in range(N):
        ksg = float(k) - float(half)
        for i in range(N):
            isg = float(i) - float(half)
            for j in range(N):
                jsg = float(j) - float(half)
                r = np.sqrt(ksg * ksg + isg * isg + jsg * jsg)
                if r < radius:
                    continue
                v = vol[k, i, j]
                if r > radius_p:
                    sum_w += 1.0
                    sum_bg += v
                else:
                    rc = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r) / cw)
                    sum_w += rc
                    sum_bg += rc * v
    return sum_w, sum_bg


def _bp_soft_mask_outside_map(vol, radius=-1.0, cosine_width=3.0,
                              mnoise=None):
    """softMaskOutsideMap (mask.cpp:43) on a (N, N, N) xmipp-centered
    f64 map.  The background sum is a sequential C++-order loop; the
    application phase is element-wise (vectorised)."""
    N = vol.shape[0]
    if radius < 0:
        radius = float(N) / 2.0
    radius_p = radius + cosine_width
    k = (np.arange(N) - N // 2).astype(np.float64)
    r = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 +
                k[None, None, :] ** 2)
    if mnoise is None:
        sum_w, sum_bg = _soft_mask_sum_core(vol, radius, radius_p,
                                            cosine_width)
        sum_bg = sum_bg / sum_w
        out = r > radius_p
        vol[out] = sum_bg
        edge = (r >= radius) & (r <= radius_p)
        if edge.any():
            rc = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r[edge]) /
                                    cosine_width)
            vol[edge] = (1.0 - rc) * vol[edge] + rc * sum_bg
    return vol

def _bp_window_to_oridim_real_space(self, Fin):
    """BackProjector::windowToOridimRealSpace (backprojector.cpp:2562).

    Fin: (P, P, P//2+1) complex FFTW array.  Returns the (ori_size,)3
    real-space map (origin-centered cube).
    """
    from ..tomo.fft_f64 import irfftn3_raw_f64
    pf32 = float(np.float32(self.padding_factor))
    f = np.float32(pf32)
    ori = self.ori_size
    P = self.pad_size
    H = P // 2 + 1
    # padoridim = ROUND(padding_factor * ori_size): float product in C++
    # (float * int), promoted to double, then ROUND; made even.
    padoridim = _c_round_d(float(f * np.float32(ori)))
    padoridim += padoridim % 2
    Ph = padoridim // 2 + 1
    Ftmp = np.zeros((padoridim, padoridim, Ph), dtype=np.complex128)
    # windowFourierTransform(Fin, padoridim) (fftw.h:809).  P is always
    # odd and padoridim always even, so exactly one branch runs:
    #  * downscale (P > padoridim, full-resolution stages): iterate the
    #    OUT grid, copy IN at equal signed FFTW coordinates (the outer
    #    shell of Fin is dropped).
    #  * upscale (P < padoridim, low-resolution stages): iterate the IN
    #    grid, zero-pad, and copy only kp^2+ip^2+jp^2 <= (H-1)^2 so the
    #    windowed FT has nothing in the corners (asymmetric-FT guard).
    if padoridim > P:
        kz = np.arange(P, dtype=np.int64)
        kp = np.where(kz < H, kz, kz - P)
        ki = np.arange(P, dtype=np.int64)
        ip = np.where(ki < H, ki, ki - P)
        jx = np.arange(H, dtype=np.int64)
        r2 = kp[:, None, None] ** 2 + ip[None, :, None] ** 2 +\
            jx[None, None, :] ** 2
        m = r2 <= (H - 1) ** 2
        oz = np.where(kp < 0, kp + padoridim, kp)
        oi = np.where(ip < 0, ip + padoridim, ip)
        kzm, kim, jxm = np.nonzero(m)
        Ftmp[oz[kzm], oi[kim], jxm] = Fin[kzm, kim, jxm]
    else:
        ks = _fftw_signed_grid(padoridim, Ph)
        jp = np.arange(Ph)
        z0 = np.where(ks < 0, ks + P, ks)
        zd = np.where(ks < 0, ks + padoridim, ks)
        Ftmp[zd[:, None, None], zd[None, :, None], jp[None, None, :]] =\
            Fin[z0[:, None, None], z0[None, :, None], jp[None, None, :]]
    # CenterFFTbySign: physical indices, (k^i^j)&1
    kk = np.arange(padoridim, dtype=np.int64)
    jj = np.arange(Ph)
    odd = ((kk[:, None, None] ^ kk[None, :, None] ^ jj[None, None, :]) & 1) != 0
    Ftmp[odd] *= -1.0
    # normfft: C++ float chains promoted to RFLOAT:
    #   data_dim==3: (RFLOAT)(pf*pf*pf);  else (RFLOAT)(pf*pf*pf*ori)
    if self.ref_dim == 2:
        if self.data_dim == 2:
            normfft = float(f * f)
        else:
            normfft = float(np.float32(f * f) * np.float32(ori))
    else:
        if self.data_dim == 3:
            normfft = float(np.float32(f * f) * f)
        else:
            normfft = float(np.float32(np.float32(f * f) * f) *
                            np.float32(ori))
    # setReal(Mout); inverseFourierTransform(): raw c2r (no scaling).
    Mout = irfftn3_raw_f64(Ftmp, (padoridim, padoridim, padoridim))
    # window() to the ori_size cube, Mout /= normfft, soft mask.
    c0 = padoridim // 2
    Mout = Mout[c0 - ori // 2: c0 - ori // 2 + ori,
                c0 - ori // 2: c0 - ori // 2 + ori,
                c0 - ori // 2: c0 - ori // 2 + ori]
    Mout = Mout / normfft
    _bp_soft_mask_outside_map(Mout, -1.0, 3.0, None)
    return Mout

def _bp_gridding_correct(self, vol):
    """Projector::griddingCorrect (projector.cpp:595).

    Divides the origin-centered real map by the FT of the interpolator
    (sinc for NN/r_min_nn==0, sinc^2 for trilinear or NN with
    r_min_nn>0).  C++-exact: rval = r / (ori_size * padding_factor)
    with the int*float product float-typed.
    """
    ori = self.ori_size
    pf32 = float(np.float32(self.padding_factor))
    denom = float(np.float32(ori) * np.float32(pf32))
    k = (np.arange(ori) - ori // 2).astype(np.float64)
    r = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 +
                k[None, None, :] ** 2)
    m = r > 0.0
    rval = r[m] / denom
    sinc = np.sin(np.pi * rval) / (np.pi * rval)
    if self.interpolator == NEAREST_NEIGHBOUR and self.r_min_nn == 0:
        vol[m] = vol[m] / sinc
    elif self.interpolator == TRILINEAR or (
            self.interpolator == NEAREST_NEIGHBOUR and self.r_min_nn > 0):
        vol[m] = vol[m] / (sinc * sinc)
    else:
        raise RuntimeError(
            "BUG Projector::griddingCorrect: unrecognised interpolator "
            "scheme.")
    return vol

def _bp_reconstruct(self, vol_out, max_iter_preweight=10, do_map=True,
                    tau2=None, tau2_fudge=1.0, normalise=1.0,
                    minres_map=-1, weight_out=None, diag_dir=None):
    """BackProjector::reconstruct (backprojector.cpp:1382), C++-exact.

    Consumes self.data/self.weight (the wsum grids) and writes the
    (ori_size,)^3 real-space reconstruction into vol_out (in place).

    C++ has TWO branches:
      * skip_gridding or max_iter_preweight <= 0:  Fconv = decenter(data)
        divided by max(Fweight_MAP, 0.001 * radial-mean weight) -- NO
        normalise division, NO gridding iterations.
      * otherwise (gridding): data/weight divided by normalise,
        Pipe & Menon iterative Fnewweight, Fconv = decenter(data) *
        Fnewweight.
    Both finish with windowToOridimRealSpace + griddingCorrect.

    padding_factor is C++ float: every float-typed C++ expression is
    reproduced with an f32 sub-chain promoted to f64 once.  The FFTs
    use Greco's FFTW core with FFTW_ESTIMATE plans and raw normalization.
    """
    import os as _os
    import sys as _sys
    if diag_dir is not None:
        _os.makedirs(diag_dir, exist_ok=True)

    def _dump(name, arr):
        if diag_dir is not None:
            # C++ dumps 8 bytes/element; complex arrays are written
            # interleaved (real, imag, real, imag, ...).
            a = np.ascontiguousarray(arr)
            if np.iscomplexobj(a):
                a = a.ravel().view(np.float64)
            else:
                a = a.ravel().astype(np.float64)
            a.tofile(_os.path.join(diag_dir, name))

    pf32 = float(np.float32(self.padding_factor))
    f = np.float32(pf32)
    P = self.pad_size
    H = P // 2 + 1
    rr = _c_round_d(float(np.float32(self.r_max) * f))
    max_r2 = rr * rr
    if self.ref_dim == 3:
        occ = float(np.float32(np.float32(f * f) * f))
    else:
        occ = float(f * f)

    # decenter(weight) into the FFTW array (C++ keeps it real here)
    Fweight = _bp_decenter(self, self.weight, max_r2)

    ks = _fftw_signed_grid(P, H)
    jp = np.arange(H)
    r2 = ks[:, None, None] ** 2 + ks[None, :, None] ** 2 +\
        jp[None, None, :] ** 2

    if do_map:
        # C++: int r2 = kp*kp + ip*ip + jp*jp (integer); then
        # int ires = ROUND(sqrt((RFLOAT)r2) / padding_factor);
        # ROUND(x) = (int)(x + 0.5) for x > 0 else (int)(x - 0.5), i.e.
        # truncation toward zero; x here is always >= 0.
        m = r2 < max_r2
        sq = np.sqrt(r2[m].astype(np.float64)) / pf32
        ires = np.trunc(np.where(sq > 0.0, sq + 0.5, sq - 0.5))
        ii = ires.astype(np.int64)
        invw = Fweight[m].copy()
        tt = np.asarray(tau2, dtype=np.float64)[ii]
        # C++: else-if (tau2 < 1e-20) covers negatives too; only NaN
        # reaches the REPORT_ERROR branch (backprojector.cpp:1494).
        # GT first prints " tau2= " + the vector (best-effort here; that
        # print is UB territory for NaN values even in GT).
        if np.isnan(tt).any():
            import sys as _sys
            _sys.stderr.write(" tau2= \n")
            vals = [repr(float(np.float32(v))) for v in
                    np.asarray(tau2, dtype=np.float64)]
            _sys.stderr.write(" ".join(vals) + "\n")
            from .errors import numerical_error
            numerical_error(
                "greco-source/"
                "numerical-reference/src/backprojector.cpp", 1494,
                "ERROR BackProjector::reconstruct: Negative or zero "
                "values encountered for tau2 spectrum!")
        pos = tt > 0.0
        invtau2 = np.zeros_like(invw)
        invtau2[pos] = 1.0 / (occ * tau2_fudge * tt[pos])
        fb = (~pos) & (invw > 1e-20)
        invtau2[fb] = 1.0 / (0.001 * invw[fb])
        # Only for (ires >= minres_map) add the Wiener-filter term.
        upd = ii >= minres_map
        invw[upd] = invw[upd] + invtau2[upd]
        Fweight[m] = invw

    if self.skip_gridding or max_iter_preweight <= 0:
        # ---- C++ skip branch (backprojector.cpp:1510) ----
        Fconv = _bp_decenter(self, self.data, max_r2)
        # round_max_r2 = ROUND(r_max * pf * r_max * pf), float chain
        t = np.float32(self.r_max) * f
        t = t * np.float32(self.r_max)
        t = t * f
        round_max_r2 = int(_c_round_d(float(t)))
        radavg_weight = np.zeros(self.r_max, dtype=np.float64)
        counter = np.zeros(self.r_max, dtype=np.float64)
        m1 = r2 < round_max_r2
        ires1 = np.floor(np.sqrt(r2[m1].astype(np.float64)) / pf32)\
            .astype(np.int64)
        if ires1.size and ires1.max() >= self.r_max:
            from .errors import numerical_error
            numerical_error("greco-source/"
                "numerical-reference/src/backprojector.cpp", 1533,
                         "BUG: ires >=XSIZE(radavg_weight) ")
        np.add.at(radavg_weight, ires1, Fweight[m1])
        np.add.at(counter, ires1, 1.0)
        ok = (counter > 0.0) | (radavg_weight > 0.0)
        if not ok.all():
            from .errors import numerical_error
            numerical_error("greco-source/"
                "numerical-reference/src/backprojector.cpp", 1551,
                         "BUG: zeros in counter or radavg_weight!")
        radavg_weight[ok] = radavg_weight[ok] / (1000.0 * counter[ok])
        ires2 = np.floor(np.sqrt(r2.astype(np.float64)) / pf32)\
            .astype(np.int64)
        ires2 = np.minimum(ires2, self.r_max - 1)
        w = np.maximum(Fweight, radavg_weight[ires2])
        nz = w != 0.0
        if not nz.all():
            # C++ prints once for the first zero-weight voxel with
            # non-zero Fconv (FFTW loop order == flat C order).
            cflat = Fconv.ravel()
            mag2 = cflat.real * cflat.real + cflat.imag * cflat.imag
            first = np.flatnonzero((w.ravel() == 0.0) & (mag2 > 0.0))
            if first.size:
                n0 = int(first[0])
                k0, i0, j0 = np.unravel_index(n0, (P, P, H))
                ires0 = int(ires2.ravel()[n0])
                c0 = cflat[n0]
                f0 = float(Fweight.ravel()[n0])
                r0 = float(radavg_weight[ires0])
                print(" WARNING: ignoring divide by zero in "
                      "skip_gridding: ires = " + str(ires0) +
                      " kp = " + str(int(ks[k0])) +
                      " ip = " + str(int(ks[i0])) +
                      " jp = " + str(int(j0)) + "\n"
                      " Fconv= [" + format(c0.real, ".6g") + ", " +
                      format(c0.imag, ".6g") + "] Fweight= " +
                      format(f0, ".6g") + " radavg_weight=" +
                      format(r0, ".6g") + "\n"
                      " max_r2 = " + str(max_r2) +
                      " r_max = " + str(self.r_max) +
                      " padding_factor = " + format(pf32, ".6g") +
                      " ROUND(sqrt(max_r2)) = " +
                      str(int(_c_round_d(np.sqrt(float(max_r2))))) +
                      " ROUND(r_max * padding_factor) = " + str(rr),
                      file=_sys.stderr)
        if nz.any():
            # C++ tComplex operator/=(tComplex(w, 0)): cd = w*w;
            # re' = (re*w + im*0.0)/cd; im' = (im*w - re*0.0)/cd.
            re_ = Fconv[nz].real
            im_ = Fconv[nz].imag
            wn = w[nz]
            cd = wn * wn
            Fconv[nz] = (re_ * wn + im_ * 0.0) / cd +\
                (1j * ((im_ * wn - re_ * 0.0) / cd))
        _dump("recon_Fconv_skip.bin", Fconv)
        _dump("recon_Fweight_skip.bin", Fweight)
    else:
        # ---- C++ gridding branch (backprojector.cpp:1579) ----
        Fweight = Fweight / normalise
        # C++ complex /= normalise is tComplex operator/=(tComplex(n,0)):
        # re' = (re*n + im*0.0)/n^2 ; im' = (im*n - re*0.0)/n^2
        re_ = self.data.real
        im_ = self.data.imag
        cd = normalise * normalise
        self.data = (re_ * normalise + im_ * 0.0) / cd +\
            (1j * ((im_ * normalise - re_ * 0.0) / cd))
        _dump("recon_Fweight.bin", Fweight)
        # 1/0 sphere in the projector-centered layout, then decenter.
        # C++ OVERWRITES the member weight array with the mask.
        klog = np.arange(P, dtype=np.int64) - P // 2
        r2_w = klog[:, None, None] ** 2 + klog[None, :, None] ** 2 +\
            jp[None, None, :].astype(np.int64) ** 2
        wmask = (r2_w < max_r2).astype(np.float64)
        self.weight[:] = wmask
        Fnewweight = _bp_decenter(self, wmask, max_r2)
        _dump("init_Fnewweight.bin", Fnewweight)
        _dump("init_Fweight.bin", Fweight)
        Fweight_c = Fweight.astype(np.complex128)
        for _iter in range(max_iter_preweight):
            # Fconv[n] = Fnewweight[n] * Fweight[n]: double * double ->
            # Complex(prod, +0.0)  (imag must stay exactly +0.0).
            Fconv = np.zeros_like(Fweight_c)
            Fconv.real = Fnewweight * Fweight
            _dump("iter%02d_Finput.bin" % _iter, Fconv)
            Fconv = _bp_convolute_blob_real_space(self, Fconv, False)
            m = r2 < max_r2
            # C++ abs() of tComplex = sqrt(re*re + im*im) (NOT hypot).
            w = np.maximum(1e-6,
                           np.sqrt(Fconv[m].real * Fconv[m].real +
                                   Fconv[m].imag * Fconv[m].imag))
            Fnewweight[m] = Fnewweight[m] / w
            _dump("iter%02d_Fnewweight.bin" % _iter, Fnewweight)
            _dump("iter%02d_Fconv.bin" % _iter, Fconv)
        _dump("recon_Fnewweight.bin", Fnewweight)
        Fconv = _bp_decenter(self, self.data, max_r2)
        Fconv = Fconv * Fnewweight
        _dump("recon_Fconv.bin", Fconv)

    Mout = _bp_window_to_oridim_real_space(self, Fconv)
    _dump("recon_Mout_pre_gridding.bin", Mout)
    _bp_gridding_correct(self, Mout)
    vol_out[:] = Mout
    return vol_out


# --------------------------------------------------------------------------- #
# low-resolution averaging (split random halves)
# --------------------------------------------------------------------------- #
def _bp_get_low_res_data_and_weight(self, lowres_r_max: int):
    """BackProjector::getLowResDataAndWeight (backprojector.cpp:856).

    Returns (lowres_data, lowres_weight), both centered (xinit=0) with
    shape (S, S, S//2+1), S = 2*(ROUND(pf*lowres_r_max)+1)+1.  Only the
    sphere k^2+i^2+j^2 <= (ROUND(pf*lowres_r_max))^2 is copied (plain
    value copies, so bit-exact).
    """
    if lowres_r_max > self.r_max:
        raise ValueError(
            "getLowResDataAndWeight: lowres_r_max is bigger than r_max")
    lr = _c_round_d(float(np.float32(self.padding_factor) * np.float32(lowres_r_max)))
    lr2 = lr * lr
    S = 2 * (lr + 1) + 1
    H = S // 2 + 1
    ld = np.zeros((S, S, H), dtype=np.complex128)
    lw = np.zeros((S, S, H), dtype=np.float64)
    k = np.arange(-(S // 2), S // 2 + 1, dtype=np.int64)
    i = np.arange(-(S // 2), S // 2 + 1, dtype=np.int64)
    j = np.arange(H, dtype=np.int64)
    r2 = (k[:, None, None] ** 2 + i[None, :, None] ** 2
          + j[None, None, :] ** 2)
    kk, ii, jj = np.nonzero(r2 <= lr2)
    off = self.pad_size // 2 - S // 2
    ld[kk, ii, jj] = self.data[kk + off, ii + off, jj]
    lw[kk, ii, jj] = self.weight[kk + off, ii + off, jj]
    return ld, lw


def _bp_set_low_res_data_and_weight(self, lowres_data, lowres_weight,
                                    lowres_r_max: int) -> None:
    """BackProjector::setLowResDataAndWeight (backprojector.cpp:897).

    Overwrites the sphere of self.data/self.weight with the (centered)
    lowres arrays.
    """
    if lowres_r_max > self.r_max:
        raise ValueError(
            "setLowResDataAndWeight: lowres_r_max is bigger than r_max")
    lr = _c_round_d(float(np.float32(self.padding_factor) * np.float32(lowres_r_max)))
    lr2 = lr * lr
    S = lowres_data.shape[0]
    H = lowres_data.shape[2]
    if S != 2 * (lr + 1) + 1 or H != S // 2 + 1:
        raise ValueError(
            "setLowResDataAndWeight: lowres_data is not of expected size")
    k = np.arange(-(S // 2), S // 2 + 1, dtype=np.int64)
    i = np.arange(-(S // 2), S // 2 + 1, dtype=np.int64)
    j = np.arange(H, dtype=np.int64)
    r2 = (k[:, None, None] ** 2 + i[None, :, None] ** 2
          + j[None, None, :] ** 2)
    kk, ii, jj = np.nonzero(r2 <= lr2)
    off = self.pad_size // 2 - S // 2
    self.data[kk + off, ii + off, jj] = lowres_data[kk, ii, jj]
    self.weight[kk + off, ii + off, jj] = lowres_weight[kk, ii, jj]


@njit(fastmath=False)
def _c_round_f64_njit(x):
    # C std::round: half away from zero (numba-compatible)
    if x >= 0:
        return int(np.floor(x + 0.5))
    return -int(np.floor(-x + 0.5))


@njit(fastmath=False, cache=True)
def _downsampled_average_core(data, weight, P, Hd, pf, S, half, Hout,
                              divide):
    """getDownsampledAverage accumulation, sequential k->i->j (C++ order).

    data/weight are centered in k/i (signed index = axis - P//2) and
    half-complex in j (0..P//2).  The per-bin addition order of the C++
    FOR_ALL_ELEMENTS_IN_ARRAY3D loop (k slowest, j fastest) is
    reproduced exactly by this nesting.
    """
    avg = np.zeros((S, S, Hout), dtype=np.complex128)
    wt = np.zeros((S, S, Hout), dtype=np.float64)
    for k in range(P):
        ksg = float(k) - float(P // 2)
        kp = _c_round_f64_njit(ksg / pf) + half
        for i in range(P):
            isg = float(i) - float(P // 2)
            ip = _c_round_f64_njit(isg / pf) + half
            drow = data[k, i, :]
            wrow = weight[k, i, :]
            for j in range(Hd):
                jp = _c_round_f64_njit(float(j) / pf)
                avg[kp, ip, jp] += drow[j]
                if divide:
                    wt[kp, ip, jp] += wrow[j]
                else:
                    wt[kp, ip, jp] += 1.0
    n = S * S * Hout
    aflat = avg.reshape(n)
    wflat = wt.reshape(n)
    # C++ avg[n] /= down_weight[n] is the tComplex operator/=(tComplex(c,0))
    # path: cd = c*c; re' = (re*c + im*0.0)/cd; im' = (im*c - re*0.0)/cd.
    for n0 in range(n):
        c = wflat[n0]
        if c > 0.0:
            cd = c * c
            re_ = aflat[n0].real
            im_ = aflat[n0].imag
            aflat[n0] = (re_ * c + im_ * 0.0) / cd +\
                1j * ((im_ * c - re_ * 0.0) / cd)
        else:
            aflat[n0] = 0.0
    return avg


def _bp_get_downsampled_average(self, divide: bool = True):
    """BackProjector::getDownsampledAverage (backprojector.cpp:933).

    Returns the centered average array (2*(r_max+1)+1)^3 with xinit=0,
    accumulated in the exact C++ element order (bit-exact).
    """
    r_max = self.r_max
    S = 2 * (r_max + 1) + 1
    P = self.pad_size
    # data is half-complex in j (0..P//2), as in C++ (pad_size/2+1);
    # the output avg is centered k/i and half-complex j (0..S//2)
    return _downsampled_average_core(
        np.ascontiguousarray(self.data), np.ascontiguousarray(self.weight),
        P, P // 2 + 1, float(np.float32(self.padding_factor)), S, S // 2,
        S // 2 + 1, bool(divide))


BackProjector.decenter = _bp_decenter


BackProjector.decenter = _bp_decenter
BackProjector.update_ssnr_arrays = _bp_update_ssnr_arrays
BackProjector.backrotate_3d = _bp_backrotate_3d
BackProjector.convolute_blob_real_space = _bp_convolute_blob_real_space
BackProjector.soft_mask_outside_map = _bp_soft_mask_outside_map
BackProjector.window_to_oridim_real_space = _bp_window_to_oridim_real_space
BackProjector.gridding_correct = _bp_gridding_correct
BackProjector.reconstruct = _bp_reconstruct
BackProjector.get_low_res_data_and_weight = _bp_get_low_res_data_and_weight
BackProjector.set_low_res_data_and_weight = _bp_set_low_res_data_and_weight
BackProjector.get_downsampled_average = _bp_get_downsampled_average


def _bp_calculate_downsampled_fourier_shell_correlation(self,
                                                        avg1,
                                                        avg2) -> np.ndarray:
    """BackProjector::calculateDownSampledFourierShellCorrelation
    (backprojector.cpp:995), f64-exact.

    avg1/avg2: centered (k/i) half-complex (j) arrays from
    get_downsampled_average.  GT arithmetic (RFLOAT=double build):
    R = f64 sqrt(int r2); keep R <= r_max; idx = ROUND(R) (half away
    from zero); num += re1*re2 + im1*im2; den1/den2 += re^2+im^2 (NO
    sqrt); fsc = num/sqrt(den1*den2) with a >0 guard; fsc[0] = 1.
    """
    H = self.ori_size // 2 + 1
    S = int(avg1.shape[0])
    half = S // 2
    ks = np.arange(S, dtype=np.int64) - half
    js = np.arange(S // 2 + 1, dtype=np.int64)
    r2i = (ks[:, None, None] ** 2 + ks[None, :, None] ** 2
           + js[None, None, :] ** 2)
    R = np.sqrt(r2i.astype(np.float64))
    m = R <= self.r_max
    idx = np.floor(R[m] + 0.5).astype(np.int64)
    a1 = np.ascontiguousarray(avg1[m], dtype=np.complex128)
    a2 = np.ascontiguousarray(avg2[m], dtype=np.complex128)
    num = np.zeros(H, dtype=np.float64)
    den1 = np.zeros(H, dtype=np.float64)
    den2 = np.zeros(H, dtype=np.float64)
    np.add.at(num, idx,
              a1.real * a2.real + a1.imag * a2.imag)
    np.add.at(den1, idx, a1.real * a1.real + a1.imag * a1.imag)
    np.add.at(den2, idx, a2.real * a2.real + a2.imag * a2.imag)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = den1 * den2 > 0.0
        fsc = np.zeros(H, dtype=np.float64)
        fsc[ok] = num[ok] / np.sqrt(den1[ok] * den2[ok])
    fsc[0] = 1.0
    return fsc


BackProjector.calculate_downsampled_fourier_shell_correlation = (
    _bp_calculate_downsampled_fourier_shell_correlation)


def _lazy_tab_ftblob(self):
    # blob_radius=1.9 default (ctor): tab_ftblob.initialise(radius*2, alpha, order)
    return _TabFtBlob(getattr(self, "_blob_radius", 1.9) * 2.0,
                      getattr(self, "_blob_alpha", 15.0),
                      getattr(self, "_blob_order", 0), 10000)


# attach a lazily computed tab_ftblob per instance
_orig_bp_init = BackProjector.__init__


def _bp_init(self, *args, **kwargs):
    _orig_bp_init(self, *args, **kwargs)
    self._blob_radius = self.blob_radius
    self._blob_alpha = self.blob_alpha
    self._blob_order = self.blob_order
    self._tab_ftblob = _lazy_tab_ftblob(self)


BackProjector.__init__ = _bp_init
