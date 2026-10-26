# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/jaz/math/Tait_Bryan_angles.h
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
# Changes: Python translation of Tait-Bryan rotation routines.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""Tait-Bryan rotation matrices -- port of reference's Tait_Bryan_angles.h.

reference convention:  R = Rz(chi) * Ry(theta) * Rx(phi),  with
    Rx = [[1,0,0],[0, cp,-sp],[0, sp, cp]]
    Ry = [[ct,0,st],[0,1,0],[-st,0,ct]]
    Rz = [[cc,-sc,0],[sc,cc,0],[0,0,1]]
Angles are in radians.
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations


import numpy as np

from ..libm_compat import sincos

# reference src/macros.h: PI = 3.14159265358979323846
PI = 3.14159265358979323846

__all__ = ['angles_to_matrix3', 'PI']


def mm3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 matrix product with the exact C++ d2Matrix operator* order:
    result(i, j) += a(i, k) * b(k, j) accumulated for k = 0..2."""
    C = np.empty((3, 3), dtype=np.result_type(a, b))
    C[0, 0] = (a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0]) + a[0, 2] * b[2, 0]
    C[0, 1] = (a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]) + a[0, 2] * b[2, 1]
    C[0, 2] = (a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2]) + a[0, 2] * b[2, 2]
    C[1, 0] = (a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0]) + a[1, 2] * b[2, 0]
    C[1, 1] = (a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]) + a[1, 2] * b[2, 1]
    C[1, 2] = (a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2]) + a[1, 2] * b[2, 2]
    C[2, 0] = (a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0]) + a[2, 2] * b[2, 0]
    C[2, 1] = (a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1]) + a[2, 2] * b[2, 1]
    C[2, 2] = (a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2]) + a[2, 2] * b[2, 2]
    return C


def angles_to_matrix3(phi: float, theta: float, chi: float) -> np.ndarray:
    # C++ Tait_Bryan_angles.h: sin/cos pairs fused by GCC into
    # glibc sincos (GT binary: TaitBryan::anglesToMatrix3 SINCOS=3).
    sp, cp = sincos(phi)
    st, ct = sincos(theta)
    sc, cc = sincos(chi)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cp, -sp],
                   [0.0, sp, cp]])
    Ry = np.array([[ct, 0.0, st],
                   [0.0, 1.0, 0.0],
                   [-st, 0.0, ct]])
    Rz = np.array([[cc, -sc, 0.0],
                   [sc, cc, 0.0],
                   [0.0, 0.0, 1.0]])
    # C++ TaitBryan::anglesToMatrix3: Rz * Ry * Rx (d3Matrix chain,
    # left-associative, strict per-element accumulation).
    return mm3(mm3(Rz, Ry), Rx)


# ---------------------------------------------------------------------------
# GT src/euler.cpp routines (Euler_angles2matrix family) used by
# reference_tiltpair_plot.  The (rot, tilt, psi) overload above IS GT
# Euler_angles2matrix(rot, tilt, psi, M, false); the app simply passes
# (psi, tilt, rot) as its (alpha, beta, gamma) arguments, so it is
# called here with the same (alpha, beta, gamma) meaning.
# ---------------------------------------------------------------------------

_F32_EPS = float(np.finfo(np.float32).eps)  # C FLT_EPSILON as a double
