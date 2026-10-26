# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/projector.cpp
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
# Greco port and adaptations:
# Changes: Python translation and Greco integration of Fourier projection routines.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""reference Projector port (core/projector.{h,cpp}).

Exact port of the CPU path of ``Projector::computeFourierTransformMap`` and
``Projector::project`` as driven by ``MlModel::setFourierTransformMaps``
(single-body, ``Projector(ori_size, TRILINEAR, pad, 10, 2)``).

Coordinate conventions (validated against the C++ dumper, see
tmp_diag/proj/dump_projector.cpp):

* Real-space volumes are origin-centered arrays of shape (N, N, N) with
  memory order (z slowest, y mid, x fastest) -- a plain C-order numpy array
  indexed [z, y, x].
* The padded Fourier transform is ``rfftn(Mpad)/Np**3`` on an (Np, Np, Np)
  box -- buffer-identical to reference's FFTW r2c output (halved x-axis fastest).
* ``CenterFFTbySign`` multiplies ``Faux[k, i, j]`` by ``(-1)**(k+i+j)``.
* The projector ``data`` array has shape (Z, Y, X) = (pad_size, pad_size,
  pad_size//2+1): DC sits at (x=0, y=0, z=0) where x is the unsigned half
  axis (origin at the left edge) and y/z are centered (origin -pad//2).
  A value is stored at data[z, y, x] iff the Faux element at corner indices
  (kz, ky, kx) -- with signed kp = kz or kz-Np, ip = ky or ky-Np, jp = kx --
  satisfies kp^2+ip^2+jp^2 <= max_r2.
* Projections sample ``data`` at (xp, yp, zp) = (Ainv @ (x, y, 0)) * padding
  where (x, y) are 2D-FT pixel coordinates: x in [0, N/2] (half axis),
  y in [-N/2+1, N/2] (centered).  Samples with xp < 0 are read as the
  complex conjugate at the mirrored position (8-point trilinear).

Performance notes (bit-exact rewrites, same per-element floating point
op order as the original port):

* ``project`` caches the per-size output grid and performs the 8-corner
  trilinear gather as ONE flat fancy index into ``data.ravel()``.
* ``compute_fourier_transform_map`` uses int32 index math, a flat scatter,
  ``np.bincount`` for the radial power spectrum (same accumulation order
  as ``np.add.at`` for these 1-D bins), and -- when pyfftw is available --
  a cached FFTW plan (same transform as FFTW r2c, validated vs pocketfft
  to ~1e-16 relative).
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from numba import njit

from .fftw import gt_forward_normalize

NEAREST_NEIGHBOUR = 0
TRILINEAR = 1
CONVOLVE_BLOB = 2


def _round_c(x):
    """C ROUND() macro: half away from zero (x array-like, result int64)."""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


def _inv3(A: np.ndarray) -> np.ndarray:
    """Bit-exact replica of Matrix2D<RFLOAT>::inv() for 3x3 (explicit
    cofactors, division by det = A00*R00 + A10*R01 + A20*R02)."""
    a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
    a10, a11, a12 = A[1, 0], A[1, 1], A[1, 2]
    a20, a21, a22 = A[2, 0], A[2, 1], A[2, 2]
    r00 = a22 * a11 - a21 * a12
    r01 = -(a22 * a01 - a21 * a02)
    r02 = a12 * a01 - a11 * a02
    r10 = -(a22 * a10 - a20 * a12)
    r11 = a22 * a00 - a20 * a02
    r12 = -(a12 * a00 - a10 * a02)
    r20 = a21 * a10 - a20 * a11
    r21 = -(a21 * a00 - a20 * a01)
    r22 = a11 * a00 - a10 * a01
    det = a00 * r00 + a10 * r01 + a20 * r02
    R = np.array([[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]], dtype=np.float64)
    R /= det
    return R


WIDTH_FMASK_EDGE = 2.0


def initial_lowpass_filter_reference(vol: np.ndarray, ori_size: int,
                                     pixel_size: float,
                                     ini_high: float) -> np.ndarray:
    """MlOptimiser::initialLowPassFilterReferences (applied at iter==0 only).

    Soft raised-cosine low-pass on the reference: keeps r < radius,
    kills r > radius_p, cosine edge in between (WIDTH_FMASK_EDGE px).

    Transforms use the Greco FFTW core in f64: raw r2c followed by
    division by N^3, and raw c2r on the inverse. Double precision
    preserves the numerical reference's low-amplitude spectral tail.
    """
    if ini_high <= 0.0:
        return vol
    from ..tomo.fft_f64 import (rfftn3_fwdonly_f64,
                                irfftn3_fwdonly_f64)
    N = int(ori_size)
    radius = float(N) * float(pixel_size) / float(ini_high)
    radius -= WIDTH_FMASK_EDGE / 2.0
    radius_p = radius + WIDTH_FMASK_EDGE
    F = rfftn3_fwdonly_f64(np.ascontiguousarray(vol, dtype=np.float64))
    kk = np.arange(N, dtype=np.int64)
    kp = np.where(kk < N // 2 + 1, kk, kk - N).astype(np.float64)
    # GT (FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM): z and y are fftw-centred
    # (kp/ip = k-N for k >= N/2) but x is NOT (jp = j, 0..N/2); the
    # negative x side therefore sits at r >= N/2 > radius_p and is
    # hard-zeroed (no cosine ramp there).
    r = np.sqrt(kp[:, None, None] ** 2 + kp[None, :, None] ** 2
                + (kk[None, None, : N // 2 + 1].astype(np.float64)) ** 2)
    mask = np.ones_like(r)
    mask[r > radius_p] = 0.0
    edge = (r >= radius) & (r <= radius_p)
    mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge])
                                    / WIDTH_FMASK_EDGE)
    F = F * mask
    # GT assigns F = 0 (not *0) above radius_p; keep +0.0 explicitly.
    F = np.where(r > radius_p, 0.0 + 0.0j, F)
    return irfftn3_fwdonly_f64(F, s=(N, N, N))


class Projector:
    """reference Projector (3D reference -> 2D projections in Fourier space)."""

    def __init__(self, ori_size: int, interpolator: int = TRILINEAR,
                 padding_factor: float = 2.0, r_min_nn: int = 10, data_dim: int = 2):
        if padding_factor < 1.0:
            raise ValueError("padding factor cannot be less than 1")
        self.ori_size = int(ori_size)
        self.padding_factor = float(padding_factor)
        self.interpolator = int(interpolator)
        self.r_min_nn = int(r_min_nn)
        self.data_dim = int(data_dim)
        self.ref_dim = 0
        self.r_max = 0
        self.pad_size = 0
        self.normfft = 0.0
        self.max_r2 = 0
        self.min_r2 = -1
        self.data: Optional[np.ndarray] = None
        self.power_spectrum: Optional[np.ndarray] = None
        # --- fast-path caches (filled by compute_fourier_transform_map)
        self._data_flat: Optional[np.ndarray] = None
        self._off8: Optional[np.ndarray] = None
        self._grid_cache: Dict[int, Tuple[np.ndarray, ...]] = {}
        # --- FFT plan cache (pyfftw), one entry per padded size
        self._fft_state: Optional[Dict] = None
        # --- invariant caches (keyed by geometry), built once per job
        self._ires_tables: Dict[Tuple, np.ndarray] = {}
        self._gridding_div: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        ncpu = os.cpu_count() or 1
        try:
            self._fftw_threads = min(int(os.environ.get("GRECO_FFTW_THREADS", "8")),
                                     ncpu)
        except ValueError:
            self._fftw_threads = 8
        # GRECO_FFTW_THREADS=0 forces the pocketfft (np.fft) path

    # ------------------------------------------------------------------ #
    # computeFourierTransformMap (CPU path)
    # ------------------------------------------------------------------ #
    def _rfftn_cached(self, Mpad: np.ndarray) -> np.ndarray:
        """rfftn with a cached pyfftw plan when available.

        Returns a view into the plan's output buffer (valid until the next
        call); the caller must finish using it before re-invoking."""
        Np = Mpad.shape[0]
        st = self._fft_state
        if st is None or st.get("Np") != Np:
            st = self._fft_state = {"Np": Np, "ok": False}
            if self._fftw_threads > 0:
                try:
                    import pyfftw
                    a = pyfftw.empty_aligned((Np, Np, Np), dtype=np.float64)
                    st["a"] = a
                    st["plan"] = pyfftw.builders.rfftn(
                        a, axes=(0, 1, 2), threads=self._fftw_threads)
                    st["ok"] = True
                except Exception:
                    st.update(Np=Np, ok=False)
        if st["ok"]:
            st["a"][:] = Mpad
            # view into the plan's internal output buffer (overwritten by
            # the next call; the caller must be done before re-invoking)
            return st["plan"](st["a"])
        return np.fft.rfftn(Mpad)

    def compute_fourier_transform_map(self, vol: np.ndarray, current_size: int = -1,
                                      do_gridding: bool = True,
                                      min_ires: int = -1,
                                      fourier_mask: np.ndarray | None = None) -> "Projector":
        N = self.ori_size
        vol = np.ascontiguousarray(vol, dtype=np.float64)
        if vol.shape != (N, N, N):
            raise ValueError(f"vol shape {vol.shape} != ({N},)*3")
        self.ref_dim = 3

        # padoridim = ROUND(padding_factor * ori_size) (float math, as in C++),
        # made even; padding factor then recomputed in float.
        padoridim = int(_round_c(np.float64(np.float32(self.padding_factor) * N))[()])
        if padoridim % 2:
            padoridim += 1
        Np = padoridim
        self.padding_factor = float(np.float32(Np) / np.float32(N))
        pf32 = np.float32(self.padding_factor)
        pfd = np.float64(self.padding_factor)

        if self.data_dim == 3:
            normfft = float(pf32 * pf32 * pf32)
        else:
            normfft = float(pf32 * pf32 * pf32 * N)
        self.normfft = normfft

        # gridding pre-correction (in place on the centered real-space map)
        if do_gridding:
            self._gridding_correct(vol)

        # zero-pad to the Np^3 box (centered on the centered volume)
        Mpad = np.zeros((Np, Np, Np), dtype=np.float64)
        off = Np // 2 - N // 2
        Mpad[off:off + N, off:off + N, off:off + N] = vol

        # oversampled Fourier transform, reference-normalized (divide by Np^3,
        # in place to avoid a second full-size temporary)
        Faux = self._rfftn_cached(Mpad)
        gt_forward_normalize(Faux, Np ** 3)
        del Mpad

        # CenterFFTbySign: multiply by (-1)^(k+i+j)
        signs_x = np.ones(Np // 2 + 1); signs_x[1::2] = -1.0
        signs_y = np.ones(Np); signs_y[1::2] = -1.0
        signs_z = np.ones(Np); signs_z[1::2] = -1.0
        Faux *= signs_x[None, None, :]
        Faux *= signs_y[None, :, None]
        Faux *= signs_z[:, None, None]

        # data array: (Z, Y, X) = (pad_size, pad_size, pad_size//2+1)
        if current_size < 0:
            r_max = N // 2
        else:
            r_max = min(current_size // 2, N // 2)
        self.r_max = r_max
        pad_size = 2 * (int(_round_c(pf32 * r_max)[()]) + 1) + 1
        self.pad_size = pad_size
        data = np.zeros((pad_size, pad_size, pad_size // 2 + 1), dtype=np.complex128)

        max_r2 = int(_round_c(pf32 * r_max)[()]) ** 2
        self.max_r2 = max_r2
        self.min_r2 = (int(_round_c(pf32 * min_ires)[()]) ** 2 if min_ires > 0 else -1)

        ps = np.zeros(N // 2 + 1, dtype=np.float64)
        counter = np.zeros(N // 2 + 1, dtype=np.float64)

        do_mask = fourier_mask is not None
        self._last_fourier_mask = fourier_mask
        if do_mask:
            fm = np.ascontiguousarray(fourier_mask, dtype=np.float64)
            if fm.shape != (N, N, N):
                raise ValueError(f"fourier_mask shape {fm.shape} != ({N},)*3")

        zmid = pad_size // 2
        ky = np.arange(Np, dtype=np.int32)
        ip = np.where(ky <= Np // 2, ky, ky - Np)              # signed int32
        kx = np.arange(Np // 2 + 1, dtype=np.int32)            # unsigned half
        ip2 = ip * ip
        kx2 = kx * kx
        PY = pad_size
        PX = pad_size // 2 + 1
        zspan = PY * PX
        data_flat = data.ravel()
        minlength = N // 2 + 1

        if do_mask:
            self._subsampling_masked(Faux, data_flat, ps, counter)
        else:
            self._subsampling_fast(Faux, data_flat, ps, counter)
        self.power_spectrum = np.where(counter < 1.0, 0.0,
                                       ps / np.maximum(counter, 1.0))
        self.data = data
        self._data_flat = data_flat
        self._off8 = np.array([0, 1, PX, PX + 1, zspan, zspan + 1,
                               zspan + PX, zspan + PX + 1], dtype=np.int64)
        return self

    def _gridding_correct(self, vol: np.ndarray):
        N = self.ori_size
        pf = self.padding_factor
        key = (N, pf, self.interpolator, self.r_min_nn)
        ent = self._gridding_div.get(key)
        if ent is None:
            # centered squared coords; r2 = k^2 + i^2 + j^2 in the same
            # order as the C++ loop (k slowest), broadcast to (N, N, N)
            idx2 = (np.arange(N, dtype=np.float64) - N // 2) ** 2
            r = np.sqrt(idx2[:, None, None] + idx2[None, :, None]
                        + idx2[None, None, :])
            sel = r > 0.0
            rval = r[sel] / (N * pf)
            div_full = np.empty_like(r)
            div_full[sel] = 1.0
            if self.interpolator == NEAREST_NEIGHBOUR and self.r_min_nn == 0:
                sinc = np.sin(np.pi * rval) / (np.pi * rval)
                div_full[sel] = sinc
            elif self.interpolator == TRILINEAR or (
                    self.interpolator == NEAREST_NEIGHBOUR
                    and self.r_min_nn > 0):
                sinc = np.sin(np.pi * rval) / (np.pi * rval)
                div_full[sel] = sinc * sinc
            else:
                raise RuntimeError(
                    "unrecognised interpolator scheme in griddingCorrect")
            ent = self._gridding_div[key] = (sel, div_full[sel].copy())
        sel, div = ent
        vol[sel] /= div

    # ------------------------------------------------------------------ #
    # project (get2DFourierTransform for ref_dim=3, data_dim=2)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # subsampling: scatter the padded FT sphere into data + power spectrum
    # ------------------------------------------------------------------ #
    def _get_ires_table(self, Np: int, pfd: float) -> Optional[np.ndarray]:
        """round(sqrt(kp^2+ip^2+kx^2)/pfd) over the full padded grid,
        centered (signed) coords exactly as in the per-element C++
        formula, uint8 clipped at N/2 (points beyond the kept region are
        never queried), cached.  Built in z-chunks to keep temporaries
        small."""
        key = (Np, pfd)
        tab = self._ires_tables.get(key)
        if tab is None:
            if Np < 1024:
                return None
            clip = self.ori_size // 2
            tab = np.empty((Np, Np, Np // 2 + 1), dtype=np.uint16)
            kp = np.arange(Np, dtype=np.int32)
            kp = np.where(kp <= Np // 2, kp, kp - Np)
            ip = np.arange(Np, dtype=np.int32)
            ip = np.where(ip <= Np // 2, ip, ip - Np)
            kx = np.arange(Np // 2 + 1, dtype=np.int32)
            kp2 = kp * kp
            ip2 = ip * ip
            kx2 = kx * kx
            for z0 in range(0, Np, 128):
                z1 = min(z0 + 128, Np)
                r2 = (kp2[z0:z1, None, None] + ip2[None, :, None]
                      + kx2[None, None, :])
                tab[z0:z1] = np.minimum(
                    _round_c(np.sqrt(r2.astype(np.float64)) / pfd),
                    clip).astype(np.uint16)
            self._ires_tables[key] = tab
        return tab

    def _subsampling_fast(self, Faux: np.ndarray, data_flat: np.ndarray,
                          ps: np.ndarray, counter: np.ndarray):
        """Row-prefix subsampling (no fourier mask).  The kept j per
        (kp, ip) row is the contiguous prefix 0..jmax and kept rows are a
        prefix + suffix of the C-order i axis (ip runs 0,+1,...,-1), so a
        C-order repeat over the per-row kept counts reproduces the legacy
        boolean-mask traversal element-for-element -> bit-exact data /
        power_spectrum / counter.  Note the Faux strides (Np, Np//2+1)
        and the data strides (pad_size, pad_size//2+1) differ whenever
        pad_size != Np."""
        Np = Faux.shape[0]
        zmid = self.pad_size // 2
        ip = np.arange(Np, dtype=np.int32)
        ip = np.where(ip <= Np // 2, ip, ip - Np)
        ip2 = ip * ip
        px = Np // 2 + 1              # Faux x extent
        zbase = Np * px               # Faux slab stride
        PX = self.pad_size // 2 + 1   # data x extent
        zspan = self.pad_size * PX    # data z span
        max_r2 = self.max_r2
        min_r2 = self.min_r2
        normfft = self.normfft
        pfd = self.padding_factor
        minlength = self.ori_size // 2 + 1
        ires_tab = self._get_ires_table(Np, pfd)
        Ffull = Faux
        for kz0 in range(0, Np, 128):
            kz1 = min(kz0 + 128, Np)
            nz = kz1 - kz0
            kp = np.arange(kz0, kz1, dtype=np.int32)
            kp = np.where(kp <= Np // 2, kp, kp - Np)
            kp2 = kp * kp
            b2 = max_r2 - (kp2[:, None] + ip2[None, :])
            ok_row = b2 >= 0
            if not ok_row.any():
                continue
            jmax = np.floor(np.sqrt(
                np.where(ok_row, b2, 0).astype(np.float64))).astype(
                    np.int32)
            if jmax.max() > Np // 2:
                jmax = np.minimum(jmax, Np // 2)
            # kept j-count per (kp, ip) row; rows with b2 < 0 keep none.
            # kept rows are a prefix + suffix of the C-order i axis (ip
            # runs 0,+1,...,-1), so a C-order repeat reproduces the
            # legacy boolean-mask traversal element-for-element.
            n_j_all = np.where(ok_row, jmax + 1, 0)
            Fslab = Ffull[kz0:kz1].reshape(nz * Np * px)
            if min_r2 >= 0:
                bm_all = min_r2 - (kp2[:, None] + ip2[None, :])
                okm_row = bm_all >= 0
                jmaxm = np.floor(np.sqrt(
                    np.where(okm_row, bm_all, 0).astype(
                        np.float64))).astype(np.int32)
                n_jm_all = np.where(okm_row, jmaxm + 1, 0)
            for z in range(nz):
                n_j = n_j_all[z]
                rows = np.flatnonzero(n_j)
                if rows.size == 0:
                    continue
                M = int(np.sum(n_j[rows], dtype=np.int64))
                offsets = np.r_[0, np.cumsum(n_j)[:-1]]
                row_idx = np.repeat(rows, n_j[rows])
                jflat = np.arange(M) - offsets[row_idx]
                i_abs = row_idx
                src = z * zbase + i_abs * px + jflat
                Fsub = Fslab[src] * normfft
                lin = ((int(kp[z]) + zmid) * zspan
                       + (ip[i_abs] + zmid) * PX + jflat)
                data_flat[lin] = Fsub
                if ires_tab is not None:
                    ires_z = ires_tab[kz0 + z, i_abs, jflat]
                else:
                    r2_j = kp2[z] + ip2[i_abs] + jflat * jflat
                    ires_z = _round_c(np.sqrt(r2_j.astype(np.float64))
                                      / pfd)
                magsq = (np.abs(Fsub) ** 2) / 2.0
                np.add.at(ps, ires_z.astype(np.intp, copy=False), magsq)
                counter += np.bincount(ires_z.astype(np.intp, copy=False),
                                       minlength=minlength)
                if min_r2 >= 0:
                    n_jm = n_jm_all[z]
                    rowsm = np.flatnonzero(n_jm)
                    if rowsm.size:
                        Mm = int(np.sum(n_jm[rowsm], dtype=np.int64))
                        offsetsm = np.r_[0, np.cumsum(n_jm)[:-1]]
                        row_idxm = np.repeat(rowsm, n_jm[rowsm])
                        jflatm = np.arange(Mm) - offsetsm[row_idxm]
                        i_absm = row_idxm
                        linm = ((int(kp[z]) + zmid) * zspan
                                + (ip[i_absm] + zmid) * PX + jflatm)
                        data_flat[linm] = 0.0

    def _subsampling_masked(self, Faux: np.ndarray, data_flat: np.ndarray,
                            ps: np.ndarray, counter: np.ndarray):
        """Legacy boolean-mask subsampling (fourier_mask path only)."""
        Np = Faux.shape[0]
        N = Faux.shape[0]
        zmid = self.pad_size // 2
        pfd = self.padding_factor
        ky = np.arange(Np, dtype=np.int32)
        ip = np.where(ky <= Np // 2, ky, ky - Np)
        kx = np.arange(Np // 2 + 1, dtype=np.int32)
        ip2 = ip * ip
        kx2 = kx * kx
        PY = self.pad_size
        PX = self.pad_size // 2 + 1
        zspan = PY * PX
        minlength = N // 2 + 1
        max_r2 = self.max_r2
        fm = np.ascontiguousarray(self._last_fourier_mask,
                                  dtype=np.float64)
        Nf = fm.shape[0]
        for kz0 in range(0, Np, 128):
            kz = np.arange(kz0, min(kz0 + 128, Np), dtype=np.int32)
            kp = np.where(kz <= Np // 2, kz, kz - Np)
            r2 = kp[:, None, None] ** 2 + ip2[None, :, None]\
                + kx2[None, None, :]
            keep = r2 <= max_r2
            if not keep.any():
                continue
            Fsub = Faux[kz0:kz0 + kz.size][keep] * self.normfft
            lkp = _round_c(kp.astype(np.float64) / pfd)
            lip = _round_c(ip.astype(np.float64) / pfd)
            ljp = _round_c(kx.astype(np.float64) / pfd)
            lkp = np.where(lkp < 0, lkp + Nf, lkp)
            lip = np.where(lip < 0, lip + Nf, lip)
            ljp = np.where(ljp < 0, ljp + Nf, ljp)
            w = fm[lkp[:, None, None], lip[None, :, None],
                   ljp[None, None, :]][keep]
            Fsub = Fsub * w
            dz = kp[:, None, None] + zmid
            dy = ip[None, :, None] + zmid
            dx = kx[None, None, :]
            lin = (dz * zspan + dy * PX + dx)[keep].astype(np.int64)
            data_flat[lin] = Fsub
            ires = _round_c(np.sqrt(r2[keep].astype(np.float64)) / pfd)
            magsq = (np.abs(Fsub) ** 2) / 2.0
            np.add.at(ps, ires, magsq)
            np.add.at(counter, ires, w)
            if self.min_r2 >= 0:
                hp = keep & (r2 <= self.min_r2)
                if hp.any():
                    data_flat[(dz * zspan + dy * PX + dx)[hp]
                              .astype(np.int64)] = 0.0

    def _output_grid(self, N: int) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
        g = self._grid_cache.get(N)
        if g is None:
            i = np.arange(N, dtype=np.int64)
            y = np.where(i <= N // 2, i, i - N)
            x = np.arange(N // 2 + 1, dtype=np.int64)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            disc = xx * xx + yy * yy <= (N // 2) ** 2
            iy_idx, ix_idx = np.nonzero(disc)
            g = (iy_idx, ix_idx, xx[disc].astype(np.float64),
                 yy[disc].astype(np.float64))
            self._grid_cache[N] = g
        return g

    def project(self, f2d: np.ndarray, A: np.ndarray,
                Ainv: np.ndarray | None = None):
        """Fill ``f2d`` (shape (S, S//2+1), y slowest / x fastest half) for the
        rotation matrix A, exactly as Projector::project.

        ``S`` may be smaller than ``ori_size``: the C++ code derives the
        output radius from ``XSIZE(f2d) - 1`` (the reference is then only
        projected within that shell).

        ``Ainv`` optionally supplies a precomputed 3x3 inverse. Without it,
        the internal 3x3 cofactor inverse is used."""
        if self.data is None:
            raise RuntimeError("call compute_fourier_transform_map first")
        N = f2d.shape[0]
        data_flat = self._data_flat
        off8 = self._off8
        pad = self.pad_size
        pf32 = np.float32(self.padding_factor)
        pf = np.float64(self.padding_factor)

        A = np.asarray(A, dtype=np.float64)
        if A.shape != (3, 3):
            raise ValueError("A must be 3x3")
        if Ainv is None:
            Ainv = _inv3(A)
        else:
            Ainv = np.ascontiguousarray(Ainv, dtype=np.float64)
        Ainv = Ainv * pf

        iy_idx, ix_idx, xv, yv = self._output_grid(N)

        r_max_ref = int(np.float32(self.r_max) * pf32)  # C++: int * float, truncated
        xp = Ainv[0, 0] * xv + Ainv[0, 1] * yv
        yp = Ainv[1, 0] * xv + Ainv[1, 1] * yv
        zp = Ainv[2, 0] * xv + Ainv[2, 1] * yv
        r_ref_2 = xp * xp + yp * yp + zp * zp
        keep = r_ref_2 <= r_max_ref * r_max_ref
        if not keep.any():
            return
        iy_idx, ix_idx = iy_idx[keep], ix_idx[keep]
        xp, yp, zp = xp[keep], yp[keep], zp[keep]
        r_ref_2 = r_ref_2[keep]

        if self.interpolator == NEAREST_NEIGHBOUR:
            # GT (projector.cpp:683/749): with --NN, trilinear is used
            # only below r_min_nn in reference space, nearest-neighbour
            # above (the C++ comment "never actually used" is stale:
            # with the default --r_min_nn 10 the NN branch covers all
            # higher frequencies).
            r_min_NN_ref_2 = int(
                (self.r_min_nn * self.r_min_nn) * pf * pf)
            tri_sel = r_ref_2 < r_min_NN_ref_2
        else:
            tri_sel = np.ones(r_ref_2.shape, dtype=bool)
        nn_sel = ~tri_sel

        self._project_trilinear(
            f2d, iy_idx[tri_sel], ix_idx[tri_sel],
            xp[tri_sel], yp[tri_sel], zp[tri_sel])
        if nn_sel.any():
            self._project_nearest(
                f2d, iy_idx[nn_sel], ix_idx[nn_sel],
                xp[nn_sel], yp[nn_sel], zp[nn_sel])
        return

    def _project_trilinear(self, f2d, iy_idx, ix_idx, xp, yp, zp):
        """GT Projector::project TRILINEAR branch (projector.cpp:683-748),
        vectorised.  Fills f2d (shape (S, S//2+1)) in place."""
        if iy_idx.size == 0:
            return
        N = f2d.shape[0]
        data_flat = self._data_flat
        off8 = self._off8
        pad = self.pad_size

        is_neg_x = xp < 0.0
        if is_neg_x.any():
            xs = np.where(is_neg_x, -xp, xp)
            ys = np.where(is_neg_x, -yp, yp)
            zs = np.where(is_neg_x, -zp, zp)
        else:
            xs, ys, zs = xp, yp, zp

        x0 = np.floor(xs).astype(np.int64)
        fx = xs - x0
        y0 = np.floor(ys).astype(np.int64)
        fy = ys - y0
        z0 = np.floor(zs).astype(np.int64)
        fz = zs - z0

        ix0 = x0                       # X: xmin = 0
        iy0 = y0 + pad // 2            # Y centered
        iz0 = z0 + pad // 2            # Z centered

        data_shape = self.data.shape
        valid = ((ix0 >= 0) & (ix0 + 1 < data_shape[2]) &
                 (iy0 >= 0) & (iy0 + 1 < data_shape[1]) &
                 (iz0 >= 0) & (iz0 + 1 < data_shape[0]))
        if not valid.any():
            return

        ix0, iy0, iz0 = ix0[valid], iy0[valid], iz0[valid]
        fx, fy, fz = fx[valid], fy[valid], fz[valid]

        # 8 corner values in ONE flat gather (C-order ravel of data)
        PX = data_shape[2]
        zspan = data_shape[1] * PX
        ib = iz0 * zspan + iy0 * PX + ix0
        vals = data_flat[ib[None, :] + off8[:, None]]
        v0, v1, v2, v3, v4, v5, v6, v7 = vals
        dx00 = v0 + (v1 - v0) * fx
        dx01 = v4 + (v5 - v4) * fx
        dx10 = v2 + (v3 - v2) * fx
        dx11 = v6 + (v7 - v6) * fx
        dxy0 = dx00 + (dx10 - dx00) * fy
        dxy1 = dx01 + (dx11 - dx01) * fy
        val = dxy0 + (dxy1 - dxy0) * fz

        NX = N // 2 + 1
        oidx = (iy_idx[valid] * NX + ix_idx[valid]).astype(np.int64)
        f2d_flat = f2d.ravel()
        f2d_flat[oidx] = val
        negsel = is_neg_x[valid]
        if negsel.any():
            f2d_flat[oidx[negsel]] = np.conj(val[negsel])

    def _project_nearest(self, f2d, iy_idx, ix_idx, xp, yp, zp):
        """GT Projector::project NEAREST_NEIGHBOUR branch
        (projector.cpp:749-787), vectorised.  Rounds to the nearest
        reference voxel (C ROUND macro, half away from zero) and applies
        the hermitian mirror on the ROUNDED x0.

        IMPORTANT GT quirk reproduced bit-exactly: the two arms use
        DIFFERENT access macros with already-shifted indices
        (xr = x0 - xinit, yr = y0 + pad//2, zr = z0 + pad//2):

        * ``is_neg_x``:  ``DIRECT_A3D_ELEM(data, zr, yr, xr)`` -- the
          shifted indices are used as PHYSICAL coordinates: flat
          ``zr*zyx + yr*PX + xr`` (conjugated when written).
        * ``!is_neg_x``: ``A3D_ELEM(data, zr, yr, xr)`` -- a LOGICAL
          access that subtracts the starts a SECOND time, i.e. the
          physical voxel is ``(zr + pad//2, yr + pad//2, xr)``.  The
          C++ flat index (no bounds checking) wraps around inside the
          allocation, so this reads an unrelated location of the fine
          grid; when the flat index passes the end of the array the GT
          binary performs an out-of-bounds heap read (observed to
          return 0; we emit 0 in that case).
        """
        if iy_idx.size == 0:
            return
        N = f2d.shape[0]
        data_flat = self._data_flat
        half = self.pad_size // 2

        x0 = _round_c(xp)
        y0 = _round_c(yp)
        z0 = _round_c(zp)
        is_neg_x = x0 < 0
        x0 = np.where(is_neg_x, -x0, x0)
        y0 = np.where(is_neg_x, -y0, y0)
        z0 = np.where(is_neg_x, -z0, z0)

        ds = self.data.shape
        # GT bounds are checked on the single-offset indices
        # (projector.cpp:774-779): 0 <= xr < xdim, 0 <= yr < ydim,
        # 0 <= zr < zdim (xinit = 0, yinit = zinit = -pad//2).
        ix = x0
        iy = y0 + half
        iz = z0 + half
        valid = ((ix >= 0) & (ix < ds[2]) &
                 (iy >= 0) & (iy < ds[1]) &
                 (iz >= 0) & (iz < ds[0]))
        if not valid.any():
            return

        PX = ds[2]
        zspan = ds[1] * PX
        total = data_flat.size

        NX = N // 2 + 1
        f2d_flat = f2d.ravel()
        oidx_all = (iy_idx * NX + ix_idx).astype(np.int64)

        neg = valid & is_neg_x
        if neg.any():
            lin = (iz[neg] * zspan + iy[neg] * PX
                   + ix[neg]).astype(np.int64)
            f2d_flat[oidx_all[neg]] = np.conj(data_flat[lin])

        pos = valid & ~is_neg_x
        if pos.any():
            # second (logical) offset: physical (iz + half, iy + half, ix),
            # flattened without bounds checks (C++ semantics)
            lin = ((iz[pos] + half) * zspan + (iy[pos] + half) * PX
                   + ix[pos]).astype(np.int64)
            inb = (lin >= 0) & (lin < total)
            if inb.any():
                f2d_flat[oidx_all[pos][inb]] = data_flat[lin[inb]]
            # lin >= total: GT out-of-bounds heap read (returns 0 in the
            # GT binary) -> f2d stays zero
        return

    def rotate_3d(self, A: np.ndarray, f3d: np.ndarray,
                  Ainv: np.ndarray | None = None) -> None:
        """GT ``Projector::rotate3D`` (src/projector.cpp:1124), in place
        on ``f3d`` -- either the caller's zeroed (S, S, S//2+1) complex
        array (3-D output) or a zeroed (S, S//2+1) half-complex array
        (2-D output: the z=0 slice, as in the autopick 3-D reference
        branch, where GT's Fref has ZSIZE == 1).  ``A`` is the full
        orientation matrix (the GT inverts it internally and folds in
        the padding factor).

        ``Ainv`` optionally supplies a precomputed 3x3 inverse of ``A``;
        otherwise the internal 3x3 cofactor inverse is used."""
        S = f3d.shape[0]
        if f3d.ndim == 2:
            # GT rotate3D on a 2-D f3d (autopick 3-D reference branch):
            # ZSIZE(f3d) == 1, the slice is filled at (k=0, i, x) with
            # x on the unsigned half axis.
            if f3d.shape != (S, S // 2 + 1):
                raise ValueError(f"f3d shape {f3d.shape} != ({S}, {S} // 2 + 1)"
                                 )
        elif f3d.shape != (S, S, S // 2 + 1):
            raise ValueError(f"f3d shape {f3d.shape} != ({S},)*2+({S}//2+1,)"
                             )
        if self.r_max <= 0:
            raise RuntimeError("rotate_3d before compute_fourier_transform_map")
        if Ainv is None:
            Ainv = _inv3(np.ascontiguousarray(A, dtype=np.float64))
        else:
            Ainv = np.ascontiguousarray(Ainv, dtype=np.float64)
        Ainv = Ainv * float(self.padding_factor)
        # GT: "const int r_max_ref = r_max * padding_factor;" (double product,
        # then (int) truncation) and "r_min_NN_ref_2" likewise truncated.
        pf = float(self.padding_factor)
        r_max_ref = int(self.r_max * self.padding_factor)
        r_max_ref_2 = r_max_ref * r_max_ref
        r_min_NN_ref_2 = int((self.r_min_nn * self.r_min_nn) * pf * pf)
        if f3d.ndim == 2:
            _rotate3d_slice_kernel(np.ascontiguousarray(self.data),
                                   np.ascontiguousarray(f3d),
                                   np.ascontiguousarray(Ainv),
                                   int(self.interpolator), r_max_ref_2,
                                   r_min_NN_ref_2, self.pad_size // 2)
        else:
            _rotate3d_kernel(np.ascontiguousarray(self.data),
                             np.ascontiguousarray(f3d),
                             np.ascontiguousarray(Ainv),
                             int(self.interpolator), r_max_ref_2,
                             r_min_NN_ref_2, self.pad_size // 2)


# ------------------------------------------------------------------ #
# rotate3D (GT src/projector.cpp:1124) for data_dim == 3
# ------------------------------------------------------------------ #
@njit(cache=True, fastmath=False)
def _floor_d(x):
    """GT macros.h FLOOR(x) for doubles (exact on integers, else the
    C macro's (int) truncation semantics)."""
    xi = int(x)
    if x == xi:
        return xi
    if x > 0:
        return xi
    return int(x - 1.0)


@njit(cache=True, fastmath=False)
def _round_d(x):
    """GT macros.h ROUND(x) = (x>0)?(int)(x+0.5):(int)(x-0.5)."""
    if x > 0:
        return int(x + 0.5)
    return int(x - 0.5)


@njit(cache=True, fastmath=False)
def _rotate3d_kernel(data, f3d, ainv, interpolator, r_max_ref_2,
                     r_min_NN_ref_2, pad_half):
    S = f3d.shape[0]
    r_max_out = f3d.shape[2] - 1
    r_max_out_2 = r_max_out * r_max_out
    for k in range(S):
        z = k if k <= r_max_out else k - S
        z2 = z * z
        for i in range(S):
            y = i if i <= r_max_out else i - S
            y2 = y * y
            yz2 = y2 + z2
            if yz2 > r_max_out_2:
                continue
            x_max = _floor_d(float(np.sqrt(r_max_out_2 - yz2)))
            for x in range(x_max + 1):
                xp = ainv[0, 0] * x + ainv[0, 1] * y + ainv[0, 2] * z
                yp = ainv[1, 0] * x + ainv[1, 1] * y + ainv[1, 2] * z
                zp = ainv[2, 0] * x + ainv[2, 1] * y + ainv[2, 2] * z
                # GT: "const int r_ref_2 = xp*xp + yp*yp + zp*zp;" (truncated)
                r_ref_2 = int(xp * xp + yp * yp + zp * zp)
                if r_ref_2 > r_max_ref_2:
                    continue
                if interpolator == TRILINEAR or r_ref_2 < r_min_NN_ref_2:
                    is_neg_x = xp < 0.0
                    xp2, yp2, zp2 = xp, yp, zp
                    if is_neg_x:
                        xp2, yp2, zp2 = -xp, -yp, -zp
                    x0 = _floor_d(xp2)
                    fx = xp2 - x0
                    y0 = _floor_d(yp2)
                    fy = yp2 - y0
                    z0 = _floor_d(zp2)
                    fz = zp2 - z0
                    zy0 = z0 + pad_half
                    zy1 = zy0 + 1
                    yi0 = y0 + pad_half
                    yi1 = yi0 + 1
                    d000 = data[zy0, yi0, x0]
                    d001 = data[zy0, yi0, x0 + 1]
                    d010 = data[zy0, yi1, x0]
                    d011 = data[zy0, yi1, x0 + 1]
                    d100 = data[zy1, yi0, x0]
                    d101 = data[zy1, yi0, x0 + 1]
                    d110 = data[zy1, yi1, x0]
                    d111 = data[zy1, yi1, x0 + 1]
                    # GT: LIN_INTERP(a, l, h) = l + (h - l) * a
                    dx00 = d000 + (d001 - d000) * fx
                    dx01 = d100 + (d101 - d100) * fx
                    dx10 = d010 + (d011 - d010) * fx
                    dx11 = d110 + (d111 - d110) * fx
                    dxy0 = dx00 + (dx10 - dx00) * fy
                    dxy1 = dx01 + (dx11 - dx01) * fy
                    val = dxy0 + (dxy1 - dxy0) * fz
                    if is_neg_x:
                        val = np.conj(val)
                    f3d[k, i, x] = val
                elif interpolator == NEAREST_NEIGHBOUR:
                    x0 = _round_d(xp)
                    y0 = _round_d(yp)
                    z0 = _round_d(zp)
                    if x0 < 0:
                        f3d[k, i, x] = np.conj(data[-z0 + pad_half,
                                                     -y0 + pad_half, -x0])
                    else:
                        f3d[k, i, x] = data[z0 + pad_half, y0 + pad_half,
                                            x0]
                else:
                    raise RuntimeError(
                        "Unrecognized interpolator in Projector::project")


@njit(cache=True, fastmath=False)
def _rotate3d_slice_kernel(data, f2d, ainv, interpolator, r_max_ref_2,
                           r_min_NN_ref_2, pad_half):
    """GT ``Projector::rotate3D`` for a 2-D f3d (ZSIZE(f3d) == 1).

    f2d is (S, S//2+1): i indexes the centered y axis, x the unsigned
    half axis.  Only the z=0 plane of the padded 3-D Fourier transform
    is sampled: (xp, yp, zp) = Ainv * (x, y, 0) * padding_factor.
    """
    S = f2d.shape[0]
    r_max_out = f2d.shape[1] - 1
    r_max_out_2 = r_max_out * r_max_out
    for i in range(S):
        y = i if i <= r_max_out else i - S
        y2 = y * y
        yz2 = y2  # z = 0
        if yz2 > r_max_out_2:
            continue
        x_max = _floor_d(float(np.sqrt(r_max_out_2 - yz2)))
        for x in range(x_max + 1):
            xp = ainv[0, 0] * x + ainv[0, 1] * y + ainv[0, 2] * 0
            yp = ainv[1, 0] * x + ainv[1, 1] * y + ainv[1, 2] * 0
            zp = ainv[2, 0] * x + ainv[2, 1] * y + ainv[2, 2] * 0
            # GT: "const int r_ref_2 = xp*xp + yp*yp + zp*zp;" (truncated)
            r_ref_2 = int(xp * xp + yp * yp + zp * zp)
            if r_ref_2 > r_max_ref_2:
                continue
            if interpolator == TRILINEAR or r_ref_2 < r_min_NN_ref_2:
                is_neg_x = xp < 0.0
                xp2, yp2, zp2 = xp, yp, zp
                if is_neg_x:
                    xp2, yp2, zp2 = -xp, -yp, -zp
                x0 = _floor_d(xp2)
                fx = xp2 - x0
                y0 = _floor_d(yp2)
                fy = yp2 - y0
                z0 = _floor_d(zp2)
                fz = zp2 - z0
                zy0 = z0 + pad_half
                zy1 = zy0 + 1
                yi0 = y0 + pad_half
                yi1 = yi0 + 1
                d000 = data[zy0, yi0, x0]
                d001 = data[zy0, yi0, x0 + 1]
                d010 = data[zy0, yi1, x0]
                d011 = data[zy0, yi1, x0 + 1]
                d100 = data[zy1, yi0, x0]
                d101 = data[zy1, yi0, x0 + 1]
                d110 = data[zy1, yi1, x0]
                d111 = data[zy1, yi1, x0 + 1]
                # GT: LIN_INTERP(a, l, h) = l + (h - l) * a
                dx00 = d000 + (d001 - d000) * fx
                dx01 = d100 + (d101 - d100) * fx
                dx10 = d010 + (d011 - d010) * fx
                dx11 = d110 + (d111 - d110) * fx
                dxy0 = dx00 + (dx10 - dx00) * fy
                dxy1 = dx01 + (dx11 - dx01) * fy
                val = dxy0 + (dxy1 - dxy0) * fz
                if is_neg_x:
                    val = np.conj(val)
                f2d[i, x] = val
            elif interpolator == NEAREST_NEIGHBOUR:
                x0 = _round_d(xp)
                y0 = _round_d(yp)
                z0 = _round_d(zp)
                if x0 < 0:
                    f2d[i, x] = np.conj(data[-z0 + pad_half,
                                                 -y0 + pad_half, -x0])
                else:
                    f2d[i, x] = data[z0 + pad_half, y0 + pad_half, x0]
            else:
                raise RuntimeError(
                    "Unrecognized interpolator in Projector::project")
