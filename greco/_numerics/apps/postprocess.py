# SPDX-License-Identifier: GPL-2.0-or-later
# RELION-derived numerical routines.
# Upstream source: RELION src/postprocessing.cpp
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
# Changes: Python port and Greco adaptation of postprocessing, output formats and reporting integration.
# Modified 2026-10-23: restored source attribution and per-file notices.
# See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
# and licenses/THIRD_PARTY_NOTICES.md.
#

"""Greco map postprocessing: FSC, phase randomization and sharpening.

Numerical reference routines retain their attribution in
licenses/THIRD_PARTY_NOTICES.md. Outputs use Greco STAR schemas and a vector PDF.
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from ..io.mrc import read_mrc, write_mrc, application_label
from ..libm_compat import sincos_array
from ..tomo import fft_f64

__all__ = ["Postprocessor", "main", "calculate_fsc", "glibc_rand_seq"]

M_2PI = 2.0 * 3.14159265358979323846  # 2*PI exactly as in reference (macros.h)
RAND_MAX = 2147483647


# --------------------------------------------------------------------------
# RNG: glibc rand()/srand() clone (TYPE_3, degree 31, sep 3), bit-exact
# --------------------------------------------------------------------------

def glibc_rand_seq(seed: int, n: int) -> np.ndarray:
    """First ``n`` values of glibc ``rand()`` after ``srand(seed)``.

    Mirrors glibc 2.34 ``__srandom_r``/``__random_r`` (stdlib/random_r.c):
    seed 0 is treated as 1, the state is seeded with the Park-Miller LCG
    (overflow-safe 127773 split, mod 2^31-1), pointers start at
    (sep, 0), 310 words are generated and discarded, and each output word
    is ``val = state[f] += (uint32) state[r]; val >> 1`` with the two
    pointers advancing cyclically (only one may wrap per step).
    """
    DEG, SEP = 31, 3
    seed = 1 if seed == 0 else seed
    st = [0] * DEG
    st[0] = seed
    word = seed
    for i in range(1, DEG):
        hi = word // 127773
        lo = word % 127773
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += 2147483647
        st[i] = word
    st = [w & 0xFFFFFFFF for w in st]
    fptr, rptr = SEP, 0
    out = np.empty(n, dtype=np.int32)
    k = 0
    total = DEG * 10 + n
    for it in range(total):
        val = (st[fptr] + st[rptr]) & 0xFFFFFFFF
        st[fptr] = val
        fptr += 1
        if fptr >= DEG:
            fptr = 0
            rptr += 1
        else:
            rptr += 1
            if rptr >= DEG:
                rptr = 0
        if it >= DEG * 10:
            out[k] = val >> 1
            k += 1
    return out


def glibc_unif_phases(seed: int, n: int,
                      a: float = 0.0, b: float = M_2PI) -> np.ndarray:
    """reference ``rnd_unif(a, b)`` = ``a + (float)rand()/(float)(RAND_MAX/(b-a))``.

    All intermediate arithmetic in float32, exactly like the C++ code.
    """
    if a == b:
        return np.full(n, a, dtype=np.float32)
    fa, fb = np.float32(a), np.float32(b)
    denom = np.float32(RAND_MAX) / (fb - fa)
    r = glibc_rand_seq(seed, n).astype(np.float32)
    return (fa + r / denom).astype(np.float32)


def _phase_cache_dir() -> str:
    d = os.environ.get("GRECO_CACHE",
                       os.path.join(os.path.expanduser("~"), ".cache", "greco._numerics"))
    os.makedirs(d, exist_ok=True)
    return d


def random_phases_sequence(seed: int, N: int, index: int,
                           npass: int) -> np.ndarray:
    """Phase sequence consumed by ``randomize_phases_beyond`` over ``npass``
    consecutive passes on an N-cube (r2c half spectrum).

    Caches the f32 sequence on disk, keyed by (seed, N, index, npass).
    """
    key = f"phases_seed{seed}_N{N}_idx{index}_p{npass}.f32"
    fn = os.path.join(_phase_cache_dir(), key)
    if os.path.exists(fn):
        arr = np.fromfile(fn, dtype="<f4")
        if arr.size >= npass * _n_phases(N, index):
            return arr
    # Count selected bins (k, i, j in FFTW logical coords, r2c layout)
    wh = N // 2 + 1
    k = np.arange(N)
    kp = np.where(k < wh, k, k - N).astype(np.int64)
    i = np.arange(N)
    ip = np.where(i < wh, i, i - N).astype(np.int64)
    j = np.arange(wh)
    r2 = (kp[:, None, None] ** 2 + ip[None, :, None] ** 2
          + j[None, None, :] ** 2)
    nsel = int((r2 >= index * index).sum())
    del r2
    phases = glibc_unif_phases(seed, npass * nsel, 0.0, M_2PI)
    tmp = fn + ".tmp"
    phases.astype("<f4").tofile(tmp)
    os.replace(tmp, fn)
    return phases


def _n_phases(N: int, index: int) -> int:
    wh = N // 2 + 1
    k = np.arange(N)
    kp = np.where(k < wh, k, k - N).astype(np.int64)
    i = np.arange(N)
    ip = np.where(i < wh, i, i - N).astype(np.int64)
    j = np.arange(wh)
    r2 = (kp[:, None, None] ** 2 + ip[None, :, None] ** 2
          + j[None, None, :] ** 2)
    return int((r2 >= index * index).sum())


# --------------------------------------------------------------------------
# FFT helpers (reference FourierTransformer conventions)
# --------------------------------------------------------------------------

def _fftw_logical_r2(N: int) -> np.ndarray:
    """(N, N, N/2+1) array of kp^2 + ip^2 + jp^2 in FFTW logical coords."""
    wh = N // 2 + 1
    return _fftw_r2_from_shape(N, N, wh)


def _fftw_r2_from_shape(nz: int, ny: int, wh: int) -> np.ndarray:
    """r2 grid over an r2c array of shape (nz, ny, wh) using the GT
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM coords (fftw.h:99): the wrap
    threshold for both k and i is XSIZE (= wh), so 2D FTs (1, N, wh)
    and cubic (N, N, wh) both work with identical arithmetic."""
    k = np.arange(nz)
    kp = np.where(k < wh, k, k - nz).astype(np.int64)
    i = np.arange(ny)
    ip = np.where(i < wh, i, i - ny).astype(np.int64)
    j = np.arange(wh)
    return (kp[:, None, None] ** 2 + ip[None, :, None] ** 2
            + j[None, None, :] ** 2)


def _fwd(m: np.ndarray) -> np.ndarray:
    """reference forward FFT: raw r2c / N**3.

    Greco's minimal native core calls FFTW directly and preserves the
    reference normalization and rounding. Engine selection
    (auto/native/pyfftw/numpy, GRECO_FFT_ENGINE) lives in fft_f64.
    """
    return fft_f64.rfftn3_fwdonly_f64(np.ascontiguousarray(m,
                                                           dtype=np.float64))


def _inv(FT: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """reference inverse FFT: raw c2r (unnormalised)."""
    return fft_f64.irfftn3_fwdonly_f64(FT, tuple(shape))


def _radial_native():
    """(fsc_accum, guinier_accum) from Greco's numerical core, or None.

    Sequential shell accumulation preserves the reference loop order for
    FSC and Guinier sums; NumPy indexed reductions may round differently.
    """
    try:
        lib = fft_f64._native_lib()
    except RuntimeError:
        return None
    if lib is None or not (hasattr(lib, "fsc_radial_accum")
                           and hasattr(lib, "guinier_accum")):
        return None
    fa = lib.fsc_radial_accum
    fa.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_long] * 4\
        + [ctypes.c_void_p] * 3
    fa.restype = None
    ga = lib.guinier_accum
    ga.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_long] * 4\
        + [ctypes.c_void_p] * 2
    ga.restype = None
    return fa, ga


def calculate_fsc(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """reference ``calculate_fsc(FT1, FT2, fsc)`` (shell 0..N/2, r2c logical coords).

    The radial sums use Greco's native ``fsc_radial_accum``, preserving
    the reference accumulation order.  The
    numpy ``np.add.at`` fallback below is NOT bit-exact with GT.
    """
    N = A.shape[-1]
    wh = N // 2 + 1
    FT1 = _fwd(A)
    FT2 = _fwd(B)
    num = np.zeros(wh, dtype=np.float64)
    den1 = np.zeros(wh, dtype=np.float64)
    den2 = np.zeros(wh, dtype=np.float64)
    native = _radial_native()
    if native is not None:
        r1 = np.ascontiguousarray(FT1.real)
        i1 = np.ascontiguousarray(FT1.imag)
        r2 = np.ascontiguousarray(FT2.real)
        i2 = np.ascontiguousarray(FT2.imag)
        native[0](r1.ctypes.data, i1.ctypes.data, r2.ctypes.data, i2.ctypes.data,
                wh, N, N, FT1.size,
                num.ctypes.data, den1.ctypes.data, den2.ctypes.data)
    else:
        idx = np.rint(np.sqrt(_fftw_logical_r2(N))).astype(np.int64)
        sel = idx < wh
        c1 = FT1[sel]
        c2 = FT2[sel]
        ii = idx[sel]
        # GT tComplex<float> product (strict f32 components, no FMA):
        # Re(conj(c1)*c2) = c1.re*c2.re + c1.im*c2.im
        np.add.at(num, ii, (c1.real * c2.real + c1.imag * c2.imag))
        # GT tComplex::abs() = sqrt(re*re + im*im); term = abs^2
        a1 = np.sqrt(c1.real * c1.real + c1.imag * c1.imag)
        a2 = np.sqrt(c2.real * c2.real + c2.imag * c2.imag)
        np.add.at(den1, ii, a1 * a1)
        np.add.at(den2, ii, a2 * a2)
    with np.errstate(invalid="ignore", divide="ignore"):
        fsc = num / np.sqrt(den1 * den2)
    return fsc


def randomize_phases_beyond(m: np.ndarray, index: int,
                          phases: np.ndarray) -> np.ndarray:
    """reference ``randomize_phases_beyond(v, index)`` with a precomputed phase
    sequence (glibc ``rnd_unif(0, 2*PI)`` values in FFTW loop order)."""
    N = m.shape[-1]
    r2 = _fftw_logical_r2(N)
    sel = (r2 >= index * index).reshape(-1)
    FT = _fwd(m)
    FTflat = FT.reshape(-1)
    # GT: mag = abs(Complex) = JAZ tComplex::abs() = sqrt(re*re+im*im)
    mag = np.sqrt(FT.real * FT.real + FT.imag * FT.imag)
    mag = mag.reshape(-1)[sel]
    ph = np.asarray(phases, dtype=np.float32)
    if ph.size != int(sel.sum()):
        raise RuntimeError("phase sequence length mismatch "
                           f"({ph.size} vs {int(sel.sum())})")
    s, c = sincos_array(ph.astype(np.float64))
    FTflat[sel] = mag.astype(np.float64) * (c + 1j * s)
    del FTflat
    return _inv(FT, m.shape)


def apply_fsc_weighting(FT: np.ndarray, fsc: np.ndarray) -> None:
    """In-place reference ``apply_fsc_weighting(FT, my_fsc)``."""
    ires_max = 0
    for i in range(len(fsc)):
        if fsc[i] < 0.0001:
            break
        ires_max = i
    N = FT.shape[0]
    ires = np.rint(np.sqrt(_fftw_logical_r2(N).astype(np.float64))).astype(np.int64)
    le = ires <= ires_max
    w = np.zeros(FT.shape, dtype=np.float64)
    fv = fsc[ires[le]]
    ok = fv > 0.0
    wf = np.zeros_like(fv)
    wf[ok] = np.sqrt((2.0 * fv[ok]) / (1.0 + fv[ok]))
    w[le] = wf
    FT *= w
    # GT zeroes beyond ires_max by direct assignment (= 0. -> +0.0), not by
    # multiplication (which would leave the sign of the original FT in the
    # zero bits); match that exactly.
    FT[~le] = 0.0


def apply_bfactor(FT: np.ndarray, ori_size: int,
                      bfactor: float, angpix: float) -> None:
    # GT loops over the FT's own dims (2D: (1, N, wh); 3D: (N, N, wh));
    # res denominator uses ori_size (the real image XSIZE).
    nz, ny, wh = FT.shape
    r2f = _fftw_r2_from_shape(nz, ny, wh).astype(np.float64)
    res = np.sqrt(r2f) / (ori_size * angpix)
    keep = res <= 1.0 / (angpix * 2.0)
    FT[keep] *= np.exp(-(bfactor / 4.0) * res[keep] ** 2)
    FT[~keep] = 0.0


def low_pass_filter(FT: np.ndarray, ori_size: int, low_pass: float,
                     angpix: float, filter_edge_width: int = 2,
                     do_highpass_instead: bool = False) -> None:
    # GT ROUND(x) = (x>=0) ? (int)(x+0.5) : (int)(x-0.5) (half away
    # from zero), not banker's rounding.
    xf = ori_size * angpix / low_pass
    ires_filter = int(xf + 0.5) if xf >= 0 else int(xf - 0.5)
    edge_half = filter_edge_width // 2
    edge_low = max(0.0, (ires_filter - edge_half) / ori_size)
    edge_high = min(FT.shape[2], (ires_filter + edge_half) / ori_size)
    edge_w = edge_high - edge_low
    nz, ny, wh = FT.shape
    r2f = _fftw_r2_from_shape(nz, ny, wh).astype(np.float64)
    res = np.sqrt(r2f) / ori_size
    mid = (res >= edge_low) & (res <= edge_high)
    if do_highpass_instead:
        FT[res < edge_low] = 0.0
        if edge_w > 0:
            FT[mid] *= 0.5 - 0.5 * np.cos(math.pi * (res[mid] - edge_low) / edge_w)
    else:
        FT[res > edge_high] = 0.0
        if edge_w > 0:
            FT[mid] *= 0.5 + 0.5 * np.cos(math.pi * (res[mid] - edge_low) / edge_w)


def make_guinier_plot(FT: np.ndarray, N: int, angpix: float,
                    fit_minres: float, fit_maxres: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """reference ``make_guinier_plot`` -> (x, y, w) arrays."""
    wh = N // 2 + 1
    lnF = np.zeros(wh, dtype=np.float64)
    cnt = np.zeros(wh, dtype=np.int64)
    native = _radial_native()
    if native is not None:
        r = np.ascontiguousarray(FT.real)
        im = np.ascontiguousarray(FT.imag)
        native[1](r.ctypes.data, im.ctypes.data, wh, N, N, FT.size,
                lnF.ctypes.data, cnt.ctypes.data)
    else:
        idx = np.rint(np.sqrt(_fftw_logical_r2(N).astype(np.float64))).astype(np.int64)
        inb = idx < wh
        ii = idx[inb]
        mag = np.sqrt(FT.real * FT.real + FT.imag * FT.imag)[inb]
        np.add.at(lnF, ii, mag)
        np.add.at(cnt, ii, 1)
    xsize_ang = N * angpix
    xs, ys, ws = [], [], []
    for i in range(wh):
        res = xsize_ang / i if i > 0 else math.inf
        if res >= angpix * 2.0:
            xs.append(1.0 / (res * res) if res != math.inf else 0.0)
            if lnF[i] > 0.0:
                ys.append(math.log(lnF[i] / cnt[i]))
                ws.append(1.0 if (res <= fit_minres and res >= fit_maxres) else 0.0)
            else:
                ys.append(-99.0)
                ws.append(0.0)
    return np.array(xs), np.array(ys), np.array(ws)


def fit_weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray
                    ) -> Tuple[float, float, float]:
    """reference ``fit_weighted_line`` (weighted least squares)."""
    ss_xy = np.sum(w * x * y)
    ss_xx = np.sum(w * x * x)
    ss_yy = np.sum(w * y * y)
    sum_w = float(np.sum(w))
    if sum_w <= 0.0:
        return 0.0, 0.0, 0.0
    ave_x = ss_xy  # placeholder, recompute properly below
    ave_x = np.sum(w * x) / sum_w
    ave_y = np.sum(w * y) / sum_w
    ss_xx -= sum_w * ave_x * ave_x
    ss_yy -= sum_w * ave_y * ave_y
    ss_xy -= sum_w * ave_x * ave_y
    if ss_xx > 0.0:
        slope = ss_xy / ss_xx
        intercept = ave_y - slope * ave_x
        corr = ss_xy * ss_xy / (ss_xx * ss_yy) if ss_yy > 0 else 0.0
        return float(slope), float(intercept), float(corr)
    return 0.0, 0.0, 0.0


def divide_by_mtf(FT: np.ndarray, N: int, angpix: float, mtf_angpix: float,
                mtf_resol: np.ndarray, mtf_value: np.ndarray) -> None:
    if np.any(mtf_value < 1e-10):
        raise RuntimeError("zero or negative values encountered in MTF curve")
    res_per_elem = (mtf_resol[-1] - mtf_resol[0]) / len(mtf_resol)
    if res_per_elem < 1e-10:
        raise RuntimeError("the resolution in the MTF star file does not go up")
    xsize_ang = angpix * N
    r2f = _fftw_logical_r2(N).astype(np.float64)
    res = np.sqrt(r2f) / xsize_ang
    sel = res < 1.0 / (2.0 * angpix)
    r = res[sel]
    i0 = np.floor(r / res_per_elem).astype(np.int64)
    mtf = np.empty(r.shape, dtype=np.float64)
    lo = i0 <= 0
    hi = i0 >= len(mtf_value) - 1
    mid = ~(lo | hi)
    mtf[lo] = mtf_value[0]
    mtf[hi] = mtf_value[-1]
    if mid.any():
        i0m = i0[mid]
        x0, x1 = mtf_resol[i0m], mtf_resol[i0m + 1]
        y0, y1 = mtf_value[i0m], mtf_value[i0m + 1]
        mtf[mid] = y0 + (y1 - y0) * (r[mid] - x0) / (x1 - x0)
    FT[sel] /= mtf


# --------------------------------------------------------------------------
# Postprocessing workflow
# --------------------------------------------------------------------------

class Postprocessor:
    def __init__(self, args, argv: Optional[Sequence[str]] = None):
        self.args = args
        self.do_fsc_weighting = not args.skip_fsc_weighting
        self.verb = args.verb

    def _v(self, msg: str) -> None:
        if self.verb > 0:
            print(msg, flush=True)

    def run(self) -> int:
        a = self.args
        t0 = time.time()

        # ---- read input maps
        self._v("== Reading input half-reconstructions: ")
        I1img = read_mrc(a.i)
        self.I1 = I1img.data.astype(np.float64)
        self._v("  + half1-map:                     " + a.i)
        fn_i2 = a.i2
        if not fn_i2:
            fn_i2 = a.i.replace("half1", "half2")
        I2img = read_mrc(fn_i2)
        I2 = I2img.data.astype(np.float64)
        self._v("  + half2-map:                     " + fn_i2)
        if self.I1.shape != I2.shape:
            raise RuntimeError("input half maps have different shapes")
        N = self.I1.shape[-1]
        if N != self.I1.shape[0] or N != self.I1.shape[1]:
            raise RuntimeError("postprocess requires cubic input maps")

        # ---- pixel size
        self.angpix = a.angpix
        if self.angpix <= 0:
            self.angpix = I1img.header["cell_a"] / N
            print("WARNING: You did not specify --angpix. The pixel size in the "
                  "image header, %s A/px, is used." % ("%.2f" % self.angpix),
                  file=sys.stderr)
        self.mtf_angpix = a.mtf_angpix
        if self.mtf_angpix < 0:
            self._v(" + --mtf_angpix was not provided, assuming pixel size in "
                    "raw micrographs is the same as in particles, "
                    "%s A/px." % ("%.2f" % self.angpix))
            self.mtf_angpix = self.angpix

        # ---- molecular weight / fractions
        self.molweight = a.molweight
        self.frac_molweight = 0.0
        if self.molweight > 0:
            self.frac_molweight = (0.476 * (N * self.angpix) ** 3 * 0.81
                                   / (self.molweight * 1000))

        # ---- option conflicts (mirrors C++)
        if a.auto_bfac and abs(a.adhoc_bfac) > 0:
            raise RuntimeError("provide either --auto_bfac OR --adhoc_bfac, "
                               "but not both!")

        os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True)

        # ---- FSC of unmasked maps
        self.fsc_unmasked = calculate_fsc(self.I1, I2)

        # ---- mask
        self.do_mask = bool(a.mask)
        self.frac_solvent_mask = 0.0
        self.molweight_frommask = 0.0
        self.Im = None
        if self.do_mask:
            self.load_mask()
            self._v("== Masking input maps ...")
            I1m = self.I1 * self.Im
            I2m = I2 * self.Im
            self.fsc_masked = calculate_fsc(I1m, I2m)

            # ---- randomize phases beyond threshold
            self.randomize_at = -1
            if a.randomize_at_A > 0:
                self.randomize_at = int(self.angpix * N / a.randomize_at_A)
            else:
                for i in range(1, len(self.fsc_unmasked)):
                    if self.fsc_unmasked[i] < a.randomize_at_fsc:
                        self.randomize_at = i
                        break
            if self.randomize_at <= 0:
                raise RuntimeError("FSC curve never drops below "
                                   "randomize_fsc_at.")
            self._v("  + randomize phases beyond:       "
                    "%s Angstroms" % ("%.4f" % (N * self.angpix / self.randomize_at)))

            nsel = _n_phases(N, self.randomize_at)
            self._v("  + generating %d random phases (seed %d) ..."
                    % (2 * nsel, a.random_seed))
            t1 = time.time()
            phases = random_phases_sequence(a.random_seed, N,
                                            self.randomize_at, npass=2)
            self._v("    phases ready in %.1f s" % (time.time() - t1))
            I1p = randomize_phases_beyond(self.I1, self.randomize_at, phases[:nsel])
            I2p = randomize_phases_beyond(I2, self.randomize_at, phases[nsel:])
            self.fsc_random_masked = calculate_fsc(I1p * self.Im, I2p * self.Im)
            del I1m, I2m, I1p, I2p, phases

            # ---- corrected FSC + cisTEM-like part FSCs
            self.fsc_true = self._correct_masked_fsc(
                self.fsc_unmasked, self.fsc_masked,
                self.fsc_random_masked, self.randomize_at)
            self.fsc_part_fracmask = self._estimate_partial_fsc(
                self.fsc_unmasked, self.frac_solvent_mask)
            self.fsc_part_molweight = (self._estimate_partial_fsc(
                self.fsc_unmasked, self.frac_molweight)
                if self.molweight > 0 else None)
        else:
            self.fsc_true = self.fsc_unmasked
            self.fsc_part_molweight = (self._estimate_partial_fsc(
                self.fsc_unmasked, self.frac_molweight)
                if self.molweight > 0 else None)

        # ---- global resolution
        self.global_resol = 999.0
        self.global_resol_i = 0
        for i in range(len(self.fsc_true)):
            if self.fsc_true[i] < 0.143:
                break
            self.global_resol = (N * self.angpix / i) if i > 0 else math.inf
            self.global_resol_i = i

        # fractional (interpolated) resolution
        self.fract_resol = self.global_resol
        if a.interpolate and self.global_resol_i < len(self.fsc_true) - 1:
            v0 = self.fsc_true[self.global_resol_i] - 0.143
            v1 = self.fsc_true[self.global_resol_i + 1] - 0.143
            r = v0 / (v0 - v1)
            self.fract_resol = N * self.angpix / (self.global_resol_i + r)

        # ---- sanity checks
        if self.do_mask:
            unmasked_resol_i = 0
            for i in range(len(self.fsc_unmasked)):
                if self.fsc_unmasked[i] < 0.143:
                    break
                unmasked_resol_i = i
            if unmasked_resol_i > self.global_resol_i:
                if a.force_mask:
                    print(" WARNING: the unmasked FSC extends beyond the "
                          "solvent-corrected FSC.", file=sys.stderr)
                else:
                    print(" WARNING: the unmasked FSC extends beyond the "
                          "solvent-corrected FSC. Skip masking for now, but "
                          "you may want to adjust you mask!", file=sys.stderr)
                    print("          You can force the mask by the "
                          "'--force_mask' option.", file=sys.stderr)
                    self.fsc_true = self.fsc_unmasked
                    self.global_resol = (N * self.angpix / unmasked_resol_i
                                         if unmasked_resol_i > 0 else math.inf)
            elif (self.global_resol_i < len(self.fsc_random_masked)
                    and self.fsc_random_masked[self.global_resol_i] > 0.1):
                print(" WARNING: The phase-randomised FSC is larger than 0.10 "
                      "at the estimated resolution!", file=sys.stderr)
                print(" WARNING: This may result in an incorrect resolution "
                      "estimation. Provide a softer mask with less features "
                      "to get lower phase-randomised FSCs.", file=sys.stderr)

        # ---- merge halves
        I2 = I2img.data.astype(np.float64)
        self.I1 = 0.5 * (self.I1 + I2)

        # ---- sharpen
        self.applied_filter = self.sharpen_map()

        # ---- outputs
        self.write_output()

        # ---- optional half maps (C++ forces: no auto_bfac, ad-hoc bfac =
        # the global one, low_pass = applied_filter, silent)
        if a.half_maps:
            from types import SimpleNamespace
            print("== Writing half maps after applying same sharpening and "
                  "low-pass filter", flush=True)
            for tag, fn in (("half1", a.i), ("half2", fn_i2)):
                print("  + %s" % tag, flush=True)
                a2 = SimpleNamespace(**vars(a))
                a2.auto_bfac = False
                a2.adhoc_bfac = self.global_bfactor
                a2.low_pass = self.applied_filter
                a2.verb = 0
                m = read_mrc(fn).data.astype(np.float64)
                self.args = a2
                self.I1 = m
                self.sharpen_map()
                from pathlib import Path
                destination=Path(a.o)/'volumes'/(tag+'.mrc')
                write_mrc(str(destination),self.I1,pixel_size=self.angpix,title=application_label())
                if self.do_mask:
                    write_mrc(str(destination.with_stem(tag+'_masked')),self.I1*self.Im,pixel_size=self.angpix,title=application_label())
            self.args = a

        print("done in %.1f s" % (time.time() - t0), flush=True)
        return 0

    # ------------------------------------------------------------------ mask

    def load_mask(self) -> None:
        a = self.args
        self._v("== Using a user-provided mask ...")
        self._v("  + input mask:                    " + a.mask)
        Imimg = read_mrc(a.mask)
        self.Im = Imimg.data.astype(np.float64)
        N = self.I1.shape[-1]
        if self.Im.shape != self.I1.shape:
            raise RuntimeError("mask size differs from input maps")
        minval = float(self.Im.min())
        maxval = float(self.Im.max())
        if minval < -1e-6 or maxval - 1.0 > 1e-6:
            raise RuntimeError("mask values not in range [0,1] (min=%s max=%s)"
                               % (minval, maxval))
        avg = float((self.Im > 0.5).sum()) / (N ** 3)
        self.frac_solvent_mask = 0.476 / avg
        self.molweight_frommask = avg * (N * self.angpix) ** 3 * 0.81
        self._v("  + fraction f (solvent mask based): "
                "%.4f" % self.frac_solvent_mask)
        self._v("  + molecular weight inside protein mask: "
                "%.4e" % self.molweight_frommask)

    # --------------------------------------------------------------- FSC math

    @staticmethod
    def _correct_masked_fsc(fsc_unmasked, fsc_masked, fsc_random_masked,
                          randomize_at) -> np.ndarray:
        if fsc_masked[0] <= 0.0:
            fsc_masked[0] = 1.0
        if fsc_unmasked[0] <= 0.0:
            fsc_unmasked[0] = 1.0
        if fsc_random_masked[0] <= 0.0:
            fsc_random_masked[0] = 1.0
        fsc_true = np.empty_like(fsc_masked)
        lo = np.arange(len(fsc_masked)) < randomize_at + 2
        fsc_true[lo] = fsc_masked[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            fsc_true[~lo] = ((fsc_masked[~lo] - fsc_random_masked[~lo])
                             / (1.0 - fsc_random_masked[~lo]))
        return fsc_true

    @staticmethod
    def _estimate_partial_fsc(fsc_unmasked, fraction) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return (fraction * fsc_unmasked
                    / (1.0 + (fraction - 1.0) * fsc_unmasked))

    # ---------------------------------------------------------------- sharpen

    def sharpen_map(self) -> float:
        a = self.args
        N = self.I1.shape[-1]
        FT = _fwd(self.I1)

        self.guinierin = make_guinier_plot(FT, N, self.angpix,
                                         a.autob_lowres, a.autob_highres)
        if a.mtf:
            self._v("== Dividing map by the MTF of the detector ...")
            self._v("  + mtf STAR-file: " + a.mtf)
            mtf_resol, mtf_value = _read_mtf(a.mtf, self.mtf_angpix)
            divide_by_mtf(FT, N, self.angpix, self.mtf_angpix,
                        mtf_resol, mtf_value)
            self.guinierinvmtf = make_guinier_plot(FT, N, self.angpix,
                                                 a.autob_lowres, a.autob_highres)
        if self.do_fsc_weighting:
            self._v("== Applying sqrt(2*FSC/(FSC+1)) weighting "
                    "(as in Rosenthal & Henderson, 2003) ...")
            apply_fsc_weighting(FT, self.fsc_true)
        self.guinierweighted = make_guinier_plot(FT, N, self.angpix,
                                               a.autob_lowres, a.autob_highres)

        self.global_bfactor = 0.0
        self.global_slope = self.global_intercept = self.global_corr_coeff = 0.0
        if a.auto_bfac:
            self._v("== Fitting straight line through Guinier plot to find "
                    "B-factor ...")
            self._v("  + fit from resolution: " + str(a.autob_lowres))
            self._v("  + fit until resolution: " + str(a.autob_highres))
            x, y, w = self.guinierweighted
            self.global_slope, self.global_intercept, self.global_corr_coeff =\
                fit_weighted_line(x, y, w)
            self.global_bfactor = 4.0 * self.global_slope
            self._v("  + slope of fit: " + str(self.global_slope))
            self._v("  + intercept of fit: " + str(self.global_intercept))
            self._v("  + correlation of fit: " + str(self.global_corr_coeff))
        elif abs(a.adhoc_bfac) > 0:
            self._v("== Using a user-provided (ad-hoc) B-factor ...")
            if a.adhoc_bfac > 0:
                print(" WARNING: using a positive B-factor. This will "
                      "effectively dampen your map. Use negative value to "
                      "sharpen it!", file=sys.stderr)
            self.global_bfactor = a.adhoc_bfac

        if abs(self.global_bfactor) > 0:
            self._v("  + apply b-factor of: " + str(self.global_bfactor))
            apply_bfactor(FT, N, self.global_bfactor, self.angpix)
        else:
            print(" WARNING: map will not be sharpened. This map may be used "
                  "as reference for subsequent refinements, but you would "
                  "need to run a postprocessing job with B-factor sharpening "
                  "to better visualise high-resolution features.",
                  file=sys.stderr)

        self.guiniersharpen = make_guinier_plot(FT, N, self.angpix,
                                              a.autob_lowres, a.autob_highres)

        applied_filter = a.low_pass
        if a.low_pass != 0:
            if a.low_pass < 0:
                applied_filter = self.global_resol
            self._v("== Low-pass filtering final map ...")
            self._v("  + filter frequency: " + str(applied_filter))
            low_pass_filter(FT, N, applied_filter, self.angpix,
                             a.filter_edge_width)

        self.I1 = _inv(FT, self.I1.shape)
        return applied_filter

    # ------------------------------------------------------------------ maps

    def write_output(self) -> None:
        from greco.sta.postprocess_report import publish
        self.output_metadata = publish(self)


def _read_mtf(fn: str, mtf_angpix: float
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Read the Greco MTF loop with explicit inverse-pixel frequency units."""
    from greco.particle_star import rows
    samples=[row for block,row in rows(fn) if block=='mtf']
    if not samples:raise ValueError('MTF STAR requires a data_mtf loop.')
    rr=np.array([row['grecoSpatialFrequencyInvPixel']/mtf_angpix for row in samples])
    vv=np.array([row['grecoMtfValue'] for row in samples])
    return rr,vv


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="greco._numerics postprocess",
        description="Greco map postprocessing")
    p.add_argument("--i", required=True,
                   help="input name of half1 (e.g. run_half1_unfil.mrc)")
    p.add_argument("--i2", default="",
                   help="input name of half2 (default: half1 -> half2)")
    p.add_argument("--ios", default="",
                   help="input optimisation set (updated set is written)")
    p.add_argument("--o", default="postprocess", help="output workspace directory")
    p.add_argument("--angpix", type=float, default=-1.0,
                   help="pixel size in Angstroms (default: from header)")
    p.add_argument("--mtf_angpix", type=float, default=-1.0,
                   help="pixel size of the raw micrographs/movies (A)")
    p.add_argument("--molweight", type=float, default=-1.0,
                   help="molecular weight (kDa) of ordered protein mass")
    p.add_argument("--mask", default="",
                   help="user-provided mask (1=protein, 0=solvent)")
    p.add_argument("--force_mask", action="store_true",
                   help="use the mask even when masked resolution is worse")
    p.add_argument("--mtf", default="", help="STAR file with the MTF curve")
    p.add_argument("--autob_lowres", type=float, default=10.0,
                   help="lowest resolution (A) for the B-factor fit")
    p.add_argument("--autob_highres", type=float, default=0.0,
                   help="highest resolution (A) for the B-factor fit")
    p.add_argument("--auto_bfac", action="store_true",
                   help="automatic B-factor (Rosenthal & Henderson 2003)")
    p.add_argument("--adhoc_bfac", type=float, default=0.0,
                   help="user-provided B-factor (A^2), e.g. -400")
    p.add_argument("--skip_fsc_weighting", action="store_true",
                   help="do not FSC-weight the sharpening")
    p.add_argument("--low_pass", type=float, default=0.0,
                   help="low-pass resolution (A); 0: off, "
                        "negative: resolution at FSC=0.143")
    p.add_argument("--randomize_at_fsc", type=float, default=0.8,
                   help="randomize phases where FSC drops below this value")
    p.add_argument("--randomize_at_A", type=float, default=-1.0,
                   help="randomize phases from this resolution (A)")
    p.add_argument("--filter_edge_width", type=int, default=2,
                   help="raised-cosine edge width in shells")
    p.add_argument("--interpolate", action="store_true",
                   help="interpolate the FSC for a fractional resolution")
    p.add_argument("--half_maps", action="store_true",
                   help="write post-processed half maps")
    p.add_argument("--verb", type=int, default=1)
    p.add_argument("--random_seed", type=int, default=0,
                   help="RNG seed (default 0)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    args = _build_parser().parse_args(argv)
    pp = Postprocessor(args, argv)
    return pp.run()


if __name__ == "__main__":
    sys.exit(main())
