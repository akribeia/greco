# Third-party references and adapted components
# MotionCor3 full-frame alignment adaptation. Source: Align/GCorrelateSum2D.cu, CPeak2D and the full-frame alignment schedule.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/motioncor3/LICENSE.txt
#
# Upstream notice (retained):
# Copyright 2023 Chan Zuckerberg Institute for Advanced Biological Imaging
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""Greco full-frame motion correction with a MotionCor3-compatible algorithm.

MotionCor3 reference parameters::

    -InTiff movie -Serial 0 -Gain <reciprocal-gain tiff>
    -Patch 0 0 0   # full-frame alignment only, no patches / local field
    -Iter 15 -Tol 0.10 -Bft 500 100 -PhaseOnly 0 -FtBin 1
    -InitDose 0 -FmDose 0              # no dose weighting in the alignment
    -SumRange 3 25 -SplitSum 1 -Group 1 4 -FmRef -1 -Align 1

Algorithm follows the MotionCor3 full-frame alignment source:

* Each frame is correlated against the aligned sum of all *other* frames
  (the reference sum has the frame's own spectrum subtracted, as in
  ``GCorrelateSum2D::mGConv`` with ``bSubtract``).
* The cross-spectrum is filtered with MC3's B-factor kernel, derived from
  ``mGLowpass`` in ``Align/GCorrelateSum2D.cu``::

      W(k) = exp(-0.25 * B * (ix^2 + iy^2) / (N/2) * N)
           = exp(-0.5 * B * |k|^2)          (k in cycles/pixel)

  ``Bft 500 100`` sets the initial B-factor to 500 px^2 (the 2nd value is
  the patch B-factor, unused without patches); B decreases by 2 each
  iteration, floored at 20 (over 15 iterations: 500 -> 472). Frequencies
  are measured on the Fourier-cropped correlation buffer, whose side is
  2048 for native square movies larger than 2048.
* The positive peak is searched in a 128x128 correlation-buffer window,
  then refined with independent three-point parabolas (``CPeak2D``).
* 15 iterations; each iteration rebuilds the aligned sum with the current
  shifts and re-measures every frame's residual shift; stops when the max
  residual norm < 0.10 px (from iteration 2 on).
* Post-processing on the total shift trajectory: spike removal
  (``CRemoveSpikes1D`` quadratic spline cleanup) + temporal smoothing
  ``s_i <- w s_i + 0.5(1-w)(s_{i-1}+s_{i+1})`` with w = 0.5 (inside
  ``CIterativeAlign``) and w = 0.4 (``0.3 + 0.1*binZ``, in
  ``CFullAlign::DoIt``).
* Full-frame shifts retain MC3's iterative gauge. ``-FmRef`` is applied
  only by patch alignment, not by ``CFullAlign`` with ``Patch 0 0``.

Shift convention (unchanged): ``full_shifts[i] = (dy, dx)`` is the signed
content displacement of frame *i* in the iterative full-frame gauge; aligning frame
*i* therefore samples it at ``(y + dy, x + dx)``.

The returned ``aligned`` image is the dose-weighted mean of the warped
centred frames (greco convention; with uniform frames it equals the MC3
plain sum up to a constant).  ``preprocess`` applies the detector gain and
the TIFF->MC3 raster flip (measured D4: ``flipud``) outside this module.
"""

from __future__ import annotations

import time

import numpy as np


class _Dev:
    """Tiny device adapter so the same code runs on cupy or numpy."""

    def __init__(self, gpu: bool = True, device: int | None = None):
        import contextlib
        self._ctx = contextlib.nullcontext()
        if gpu:
            import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter, map_coordinates
            self.asarray = cp.asarray
            self.fft2 = cp.fft.fft2
            self.rfft2 = cp.fft.rfft2
            self.irfft2 = cp.fft.irfft2
            self.ifft2 = cp.fft.ifft2
            self.map_coordinates = map_coordinates
            self.gaussian_filter = gaussian_filter
            self.abs = cp.abs
            self.argmax = cp.argmax
            self.exp = cp.exp
            self.conj = cp.conj
            self.name = "cupy"
            if device is not None:
                self._ctx = cp.cuda.Device(device)
        else:
            from scipy.ndimage import gaussian_filter, map_coordinates
            self.asarray = np.asarray
            self.fft2 = np.fft.fft2
            self.rfft2 = np.fft.rfft2
            self.irfft2 = np.fft.irfft2
            self.ifft2 = np.fft.ifft2
            self.map_coordinates = map_coordinates
            self.gaussian_filter = gaussian_filter
            self.abs = np.abs
            self.argmax = np.argmax
            self.exp = np.exp
            self.conj = np.conjugate
            self.name = "numpy"
        array_module = np if self.name == "numpy" else cp
        self.roll = array_module.roll
        self.sqrt = array_module.sqrt
        self.sin = array_module.sin
        self.cos = array_module.cos
        self.zeros = array_module.zeros
        self.zeros_like = array_module.zeros_like
        self.flip = array_module.flip
        self.where = array_module.where
        self.maximum = array_module.maximum

    def asnumpy(self, x):
        import cupy as cp
        return cp.asnumpy(x) if self.name == "cupy" else np.asarray(x)

    def cuda(self):
        """Context manager binding this thread to the configured device."""
        return self._ctx


def _bfactor_grid(size: int):
    """|k|^2 in cycles/pixel^2, unshifted numpy FFT layout (DC at [0, 0])."""
    k = np.fft.fftfreq(size).astype(np.float32)
    k2 = (k[None, :] ** 2 + k[:, None] ** 2).astype(np.float32)
    return k2


def _phase_shift(dev: _Dev, F, ramp_y, ramp_x, dy: float, dx: float):
    """Spectrum of the image whose content is displaced by (dy, dx):
    F * exp(+i 2pi (ky dy + kx dx))."""
    if dy == 0.0 and dx == 0.0:
        return F
    ramp = ramp_y * float(dy) + ramp_x * float(dx)
    return F * dev.exp(1j * ramp)


def _windowed_peak(dev: _Dev, cc, radius: int):
    """Signed (dy, dx) peak of the unshifted-layout circular cross-
    correlation, searched inside the central (2R+1)^2 lag window around
    zero (lags >= half the size wrap negative)."""
    n = cc.shape[0]
    radius = min(radius, n // 2)
    lags = np.arange(-radius, radius) % n
    idx = dev.asarray(lags, dtype=np.int64)
    sub = cc[idx][:, idx]
    flat = int(dev.argmax(sub))
    py = int(np.clip(flat // sub.shape[1], 1, len(lags) - 2))
    px = int(np.clip(flat % sub.shape[1], 1, len(lags) - 2))
    iy, ix = int(lags[py]), int(lags[px])
    ny = np.array([(iy + d) % n for d in (-1, 0, 1)], dtype=np.int64)
    nx = np.array([(ix + d) % n for d in (-1, 0, 1)], dtype=np.int64)
    patch = dev.asnumpy(cc[np.ix_(ny, nx)])
    def offset(a, b, c):
        value = -(c - a) / (2 * (a + c - 2 * b) + 1e-30)
        return float(value) if abs(value) <= 1 else 0.0
    dy = offset(patch[0, 1], patch[1, 1], patch[2, 1])
    dx = offset(patch[1, 0], patch[1, 1], patch[1, 2])
    sy = iy if iy < n // 2 else iy - n
    sx = ix if ix < n // 2 else ix - n
    return sy + float(dy), sx + float(dx)


def _smooth_shifts(shifts: np.ndarray, w: float) -> np.ndarray:
    """MC3 ``CStackShift::Smooth``: interior frame i gets
    w*s_i + 0.5(1-w)*(s_{i-1} + s_{i+1}); the two end frames are kept."""
    s = np.asarray(shifts, float)
    if w >= 1.0 or s.shape[0] < 3:
        return s
    out = s.copy()
    c = 0.5 * (1.0 - w)
    out[1:-1] = w * s[1:-1] + c * (s[:-2] + s[2:])
    return out


def _remove_spikes(shifts: np.ndarray) -> np.ndarray:
    """MC3 single-spike passes and quadratic truncated-power spline cleanup.

    Stable least squares replaces the C++ float normal-equation solver.
    """
    s = np.asarray(shifts, float).copy()
    n = len(s)
    if n < 3:
        return s
    for _ in range(2):
        old = s.copy()
        bad = (abs(old[1:-1] - old[:-2]) >= 5) & (abs(old[1:-1] - old[2:]) >= 5)
        s[1:-1] = np.where(bad, (old[:-2] + old[2:]) / 2, s[1:-1])
        for i, j, k in ((0, 1, 2), (n-1, n-2, n-3)):
            bad = (abs(old[i]-old[j]) > 10) & (abs(old[i]) > abs(old[j]))
            s[i] = np.where(bad, 2*old[j]-old[k], s[i])
    t = np.sqrt(np.arange(n) / n)
    count = min(6, max(0, (n-30)//15))
    if count == 0 and n >= 30:
        count = 1
    knots = t[15*np.arange(1, count+1)]
    design = np.column_stack([np.ones(n), t, t*t] + [np.maximum(t-k, 0)**2 for k in knots])
    fit = design @ np.linalg.lstsq(design, s, rcond=None)[0]
    # MC3 refits after replacing large outward deviations (not dropping rows).
    revised = np.where((abs(s-fit)>3) & (abs(s)>abs(fit)), fit, s)
    # The upstream Y refit passes original pfShiftY, not the modified buffer.
    revised[:, 0] = s[:, 0]  # dy is the upstream Y coordinate
    fit = design @ np.linalg.lstsq(design, revised, rcond=None)[0]
    tolerance = np.where(np.arange(n)<50, 3., np.where(np.arange(n)>20, 4., 5.))
    s = np.where(abs(s-fit)>tolerance[:, None], fit, s)
    return s


def fourier_warp(dev: _Dev, image, dy: float, dx: float):
    """Periodic frame alignment: sample at (y+dy,x+dx), without bilinear blur."""
    if dy == 0 and dx == 0:
        return image
    h, w = image.shape
    ky = dev.asarray(np.fft.fftfreq(h).astype(np.float32)[:, None])
    kx = dev.asarray(np.fft.rfftfreq(w).astype(np.float32)[None, :])
    phase = dev.exp(2j*np.pi*(ky*float(dy)+kx*float(dx)))
    return dev.irfft2(dev.rfft2(image)*phase, s=(h,w)).astype(np.float32)


def mc3_sum_warp(dev: _Dev, image, dy: float, dx: float):
    """MC3's in-place sum correction skips native shifts smaller than .1 px.

    This cutoff does not apply to phase shifts inside the MC fitting loop.
    """
    if np.hypot(dy,dx) < .1:
        return image
    return fourier_warp(dev,image,dy,dx)


def motion_correct(frames, frame_doses, prior_doses, *, gpu: bool = True,
                   b_factor: float = 500.0, n_iter: int = 15, tol: float = 0.10,
                   search_radius: int = 64,
                   damage_decay_e_per_A2: float = 40.0,
                   sum_range=None):
    """Align one tilt's frames (MC3 full-frame recipe) and return the
    dose-weighted aligned sum.

    frames: (N, H, W) array (CPU or GPU); frame_doses/prior_doses: (N,)
    e-/A^2, used ONLY for the output sum weights (the alignment reference
    sum is unweighted, matching ``-FmDose 0``).

    ``sum_range``: optional Greco-only (lo, hi) zero-based half-open frame
    range for the output sum, NOT MC3's dose interval. Defaults to all frames.

    Returns dict with:
      aligned: (H, W) float32 dose-weighted aligned mean (input device)
      full_shifts: (N, 2) content displacement in MC3's full-frame gauge
      weights: (N,) output-sum weights
      iterations: int number of alignment iterations actually run
      runtime_s: float
    """
    dev = _Dev(gpu)
    frames = dev.asarray(frames, dtype=np.float32)
    n, h, w = frames.shape
    if h != w:
        raise ValueError("motion correction expects square frames")
    if n < 2:
        raise ValueError("motion correction needs at least 2 frames")
    if sum_range is None:
        lo, hi = 0, n
    else:
        lo, hi = max(0, int(sum_range[0])), min(n, int(sum_range[1]))
        if lo >= hi:
            lo, hi = 0, n

    frame_mean = frames.mean(axis=(1, 2))
    centred = frames - frame_mean[:, None, None]
    weights = np.asarray(frame_doses, float) * np.exp(
        -np.asarray(prior_doses, float) / damage_decay_e_per_A2)

    # MC3 estimates motion on a Fourier-cropped correlation buffer, not FtBin.
    size = h if h <= 2048 else 2048
    scale = h / size
    def spectrum(image):
        f = dev.fft2(image)
        if size == h:
            return f
        indices = dev.asarray(np.concatenate((np.arange(size//2), np.arange(h-size//2,h))))
        return f[indices][:, indices]
    k2 = dev.asarray(_bfactor_grid(size))
    ky = dev.asarray(np.fft.fftfreq(size).astype(np.float32)[:, None])
    kx = dev.asarray(np.fft.fftfreq(size).astype(np.float32)[None, :])
    ramp_y = (2.0 * np.pi) * ky
    ramp_x = (2.0 * np.pi) * kx
    F = [spectrum(c) for c in centred]

    shifts = np.zeros((n, 2), float)
    started = time.monotonic()
    iters = 0
    for it in range(n_iter):
        iters = it + 1
        B = max(b_factor - 2.0 * it, 20.0)
        wfilt = dev.exp(-0.5 * float(B) * k2)
        A = [_phase_shift(dev, F[i], ramp_y, ramp_x, *shifts[i]) for i in range(n)]
        S = A[0]
        for a in A[1:]:
            S = S + a
        max_err = 0.0
        for i in range(n):
            cc = dev.ifft2(wfilt * A[i] * dev.conj(S - A[i])).real
            dy, dx = _windowed_peak(dev, cc, search_radius)
            shifts[i, 0] += dy
            shifts[i, 1] += dx
            max_err = max(max_err, float(np.hypot(dy, dx)))
        # note: MC3 updates the running shifts with the RAW residuals and
        # only smooths the total trajectory after the loop (smoothings in
        # the loop would bias the fixed point away from the true shifts)
        if it > 0 and max_err < tol / scale:
            break

    # MC3 post-processing on the total shift trajectory:
    #   RemoveSpikes -> Smooth(0.5)  (CIterativeAlign)
    #   Smooth(0.4)  = 0.3 + 0.1*binZ, binZ=1   (CFullAlign::DoIt)
    shifts = _smooth_shifts(_smooth_shifts(_remove_spikes(shifts), 0.5), 0.4)
    shifts *= scale
    # CFullAlign scales back to native pixels BEFORE TruncateDecimal.
    shifts = np.trunc(shifts * 100) / 100
    # CFullAlign does NOT MakeRelative: FmRef is only applied by CPatchAlign.
    # Preserve the full-frame gauge used by MC3's FULL sum.

    # dose-weighted aligned mean over the selected frame range
    sum_img = None
    sum_w = 0.0
    for i in range(lo, hi):
        dy, dx = float(shifts[i, 0]), float(shifts[i, 1])
        if dy == 0.0 and dx == 0.0:
            aligned = centred[i]
        else:
            aligned = mc3_sum_warp(dev, centred[i], dy, dx)
        w_i = float(weights[i])
        sum_img = w_i * aligned if sum_img is None else sum_img + w_i * aligned
        sum_w += w_i
    if sum_w <= 0:
        raise ValueError("empty output frame range")
    mean0 = float(dev.asnumpy(frame_mean[lo:hi].mean())
                  if dev.name == "cupy" else frame_mean[lo:hi].mean())
    aligned = (sum_img / sum_w + dev.asarray(np.float32(mean0))).astype(
        np.float32, copy=False)
    return dict(aligned=aligned, full_shifts=shifts, weights=weights,
                iterations=iters, runtime_s=time.monotonic() - started)
