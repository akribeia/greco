# Third-party references and adapted components
# AreTomo3 FindCtf adaptations: CTsTiles, CFindCtf2D and CFindCtfMain weak-fit refinement.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/aretomo3/LICENSE.txt
#
# Upstream notice (retained):
# Copyright 2023 Chan Zuckerberg Institute for Advanced Biological Imaging
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
#
#
#
#

"""Native Greco per-tilt CTF estimation from motion-corrected bin1 images.

Pipeline per tilt (float32, GPU):

1. Tiling: 512 px tiles on a 256 px pitch (50% overlap), centred in the
   plane (Greco ``CTsTiles::mSetSize``; for a 4096 px plane: 15x15 = 225).
2. Per tile: z-score normalisation (population std), rounded-edge window
   (super-ellipse, power 3), 2 px zero padding in x, FFT -> AMPLITUDE
   half-spectrum (257x512, DC at (0, 256)); no log.
3. The half-spectra are averaged over all tiles of the plane.
4. Background removal: a 35x35 box of the local mean of sqrt(amplitude)
   (conjugate-mirror wrap at x = 0), squared and subtracted; radii
   r < 0.04 (normalised) are zeroed.
5. Real-space Gaussian low-pass of the mirrored spectrum
   (exp(-10 * r^2 / (Nx^2 + Ny^2)) on the 512x512 spatial image).
6. 1-D stage: radial average of the cleaned spectrum; brute-force defocus
   (501 steps over [2000, 40000] * pix^2 A, min step 100 A) x phase
   (37 steps, min 1 deg; a single value when the range is < 36 deg, as in
   the C++); score = normalised CC of (|CTF|-0.5)*exp(-r^2) against the
   radial spectrum on the band [tile*pix/35.6 A, tile*pix/5.28 A].  A
   second pass refines df over [max(df-0.3df, 2000)...] with the phase
   window narrowed to 0.2 of the range.
7. 2-D stage: parameters (df_mean, astig_ratio, azimuth, ext_phase),
   coordinate ascent (``_refine_param``: scan a range at a fixed step,
   keep the first strict maximum, commit only if the stored score
   improves) interleaved with a (df, phase) grid search (df +/- 3000 A
   step 100, phase step 1), then 2 medium + 5 fine refinement rounds
   (df range 6000/3000/2000, astig ratio +/-0.1/0.05/0.05, azimuth
   +/-180/90/20 step 2, phase +/-range step 1).  The 2-D CTF is
   CTF(r, phi) = -sin(phase + pi*lam*r^2 * (df(phi) - 0.5*lam^2*r^2*Cs))
   with df(phi) = df_mean * (1 + ratio * cos(2 (phi - azimuth))); the
   score is the B-weighted (exp(-r^2)) normalised CC against the cleaned
   half-spectrum on the same resolution band.
8. CTF resolution: per-shell (width 10 px) normalised CC between |CTF|-0.5
   and the spectrum; the shell of maximum CC is extended while the CC
   stays >= 0.143; res = tile*pix/shell.

``refine_series_bad_tilts`` mirrors CFindCtfMain::mRefineTilts (re-estimate
any tilt whose score is >= 30% below the best in its 7-tilt window, with
the reference tilt's astig/azimuth/phase fixed and a df scan of the
reference mean +/- 5000 A at 100 A steps).

Output columns match the Greco table: df_min, df_max, df_mean,
astig_azimuth, phase, score, resolution.
"""

from __future__ import annotations
from greco.project_layout import join as storage_path

import time
from dataclasses import dataclass

import numpy as np

from .motion import _Dev


def electron_wavelength_A(kv: float) -> float:
    """Electron wavelength in Angstrom (Greco CCtfTheory::mCalcWavelength)."""
    return 12.26 / np.sqrt(kv * 1000.0 + 0.9784 * kv * kv)


@dataclass
class TiltCTF:
    """Per-tilt estimate in Greco's native CTF convention.

    The azimuth points along ``df_min_A`` (defocus U), exactly as expected by
    :mod:`greco.ctf_model`. The estimator normalizes its internal result at
    this boundary so preprocessing never carries a foreign angle convention.
    """
    tilt_deg: float
    df_min_A: float
    df_max_A: float
    astig_azimuth_deg: float
    phase_deg: float
    score: float
    resolution_A: float

    @property
    def df_mean_A(self) -> float:
        return 0.5 * (self.df_min_A + self.df_max_A)

    @property
    def astig_magnitude_A(self) -> float:
        return self.df_max_A - self.df_min_A

    @property
    def astig_ratio(self) -> float:
        return (self.df_max_A - self.df_min_A) / (self.df_max_A + self.df_min_A)

    @property
    def phase_rad(self) -> float:
        return float(np.deg2rad(self.phase_deg))

@dataclass(frozen=True)
class TiltDefocusGeometry:
    """Saved image gradient or the tilt-constrained physical defocus model.

    Coordinates are detector pixels relative to the image centre, before any
    raster rotation.  ``handedness`` is the Greco defocus hand (+1/-1).
    Alpha/beta offsets are retained explicitly so the serialized model is
    sufficient to reproduce the per-tile defocus later.
    """

    tilt_deg: float
    tilt_axis_deg: float
    pixel_size_A: float
    handedness: int = 1
    alpha_offset_deg: float = 0.0
    beta_offset_deg: float = 0.0
    gradient_A_per_pixel: tuple | None = None

    def local_mean_defocus_A(self, central_defocus_A, x_px, y_px):
        """Evaluate saved image coefficients or the physical tilt model."""
        if self.gradient_A_per_pixel is not None:
            gradient=np.asarray(self.gradient_A_per_pixel,dtype=float)
            if gradient.shape!=(2,) or not np.isfinite(gradient).all():
                raise ValueError('Invalid saved image defocus gradient.')
            return np.asarray(central_defocus_A)+gradient[0]*np.asarray(x_px)+gradient[1]*np.asarray(y_px)
        axis = np.deg2rad(self.tilt_axis_deg)
        rot_x = np.asarray(x_px) * np.cos(axis) + np.asarray(y_px) * np.sin(axis)
        rot_y = -np.asarray(x_px) * np.sin(axis) + np.asarray(y_px) * np.cos(axis)
        tilt = np.deg2rad(self.tilt_deg)
        alpha = np.deg2rad(self.tilt_deg + self.alpha_offset_deg)
        beta = np.deg2rad(self.beta_offset_deg)
        delta_z_px = -rot_x * np.tan(alpha) + rot_y * np.tan(beta) * np.tan(tilt)
        return (np.asarray(central_defocus_A)
                + self.handedness * delta_z_px * self.pixel_size_A)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation (the Greco GCC formula)."""
    if a.size == 0:
        return 0.0
    ma, mb = float(a.mean()), float(b.mean())
    cov = float(np.mean(a * b) - ma * mb)
    sa = float(np.sqrt(max(np.mean(a * a) - ma * ma, 0.0)))
    sb = float(np.sqrt(max(np.mean(b * b) - mb * mb, 0.0)))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return cov / (sa * sb)


class _CTFEstimator:
    """Per-tilt Greco CTF estimator bound to one device and scope setup."""

    def __init__(self, *, pixel_A: float, voltage_kV: float, cs_mm: float,
                 amp_contrast: float = 0.07, tile_size: int = 512,
                 gpu: bool = True, device: int | None = None,
                 res_low_A=None, res_high_A=None):
        self.dev = _Dev(gpu, device)
        d = self.dev
        self.pix = float(pixel_A)
        self.tile = int(tile_size)
        lam_A = electron_wavelength_A(voltage_kV)
        self.lam = float(lam_A) / self.pix             # wavelength in px
        self.cs = cs_mm * 1e7 / self.pix               # Cs in px
        self.amp_phase = float(np.arctan(amp_contrast
                                         / np.sqrt(1.0 - amp_contrast * amp_contrast)))
        self.cmp_x = self.tile // 2 + 1                # 257
        self.cmp_y = self.tile                         # 512
        # score band (Greco CFindCtfBase::Setup1)
        self.res_lo_A = 18.0*self.pix if res_low_A is None else float(res_low_A)
        self.res_hi_A = max(2.0*self.pix/.75,3.5) if res_high_A is None else float(res_high_A)
        if not (np.isfinite(self.res_lo_A) and np.isfinite(self.res_hi_A)
                and self.res_lo_A>self.res_hi_A>=2*self.pix):
            raise ValueError('CTF fit requires low resolution > high resolution >= Nyquist (2 × pixel size), in Å.')
        # brute-force defocus range (Greco: scaled by pix^2)
        self.df_lo_A = 2000.0 * self.pix * self.pix
        self.df_hi_A = 40000.0 * self.pix * self.pix
        # --- static 2-D grids (comparison grid: x in [0, 0.5], y in [-0.5, 0.5))
        x = np.arange(self.cmp_x, dtype=np.float64) * 0.5 / (self.cmp_x - 1)
        y = (np.arange(self.cmp_y, dtype=np.float64) - self.cmp_y // 2) / self.cmp_y
        # NumPy representation follows the CUDA row-major buffer:
        # (cmp_y, cmp_x), with x being the contiguous half-spectrum axis.
        # The earlier draft accidentally built these grids transposed while
        # the spectra/background/radial code correctly used (512, 257).
        fx, fy = x[None, :], y[:, None]
        f2 = fx * fx + fy * fy
        self.f2 = d.asarray(f2.astype(np.float32))
        self.phi = d.asarray(np.arctan2(fy, fx + 1e-30).astype(np.float32))
        # C++ uses wavelength and defocus in pixels.  ``dfp`` below already
        # converts Angstrom to pixels, so dividing chiA by pixel size again
        # (as the draft did) was a second, erroneous conversion.
        self.chiA = d.asarray((np.pi * self.lam * f2).astype(np.float32))
        # pi*lambda*s^2 * (df - 0.5*lambda^2*s^2*Cs): the spherical-
        # aberration term therefore contains lambda**3.  The old 2-D path
        # accidentally used lambda**2 (the 1-D path was already correct),
        # which moved the 2-D optimum by roughly 2--3 kA on yeast.
        self.chiB = d.asarray((0.5 * np.pi * self.lam**3 * self.cs * f2 * f2)
                              .astype(np.float32))
        # 2-D score band (normalised r^2; Greco GCC2D::Setup2/DoIt)
        fRes1 = self.cmp_y * self.pix
        lo2 = (fRes1 / self.res_lo_A / self.cmp_y) ** 2
        hi2 = min(fRes1 / self.res_hi_A / self.cmp_y, 0.75) ** 2
        band = (f2 >= lo2) & (f2 <= hi2)
        self.idx2 = d.asarray(np.nonzero(band.ravel())[0].astype(np.int64))
        self.b2w = d.asarray(np.exp(-f2[band]).astype(np.float32))
        # --- static 1-D grid (radial index in px)
        r = np.arange(self.cmp_x, dtype=np.float64) * 0.5 / (self.cmp_x - 1)
        fRes1d = (self.cmp_x - 1) * 2 * self.pix
        b1_lo = fRes1d / self.res_lo_A                  # ~28.4 px
        b1_hi = fRes1d / self.res_hi_A                  # ~192 px
        m1 = (np.arange(self.cmp_x) >= b1_lo) & (np.arange(self.cmp_x) < b1_hi)
        self.b1_idx = np.nonzero(m1)[0]
        self.b1w = np.exp(-r[m1] ** 2).astype(np.float32)
        if not self.b1_idx.size or not np.any(band):
            raise ValueError('CTF fit resolution range contains no sampled frequencies.')
        # --- background-removal radius mask (r < 0.04 -> zero)
        X, Y = np.meshgrid(np.arange(self.cmp_x), np.arange(self.cmp_y), indexing="xy")
        r2 = (X / self.cmp_y) ** 2 + ((Y - self.cmp_y // 2) / self.cmp_y) ** 2
        self.bg_keep = d.asarray((r2 >= 0.04 ** 2).astype(bool))
        # --- radial-average sampling (Greco GRadialAvg)
        ra_x = np.zeros((self.cmp_x, 180), dtype=np.int64)
        ra_y = np.zeros((self.cmp_x, 180), dtype=np.int64)
        self.ra_n = np.zeros(self.cmp_x, dtype=np.int64)
        for ri in range(self.cmp_x):
            npts = int(min(max(int(np.floor(np.pi * ri + 0.5)), 1), 180))
            ang = np.deg2rad(-90.0 + np.arange(npts) * 180.0 / npts)
            # note: at r = 256 the last sample rounds to y = 512 (Greco
            # reads one row past the spectrum there; that sample is outside
            # the 1-D score band, so clipping is result-equivalent)
            ra_x[ri, :npts] = np.clip(np.floor(ri * np.cos(ang) + 0.5), 0, self.cmp_x - 1)
            ra_y[ri, :npts] = np.clip(
                np.floor(ri * np.sin(ang) + self.cmp_y // 2 + 0.5), 0, self.cmp_y - 1)
            self.ra_n[ri] = npts
        self.ra_x_dev = d.asarray(ra_x)
        self.ra_y_dev = d.asarray(ra_y)
        # --- CTF-resolution shells (Greco GSpectralCC2D, width 10)
        rr = np.sqrt(X.astype(float) ** 2 + (Y - self.cmp_y // 2) ** 2)
        rr_flat = rr.ravel().astype(np.float32)
        order = np.argsort(rr_flat, kind="stable").astype(np.int64)
        rr_sorted = rr_flat[order]
        self.shell_order = d.asarray(order)
        self.shell_masks = []
        self.shell_s = None
        self.shell_stat = None
        # GSpectralCC2D compares against iSpectX (the R2C width, 257 for a
        # 512-pixel tile), not the largest valid x index.  The upper radius
        # bound is exclusive, so the terminal window is [247, 257).
        spect_x = self.cmp_x
        ab = np.zeros((self.cmp_x, 2), dtype=np.int64)
        for i in range(1, self.cmp_x):
            iLow, iHigh = i - 5, i + 5
            if iLow < 0:
                iLow, iHigh = 0, 10
            elif iHigh > spect_x:
                iLow, iHigh = spect_x - 10, spect_x
            ab[i] = (np.searchsorted(rr_sorted, iLow, side="left"),
                     np.searchsorted(rr_sorted, iHigh, side="left"))
        self.shell_ab = ab
        # rounded-edge tile window (Greco GRoundEdge, centre/size = tile)
        c = self.tile * 0.5
        ax = 2.0 * np.abs(np.arange(self.tile) - c) / self.tile
        rrw = np.sqrt(ax[None, :] ** 2 + ax[:, None] ** 2)
        w = np.where(rrw < 1.0, 1.0 - (0.5 * (1.0 - np.cos(np.pi * rrw))) ** 3, 0.0)
        self.tile_win = d.asarray(w.astype(np.float32))
        # tile layout (Greco CTsTiles::mSetSize)
        self._ov = int(self.tile * 0.5) // 2 * 2

    # ------------------------------------------------------------------
    # spectrum generation
    # ------------------------------------------------------------------
    def half_spect(self, plane, retain_tiles=False,valid_bounds_yx=None):
        """Averaged amplitude half-spectrum (cmp_y, cmp_x), DC at (cmp_y/2, 0)."""
        d = self.dev
        n = plane.shape[0]
        ts = self.tile
        ov = self._ov
        pitch = ts - ov
        ntx = (n - ov) // pitch
        nty = (n - ov) // pitch
        if ntx <= 0 or nty <= 0:
            raise ValueError(f"plane {n}px too small for {ts}px CTF tiles")
        ox = (n - ntx * ts + (ntx - 1) * ov) // 2
        oy = (n - nty * ts + (nty - 1) * ov) // 2
        acc = None;count=0
        bounds=valid_bounds_yx if valid_bounds_yx is not None else (0,n,0,n)
        if d.name == 'cupy':
            from .ctf_gpu import half_spectrum
            starts = [(oy+ty*pitch, ox+tx*pitch) for ty in range(nty) for tx in range(ntx)
                      if oy+ty*pitch >= bounds[0] and oy+ty*pitch+ts <= bounds[1]
                      and ox+tx*pitch >= bounds[2] and ox+tx*pitch+ts <= bounds[3]]
            if not starts:
                raise ValueError('No complete CTF tiles inside the recorded movie pixels.')
            return half_spectrum(self, plane, starts, retain_tiles)
        saved, centers, stds = [], [], []
        for ty in range(nty):
            for tx in range(ntx):
                y0=oy+ty*pitch;x0=ox+tx*pitch
                if y0<bounds[0] or y0+ts>bounds[1] or x0<bounds[2] or x0+ts>bounds[3]:continue
                count+=1
                t = plane[oy + ty * pitch: oy + ty * pitch + ts,
                          ox + tx * pitch: ox + tx * pitch + ts]
                t = d.asarray(t, dtype=np.float32)
                m = float(d.asnumpy(t.mean()))
                m2 = float(d.asnumpy((t * t).mean()))
                s = float(np.sqrt(max(m2 - m * m, 0.0)))
                t = (t - d.asarray(np.float32(m))) / d.asarray(
                    np.float32(s if s > 0 else 1.0))
                t = t * self.tile_win
                # Greco stores an in-place R2C transform in a real buffer
                # whose *stride* is tile+2.  The two padding floats are not
                # image columns.  The discarded draft incorrectly ran a
                # complex FFT over a 512x514 image, changing every frequency.
                half = d.roll(d.abs(d.rfft2(t)), ts // 2, axis=0)
                if retain_tiles:
                    saved.append(d.asnumpy(half).astype(np.float32))
                    centers.append((ox+tx*pitch+(ts-1)/2-(n-1)/2,
                                    oy+ty*pitch+(ts-1)/2-(n-1)/2))
                    stds.append(s)
                acc = half if acc is None else acc + half
        if count==0:raise ValueError('No complete CTF tiles inside the recorded movie pixels.')
        if retain_tiles:
            self.retained_tiles=(np.stack(saved),np.asarray(centers),np.asarray(stds))
        return acc / count

    def clean(self, half):
        """Background removal + real-space Gaussian low-pass (Greco
        CFindCtfBase::mRemoveBackground + mLowpass)."""
        d = self.dev
        # --- background: 33x33 box mean of sqrt(amp), squared, subtracted.
        # Offset (j, k) on the half-spectrum (x0 in [0, 256], y0 in [0, 512)):
        #   xx = x0 + j  (>= 256 wraps to xx - 512)
        #   xx >= 0 -> (xx, y0 + k)   (y0 + k mod 512)
        #   xx < 0  -> (-xx, -(y0 + k))  (conjugate mirror, Greco mGRemove)
        # fMinFreq=1/15, so Greco computes int(512/15)=34 and rounds it
        # to the next odd size: a 35x35 box (half-width 17).
        m = 17
        iBox = 2 * m + 1
        nx = self.cmp_x
        A = d.sqrt(d.maximum(half, 0.0))
        if d.name == 'cupy' and A.dtype == np.float32:
            from .ctf_gpu import background_sum
            acc = background_sum(A, m)
            bg = (acc / (iBox * iBox)) ** 2
            return self._lowpass(d.where(self.bg_keep, half - bg, 0.0))
        negA = d.roll(d.flip(A, axis=0), 1, axis=0)
        # negA[y, x] = A[(-y) mod 512, x]
        acc = d.zeros_like(half)
        for k in range(-m, m + 1):
            ry = d.roll(A, -k, axis=0)                # ry[y, x]  = A[y+k, x]
            rny = d.roll(negA, -k, axis=0)            # rny[y, x] = A[(-y-k) mod 512, x]
            for j in range(-m, m + 1):
                if j > 0:
                    # Greco wraps at iHalfX=256, not at the allocated
                    # half-spectrum width nx=257.  Thus x+j==256 already
                    # uses the conjugate y and the wrapped x indices run
                    # 256,255,...,256-j (j+1 samples).
                    cut = (nx - 1) - j
                    acc[:, :cut] = acc[:, :cut] + ry[:, j:j + cut]
                    acc[:, cut:] = acc[:, cut:] + d.flip(rny[:, cut:nx], axis=1)
                elif j == 0:
                    acc[:, :nx - 1] = acc[:, :nx - 1] + ry[:, :nx - 1]
                    # Greco wraps the Nyquist column (x=256 -> -256, y-flip)
                    acc[:, nx - 1] = acc[:, nx - 1] + rny[:, nx - 1]
                else:
                    mb = -j
                    acc[:, mb:] = acc[:, mb:] + ry[:, :nx - mb]
                    # x0 in [0, mb): mirror samples (mb - x0, (-y-k) mod 512)
                    seg = d.flip(rny[:, 1:mb + 1], axis=1)
                    acc[:, :mb] = acc[:, :mb] + seg
        bg = (acc / (iBox * iBox)) ** 2
        out = d.where(self.bg_keep, half - bg, 0.0)
        # --- real-space Gaussian low-pass
        return self._lowpass(out)

    def _lowpass(self, half, b: float = 10.0):
        d = self.dev
        n = self.cmp_y
        hx = self.cmp_x - 1
        # GCalcSpectrum::GenFullSpect: make a real 512x512 spectrum image
        # with DC at (256,256), mirroring the negative-x half.
        full = d.zeros((n, n), dtype=np.float32)
        full[:, hx:] = half[:, :hx]
        yp = d.asarray((-np.arange(n)) % n)
        full[:, :hx] = half[yp, hx:0:-1]
        # CFindCtfBase::mLowpass forwards that real image directly, applies
        # GFFTUtil2D::Lowpass in unshifted R2C coordinates, then inverses it.
        comp = d.rfft2(full)
        yy = np.arange(n, dtype=np.float32)
        yy[yy > n // 2] -= n
        xx = np.arange(self.cmp_x, dtype=np.float32)
        filt = np.exp(-b * (yy[:, None] ** 2 + xx[None, :] ** 2)
                      / float(n * n + n * n)).astype(np.float32)
        smooth = d.irfft2(comp * d.asarray(filt), s=(n, n)).real
        out = d.zeros((n, self.cmp_x), dtype=np.float32)
        out[:, :hx] = smooth[:, hx:]
        # The C++ copies one padded value for the terminal Nyquist column;
        # it lies outside every scoring band.  Keep it deterministically 0.
        return out

    # ------------------------------------------------------------------
    # 1-D stage
    # ------------------------------------------------------------------
    def radial_average(self, cleans):
        d = self.dev
        A = cleans
        out = np.empty(self.cmp_x, dtype=np.float64)
        for ri in range(self.cmp_x):
            npts = int(self.ra_n[ri])
            vals = A[self.ra_y_dev[ri, :npts], self.ra_x_dev[ri, :npts]]
            out[ri] = float(d.asnumpy(vals.mean()))
        return out


    def _bf1d(self, df_lo: float, df_hi: float, ph_lo: float, ph_hi: float,
              radial: np.ndarray):
        """Greco CFindDefocus1D::mBrutalForceSearch."""
        rng = df_hi - df_lo
        step = max(rng / 500.0, 100.0)
        n_df = int(rng / step) // 2 * 2 + 1
        df_grid = df_lo + step * np.arange(n_df)
        prng = ph_hi - ph_lo
        if prng / 36.0 < 1.0:
            ph_grid = np.array([ph_lo])
        else:
            pstep = prng / 36.0
            n_ph = int(prng / pstep) // 2 * 2 + 1
            ph_grid = ph_lo + pstep * np.arange(n_ph)
        r2 = (np.arange(self.cmp_x, dtype=np.float64) * 0.5 / (self.cmp_x - 1)) ** 2
        best = (-1e20, 0.0, 0.0)
        for ph in ph_grid:
            chi = (self.amp_phase + np.deg2rad(ph)
                   + np.pi * self.lam * r2[None, :]
                   * (df_grid[:, None] / self.pix
                      - 0.5 * self.lam * self.lam * r2[None, :] * self.cs))
            C = -np.sin(chi)
            c_all = (np.abs(C[:, self.b1_idx]) - 0.5) * self.b1w[None, :]
            s = radial[self.b1_idx]
            cc = np.array([_pearson(c_all[i], s) for i in range(n_df)])
            i = int(np.argmax(cc))
            if cc[i] > best[0]:
                best = (float(cc[i]), float(df_grid[i]), float(ph))
        return best  # (cc, df_A, phase_deg)

    def do_1d(self, cleans):
        """CFindCtf1D::Do1D: brute-force + refine. Returns (df_A, phase_deg)."""
        radial = self.radial_average(cleans)
        _, df, ph = self._bf1d(self.df_lo_A, self.df_hi_A,
                               self._ph_lo, self._ph_hi, radial)
        # CFindCtf1D::mRefineDefocus: df range max(0.3*df, 2000), floored at
        # 1000*pix^2 (window placed at [max(df-0.5R, floor), +R])
        fDfRange = max(0.3 * df, 2000.0)
        lo = max(df - 0.5 * fDfRange, 1000.0 * self.pix * self.pix)
        hi = lo + fDfRange
        pw = 0.2 * (self._ph_hi - self._ph_lo)
        plo = max(ph - pw / 2.0, self._ph_lo)
        phi = min(ph + pw / 2.0, self._ph_hi)
        _, df, ph = self._bf1d(lo, hi, plo, phi, radial)
        return df, ph

    # ------------------------------------------------------------------
    # 2-D stage
    # ------------------------------------------------------------------
    def _ctf2d(self, p: np.ndarray):
        """One 2-D CTF surface (cmp_x, cmp_y) for params p
        [df_mean_A, astig_ratio, azimuth_deg, phase_deg, score, res]."""
        d = self.dev
        c2 = d.cos(2.0 * (self.phi - d.asarray(np.float32(np.deg2rad(float(p[2]))))))
        dfp = float(p[0]) * (1.0 + float(p[1]) * c2) / self.pix
        chi = (d.asarray(np.float32(self.amp_phase + np.deg2rad(float(p[3]))))
               + self.chiA * dfp - self.chiB)
        return -d.sin(chi)

    def _score_rows(self, Cb, sb):
        """Row-wise normalised CC of (|C|-0.5)*exp(-r^2) rows vs sb
        (device arrays; Cb (k, nb), sb (nb,))."""
        d = self.dev
        cm = Cb.mean(axis=1)
        sm = float(d.asnumpy(sb.mean()))
        cs = (Cb * sb[None]).mean(axis=1)
        c2m = (Cb * Cb).mean(axis=1)
        s2m = float(d.asnumpy((sb * sb).mean()))
        cov = cs - cm * sm
        vc = d.maximum(c2m - cm * cm, 0.0)
        vs = max(s2m - sm * sm, 0.0)
        denom = d.sqrt(vc) * d.asarray(np.float32(np.sqrt(vs) if vs > 0 else 0.0))
        cc = d.where(denom > 0.0, cov / d.maximum(denom, 1e-30), 0.0)
        return d.asnumpy(cc)

    def _cc2d(self, p: np.ndarray, sb) -> float:
        d = self.dev
        ctf = self._ctf2d(p)
        Cb = (d.abs(ctf).reshape(-1)[self.idx2] - d.asarray(np.float32(0.5))) * self.b2w
        return float(self._score_rows(Cb[None], sb)[0])

    def _cc2d_many(self, values: np.ndarray, iParam: int, p: np.ndarray, sb):
        """Scores for a batch of values of parameter iParam (ascending)."""
        d = self.dev
        k = len(values)
        out = np.empty(k, dtype=np.float32)
        chunk = 48
        # For scans of df/ratio/azimuth the current external phase remains
        # fixed.  Only a phase-coordinate scan replaces p[3] with candidates.
        # This is invisible for conventional zero-phase data but essential
        # for phase-plate series.
        ph_ext = d.asarray(np.float32(
            self.amp_phase + np.deg2rad(float(p[3]))))
        azr0 = d.asarray(np.float32(np.deg2rad(float(p[2]))))
        for a in range(0, k, chunk):
            vals = d.asarray(values[a:a + chunk].astype(np.float32))
            kc = vals.shape[0]
            if iParam == 2:  # azimuth
                azr = d.asarray(np.deg2rad(values[a:a + chunk].astype(np.float64))
                                .astype(np.float32)).reshape(kc, 1, 1)
                c2 = d.cos(2.0 * (self.phi[None, :, :] - azr))
                dfp = float(p[0]) * (1.0 + float(p[1]) * c2) / self.pix
                chi = ph_ext + self.chiA[None] * dfp - self.chiB[None]
            elif iParam == 0:  # df_mean
                c2 = d.cos(2.0 * (self.phi - azr0))
                dfb = (vals / self.pix).reshape(kc, 1, 1)
                dfp = dfb * (1.0 + float(p[1]) * c2)
                chi = ph_ext + self.chiA[None] * dfp - self.chiB[None]
            elif iParam == 1:  # astig ratio
                c2 = d.cos(2.0 * (self.phi - azr0))
                dfp = (float(p[0]) * (1.0 + vals.reshape(kc, 1, 1) * c2) / self.pix)
                chi = ph_ext + self.chiA[None] * dfp - self.chiB[None]
            else:  # phase
                c2 = d.cos(2.0 * (self.phi - azr0))
                ph = d.asarray(np.deg2rad(values[a:a + chunk].astype(np.float64))
                               .astype(np.float32)).reshape(kc, 1, 1)
                dfp = float(p[0]) * (1.0 + float(p[1]) * c2) / self.pix
                chi = (d.asarray(np.float32(self.amp_phase)) + ph
                       + self.chiA[None] * dfp[None] - self.chiB[None])
            C = -d.sin(chi)
            Cb = (d.abs(C.reshape(kc, -1)[:, self.idx2])
                  - d.asarray(np.float32(0.5))) * self.b2w[None]
            out[a:a + kc] = self._score_rows(Cb, sb)
        return out

    def _refine_param(self, p: np.ndarray, sb, lo: float, hi: float,
                      step: float, iParam: int):
        """Greco CFindDefocus2D::RefineParam (coordinate ascent)."""
        rng = hi - lo
        if rng == 0.0 or step <= 0:
            return
        iNumSteps = int(rng / step + 0.5) // 2 * 2 + 1
        cent = iNumSteps // 2
        # C++ snapshots the complete parameter vector and restores it when
        # this coordinate scan does not improve the stored score.  Merely
        # leaving ``p`` alone here is not enough: scoring the candidates
        # mutates p[iParam], so without the restore the final (usually upper
        # bound) candidate leaks into later refinement rounds.
        old = p.copy()
        init = float(p[iParam])
        cands = init + step * (np.arange(iNumSteps) - cent)
        cands = cands[(cands >= lo) & (cands <= hi)]
        best_cc = self._cc2d(p, sb)
        best_val = init
        if cands.size:
            ccs = self._cc2d_many(cands, iParam, p, sb)
            for v, cc in zip(cands, ccs):
                if cc > best_cc:
                    best_cc, best_val = float(cc), float(v)
        if best_cc > p[4]:
            p[iParam] = best_val
            p[4] = best_cc
            if p[3] > 180.0:
                p[3] -= 180.0
        else:
            p[:] = old

    def _grid_search(self, p: np.ndarray, sb, df_lo: float, df_hi: float,
                     ph_lo: float, ph_hi: float):
        """Greco CFindDefocus2D::mGridSearch (df x phase, 100 A / 1 deg)."""
        old = p.copy()
        fvals = df_lo + 100.0 * np.arange(int(np.floor((df_hi - df_lo) / 100.0)) + 1)
        fvals = fvals[fvals <= df_hi + 1e-6]
        pvals = ph_lo + 1.0 * np.arange(int(np.floor((ph_hi - ph_lo) / 1.0)) + 1)
        pvals = pvals[pvals <= ph_hi + 1e-6]
        best_cc, best_df, best_ph = -1e20, 0.0, ph_lo
        for ph in pvals:
            p[3] = float(ph)
            ccs = self._cc2d_many(fvals, 0, p, sb)
            i = int(np.argmax(ccs))
            if ccs[i] > best_cc:
                best_cc, best_df, best_ph = float(ccs[i]), float(fvals[i]), float(ph)
        p[0], p[3], p[4] = best_df, best_ph, best_cc
        if p[4] < old[4]:
            p[:] = old

    def _ctf_res(self, p: np.ndarray, cleans) -> float:
        """Greco GSpectralCC2D: per-shell CC (width 10 px), extend the
        maximum while the CC stays >= 0.143; res = tile*pix/shell."""
        d = self.dev
        ctf = self._ctf2d(p)
        c = (d.abs(ctf).reshape(-1)[self.shell_order] - d.asarray(np.float32(0.5)))
        s = cleans.reshape(-1)[self.shell_order]
        cc = d.zeros((self.cmp_x,), dtype=np.float32)
        for i in range(1, self.cmp_x):
            a, b = int(self.shell_ab[i, 0]), int(self.shell_ab[i, 1])
            if b - a < 8:
                continue
            cm = c[a:b].mean()
            sm = s[a:b].mean()
            cs = (c[a:b] * s[a:b]).mean()
            c2m = (c[a:b] * c[a:b]).mean()
            s2m = (s[a:b] * s[a:b]).mean()
            cov = cs - cm * sm
            vc = d.maximum(c2m - cm * cm, 0.0)
            vs = d.maximum(s2m - sm * sm, 0.0)
            denom = d.sqrt(vc) * d.sqrt(vs)
            cc[i] = d.where(denom > 0.0, cov / d.maximum(denom, 1e-30), 0.0)
        cc = d.asnumpy(cc)
        iMax = 1 + int(np.argmax(cc[1:]))
        if cc[iMax] < 0.143:
            iShell = iMax
        else:
            iShell = iMax
            for i in range(iMax, self.cmp_x):
                if cc[i] < 0.143:
                    break
                iShell = i
        return self.cmp_y * self.pix / max(iShell, 1)

    # ------------------------------------------------------------------
    # per-tilt driver
    # ------------------------------------------------------------------
    def do_tilt(self, cleans, tilt_deg: float, ph_range=(0.0, 0.0)) -> TiltCTF:
        """CFindCtf2D::Do2D on a cleaned half-spectrum."""
        self._ph_lo, self._ph_hi = float(ph_range[0]), float(ph_range[1])
        d = self.dev
        df1, ph1 = self.do_1d(cleans)
        sb = cleans.reshape(-1)[self.idx2]
        p = np.array([df1, 0.01, 0.0, ph1, -1e20, 1e20])
        # --- DoIt
        self._refine_param(p, sb, 0.0, 180.0, 1.0, 2)
        self._refine_param(p, sb, 0.0, 0.5, 0.005, 1)
        p[5] = self._ctf_res(p, cleans)
        df_lo = max(p[0] - 3000.0, 2000.0)
        self._grid_search(p, sb, df_lo, p[0] + 3000.0, self._ph_lo, self._ph_hi)
        self._refine_param(p, sb, 0.0, 0.5, 0.005, 1)
        self._refine_param(p, sb, 0.0, 180.0, 1.0, 2)
        if self._ph_hi - self._ph_lo > 0.5:
            self._refine_param(p, sb, 0.0, 180.0, 1.0, 3)
        p[5] = self._ctf_res(p, cleans)
        # --- 2 medium + 5 fine rounds (CFindCtf2D::Do2D)
        fPhaseRange = self._ph_hi - self._ph_lo
        for i in range(2):
            f = 1.0 / (1 + i)
            self._mrefine(p, sb, cleans, 6000.0 * f, 0.1 * f, 180.0 * f, fPhaseRange * f)
        for _ in range(5):
            fPhaseRange = min(fPhaseRange, 20.0)
            self._mrefine(p, sb, cleans, 2000.0, 0.05, 20.0, fPhaseRange)
        return TiltCTF(
            tilt_deg=float(tilt_deg),
            df_min_A=float(p[0] * (1.0 - p[1])),
            df_max_A=float(p[0] * (1.0 + p[1])),
            astig_azimuth_deg=float(p[2] + 90.0),
            phase_deg=float(p[3]),
            score=float(p[4]),
            resolution_A=float(p[5]),
        )

    def _mrefine(self, p, sb, cleans, fDfRange, fAstMag, fAstAng, fPhaseRange):
        """CFindCtf2D::mRefine (df -> azimuth -> astig ratio -> phase)."""
        fMinDf = max(p[0] - 0.5 * fDfRange, 2000.0)
        fMaxDf = p[0] + 0.5 * fDfRange
        fStepDf = (fMaxDf - fMinDf) / 100.0
        self._refine_param(p, sb, fMinDf, fMaxDf, fStepDf, 0)
        self._refine_param(p, sb, p[2] - 0.5 * fAstAng, p[2] + 0.5 * fAstAng, 2.0, 2)
        fMinRatio = max(p[1] - 0.5 * fAstMag, 0.0)
        fMaxRatio = p[1] + 0.5 * fAstMag
        self._refine_param(p, sb, fMinRatio, fMaxRatio, 0.001, 1)
        if fPhaseRange <= 0.5:
            return
        self._refine_param(p, sb,
                           max(p[3] - fPhaseRange, self._ph_lo),
                           min(p[3] + fPhaseRange, self._ph_hi), 1.0, 3)

    def refine_from_reference(self, cleans, ref: TiltCTF,
                              tilt_deg: float) -> TiltCTF:
        """CFindCtfMain::mRefineTilt: df-only scan around the reference tilt."""
        d = self.dev
        pix2 = self.pix * self.pix
        dfc = ref.df_mean_A
        lo = max(dfc - 5000.0, 2000.0 * pix2)
        hi = min(dfc + 5000.0, 40000.0 * pix2)
        center = 0.5 * (lo + hi)
        rng = hi - lo
        sb = cleans.reshape(-1)[self.idx2]
        p = np.array([center, ref.astig_ratio, ref.astig_azimuth_deg - 90.0, ref.phase_deg,
                      -1e20, 1e20])
        fMinDf = max(center - 0.5 * rng, 1000.0)
        fMaxDf = center + 0.5 * rng
        self._refine_param(p, sb, fMinDf, fMaxDf, 100.0, 0)
        p[5] = self._ctf_res(p, cleans)
        return TiltCTF(
            tilt_deg=float(tilt_deg),
            df_min_A=float(p[0] * (1.0 - p[1])),
            df_max_A=float(p[0] * (1.0 + p[1])),
            astig_azimuth_deg=float(p[2] + 90.0),
            phase_deg=float(p[3]),
            score=float(p[4]),
            resolution_A=float(p[5]),
        )


def estimate_plane_ctf(plane, *, pixel_A, voltage_kV, cs_mm,
                               amp_contrast: float = 0.07,
                               ext_phase_deg: float = 0.0,
                               ext_phase_range_deg: float = 0.0,
                               tilt_deg: float = 0.0, tile_size: int = 512,
                               gpu: bool = True, device: int | None = None,
                               tile_output=None,res_low_A=None,res_high_A=None,valid_bounds_yx=None):
    """Estimate one full-resolution plane's CTF.

    plane: (N, N) square motion-corrected bin1 image (CPU or GPU array).
    Returns (TiltCTF, cleaned_half_spect, runtime_s) where the
    spectrum is a CPU (tile, tile//2+1) float32 array (DC at (tile//2, 0))
    that can be persisted for the series refinement pass.

    A per-plane call does not benefit from the zero-tilt phase narrowing used
    by the series driver.
    """
    t0 = time.monotonic()
    est = _CTFEstimator(pixel_A=pixel_A, voltage_kV=voltage_kV, cs_mm=cs_mm,
                            amp_contrast=amp_contrast, tile_size=tile_size,
                            gpu=gpu, device=device,res_low_A=res_low_A,res_high_A=res_high_A)
    half = est.half_spect(plane, retain_tiles=tile_output is not None,valid_bounds_yx=valid_bounds_yx)
    if tile_output is not None:
        from pathlib import Path
        folder=Path(tile_output);folder.mkdir(parents=True,exist_ok=True)
        tiles,centers,stds=est.retained_tiles
        np.save(storage_path(folder, 'ctf_tiles.npy'),tiles)
        np.savez(storage_path(folder, 'ctf_tiles_metadata.npz'),centers=centers,stds=stds)
    cleans = est.clean(half)
    ph_r = (max(ext_phase_deg - ext_phase_range_deg / 2.0, 0.0),
            min(ext_phase_deg + ext_phase_range_deg / 2.0, 180.0))
    res = est.do_tilt(cleans, float(tilt_deg), ph_r)
    return res, est.dev.asnumpy(cleans), time.monotonic() - t0


def refine_series_bad_tilts(spectra, results, *, pixel_A, voltage_kV, cs_mm,
                            amp_contrast: float = 0.07, tile_size: int = 512,
                            gpu: bool = True, device: int | None = None,
                            angles_deg=None,res_low_A=None,res_high_A=None):
    """CFindCtfMain::mRefineTilts: for each tilt in physical-angle order, compare its score with the
    best in its 7-tilt window; if the deficit is >= 30% of the best,
    re-estimate from the best neighbour (df scan of the reference mean
    +/- 5000 A at 100 A steps; astig/azimuth/phase fixed to the reference).

    Returns (new_results, changed_indices).
    """
    est = _CTFEstimator(pixel_A=pixel_A, voltage_kV=voltage_kV, cs_mm=cs_mm,
                            amp_contrast=amp_contrast, tile_size=tile_size,
                            gpu=gpu, device=device,res_low_A=res_low_A,res_high_A=res_high_A)
    angles = (np.asarray(angles_deg, float) if angles_deg is not None
              else np.array([r.tilt_deg for r in results]))
    n = len(results)
    if n < 7:
        return list(results), []
    scores = np.array([r.score for r in results], float)
    new_results = list(results)
    changed = []
    win = 7
    order=np.argsort(angles,kind="stable")
    for position,i in enumerate(order):
        iStart = max(position - win // 2, 0)
        iEnd = min(iStart + win, n)
        iStart = iEnd - win
        neighbours=order[iStart:iEnd]
        iBest = int(neighbours[np.argmax(scores[neighbours])])
        best = scores[iBest]
        diff = (best - scores[i]) / (best + 1e-30)
        if diff < 0.30:
            continue
        new_results[i] = est.refine_from_reference(est.dev.asarray(spectra[i]),
                                                   results[iBest], float(angles[i]))
        scores[i] = new_results[i].score
        changed.append(i)
    return new_results, changed
