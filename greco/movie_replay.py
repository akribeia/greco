"""Saved movie replay helpers for geometry jobs, with lightweight imports.

These preprocessing operations retain the saved Greco motion correction conventions.
No motion or CTF estimation is performed here.
"""
from greco.project_layout import join as storage_path
from pathlib import Path
import numpy as np
from .mdoc import parse_mdoc
from .movie_io import load_gain as _load_gain, load_movie, prepare_frames as _prepare_frames
from .motion import _Dev
from .ctf_model import CTFParameters,ctf_transfer,regularized_ctf_inverse
from .ctf_spatial import correct_tilt_ctf_spatial


def _rebuild_aligned_sum(dev: _Dev, frames, full_shifts, parity=None,
                         frame_range=None):
    """Rebuild the plain aligned sum (MC3 scale, i.e. the sum of the full
    intensities, mean ~ N * 0.334 on kai) of the frames warped by their full
    shifts (content displacement in the MC3 full-frame gauge). Every frame,
    including the central one, is Fourier-shifted when its shift is nonzero.
    ``parity`` uses the MC3 convention: 'even' = the
    1-based even frames = i % 2 == 1 here (0-based)."""
    n_frames = frames.shape[0]
    lo, hi = frame_range if frame_range is not None else (0, n_frames)
    acc = None
    h, w = frames.shape[-2:]
    ky = dev.asarray(np.fft.fftfreq(h).astype(np.float32)[:, None])
    kx = dev.asarray(np.fft.rfftfreq(w).astype(np.float32)[None, :])
    for i in range(lo, hi):
        if parity is not None and (i % 2) != parity:
            continue
        dy, dx = float(full_shifts[i][0]), float(full_shifts[i][1])
        spectrum = dev.rfft2(frames[i])
        if np.hypot(dy, dx) >= .1:
            spectrum = spectrum * dev.exp(2j*np.pi*(ky*dy+kx*dx))
        acc = spectrum if acc is None else acc + spectrum
    if acc is None:
        raise ValueError("no frames in subset")
    # CCorrectFullShift accumulates spectra, then performs ONE inverse FFT.
    # MC3 GPositivity2D clamps the FINAL native sum, not individual frames.
    # Fractional Fourier shifts can create negative ringing even from counts.
    return dev.maximum(dev.irfft2(acc, s=(h,w)),0).astype(np.float32)


def _movie_mdoc(args):
    mdoc = parse_mdoc(args.mdoc)
    scale = getattr(args,'eer_upscaling',1)
    if scale==2 and any(Path(tilt.movie).suffix.lower()=='.eer' for tilt in mdoc.tilts):
        if not all(Path(tilt.movie).suffix.lower()=='.eer' for tilt in mdoc.tilts):
            raise ValueError('Cannot mix TIFF and upscaled EER in one stack.')
        mdoc.pixel_size_A /= scale
        mdoc.image_size = tuple(size*scale for size in mdoc.image_size)
    return mdoc


def _load_detector_correction(tdir):
    path = storage_path(tdir, 'detector_correction.npz')
    if not path.exists():
        raise ValueError('Missing saved detector correction. Rerun preprocessing stage A; '
                         'do not mix old MC shifts with newly corrected frames.')
    with np.load(path, allow_pickle=False) as handle:
        correction = {key:handle[key].copy() for key in handle.files}
    if str(correction.get('algorithm','')) != 'mc3_detector_v1':
        raise ValueError('Unsupported detector correction; rerun preprocessing stage A.')
    return correction
