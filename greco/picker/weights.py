# Third-party references and adapted components
# Greco spectral weighting with CPU interpolation compatible with voltools conventions. The Greco implementation uses SciPy; this notice records the interpolation reference.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/voltools/LICENSE.txt
#
# Copyright 2019 Ilja Gubins
# Full license conditions and disclaimers are retained at the paths above.
#

"""Fourier-space weights: band-pass filters, binary and tilt-weighted wedges
with per-tilt CTF (phase-flip) and dose weighting.

Numerical kernels are implemented in spectral.py. This interface preserves saved
CTF metadata, Fourier layout and tilt accumulation conventions. Section rotation
retains the SciPy implementation of the voltools CPU filt_bspline convention.
Historical PyTom/pytom-match-pick provenance remains documented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import (
    radial_grid,
    gaussian_transfer as create_gaussian_band_pass,
    ctf_transfer as create_ctf,
    binary_wedge as _create_binary_wedge,
    weighted_wedge as _create_tilt_weighted_wedge,
    randomize_phases as phase_randomize_template,
)

# Physical constants
CONSTANTS = {
    "c": 299792458,
    "el": 1.60217646e-19,
    "h": 6.62606896e-34,
    "me": 9.10938215e-31,
}


@dataclass(frozen=True)
class CtfData:
    defocus: float  # meters
    amplitude_contrast: float
    voltage: float  # eV
    spherical_aberration: float  # meters
    phase_shift_deg: float = 0.0
    flip_phase: bool = False


@dataclass(frozen=True)
class TiltSeriesMetaData:
    tilt_angles: list  # degrees
    ctf_data: list | None = None  # list[CtfData] length 1 or len(tilts)
    dose_accumulation: list | None = None  # e-/A^2 accumulated per tilt
    per_tilt_weighting: bool = True
    defocus_handedness: int = 0
    level_angle_x: float = 0.0  # degrees
    level_angle_y: float = 0.0  # degrees
    angles_in_degrees: bool = True


def hwhm_to_sigma(hwhm: float) -> float:
    return hwhm / np.sqrt(2 * np.log(2))


def wavelength_ev2m(voltage: float) -> float:
    h, e, m, c = CONSTANTS["h"], CONSTANTS["el"], CONSTANTS["me"], CONSTANTS["c"]
    return h / np.sqrt(e * voltage * m * (e / m * voltage / c**2 + 2))


def create_wedge(shape, ts_metadata: TiltSeriesMetaData, voxel_size: float,
                 cut_off_radius: float = 1.0, low_pass=None, high_pass=None,
                 per_tilt_weighting: bool | None = None):
    """Wedge volume in reduced Fourier form (last dim shape[2]//2+1)."""
    if ts_metadata.angles_in_degrees:
        tilt_angles_rad = np.deg2rad(ts_metadata.tilt_angles)
    else:
        tilt_angles_rad = ts_metadata.tilt_angles

    level_angle_x_rad = np.deg2rad(ts_metadata.level_angle_x)
    level_angle_y_rad = np.deg2rad(ts_metadata.level_angle_y)

    if per_tilt_weighting is None:
        per_tilt_weighting = ts_metadata.per_tilt_weighting

    if per_tilt_weighting:
        wedge = _create_tilt_weighted_wedge(
            shape,
            tilt_angles_rad,
            cut_off_radius,
            voxel_size,
            accumulated_dose_per_tilt=ts_metadata.dose_accumulation,
            ctf_params_per_tilt=ts_metadata.ctf_data,
            level_angle_x=level_angle_x_rad,
            level_angle_y=level_angle_y_rad,
        ).astype(np.float32)
    else:
        alpha_min = min(tilt_angles_rad) + level_angle_y_rad
        alpha_max = max(tilt_angles_rad) + level_angle_y_rad
        wedge = _create_binary_wedge(
            shape,
            alpha_min,
            alpha_max,
            cut_off_radius,
            level_angle_x=level_angle_x_rad,
        ).astype(np.float32)
        if ts_metadata.ctf_data is not None:
            ctf_data = ts_metadata.ctf_data[len(ts_metadata.tilt_angles) // 2]
            wedge *= create_ctf(shape, voxel_size * 1e-10, ctf_data)

    if not (low_pass is None and high_pass is None):
        return wedge * create_gaussian_band_pass(
            shape, voxel_size, low_pass, high_pass
        ).astype(np.float32)
    return wedge


def spherical_mask(size: int, radius: float, center: float):
    """Binary sphere on the supplied voxel-centre grid."""
    coords = np.arange(size) - center
    x = coords[:, np.newaxis, np.newaxis]
    y = coords[np.newaxis, :, np.newaxis]
    z = coords[np.newaxis, np.newaxis, :]
    r = np.sqrt(x**2 + y**2 + z**2)
    return (r <= radius).astype(np.float32)




def create_template_wedge(shape, metadata, voxel_size, *, impulse_spectral_weighting=False):
    """Template-side wedge, with optional impulse-derived radial weighting."""
    base = create_wedge(shape, metadata, voxel_size, cut_off_radius=1.0).astype(np.float32)
    if impulse_spectral_weighting:
        from .impulse_weighting import radial_weighting
        return radial_weighting(base, shape, metadata, voxel_size)
    return base
