"""CTF observation operator for preprocessing and projection matching.

This module does not reconstruct or phase-flip data.  It only constructs the
real Fourier transfer function for one projection and applies it to a model
projection.  The sign convention is recorded here so that a future 3-D
consumer cannot silently inherit a microscope-package convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fftshift, ifftshift


FloatImage = NDArray[np.float32]
FloatSpectrum = NDArray[np.float64]
ComplexSpectrum = NDArray[np.complex128]


def _symmetric_shifted(array: np.ndarray) -> np.ndarray:
    """Project a shifted real Fourier map onto discrete Hermitian symmetry."""

    unshifted = ifftshift(array)
    y_partner = np.mod(-np.arange(unshifted.shape[0]), unshifted.shape[0])
    x_partner = np.mod(-np.arange(unshifted.shape[1]), unshifted.shape[1])
    partner = unshifted[np.ix_(y_partner, x_partner)]
    return fftshift(0.5 * (unshifted + partner))


@dataclass(frozen=True)
class CTFParameters:
    """Global per-projection CTF parameters in explicit physical units.

    ``defocus_*`` are positive underfocus values in Angstrom.  The astigmatism
    angle is measured from +x toward increasing array rows (the row-down
    frequency convention used by this module), modulo 180 degrees.
    ``phase_shift_rad`` is an additive phase in the aberration phase.
    ``b_factor_A2`` is optional and is zero unless supplied by a separately
    calibrated envelope model; it is not Greco motion correction dose wiring.
    """

    voltage_kV: float
    cs_mm: float
    amplitude_contrast: float
    defocus_u_A: float
    defocus_v_A: float
    astigmatism_angle_deg: float = 0.0
    phase_shift_rad: float = 0.0
    b_factor_A2: float = 0.0


def astigmatism_angle_after_raster_rotation(raw_angle_deg: float, rotation_deg: float) -> float:
    """Map a raw-raster astigmatism azimuth into this module's array frame.

    ``ctf_transfer`` measures frequency angles with ``fy`` increasing in the
    array row-down direction.  ``scipy.ndimage.rotate(image, rotation_deg)``
    rotates image content counter-clockwise in the displayed raster, hence a
    raw azimuth maps to ``raw - rotation`` in this row-down coordinate.  The
    result is intentionally not wrapped: CTF evaluation is 180-degree
    periodic for astigmatism and retaining the unwrapped value makes the
    provenance auditable.
    """

    if not np.isfinite(raw_angle_deg) or not np.isfinite(rotation_deg):
        raise ValueError("raster and astigmatism angles must be finite")
    return float(raw_angle_deg - rotation_deg)


def electron_wavelength_A(voltage_kV: float) -> float:
    """Relativistic electron wavelength in Angstrom."""

    if voltage_kV <= 0:
        raise ValueError("voltage_kV must be positive")
    voltage_V = float(voltage_kV) * 1000.0
    return float(12.2639 / np.sqrt(voltage_V * (1.0 + 0.978466e-6 * voltage_V)))


def _validate_parameters(parameters: CTFParameters) -> None:
    if parameters.cs_mm < 0:
        raise ValueError("cs_mm must be non-negative")
    if not 0.0 <= parameters.amplitude_contrast < 1.0:
        raise ValueError("amplitude_contrast must be in [0, 1)")
    if parameters.defocus_u_A < 0 or parameters.defocus_v_A < 0:
        raise ValueError("defocus values must be non-negative underfocus values")
    if parameters.b_factor_A2 < 0:
        raise ValueError("b_factor_A2 must be non-negative")


def ctf_transfer(
    shape: tuple[int, int],
    pixel_size_A: float,
    parameters: CTFParameters,
) -> FloatSpectrum:
    """Return a shifted real CTF transfer function for one raster.

    The explicit convention is

    ``chi = pi*lambda*df(theta)*s^2 - pi/2*Cs*lambda^3*s^4 + phase``

    and

    ``H = -sqrt(1-A^2)*sin(chi) - A*cos(chi)``.

    Spatial frequency ``s`` is in cycles/Angstrom.  The returned array uses
    ``fftshift`` ordering, matching the reliability maps in this package.
    ``df(theta)`` interpolates U/V as
    ``mean + (U-V)/2*cos(2*(theta-astig_angle))`` with theta from +x.
    """

    _validate_parameters(parameters)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"shape must be a positive 2-D shape, got {shape}")
    if pixel_size_A <= 0:
        raise ValueError("pixel_size_A must be positive")
    height, width = shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None] / float(pixel_size_A)
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :] / float(pixel_size_A)
    s2 = fx * fx + fy * fy
    defocus = ctf_defocus_map(shape, pixel_size_A, parameters)
    wavelength = electron_wavelength_A(parameters.voltage_kV)
    cs_A = parameters.cs_mm * 1.0e7
    chi = (
        np.pi * wavelength * defocus * s2
        - 0.5 * np.pi * cs_A * wavelength**3 * s2**2
        + parameters.phase_shift_rad
    )
    amplitude = float(parameters.amplitude_contrast)
    transfer = -np.sqrt(1.0 - amplitude * amplitude) * np.sin(chi) - amplitude * np.cos(chi)
    if parameters.b_factor_A2:
        transfer *= np.exp(-0.25 * parameters.b_factor_A2 * s2)
    # Even-sized FFT grids have self-conjugate Nyquist lines.  Enforce the
    # discrete symmetry explicitly so applying H to a real image cannot leave
    # an implementation-dependent imaginary residue at those bins.
    return np.asarray(_symmetric_shifted(transfer), dtype=np.float64)


def ctf_defocus_map(
    shape: tuple[int, int],
    pixel_size_A: float,
    parameters: CTFParameters,
) -> NDArray[np.float64]:
    """Return the U/V-interpolated defocus map in shifted FFT coordinates."""

    _validate_parameters(parameters)
    if len(shape) != 2 or min(shape) < 1 or pixel_size_A <= 0:
        raise ValueError("shape must be positive 2-D and pixel_size_A must be positive")
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1]))[None, :]
    theta = np.arctan2(fy, fx)
    astig_angle = np.deg2rad(parameters.astigmatism_angle_deg)
    mean_df = 0.5 * (parameters.defocus_u_A + parameters.defocus_v_A)
    return np.asarray(
        mean_df + 0.5 * (parameters.defocus_u_A - parameters.defocus_v_A) * np.cos(2.0 * (theta - astig_angle)),
        dtype=np.float64,
    )


def regularized_ctf_inverse(transfer_shifted: ArrayLike, regularization: float) -> FloatSpectrum:
    """Return the bounded Tikhonov inverse ``H/(H²+lambda)``.

    This is an explicitly regularized preconditioner for a
    comparison reconstruction.  It is not phase flipping and it is not an
    unregularized CTF deconvolution.  The gain is bounded by
    ``1/(2*sqrt(lambda))`` for real ``H``.
    """

    transfer = np.asarray(transfer_shifted, dtype=np.float64)
    if transfer.ndim != 2 or not np.isfinite(transfer).all():
        raise ValueError("transfer must be a finite 2-D array")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    return transfer / (transfer * transfer + float(regularization))
