"""Data-driven projection-domain reliability from independent half sums."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter


FloatImage = NDArray[np.float32]


@dataclass(frozen=True)
class ReliabilityEstimate:
    """Frequency-domain half/full reliability and diagnostic components.

    ``reliability`` is the half-set estimate.  ``reliability_full`` is the corresponding reliability for the
    average/sum of two independent halves under the usual equal-noise model.
    ``phase_agreement`` is a bounded signed cosine-like diagnostic; it is not
    a statistical coherence estimate.
    """

    reliability: FloatImage
    reliability_full: FloatImage
    signal_power: FloatImage
    noise_power: FloatImage
    cross_power: FloatImage
    phase_agreement: FloatImage
    scale_even: float
    scale_odd: float

    @property
    def reliability_half(self) -> FloatImage:
        """Explicit name for the per-half reliability mask."""

        return self.reliability



def _validate_pair(even: ArrayLike, odd: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    even_array = np.asarray(even, dtype=np.float64)
    odd_array = np.asarray(odd, dtype=np.float64)
    if even_array.ndim != 2 or odd_array.ndim != 2:
        raise ValueError("EVEN and ODD must be 2-D images")
    if even_array.shape != odd_array.shape:
        raise ValueError(f"EVEN/ODD shape mismatch: {even_array.shape} vs {odd_array.shape}")
    if not np.isfinite(even_array).all() or not np.isfinite(odd_array).all():
        raise ValueError("EVEN and ODD must contain only finite values")
    return even_array, odd_array


def _symmetric_shifted_mask(mask_shifted: np.ndarray) -> np.ndarray:
    """Enforce Hermitian symmetry on a real mask in fftshifted coordinates."""

    unshifted = ifftshift(mask_shifted)
    y_partner = np.mod(-np.arange(unshifted.shape[0]), unshifted.shape[0])
    x_partner = np.mod(-np.arange(unshifted.shape[1]), unshifted.shape[1])
    partner = unshifted[np.ix_(y_partner, x_partner)]
    return fftshift(0.5 * (unshifted + partner))


def _tukey_window(length: int, alpha: float) -> np.ndarray:
    if length < 1:
        return np.empty(0, dtype=np.float64)
    if alpha <= 0.0:
        return np.ones(length, dtype=np.float64)
    if alpha >= 1.0:
        return np.hanning(length) if length > 1 else np.ones(1, dtype=np.float64)
    window = np.ones(length, dtype=np.float64)
    edge = alpha * (length - 1.0) / 2.0
    index = np.arange(length, dtype=np.float64)
    left = index < edge
    right = index > (length - 1.0) * (1.0 - alpha / 2.0)
    window[left] = 0.5 * (1.0 + np.cos(np.pi * (index[left] / edge - 1.0)))
    window[right] = 0.5 * (1.0 + np.cos(np.pi * (index[right] / edge - 2.0 / alpha + 1.0)))
    return window


def _apodization_window(shape: tuple[int, int], kind: str, tukey_alpha: float) -> np.ndarray:
    if kind == "none":
        return np.ones(shape, dtype=np.float64)
    if kind != "tukey":
        raise ValueError("apodization must be 'none' or 'tukey'")
    if not 0.0 <= tukey_alpha <= 1.0:
        raise ValueError("tukey_alpha must be in [0, 1]")
    return np.outer(_tukey_window(shape[0], tukey_alpha), _tukey_window(shape[1], tukey_alpha))


def reliability_full_from_half(reliability_half: ArrayLike) -> FloatImage:
    """Convert half-set reliability to full-set reliability.

    The relation ``R_full = 2*R_half/(1+R_half)`` assumes that the two halves
    have first been put on a common signal-amplitude scale and have
    independent, equal noise.  It is a reliability-mask conversion, not an
    amplitude rescaling of the FULL image.
    """

    values = np.asarray(reliability_half, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("reliability_half must be a finite 2-D array")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("reliability_half must be bounded in [0, 1]")
    return np.asarray(np.clip(2.0 * values / (1.0 + values + np.finfo(np.float64).eps), 0.0, 1.0), dtype=np.float32)


def _half_scales(
    even_spectrum: np.ndarray,
    odd_spectrum: np.ndarray,
    low_frequency_fraction: float,
) -> tuple[float, float]:
    height, width = even_spectrum.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    amplitude_even = np.abs(even_spectrum)
    amplitude_odd = np.abs(odd_spectrum)
    valid = radius <= low_frequency_fraction
    valid &= amplitude_even > np.finfo(np.float64).eps
    valid &= amplitude_odd > np.finfo(np.float64).eps
    if not np.any(valid):
        return 1.0, 1.0
    log_ratio = np.log(amplitude_odd[valid]) - np.log(amplitude_even[valid])
    log_ratio = log_ratio[np.isfinite(log_ratio)]
    if log_ratio.size == 0:
        return 1.0, 1.0
    # Trim unstable ratios from spectral nulls before taking the robust centre.
    low, high = np.percentile(log_ratio, (10.0, 90.0))
    log_ratio = np.clip(log_ratio, low, high)
    ratio = float(np.exp(np.median(log_ratio)))
    ratio = float(np.clip(ratio, 0.25, 4.0))
    return float(np.sqrt(ratio)), float(1.0 / np.sqrt(ratio))


def estimate_reliability(
    even: ArrayLike,
    odd: ArrayLike,
    *,
    smoothing_sigma: float = 2.0,
    low_frequency_fraction: float = 0.25,
    method: str = "cross_power",
    apodization: str = "none",
    tukey_alpha: float = 0.1,
) -> ReliabilityEstimate:
    """Estimate a bounded per-frequency signal reliability from EVEN/ODD.

    The halves are symmetrically amplitude-matched.  For independent equal
    noise, the expected cross power estimates signal power and
    ``0.5*|E-O|**2`` estimates noise power.  For the cross-power estimator,
    the *signed* cross power is smoothed before clipping its signal estimate
    at zero; clipping first would create a positive noise bias.  Callers may
    request a light Tukey taper to suppress reproducible constant-padding
    borders before the FFT; the default remains untapered, while the
    isolated central7 route opts into the taper explicitly.  No CTF
    deconvolution is performed here; CTF zeros therefore remain unsupported
    frequencies.
    """

    if smoothing_sigma < 0 or low_frequency_fraction <= 0:
        raise ValueError("smoothing_sigma must be non-negative and frequency fraction positive")
    if method not in {"cross_power", "half_average_wiener"}:
        raise ValueError(f"unknown reliability method: {method}")
    even_array, odd_array = _validate_pair(even, odd)
    taper = _apodization_window(even_array.shape, apodization, tukey_alpha)
    even_spectrum = fftshift(fft2(even_array * taper))
    odd_spectrum = fftshift(fft2(odd_array * taper))
    scale_even, scale_odd = _half_scales(
        even_spectrum,
        odd_spectrum,
        low_frequency_fraction,
    )
    even_spectrum *= scale_even
    odd_spectrum *= scale_odd

    cross_power = np.real(even_spectrum * np.conj(odd_spectrum))
    noise_power = 0.5 * np.abs(even_spectrum - odd_spectrum) ** 2
    if method == "cross_power":
        signed_signal_power = gaussian_filter(cross_power, smoothing_sigma, mode="wrap") if smoothing_sigma else cross_power
        signal_power = np.maximum(signed_signal_power, 0.0)
        effective_noise_power = noise_power
    else:
        # M=(E+O)/2 has noise variance N/2 when N estimates the variance of
        # either independent half.  Subtract that bias from |M|^2 before the
        # Wiener gain is formed; this is a diagnostic alternative to the
        # cross-power estimator, not a silent replacement of it.
        mean_power = np.abs(0.5 * (even_spectrum + odd_spectrum)) ** 2
        signed_signal_power = mean_power - 0.5 * noise_power
        if smoothing_sigma:
            signed_signal_power = gaussian_filter(signed_signal_power, smoothing_sigma, mode="wrap")
        signal_power = np.maximum(signed_signal_power, 0.0)
        effective_noise_power = 0.5 * noise_power
    if smoothing_sigma:
        effective_noise_power = gaussian_filter(effective_noise_power, smoothing_sigma, mode="wrap")
    denominator = signal_power + effective_noise_power + np.finfo(np.float64).eps
    reliability = np.clip(signal_power / denominator, 0.0, 1.0)
    power_even = np.abs(even_spectrum) ** 2
    power_odd = np.abs(odd_spectrum) ** 2
    phase_agreement = np.clip(
        cross_power / (np.sqrt(power_even * power_odd) + np.finfo(np.float64).eps),
        -1.0,
        1.0,
    )
    if smoothing_sigma:
        phase_agreement = gaussian_filter(phase_agreement, smoothing_sigma, mode="wrap")
    reliability = _symmetric_shifted_mask(reliability)
    phase_agreement = _symmetric_shifted_mask(phase_agreement)
    reliability[reliability.shape[0] // 2, reliability.shape[1] // 2] = 1.0
    phase_agreement[phase_agreement.shape[0] // 2, phase_agreement.shape[1] // 2] = 1.0
    reliability_full = reliability_full_from_half(reliability)
    reliability_full[reliability_full.shape[0] // 2, reliability_full.shape[1] // 2] = 1.0
    return ReliabilityEstimate(
        reliability=reliability.astype(np.float32),
        reliability_full=reliability_full.astype(np.float32),
        signal_power=signal_power.astype(np.float32),
        noise_power=effective_noise_power.astype(np.float32),
        cross_power=cross_power.astype(np.float32),
        phase_agreement=phase_agreement.astype(np.float32),
        scale_even=scale_even,
        scale_odd=scale_odd,
    )


def apply_reliability_filter(full: ArrayLike, reliability: ArrayLike) -> FloatImage:
    """Apply a shifted reliability mask to one real projection."""

    image = np.asarray(full, dtype=np.float64)
    mask = np.asarray(reliability, dtype=np.float64)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("full image and reliability mask must have matching 2-D shapes")
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError("full image and reliability mask must be finite")
    filtered = ifft2(ifftshift(fftshift(fft2(image)) * mask)).real
    # The zero-frequency component is retained by the mask; this also makes
    # the invariant explicit against small floating-point drift.
    filtered += float(image.mean() - filtered.mean())
    return filtered.astype(np.float32)
