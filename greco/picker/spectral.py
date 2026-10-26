"""Fourier transfer functions and deterministic support projections for Greco Picker.

Author: Carlos Fernández Rodríguez, 2026. This is not a clean-room audit:
previous PyTom and pytom-match-pick implementations were inspected. Historical
provenance is recorded in licenses/THIRD_PARTY_NOTICES.md. This rewrite does not
claim that prior attribution or applicable license obligations are extinguished.
"""
import numpy as np


def frequency_axes(shape, reduced=True, centered=False, reduced_shape=False):
    """Independent one-dimensional coordinates for the real/full FFT layout.

    Preserve the established Nyquist convention, including odd input sizes and
    the inferred last-axis Nyquist when a reduced shape is supplied.
    """
    if len(shape) not in (1, 2, 3):
        raise ValueError('radial_grid() only works for 1D, 2D or 3D shapes')
    axes = []
    for dimension, length in enumerate(shape):
        if reduced and dimension == len(shape) - 1:
            samples = length if reduced_shape else length // 2 + 1
            axis = np.arange(samples) / (samples - 1)
        else:
            half = length // 2
            start = -half
            # Floating arange is deliberate: the compatibility contract includes
            # FP64 division, rather than a float32 or fftfreq approximation.
            axis = np.abs(np.arange(start, half + length % 2, dtype=np.float64)) / half
            if not centered:
                axis = np.fft.ifftshift(axis)
        layout = [1] * len(shape)
        layout[dimension] = axis.size
        axes.append(axis.reshape(layout))
    return axes


def radial_grid(shape, reduced=True, fftshifted=False, shape_is_reduced=False):
    coordinates = frequency_axes(shape, reduced, fftshifted, shape_is_reduced)
    if len(coordinates) == 1:
        return coordinates[0]
    squared = coordinates[0] ** 2
    for coordinate in coordinates[1:]:
        squared = squared + coordinate ** 2
    return np.sqrt(squared)


def gaussian_transfer(shape, spacing, low_pass=None, high_pass=None):
    if low_pass is None and high_pass is None:
        raise ValueError('Either low-pass or high-pass needs to be set')
    radius = radial_grid(shape)
    half_width_scale = np.sqrt(2 * np.log(2))
    def attenuation(resolution):
        sigma = (2 * spacing / resolution) / half_width_scale
        return np.exp(-(radius ** 2) / (2 * sigma ** 2))
    if high_pass is None:
        return attenuation(low_pass)
    high = 1 - attenuation(high_pass)
    return high if low_pass is None else high * attenuation(low_pass)


def ctf_transfer(shape, pixel_size, ctf_data, reduced=True, fftshifted=False):
    parameters = ctf_data
    # Relativistic de Broglie wavelength. Keep the historical constants and
    # operation order: updating constants would change established picker scores.
    voltage = parameters.voltage
    h, electron_charge = 6.62606896e-34, 1.60217646e-19
    electron_mass, speed_of_light = 9.10938215e-31, 299792458
    wavelength = h / np.sqrt(electron_charge * voltage * electron_mass *
                            (electron_charge / electron_mass * voltage / speed_of_light ** 2 + 2))
    frequency = radial_grid(shape, reduced, fftshifted) / (2 * pixel_size)
    defocus_phase = np.pi * wavelength * parameters.defocus * frequency ** 2
    aberration_phase = .5 * np.pi * parameters.spherical_aberration * wavelength ** 3 * frequency ** 4
    phase = defocus_phase - aberration_phase
    amplitude_phase = np.arctan(parameters.amplitude_contrast /
                                np.sqrt(1 - parameters.amplitude_contrast ** 2))
    transfer = -np.sin(phase + amplitude_phase + np.deg2rad(parameters.phase_shift_deg))
    if parameters.flip_phase:
        return np.abs(transfer)
    transfer *= -1 if parameters.defocus > 0 else 1
    return transfer


def binary_wedge(shape, alpha_min, alpha_max, cut_off_radius, level_angle_x=0.):
    # Signed distances to the two limiting section planes, with a one-cell
    # transition. Retain the legacy Fourier-axis signs and edge normalization.
    coordinates = []
    for dimension in (0, 1):
        axis = -(np.fft.fftfreq(shape[dimension]) * shape[dimension] / (shape[dimension] // 2))
        layout = [1, 1, 1]
        layout[dimension] = len(axis)
        coordinates.append(axis.reshape(layout))
    across, along = coordinates
    vertical = radial_grid((shape[2],)).reshape(1, 1, -1)
    section_z = vertical * np.cos(level_angle_x) - along * np.sin(level_angle_x)
    radius = np.sqrt(across ** 2 + section_z ** 2)
    direction = np.arctan2(across, section_z)
    distances = [across * np.sin(boundary) + section_z * np.cos(boundary)
                 for boundary in (alpha_min, alpha_max)]
    inside = (direction >= alpha_min) & (direction <= alpha_max)
    lower = np.minimum(*distances)
    upper = np.where(inside, radius, np.maximum(*distances))
    signed_distance = np.minimum(-lower, upper)
    transition = (signed_distance.max() - signed_distance.min()) / (2 * min(shape[0], shape[2]) // 2)
    result = (np.clip(signed_distance, -transition, transition) + transition) / (2 * transition)
    result[radial_grid(shape) > cut_off_radius] = 0
    return result




class SectionAccumulator:
    """Accumulate interpolated Fourier sections without changing tilt order."""
    def __init__(self, shape, radians, pixel_size, rotate):
        self.shape = shape
        self.size = shape[0]
        self.rotate = rotate
        self.buffer = np.zeros(shape)
        self.sum = np.zeros((self.size, self.size, self.size // 2 + 1))
        delta = np.min(np.abs(np.diff(radians)))
        overlap = 1 / (delta * self.size)
        half = self.size // 2
        coordinate = np.abs(np.arange(-half, half + self.size % 2, dtype=float)) / half * .5
        ramp = coordinate / overlap
        ramp[ramp > 1] = 1
        self.ramp = np.broadcast_to(ramp[:, None], (self.size, self.size))
        self.frequency_squared = (radial_grid((self.size, self.size), False, True) / (2 * pixel_size)) ** 2
        self.pixel_size = pixel_size

    def add(self, alpha, ctf, dose, level_x, level_y):
        section = self.ramp if ctf is None else ctf_transfer(
            (self.size, self.size), self.pixel_size * 1e-10, ctf, False, True) * self.ramp
        section = section * np.cos(alpha)
        if dose is not None:
            displacement = np.sqrt(dose * 4 / (8 * np.pi ** 2))
            attenuation = np.exp(-2 * np.pi ** 2 * displacement ** 2 * self.frequency_squared)
            section = section * attenuation
        if self.size % 2 == 0:
            section[0, :] = 0
            section[:, 0] = 0
        self.buffer[:, :, self.size // 2] = section
        rotated = self.rotate(self.buffer, rotation=(alpha + level_y, level_x, 0),
                              rotation_units='rad', rotation_order='ryxz',
                              center=(self.size // 2,) * 3, order=3)
        uncentered = np.fft.ifftshift(rotated, axes=(0, 1, 2))
        self.sum += uncentered[:, :, :self.size // 2 + 1]

    def finish(self, cutoff):
        result = np.roll(self.sum[::-1, ::-1, :], (1, 1), axis=(0, 1))
        result[radial_grid(self.shape) > cutoff] = 0
        return result


def weighted_wedge(shape, tilt_angles_rad, cut_off_radius, pixel_size_angstrom,
                   accumulated_dose_per_tilt=None, ctf_params_per_tilt=None,
                   level_angle_x=0., level_angle_y=0.):
    from greco.picker.geometry import rotate_cpu
    count = len(tilt_angles_rad)
    for values, label in ((accumulated_dose_per_tilt, 'dose'), (ctf_params_per_tilt, 'ctf')):
        if values is not None and len(values) != count:
            raise ValueError(f'{label} list must match tilt angle list length')
    doses = [None] * count if accumulated_dose_per_tilt is None else accumulated_dose_per_tilt
    ctfs = [None] * count if ctf_params_per_tilt is None else ctf_params_per_tilt
    accumulator = SectionAccumulator(shape, tilt_angles_rad, pixel_size_angstrom, rotate_cpu)
    for angle, ctf, dose in zip(tilt_angles_rad, ctfs, doses):
        accumulator.add(angle, ctf, dose, level_angle_x, level_angle_y)
    return accumulator.finish(cut_off_radius)


def randomize_phases(template, mask, n_iter=40, seed=321):
    """Alternate fixed Fourier amplitudes with the supplied real-space support.

    The generator and projection sequence are part of reproducibility: changing
    either changes the noise reference, even if its distribution is equivalent.
    """
    source = np.asarray(template, dtype=np.float64) * mask
    magnitude = np.abs(np.fft.rfftn(source))
    def project_amplitudes(field):
        spectrum = np.fft.rfftn(field)
        phases = np.angle(spectrum)
        return np.fft.irfftn(magnitude * np.exp(1j * phases), s=source.shape,
                            axes=(-3, -2, -1))
    noise = np.random.default_rng(seed).standard_normal(source.shape)
    reconstructed = project_amplitudes(noise)
    for iteration in range(n_iter):
        reconstructed = project_amplitudes(reconstructed * mask)
    reconstructed = reconstructed * mask
    if np.sign(reconstructed.sum()) != np.sign(source.sum()):
        reconstructed = -reconstructed
    return reconstructed.astype(np.float32)
