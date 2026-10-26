"""Spatially varying CTF correction for tilted projections.

The geometry and 512/256 tile/core layout follow Greco ``CCorrImgCtf``.
Greco deliberately keeps its existing bounded inverse ``H/(H^2+lambda)``;
only the invalid global-defocus assumption is replaced by the local defocus
at each core centre.  This makes the change independently testable from a
change of deconvolution filter.
"""

from __future__ import annotations

import numpy as np

from .ctf_estimator import TiltDefocusGeometry
from .ctf_model import electron_wavelength_A


def _core_starts(length: int, core_size: int) -> list[int]:
    if length < core_size or length % core_size:
        raise ValueError(f"image side {length} must be a multiple of core size {core_size}")
    offset = (length % core_size) // 2
    return [offset + i * core_size for i in range(length // core_size)]


def _greco_keep_center_window(tile_size: int, core_size: int) -> np.ndarray:
    """Rounded edge window that preserves the central core of each tile."""
    c = tile_size * 0.5
    yy, xx = np.indices((tile_size, tile_size), dtype=np.float64)
    radius = np.hypot(xx - c, yy - c)
    mask_radius = np.hypot(core_size, core_size) * 0.5
    q = radius / mask_radius - 1.0
    window = np.ones_like(radius)
    outside = q > 0.0
    taper = 0.5 * (1.0 - np.cos(np.pi * np.clip(q[outside], 0.0, 1.0)))
    window[outside] = 1.0 - taper * taper
    return window.astype(np.float32)


def correction_filter(transfer, frequency_cycles_pixel, *, mode='tikhonov',
                      regularization=.05, tilt_deg=0., wiener_lowpass=15., xp=np):
    """Spectral law; optional phase-flip/SZ preserve dark-particle contrast.

    Inputs are fftshift ordered. DC is untouched in dark-preserving modes,
    as in the reference kernel. Historical Tikhonov behavior is unchanged.
    """
    if mode=='tikhonov':return transfer/(transfer*transfer+regularization)
    if mode not in ('phase_flip','wiener_sz'):raise ValueError('Unknown spatial CTF filter')
    sign=xp.where(transfer<=0,1.,-1.)
    if mode=='phase_flip':result=sign
    else:
        if wiener_lowpass<0 or not np.isfinite(wiener_lowpass):raise ValueError('Invalid Wiener lowpass')
        noise=8.*xp.exp(2.*frequency_cycles_pixel**2)
        bfactor=wiener_lowpass/(np.cos(tilt_deg*.01745)+.001)
        result=xp.exp(-bfactor*frequency_cycles_pixel)*(noise+1.)/(xp.abs(transfer)+noise)*sign
    result=result.copy();result[result.shape[0]//2,result.shape[1]//2]=1.
    return result


def correct_tilt_ctf_spatial(
    image,
    *,
    pixel_size_A: float,
    tilt_deg: float,
    tilt_axis_deg: float,
    defocus_u_A: float,
    defocus_v_A: float,
    astigmatism_angle_deg: float,
    voltage_kV: float,
    cs_mm: float,
    amplitude_contrast: float,
    phase_shift_rad: float = 0.0,
    regularization: float = 0.05,
    handedness: int = 1,
    alpha_offset_deg: float = 0.0,
    beta_offset_deg: float = 0.0,
    tile_size: int = 512,
    core_size: int = 256,
    gpu: bool = False,
    return_device: bool = False,
    filter_mode: str = 'tikhonov',
    wiener_lowpass: float = 15.,
    gradient_A_per_pixel=None,
    coordinate_origin=None,
) -> np.ndarray:
    """Apply a local-defocus bounded inverse and paste non-overlapping cores.

    The astigmatism ratio is held fixed, as in Greco: changing the local
    mean defocus scales both the mean and astigmatic half-difference.
    """
    if gpu:
        import cupy as xp
    else:
        xp = np
    source = xp.asarray(image, dtype=xp.float32)
    if source.ndim != 2:
        raise ValueError("image must be 2-D")
    if tile_size < core_size or tile_size % 2 or core_size % 2:
        raise ValueError("tile/core sizes must be even and tile >= core")
    ys = _core_starts(source.shape[0], core_size)
    xs = _core_starts(source.shape[1], core_size)
    out = xp.empty_like(source)
    window = xp.asarray(_greco_keep_center_window(tile_size, core_size))
    geometry = TiltDefocusGeometry(
        tilt_deg=tilt_deg,
        tilt_axis_deg=tilt_axis_deg,
        pixel_size_A=pixel_size_A,
        handedness=handedness,
        alpha_offset_deg=alpha_offset_deg,
        beta_offset_deg=beta_offset_deg,
        gradient_A_per_pixel=gradient_A_per_pixel,
    )
    central = 0.5 * (defocus_u_A + defocus_v_A)
    ratio = 0.5 * (defocus_v_A - defocus_u_A) / max(central, np.finfo(float).eps)
    core_in_tile = (tile_size - core_size) // 2
    half = tile_size // 2
    padded = xp.pad(source, ((half, half), (half, half)), mode="reflect")
    # Static shifted frequency grid.  Only the local mean defocus changes
    # between cores; doing the transfer and FFT on the active device avoids
    # hundreds of NumPy FFTs for one 4096x4096 projection.
    fy = xp.fft.fftshift(xp.fft.fftfreq(tile_size))[:, None] / pixel_size_A
    fx = xp.fft.fftshift(xp.fft.fftfreq(tile_size))[None, :] / pixel_size_A
    s2 = fx * fx + fy * fy
    theta = xp.arctan2(fy, fx)
    astig = xp.deg2rad(astigmatism_angle_deg)
    astig_shape = 1.0 - ratio * xp.cos(2.0 * (theta - astig))
    frequency_cycles_pixel = None if filter_mode=='tikhonov' else xp.sqrt(s2)*pixel_size_A
    wavelength = electron_wavelength_A(voltage_kV)
    cs_A = cs_mm * 1.0e7
    amp = float(amplitude_contrast)
    origin=(np.asarray(source.shape,dtype=float)-1)/2 if gradient_A_per_pixel is not None or coordinate_origin=="image_pixel_centres" else np.asarray(source.shape,dtype=float)/2
    if gpu and filter_mode == 'tikhonov':
        out = _correct_batched(source, padded, window, ys, xs, geometry,
            central, origin, astig_shape, s2, wavelength, cs_A, amp,
            phase_shift_rad, regularization, tile_size, core_size, xp)
        return out if return_device else xp.asnumpy(out)
    for y0 in ys:
        for x0 in xs:
            cy = y0 + core_size // 2
            cx = x0 + core_size // 2
            tile = padded[cy:cy + tile_size, cx:cx + tile_size] * window
            local = float(geometry.local_mean_defocus_A(
                central, cx-origin[1], cy-origin[0]))
            chi = (xp.pi * wavelength * (local * astig_shape) * s2
                   - 0.5 * xp.pi * cs_A * wavelength**3 * s2**2
                   + phase_shift_rad)
            transfer = (-np.sqrt(1.0 - amp * amp) * xp.sin(chi)
                        - amp * xp.cos(chi))
            # Match ctf_model's explicit discrete Hermitian projection,
            # including the self-conjugate Nyquist rows/columns.
            unshifted = xp.fft.ifftshift(transfer)
            partner = unshifted[xp.ix_((-xp.arange(tile_size)) % tile_size,
                                       (-xp.arange(tile_size)) % tile_size)]
            transfer = xp.fft.fftshift(0.5 * (unshifted + partner))
            inverse = correction_filter(transfer,frequency_cycles_pixel,
                mode=filter_mode,regularization=regularization,tilt_deg=tilt_deg,
                wiener_lowpass=wiener_lowpass,xp=xp)
            corrected = xp.fft.ifft2(
                xp.fft.fft2(tile) * xp.fft.ifftshift(inverse)).real.astype(xp.float32)
            a = core_in_tile
            b = a + core_size
            out[y0:y0 + core_size, x0:x0 + core_size] = corrected[a:b, a:b]
    return out if gpu and return_device else xp.asnumpy(out) if gpu else np.asarray(out)


def _correct_batched(source, padded, window, ys, xs, geometry, central, origin,
                     astig_shape, s2, wavelength, cs_A, amp, phase, regularization,
                     tile_size, core_size, xp):
    """Bound transient double-precision transfer/FFT work to small core batches."""
    free, _ = xp.cuda.runtime.memGetInfo()
    pool = xp.get_default_memory_pool()
    available = free + pool.total_bytes() - pool.used_bytes()
    # Complex FFTs and float64 transfer arrays dominate the peak. Never build
    # the old all-core workspace (~15 GiB for one 4096² image).
    batch = max(1, min(32, int(min(2*1024**3, available//4)//(tile_size**2*224))))
    starts = np.asarray([(y, x) for y in ys for x in xs])
    centers = starts + core_size//2
    local = geometry.local_mean_defocus_A(central,
        centers[:, 1]-origin[1], centers[:, 0]-origin[0])
    out = xp.empty_like(source)
    ar = xp.arange(tile_size)
    neg = (-xp.arange(tile_size)) % tile_size
    iy, ix = neg[:, None], neg[None, :]
    a = (tile_size-core_size)//2
    for first in range(0, len(starts), batch):
        stop = min(first+batch, len(starts))
        cy, cx = centers[first:stop].T
        tiles = padded[xp.asarray(cy)[:, None, None]+ar[None, :, None],
                       xp.asarray(cx)[:, None, None]+ar[None, None, :]] * window[None]
        L = xp.asarray(local[first:stop], dtype=xp.float64)[:, None, None]
        chi = (xp.pi*wavelength*(L*astig_shape[None])*s2[None]
               - .5*xp.pi*cs_A*wavelength**3*s2[None]**2 + phase)
        transfer = -np.sqrt(1.-amp*amp)*xp.sin(chi) - amp*xp.cos(chi)
        del chi
        unshifted = xp.fft.ifftshift(transfer, axes=(-2, -1))
        partner = unshifted[:, iy, ix]
        transfer = xp.fft.fftshift(.5*(unshifted+partner), axes=(-2, -1))
        del unshifted, partner
        inverse = transfer/(transfer*transfer+regularization)
        del transfer
        corrected = xp.fft.ifft2(xp.fft.fft2(tiles)*
            xp.fft.ifftshift(inverse, axes=(-2, -1))).real.astype(xp.float32)
        for k, (y, x) in enumerate(starts[first:stop]):
            out[y:y+core_size, x:x+core_size] = corrected[k, a:a+core_size, a:a+core_size]
        del tiles, inverse, corrected
    return out
