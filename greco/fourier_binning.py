"""Small, explicit Fourier crop resampler for central landmark seeds.

This operator is deliberately separate from the production ``local_mean``
preprocessing path.  It is a conservative even-grid Fourier resampler:
the centred spectrum is cropped (with source +/-Nyquist pairs folded into an
even target's single Nyquist bin), the crop is projected onto discrete
Hermitian symmetry, and the inverse transform is normalized so that the DC
component (and therefore the image mean) is preserved.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatImage = NDArray[np.float32]


def _hermitianize_unshifted(spectrum_unshifted: np.ndarray) -> np.ndarray:
    """Enforce Hermitian symmetry in unshifted FFT coordinates."""

    unshifted = np.asarray(spectrum_unshifted, dtype=np.complex128)
    y_partner = np.mod(-np.arange(unshifted.shape[0]), unshifted.shape[0])
    x_partner = np.mod(-np.arange(unshifted.shape[1]), unshifted.shape[1])
    partner = unshifted[np.ix_(y_partner, x_partner)]
    symmetric = 0.5 * (unshifted + np.conj(partner))
    return symmetric


def _folded_center_crop(spectrum: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Crop physical frequencies and fold source Nyquist pairs correctly."""

    source_shape = spectrum.shape

    def base_indices(target_length: int, source_length: int) -> np.ndarray:
        indices = np.arange(target_length, dtype=np.int64)
        half = target_length // 2
        signed = np.where(indices <= half, indices, indices - target_length)
        if target_length % 2 == 0:
            signed[half] = -half
        return np.mod(signed, source_length)

    y_base = base_indices(target_shape[0], source_shape[0])
    x_base = base_indices(target_shape[1], source_shape[1])
    result = spectrum[np.ix_(y_base, x_base)].copy()

    # An even target has one self-conjugate Nyquist bin. Both source members
    # of each +/- Nyquist pair fold into that one output bin. These additions
    # are vectorized; the central crop must not introduce a Python loop over
    # image pixels.
    if target_shape[0] % 2 == 0:
        y_nyquist = target_shape[0] // 2
        y_partner = (-y_nyquist) % source_shape[0]
        result[y_nyquist, :] += spectrum[y_partner, x_base]
    if target_shape[1] % 2 == 0:
        x_nyquist = target_shape[1] // 2
        x_partner = (-x_nyquist) % source_shape[1]
        result[:, x_nyquist] += spectrum[y_base, x_partner]
    if target_shape[0] % 2 == 0 and target_shape[1] % 2 == 0:
        y_nyquist = target_shape[0] // 2
        x_nyquist = target_shape[1] // 2
        result[y_nyquist, x_nyquist] += spectrum[(-y_nyquist) % source_shape[0], (-x_nyquist) % source_shape[1]]
    return result


def fourier_bin_2d(image: ArrayLike, factor: int = 8) -> FloatImage:
    """Downsample a real 2-D image by a centred Fourier-spectrum crop.

    ``factor`` must divide both dimensions.  The returned pixel represents
    ``factor`` input pixels in each direction.  No raster transpose, mirror,
    spatial padding, or phase convention is introduced by this function.
    """

    source = np.asarray(image, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError(f"image must be a non-empty 2-D array, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("image must contain only finite values")
    if not isinstance(factor, (int, np.integer)) or int(factor) < 1:
        raise ValueError("factor must be a positive integer")
    factor = int(factor)
    if source.shape[0] % factor or source.shape[1] % factor:
        raise ValueError(f"shape {source.shape} is not divisible by factor {factor}")
    if factor == 1:
        return source.astype(np.float32, copy=True)

    target_shape = (source.shape[0] // factor, source.shape[1] // factor)
    spectrum = np.fft.fft2(source)
    cropped = _folded_center_crop(spectrum, target_shape)

    # numpy's inverse transform divides by the target number of pixels, while
    # the forward transform above contains the source number of pixels.
    cropped *= float(np.prod(target_shape)) / float(np.prod(source.shape))
    cropped = _hermitianize_unshifted(cropped)
    restored = np.fft.ifft2(cropped)
    # Hermitianization makes the imaginary component roundoff only.  Do not
    # silently hide a broken crop: a large residual indicates a programming
    # error and is rejected instead of being discarded by ``.real``.
    imaginary_residual = float(np.max(np.abs(restored.imag)))
    real_scale = max(1.0, float(np.max(np.abs(restored.real))))
    if imaginary_residual > 5e-10 * real_scale:
        raise RuntimeError(f"Fourier crop lost Hermitian symmetry: imaginary residual={imaginary_residual}")
    result = np.asarray(restored.real, dtype=np.float64)
    # Make the DC/mean invariant explicit against accumulated FFT roundoff.
    result += float(source.mean() - result.mean())
    return result.astype(np.float32)


def fourier_bin_2d_gpu(image, factor=8):
    """Device-resident equivalent of the CPU crop, including Nyquist folding.

    Float64 FFT preserves the historical numerical contract; only the small
    output needs to be copied to host by the caller.
    """
    import cupy as cp
    source=cp.asarray(image,dtype=cp.float64)
    if source.ndim!=2 or min(source.shape)<2 or not isinstance(factor,(int,np.integer)) or factor<1:
        raise ValueError('invalid image or binning')
    if any(n%factor for n in source.shape):raise ValueError('binning must divide image shape')
    if not bool(cp.isfinite(source).all()):raise ValueError('image must be finite')
    if factor==1:return source.astype(cp.float32)
    target=tuple(n//factor for n in source.shape)
    spectrum=cp.fft.fft2(source)
    indices=[]
    for n,m in zip(target,source.shape):
        v=cp.arange(n);signed=cp.where(v<=n//2,v,v-n)
        if n%2==0:signed[n//2]=-n//2
        indices.append(signed%m)
    y,x=indices;crop=spectrum[cp.ix_(y,x)].copy()
    if target[0]%2==0:crop[target[0]//2,:]+=spectrum[(-target[0]//2)%source.shape[0],x]
    if target[1]%2==0:crop[:,target[1]//2]+=spectrum[y,(-target[1]//2)%source.shape[1]]
    if target[0]%2==0 and target[1]%2==0:
        crop[target[0]//2,target[1]//2]+=spectrum[(-target[0]//2)%source.shape[0],(-target[1]//2)%source.shape[1]]
    crop*=float(np.prod(target))/float(np.prod(source.shape))
    partner=cp.conj(crop[cp.ix_((-cp.arange(target[0]))%target[0],(-cp.arange(target[1]))%target[1])])
    restored=cp.fft.ifft2(.5*(crop+partner))
    if float(cp.max(cp.abs(restored.imag)))>5e-10*max(1.,float(cp.max(cp.abs(restored.real)))):
        raise RuntimeError('Fourier crop lost Hermitian symmetry')
    result=restored.real;result+=source.mean()-result.mean()
    return result.astype(cp.float32)


__all__ = ["fourier_bin_2d", "fourier_bin_2d_gpu"]
