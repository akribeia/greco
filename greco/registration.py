"""Comparable bounded local-registration surfaces for bin8 landmark SPA."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import correlate, map_coordinates, shift as ndi_shift, sobel


@dataclass(frozen=True)
class LocalRegistrationResult:
    correction_yx: NDArray[np.float64]
    best_score: float
    zero_score: float
    score_improvement: float
    second_best_score: float
    peak_prominence: float
    second_peak_distance: float
    hessian_yx: NDArray[np.float64]
    hessian_eigenvalues: NDArray[np.float64]
    anisotropy: float
    boundary: bool
    multimodal: bool
    subpixel_valid: bool
    valid: bool
    surface: NDArray[np.float32]


def cosine_disk_mask(size: int = 33, radius: float = 14.0, edge: float = 3.0) -> NDArray[np.float64]:
    """Return the 2-D projection-support analogue of the local sphere mask."""
    if size < 3 or size % 2 != 1 or radius <= 0 or edge <= 0 or edge > radius:
        raise ValueError("invalid cosine disk")
    c = (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    r = np.hypot(y - c, x - c)
    mask = np.ones((size, size), dtype=np.float64)
    mask[r >= radius] = 0
    transition = (r > radius - edge) & (r < radius)
    mask[transition] = 0.5 * (1 + np.cos(np.pi * (r[transition] - (radius - edge)) / edge))
    return mask


@lru_cache(maxsize=8)
def _hann2(size: int) -> np.ndarray:
    window = np.outer(np.hanning(size), np.hanning(size))
    window.setflags(write=False)
    return window


def sample_search_region(image: ArrayLike, centre_yx: ArrayLike, patch_size: int = 33, search_radius: int = 4) -> NDArray[np.float32]:
    """Sample a search region at a floating projected landmark centre."""
    data = np.asarray(image, dtype=np.float32)
    centre = np.asarray(centre_yx, dtype=float)
    extent = patch_size // 2 + int(search_radius)
    if data.ndim != 2 or centre.shape != (2,) or not np.isfinite(centre).all():
        raise ValueError("invalid image or centre")
    if np.any(centre - extent < 0) or np.any(centre + extent > np.asarray(data.shape) - 1):
        raise ValueError("search region outside detector support")
    offsets = np.mgrid[-extent:extent + 1, -extent:extent + 1].astype(float)
    return map_coordinates(data, offsets + centre[:, None, None], order=1, mode="constant", prefilter=False).astype(np.float32)


def _weighted_zncc_batch(model: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Evaluate the identical weighted ZNCC for a batch of trailing 2-D patches."""
    total = float(weights.sum())
    if total <= 0:
        return np.full(observed.shape[:-2], -1.0)
    model_mean = float(np.sum(weights * model) / total)
    centred_model = model - model_mean
    observed_mean = np.sum(observed * weights, axis=(-2, -1)) / total
    centred_observed = observed - observed_mean[..., None, None]
    numerator = np.sum(weights * centred_model * centred_observed, axis=(-2, -1))
    model_energy = float(np.sum(weights * centred_model * centred_model))
    observed_energy = np.sum(weights * centred_observed * centred_observed, axis=(-2, -1))
    denominator = np.sqrt(model_energy * observed_energy)
    return np.divide(numerator, denominator, out=np.full_like(numerator, -1.0), where=denominator > 1e-12)


def _patch_array(region: np.ndarray, patch_size: int, radius: int) -> np.ndarray:
    """Return patches in correction-grid order without copying pixel data."""
    windows = np.lib.stride_tricks.sliding_window_view(region, (patch_size, patch_size))
    # sliding-window displacement is the negative raster correction.
    return windows[::-1, ::-1]


def _zncc_surface(model: np.ndarray, region: np.ndarray, weights: np.ndarray, radius: int) -> np.ndarray:
    return _weighted_zncc_batch(model, _patch_array(region, model.shape[0], radius), weights)


def _pcc_surface(model: np.ndarray, region: np.ndarray, radius: int) -> np.ndarray:
    """Reproduce the validated main-path bounded-patch PCC surface."""
    observed = region[radius:radius + model.shape[0], radius:radius + model.shape[1]]
    centred_model = model - model.mean()
    model_norm = np.linalg.norm(centred_model)
    if model_norm <= 1e-12:
        return np.full((2 * radius + 1, 2 * radius + 1), -1.0)
    surface = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    for iy, dy in enumerate(range(-radius, radius + 1)):
        for ix, dx in enumerate(range(-radius, radius + 1)):
            moved = ndi_shift(observed, (dy, dx), order=1, mode="constant",
                cval=float(observed.mean()), prefilter=False)
            moved -= moved.mean()
            denominator = model_norm * np.linalg.norm(moved)
            surface[iy, ix] = np.dot(centred_model.ravel(), moved.ravel()) / denominator if denominator > 1e-12 else -1.0
    return surface


def _gradient_surface(model: np.ndarray, region: np.ndarray, weights: np.ndarray, radius: int, intensity_fraction: float) -> np.ndarray:
    patches = _patch_array(region, model.shape[0], radius)
    intensity = _weighted_zncc_batch(model, patches, weights)
    model_y, model_x = sobel(model, axis=0, mode="nearest") / 8, sobel(model, axis=1, mode="nearest") / 8
    # Explicit singleton batch dimensions prevent scipy.ndimage.sobel from
    # smoothing across the two correction-grid axes of the 4-D patch view.
    kernel_y = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    kernel_x = kernel_y.T
    observed_y = correlate(patches, kernel_y[None, None], mode="nearest") / 8
    observed_x = correlate(patches, kernel_x[None, None], mode="nearest") / 8
    gradient = 0.5 * (
        _weighted_zncc_batch(model_y, observed_y, weights)
        + _weighted_zncc_batch(model_x, observed_x, weights)
    )
    return intensity_fraction * intensity + (1 - intensity_fraction) * gradient


def _generalized_cc_surface(model: np.ndarray, region: np.ndarray, radius: int, alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    size = region.shape[0]
    support = model.shape[0]
    offset = (size - support) // 2
    padded = np.zeros_like(region, dtype=np.float64)
    window = _hann2(support)
    padded[offset:offset + support, offset:offset + support] = (model - model.mean()) * window
    observed = (region - region.mean()) * _hann2(size)
    fm = np.fft.fft2(padded)
    fo = np.fft.fft2(observed)
    cross = np.conj(fm) * fo
    magnitude = np.abs(cross)
    epsilon = max(float(np.median(magnitude[magnitude > 0])) if np.any(magnitude > 0) else 1.0, 1.0) * 1e-12
    correlation = np.fft.ifft2(cross / (magnitude + epsilon) ** alpha).real
    surface = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    # IFFT peak d is observed displacement; the returned grid is the raster
    # correction -d, matching every spatial method and the joint solver.
    for iy, correction_y in enumerate(range(-radius, radius + 1)):
        for ix, correction_x in enumerate(range(-radius, radius + 1)):
            surface[iy, ix] = correlation[(-correction_y) % size, (-correction_x) % size]
    scale = float(np.max(np.abs(surface)))
    return surface / scale if scale > 1e-12 else np.full_like(surface, -1.0)


def _quadratic_peak(surface: np.ndarray, peak: tuple[int, int], radius: int) -> tuple[np.ndarray, np.ndarray, bool]:
    integer = np.asarray(peak, dtype=float) - radius
    if any(index == 0 or index == size - 1 for index, size in zip(peak, surface.shape)):
        return integer, np.full((2, 2), np.nan), False
    rows = []
    values = []
    for y in (-1.0, 0.0, 1.0):
        for x in (-1.0, 0.0, 1.0):
            rows.append((y * y, y * x, x * x, y, x, 1.0))
            values.append(surface[peak[0] + int(y), peak[1] + int(x)])
    a, b, c, d, e, _ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]
    hessian = np.asarray([[2 * a, b], [b, 2 * c]], dtype=np.float64)
    eig = np.linalg.eigvalsh(hessian)
    if eig.max() >= -1e-10:
        return integer, hessian, False
    local = -np.linalg.solve(hessian, np.asarray([d, e]))
    if np.max(np.abs(local)) > 0.75:
        return integer, hessian, False
    return integer + local, hessian, True


def _summarize(surface: np.ndarray, radius: int) -> LocalRegistrationResult:
    if not np.isfinite(surface).all():
        raise ValueError("non-finite registration surface")
    peak = np.unravel_index(np.argmax(surface), surface.shape)
    correction, hessian, subpixel = _quadratic_peak(surface, peak, radius)
    suppressed = surface.copy()
    y0, x0 = peak
    suppressed[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2] = -np.inf
    second_index = np.unravel_index(np.argmax(suppressed), suppressed.shape)
    best = float(surface[peak])
    second = float(suppressed[second_index])
    prominence = best - second
    distance = float(np.linalg.norm(np.subtract(second_index, peak)))
    eig = np.linalg.eigvalsh(-hessian) if np.isfinite(hessian).all() else np.asarray([np.nan, np.nan])
    anisotropy = float(eig.max() / eig.min()) if np.all(eig > 1e-12) else float("inf")
    boundary = bool(np.any(np.abs(correction) >= radius - 0.05))
    multimodal = bool(prominence <= max(abs(best) * 1e-3, 1e-6) or distance <= 1.5)
    valid = bool(np.isfinite(correction).all() and best > -1 and not boundary)
    return LocalRegistrationResult(correction, best, float(surface[radius, radius]),
        best - float(surface[radius, radius]), second, prominence, distance, hessian, eig,
        anisotropy, boundary, multimodal, subpixel, valid, np.asarray(surface, np.float32))


def register_local_model(
    model: ArrayLike,
    search_region: ArrayLike,
    *,
    method: str,
    search_radius: int = 4,
    alpha: float = 0.5,
    intensity_fraction: float = 0.75,
    mask: ArrayLike | None = None,
) -> LocalRegistrationResult:
    """Measure one correction using a common grid and subpixel policy."""
    model_array = np.asarray(model, dtype=np.float64)
    region = np.asarray(search_region, dtype=np.float64)
    expected = model_array.shape[0] + 2 * int(search_radius)
    if model_array.ndim != 2 or model_array.shape[0] != model_array.shape[1] or model_array.shape[0] % 2 != 1:
        raise ValueError("model must be square and odd")
    if region.shape != (expected, expected):
        raise ValueError("search region has incompatible shape")
    if method == "pcc":
        surface = _pcc_surface(model_array, region, search_radius)
    elif method == "zncc":
        surface = _zncc_surface(model_array, region, np.ones_like(model_array), search_radius)
    elif method == "masked_zncc":
        weights = cosine_disk_mask(model_array.shape[0]) if mask is None else np.asarray(mask, dtype=np.float64)
        if weights.shape != model_array.shape or np.min(weights) < 0:
            raise ValueError("invalid registration mask")
        surface = _zncc_surface(model_array, region, weights, search_radius)
    elif method == "generalized_cc":
        surface = _generalized_cc_surface(model_array, region, search_radius, alpha)
    elif method == "gradient_zncc":
        weights = cosine_disk_mask(model_array.shape[0]) if mask is None else np.asarray(mask, dtype=np.float64)
        surface = _gradient_surface(model_array, region, weights, search_radius, intensity_fraction)
    else:
        raise ValueError(f"unknown local registration method: {method}")
    return _summarize(surface, search_radius)


def register_local_models(
    models: ArrayLike,
    search_regions: ArrayLike,
    *,
    method: str,
    search_radius: int = 4,
    alpha: float = 0.5,
    intensity_fraction: float = 0.75,
    mask: ArrayLike | None = None,
    batch_size: int = 16,
) -> list[LocalRegistrationResult]:
    """Batch local observations without changing their scalar mathematics.

    Chunking bounds the temporary gradient patch tensor and is particularly
    useful for larger patches at finer sampling. PCC retains its exact scalar
    reference implementation; all other methods share the batch route.
    """
    model_array = np.asarray(models, dtype=np.float64)
    regions = np.asarray(search_regions, dtype=np.float64)
    if model_array.ndim != 3 or regions.ndim != 3 or len(model_array) != len(regions):
        raise ValueError("expected matching batches of 2-D models and regions")
    if method == "pcc":
        return [register_local_model(m, r, method=method, search_radius=search_radius,
            alpha=alpha, intensity_fraction=intensity_fraction, mask=mask)
            for m, r in zip(model_array, regions, strict=True)]
    if method == "generalized_cc":
        # NumPy's many very small 41x41 batched FFTs are slower than its
        # scalar FFT path. Keep the measured faster kernel while sharing all
        # control/subpixel logic.
        return [register_local_model(m, r, method=method, search_radius=search_radius,
            alpha=alpha, intensity_fraction=intensity_fraction, mask=mask)
            for m, r in zip(model_array, regions, strict=True)]
    weights = np.ones(model_array.shape[-2:], dtype=float) if method in ("zncc", "generalized_cc") else (
        cosine_disk_mask(model_array.shape[-1]) if mask is None else np.asarray(mask, dtype=float))
    results = []
    for start in range(0, len(model_array), batch_size):
        mm = model_array[start:start + batch_size]
        rr = regions[start:start + batch_size]
        if method != "generalized_cc":
            patches = np.lib.stride_tricks.sliding_window_view(rr, mm.shape[-2:], axis=(-2, -1))[:, ::-1, ::-1]
            total = float(weights.sum())
            model_centred = mm - np.sum(mm * weights, axis=(-2, -1), keepdims=True) / total
            patch_centred = patches - np.sum(patches * weights, axis=(-2, -1), keepdims=True) / total
            def batch_corr(model_values, patch_values):
                numerator = np.sum(weights * model_values[:, None, None] * patch_values, axis=(-2, -1))
                denominator = np.sqrt(
                    np.sum(weights * model_values * model_values, axis=(-2, -1))[:, None, None]
                    * np.sum(weights * patch_values * patch_values, axis=(-2, -1)))
                return np.divide(numerator, denominator, out=np.full_like(numerator, -1), where=denominator > 1e-12)
            intensity = batch_corr(model_centred, patch_centred)
            if method in ("zncc", "masked_zncc"):
                surfaces = intensity
            elif method == "gradient_zncc":
                ky = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
                my = correlate(mm, ky[None], mode="nearest") / 8
                mx = correlate(mm, ky.T[None], mode="nearest") / 8
                py = correlate(patches, ky[None, None, None], mode="nearest") / 8
                px = correlate(patches, ky.T[None, None, None], mode="nearest") / 8
                my -= np.sum(my * weights, axis=(-2, -1), keepdims=True) / total
                mx -= np.sum(mx * weights, axis=(-2, -1), keepdims=True) / total
                py -= np.sum(py * weights, axis=(-2, -1), keepdims=True) / total
                px -= np.sum(px * weights, axis=(-2, -1), keepdims=True) / total
                gradient = 0.5 * (batch_corr(my, py) + batch_corr(mx, px))
                surfaces = intensity_fraction * intensity + (1 - intensity_fraction) * gradient
            else:
                raise ValueError(f"unknown local registration method: {method}")
        results.extend(_summarize(surface, search_radius) for surface in surfaces)
    return results


def _quadratic_fit_rows(values: np.ndarray):
    """Batched quadratic-surface fit over the 3x3 neighbourhood.

    values: (M, 3, 3) in (y, x) order with (y, x) in {-1, 0, 1}.  Returns
    (a, b, c, d, e) coefficient arrays matching the per-row lstsq parameter
    order ((y^2, yx, x^2, y, x, 1)).  The normal-equation closed form agrees
    with the SVD solve to relative ~1e-15 (design matrix is fixed; b, d, e
    decouple exactly).
    """
    values = np.asarray(values, dtype=np.float64)
    wy2 = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    wx2 = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    wxy = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]])
    wy1 = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    wx1 = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    s1 = values.sum(axis=(-2, -1))
    wy2v = (values * wy2).sum(axis=(-2, -1))
    wx2v = (values * wx2).sum(axis=(-2, -1))
    wxyv = (values * wxy).sum(axis=(-2, -1))
    wy1v = (values * wy1).sum(axis=(-2, -1))
    wx1v = (values * wx1).sum(axis=(-2, -1))
    inv3 = np.linalg.inv(np.array([[6.0, 4.0, 6.0], [4.0, 6.0, 6.0], [6.0, 6.0, 9.0]]))
    acf = np.stack((wy2v, wx2v, s1), axis=1) @ inv3.T
    return acf[:, 0], wxyv / 4.0, acf[:, 1], wy1v / 6.0, wx1v / 6.0


def _summarize_vectorized(host: np.ndarray, radius: int) -> list[LocalRegistrationResult]:
    """Batched _summarize over (N, 2r+1, 2r+1) float32 surfaces.

    Vectorizes argmax, 3x3 suppression, second peak, Hessian eigenvalues and
    the soft decisions; the quadratic fit uses _quadratic_fit_rows.  All
    per-row values match the scalar _summarize to the fit's ~1e-15 relative
    agreement (far below any reportable correction precision).
    """
    host = np.asarray(host)
    n, h, w = host.shape
    if n == 0:
        return []
    if min(h, w) < 3:
        return [_summarize(surface, radius) for surface in host]
    flat = host.reshape(n, -1)
    if not np.isfinite(flat).all():
        raise ValueError('non-finite registration surface')
    pf = np.argmax(flat, axis=1)
    py = (pf // w).astype(np.int64)
    px = (pf - py * w).astype(np.int64)
    integer = np.stack((py, px), axis=1).astype(np.float64) - float(radius)
    boundary_peak = (py == 0) | (py == h - 1) | (px == 0) | (px == w - 1)
    rows = np.arange(n)[:, None, None]
    cyy = np.clip(py, 1, h - 2)[:, None, None]
    cxx = np.clip(px, 1, w - 2)[:, None, None]
    oy = np.array([-1, 0, 1], dtype=np.int64)[None, :, None]
    ox = np.array([-1, 0, 1], dtype=np.int64)[None, None, :]
    vals = host[rows, cyy + oy, cxx + ox]
    a, b, c, d, e = _quadratic_fit_rows(vals)
    hessian = np.stack((np.stack((2 * a, b), 1), np.stack((b, 2 * c), 1)), 1)
    eig = np.linalg.eigvalsh(hessian)
    use_int_only = boundary_peak | (eig.max(axis=1) >= -1e-10)
    okmask = ~use_int_only
    local = np.zeros((n, 2))
    if okmask.any():
        h_ok = hessian[okmask]
        rhs_ok = np.stack((d[okmask], e[okmask]), 1)[:, :, None]
        local_ok = -np.linalg.solve(h_ok, rhs_ok).squeeze(-1)
        local[okmask] = local_ok
        far = np.flatnonzero(okmask)[np.max(np.abs(local_ok), axis=1) > 0.75]
        use_int_only[far] = True
    correction = integer + np.where(use_int_only[:, None], 0.0, local)
    subpixel = ~use_int_only
    hessian = np.where(boundary_peak[:, None, None], np.nan, hessian)
    suppressed = np.where(np.isfinite(host), host, -np.inf).copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            m = (py + dy >= 0) & (py + dy <= h - 1) & (px + dx >= 0) & (px + dx <= w - 1)
            if m.any():
                suppressed[m, py[m] + dy, px[m] + dx] = -np.inf
    pf2 = np.argmax(suppressed.reshape(n, -1), axis=1)
    sy2 = (pf2 // w).astype(np.int64)
    sx2 = (pf2 - sy2 * w).astype(np.int64)
    best = host[np.arange(n), py, px].astype(np.float64)
    second = suppressed[np.arange(n), sy2, sx2].astype(np.float64)
    prominence = best - second
    distance = np.sqrt(((sy2 - py) ** 2 + (sx2 - px) ** 2)).astype(np.float64)
    eig2 = np.linalg.eigvalsh(-hessian)
    eig2 = np.where(~np.isfinite(hessian).all(axis=(-2, -1))[:, None], np.nan, eig2)
    anisotropy = np.where((eig2 > 1e-12).all(axis=1), eig2.max(axis=1) / eig2.min(axis=1), np.inf)
    boundary = (np.abs(correction) >= radius - 0.05).any(axis=1)
    multimodal = (prominence <= np.maximum(np.abs(best) * 1e-3, 1e-6)) | (distance <= 1.5)
    valid = np.isfinite(correction).all(axis=1) & (best > -1) & ~boundary
    zero = host[:, radius, radius].astype(np.float64)
    results = []
    for i in range(n):
        results.append(LocalRegistrationResult(
            correction[i], float(best[i]), float(zero[i]), float(best[i]) - float(zero[i]),
            float(second[i]), float(prominence[i]), float(distance[i]),
            np.asarray(hessian[i]), np.asarray(eig2[i]), float(anisotropy[i]),
            bool(boundary[i]), bool(multimodal[i]), bool(subpixel[i]), bool(valid[i]),
            np.asarray(host[i], np.float32)))
    return results
