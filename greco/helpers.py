"""Shared robust statistics, masks, angular compensation and projection helpers."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np

from .local_projection import project_local_volume_patch
from .selection import project_points_aligned_axis0
from .tomography import gpu_reconstruct_wbp_batch


def geometric_median(points, iterations=64):
    p = np.asarray(points, float)
    x = np.median(p, axis=0)
    for _ in range(iterations):
        d = np.linalg.norm(p - x, axis=1)
        new = np.average(p, axis=0, weights=1 / np.maximum(d, 1e-6))
        if np.linalg.norm(new - x) < 1e-4:
            return new
        x = new
    return x


def consensus(votes):
    p = np.asarray(votes, float)
    x = geometric_median(p)
    d = np.linalg.norm(p - x, axis=1)
    scale = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-6
    keep = d <= np.median(d) + 3 * scale
    if keep.sum() >= 3:
        x = geometric_median(p[keep])
        d = np.linalg.norm(p - x, axis=1)
    return x, keep, d


def jackknife_max(votes):
    votes = np.asarray(votes, float)
    if len(votes) < 4:
        return None
    full = consensus(votes)[0]
    estimates = [consensus(np.delete(votes, i, axis=0))[0] for i in range(len(votes))]
    return float(np.max(np.linalg.norm(np.asarray(estimates) - full, axis=1)))


def soft_spherical_mask(size=33, radius=14., edge=3.):
    c = (size - 1) / 2
    z, y, x = np.ogrid[:size, :size, :size]
    r = np.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2)
    mask = np.ones((size, size, size), np.float32)
    mask[r >= radius] = 0
    transition = (r > radius - edge) & (r < radius)
    mask[transition] = .5 * (1 + np.cos(np.pi * (r[transition] - (radius - edge)) / edge))
    return mask


_impulse_projection_cache = {}


def _impulse_projections(angles, size):
    """Impulse-response projections per angle, cached for the process.

    The projections depend only on the angle values and the support size
    (a fixed impulse at the cube centre), never on the view masks, so they
    are computed once per (angles, size) and reused by every
    transfer_cache_for_supports call in the process.  project_local_volume_patch
    is deterministic (scipy float64), so the cached values are bit-identical
    to recomputing them.
    """
    key = (tuple(float(a) for a in np.asarray(angles, float)), int(size))
    cached = _impulse_projection_cache.get(key)
    if cached is not None:
        return cached
    impulse = np.zeros((size, size, size), np.float32)
    impulse[size // 2, size // 2, size // 2] = 1
    projections = np.asarray([project_local_volume_patch(impulse, (0., 0., 0.), float(a),
                                     observation_radius=size // 2, model_radius=size // 2)
                              for a in np.asarray(angles, float)], np.float32)
    _impulse_projection_cache[key] = projections
    return projections


_TRANSFER_CACHE = {}
_TRANSFER_CACHE_MAX = 8192
_TRANSFER_CACHE_MAX_BYTES = 8192 * 33**3 * np.dtype(np.complex64).itemsize


def _fft_transfers(part):
    keys, psfs = part
    cache = {}
    # Shifted impulse-response PSFs are confined to the central Y plane.
    # Batch the two nontrivial NumPy FFT axes, keeping independent output arrays.
    if (isinstance(psfs, np.ndarray) and psfs.ndim == 4
            and psfs.dtype == np.float32 and psfs.flags.c_contiguous
            and len(keys) == len(psfs) and len(keys) > 0
            and len(set(psfs.shape[1:])) == 1 and psfs.shape[1] % 2 == 1):
        middle = psfs.shape[2] // 2
        planes = psfs[:, :, middle, :]
        if (not np.any(psfs[:, :, :middle, :].view(np.uint32))
                and not np.any(psfs[:, :, middle + 1:, :].view(np.uint32))
                and np.isfinite(planes).all()
                and not np.any((planes == 0) & np.signbit(planes))
                and np.max(np.abs(planes)) <= np.finfo(np.float32).max / psfs[0].size):
            spectra = np.fft.fftn(np.fft.ifftshift(planes, axes=(-2, -1)), axes=(-2, -1))
            if spectra.dtype == np.complex64 and spectra.flags.c_contiguous:
                bits = spectra.view(np.uint32)
                fragile = ((bits & np.uint32(0x7f800000)) == 0) & (bits != 0)
                # The omitted axis can change signs of zero when FFT
                # intermediates underflow. Keep those spectra on the full FFT.
                if not np.any(fragile):
                    scales = np.maximum(np.max(np.abs(spectra), axis=(-2, -1)), 1e-12)
                    normalized = spectra / scales[:, None, None]
                    for key, plane in zip(keys, normalized, strict=True):
                        cache[key] = np.broadcast_to(plane[:, None, :], psfs.shape[1:]).copy()
                    return cache
    for key, psf in zip(keys, psfs, strict=True):
        transfer = np.fft.fftn(np.fft.ifftshift(psf))
        cache[key] = transfer / max(float(np.max(np.abs(transfer))), 1e-12)
    return cache


def _transfer_fft_workers():
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        assigned = int(os.environ.get('SLURM_CPUS_PER_TASK', available))
    except ValueError:
        assigned = 1
    return max(1, min(4, available, assigned))


def _compute_transfers(angles, keys, size, batch_size):
    projections = _impulse_projections(angles, size)
    cache = {}
    workers = _transfer_fft_workers() if len(keys) >= 32 else 1
    context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    with context as pool:
        for start in range(0, len(keys), batch_size):
            part = keys[start:start + batch_size]
            valid = np.zeros((len(part), len(angles)), bool)
            for row, key in enumerate(part):
                valid[row, list(key)] = True
            stacks = np.broadcast_to(projections[None], (len(part),) + projections.shape)
            psfs = gpu_reconstruct_wbp_batch(stacks, angles, valid_bt=valid, y_chunk=8)
            if pool is None or len(part) < 32:
                cache.update(_fft_transfers((part, psfs)))
            else:
                # Finish every CPU chunk before starting another GPU WBP batch.
                width = max(1, (len(part) + workers - 1) // workers)
                groups = [(part[i:i + width], psfs[i:i + width])
                          for i in range(0, len(part), width)]
                for completed in pool.map(_fft_transfers, groups):
                    cache.update(completed)
    return cache


def transfer_cache_for_supports(angles, supports, size=33, batch_size=128):
    """Reuse exact ordered-angle transfers across stages within bounded RAM.

    Recent bulk requests take priority, retaining smaller supports first when
    the request exceeds the budget. Sparse boundary supports are more likely
    to stay unchanged as the tilt range grows. One-off raw misses bypass this
    memo, as before. Returned mappings retain every requested operator even
    when only part of that mapping fits the process cache.
    """
    angles = np.asarray(angles, float)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError('angles must be finite and one-dimensional')
    keys = sorted({tuple(np.flatnonzero(np.asarray(mask, bool))) for mask in supports})
    if any(not key for key in keys):
        raise ValueError('empty angular support')
    if len(keys) < 16:
        return _compute_transfers(angles, keys, size, batch_size)
    # Keep view order, duplicate angles and signed-zero bits. Unused views
    # cannot affect this PSF; the supported-angle sequence and size define it.
    physical = {key: (int(size), angles[list(key)].tobytes()) for key in keys}
    representatives = {physical[key]: key for key in keys}
    hits = {name: _TRANSFER_CACHE[name] for name in representatives
            if name in _TRANSFER_CACHE}
    missing = [key for name, key in representatives.items() if name not in hits]
    fresh = _compute_transfers(angles, missing, size, batch_size) if missing else {}
    values = {name: hits[name] if name in hits else fresh[key]
              for name, key in representatives.items()}
    result = {key: values[physical[key]] for key in keys}
    # Admit only after successful computation. Keep every requested reference
    # alive in result while replacing the bounded process-level memo.
    priority = sorted(values, key=lambda name: (len(name[1]), name))
    candidates = [(name, values[name]) for name in priority]
    candidates.extend((name, value) for name, value in reversed(_TRANSFER_CACHE.items())
                      if name not in values)
    retained = {}; payload = 0
    for name, value in candidates:
        if len(retained) >= _TRANSFER_CACHE_MAX:
            break
        if payload + value.nbytes <= _TRANSFER_CACHE_MAX_BYTES:
            retained[name] = value
            payload += value.nbytes
    _TRANSFER_CACHE.clear()
    _TRANSFER_CACHE.update(retained)
    return result


def projected_centres(positions, angles, size):
    return np.stack([project_points_aligned_axis0(positions, a, size) for a in angles], axis=1)


def rerank_tier(indices, scores, count, excluded=()):
    excluded = set(np.asarray(excluded, int).tolist())
    scores = np.asarray(scores)
    eligible = np.asarray([int(i) for i in np.asarray(indices, int)
                           if int(i) not in excluded and np.isfinite(scores[i])], dtype=np.int64)
    if count <= 0 or not len(eligible):
        return np.empty(0, dtype=np.int32)
    order = np.argsort(scores[eligible], kind='stable')[::-1]
    return eligible[order[:count]].astype(np.int32)
