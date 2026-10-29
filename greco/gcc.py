# Greco optimizer — implementation of the GRECO algorithm
# GRECO: Geometric Registration of Electron Cryotomography Observations
# Copyright (C) 2026 Carlos Fernández Rodríguez
#
# This file is part of Greco.
#
# Greco is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3 only.
#
# Greco is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Greco. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Greco in-process generalized-CC (GCC) joint refinement.

Port of Greco scripts/refine_spa_joint_local_registration_bin8.py with the
hot path made GPU-resident and the per-pass file I/O removed:

  * the cumulative stack and its per-pass re-materialization stay on the GPU;
  * subpixel 33^3 patch extraction is one batched GPU map_coordinates;
  * leave-one-out WBP, angular-transfer correction, local projection and the
    GCC registration all run on the device (validated kernels);
  * the per-tilt search regions are extracted in one batched GPU call;
  * the joint solve, local-containment quarantine and stop rules are the
    unchanged CPU contract; only a compact in-memory history is kept.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import map_coordinates as _cpu_map_coordinates


# ---------------------------------------------------------------------------
# GPU-resident helpers
# ---------------------------------------------------------------------------

def _gpu():
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    return cp, map_coordinates


def materialize_cumulative_stack_gpu(original, cumulative_shifts_yx):
    """GPU re-materialization: immutable original + total cumulative shifts.

    Same contract as stack_materialization.materialize_cumulative_stack
    (order-1, constant edge fill).  ``original`` must be a device array.
    """
    cp, map_coordinates = _gpu()
    shifts = np.asarray(cumulative_shifts_yx, float)
    if shifts.shape != (len(original), 2):
        raise ValueError('cumulative shifts must have shape (n_tilts, 2)')
    t, y, x = original.shape
    yy, xx = np.mgrid[0:y, 0:x].astype(np.float32)
    yy = cp.asarray(np.broadcast_to(yy[None], (t, y, x)))
    xx = cp.asarray(np.broadcast_to(xx[None], (t, y, x)))
    cval = float(cp.mean(original).get())
    # same contract as the CPU driver: ndi_shift(image, delta) moves content
    # BY +delta, i.e. output[i] = input[i - delta] -> coords = grid - shift
    coords = cp.stack((
        cp.broadcast_to(cp.arange(t, dtype=cp.float32)[:, None, None], (t, y, x)),
        yy - cp.asarray(shifts[:, 0], cp.float32)[:, None, None],
        xx - cp.asarray(shifts[:, 1], cp.float32)[:, None, None],
    ))
    return cp.asarray(map_coordinates(original, coords, order=1, mode='constant', cval=cval), cp.float32)


def subpixel_patch_stacks_gpu(current, centres, valid, radius=16):
    """GPU equivalent of solver.subpixel_patch_stacks.

    current: device (T, Y, X) float32.
    centres: (L, T, 2) float64 projected (y, x) centres.
    valid:   (L, T) bool.

    Returns device (L, T, 2r+1, 2r+1) float32 normalized patches.
    Invalid views are sampled at the detector centre (masked downstream).
    """
    cp, map_coordinates = _gpu()
    t, y, _ = current.shape
    centres = np.asarray(centres, float)
    valid = np.asarray(valid, bool)
    if centres.shape != (len(valid), t, 2):
        raise ValueError('centres/valid shape mismatch')
    extraction = centres.copy()
    # invalid views (and, defensively, views whose patch would leave the
    # detector support) are sampled at the detector centre; they never vote
    # (downstream 'inside' filter) and the joint solver keeps the margin
    extraction[~valid] = (y - 1) / 2.0
    out_of_support = (np.any(extraction - radius < 0, axis=2)
                      | np.any(extraction + radius > y - 1, axis=2))
    extraction[out_of_support] = (y - 1) / 2.0
    offy, offx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    n = 2 * radius + 1
    l = len(valid)
    view_idx = np.broadcast_to(np.arange(t)[None, :], (l, t)).reshape(l, t, 1, 1).astype(np.float32)
    cy = extraction[:, :, 0][:, :, None, None].astype(np.float32)
    cx = extraction[:, :, 1][:, :, None, None].astype(np.float32)
    coords = cp.asarray(np.stack((
        np.broadcast_to(view_idx, (l, t, n, n)),
        cy + offy,
        cx + offx,
    )))
    patches = cp.asarray(map_coordinates(current, coords, order=1, mode='constant'), cp.float32)
    mean = patches.mean(axis=(-2, -1), keepdims=True)
    std = patches.std(axis=(-2, -1), keepdims=True)
    if float(cp.min(std[cp.asarray(valid).reshape(l, t, 1, 1) & cp.ones((1, 1), cp.bool_)]).get()) < 1e-6:
        raise ValueError('constant landmark patch')
    safe = cp.where(std < 1e-6, cp.float32(1.0), std)
    return ((patches - mean) / safe).astype(cp.float32, copy=False)


def transfer_cache_gpu(cache_cpu):
    cp, _ = _gpu()
    return {key: cp.asarray(value) for key, value in cache_cpu.items()}


def _loo_support_table(valid):
    """Build each leave-one-out key once per distinct validity row."""
    patterns = {}
    rows = []
    supports = {}
    for mask in valid:
        signature = mask.tobytes()
        lookup = patterns.get(signature)
        if lookup is None:
            ids = tuple(np.flatnonzero(mask))
            lookup = [None] * valid.shape[1]
            for offset, target in enumerate(ids):
                key = ids[:offset] + ids[offset + 1:]
                lookup[target] = key
                if key not in supports:
                    support = np.zeros(valid.shape[1], bool)
                    support[list(key)] = True
                    supports[key] = support
            lookup = tuple(lookup)
            patterns[signature] = lookup
        rows.append(lookup)
    return rows, supports


def _angular_operators_batched(transfers, keys, wiener_lambda, max_inverse_gain,
                               lowpass_fraction, chunk=1024):
    """Precompute independent operators; retain lazy lookup on smaller GPUs."""
    from .angular_transfer_gpu import angular_transfer_operator_gpu
    cp, _ = _gpu()
    if not keys:
        return {}
    free, _ = cp.cuda.runtime.memGetInfo()
    per_key = transfers[keys[0]].nbytes
    # Do not introduce an eager-allocation memory regression over lazy lookup.
    if len(keys) * per_key > free * .6:
        return {}
    chunk = min(chunk, max(1, int(free * .15 // (per_key * 6))))
    operators = {}
    for start in range(0, len(keys), chunk):
        part = keys[start:start + chunk]
        batch = cp.stack([transfers[key] for key in part])
        inverse = angular_transfer_operator_gpu(batch, wiener_lambda,
                                                max_inverse_gain, lowpass_fraction)
        operators.update((key, inverse[i]) for i, key in enumerate(part))
    return operators


# ---------------------------------------------------------------------------
# raw-source (maximum-FOV) sampling
#
# The source stack is the CTF+reliability planes in the RAW MC raster: no
# frozen shift and no axis rotation baked in.  The frozen shift, the axis
# rotation and the current cumulative shift are applied ON THE FLY at sample
# time, so no edge fill is ever baked into the data and features that sit
# near the detector edge remain available to the loop at every pass:
#
#   aligned_final[p] = Q[c0 + M(p - S_c - c0) - S_f]
#
# with M the inverse-map matrix of skimage.transform.rotate
# (out[p] = in[c0 + M(p - c0)]), i.e. M = [[cos, sin], [-sin, cos]] for the
# axis angle.  (Verified: wrong-sign control gives CC ~ 0 in the overlap.)
# Raw mode uses plain CuPy order-1 sampling (it is a new protocol variant,
# not a reproduction of the reference driver).
# ---------------------------------------------------------------------------

def raw_affine(axis_deg, size):
    a = np.deg2rad(float(axis_deg))
    c0 = (size - 1) / 2.0
    return (float(np.cos(a)), float(np.sin(a)), float(-np.sin(a)),
            float(np.cos(a)), c0)


def raw_sample_bases(centres, cumulative, frozen, axis_deg, size):
    """Raw-raster sample bases for aligned-raster centres.

    centres: (L, T, 2); cumulative: (T, 2); frozen: (T, 2).
    Returns (base_y, base_x), each (L, T).
    """
    if np.ndim(axis_deg):
        a=np.deg2rad(np.asarray(axis_deg,float))
        if a.shape!=(centres.shape[1],) or not np.isfinite(a).all():raise ValueError('Invalid per-view raster angles')
        M00=np.cos(a);M01=np.sin(a);M10=-np.sin(a);M11=np.cos(a);c0=(size-1)/2.
    else:
        M00, M01, M10, M11, c0 = raw_affine(axis_deg, size)
    cumulative = np.asarray(cumulative, float)
    frozen = np.asarray(frozen, float)
    p = np.asarray(centres, float) - cumulative[None, :, :]
    py = p[:, :, 0] - c0
    px = p[:, :, 1] - c0
    by = M00 * py + M01 * px + c0 - frozen[:, 0][None, :]
    bx = M10 * py + M11 * px + c0 - frozen[:, 1][None, :]
    return by, bx


def _rotated_offsets(radius, axis_deg):
    if np.ndim(axis_deg):
        values=[_rotated_offsets(radius,float(a)) for a in axis_deg]
        return (np.stack([v[0] for v in values]),np.stack([v[1] for v in values]),
                np.array([v[2] for v in values]),np.array([v[3] for v in values]))
    M00, M01, M10, M11, _ = raw_affine(axis_deg, 0)
    offy, offx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(float)
    roty = (M00 * offy + M01 * offx).astype(np.float32)
    rotx = (M10 * offy + M11 * offx).astype(np.float32)
    ext_y = (abs(M00) + abs(M01)) * radius
    ext_x = (abs(M10) + abs(M11)) * radius
    return roty, rotx, ext_y, ext_x


def raw_search_support(base_y, base_x, extent, axis_deg, size):
    """True where the ENTIRE rotated raw search footprint is inside the detector.

    The sampled window is ``base + (M00*oy + M01*ox, M10*oy + M11*ox)`` for
    integer ``oy, ox`` in ``[-extent, extent]``; the linear extrema of that
    footprint are attained at the square corners, so the bounding test
    ``base +/- (|M00|+|M01|)*extent`` / ``base +/- (|M10|+|M11|)*extent`` is
    exact for "every sampled coordinate lies in [0, size-1]".

    base_y/base_x: raw-raster bases (raw_sample_bases).  This is the correct
    raw-mode gate for GCC search regions: the aligned-raster box test on the
    projected centres does NOT guarantee raw support, because the window is
    rotated (footprint half-width grows from extent to ~1.095*extent at
    -84.25 deg) and the raw base carries the frozen shift.
    """
    base_y = np.asarray(base_y, float)
    base_x = np.asarray(base_x, float)
    _, _, ext_y, ext_x = _rotated_offsets(extent, axis_deg)
    return ((base_y - ext_y >= 0) & (base_y + ext_y <= size - 1)
            & (base_x - ext_x >= 0) & (base_x + ext_x <= size - 1))


_raw_patch_coordinates_kernel = None


def _raw_patch_coordinates_gpu(base_y, base_x, roty, rotx):
    """Expand compact host geometry on device with NumPy float32 add rounding."""
    import cupy as cp
    global _raw_patch_coordinates_kernel
    if _raw_patch_coordinates_kernel is None:
        _raw_patch_coordinates_kernel = cp.ElementwiseKernel(
            'raw float32 by, raw float32 bx, raw float32 dy, raw float32 dx, '
            'int32 tilts, int32 n, int32 per_view, int64 count', 'raw float32 coords',
            r"""
            long long row = i / ((long long)n * n);
            long long pixel = i % ((long long)n * n);
            if(per_view)pixel += (row % tilts) * n * n;
            coords[i] = (float)(row % tilts);
            float y, x;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(by[row]), "f"(dy[pixel]));
            asm("add.rn.f32 %0, %1, %2;" : "=f"(x) : "f"(bx[row]), "f"(dx[pixel]));
            coords[count + i] = y;
            coords[2 * count + i] = x;
            """, 'greco_raw_patch_coordinates_views', options=('--fmad=false',))
    landmarks, tilts = base_y.shape
    n = roty.shape[-1]
    per_view=int(roty.ndim==3)
    coords = cp.empty((3, landmarks, tilts, n, n), cp.float32)
    count = landmarks * tilts * n * n
    if count:
        _raw_patch_coordinates_kernel(cp.asarray(np.asarray(base_y, np.float32)),
            cp.asarray(np.asarray(base_x, np.float32)), cp.asarray(roty), cp.asarray(rotx),
            tilts, n, per_view, count, coords, size=count)
    return coords



def raw_patch_stacks_gpu(original_dev, base_y, base_x, valid, radius, axis_deg):
    """Observation patches sampled from the raw-raster source on the fly.

    base_y/base_x: (L, T) raw-raster sample bases (raw_sample_bases).
    Returns (device (L, T, 2r+1, 2r+1) z-scored patches, raw_inside (L, T)
    bool).  Views whose rotated patch footprint leaves the raw frame are
    sampled at the detector centre (placeholders) and reported in
    raw_inside; callers must mask them out of the LOO support and the votes.
    """
    cp, map_coordinates = _gpu()
    t, y, _ = original_dev.shape
    valid = np.asarray(valid, bool)
    by, bx = np.asarray(base_y, float), np.asarray(base_x, float)
    roty, rotx, ext_y, ext_x = _rotated_offsets(radius, axis_deg)
    inside = (valid & (by - ext_y >= 0) & (by + ext_y <= y - 1)
              & (bx - ext_x >= 0) & (bx + ext_x <= y - 1))
    bad = ~inside
    by = np.where(bad, (y - 1) / 2.0, by)
    bx = np.where(bad, (y - 1) / 2.0, bx)
    n = 2 * radius + 1
    l = len(valid)
    coords = _raw_patch_coordinates_gpu(by, bx, roty, rotx)
    patches = cp.asarray(map_coordinates(original_dev, coords, order=1,
                                         mode='constant', cval=0.0), cp.float32)
    mean = patches.mean(axis=(-2, -1), keepdims=True)
    std = patches.std(axis=(-2, -1), keepdims=True)
    # A flat local patch supplies no registration evidence. Mask that
    # observation just like an out-of-frame footprint, preserving other views.
    inside &= cp.asnumpy((cp.isfinite(std) & (std >= 1e-6)).reshape(l,t))
    safe = cp.where(std < 1e-6, cp.float32(1.0), std)
    return ((patches - mean) / safe).astype(cp.float32, copy=False), inside


def raw_search_regions_gpu(original_dev, target, base_yx, extent, axis_deg):
    """Search windows on the raw-raster source for one target plane.

    base_yx: (R, 2) raw-raster bases of the target centres.
    """
    cp, map_coordinates = _gpu()
    if np.ndim(axis_deg):axis_deg=float(np.asarray(axis_deg)[target])
    roty, rotx, _, _ = _rotated_offsets(extent, axis_deg)
    r = len(base_yx)
    n = 2 * extent + 1
    cy = base_yx[:, 0][:, None, None].astype(np.float32) + roty
    cx = base_yx[:, 1][:, None, None].astype(np.float32) + rotx
    coords = cp.stack((
        cp.broadcast_to(cp.full(r, float(target), cp.float32)[:, None, None], (r, n, n)),
        cp.broadcast_to(cp.asarray(cy), (r, n, n)),
        cp.broadcast_to(cp.asarray(cx), (r, n, n)),
    )).astype(np.float32)
    return cp.asarray(map_coordinates(original_dev, coords, order=1,
                                      mode='constant', cval=0.0), cp.float32)


def materialize_raw_stack_gpu(original_dev, plane_idx, cumulative, frozen,
                              axis_deg, chunk=8):
    """Aligned-raster materialization from the raw-raster source, in chunks.

    aligned[j][p] = original_dev[plane_idx[j]][c0 + M(p - S_c_j - c0) - S_f_j]

    Pixels outside the raw frame are filled with that plane's mean, matching
    the treated-path convention without coupling planes through a stack-wide
    fill value.
    """
    cp, map_coordinates = _gpu()
    t, y, x = original_dev.shape
    if np.ndim(axis_deg):
        axis=np.asarray(axis_deg,float)
        if axis.shape!=(len(plane_idx),) or not np.isfinite(axis).all():raise ValueError('Invalid materialization raster angles')
        out=cp.empty((len(plane_idx),y,x),cp.float32)
        yy,xx=np.mgrid[:y,:x].astype(np.float32);c0=(y-1)/2
        a=np.deg2rad(axis);co=np.cos(a);si=np.sin(a)
        cumulative=np.asarray(cumulative,float);frozen=np.asarray(frozen,float)
        for first in range(0,len(plane_idx),chunk):
            end=min(first+chunk,len(plane_idx))
            py=yy[None]-c0-cumulative[first:end,0,None,None]
            px=xx[None]-c0-cumulative[first:end,1,None,None]
            ry=co[first:end,None,None]*py+si[first:end,None,None]*px+c0-frozen[first:end,0,None,None]
            rx=-si[first:end,None,None]*py+co[first:end,None,None]*px+c0-frozen[first:end,1,None,None]
            coords=cp.stack((cp.broadcast_to(cp.asarray(plane_idx[first:end],cp.float32)[:,None,None],(end-first,y,x)),cp.asarray(ry),cp.asarray(rx)))
            vals=map_coordinates(original_dev,coords,order=1,mode='constant',cval=0.)
            oob=(coords[1]<0)|(coords[1]>y-1)|(coords[2]<0)|(coords[2]>x-1)
            fill=cp.stack([original_dev[int(k)].mean() for k in plane_idx[first:end]])[:,None,None]
            out[first:end]=cp.where(oob,fill,cp.asarray(vals,cp.float32))
        return out
    M00, M01, M10, M11, c0 = raw_affine(axis_deg, y)
    cumulative = np.asarray(cumulative, float)
    frozen = np.asarray(frozen, float)
    out = cp.empty((len(plane_idx), y, x), cp.float32)
    yy, xx = np.mgrid[0:y, 0:x].astype(np.float32)
    for s in range(0, len(plane_idx), chunk):
        e = min(s + chunk, len(plane_idx))
        py = yy[None] - c0 - cumulative[s:e, 0][:, None, None]
        px = xx[None] - c0 - cumulative[s:e, 1][:, None, None]
        ry = M00 * py + M01 * px + c0 - frozen[s:e, 0][:, None, None]
        rx = M10 * py + M11 * px + c0 - frozen[s:e, 1][:, None, None]
        coords = cp.stack((
            cp.broadcast_to(cp.asarray(plane_idx[s:e].astype(np.float32))[:, None, None],
                            (e - s, y, x)),
            cp.asarray(ry), cp.asarray(rx)))
        vals = map_coordinates(original_dev, coords, order=1,
                               mode='constant', cval=0.0)
        ryc, rxc = cp.asarray(ry), cp.asarray(rx)
        oob = (ryc < 0) | (ryc > y - 1) | (rxc < 0) | (rxc > x - 1)
        fillval = cp.stack([cp.mean(original_dev[int(plane_idx[k])])
                            for k in range(s, e)])[:, None, None]
        out[s:e] = cp.where(oob, fillval, cp.asarray(vals, cp.float32))
    return out.astype(cp.float32, copy=False)


# ---------------------------------------------------------------------------
# CPU joint-solve contract (unchanged from the validated driver)
# ---------------------------------------------------------------------------

def _surface_value(surface, correction, radius):
    coordinate = np.clip(np.asarray(correction, float) + radius, 0, 2 * radius)
    return float(_cpu_map_coordinates(surface, coordinate[:, None], order=1, mode='nearest')[0])


def _surface_values(surfaces, corrections, radius):
    """Vectorized bilinear equivalent of ``_surface_value`` for many rows."""
    surfaces = np.asarray(surfaces)
    coordinates = np.clip(np.asarray(corrections, float) + radius, 0, 2 * radius)
    y0 = np.floor(coordinates[:, 0]).astype(np.intp)
    x0 = np.floor(coordinates[:, 1]).astype(np.intp)
    y1 = np.minimum(y0 + 1, 2 * radius)
    x1 = np.minimum(x0 + 1, 2 * radius)
    wy = coordinates[:, 0] - y0
    wx = coordinates[:, 1] - x0
    row = np.arange(len(surfaces))
    return ((1 - wy) * (1 - wx) * surfaces[row, y0, x0]
            + wy * (1 - wx) * surfaces[row, y1, x0]
            + (1 - wy) * wx * surfaces[row, y0, x1]
            + wy * wx * surfaces[row, y1, x1])


def _solve_observed_tilts(votes, angles, robust_scale, max_nfev, tolerance, workspace=None,
                          backend='cpu'):
    from .solver import solve_joint_votes
    observed_tilts = np.any(np.isfinite(votes).all(axis=2), axis=0)
    if np.sum(observed_tilts) < 3:
        raise ValueError("insufficient observed tilts for joint fit")
    supported_shifts, dr, report = solve_joint_votes(
        votes[:, observed_tilts], angles[observed_tilts], robust_scale, max_nfev, tolerance,
        workspace=workspace, backend=backend)
    shifts = np.zeros((len(angles), 2), dtype=float)
    shifts[observed_tilts] = supported_shifts
    report["observed_tilt_count"] = int(observed_tilts.sum())
    report["unsupported_tilt_indices"] = np.flatnonzero(~observed_tilts).astype(int).tolist()
    return shifts, dr, report


def _subset_agreement(votes, angles, robust_scale, max_nfev, tolerance,
                      backend='cpu'):
    estimates = []
    for parity in (0, 1):
        subset = np.arange(len(votes)) % 2 == parity
        active = subset & (np.sum(np.isfinite(votes).all(axis=2), axis=1) >= 2)
        if np.sum(active) < 2:
            return None
        try:
            shifts, _, _ = _solve_observed_tilts(
                votes[active], angles, robust_scale, max_nfev, tolerance,
                backend=backend)
        except (ValueError, RuntimeError):
            return None
        estimates.append(shifts)
    distance = np.linalg.norm(estimates[0] - estimates[1], axis=1)
    return dict(median=float(np.median(distance)), p90=float(np.quantile(distance, .9)), maximum=float(np.max(distance)))


def _xy_update_is_stable(shifts_yx, median_limit, p90_limit):
    norm = np.linalg.norm(np.asarray(shifts_yx, float), axis=1)
    return bool(np.median(norm) < median_limit and np.quantile(norm, .9) < p90_limit)


def _solve_with_local_containment(votes, valid, classes, positions, angles, size,
                                  search_radius, robust_scale, max_nfev, tolerance,
                                  workspace=None, backend='cpu', containment_margin=20):
    from .support_containment import solve_with_view_containment
    return solve_with_view_containment(votes, valid, classes, positions, angles, size,
        search_radius, robust_scale, max_nfev, tolerance, workspace, backend,
        containment_margin, solve=_solve_observed_tilts)


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------


@contextmanager
def _local_fft_plan_cache():
    """Retain local FFT plans during GCC, then restore the caller's limits."""
    cp, _ = _gpu()
    cache = cp.fft.config.get_plan_cache()
    size, memory = cache.get_size(), cache.get_memsize()
    if size <= 0 or size >= 128 or memory == 0:
        yield
        return
    try:
        cache.set_size(128)
        cache.set_memsize(min(memory, 64 * 1024**2) if memory > 0 else 64 * 1024**2)
        yield
    finally:
        cache.set_size(size)
        cache.set_memsize(memory)


@_local_fft_plan_cache()
def run_gcc(original, angles, initial_tracks, initial_cumulative, *,
            gcc_alpha=.25, search_radius=4, max_passes=60, stop_shift=.10,
            stop_shift_p90=.25, stop_consecutive=3, robust_scale=.2,
            solver_max_nfev=2000, solver_tolerance=1e-6, wiener_lambda=.05,
            solver_backend='cpu',
            max_inverse_gain=4., lowpass_fraction=.8,
            subset_agreement_every=5, resample='exact',
            source='treated', frozen_shifts_yx=None, axis_deg=None,
            trace_planes=None, search_support_diag=None, diag_plane_map=None,
            containment_solver=None, measurement_rms_ratio=None, normal_radius_only=False):
    """Run the validated full-support LOO GCC loop in-process on the GPU.

    original: (T, Y, X) float32 bin8 stack (CPU or GPU); T = len(angles).
    initial_tracks: dict with positions_zyx, valid_views, classes.
    initial_cumulative: (T, 2) float.

    Returns (final_state, manifest): final_state has positions_zyx,
    cumulative_shifts_yx, valid_views, classes; manifest has stop_reason,
    compact per-pass history and runtime.

    resample: how the three resampling steps (cumulative re-
    materialization, patch extraction, search-region sampling) are done
    (treated mode only):
      'exact' (default): GPU kernels replicating the SciPy order-1
             arithmetic bit-for-bit (greco/scipy_resample_gpu.py) ->
             whole pipeline on the GPU AND bit-exact vs the reference.
      'cpu':  SciPy on the CPU, exactly as the reference driver does.
      'fast': plain CuPy map_coordinates (fastest to write, but its
             order-1 kernel differs from SciPy at ~1e-6 and the loop
             amplifies that into a ~2 px drift; do not use for results).
    trace_planes: optional list of plane indices (into ``original``) to
             record the per-pass cumulative shift of in each history entry
             (``trace_cumulative``); diagnostic only, no effect on the fit.

    source: 'treated' (default, the production/reference behaviour: the
    original stack has the frozen shift + axis rotation baked in) or
    'raw' (maximum-FOV: the original stack is the CTF+reliability planes
    in the raw MC raster; frozen_shifts_yx (T, 2) and axis_deg are
    required, and the frozen shift + rotation + cumulative shift are
    applied on the fly at every sample) or
    'raw_materialized_control' (DIAGNOSTIC ONLY, not a protocol: the
    raw-raster stack is materialized to the aligned raster each pass with
    materialize_raw_stack_gpu and sampled with the SAME aligned-raster
    bit-exact path as treated; isolates the on-the-fly raw sampler from
    the raw data itself).

    search_support_diag / diag_plane_map: diagnostic-only raw-mode
    bookkeeping (no effect on the fit).  When search_support_diag is a
    dict and source=='raw', every pass records, per target plane, the
    GCC candidate set under the aligned-raster gate and how many of the
    candidate search windows have their FULL rotated raw footprint
    inside the raw detector (the rest would be sampled partially outside,
    injecting the constant fill into the window).  diag_plane_map maps
    the local target index of this call to the global plane index used
    as the report key.  Accumulates across calls via the shared dict.
    """
    fit_contained = containment_solver or _solve_with_local_containment

    cp, map_coordinates = _gpu()
    started = time.monotonic()
    angles = np.asarray(angles, float)
    if original is None:
        raise ValueError('original stack required')
    if source not in ('treated', 'raw', 'raw_materialized_control'):
        raise ValueError("source must be 'treated', 'raw' or 'raw_materialized_control'")
    if source in ('raw', 'raw_materialized_control'):
        if frozen_shifts_yx is None or axis_deg is None:
            raise ValueError(f"source={source!r} requires frozen_shifts_yx and axis_deg")
        frozen = np.asarray(frozen_shifts_yx, float)
        if frozen.shape != (len(angles), 2):
            raise ValueError('frozen shifts must have shape (n_tilts, 2)')
    else:
        frozen = None
    if resample not in ('exact', 'cpu', 'fast'):
        raise ValueError("resample must be 'exact', 'cpu' or 'fast'")
    original_host = None
    if isinstance(original, np.ndarray):
        original_host = np.asarray(original, np.float32)
        original_dev = cp.asarray(original_host)
    else:
        original_dev = original
    if source == 'treated' and resample in ('exact', 'cpu') and original_host is None:
        raise ValueError(f"resample={resample!r} requires a host (numpy) original stack")
    cumulative = np.asarray(initial_cumulative, float).copy()
    size = original.shape[-1]

    positions = np.asarray(initial_tracks['positions_zyx'], float)
    valid = np.asarray(initial_tracks['valid_views'], bool)
    classes = np.asarray(initial_tracks['classes'])
    if valid.shape != (len(positions), len(angles)):
        raise ValueError('input tracks, stack and angles do not match')

    from .gcc_observations import measure_observations
    from .solver import projection_jacobians
    measurement_context={}
    solver_workspace={}
    history=[]
    stable_passes=0
    stop_reason='max_passes'
    first_measured_rms=None

    for iteration in range(1, max_passes + 1):
        iteration_started = time.monotonic()
        timing = {}
        stamp = time.monotonic()
        evidence=measure_observations(original_host if original_host is not None else original_dev,angles,
            dict(positions_zyx=positions,valid_views=valid,classes=classes),cumulative,
            search_radius=search_radius,gcc_alpha=gcc_alpha,wiener_lambda=wiener_lambda,
            max_inverse_gain=max_inverse_gain,lowpass_fraction=lowpass_fraction,
            source=source,frozen_shifts_yx=frozen,axis_deg=axis_deg,resample=resample,
            search_support_diag=search_support_diag,diag_plane_map=diag_plane_map,
            iteration=iteration,context=measurement_context)
        common_before,common_after=[],[]
        votes=evidence['votes'];score_surfaces=evidence['score_surfaces']
        predicted_rows=evidence['predicted_rows'];prominences=evidence['prominences'];curvatures=evidence['curvatures']
        boundary_count=evidence['boundary_count'];multimodal_count=evidence['multimodal_count'];timing=evidence['timing_s']


        stamp = time.monotonic()
        try:
            shifts, dr, report, votes, revised_valid, active, quarantined, safe, offending = (
                fit_contained(votes, valid, classes, positions, angles, size,
                                              search_radius, robust_scale, solver_max_nfev,
                                              solver_tolerance, solver_workspace,
                                              solver_backend))
        except (ValueError, RuntimeError) as error:
            stop_reason = 'joint_fit_failed'
            history.append(dict(iteration=iteration, error=str(error),
                                valid_observations=int(np.isfinite(votes).all(axis=2).sum()),
                                boundary_observations=boundary_count))
            break
        timing['solve_s'] = time.monotonic() - stamp
        if quarantined:
            valid = revised_valid

        projected_dr = np.einsum('tac,kc->kta', projection_jacobians(angles), dr)
        predicted = shifts[None] - projected_dr
        if predicted_rows:
            row_landmarks = np.fromiter((row[0] for row in predicted_rows), int)
            row_targets = np.fromiter((row[1] for row in predicted_rows), int)
            common_before.extend(row[2] for row in predicted_rows)
            common_after.extend(_surface_values(
                score_surfaces[row_landmarks, row_targets],
                predicted[row_landmarks, row_targets], search_radius).tolist())
        new_positions = positions + dr
        shift_norm = np.linalg.norm(shifts, axis=1)
        landmark_norm = np.linalg.norm(dr, axis=1)
        observed = np.isfinite(votes).all(axis=2)
        projected_landmark_norm = np.linalg.norm(projected_dr, axis=2)[observed]
        raster_stable = _xy_update_is_stable(shifts, stop_shift, stop_shift_p90)
        if measurement_rms_ratio is not None:
            measured=float(report.get('rms_before',np.sqrt(np.mean(votes[np.isfinite(votes).all(2)]**2))))
            if first_measured_rms is None:first_measured_rms=measured
            raster_stable=raster_stable and measured<=measurement_rms_ratio*first_measured_rms
        if normal_radius_only:raster_stable=raster_stable and search_radius==4
        stable_passes = stable_passes + 1 if raster_stable else 0
        compute_subset = subset_agreement_every > 0 and (
            iteration == 1 or iteration % subset_agreement_every == 0 or iteration == max_passes)
        entry = dict(
            iteration=iteration, applied=safe,
            active_landmarks=int(active.sum()),
            valid_observations=int(np.isfinite(votes).all(axis=2).sum()),
            boundary_observations=boundary_count, multimodal_observations=multimodal_count,
            median_shift=float(np.median(shift_norm)), p90_shift=float(np.quantile(shift_norm, .9)),
            max_shift=float(np.max(shift_norm)),
            median_landmark_update=float(np.median(landmark_norm)),
            p90_landmark_update=float(np.quantile(landmark_norm, .9)),
            median_projected_landmark_update=float(np.median(projected_landmark_norm)),
            p90_projected_landmark_update=float(np.quantile(projected_landmark_norm, .9)),
            median_after=float(report.get('median_after', float('nan'))),
            rms_after=float(report.get('rms_after', float('nan'))),
            solver_nfev=int(report.get('nfev', 0)),
            median_peak_prominence=float(np.median(prominences)) if prominences else None,
            median_hessian_curvature=float(np.median(curvatures)) if curvatures else None,
            common_score_before=float(np.mean(common_before)) if common_before else None,
            common_score_after=float(np.mean(common_after)) if common_after else None,
            landmark_subset_agreement=_subset_agreement(votes, angles, robust_scale,
                                                         solver_max_nfev, solver_tolerance,
                                                         solver_backend)
            if compute_subset else None,
            raster_stable=raster_stable,
            stable_passes=stable_passes, quarantined=bool(quarantined),
            containment_removed_pairs=int(report.get("containment_removed_pairs",0)), support_exclusions=quarantined,
            timing_s=timing, elapsed_s=time.monotonic() - iteration_started,
        )
        if not safe:
            stop_reason = 'joint_solution_outside_support_or_shift_limit'
            if trace_planes is not None:
                entry['trace_cumulative'] = {int(p): [float(cumulative[p, 0]),
                                                      float(cumulative[p, 1])]
                                             for p in trace_planes}
            history.append(entry)
            break
        cumulative += shifts
        stamp = time.monotonic()
        positions = new_positions
        timing['apply_s'] = time.monotonic() - stamp
        if trace_planes is not None:
            entry['trace_cumulative'] = {int(p): [float(cumulative[p, 0]),
                                                  float(cumulative[p, 1])]
                                         for p in trace_planes}
        history.append(entry)
        if stable_passes >= stop_consecutive:
            stop_reason = 'converged'
            break

    final_state = dict(positions_zyx=positions, cumulative_shifts_yx=cumulative,
                       initial_cumulative_shifts_yx=np.asarray(initial_cumulative, float),
                       round_correction_shifts_yx=cumulative - np.asarray(initial_cumulative, float),
                       valid_views=valid, classes=classes, angles_deg=angles)
    manifest = dict(stop_reason=stop_reason, passes=len(history),
                    history=history,
                    last=history[-1] if history else None,
                    runtime_s=time.monotonic() - started)
    return final_state, manifest
