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

"""Shared GPU image measurements for fixed and angular geometry fitting."""
import time
import numpy as np
from .gcc import (_gpu, _loo_support_table,
    _angular_operators_batched, raw_sample_bases, raw_patch_stacks_gpu,
    raw_search_support, _rotated_offsets, raw_search_regions_gpu,
    materialize_raw_stack_gpu, materialize_cumulative_stack_gpu,
    subpixel_patch_stacks_gpu, _local_fft_plan_cache)


@_local_fft_plan_cache()
def measure_observations(original, angles, initial_tracks, initial_cumulative, *,
                         search_radius=4, gcc_alpha=.25, wiener_lambda=.05,
                         max_inverse_gain=4., lowpass_fraction=.8,
                         source='raw', frozen_shifts_yx=None, axis_deg=None,
                         resample='exact', search_support_diag=None,
                         diag_plane_map=None, iteration=1, context=None, patch_radius=16):
    """Return image-derived LOO/GCC observations without fitting geometry.

    An optional call-local context retains transfer operators across iterations
    with unchanged angles. A changing-angle caller uses a fresh context.
    """
    if type(patch_radius) is not int or patch_radius<1:raise ValueError("Invalid landmark patch radius")
    patch_size=2*patch_radius+1;patch_scale=patch_radius/16
    cp, map_coordinates = _gpu()
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
        original_dev = context['original_dev'] if context and 'original_dev' in context else cp.asarray(original_host)
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

    from .helpers import soft_spherical_mask, transfer_cache_for_supports, projected_centres
    from .registration import cosine_disk_mask, sample_search_region
    from .local_loo_gpu import BatchedLocalLOOWBPGPU
    from .local_projection_gpu import project_local_volumes_gpu_device
    from .local_registration_gpu import register_local_models_gpu
    from .angular_transfer_gpu import (angular_transfer_operator_gpu,
        apply_angular_transfer_operator_gpu)
    from .solver import subpixel_patch_stacks
    from .stack_materialization import materialize_cumulative_stack
    from .scipy_resample_gpu import shift_stack_scipy_gpu, patch_stack_scipy_gpu

    context={} if context is None else context
    if not context:
        mask3d = cp.asarray(soft_spherical_mask(size=patch_size,radius=14*patch_scale,edge=3*patch_scale), cp.float32)
        mask2d = cosine_disk_mask(size=patch_size,radius=14*patch_scale,edge=3*patch_scale)
        support_rows, support_masks = _loo_support_table(valid)
        support_valid = valid.copy()
        loo_supports = list(support_masks.values())
        # Retain transfers on the host; only their prepared operators need GPU residency.
        transfer_cache = transfer_cache_for_supports(angles, loo_supports, size=patch_size)

        operator_cache = _angular_operators_batched(
            transfer_cache, list(transfer_cache), wiener_lambda,
            max_inverse_gain, lowpass_fraction)

        context.update(original_dev=original_dev,mask3d=mask3d,mask2d=mask2d,support_rows=support_rows,support_masks=support_masks,support_valid=support_valid,transfer_cache=transfer_cache,operator_cache=operator_cache)
    else:
        mask3d=context['mask3d'];mask2d=context['mask2d'];support_rows=context['support_rows'];support_masks=context['support_masks'];support_valid=context['support_valid'];transfer_cache=context['transfer_cache'];operator_cache=context['operator_cache']

    def operator_lookup(key):
        operator = operator_cache.get(key)
        if operator is None:
            operator = angular_transfer_operator_gpu(
                transfer_cache[key], wiener_lambda, max_inverse_gain,
                lowpass_fraction)
            operator_cache[key] = operator
        return operator

    if resample == 'exact' and source == 'treated':
        # per-plane border value = plane mean, exactly as the reference
        # scipy ndi_shift CPU path computes it
        exact_cvals = cp.asarray(
            np.asarray([float(im.mean()) for im in original_host], np.float64))

    def _materialize(cum):
        if source in ('raw', 'raw_materialized_control'):
            # aligned raster = raw MC raster sampled on the fly with
            # rotation + frozen shift + cumulative shift
            return materialize_raw_stack_gpu(original_dev, np.arange(len(cum)),
                                             cum, frozen, axis_deg)
        if resample == 'cpu':
            return materialize_cumulative_stack(original_host, cum)
        if resample == 'fast':
            return materialize_cumulative_stack_gpu(original_dev, cum)
        # 'exact': bit-exact GPU replica of the reference scipy ndi_shift
        return shift_stack_scipy_gpu(original_dev, cp.asarray(cum, np.float64),
                                     exact_cvals)

    current = None if source == 'raw' else _materialize(cumulative)

    timing={}
    stamp=time.monotonic()
    centres = projected_centres(positions, angles, size)
    valid_pass = valid
    if source == 'raw':
        by, bx = raw_sample_bases(centres, cumulative, frozen, axis_deg, size)
        patches, raw_inside = raw_patch_stacks_gpu(
            original_dev, by, bx, valid, patch_radius, axis_deg)
        valid_pass = valid & raw_inside
    elif source == 'raw_materialized_control':
        # DIAGNOSTIC control: the raw-raster stack is already
        # materialized to the aligned raster in ``current`` (see the
        # _materialize branch above), so patches are extracted with the
        # SAME aligned-raster bit-exact path as treated 'exact' mode.
        # This isolates the raw DATA from the on-the-fly raw sampler.
        extraction = centres.copy()
        extraction[~valid] = (size - 1) / 2.0
        patches = patch_stack_scipy_gpu(current, cp.asarray(extraction, np.float64),
                                        patch_radius, cval=0.0).get()
        valid_pass = np.array(valid, copy=True)
        n_const = 0
        for l in range(patches.shape[0]):
            for t in range(patches.shape[1]):
                p = patches[l, t].copy()
                p -= p.mean()
                scale = float(p.std())
                if valid[l, t] and scale < 1e-6:
                    # fill-only patch: the footprint lies entirely outside
                    # the raw FOV for this view (plane-mean fill), so it
                    # carries no information.  In the treated 'exact' path
                    # this is asserted impossible (valid centres stay
                    # inside the 16-px margin); in the control the raw FOV
                    # is smaller than the aligned plane, so at high tilt
                    # and after cumulative updates it can happen.  Mask
                    # the view for this landmark (diagnostic only) rather
                    # than aborting the control run.
                    n_const += 1
                    valid_pass[l, t] = False
                    patches[l, t] = 0.0
                    continue
                patches[l, t] = p / (scale if scale >= 1e-6 else 1.0)
        patches = cp.asarray(patches, cp.float32)
        if n_const:
            print(f"raw_materialized_control: iteration {iteration}: "
                  f"masked {n_const} fill-only (constant) valid patch(es)",
                  flush=True)
    elif resample == 'cpu':
        extraction = centres.copy()
        extraction[~valid] = (size - 1) / 2.0
        patches = cp.asarray(subpixel_patch_stacks(current, extraction,radius=patch_radius), cp.float32)
    elif resample == 'fast':
        patches = subpixel_patch_stacks_gpu(current, centres, valid,radius=patch_radius)
    else:  # 'exact': bit-exact GPU replica of the reference patches
        extraction = centres.copy()
        extraction[~valid] = (size - 1) / 2.0
        if np.any((extraction[valid] < patch_radius) | (extraction[valid] > size - 1 - patch_radius)):
            raise ValueError('landmark patch outside detector support')
        patches = patch_stack_scipy_gpu(current, cp.asarray(extraction, np.float64),
                                         patch_radius, cval=0.0).get()
        # same per-patch float32 normalization as subpixel_patch_stacks
        # (the raw GPU resampling above is bit-exact vs map_coordinates)
        for l in range(patches.shape[0]):
            for t in range(patches.shape[1]):
                p = patches[l, t].copy()
                p -= p.mean()
                scale = float(p.std())
                if scale < 1e-6:
                    raise ValueError('constant landmark patch')
                patches[l, t] = p / scale
        patches = cp.asarray(patches, cp.float32)
    timing['extract_s'] = time.monotonic() - stamp

    stamp = time.monotonic()
    loo_indices = np.flatnonzero(valid_pass.sum(axis=1) >= 2)
    loo = BatchedLocalLOOWBPGPU(patches[loo_indices], angles, valid_pass[loo_indices])
    timing['loo_setup_s'] = time.monotonic() - stamp

    # raw mode: valid_pass (valid & raw_inside) can shrink each pass, so
    # the per-(landmark,target) LOO supports can be keys the initial
    # cache does not hold.  Compute every missing support for this pass
    # in ONE batched call instead of one at a time inside the target
    # loop.  The batched WBP is row-independent (each row is scaled by
    # its own valid.sum, FFT/gather are per-row), so a key computed in a
    # batch of N is bit-identical to one computed in a batch of 1.
    if not np.array_equal(valid_pass, support_valid):
        support_rows, support_masks = _loo_support_table(valid_pass)
        support_valid = valid_pass.copy()
    if source in ('raw', 'raw_materialized_control'):
        missing = {key: mask for key, mask in support_masks.items()
                   if key and key not in transfer_cache}
        if missing:
            transfer_cache.update(transfer_cache_for_supports(angles, list(missing.values()), size=patch_size))

    votes = np.full((len(positions), len(angles), 2), np.nan, np.float32)
    score_surfaces = np.full((len(positions), len(angles),
                              2 * search_radius + 1, 2 * search_radius + 1), np.nan, np.float32)
    boundary_count = multimodal_count = 0
    prominences, curvatures, common_before, common_after = [], [], [], []
    predicted_rows = []
    registration_models = []
    registration_regions = []
    registration_pairs = []

    stamp = time.monotonic()
    for target, angle in enumerate(angles):
        for local_ids, raw_models in loo.iter_target_models(target):
            landmark_ids = loo_indices[local_ids]
            if not len(landmark_ids):
                continue
            gpu_operators = []
            for r, landmark in enumerate(landmark_ids):
                gpu_operators.append(operator_lookup(support_rows[landmark][target]))
            models = apply_angular_transfer_operator_gpu(
                raw_models, cp.stack(gpu_operators)) * mask3d[None]
            projected_models = project_local_volumes_gpu_device(models, float(angle),observation_size=patch_size,model_radius=patch_radius)
            target_centres = centres[landmark_ids, target]
            target_valid = valid_pass[landmark_ids, target]
            # batched search regions for this target.  The aligned-raster
            # box gate is the reference (treated) contract.  In raw mode the
            # sampled window is a ROTATED footprint in the RAW raster
            # (half-width ~1.095*extent at -84.25 deg, plus the frozen-shift
            # offset of the raw base), so the aligned box does NOT guarantee
            # raw support: a window whose rotated footprint leaves the raw
            # detector would be sampled with constant fill (cval=0) and bias
            # the GCC vote.  The raw gate therefore additionally requires
            # the full rotated footprint to be inside (raw_search_support).
            extent = patch_radius + search_radius
            inside = (target_valid & np.all(target_centres >= extent, axis=1)
                      & np.all(target_centres <= size - 1 - extent, axis=1))
            if source == 'raw':
                target_axis=float(np.asarray(axis_deg)[target]) if np.ndim(axis_deg) else axis_deg
                inside = inside & raw_search_support(
                    by[landmark_ids, target], bx[landmark_ids, target],
                    extent, target_axis, size)
                if search_support_diag is not None:
                    # diagnostic only (zero effect on the fit): per target,
                    # the candidate set under the aligned-raster gate and
                    # whether the FULL rotated raw search footprint is
                    # inside the raw detector.  Independent of the gate
                    # above, so the same run reports what the PRE-fix code
                    # would have fed to the sampler.
                    ac = np.flatnonzero(target_valid
                                        & np.all(target_centres >= extent, axis=1)
                                        & np.all(target_centres <= size - 1 - extent, axis=1))
                    if len(ac):
                        bys = by[landmark_ids[ac], target]
                        bxs = bx[landmark_ids[ac], target]
                        oob = ~raw_search_support(bys, bxs, extent, target_axis, size)
                        _, _, d_ext_y, d_ext_x = _rotated_offsets(extent, target_axis)
                        # footprint-edge clearance (negative == out of bounds)
                        clr = np.minimum(np.minimum(bys - d_ext_y, size - 1 - bys - d_ext_y),
                                         np.minimum(bxs - d_ext_x, size - 1 - bxs - d_ext_x))
                        entry = search_support_diag.setdefault('targets', {}).setdefault(
                            int(np.asarray(diag_plane_map)[target]),
                            dict(gcc_passes=0, n_candidate=0, n_raw_oob=0, clearances=[]))
                        entry['gcc_passes'] += 1
                        entry['n_candidate'] += int(len(ac))
                        entry['n_raw_oob'] += int(np.sum(oob))
                        entry['clearances'].extend(np.asarray(clr, float).tolist())
            rows = np.flatnonzero(inside)
            if not len(rows):
                continue
            if source == 'raw':
                base_yx = np.stack((by[landmark_ids[rows], target],
                                    bx[landmark_ids[rows], target]), axis=1)
                regions = raw_search_regions_gpu(original_dev, target, base_yx,
                                                 extent, axis_deg)
            elif source == 'raw_materialized_control':
                # DIAGNOSTIC control: aligned-raster sampling from the
                # materialized stack, exactly the treated 'exact' regions.
                regions = patch_stack_scipy_gpu(
                    current[target:target + 1],
                    cp.asarray(target_centres[rows][:, None, :], np.float64),
                    extent, cval=0.0)[:, 0]
            elif resample == 'cpu':
                # reference driver: per-landmark scipy sampling (default cval 0.0)
                regions = np.stack([sample_search_region(current[target],
                               target_centres[r], patch_size, search_radius) for r in rows])
            elif resample == 'fast':
                n = 2 * extent + 1
                off = np.mgrid[-extent:extent + 1, -extent:extent + 1].astype(np.float32)
                cys = target_centres[rows, 0].astype(np.float32)
                cxs = target_centres[rows, 1].astype(np.float32)
                coords = cp.stack((
                    cp.broadcast_to(cp.full(len(rows), float(target), cp.float32)[:, None, None],
                                    (len(rows), n, n)),
                    cp.broadcast_to(cp.asarray(cys)[:, None, None], (len(rows), n, n)) + cp.asarray(off[0]),
                    cp.broadcast_to(cp.asarray(cxs)[:, None, None], (len(rows), n, n)) + cp.asarray(off[1]),
                )).astype(cp.float32)
                regions = cp.asarray(map_coordinates(current, coords, order=1, mode='constant'), cp.float32)
            else:  # 'exact': bit-exact GPU replica of the reference regions
                regions = patch_stack_scipy_gpu(
                    current[target:target + 1],
                    cp.asarray(target_centres[rows][:, None, :], np.float64),
                    extent, cval=0.0)[:, 0]
            registration_models.append(projected_models[rows])
            registration_regions.append(regions)
            registration_pairs.extend((int(landmark), int(target))
                                      for landmark in landmark_ids[rows])
            del raw_models, models, projected_models
    del loo

    # Registration used to return to the host once per target tilt.  Queue
    # every independent model/region pair for this pass and synchronize
    # only once. Larger batches preserve each pair's independent
    # correlation arithmetic and avoid repeated GPU/host overhead.
    if registration_pairs:
        results = register_local_models_gpu(
            cp.concatenate(registration_models), cp.concatenate(registration_regions),
            method='generalized_cc', search_radius=search_radius,
            alpha=gcc_alpha, mask=mask2d, batch_size=1024)
        for (landmark, target), result in zip(registration_pairs, results, strict=True):
            boundary_count += int(result.boundary)
            multimodal_count += int(result.multimodal)
            if result.valid:
                votes[landmark, target] = result.correction_yx
                score_surfaces[landmark, target] = result.surface
                prominences.append(result.peak_prominence)
                if np.isfinite(result.hessian_eigenvalues[1]):
                    curvatures.append(-result.hessian_eigenvalues[1])
                predicted_rows.append((landmark, target, result.zero_score))
    timing['tilts_s'] = time.monotonic() - stamp
    context.update(support_rows=support_rows,support_masks=support_masks,support_valid=support_valid)
    return dict(votes=votes,valid_views=valid_pass,score_surfaces=score_surfaces,
        predicted_rows=predicted_rows,prominences=prominences,curvatures=curvatures,
        boundary_count=boundary_count,multimodal_count=multimodal_count,timing_s=timing)
