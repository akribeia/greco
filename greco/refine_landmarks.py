"""Greco native-resolution (bin1) landmark refinement.

Refines a bin8 2-D solution to full resolution working only on landmark
patches: the treated detector planes are read, never fully resampled, and no
full-resolution WBP volume is built.  Per pass:

  * observation patches ((2r+1)^2) are sampled directly from the raw
    treated bin1 planes at ``projected centre - current cumulative shift``
    (one batched GPU map_coordinates);
  * the LOO WBP models, angular-transfer correction and local projection
    use the same validated kernels as the bin8 loop (local_loo_gpu,
    angular_transfer_gpu, local_projection_gpu) at the patch scale;
  * every (landmark, target tilt) pair is registered with the GPU
    generalized CC (or gradient ZNCC) over a small search window;
  * the joint solve uses the same soft-L1 model as bin8, with a GPU backend;
    containment follows the native patch/search footprint.

Memory cost is O(L*T*(2r+1)^2), not O(T*Y*X): the heavy stack is only
read.  This is the final-resolution step of the protocol: the bin8 solution scaled
x8 initializes the loop and the per-tilt corrections land on the
full-resolution grid (subpixel).
"""

from __future__ import annotations
from greco.project_layout import join as storage_path

import time

import numpy as np


def _gpu():
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    return cp, map_coordinates


def lift_geometry(positions,cumulative,binning,coarse_size,native_size,axis_deg,flip_y=False):
    """Transport XYZ and shifts while preserving the Fourier-bin raw grid.

    Detector/volume centers and reflection have distinct affine offsets.
    Native raw bases equal b*coarse bases plus (b-1,0) when flipped.
    """
    from .gcc import raw_affine
    b=float(binning)
    if b<1 or native_size!=b*coarse_size:raise ValueError('Native/coarse raster sizes do not match binning.')
    delta=(native_size-1)/2-b*(coarse_size-1)/2
    m00,m01,m10,m11,_=raw_affine(axis_deg,native_size)
    matrix=np.array([[m00,m01],[m10,m11]])
    raw_offset=np.array([b-1 if flip_y else 0,0.])
    shift_offset=matrix.T@(np.full(2,delta)-raw_offset)
    return np.asarray(positions,float)*b+delta,np.asarray(cumulative,float)*b+shift_offset


# ---------------------------------------------------------------------------
# patch-only sampling from the raw (unshifted) stack
# ---------------------------------------------------------------------------

def raw_patch_stacks_gpu(original_dev, centres, cumulative, valid, radius):
    """Observation patches sampled directly from the raw planes.

    ``centres`` are aligned coordinates, (L, T, 2) (y, x); the raw sample
    position of view t is ``centre - cumulative[t]`` (the same relation
    the materialized aligned stack satisfies: aligned[c] =
    original[c - shift]).

    Returns device (L, T, 2r+1, 2r+1) float32, per-patch z-scored (the same
    normalization as gcc.subpixel_patch_stacks_gpu).  Invalid views are
    sampled at the detector centre and never vote downstream.
    """
    cp, map_coordinates = _gpu()
    t, y, _ = original_dev.shape
    centres = np.asarray(centres, float)
    cumulative = np.asarray(cumulative, float)
    valid = np.asarray(valid, bool)
    if centres.shape != (len(valid), t, 2):
        raise ValueError('centres/valid shape mismatch')
    extraction = centres - cumulative[None, :, :]
    extraction[~valid] = (y - 1) / 2.0
    # defensive: keep any out-of-support patch on the detector centre (it is
    # filtered by the registration loop's 'inside' test and cannot vote)
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
    patches = cp.asarray(map_coordinates(original_dev, coords, order=1, mode='constant'), cp.float32)
    mean = patches.mean(axis=(-2, -1), keepdims=True)
    std = patches.std(axis=(-2, -1), keepdims=True)
    if float(cp.min(std[cp.asarray(valid).reshape(l, t, 1, 1) & cp.ones((1, 1), cp.bool_)]).get()) < 1e-6:
        raise ValueError('constant landmark patch')
    safe = cp.where(std < 1e-6, cp.float32(1.0), std)
    return ((patches - mean) / safe).astype(cp.float32, copy=False)


def raw_search_regions_gpu(original_dev, target, centres_raw, extent):
    """Search windows on the raw target plane around ``centres_raw`` (R, 2)."""
    cp, map_coordinates = _gpu()
    n = 2 * extent + 1
    offy, offx = np.mgrid[-extent:extent + 1, -extent:extent + 1].astype(np.float32)
    r = len(centres_raw)
    cy = centres_raw[:, 0][:, None, None].astype(np.float32)
    cx = centres_raw[:, 1][:, None, None].astype(np.float32)
    coords = cp.stack((
        cp.broadcast_to(cp.full(r, float(target), cp.float32)[:, None, None], (r, n, n)),
        cp.broadcast_to(cp.asarray(cy), (r, n, n)) + cp.asarray(offy),
        cp.broadcast_to(cp.asarray(cx), (r, n, n)) + cp.asarray(offx),
    )).astype(cp.float32)
    return cp.asarray(map_coordinates(original_dev, coords, order=1, mode='constant',
                                      cval=0.0), cp.float32)


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def refine_landmarks(original, angles, initial_tracks, initial_cumulative, *,
                method='generalized_cc', patch_radius=16, search_radius=8,
                gcc_alpha=.25, max_passes=60, stop_shift=.10, stop_shift_p90=.25,
                stop_consecutive=3, robust_scale=.2, solver_max_nfev=2000,
                solver_tolerance=1e-6, wiener_lambda=.05, max_inverse_gain=4.,
                lowpass_fraction=.8, plateau_window=10, plateau_eps=.05,
                raw_source=False, frozen_shifts_yx=None, axis_deg=None,
                batch_size=8, solver_backend='gpu_lsmr32', lowpass_schedule=None):
    """Native-resolution landmark refinement (patch-only run_gcc variant).

    original: (T, Y, X) float32 treated bin1 stack (CPU numpy; Y == X).
    initial_tracks: dict with positions_zyx (full-res px), valid_views,
    classes.
    initial_cumulative: (T, 2) full-res px (typically the bin8 solution x8).

    raw_source: True = maximum-FOV mode: ``original`` is the raw-source bin1
    stack (CTF+reliability in the raw MC raster, no frozen shift/rotation)
    and frozen_shifts_yx (T, 2) + axis_deg feed the on-the-fly affine
    sampling (rotation + frozen + current shift at sample time, no baked
    edge fill; gcc.raw_sample_bases).

    Returns (final_state, manifest) with the same keys as gcc.run_gcc.
    """
    cp, _ = _gpu()
    from .gcc import (raw_sample_bases,
                      raw_patch_stacks_gpu as affine_patch_stacks_gpu,
                      raw_search_regions_gpu as affine_search_regions_gpu,
                      _surface_value, _subset_agreement,
                      _xy_update_is_stable, _solve_with_local_containment,
                      raw_search_support)
    from .helpers import soft_spherical_mask, projected_centres
    from .registration import cosine_disk_mask
    from .local_loo_gpu import BatchedLocalLOOWBPGPU
    from .local_projection_gpu import project_local_volumes_gpu_device
    from .local_registration_gpu import register_local_models_gpu
    from .angular_transfer_gpu import compensate_angular_transfer_gpu_device
    from .solver import projection_jacobians
    from .tomography import gpu_reconstruct_wbp_batch_device
    from .local_projection_gpu import project_volume_patches_gpu_device

    started = time.monotonic()
    original = cp.asarray(original, cp.float32)
    angles = np.asarray(angles, float)
    if len(angles) != original.shape[0]:
        raise ValueError('stack/angle mismatch')
    if method not in ('generalized_cc', 'gradient_zncc'):
        raise ValueError("method must be 'generalized_cc' or 'gradient_zncc' (GPU batch)")
    if raw_source:
        if frozen_shifts_yx is None or axis_deg is None:
            raise ValueError("raw_source requires frozen_shifts_yx and axis_deg")
        frozen = np.asarray(frozen_shifts_yx, float)
        if frozen.shape != (len(angles), 2):
            raise ValueError('frozen shifts must have shape (n_tilts, 2)')
    else:
        frozen = None
    size = original.shape[-1]
    s = 2 * patch_radius + 1
    extent = s // 2 + search_radius

    original_dev = cp.asarray(original)
    cumulative = np.asarray(initial_cumulative, float).copy()
    positions = np.asarray(initial_tracks['positions_zyx'], float)
    valid = np.asarray(initial_tracks['valid_views'], bool)
    classes = np.asarray(initial_tracks['classes'])
    if valid.shape != (len(positions), len(angles)):
        raise ValueError('input tracks, stack and angles do not match')

    mask3d = cp.asarray(soft_spherical_mask(size=s, radius=s // 2 - 1, edge=3.0), cp.float32)
    mask2d = cosine_disk_mask(size=s, radius=s // 2 - 1, edge=3.0)
    # Native PSFs and their FFTs stay on-device. Keep a bounded cache for
    # genuinely local tracks, which can otherwise create thousands of PSFs.
    from collections import OrderedDict
    transfer_cache=OrderedDict()
    impulse=cp.zeros((s,s,s),cp.float32);impulse[s//2,s//2,s//2]=1
    impulse_projections=cp.stack([project_volume_patches_gpu_device(
        impulse,np.zeros((1,3)),float(a),observation_radius=patch_radius,
        model_radius=patch_radius)[0] for a in angles])
    cache_limit=max(1,min(len(angles)*2, int(cp.cuda.runtime.memGetInfo()[0]*.15/(s**3*8))))
    def transfer_lookup(support):
        key=tuple(np.flatnonzero(support))
        if not key:raise ValueError('empty leave-one-out support')
        if key not in transfer_cache:
            if len(transfer_cache)>=cache_limit:transfer_cache.popitem(last=False)
            # WBP masks projections in-place: never mutate the shared impulse.
            psf=gpu_reconstruct_wbp_batch_device(impulse_projections[None].copy(),angles,
                valid_bt=np.asarray(support,bool)[None],y_chunk=8)[0]
            transfer=cp.fft.fftn(cp.fft.ifftshift(psf))
            transfer_cache[key]=(transfer/cp.maximum(cp.max(cp.abs(transfer)),cp.float32(1e-12))).astype(cp.complex64)
        transfer_cache.move_to_end(key)
        return transfer_cache[key]
    bands=list(lowpass_schedule or [lowpass_fraction])
    if not bands or any(not 0<b<=1 for b in bands) or bands!=sorted(bands):
        raise ValueError('invalid frequency continuation schedule')
    band_index=0;lowpass_fraction=bands[0]

    history = []
    stable_passes = 0
    stop_reason = 'max_passes'
    recent_rms = []
    # Retain only two observation graphs: reuse rank/design work when votes
    # have the same support, without accumulating dense systems across passes.
    solver_workspace=OrderedDict()

    for iteration in range(1, max_passes + 1):
        iteration_started = time.monotonic()
        timing = {}
        stamp = time.monotonic()
        centres = projected_centres(positions, angles, size)
        if raw_source:
            by, bx = raw_sample_bases(centres, cumulative, frozen, axis_deg, size)
            patches, raw_inside = affine_patch_stacks_gpu(
                original_dev, by, bx, valid, patch_radius, axis_deg)
            valid_pass = valid & raw_inside
            search_inside=raw_search_support(by,bx,extent,axis_deg,size)
        else:
            extraction=centres-cumulative[None]
            valid_pass=valid & np.all((extraction>=patch_radius)&(extraction<=size-1-patch_radius),axis=2)
            search_inside=np.all((extraction>=extent)&(extraction<=size-1-extent),axis=2)
            patches = raw_patch_stacks_gpu(original_dev, centres, cumulative, valid, patch_radius)
        timing['extract_s'] = time.monotonic() - stamp

        stamp = time.monotonic()
        loo_indices = np.flatnonzero(valid_pass.sum(axis=1) >= 2)
        if not len(loo_indices):
            stop_reason='insufficient_supported_landmarks'
            break
        loo = BatchedLocalLOOWBPGPU(patches[loo_indices], angles, valid_pass[loo_indices],batch_size=batch_size)
        timing['loo_setup_s'] = time.monotonic() - stamp

        votes = np.full((len(positions), len(angles), 2), np.nan, np.float32)
        score_surfaces = np.full((len(positions), len(angles),
                                  2 * search_radius + 1, 2 * search_radius + 1), np.nan, np.float32)
        boundary_count = multimodal_count = 0
        prominences, curvatures, common_before, common_after = [], [], [], []
        predicted_rows = []

        stamp = time.monotonic()
        target_batches=[(target,float(angle),start) for target,angle in enumerate(angles)
                        for start in range(0,len(loo_indices),batch_size)]
        for target, angle, start in target_batches:
            local_ids, raw_models = loo.target_models(target,np.arange(start,min(start+batch_size,len(loo_indices))))
            landmark_ids = loo_indices[local_ids]
            if not len(landmark_ids):
                continue
            gpu_transfers = []
            for landmark in landmark_ids:
                support = valid_pass[landmark].copy()
                support[target] = False
                gpu_transfers.append(transfer_lookup(support))
            models = compensate_angular_transfer_gpu_device(
                raw_models, cp.stack(gpu_transfers), wiener_lambda, max_inverse_gain,
                lowpass_fraction) * mask3d[None]
            projected_models = project_local_volumes_gpu_device(models, float(angle),
                                                                observation_size=s,
                                                                model_radius=patch_radius)
            target_centres = centres[landmark_ids, target]
            target_valid = valid_pass[landmark_ids, target]
            inside = (target_valid & search_inside[landmark_ids,target] & np.all(target_centres >= extent, axis=1)
                      & np.all(target_centres <= size - 1 - extent, axis=1))
            rows = np.flatnonzero(inside)
            if not len(rows):
                continue
            if raw_source:
                base_yx = np.stack((by[landmark_ids[rows], target],
                                    bx[landmark_ids[rows], target]), axis=1)
                regions = affine_search_regions_gpu(original_dev, target, base_yx,
                                                    extent, axis_deg)
            else:
                regions = raw_search_regions_gpu(original_dev, target,
                                                 target_centres[rows] - cumulative[target], extent)
            results = register_local_models_gpu(projected_models[rows], regions,
                                                method=method, search_radius=search_radius,
                                                alpha=gcc_alpha, mask=mask2d)
            if len(results)!=len(rows):raise ValueError('registration result count mismatch')
            for landmark, result in zip(landmark_ids[rows], results):
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
        del loo, patches

        stamp = time.monotonic()
        try:
            shifts, dr, report, votes, revised_valid, active, quarantined, safe, offending = (
                _solve_with_local_containment(votes, valid_pass, classes, positions, angles, size,
                                              search_radius, robust_scale, solver_max_nfev,
                                              solver_tolerance,workspace=solver_workspace,backend=solver_backend,
                                              containment_margin=extent))
        except (ValueError, RuntimeError) as error:
            stop_reason = 'joint_fit_failed'
            history.append(dict(iteration=iteration, error=str(error),
                                valid_observations=int(np.isfinite(votes).all(axis=2).sum()),
                                boundary_observations=boundary_count))
            break
        timing['solve_s'] = time.monotonic() - stamp
        while len(solver_workspace)>2:solver_workspace.popitem(last=False)
        if report.get('unsupported_tilt_indices'):
            stop_reason='insufficient_observed_tilts'
            history.append(dict(iteration=iteration,unsupported_tilt_indices=report['unsupported_tilt_indices'],applied=False))
            break
        if quarantined:
            valid = revised_valid

        predicted = shifts[None] - np.einsum('tac,kc->kta', projection_jacobians(angles), dr)
        for (landmark, target, zero) in predicted_rows:
            common_before.append(zero)
            common_after.append(_surface_value(score_surfaces[landmark, target],
                                               predicted[landmark, target], search_radius))
        new_positions = positions + dr
        if raw_source:
            proposed_centres=projected_centres(new_positions,angles,size)
            py,px=raw_sample_bases(proposed_centres,cumulative+shifts,frozen,axis_deg,size)
            proposed_support=raw_search_support(py,px,patch_radius,axis_deg,size)
            safe=bool(safe and not np.any(valid_pass & ~proposed_support))
        shift_norm = np.linalg.norm(shifts, axis=1)
        landmark_norm = np.linalg.norm(dr, axis=1)
        stable = _xy_update_is_stable(shifts, stop_shift, stop_shift_p90)
        stable_passes = stable_passes + 1 if stable else 0
        compute_subset = iteration == 1 or iteration % 5 == 0 or iteration == max_passes
        entry = dict(
            iteration=iteration, applied=safe,
            lowpass_fraction=lowpass_fraction,
            active_landmarks=int(active.sum()),
            valid_observations=int(np.isfinite(votes).all(axis=2).sum()),
            boundary_observations=boundary_count, multimodal_observations=multimodal_count,
            median_shift=float(np.median(shift_norm)), p90_shift=float(np.quantile(shift_norm, .9)),
            max_shift=float(np.max(shift_norm)),
            median_landmark_update=float(np.median(landmark_norm)),
            p90_landmark_update=float(np.quantile(landmark_norm, .9)),
            median_after=float(report.get('median_after', float('nan'))),
            rms_after=float(report.get('rms_after', float('nan'))),
            median_peak_prominence=float(np.median(prominences)) if prominences else None,
            median_hessian_curvature=float(np.median(curvatures)) if curvatures else None,
            common_score_before=float(np.mean(common_before)) if common_before else None,
            common_score_after=float(np.mean(common_after)) if common_after else None,
            landmark_subset_agreement=_subset_agreement(votes, angles, robust_scale,
                                                         solver_max_nfev, solver_tolerance,backend=solver_backend)
            if compute_subset else None,
            stable_passes=stable_passes, quarantined=bool(quarantined),
            timing_s=timing, elapsed_s=time.monotonic() - iteration_started,
        )
        print(f"final-resolution pass {iteration:02d}: band={lowpass_fraction:.3f}, "
              f"observations={entry['valid_observations']}, "
              f"shift median/p90={entry['median_shift']:.3f}/{entry['p90_shift']:.3f} grid px, "
              f"{entry['elapsed_s']:.1f}s, applied={safe}",flush=True)
        if not safe:
            stop_reason = 'joint_solution_outside_support_or_shift_limit'
            history.append(entry)
            break
        cumulative += shifts
        positions = new_positions
        history.append(entry)
        if stable_passes >= stop_consecutive:
            if band_index<len(bands)-1:
                band_index+=1;lowpass_fraction=bands[band_index];stable_passes=0;recent_rms=[]
                continue
            stop_reason = 'converged'
            break
        # plateau stop: at native resolution the per-pass measurement noise
        # of the small landmark patches (~0.3 px) sits above the production
        # .10/.25 stability thresholds, so the loop otherwise limit-cycles
        # on the noise floor; stop once the residual stops improving
        rms = entry.get('rms_after')
        if rms is not None and np.isfinite(rms):
            recent_rms.append(float(rms))
            recent_rms = recent_rms[-plateau_window:]
            if len(recent_rms) == plateau_window and (max(recent_rms) - min(recent_rms)) <= plateau_eps:
                if band_index<len(bands)-1:
                    band_index+=1;lowpass_fraction=bands[band_index];stable_passes=0;recent_rms=[]
                    continue
                stop_reason = 'plateau'
                break

    if raw_source:
        final_centres=projected_centres(positions,angles,size)
        fy,fx=raw_sample_bases(final_centres,cumulative,frozen,axis_deg,size)
        valid=valid & raw_search_support(fy,fx,patch_radius,axis_deg,size)
    final_state = dict(positions_zyx=positions, cumulative_shifts_yx=cumulative,
                       initial_cumulative_shifts_yx=np.asarray(initial_cumulative, float),
                       round_correction_shifts_yx=cumulative - np.asarray(initial_cumulative, float),
                       valid_views=valid, classes=classes, angles_deg=angles)
    manifest = dict(stop_reason=stop_reason, passes=len(history),
                    method=method, patch_radius=patch_radius, search_radius=search_radius,
                    lowpass_schedule=bands,final_lowpass_fraction=lowpass_fraction,
                    history=history,
                    last=history[-1] if history else None,
                    runtime_s=time.monotonic() - started)
    return final_state, manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _max_step(cumulative, angles, lo=46.0, hi=66.5):
    idx = [i for i in range(len(angles)) if lo <= angles[i] <= hi]
    best = None
    for i, j in zip(idx, idx[1:]):
        d = float(np.abs(np.diff(cumulative[[i, j]], axis=0)).max())
        if best is None or d > best[2]:
            best = (float(angles[i]), float(angles[j]), d)
    return best


def _clean(obj):
    import math
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()
                if not (isinstance(v, float) and math.isnan(v))}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj if not (isinstance(v, float) and math.isnan(v))]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main() -> int:
    import argparse
    import json
    from pathlib import Path

    import mrcfile

    p = argparse.ArgumentParser(
        description='greco native-resolution (bin1) landmark refinement (patches only)')
    p.add_argument('--bin1-stack', type=Path, required=True,
                   help='native-resolution mrcs stack: treated (preprocess '
                        '--bin1) or raw-source (preprocess --bin1 --raw-source '
                        'with --raw-source below)')
    p.add_argument('--raw-source', action='store_true',
                   help='maximum-FOV mode: --bin1-stack is the raw-source bin1 '
                        'stack (CTF+reliability in the raw MC raster, no frozen '
                        'shift/rotation) and the rotation + frozen + current '
                        'shift are applied on the fly at sample time.  Requires '
                        '--axis-deg and --frozen-shifts.')
    p.add_argument('--axis-deg', type=float, default=None,
                   help='tilt axis angle (raster rotation) for --raw-source')
    p.add_argument('--frozen-shifts', type=Path, default=None,
                   help='per-plane frozen 2-D shifts csv (plane, angle_deg, '
                        'dy_raw, dx_raw) for --raw-source')
    p.add_argument('--bin8-run', type=Path, required=True,
                   help='greco 2-D run dir with final_state.npz (the solution to refine)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=('generalized_cc', 'gradient_zncc'),
                   default='generalized_cc',
                   help='local registration metric (GCC = production metric; '
                        'gradient_zncc = ZNCC family, GPU)')
    # defaults (patch_radius=16, search_radius=8): the patch is the bin8
    # loop's 16 px bin8 at native scale; the search window is 8 fullres px
    # because the initial state (bin8 solution x8) is only off by a few
    # fullres px per observation (the bin8 loop's 4 px bin8 = 32 fullres px
    # window would be 16x wider in area and multimodal-prone at this scale)
    p.add_argument('--patch-radius', type=int, default=16,
                   help='full-res observation/model patch radius (default 16 = '
                        'the bin8 loop 16 px bin8 at native scale)')
    p.add_argument('--search-radius', type=int, default=8,
                   help='full-res per-observation search radius (default 8 = '
                        'the bin8 loop 4 px bin8 at native scale)')
    p.add_argument('--max-passes', type=int, default=60)
    p.add_argument('--plateau-window', type=int, default=10,
                   help='plateau stop: window of recent rms_after values')
    p.add_argument('--plateau-eps', type=float, default=.05,
                   help='plateau stop: stop when max-min of the window <= this (fullres px)')
    p.add_argument('--tilts', type=str, default=None,
                   help='comma-separated tilt subset (development/smoke test)')
    p.add_argument('--max-landmarks', type=int, default=None,
                   help='keep at most this many landmarks (development/smoke test)')
    p.add_argument('--write-aligned', action='store_true',
                   help='materialize the final aligned bin1 stack')
    a = p.parse_args()

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(storage_path(a.bin8_run, 'final_state.npz')) as d:
        b = float(d['binning']) if 'binning' in d.files else 8.0
        positions = np.asarray(d['positions_zyx'], float)
        cumulative = np.asarray(d['cumulative_shifts_yx'], float)
        valid = np.asarray(d['valid_views'], bool)
        classes = np.asarray(d['classes'])
        angles = np.asarray(d['angles_deg'], float)
        # the 2-D loop may stop before full range: angles_deg/cumulative are
        # full-length but valid_views covers only the active subset, so it is
        # expanded back to the full angle set (kai full_range runs already
        # have valid_views full-length).
        active = (np.asarray(d['active_indices'], int)
                  if 'active_indices' in d.files else None)
    if active is not None and valid.shape[1] == len(active) != len(angles):
        full = np.zeros((valid.shape[0], len(angles)), bool)
        full[:, active] = valid
        valid = full
    with mrcfile.open(str(a.bin1_stack), permissive=True) as h:
        original = np.asarray(h.data, np.float32).copy()
        voxel = float(h.voxel_size.x)
    if valid.shape!=(len(positions),len(angles)) or cumulative.shape!=(len(angles),2):
        p.error('accepted state has inconsistent tilt/track dimensions')
    positions,cumulative=lift_geometry(positions,cumulative,b,original.shape[-1]/b,
        original.shape[-1],a.axis_deg if a.raw_source else 0.)

    frozen = None
    if a.raw_source:
        if a.axis_deg is None:
            p.error('--raw-source requires --axis-deg')
        if a.frozen_shifts is None:
            p.error('--raw-source requires --frozen-shifts')
        from .preprocessing_star import read_table
        rows=read_table(a.frozen_shifts)
        frozen=np.asarray([[float(row['dy_raw']),float(row['dx_raw'])] for row in rows])
        if len(frozen) != len(angles):
            p.error(f'--frozen-shifts has {len(frozen)} rows, expected {len(angles)}')

    indices = np.arange(len(angles))
    if a.tilts:
        indices = np.asarray([int(v) for v in a.tilts.split(',')], int)
        angles = angles[indices]
        cumulative = cumulative[indices]
        valid = valid[:, indices]
        original = original[indices]
        if frozen is not None:
            frozen = frozen[indices]
    if a.max_landmarks is not None:
        keep = np.flatnonzero(valid.sum(axis=1) > 0)[:a.max_landmarks]
        positions = positions[keep]
        valid = valid[keep]
        classes = classes[keep]

    final_state, manifest = refine_landmarks(
        original, angles,
        dict(positions_zyx=positions, valid_views=valid, classes=classes),
        cumulative, method=a.method, patch_radius=a.patch_radius,
        search_radius=a.search_radius, max_passes=a.max_passes,
        plateau_window=a.plateau_window, plateau_eps=a.plateau_eps,
        raw_source=a.raw_source, frozen_shifts_yx=frozen, axis_deg=a.axis_deg)

    np.savez_compressed(storage_path(out, 'refine_landmarks.npz'), **final_state, binning=1,
                        input_binning=b,coordinate_pixel_size_A=voxel)
    init = np.asarray(final_state['initial_cumulative_shifts_yx'], float)
    final = np.asarray(final_state['cumulative_shifts_yx'], float)
    with (storage_path(out, 'shifts_bin1.csv')).open('w') as h:
        h.write('stack_index,angle_deg,dy_init_fullres,dx_init_fullres,'
                'dy_refined_fullres,dx_refined_fullres,ddy_fullres,ddx_fullres\n')
        for k in range(len(angles)):
            h.write(f'{indices[k]},{angles[k]:.6f},{init[k, 0]:.6f},{init[k, 1]:.6f},'
                    f'{final[k, 0]:.6f},{final[k, 1]:.6f},'
                    f'{final[k, 0] - init[k, 0]:+.6f},{final[k, 1] - init[k, 1]:+.6f}\n')

    correction = np.linalg.norm(final - init, axis=1)
    report = dict(
        method=a.method, patch_radius=a.patch_radius, search_radius=a.search_radius,
        n_tilts=len(angles), n_landmarks=int(len(positions)),
        stop_reason=manifest['stop_reason'], passes=manifest['passes'],
        runtime_s=manifest['runtime_s'],
        max_step_46_66_before=_max_step(init, angles),
        max_step_46_66_after=_max_step(final, angles),
        correction_median_fullres_px=float(np.median(correction)),
        correction_p90_fullres_px=float(np.quantile(correction, .9)),
        per_pass=[{k: h.get(k) for k in ('iteration', 'median_shift', 'p90_shift',
                                         'max_shift', 'median_after', 'rms_after')}
                  for h in manifest['history']],
        last=_clean(manifest['last']),
    )
    (storage_path(out, 'refine_landmarks_report.json')).write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + '\n')

    if a.write_aligned:
        import cupy as cp
        if a.raw_source:
            # original is the (possibly --tilts-subset) raw-source stack:
            # plane indices are relative to it
            from .gcc import materialize_raw_stack_gpu
            aligned = materialize_raw_stack_gpu(cp.asarray(original),
                                                np.arange(len(original)),
                                                final, frozen,
                                                a.axis_deg).get()
        else:
            from .scipy_resample_gpu import shift_stack_scipy_gpu
            cvals = np.asarray([float(im.mean()) for im in original], np.float64)
            aligned = shift_stack_scipy_gpu(cp.asarray(original), cp.asarray(final, np.float64),
                                            cp.asarray(cvals)).get()
        with mrcfile.new(str(storage_path(out, 'aligned_bin1.mrcs')), overwrite=True) as h:
            h.set_data(np.ascontiguousarray(aligned))
            h.voxel_size = voxel

    last = manifest['last'] or {}
    print(f"refine_landmarks: {manifest['passes']} passes, stop={manifest['stop_reason']}, "
          f"median |correction| {report['correction_median_fullres_px']:.3f} fullres px, "
          f"p90 {report['correction_p90_fullres_px']:.3f}, "
          f"last median_after {last.get('median_after')}, "
          f"max_step_46_66 {report['max_step_46_66_before']} -> {report['max_step_46_66_after']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
