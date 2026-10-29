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

"""Greco 2D progressive all-active landmark registration (single process).

The loop keeps the immutable treated bin8 original in RAM, re-materializes
the accepted subset per stage, rebuilds the working volume on the GPU,
pre-positions incoming views with masked cross-correlation, selects landmarks
with the all-active column bootstrap and refines them with full-support
leave-one-out generalized cross-correlation.

Registration parameters:
  * seed: 7 central tilts (6 for even series); expansion: one neighbour per
    side per stage; a failed incoming CC blocks only its own frontier;
  * column directions: seed for stage 0, every proposed tilt afterwards
    (all-active);
  * bright_structure selector with column-based landmark selection;
  * GCC: generalized_cc alpha .25 radius 4, full-support LOO, 60 passes,
    stop .10/.25 after 3 stable passes, robust .2, wiener .05, max gain 4,
    lowpass .8;
  * stage acceptance: only converged / max_passes (production GCC safety).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mrcfile
import numpy as np

from .bootstrap import (accepted_stop_reason, central_index_seed, incoming_pair,
                        predict_neighbor_shift, stage_column_directions)
from .column_bootstrap import (enforce_global_margin_headroom,
                               detect_column_landmarks)
from ..gcc import run_gcc
from ..helpers import consensus, jackknife_max
from ..selection import masked_landmark_projection_gpu, project_points_aligned_axis0, refine_from_selected_landmarks
from ..stack_materialization import materialize_cumulative_stack
from ..tomography import gpu_reconstruct_wbp
from .entry import select_entry_landmarks, validate_sigma
from ..tomogram_thickness import reconstruct_stage, validate_percent


def _write_landmark_projections(stage_dir, volume, landmarks, angles, voxel):
    """Persist only landmarks that actually enter GCC for each stage view."""
    positions = np.asarray(landmarks['positions_zyx'], np.int32)
    valid = np.asarray(landmarks['valid_views'], bool)
    if valid.shape != (len(positions), len(angles)):
        raise ValueError('landmark/view mask does not match stage angles')
    import cupy as cp
    volume = cp.asarray(volume, dtype=cp.float32)
    projections = np.stack([
        masked_landmark_projection_gpu(volume, positions[valid[:, view]], float(angle), return_support=False)[0]
        for view, angle in enumerate(angles)
    ]).astype(np.float32, copy=False)
    with mrcfile.new(stage_dir / 'landmarks_projection.mrcs', overwrite=False) as handle:
        handle.set_data(projections)
        handle.voxel_size = (float(voxel),) * 3


# ---------------------------------------------------------------------------
# Incoming-view pre-positioning with masked cross-correlation
# ---------------------------------------------------------------------------

def masked_cc_target(volume, active_stack, active_angles, target_image, target_angle,
                     predicted, *, radius=32, selector='bright_structure',
                     cc_method='intensity', column_cell=32, column_width=30,
                     entry_search_sigma_px=8., thickness_percent=50., score_volume=None):
    started = time.monotonic()
    sigma = validate_sigma(entry_search_sigma_px)
    landmarks, ids, column_angle, entry_reference = select_entry_landmarks(
        volume, active_stack, active_angles, target_angle, target_image.shape,
        selector=selector, thickness_percent=thickness_percent, score_volume=score_volume,
        column_cell=column_cell, column_width=column_width)
    if not len(ids):
        return dict(applied=False, reason='no_reference_supported_maxima',
                    angle_deg=float(target_angle), entry_column_angle_deg=column_angle,
                    entry_reference_index_in_active=entry_reference,
                    entry_search_sigma_px=sigma, raw_columns=len(landmarks), selected_columns=0)
    selected = landmarks[ids]
    if cc_method == 'gradient':
        from types import SimpleNamespace
        from ..selection import masked_landmark_projection_gpu
        projection, support = masked_landmark_projection_gpu(
            volume, selected, float(target_angle))
        square = np.zeros(projection.shape, dtype=np.float32)
        square[32:-32, 32:-32] = 1
        mask = (support > .25).astype(np.float32) * square
        gradient = _gradient_cc(projection, target_image, mask, radius, search_sigma=sigma)
        result = SimpleNamespace(projection_yx=projection, projection_mask_yx=mask,
                                 correction_yx=gradient['correction_yx'],
                                 score=gradient['score'], at_boundary=gradient['boundary'])
        chosen = np.asarray(gradient['correction_yx'], float)
    elif cc_method == 'intensity':
        gradient = None
        result = refine_from_selected_landmarks(volume, target_image, selected, float(target_angle),
                                             search_radius=radius, search_sigma=sigma,
                                             search_center_yx=(0., 0.))
        chosen = np.asarray(result.correction_yx, float)
    else:
        raise ValueError("cc_method must be 'intensity' or 'gradient'")
    votes = []
    boundary = 0
    centre = (volume.shape[0] - 1) / 2
    points = []
    regions = []
    extent = 16 + radius
    for xyz in selected:
        point = np.asarray([xyz[2] - centre, xyz[1] - centre, xyz[0] - centre])
        detector = project_points_aligned_axis0(xyz[None], float(target_angle), volume.shape[0])[0]
        cy, cx = np.rint(detector).astype(int)
        if (cy - extent < 0 or cx - extent < 0
                or cy + extent >= target_image.shape[0]
                or cx + extent >= target_image.shape[1]):
            continue
        points.append(point)
        regions.append(target_image[cy-extent:cy+extent+1, cx-extent:cx+extent+1])
    if points:
        from ..local_projection_gpu import project_volume_patches_gpu_device
        from ..local_registration_gpu import register_template_zncc_gpu
        templates = project_volume_patches_gpu_device(
            volume, np.asarray(points), float(target_angle),
            observation_radius=16, model_radius=16)
        results = register_template_zncc_gpu(templates, np.asarray(regions), radius)
    else:
        results = []
    for vote, _, edge in results:
        boundary += int(edge)
        if not edge:
            votes.append(vote)
    decision = 'insufficient_local_votes'
    local_dispersion = local_jackknife = agreement = None
    local = None
    if len(votes) >= 3:
        local, inlier, residual = consensus(votes)
        ninlier = int(inlier.sum())
        local_dispersion = float(np.median(residual[inlier])) if ninlier else None
        local_jackknife = jackknife_max(np.asarray(votes)[inlier]) if ninlier >= 3 else None
        agreement = float(np.linalg.norm(chosen - local))
        stable = (ninlier >= 3 and local_dispersion <= 2.
                  and (local_jackknife is None or local_jackknife <= 2.))
        if result.at_boundary and stable:
            chosen = np.asarray(local)
            decision = 'local_rescue'
        elif not result.at_boundary and stable and agreement <= 3.:
            decision = 'global_confirmed'
        else:
            decision = 'global_local_disagreement'
    if (decision == 'global_local_disagreement' and not result.at_boundary
            and agreement is not None and agreement <= 3. and np.linalg.norm(chosen) <= 3.):
        decision = 'centres_confirm_global'
    elif (decision == 'global_local_disagreement' and not result.at_boundary
            and np.linalg.norm(chosen) <= 2.):
        decision = 'prior_confirms_global'
    if gradient is not None:
        decision = 'gradient_global' if not gradient['boundary'] else 'gradient_boundary'
        applied = not gradient['boundary']
    else:
        applied = decision in ('global_confirmed', 'local_rescue',
                               'centres_confirm_global', 'prior_confirms_global')
    output = dict(applied=applied, reason=decision, angle_deg=float(target_angle),
                predicted_shift_yx=predicted.tolist(), cc_residual_yx=chosen.tolist(),
                initial_shift_for_gcc_yx=(predicted + chosen).tolist() if applied else predicted.tolist(),
                global_score=float(result.score), global_boundary=bool(result.at_boundary),
                raw_columns=len(landmarks), selected_columns=len(selected),
                signal_q_min_median=None,
                global_columns=0, neighbor_columns=len(ids), regional_columns=0,
                local_votes=len(votes), local_boundary_votes=boundary,
                local_dispersion=local_dispersion, local_jackknife=local_jackknife,
                global_local_distance=agreement)
    if gradient is not None:
        output['gradient_score'] = gradient['score']
        output['gradient_prominence'] = gradient['prominence']
        output['gradient_boundary'] = gradient['boundary']
    output.update(entry_column_angle_deg=column_angle,
                  entry_reference_index_in_active=entry_reference,
                  entry_search_sigma_px=sigma,
                  entry_support_policy='nearest accepted contrast >0.5, search radius 2; central Z50 XY80 ROI')
    output['runtime_s'] = time.monotonic()-started
    return output


def _gradient_cc(model, observed, mask, radius, search_sigma=None):
    """Windowed gradient cross-correlation with axis-parabolic refinement."""
    from scipy.ndimage import sobel
    model = np.asarray(model, float); observed = np.asarray(observed, float)
    mask = np.asarray(mask, float)
    taper = np.outer(np.hanning(model.shape[0]), np.hanning(model.shape[1])) * mask
    cross = 0j
    for axis in (0, 1):
        fixed = sobel(model, axis)
        moving = sobel(observed, axis)
        fixed = (fixed - np.sum(fixed*taper)/max(taper.sum(), 1e-12))*taper
        moving = (moving - np.sum(moving*taper)/max(taper.sum(), 1e-12))*taper
        cross += np.conj(np.fft.fft2(fixed))*np.fft.fft2(moving)
    corr = np.fft.ifft2(cross).real
    offsets = np.arange(-radius, radius+1)
    surface = corr[(-offsets) % corr.shape[0]][:, (-offsets) % corr.shape[1]]
    if search_sigma is not None:
        from .entry import gaussian_search_surface
        surface = gaussian_search_surface(surface, offsets, search_sigma)
    peak = np.unravel_index(np.argmax(surface), surface.shape)
    correction = np.asarray(peak, float) - radius
    for axis in (0, 1):
        index = peak[axis]
        if 0 < index < surface.shape[axis] - 1:
            before = list(peak); after = list(peak)
            before[axis] -= 1; after[axis] += 1
            fm, f0, fp = surface[tuple(before)], surface[peak], surface[tuple(after)]
            curvature = fm - 2*f0 + fp
            if curvature < -1e-12:
                correction[axis] += float(np.clip(.5*(fm-fp)/curvature, -.5, .5))
    flat = np.sort(surface.ravel()); best = float(surface[peak])
    return dict(correction_yx=correction.tolist(), score=best,
                prominence=float(best-flat[-2]) if len(flat) > 1 else None,
                boundary=bool(np.any(np.abs(correction) >= radius-.05)))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _json_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'unsupported JSON value: {type(value).__name__}')


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=_json_numpy) + '\n')


def _write_mrcs(path: Path, stack, voxel) -> None:
    with mrcfile.new(str(path), overwrite=True) as h:
        h.set_data(np.asarray(stack, np.float32))
        h.voxel_size = voxel


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def run(original, angles, output, *, selector='bright_structure',
        landmark_support='accepted',
        global_margin_buffer=0.0, binning=8, voxel=None, resume=False,
        resample='exact', extra_rounds=0,
        raw_source=False, frozen_shifts_yx=None, axis_deg=None,
        search_support_diagnostics=False, diagnostic_source=None,
        solver_backend='cpu', cc_method='intensity',
        column_cell=32, column_width=30,
        entry_search_sigma_px=8., tomogram_thickness_percent=50., gcc_runner=None, exact_extra_rounds=False,stage_acceptor=None):
    """Run the progressive all-active 2D registration in-process.

    original: (T, Y, X) float32 bin8 stack (any tilt order).
    angles:   (T,) physical tilt angles matching ``original``.
    extra_rounds: backstop cap on extra evidence rounds over the accepted
    tilts after progressive expansion stops, including incomplete ranges.  Each round repeats the stage machinery from the current
    state (fresh all-active column landmarks + selection + GCC from the current
    cumulative shifts).  The loop runs until convergence: it stops at the
    first 'converged' round (the solver's no-improvement plateau — further
    rounds are no-ops), at the first non-accepted (safety) stop, or at the
    cap.  The default (0) preserves the production driver behaviour exactly.
    raw_source: True = maximum-FOV mode: ``original`` is the raw-source bin8
    stack (CTF + reliability in the raw MC raster, no frozen shift/rotation
    baked in) and frozen_shifts_yx (T, 2) + axis_deg feed the on-the-fly
    affine sampling (rotation + frozen + current shift at sample time, no
    baked edge fill).  Returns the trajectory list (also written to
    ``output/trajectory.json``).
    """
    gcc_runner = run_gcc if gcc_runner is None else gcc_runner
    entry_search_sigma_px = validate_sigma(entry_search_sigma_px)
    tomogram_thickness_percent = validate_percent(tomogram_thickness_percent)
    if (not isinstance(column_cell, int) or not isinstance(column_width, int)
            or column_cell <= 0 or column_width <= 0 or column_width > column_cell):
        raise ValueError('column cell/width must be positive integers, width <= cell')
    def detection_proxy(volume, directory):
        return volume

    if landmark_support not in ('accepted', 'all'):
        raise ValueError('landmark_support must be accepted or all')
    if raw_source:
        if frozen_shifts_yx is None or axis_deg is None:
            raise ValueError('raw_source requires frozen_shifts_yx and axis_deg')
        frozen = np.asarray(frozen_shifts_yx, float)
        if frozen.shape != (len(angles), 2):
            raise ValueError('frozen shifts must have shape (n_tilts, 2)')
        # the frozen-shifts csv is in the raw MC raster (fullres) px, while
        # this loop's raster is the BINNED raw stack: scale the translation
        # to loop units (translation commutes with the binning scale,
        # s_fullres -> s_fullres / binning)
        frozen = frozen / float(binning)
    else:
        frozen = None
    if voxel is None:
        voxel = 15.84 * float(binning)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    order = np.argsort(np.asarray(angles, float), kind='stable')
    original = np.asarray(original, np.float32)[order]
    angles = np.asarray(angles, float)[order]
    if frozen is not None:
        frozen = frozen[order]
    if len(angles) != original.shape[0]:
        raise ValueError('stack/angle mismatch')
    if np.any(np.diff(angles) <= 0):
        raise ValueError('physical angles must be strictly increasing')
    if diagnostic_source is not None and diagnostic_source != 'raw_materialized_control':
        raise ValueError("diagnostic_source must be 'raw_materialized_control'")
    if diagnostic_source is not None and not raw_source:
        raise ValueError('diagnostic_source requires raw_source')
    # the gcc source mode: 'treated' | 'raw' | diagnostic 'raw_materialized_control'
    source = ('raw_materialized_control' if diagnostic_source
              else 'raw' if raw_source else 'treated')
    # raw search-support diagnostic (zero effect on the fit): shared
    # accumulator across all run_gcc calls of this run
    search_support_diag = (dict(targets={})
                           if (search_support_diagnostics and source == 'raw') else None)
    t0 = original.shape[0]
    from .bootstrap import signal_tilt_indices
    usable = signal_tilt_indices(original)
    excluded_signal = np.setdiff1d(np.arange(t0),usable)
    if len(usable)<6:raise ValueError(f'At least six projections with finite, nonconstant signal are required; found {len(usable)}.')
    seed = usable[central_index_seed(angles[usable])]
    if len(excluded_signal):
        print('Excluded projections without finite, nonconstant signal: '+
              ', '.join(f'{int(i)} ({angles[i]:.3f} deg)' for i in excluded_signal),flush=True)
    active = seed.tolist()
    cumulative = np.zeros((t0, 2), float)
    config = dict(
        exact_extra_rounds=exact_extra_rounds,stage_continuation=stage_acceptor.config if stage_acceptor else None,
        seed_indices=active, seed_angles_deg=angles[seed].tolist(),
        expansion='independent neighbouring frontiers; retry blocked after opposite growth',
        input=('greco raw-source bin8 (CTF+reliability in raw MC raster; '
               'rotation + frozen + current shift applied on the fly, maximum '
               'FOV)' if raw_source
               else 'greco treated bin8 original (in-memory)'),
        raw_source=raw_source,
        raw_source_axis_deg=axis_deg,
        column_directions='seed for stage0; every proposed tilt afterwards (all-active)',
        column_selector=selector,
        landmark_support=landmark_support,
        cc_method=cc_method,
        solver_backend=solver_backend,
        qualification=('support tiers plus signal_q_min on accepted views; incoming views '
                       'admitted geometrically for registration' if landmark_support == 'accepted'
                       else 'support tiers plus signal_q_min'),
        cc=('windowed gradient cross-correlation, radius 32, preposition from two accepted same-side '
            'residuals' if cc_method == 'gradient' else
            'masked cross-correlation, radius 32, sigma 8, local votes, preposition from two accepted same-side residuals'),
        gcc='greco in-process full-support LOO generalized CC alpha .25 radius 4, on all proposed tilts every round',
        reconstruction='original treated images plus total cumulative shifts once, accepted tilts only',
        stage_acceptance='GCC convergence safety: three stable raster updates; stop otherwise',
        global_margin_headroom_buffer_px=global_margin_buffer,
        extra_rounds=extra_rounds,
        sha256=hashlib.sha256(json.dumps(
            {k: v for k, v in locals().items() if k not in (
                'sha256', 'detection_proxy', 'column_cell', 'column_width')},
            sort_keys=True, default=str).encode()).hexdigest(),
    )
    config['branch_policy']='independent_CC_frontiers_retry_after_opposite_growth_v1'
    config.update(optimizer='greco_optimizer', optimizer_version=1,
                  picking_roi_fraction_zyx=[.5,.8,.8], max_observable_gap_deg=12.,
                  qualification='central Z50 XY80 picking ROI; geometry-gap recovery; accepted local runs; true raw support; near-track consolidation preserving view pairs')
    config['sha256'] = hashlib.sha256(json.dumps(
        {k:v for k,v in config.items() if k!='sha256'},sort_keys=True,default=str).encode()).hexdigest()
    config.update(entry_candidates='nearest_accepted_columns_support_roi',
                  entry_search_sigma_px=entry_search_sigma_px,
                  tomogram_thickness_percent=tomogram_thickness_percent)
    config['sha256'] = hashlib.sha256(json.dumps(
        {k:v for k,v in config.items() if k!='sha256'},sort_keys=True,default=str).encode()).hexdigest()
    if (column_cell, column_width) != (32, 30):
        config['column_cell'] = column_cell
        config['column_width'] = column_width
        config['sha256'] = hashlib.sha256(
            json.dumps({k: v for k, v in config.items() if k != 'sha256'},
                       sort_keys=True, default=str).encode()).hexdigest()

        config['sha256'] = hashlib.sha256(
            json.dumps({k: v for k, v in config.items() if k != 'sha256'},
                       sort_keys=True, default=str).encode()).hexdigest()
    if len(excluded_signal):
        config['excluded_signal_indices']=excluded_signal.tolist()
        config['sha256']=hashlib.sha256(json.dumps(
            {k:v for k,v in config.items() if k!='sha256'},sort_keys=True,default=str).encode()).hexdigest()
    if (output / 'config.json').exists():
        saved_config = json.loads((output / 'config.json').read_text())
        for key, value, default in [('entry_search_sigma_px', entry_search_sigma_px, 8.),
                                    ('tomogram_thickness_percent', tomogram_thickness_percent, 50.),
                                    ('exact_extra_rounds',exact_extra_rounds,False),
                                    ('stage_continuation',stage_acceptor.config if stage_acceptor else None,None)]:
            if saved_config.get(key, default) != value:
                raise ValueError('resume optimizer setting mismatch: ' + key)
        if (saved_config.get('column_cell', 32), saved_config.get('column_width', 30)) != (column_cell, column_width):
            raise ValueError('resume column geometry mismatch')

    if (output / 'config.json').exists():
        config = json.loads((output / 'config.json').read_text())
    else:
        _write_json(output / 'config.json', config)

    from ..scipy_resample_gpu import shift_stack_scipy_gpu

    def commit_axis(state):
        nonlocal axis_deg
        if 'axis_rotation_deg' not in state:return
        updated=float(state['axis_rotation_deg'])
        if not np.isfinite(updated):raise ValueError('Nonfinite common raster angle')
        delta=np.deg2rad(updated-float(axis_deg))
        old=cumulative.copy()
        cumulative[:,0]=np.cos(delta)*old[:,0]-np.sin(delta)*old[:,1]
        cumulative[:,1]=np.sin(delta)*old[:,0]+np.cos(delta)*old[:,1]
        axis_deg=updated

    if raw_source:
        from ..gcc import materialize_raw_stack_gpu
        import cupy as cp
        raw_dev = cp.asarray(original)

        def _mat(sub, shifts, idx):
            # aligned raster = raw MC raster sampled on the fly with
            # rotation + frozen shift + the given shift (no baked fill)
            idx = np.asarray(idx, int)
            return materialize_raw_stack_gpu(raw_dev, idx, shifts,
                                             frozen[idx], axis_deg).get()
    else:
        def _mat(sub, shifts, idx=None):
            """Materialize a shifted subset (host output, as the consumers
            expect).

            'cpu'  -> scipy on the CPU (reference driver behaviour);
            otherwise the bit-exact GPU replica of that scipy resampling
            (greco/scipy_resample_gpu.py) with a single ~10 ms D2H copy.
            """
            if resample == 'cpu':
                return materialize_cumulative_stack(sub, shifts)
            import cupy as cp
            cvals = np.asarray([float(im.mean()) for im in sub], np.float64)
            return shift_stack_scipy_gpu(cp.asarray(sub), cp.asarray(shifts, np.float64),
                                         cp.asarray(cvals)).get()

    trajectory = []
    blocked_incoming = {}
    last_accepted_stage = None
    stage_no = 0
    while True:
        neighbours = [] if stage_no == 0 else incoming_pair(active, t0,eligible=usable).tolist()
        incoming = [t for t in neighbours if t not in blocked_incoming]
        if stage_no > 0 and not incoming:
            # Retry blocked frontiers only after the other branch has grown.
            incoming = [t for t in neighbours if blocked_incoming[t] < len(active)]
            if not incoming:
                break
        stage = output / f'stage_{stage_no:02d}'
        if stage.exists():
            if not resume:
                raise RuntimeError(f'existing stage requires explicit resume: {stage}')
            acceptance = stage / 'acceptance.json'
            if acceptance.exists():
                acc = json.loads(acceptance.read_text())
                if acc.get('accepted'):
                    geo = json.loads((stage / 'output_geometry.json').read_text())
                    active = np.asarray(geo['active_indices'], int).tolist()
                    cumulative = np.asarray(geo['cumulative_shifts_yx'], float)
                    blocked_incoming = {int(t): int(n) for t,n in geo.get('blocked_incoming',{}).items()}
                    with np.load(stage / 'final_state.npz') as checkpoint:
                        final_state = {key: np.array(checkpoint[key]) for key in checkpoint.files}
                    if 'axis_rotation_deg' in final_state:axis_deg=float(final_state['axis_rotation_deg'])
                    last_accepted_stage = stage
                    try:
                        trajectory.append(json.loads((stage / 'metrics.json').read_text()))
                    except (OSError, json.JSONDecodeError):
                        pass
                    if stage_acceptor is not None:
                        stage_acceptor.failures=int(acc.get('consecutive_stage_failures',0))
                        stage_acceptor.stopped=bool(acc.get('continuation_stopped',False))
                    if len(active) == t0 or (stage_acceptor is not None and stage_acceptor.stopped):
                        break
                    stage_no += 1
                    continue
            import shutil
            shutil.rmtree(stage)
        stage.mkdir()
        started = time.monotonic()
        current_stack = _mat(original[active], cumulative[active], active)
        volume = reconstruct_stage(gpu_reconstruct_wbp, current_stack, angles[active],
                                       tomogram_thickness_percent)
        picking_volume = detection_proxy(volume, stage)
        from ..selection import column_score_volume_gpu
        stage_score_volume = column_score_volume_gpu(volume, selector,
            thickness_percent=tomogram_thickness_percent)
        _write_json(stage / 'input_geometry.json',
                    dict(active_indices=active, cumulative_shifts_yx=cumulative.tolist()))
        cc = []
        proposed = active.copy()
        initial = cumulative[active].copy()
        if incoming:
            for target in incoming:
                predicted = predict_neighbor_shift(float(angles[target]), angles[active],
                                                   cumulative[active])
                image = _mat(original[target:target + 1],
                                                     predicted[None],
                                                     [target])[0]
                result = masked_cc_target(volume, current_stack, angles[active], image,
                                          angles[target], predicted, selector=selector,
                                          cc_method=cc_method,
                                          column_cell=column_cell, column_width=column_width,
                                          entry_search_sigma_px=entry_search_sigma_px,
                                          thickness_percent=tomogram_thickness_percent,
                                          score_volume=stage_score_volume)
                result['target_index'] = int(target)
                cc.append(result)
                if result['applied']:
                    proposed.append(target)
            proposed = sorted(proposed)
            initial = np.asarray([cumulative[i] if i in active else
                                  next(np.asarray(r['initial_shift_for_gcc_yx'])
                                       for r in cc if r['target_index'] == i)
                                  for i in proposed])
        _write_json(stage / 'masked_cc.json', cc)
        for result in cc:
            if not result['applied']:
                blocked_incoming[int(result['target_index'])] = len(active)
        if incoming and len(proposed) == len(active):
            del stage_score_volume
            reason = 'incoming_cc_rejected'
            decision=stage_acceptor.assess(dict(stop_reason=reason,history=[]),None) if stage_acceptor else {}
            _write_json(stage / 'acceptance.json', dict(accepted=False, reason=reason,**{k:v for k,v in decision.items() if k!='accepted'}))
            trajectory.append(dict(stage=stage_no, incoming_indices=incoming,
                                   incoming_angles_deg=angles[incoming].tolist(),
                                   active_count=len(active),
                                   angular_min=float(angles[active[0]]),
                                   angular_max=float(angles[active[-1]]),
                                   cc=cc, accepted=False, gcc_stop=None, reason=reason,
                                   **{k:v for k,v in decision.items() if k!='accepted'},
                                   runtime_s=time.monotonic() - started))
            _write_json(output / 'trajectory.json', trajectory)
            if stage_acceptor is not None and stage_acceptor.stopped:break
            stage_no += 1
            continue
        directions = stage_column_directions(angles, seed if stage_no == 0 else proposed, stage_no)
        selected_stack = _mat(original[proposed], initial, proposed)
        support_views = (np.isin(proposed, active) if landmark_support == 'accepted' else None)
        landmarks, sel_report = detect_column_landmarks(selected_stack, angles[proposed], picking_volume,
                                                     directions, selector=selector,
                                                     support_views=support_views,
                                                     column_cell=column_cell, column_width=column_width, cumulative_shifts_yx=initial if raw_source else None, frozen_shifts_yx=frozen[proposed] if raw_source else None, axis_deg=axis_deg if raw_source else None,
                                                     thickness_percent=tomogram_thickness_percent,
                                                     score_volume=stage_score_volume)
        del stage_score_volume
        np.savez_compressed(stage / 'selected_landmarks.npz', **landmarks)
        _write_json(stage / 'selection.json', sel_report)
        excluded = 0
        if global_margin_buffer > 0:
            excluded = enforce_global_margin_headroom(landmarks, angles[proposed],
                                                      original.shape[1], global_margin_buffer)
        _write_landmark_projections(stage, volume, landmarks, angles[proposed], voxel)
        candidate_state, gcc_manifest = gcc_runner(original[proposed], angles[proposed],
                                            landmarks, initial, resample=resample,
                                            source=source,
                                            frozen_shifts_yx=frozen[proposed] if raw_source else None,
                                            axis_deg=axis_deg,
                                            solver_backend=solver_backend,
                                            search_support_diag=search_support_diag,
                                            diag_plane_map=np.asarray(proposed, int))
        _write_json(stage / 'gcc_manifest.json', gcc_manifest)
        decision=stage_acceptor.assess(gcc_manifest,candidate_state) if stage_acceptor else {}
        accepted = decision['accepted'] if decision else accepted_stop_reason(gcc_manifest['stop_reason'])
        if accepted:
            commit_axis(candidate_state)
            final_state = candidate_state
            cumulative[proposed] = final_state['cumulative_shifts_yx']
            active = proposed
            last_accepted_stage = stage
            for target in active:
                blocked_incoming.pop(target, None)
        elif stage_no == 0:
            # Preserve the seed-failure output; later failed trials
            # must never overwrite the last accepted landmark state.
            commit_axis(candidate_state)
            final_state = candidate_state
        last = gcc_manifest['history'][-1] if gcc_manifest['history'] else {}
        record = dict(stage=stage_no,axis_rotation_deg=axis_deg, incoming_indices=incoming,
                      incoming_angles_deg=angles[incoming].tolist(),
                      active_count=len(active),
                      angular_min=float(angles[active[0]]),
                      angular_max=float(angles[active[-1]]),
                      cc=cc,
                      selected_count=sel_report['selected_count'],
                      candidate_count=sel_report['merged_count'],
                      global_margin_excluded=excluded,
                      signal_q_min_median=sel_report['signal_q_min_median'],
                      gcc_stop=gcc_manifest['stop_reason'],
                      gcc_passes=sum(bool(h.get('applied')) for h in gcc_manifest['history']),
                      boundary_count=last.get('boundary_observations'),
                      valid_observations=last.get('valid_observations'),
                      residual_median=last.get('median_after'),
                      residual_rms=last.get('rms_after'),
                      subset_ab=next((h['landmark_subset_agreement']
                                      for h in reversed(gcc_manifest['history'])
                                      if h.get('landmark_subset_agreement')), None),
                      accepted=accepted,
                      runtime_s=time.monotonic() - started,
                      timing_s=dict(selection=sel_report['runtime_s'],
                                    gcc=gcc_manifest['runtime_s']))
        record.update(decision)
        _write_json(stage / 'metrics.json', record)
        _write_json(stage / 'acceptance.json',
                    dict(accepted=accepted, reason=gcc_manifest['stop_reason'],**{k:v for k,v in decision.items() if k!='accepted'}))
        _write_json(stage / 'output_geometry.json',
                    dict(active_indices=active, cumulative_shifts_yx=cumulative.tolist(),
                         blocked_incoming=blocked_incoming))
        np.savez_compressed(stage / 'final_state.npz', **candidate_state)
        trajectory.append(record)
        _write_json(output / 'trajectory.json', trajectory)
        if not accepted or len(active) == t0:
            break
        stage_no += 1

    # --- optional extra evidence rounds (protocol extension) ---
    # After expansion stops, repeat the stage machinery using only the
    # accepted tilts, even for an incomplete range: reconstruct -> column landmarks ->
    # selection -> GCC starting from the current cumulative shifts.
    # Unaccepted rounds stop the loop without touching the state.
    if extra_rounds > 0 and last_accepted_stage is not None and not (stage_acceptor is not None and stage_acceptor.stopped):
        for r in range(extra_rounds):
            rdir = output / f'extra_round_{r:02d}'
            rdir.mkdir(exist_ok=True)
            started = time.monotonic()
            current_stack = _mat(original[active], cumulative[active], active)
            volume = reconstruct_stage(gpu_reconstruct_wbp, current_stack, angles[active],
                                       tomogram_thickness_percent)
            picking_volume = detection_proxy(volume, rdir)
            directions = stage_column_directions(angles, np.asarray(active, int), stage_no)
            landmarks, sel_report = detect_column_landmarks(current_stack, angles[active], picking_volume,
                                                         directions, selector=selector,
                                                         column_cell=column_cell, column_width=column_width, cumulative_shifts_yx=cumulative[active] if raw_source else None, frozen_shifts_yx=frozen[active] if raw_source else None, axis_deg=axis_deg if raw_source else None,
                                                     thickness_percent=tomogram_thickness_percent)
            np.savez_compressed(rdir / 'selected_landmarks.npz', **landmarks)
            _write_json(rdir / 'selection.json', sel_report)
            _write_landmark_projections(rdir, volume, landmarks, angles[active], voxel)
            round_state, gcc_manifest = gcc_runner(original[active], angles[active], landmarks,
                                                cumulative[active], resample=resample,
                                                source=source,
                                                frozen_shifts_yx=frozen[active] if raw_source else None,
                                                axis_deg=axis_deg,
                                                solver_backend=solver_backend,
                                                search_support_diag=search_support_diag,
                                                diag_plane_map=np.asarray(active, int))
            _write_json(rdir / 'gcc_manifest.json', gcc_manifest)
            decision=stage_acceptor.assess(gcc_manifest,round_state) if stage_acceptor else {}
            accepted = decision['accepted'] if decision else accepted_stop_reason(gcc_manifest['stop_reason'])
            if accepted:
                commit_axis(round_state)
                cumulative[active] = round_state['cumulative_shifts_yx']
                final_state = round_state
                last_accepted_stage = rdir
            last = gcc_manifest['history'][-1] if gcc_manifest['history'] else {}
            record = dict(stage='extra_round', round=r,axis_rotation_deg=axis_deg,
                          active_count=len(active),
                          angular_min=float(angles[active[0]]),
                          angular_max=float(angles[active[-1]]),
                          selected_count=sel_report['selected_count'],
                          candidate_count=sel_report['merged_count'],
                          gcc_stop=gcc_manifest['stop_reason'],
                          gcc_passes=sum(bool(h.get('applied')) for h in gcc_manifest['history']),
                          boundary_count=last.get('boundary_observations'),
                          valid_observations=last.get('valid_observations'),
                          residual_median=last.get('median_after'),
                          residual_rms=last.get('rms_after'),
                          accepted=accepted,
                          runtime_s=time.monotonic() - started,
                          timing_s=dict(selection=sel_report['runtime_s'],
                                        gcc=gcc_manifest['runtime_s']))
            record.update(decision)
            _write_json(rdir / 'metrics.json', record)
            _write_json(rdir / 'acceptance.json',
                        dict(accepted=accepted, reason=gcc_manifest['stop_reason'],**{k:v for k,v in decision.items() if k!='accepted'}))
            _write_json(rdir / 'output_geometry.json',
                        dict(active_indices=active, cumulative_shifts_yx=cumulative.tolist()))
            np.savez_compressed(rdir / 'final_state.npz', **round_state)
            trajectory.append(record)
            _write_json(output / 'trajectory.json', trajectory)
            if not accepted:
                break
            # until convergence: a 'converged' round IS the convergence signal
            # (the solver reached its no-improvement plateau from the current
            # state, so further rounds are no-ops — measured on the treated
            # reference: 3 'converged' rounds with flat residuals 0.355/
            # 0.348/0.376).  Keep running only while a round exhausts its
            # pass budget still improving ('max_passes'); N is the backstop.
            if gcc_manifest['stop_reason'] == 'converged' and not exact_extra_rounds:
                break

    # final products
    if search_support_diag is not None:
        report = {}
        for key, d in search_support_diag['targets'].items():
            key = int(key)
            clr = np.asarray(d['clearances'], float)
            report[str(key)] = dict(
                angle_deg=float(angles[key]),
                gcc_passes_observed=int(d['gcc_passes']),
                n_candidate_observations=int(d['n_candidate']),
                n_full_raw_support=int(d['n_candidate'] - d['n_raw_oob']),
                n_raw_search_oob=int(d['n_raw_oob']),
                oob_fraction_of_candidates=(float(d['n_raw_oob'] / d['n_candidate'])
                                            if d['n_candidate'] else None),
                raw_clearance_min_px=float(clr.min()) if len(clr) else None,
                raw_clearance_p10_px=float(np.quantile(clr, .10)) if len(clr) else None,
                raw_clearance_median_px=float(np.median(clr)) if len(clr) else None)
        _write_json(output / 'raw_search_support_diagnostic.json', report)

    final_stack = _mat(original[active], cumulative[active], active)
    _write_mrcs(output / 'aligned_bin8.mrcs', final_stack, float(voxel))
    np.savez_compressed(output / 'final_state.npz',
                        positions_zyx=final_state['positions_zyx'],
                        cumulative_shifts_yx=cumulative,
                        valid_views=final_state['valid_views'],
                        classes=final_state['classes'],
                        angles_deg=angles, active_indices=np.asarray(active, int),
                        binning=binning,**({"axis_rotation_deg":axis_deg} if axis_deg is not None else {}))
    sel_final = (last_accepted_stage or stage) / 'selected_landmarks.npz'
    if sel_final.exists():
        with np.load(sel_final) as d:
            np.savez_compressed(output / 'selected_landmarks_final.npz', **d)
    shifts = np.zeros((t0, 2), float)
    shifts[active] = cumulative[active]
    fullres = shifts * float(binning)
    with (output / 'shifts.csv').open('w') as h:
        h.write('stack_index,angle_deg,dy_binned,dx_binned,dy_fullres,dx_fullres\n')
        for i in range(t0):
            h.write(f'{i},{angles[i]:.6f},{shifts[i, 0]:.6f},{shifts[i, 1]:.6f},'
                    f'{fullres[i, 0]:.6f},{fullres[i, 1]:.6f}\n')
    _write_json(output / 'summary.json', dict(
        stage_continuation=stage_acceptor.config if stage_acceptor else None,
        consecutive_stage_failures=stage_acceptor.failures if stage_acceptor else 0,
        continuation_stopped=stage_acceptor.stopped if stage_acceptor else False,
        config_sha256=config['sha256'], stages=len(trajectory),final_axis_rotation_deg=axis_deg,
        extra_rounds_run=sum(1 for r in trajectory if r.get('stage') == 'extra_round'),
        active_indices=active, active_angles_deg=angles[active].tolist(),
        excluded_signal_indices=excluded_signal.tolist(),excluded_signal_angles_deg=angles[excluded_signal].tolist(),
        full_range=len(active) == t0, blocked_incoming=sorted(blocked_incoming),
        branch_policy='independent_CC_frontiers_retry_after_opposite_growth_v1',
        last_gcc_stop=trajectory[-1]['gcc_stop'] if trajectory else None,
        total_runtime_s=sum(r['runtime_s'] for r in trajectory)))
    return trajectory


def main(*, gcc_runner=None) -> int:
    p = argparse.ArgumentParser(description='greco in-process 2D all-active registration')
    p.add_argument('--stack', type=Path, required=True, help='treated bin8 mrcs stack')
    p.add_argument('--rawtlt', type=Path, required=True, help='one angle (deg) per line')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binning', type=int, default=8)
    p.add_argument('--selector', default='bright_structure',
                   choices=('bright_structure', 'point', 'spherical_mean',
                            'structure_tensor', 'bright_blob'))
    p.add_argument('--entry-search-sigma-px', type=float, default=8.)
    p.add_argument('--tomogram-thickness-percent', type=float, default=50.)
    p.add_argument('--global-margin-buffer', type=float, default=0.0)
    p.add_argument('--landmark-support', choices=('accepted', 'all'), default='accepted',
                   help='views used for landmark admission/ranking; accepted excludes '
                        'incoming images and admits their observations geometrically; '
                        'all scores every proposed view')
    p.add_argument('--resample', choices=['exact', 'cpu', 'fast'], default='exact',
                   help='resampling backend for the GCC loop and stage '
                        'materialization: exact = GPU bit-exact replica of the '
                        'reference scipy resampling (default); cpu = scipy on '
                        'the CPU (reference driver behaviour); fast = plain '
                        'CuPy (NOT bit-exact, ~2 px drift)')
    p.add_argument('--column-cell', type=int, default=32)
    p.add_argument('--column-width', type=int, default=30)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--solver-backend',
                   choices=['cpu', 'gpu_lsmr', 'gpu_lsmr32', 'gpu_schur'],
                   default='cpu', help='joint robust solver backend')
    p.add_argument('--cc-method', choices=['intensity', 'gradient'],
                   default='intensity', help='incoming-tilt pre-positioning CC')
    p.add_argument('--extra-rounds', type=int, default=0,
                   help='after expansion stops, run extra evidence rounds on accepted tilts '
                        '(fresh all-active landmark selection + GCC from the current '
                        'state) UNTIL CONVERGENCE: the loop stops at the first '
                        "'converged' round or a safety stop; N is only a backstop cap "
                        '(protocol extension; 0 = production behaviour, bit-exact)')
    p.add_argument('--raw-source', action='store_true',
                   help='maximum-FOV mode: --stack is the raw-source bin8 '
                        'stack (preprocess --raw-source, CTF+reliability in '
                        'the raw MC raster with no frozen shift/rotation) and '
                        'the rotation + frozen + current shift are applied on '
                        'the fly at sample time.  Requires --axis-deg and '
                        '--frozen-shifts (alignment/tilts.star).')
    p.add_argument('--axis-deg', type=float, default=None,
                   help='tilt axis angle (raster rotation) for --raw-source')
    p.add_argument('--frozen-shifts', type=Path, default=None,
                   help='per-plane frozen 2-D shifts csv (plane, angle_deg, '
                        'dy_raw, dx_raw) for --raw-source')
    p.add_argument('--search-support-diagnostics', action='store_true',
                   help='raw mode only: record per-plane GCC search-window raw '
                        'detector-support statistics (raw_search_support_diagnostic.json); '
                        'diagnostic only, zero effect on the fit')
    p.add_argument('--diagnostic-source', choices=['raw_materialized_control'], default=None,
                   help='DIAGNOSTIC: with --raw-source, materialize the raw stack to the '
                        'aligned raster each pass and use the treated aligned-raster '
                        'extraction path instead of the on-the-fly raw sampler (causal '
                        'isolation of sampler vs data; not a protocol)')
    a = p.parse_args()
    with mrcfile.open(str(a.stack), permissive=True) as h:
        original = np.asarray(h.data, np.float32).copy()
        voxel = float(h.voxel_size.x)
    angles = np.loadtxt(a.rawtlt, ndmin=1).astype(float)
    frozen = None
    if a.raw_source:
        if a.axis_deg is None:
            p.error('--raw-source requires --axis-deg')
        if a.frozen_shifts is None:
            p.error('--raw-source requires --frozen-shifts')
        rows = np.loadtxt(a.frozen_shifts, delimiter=',', skiprows=1)
        frozen = rows[:, 2:]
    trajectory = run(original, angles, a.output, selector=a.selector,
                     landmark_support=a.landmark_support,
                     global_margin_buffer=a.global_margin_buffer, binning=a.binning,
                     voxel=voxel, resume=a.resume, resample=a.resample,
                     extra_rounds=a.extra_rounds,
                     raw_source=a.raw_source, frozen_shifts_yx=frozen,
                     axis_deg=a.axis_deg,
                     search_support_diagnostics=a.search_support_diagnostics,
                     diagnostic_source=a.diagnostic_source,
                     solver_backend=a.solver_backend,
                     cc_method=a.cc_method,
                     column_cell=a.column_cell, column_width=a.column_width,
                     entry_search_sigma_px=a.entry_search_sigma_px,
                     tomogram_thickness_percent=a.tomogram_thickness_percent,gcc_runner=gcc_runner)
    print(f'greco 2D: {len(trajectory)} stages, '
          f'{sum(r["runtime_s"] for r in trajectory):.0f}s total, '
          f'full_range={trajectory[-1]["active_count"] == len(angles) if trajectory else False}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
