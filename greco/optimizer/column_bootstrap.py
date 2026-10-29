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

"""Landmark tracking for the Greco Optimizer column detector."""
import time
import numpy as np
from ..helpers import projected_centres
from .local_view_tracks import exact_pool_union, supported_tracks, track_diagnostics, consolidate_near_tracks
from ..landmark_tracks import longest_run
from ..selection import column_score_volume_gpu, find_column_maxima_gpu, rank_maxima_multiview


def detector_support(centres, size, cumulative_shifts_yx=None, frozen_shifts_yx=None, axis_deg=None):
    """Full GCC search footprints; geometry inputs use the stage's binning.

    The aligned square includes artificial fill after rotation/shifting. Raw
    geometry must be supplied together so that fill cannot qualify a track.
    """
    supplied=(cumulative_shifts_yx is not None, frozen_shifts_yx is not None, axis_deg is not None)
    geometric=np.all(np.isfinite(centres)&(centres>=20)&(centres<=size-21),axis=2)
    if not any(supplied):
        return geometric,geometric.copy()
    if not all(supplied):
        raise ValueError('Greco Optimizer requires cumulative shifts, frozen shifts and axis together')
    cumulative=np.asarray(cumulative_shifts_yx,float);frozen=np.asarray(frozen_shifts_yx,float)
    expected=(centres.shape[1],2)
    if cumulative.shape!=expected or frozen.shape!=expected or not np.isfinite(cumulative).all() or not np.isfinite(frozen).all() or not np.isfinite(axis_deg).all():
        raise ValueError('invalid local-track raw geometry')
    from ..gcc import raw_sample_bases,raw_search_support
    by,bx=raw_sample_bases(centres,cumulative,frozen,axis_deg,size)
    raw=raw_search_support(by,bx,20,axis_deg,size)
    return geometric,geometric & raw


def picking_roi_score(score):
    """Restrict picking before column argmax: central 50% Z, central 80% XY.

    Reconstruction and registration volumes are untouched. Use a finite
    negative sentinel to avoid infinity/zero interpolation producing NaNs.
    """
    shape=np.asarray(score.shape)
    low=np.ceil(shape*np.array([.25,.10,.10])).astype(int)
    high=np.floor(shape*np.array([.75,.90,.90])).astype(int)
    if np.any(high<=low):
        raise ValueError('volume too small for central picking ROI')
    masked=score.copy()
    for axis in range(3):
        slices=[slice(None)]*3;slices[axis]=slice(None,int(low[axis]))
        masked[tuple(slices)]=-1e30
        slices[axis]=slice(int(high[axis]),None)
        masked[tuple(slices)]=-1e30
    return masked,low,high


def detect_column_landmarks(stack, angles, volume, directions, *, selector='bright_structure',
                             min_view_score=.5, support_search_radius=2,
                             support_views=None, column_cell=32, column_width=30,
                             cumulative_shifts_yx=None, frozen_shifts_yx=None, axis_deg=None,
                             thickness_percent=None, score_volume=None):
    started=time.monotonic()
    angles=np.asarray(angles,float);directions=np.sort(np.asarray(directions,float))
    reference=np.ones(len(angles),bool) if support_views is None else np.asarray(support_views,bool)
    if reference.shape!=angles.shape or not reference.any():
        raise ValueError('support_views must select matching accepted views')
    score=(column_score_volume_gpu(volume,selector,thickness_percent=thickness_percent)
           if score_volume is None else score_volume)
    score,roi_low,roi_high=picking_roi_score(score)
    pools=[np.asarray(find_column_maxima_gpu(volume,float(a),selector=selector,score_volume=score,
                                           cell=column_cell,column_width=column_width,thickness_percent=thickness_percent),int)
           for a in directions]
    # Empty/intersecting columns must not manufacture an arbitrary sentinel
    # winner; discard them, never replace them with an outside-ROI point.
    clipped=[]
    for pool in pools:
        pool=pool[np.all((pool>=roi_low)&(pool<roi_high),axis=1)]
        peaks=score[tuple(pool.T)]
        peaks=peaks.get() if hasattr(peaks,'get') else np.asarray(peaks)
        clipped.append(pool[np.isfinite(peaks)&(peaks>-1e20)])
    pools=clipped
    merged,origins=exact_pool_union(pools)
    centres=projected_centres(merged,angles,stack.shape[-1])
    aligned_support,geometric=detector_support(centres,stack.shape[-1],cumulative_shifts_yx,frozen_shifts_yx,axis_deg)
    evidence=np.concatenate([rank_maxima_multiview(chunk,stack[reference],angles[reference],
        excluded_index=None,keep=len(chunk),min_support_fraction=0.,min_view_score=min_view_score,
        support_search_radius=support_search_radius, backend='gpu').per_view_scores
        for chunk in np.array_split(merged,(len(merged)+127)//128)]) if len(merged) else np.empty((0,reference.sum()))
    invalid=~geometric[:,reference]
    false_support=int(np.sum(invalid & np.isfinite(evidence) & (evidence>min_view_score)))
    # Mask before admission and persist unknown evidence as NaN, not a huge
    # contrast score obtained by dividing by the MAD of constant frame fill.
    evidence=np.where(invalid,np.nan,evidence)
    keep,valid=supported_tracks(evidence,angles,reference,geometric,threshold=min_view_score)
    ids=np.flatnonzero(keep);valid=valid[ids]
    before_count=len(ids);before_pairs=int(valid.sum())
    # Priority only picks a representative among <=2-voxel neighbours. It is
    # never a global ranking or a threshold that rejects weak local objects.
    if len(ids):
        peaks=score[tuple(merged[ids].T)]
        peaks=peaks.get() if hasattr(peaks,'get') else np.asarray(peaks)
        retained,groups,sizes=consolidate_near_tracks(merged[ids],valid,peaks)
    else:
        retained=groups=sizes=np.empty(0,int)
    groups=ids[groups]
    ids=ids[retained];valid=valid[retained]
    classes=np.where(valid.all(1),'global','local')
    full=np.full(valid.shape,np.nan);full[:,reference]=evidence[ids]
    landmarks=dict(positions_zyx=merged[ids],candidate_indices=ids,valid_views=valid,
        support_views=reference,classes=classes,angles_deg=angles,per_view_scores=full,
        source_direction_mask=origins[ids],detector_valid_views=geometric[ids],
        near_group_id=groups,near_group_size=sizes,
        support_fraction=valid[:,reference].mean(1),
        run_length=np.asarray([longest_run(row)[1] for row in valid[:,np.argsort(angles)]]),
        signal_q_min_score=np.full(len(ids),np.nan))
    report=dict(selector=selector,optimizer='greco_optimizer',optimizer_version=1,directions_deg=directions.tolist(),
        picking_roi_low_zyx=roi_low.tolist(),picking_roi_high_exclusive_zyx=roi_high.tolist(),
        picking_roi_fraction_zyx=[.5,.8,.8],max_observable_gap_deg=12.,
        geometry_missing_policy='skip unknown views; negative observable evidence breaks chains',
        column_cell=int(column_cell),column_width=int(column_width),
        raw_count=sum(map(len,pools)),merged_count=len(merged),selected_count=len(ids),
        selected_global=int(np.sum(classes=='global')),selected_local=int(np.sum(classes=='local')),
        global_budget=None,local_budget=None,min_accepted_run=4,
        support_view_indices=np.flatnonzero(reference).tolist(),min_view_score=float(min_view_score),
        detector_support='raw rotated search footprint' if axis_deg is not None else 'aligned raster',
        excluded_raw_pairs=int(np.sum(aligned_support & ~geometric)),
        excluded_contrast_supported_pairs=false_support,
        incoming_policy='at least 2 of 3 nearest geometrically observable accepted views within 12 degrees',
        merge_policy='exact XYZ union; <=2-voxel bounded groups; preserve all view-pair support',
        near_merge_radius_vox=2.,selected_before_near_merge=before_count,
        observations_before_near_merge=before_pairs,observations_after_near_merge=int(valid.sum()),
        near_duplicates_removed=before_count-len(ids),
        signal_q_min_median=None,runtime_s=time.monotonic()-started,**track_diagnostics(valid))
    return landmarks,report


def enforce_global_margin_headroom(landmarks, stage_angles, size, buffer):
    """Drop global tracks with too little clearance to the GCC detector margin.

    Operates on in-memory landmark dictionaries. Returns the number of
    excluded landmarks.
    """
    if buffer <= 0:
        return 0
    from ..selection import project_points_aligned_axis0
    positions = np.asarray(landmarks['positions_zyx'], float)
    valid = np.asarray(landmarks['valid_views'], bool)
    classes = np.asarray(landmarks['classes'])
    lo, hi = 20.0, size - 21.0
    keep = np.ones(len(positions), bool)
    excluded = []
    for i in np.flatnonzero(classes == 'global'):
        views = np.flatnonzero(valid[i])
        if not len(views):
            continue
        proj = np.stack([project_points_aligned_axis0(positions[i][None], float(stage_angles[t]), size)[0]
                         for t in views])
        clearance = np.minimum(np.minimum(proj[:, 0] - lo, proj[:, 1] - lo),
                               np.minimum(hi - proj[:, 0], hi - proj[:, 1]))
        c = float(clearance.min())
        if c < buffer:
            bad_view = int(views[int(np.argmin(clearance))])
            keep[i] = False
            excluded.append(dict(landmark=int(i), min_clearance_px=c, angle_deg=float(stage_angles[bad_view])))
    if keep.all():
        return 0
    for key in landmarks:
        if key in ('angles_deg', 'support_views'):
            continue
        arr = landmarks[key]
        if np.asarray(arr).ndim > 0 and np.asarray(arr).shape[0] == len(positions):
            landmarks[key] = np.asarray(arr)[keep]
    landmarks['excluded_global_margin'] = excluded
    return len(excluded)

