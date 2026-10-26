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

"""Greco Optimizer entry alignment using accepted-view landmark support."""
import numpy as np
from ..selection import rank_maxima_multiview, project_points_aligned_axis0


def validate_sigma(sigma):
    if isinstance(sigma, bool):
        raise ValueError('Entry sigma must be finite and positive.')
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Entry sigma must be finite and positive.')
    return sigma


def gaussian_search_surface(surface, offsets, sigma):
    """Spatial prior, not a calibrated likelihood; remove a negative floor."""
    sigma = validate_sigma(sigma)
    surface = np.asarray(surface, float)
    offsets = np.asarray(offsets, float)
    weights = np.exp(-(offsets[:, None]**2 + offsets[None, :]**2)/(2*sigma**2))
    return (surface - min(0., float(surface.min()))) * weights


def nearest_reference(angles, target):
    angles = np.asarray(angles, float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError('Finite nonempty accepted angles required.')
    return int(np.argmin(np.abs(angles - float(target))))


def reference_support(points, image, angle, threshold=.5, *, backend='cpu'):
    points = np.asarray(points).reshape(-1, 3)
    if not len(points):
        return np.zeros(0, bool)
    scores = np.concatenate([
        rank_maxima_multiview(chunk, np.asarray(image)[None], [angle],
            excluded_index=None, keep=len(chunk), min_support_fraction=0.,
            min_view_score=threshold, support_search_radius=2, backend=backend).per_view_scores[:, 0]
        for chunk in np.array_split(points, (len(points)+127)//128)])
    return np.isfinite(scores) & (scores > threshold)


def entry_indices(points, reference, reference_angle, target_angle, target_shape, *, backend='cpu'):
    """Accepted-view evidence only; target is checked geometrically, not scored."""
    ids = reference_support(points, reference, reference_angle, backend=backend)
    for angle, shape, margin in ((reference_angle, reference.shape, 20),
                                 (target_angle, target_shape, 32)):
        projected = project_points_aligned_axis0(points, float(angle), shape[-1])
        ids &= np.isfinite(projected).all(1)
        ids &= ((projected >= margin) &
                (projected <= np.asarray(shape[-2:])-1-margin)).all(1)
    return np.flatnonzero(ids)


def select_entry_landmarks(volume, active_stack, active_angles, target_angle,
                        target_shape, *, selector='bright_structure',
                        thickness_percent=50., score_volume=None, column_cell=32, column_width=30):
    """Select columns supported by the nearest accepted tilt."""
    from .column_bootstrap import picking_roi_score
    from ..selection import column_score_volume_gpu, find_column_maxima_gpu
    reference = nearest_reference(active_angles, target_angle)
    column_angle = float(active_angles[reference])
    score, low, high = picking_roi_score(score_volume if score_volume is not None else
        column_score_volume_gpu(volume, selector, thickness_percent=thickness_percent))
    landmarks = find_column_maxima_gpu(volume, column_angle, selector=selector,
                                   score_volume=score, thickness_percent=thickness_percent,
                                   cell=column_cell, column_width=column_width)
    landmarks = np.asarray(landmarks, int).reshape(-1, 3)
    landmarks = landmarks[np.all((landmarks >= low) & (landmarks < high), axis=1)]
    if len(landmarks):
        peaks = score[tuple(landmarks.T)]
        peaks = peaks.get() if hasattr(peaks, 'get') else np.asarray(peaks)
        landmarks = landmarks[np.isfinite(peaks) & (peaks > -1e20)]
    ids = entry_indices(landmarks, active_stack[reference], column_angle,
                        target_angle, target_shape, backend='gpu')
    return landmarks, ids, column_angle, reference
