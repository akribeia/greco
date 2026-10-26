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

"""Accepted-view-only landmark tracks; no per-landmark quality ranking."""
import numpy as np
from scipy.spatial import cKDTree


def exact_pool_union(pools):
    """Deduplicate identical XYZ and retain every contributing direction."""
    rows = [np.asarray(p).reshape(-1, 3) for p in pools]
    all_points = np.concatenate(rows) if rows else np.empty((0, 3))
    points, inverse = np.unique(all_points, axis=0, return_inverse=True)
    origins = np.zeros((len(points), len(rows)), bool)
    offset = 0
    for j, row in enumerate(rows):
        origins[inverse[offset:offset + len(row)], j] = True
        offset += len(row)
    return points.astype(int), origins


def supported_tracks(scores, angles, reference, geometric, *, threshold=.5,
                     min_run=4, neighbour_count=3, min_neighbours=2,
                     max_observable_gap_deg=12.):
    """Freeze masks before GCC; retain ALL supported angular runs.

    Incoming images never enter scores. Permit an incoming view only when
    at least two of its three nearest accepted views support the candidate.
    Geometry-missing views are unknown, not negative evidence. Genuine
    negative evidence and observable angular gaps >12 degrees break chains.
    """
    angles = np.asarray(angles, float)
    reference = np.asarray(reference, bool)
    geometric = np.asarray(geometric, bool)
    scores = np.asarray(scores, float)
    refs = np.flatnonzero(reference)
    refs = refs[np.argsort(angles[refs], kind='stable')]
    if scores.shape != (len(geometric), reference.sum()):
        raise ValueError('scores must contain accepted views only')
    if min_run < 4:
        raise ValueError('joint XYZ tracks require at least four accepted views')
    supported = np.zeros_like(geometric)
    supported[:, reference] = np.isfinite(scores) & (scores > threshold)
    supported &= geometric
    valid = np.zeros_like(geometric)
    for k in range(len(geometric)):
        observed = refs[geometric[k, refs]]
        # Split on long missing stretches even if both endpoints are positive.
        breaks = np.flatnonzero(np.diff(angles[observed]) > max_observable_gap_deg) + 1
        for segment in np.split(observed, breaks):
            row = supported[k, segment]
            edges = np.diff(np.r_[False, row, False].astype(int))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if end - start >= min_run:
                    valid[k, segment[start:end]] = True
    for t in np.flatnonzero(~reference):
        ordered = refs[np.argsort(np.abs(angles[refs] - angles[t]), kind='stable')]
        ordered = ordered[np.abs(angles[ordered]-angles[t]) <= max_observable_gap_deg]
        observable = geometric[:, ordered]
        nearest = observable & (np.cumsum(observable, axis=1) <= neighbour_count)
        valid[:, t] = ((valid[:, ordered] & nearest).sum(1) >= min_neighbours) & geometric[:, t]
    keep = valid[:, reference].sum(1) >= min_run
    return keep, valid


def consolidate_near_tracks(positions, valid, priority, *, radius=2.):
    """Reduce near-identical points without inventing a single observation.

    Groups have a bounded diameter (no transitive single-linkage chains).
    Retain actual scored XYZ and their original complete track masks until
    every within-group view pair is represented. Thus disjoint local chains
    and bridging chains survive; only redundant nearby measurements disappear.
    """
    positions = np.asarray(positions, float)
    valid = np.asarray(valid, bool)
    priority = np.asarray(priority, float)
    if positions.shape != (len(valid), 3) or priority.shape != (len(valid),) or radius < 0:
        raise ValueError('invalid near-track consolidation inputs')
    if not len(valid):
        return np.empty(0, int), np.empty(0, int), np.empty(0, int)
    # Coordinates resolve equal-priority ties independently of input order.
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0], -np.nan_to_num(priority)))
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    tree = cKDTree(positions)
    assigned = np.zeros(len(valid), bool)
    retained, groups, sizes = [], [], []
    for anchor in order:
        if assigned[anchor]:
            continue
        neighbours = np.asarray(tree.query_ball_point(positions[anchor], radius), int)
        neighbours = neighbours[~assigned[neighbours]]
        neighbours = neighbours[np.argsort(rank[neighbours])]
        members = []
        for candidate in neighbours:
            if not members or np.all(np.linalg.norm(positions[members]-positions[candidate], axis=1) <= radius):
                members.append(int(candidate))
        assigned[members] = True
        covered = np.zeros((valid.shape[1], valid.shape[1]), bool)
        for candidate in members:
            pairs = valid[candidate, :, None] & valid[candidate, None, :]
            if np.any(pairs & ~covered):
                retained.append(candidate)
                groups.append(int(anchor))
                sizes.append(len(members))
                covered |= pairs
    sorting = np.argsort(retained)
    return np.asarray(retained)[sorting], np.asarray(groups)[sorting], np.asarray(sizes)[sorting]


def track_diagnostics(valid):
    """Observation-graph connectivity is diagnostic, never a hidden veto."""
    valid = np.asarray(valid, bool)
    adjacency = (valid.astype(np.int64).T @ valid.astype(np.int64)) > 0
    remaining = set(range(valid.shape[1]))
    components = []
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            t = pending.pop()
            if t in component:
                continue
            component.add(t)
            pending.extend(set(np.flatnonzero(adjacency[t])) - component)
        remaining -= component
        components.append([int(t) for t in sorted(component)])
    return dict(active_landmarks_per_view=valid.sum(0).tolist(),
                observation_components=components,
                unsupported_views=np.flatnonzero(~valid.any(0)).tolist())
