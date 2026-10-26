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

"""Deterministic center-out scheduling and shift initialization."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike,NDArray


def central_index_seed(angles_deg:ArrayLike)->NDArray[np.int32]:
    """Centre on the median tilt index: seven for odd N, six for even N."""
    a=np.asarray(angles_deg,float).reshape(-1)
    if len(a)<6 or np.any(np.diff(a)<=0):raise ValueError('at least six sorted physical tilts required')
    width=7 if len(a)%2 else 6
    start=(len(a)-width)//2
    return np.arange(start,start+width,dtype=np.int32)

def signal_tilt_indices(stack,minimum_std=1e-6):
    """Keep finite, nonconstant projections without renumbering input tilts."""
    stack=np.asarray(stack)
    if stack.ndim!=3:raise ValueError('Expected a tilt stack.')
    return np.asarray([i for i,image in enumerate(stack)
                       if np.isfinite(image).all() and np.std(image,dtype=np.float64)>=minimum_std],np.int32)


def incoming_pair(active:ArrayLike,total_tilts:int,eligible=None)->NDArray[np.int32]:
    """Immediate angular neighbours of a contiguous accepted interval."""
    ids=np.asarray(active,int)
    if eligible is not None:
        usable=np.asarray(eligible,int)
        if (not len(usable) or np.any(np.diff(usable)<=0) or usable[0]<0 or usable[-1]>=total_tilts
                or not len(ids) or not np.isin(ids,usable).all()):
            raise ValueError('Invalid eligible tilt identities')
        positions=np.searchsorted(usable,ids)
        return usable[incoming_pair(positions,len(usable))].astype(np.int32)
    if not len(ids) or not np.array_equal(ids,np.arange(ids[0],ids[-1]+1)):
        raise ValueError('active tilts must be a contiguous angle-ordered interval')
    out=[]
    if ids[0]>0:out.append(int(ids[0]-1))
    if ids[-1]<total_tilts-1:out.append(int(ids[-1]+1))
    return np.asarray(out,np.int32)

def stage_column_directions(angles_deg:ArrayLike,proposed_indices:ArrayLike,stage_no:int)->NDArray[np.float64]:
    """Return all physical directions available to a fresh stage."""
    angles=np.asarray(angles_deg,float).reshape(-1);ids=np.asarray(proposed_indices,int).reshape(-1)
    if np.any(ids<0) or np.any(ids>=len(angles)):raise ValueError('proposed indices out of range')
    return angles[np.sort(ids)].copy()


def predict_neighbor_shift(angle:float,active_angles:ArrayLike,active_shifts_yx:ArrayLike)->NDArray[np.float64]:
    """Linearly extrapolate from two same-side neighbours, else nearest."""
    aa=np.asarray(active_angles,float);ss=np.asarray(active_shifts_yx,float)
    if len(aa)!=len(ss) or ss.shape[1:]!=(2,):raise ValueError('angle/shift mismatch')
    side=np.flatnonzero(aa<angle) if angle>np.max(aa) else np.flatnonzero(aa>angle) if angle<np.min(aa) else np.array([],int)
    if len(side)>=2:
        near=side[np.argsort(np.abs(aa[side]-angle),kind='stable')[:2]]
        near=near[np.argsort(aa[near])]
        x0,x1=aa[near];y0,y1=ss[near]
        if x1!=x0:return y1+(angle-x1)*(y1-y0)/(x1-x0)
    nearest=int(np.argmin(np.abs(aa-angle)))
    return ss[nearest].copy()

def accepted_stop_reason(reason:str)->bool:
    """Existing solver safety: only applied-safe exits may grow the active set."""
    return reason == 'converged'
