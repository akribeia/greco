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

"""Production-style full-step fit extended to changing roll/inclination.
Limit the joint step only to preserve tilt ordering. Remove unsupported view pairs and
refit; never accept a shift beyond the measured search radius.
"""
import time
import numpy as np
from greco.helpers import projected_centres
from .angular_solver import fit
from .. import gcc

class UnsafeStep(RuntimeError):pass

def fit_full_step(votes,valid,positions,angles,axis,shifts,frozen,size,radius,mode,*,solve=fit,raw_support=True,patch_radius=16):
    started=time.monotonic();work=np.array(votes,float,copy=True);support=np.array(valid,bool,copy=True);work[~support]=np.nan
    required=np.isfinite(work).all(2).any(0);removed=0;solves=0
    while True:
        active=np.isfinite(work).all(2).sum(1)>=4
        work[~active]=np.nan
        if active.sum()<2 or np.any(required & ~np.isfinite(work).all(2).any(0)):raise UnsafeStep('insufficient_landmarks_after_containment')
        ds,local_dr,da,dt,report=solve(work[active],positions[active],angles,size,mode);solves+=1
        dr=np.zeros_like(positions,dtype=float);dr[active]=local_dr
        if not all(np.isfinite(x).all() for x in [ds,dr,da,dt]):raise UnsafeStep('nonfinite_joint_update')
        if np.max(abs(ds))>=radius:raise UnsafeStep('shift_outside_search_radius')
        gaps=np.diff(angles);closing=np.diff(dt)
        if np.any(gaps<=0):raise UnsafeStep('initial_tilt_order_violation')
        step_fraction=1.
        threatened=closing<0
        if np.any(threatened):
            limit=float(np.min(gaps[threatened]/(-closing[threatened])))
            if limit<=1.:step_fraction=.9*limit
        if step_fraction<1e-8:raise UnsafeStep('tilt_order_step_stalled')
        raw_update=(ds,dr,da,dt)
        ds,dr,da,dt=(v*step_fraction for v in raw_update)
        new_angles=angles+dt
        if np.any(np.diff(new_angles)<=0):raise UnsafeStep('tilt_order_violation')
        new_axis=axis+da;r=np.deg2rad(da);new_shifts=np.empty_like(shifts,dtype=float)
        new_shifts[:,0]=np.cos(r)*shifts[:,0]-np.sin(r)*shifts[:,1]+ds[:,0]
        new_shifts[:,1]=np.sin(r)*shifts[:,0]+np.cos(r)*shifts[:,1]+ds[:,1]
        centres=projected_centres(positions+dr,new_angles,size)
        extent=patch_radius+radius
        inside=np.all(np.isfinite(centres)&(centres>=extent)&(centres<=size-1-extent),axis=2)
        if raw_support:
            by,bx=gcc.raw_sample_bases(centres,new_shifts,frozen,new_axis,size)
            inside &= gcc.raw_search_support(by,bx,extent,new_axis,size)
        bad=support & ~inside
        if not bad.any():
            report=dict(report,containment_removed_pairs=removed,containment_refits=solves,solve_time_s=time.monotonic()-started,step_fraction=step_fraction,min_tilt_gap_deg=float(np.min(np.diff(new_angles))))
            return *raw_update,report,work,support,active
        removed+=int(bad.sum());work[bad]=np.nan;support[bad]=False
        weak=np.isfinite(work).all(2).sum(1)<4;work[weak]=np.nan;support[weak]=False
