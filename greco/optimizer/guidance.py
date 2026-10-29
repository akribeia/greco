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

"""Greco Optimizer image guidance and bounded wide-range GCC rescue."""
import numpy as np

RESCUE_RADIUS_PX=16


def _adaptive_range_rescue(original,raw,angles,landmarks,cumulative,kwargs,should_rescue):
    budget=kwargs.get('max_passes',120);radius=4;history=[];segments=[];runtime=0.
    tracks=landmarks;shifts=np.array(cumulative,float,copy=True)
    while len(history)<budget:
        state,manifest=original(raw,angles,tracks,shifts,**dict(kwargs,search_radius=radius,max_passes=budget-len(history)))
        rows=manifest['history'];reason=manifest['stop_reason']
        segments.append(dict(search_radius_px=radius,passes=len(rows),stop_reason=reason))
        offset=len(history)
        history.extend(dict(row,iteration=offset+i+1,search_radius_px=radius) for i,row in enumerate(rows))
        runtime+=manifest['runtime_s']
        if not rows:break
        tracks=state;shifts=state['cumulative_shifts_yx']
        failed=reason in ('joint_solution_outside_support_or_shift_limit','insufficient_measured_observations')
        veto=should_rescue() if should_rescue is not None else True
        if failed and veto and radius<32:
            radius=16 if radius==4 else 32;continue
        if reason=='converged' and radius>4:
            radius=4;continue
        break
    if segments[-1]['search_radius_px']!=4 and manifest['stop_reason']=='converged':manifest['stop_reason']='max_passes'
    state['initial_cumulative_shifts_yx']=np.array(cumulative,float,copy=True)
    state['round_correction_shifts_yx']=state['cumulative_shifts_yx']-cumulative
    manifest.update(history=history,last=history[-1] if history else None,passes=len(history),runtime_s=runtime,
        range_rescue=dict(enabled=True,policy='adaptive_4_16_32_confirm_4',segments=segments))
    return state,manifest


def run_with_range_rescue(original, raw, angles, landmarks, cumulative, kwargs, *, should_rescue=None,rescue_state=None):
    """Retry actual measurements at +/-16, never reuse narrow-window votes.

    Normal mode follows a wide pass with normal GCC confirmations. Fast
    supplies shared rescue_state to stay wide across calls and converge
    there. All attempts share one pass budget.
    """
    if rescue_state is not None and rescue_state.get('adaptive'):
        return _adaptive_range_rescue(original,raw,angles,landmarks,cumulative,kwargs,should_rescue)
    budget=kwargs.get('max_passes',60);normal=kwargs.get('search_radius',4)
    tracks=landmarks;shifts=np.array(cumulative,float,copy=True)
    history=[];runtime=0.;radius=RESCUE_RADIUS_PX if rescue_state and rescue_state.get('active') else normal;segments=[]
    while len(history)<budget:
        remaining=budget-len(history)
        options=dict(kwargs,search_radius=radius,max_passes=1 if radius!=normal and rescue_state is None else remaining)
        state,manifest=original(raw,angles,tracks,shifts,**options)
        rows=manifest['history']
        segments.append(dict(search_radius_px=radius,passes=len(rows),stop_reason=manifest['stop_reason']))
        for row in rows:
            history.append(dict(row,iteration=len(history)+1,search_radius_px=radius))
        runtime+=manifest['runtime_s']
        if not rows:break
        tracks=state;shifts=state['cumulative_shifts_yx']
        failed=manifest['stop_reason']=='joint_solution_outside_support_or_shift_limit'
        peak=rows[-1].get('max_shift',np.nan)
        shift_veto=should_rescue() if should_rescue is not None else np.isfinite(peak) and peak>=normal
        if radius==normal and failed and shift_veto and normal<RESCUE_RADIUS_PX:
            radius=RESCUE_RADIUS_PX
            if rescue_state is not None:rescue_state['active']=True
            continue
        if rescue_state is None and radius!=normal and manifest['stop_reason']=='max_passes' and rows[-1].get('applied'):
            radius=normal;continue
        break
    if len(history)>=budget and manifest['stop_reason']!='converged':
        manifest['stop_reason']='max_passes' if history[-1].get('applied') else manifest['stop_reason']
    state['initial_cumulative_shifts_yx']=np.array(cumulative,float,copy=True)
    state['round_correction_shifts_yx']=state['cumulative_shifts_yx']-cumulative
    manifest.update(history=history,last=history[-1] if history else None,passes=len(history),runtime_s=runtime,
        range_rescue=dict(enabled=True,stay_in_rescue=rescue_state is not None,normal_radius_px=normal,rescue_radius_px=RESCUE_RADIUS_PX,segments=segments,fractional_steps=False))
    return state,manifest


def guided_gcc(original, strength=.25, rescue_enabled=True,*,rescue_state=None):
    if strength==0 and not rescue_enabled:return original
    import greco.gcc as gcc
    from .image_constraints import solve_joint_image_constraints
    from ..support_containment import solve_with_view_containment
    from ..curation_2d import diagnose
    def run_gcc(raw,angles,landmarks,cumulative,**kwargs):
        import cupy as cp
        current=np.array(cumulative,float,copy=True);frozen=kwargs['frozen_shifts_yx'];axis=kwargs['axis_deg']
        raw_dev=cp.asarray(raw);cache={};evidence=[];pass_number=0
        old=gcc._solve_with_local_containment
        def fit(votes,valid,classes,positions,angles,size,search_radius,robust_scale,max_nfev,tolerance,workspace=None,backend='cpu'):
            nonlocal pass_number
            pass_number+=1
            if strength==0:
                result=old(votes,valid,classes,positions,angles,size,search_radius,robust_scale,max_nfev,tolerance,workspace,backend)
                s=result[0]
                evidence.append(dict(iteration=pass_number,search_radius_px=search_radius,edges=[],
                    applied=bool(result[7]),joint_report=result[2],
                    shift_limit_violation=bool(np.isfinite(s).all() and np.max(np.abs(s))>=search_radius)))
                return result
            measured=not cache or pass_number%3==1 or cache.get('radius')!=search_radius
            if measured:
                aligned=gcc.materialize_raw_stack_gpu(raw_dev,np.arange(len(raw)),current,frozen,axis).get()
                diagnostic=diagnose(aligned,angles)
                pairs=[p for p in diagnostic['pairs'] if p.get('reliable') and 'smoothness_residual_yx_px' in p]
                edges=np.array([[p['left_index'],p['right_index']] for p in pairs],int).reshape(-1,2)
                targets=np.array([p['smoothness_residual_yx_px'] for p in pairs],float).reshape(-1,2)
                cache.update(edges=edges,absolute=targets+(current[edges[:,1]]-current[edges[:,0]]) if len(edges) else targets,
                    quality=np.array([min(1.,p['correlation']/.4) for p in pairs]),radius=search_radius)
            edges=cache['edges'];targets=cache['absolute']-(current[edges[:,1]]-current[edges[:,0]]) if len(edges) else cache['absolute']
            def observed(v,a,scale,budget,tol,workspace=None,backend='cpu'):
                supported=np.isfinite(v).all(2).any(0);ids=np.flatnonzero(supported);mapping=np.full(len(a),-1,int);mapping[ids]=np.arange(len(ids))
                keep=supported[edges].all(1) if len(edges) else np.zeros(0,bool)
                s,d,report=solve_joint_image_constraints(v[:,supported],a[supported],mapping[edges[keep]],targets[keep],strength,
                    scale,budget,tol,workspace,quality=cache['quality'][keep])
                expanded=np.zeros((len(a),2));expanded[supported]=s;return expanded,d,report
            result=solve_with_view_containment(votes,valid,classes,positions,angles,size,
                search_radius,robust_scale,max_nfev,tolerance,workspace,backend,solve=observed)
            s,_,report,_,_,_,_,safe,_=result
            evidence.append(dict(iteration=pass_number,search_radius_px=search_radius,
                edges=edges.tolist(),targets_yx_px=targets.tolist(),measured_on_this_pass=measured,
                applied=bool(safe),joint_report=report,
                shift_limit_violation=bool(np.isfinite(s).all() and np.max(np.abs(s))>=search_radius)))
            if safe:current[:]+=s
            return result
        options=dict(kwargs,containment_solver=fit)
        if rescue_enabled:
            state,manifest=run_with_range_rescue(original,raw,angles,landmarks,cumulative,options,
                should_rescue=lambda:bool(evidence and evidence[-1]['shift_limit_violation']),rescue_state=rescue_state)
        else:
            state,manifest=original(raw,angles,landmarks,cumulative,**options)
            manifest['range_rescue']=dict(enabled=False,normal_radius_px=kwargs.get('search_radius',4),
                rescue_radius_px=RESCUE_RADIUS_PX,fractional_steps=False)
        manifest['image_smoothness']=dict(protocol='local-tracks-image-smoothness-wide-rescue-v2',strength=strength,
            rescue_enabled=rescue_enabled,xyz_fixed=False,smooth_shift_curve=False,history=evidence)
        return state,manifest
    return run_gcc


def main():
    import os
    import greco.optimizer.progressive_alignment as driver
    enabled=os.environ.get('GRECO_RESCUE_ENABLED','1')
    if enabled not in ('0','1'):raise ValueError('Invalid rescue flag')
    runner=guided_gcc(driver.run_gcc,float(os.environ.get('GRECO_IMAGE_SMOOTHNESS','.25')),enabled=='1')
    return driver.main(gcc_runner=runner)


if __name__=='__main__':raise SystemExit(main())
