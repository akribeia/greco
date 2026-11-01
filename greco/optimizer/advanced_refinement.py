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

"""Advanced-mode angular refinement using shared image measurements."""
import time
import numpy as np
from .. import gcc
from ..helpers import projected_centres
from ..solver import projection_jacobians
from .angular_support import fit_full_step,UnsafeStep
from ..gcc_observations import measure_observations
from .advanced_options import PROTOCOL,STAGE_MAX_PASSES



class AdvancedRefiner:
    """Robust refinement with adaptive search and normal-radius confirmation."""
    def __init__(self,mode='common',max_passes=STAGE_MAX_PASSES,rescue_enabled=True,rescue_state=None,patch_radius=16):
        self.patch_radius=patch_radius
        self.calls=0;self.mode=mode;self.max_passes=max_passes;self.rescue_enabled=rescue_enabled
        self.rescue_state={} if rescue_state is None else rescue_state

    def __call__(self,original,angles,landmarks,cumulative,*,source='raw',frozen_shifts_yx=None,axis_deg=None,**kwargs):
        import cupy as cp
        if source!='raw':raise ValueError('Immutable raw source required')
        self.calls+=1
        angles=np.asarray(angles,float).copy();initial_angles=angles.copy()
        axis=np.broadcast_to(np.asarray(axis_deg,float),angles.shape).copy();initial_axis=axis.copy()
        positions=np.asarray(landmarks['positions_zyx'],float).copy();valid=np.asarray(landmarks['valid_views'],bool).copy()
        classes=np.array(landmarks['classes']);shifts=np.asarray(cumulative,float).copy();initial_shifts=shifts.copy()
        frozen=np.asarray(frozen_shifts_yx,float);size=original.shape[-1];raw=cp.asarray(original)
        med=1.;p90=2.5;angular_median_limit=1.;angular_p90_limit=2.5;angular_stop_factor=1.
        history=[];stable=0;radius=4;reason='max_passes';start=time.monotonic()
        first_rms=None;rms_radius=None
        for iteration in range(1,self.max_passes+1):
            stamp=time.monotonic();P=projection_jacobians(angles);centres=projected_centres(positions,angles,size)
            valid&=np.all(np.isfinite(centres)&(centres>=self.patch_radius+4)&(centres<=size-self.patch_radius-5),axis=2)
            valid[valid.sum(1)<4]=False
            if np.any(valid.sum(0)<8):reason='insufficient_supported_landmarks';break
            tracks=dict(positions_zyx=positions,valid_views=valid,classes=classes)
            measured_start=time.monotonic()
            measurement=dict(source='raw',frozen_shifts_yx=frozen,axis_deg=float(axis[0]) if self.mode in ('common','fixed','tilt_common') else axis,resample='exact',search_radius=radius)
            if self.patch_radius==16:
                votes=np.array(measure_observations(raw,angles,tracks,shifts,**measurement)['votes'],float)
            else:
                from ..tomography import gpu_batch_size
                patch_size=2*self.patch_radius+1
                batch=gpu_batch_size(len(positions),patch_size**3*256+len(angles)*patch_size**2*32)
                votes=np.full((len(positions),len(angles),2),np.nan)
                for start_batch in range(0,len(positions),batch):
                    end_batch=min(len(positions),start_batch+batch)
                    if not np.any(valid[start_batch:end_batch].sum(1)>=4):continue
                    subset={key:value[start_batch:end_batch] for key,value in tracks.items()}
                    votes[start_batch:end_batch]=measure_observations(raw,angles,subset,shifts,patch_radius=self.patch_radius,**measurement)['votes']
            image_s=time.monotonic()-measured_start
            observed=np.isfinite(votes).all(2);active=observed.sum(1)>=4;votes[~active]=np.nan;observed&=active[:,None]
            if active.sum()<2 or np.any(observed.sum(0)<8):
                if radius<32 and self.rescue_enabled:
                    radius=16 if radius==4 else 32;stable=0
                    print('RESCUE ESCALATE',radius,flush=True);continue
                reason='insufficient_measured_observations';break
            try:
                ds,dr,da,dt,report,votes,new_valid,active=fit_full_step(votes,valid,positions,angles,axis,shifts,frozen,size,radius,self.mode,**({'patch_radius':self.patch_radius} if self.patch_radius!=16 else {}))
            except UnsafeStep as error:
                if str(error)=='shift_outside_search_radius' and radius<32 and self.rescue_enabled:
                    radius=16 if radius==4 else 32;stable=0
                    print('RESCUE ESCALATE',radius,flush=True);continue
                reason=str(error);break
            valid=new_valid;observed=np.isfinite(votes).all(2)

            projected_dr=np.einsum('tca,na->ntc',P,dr);centred=centres-(size-1)/2
            roll_flow=np.stack([-centred[:,:,1],centred[:,:,0]],2)*np.deg2rad(da)[None,:,None]
            c=(size-1)/2;rad=np.deg2rad(angles)
            tilt_j=(np.sin(rad)[None]*(positions[:,2,None]-c)+np.cos(rad)[None]*(positions[:,0,None]-c))*np.pi/180
            tilt_flow=tilt_j*dt[None]
            predicted=ds[None]-projected_dr+roll_flow;predicted[:,:,1]+=tilt_flow
            scale=report.get("step_fraction",1.)
            if not np.isfinite(scale) or not all(np.isfinite(x).all() for x in [ds,dr,da,dt]):reason='nonfinite_joint_update';break
            next_angles=angles+dt*scale
            applied_da=da*scale;r=np.deg2rad(applied_da);sy=shifts[:,0].copy();sx=shifts[:,1].copy()
            shifts[:,0]=np.cos(r)*sy-np.sin(r)*sx+ds[:,0]*scale
            shifts[:,1]=np.sin(r)*sy+np.cos(r)*sx+ds[:,1]*scale
            positions+=dr*scale;axis+=applied_da;angles=next_angles
            norm=np.linalg.norm(ds,axis=1);rf=np.linalg.norm(roll_flow,axis=2)[observed];tf=abs(tilt_flow[observed])
            raster=bool(np.median(norm)<med and np.quantile(norm,.9)<p90)
            angular=bool(np.median(rf)*angular_stop_factor<angular_median_limit and np.quantile(rf,.9)*angular_stop_factor<angular_p90_limit and np.median(tf)*angular_stop_factor<angular_median_limit and np.quantile(tf,.9)*angular_stop_factor<angular_p90_limit)
            # A new search radius changes the available measurements. Compare
            # RMS only within the current contiguous measurement regime.
            if rms_radius!=radius:
                first_rms=report['rms_before'];rms_radius=radius
            improving=bool(report['rms_before']<=1.02*first_rms)
            stable=stable+1 if raster and angular and improving and scale==1. else 0
            next_radius=4 if stable>=3 and radius>4 else radius
            row=dict(rms_reference=float(first_rms),rms_stable=improving,next_search_radius_px=next_radius,rescue_exit=next_radius!=radius,stopping_median_px=med,stopping_p90_px=p90,angular_stop_factor=angular_stop_factor,angular_median_limit_px=angular_median_limit,angular_p90_limit_px=angular_p90_limit,applied=True,iteration=iteration,mode=self.mode,axis_rotation_deg=axis.copy(),angles_deg=angles.copy(),step_scale=float(scale),search_radius_px=radius,unscaled_rotation_update_deg=da,tilt_update_deg=dt*scale,median_unscaled_shift=float(np.median(norm)),p90_unscaled_shift=float(np.quantile(norm,.9)),median_shift=float(np.median(norm*scale)),p90_shift=float(np.quantile(norm*scale,.9)),max_shift=float(max(norm*scale)),median_unscaled_angular_flow_px=float(np.median(rf)),p90_unscaled_angular_flow_px=float(np.quantile(rf,.9)),median_unscaled_tilt_flow_px=float(np.median(tf)),p90_unscaled_tilt_flow_px=float(np.quantile(tf,.9)),median_landmark_update=float(np.median(np.linalg.norm(dr[active]*scale,axis=1))),median_after=report['median_after'],rms_after=report['rms_after'],landmark_subset_agreement=None,valid_observations=int(observed.sum()),boundary_observations=0,stable_passes=stable,raster_stable=raster,angular_stable=angular,image_measurement_s=image_s,solver_s=report['solve_time_s'],elapsed_s=time.monotonic()-stamp,solver_report=report)
            history.append(row)
            print('GPU JOINT',self.mode,self.calls,iteration,'axis',round(axis.mean(),6),'tilt_delta',np.round((angles-initial_angles)[[0,len(angles)//2,-1]],5).tolist(),'image_s',round(image_s,3),'solve_s',round(report['solve_time_s'],3),'scale',round(scale,4),'radius',radius,'next_radius',next_radius,'stable',stable,flush=True)
            if stable>=3:
                if radius>4:
                    radius=4;stable=0
                    print('RESCUE CONFIRM radius 4',flush=True)
                    continue
                reason='converged';break
        state=dict(positions_zyx=positions,valid_views=valid,classes=classes,cumulative_shifts_yx=shifts,initial_cumulative_shifts_yx=initial_shifts,round_correction_shifts_yx=shifts-initial_shifts,angles_deg=angles,axis_rotation_deg=float(axis[0]) if self.mode in ('common','fixed','tilt_common') else axis)
        # Fixed world grid measures camera changes independently of landmark reselection.
        centre=(size-1)/2
        z=centre+np.array([-.2,0.,.2])*size
        y=centre+np.array([-.3,0.,.3])*size
        x=centre+np.array([-.3,0.,.3])*size
        grid=np.stack(np.meshgrid(z,y,x,indexing='ij'),axis=-1).reshape(-1,3)
        before=projected_centres(grid,initial_angles,size)
        after=projected_centres(grid,angles,size)
        by0,bx0=gcc.raw_sample_bases(before,initial_shifts,frozen,initial_axis,size)
        by1,bx1=gcc.raw_sample_bases(after,shifts,frozen,axis,size)
        motion=np.hypot(by1-by0,bx1-bx0)
        geometry_change=dict(fixed_world_grid_points=len(grid),median_raw_motion_px=float(np.median(motion)),p90_raw_motion_px=float(np.quantile(motion,.9)),max_raw_motion_px=float(motion.max()),per_view_median_raw_motion_px=np.median(motion,axis=0).tolist(),axis_change_deg=(axis-initial_axis).tolist(),tilt_change_deg=(angles-initial_angles).tolist())
        manifest=dict(protocol=PROTOCOL,rescue_exit_policy='adaptive_4_16_32_confirm_4',rms_reference_policy='current_search_radius_segment',stopping_median_px=med,stopping_p90_px=p90,outer_round_index=kwargs.get('round_index'),outer_round_count=kwargs.get('round_count'),geometry_change=geometry_change,stop_reason=reason,passes=len(history),history=history,last=history[-1] if history else None,runtime_s=time.monotonic()-start,mode=self.mode,backend='gpu_schur_angular',final_axis_rotation_deg=state['axis_rotation_deg'])
        if reason=='insufficient_supported_landmarks':
            manifest['support_counts_per_view']=valid.sum(0).tolist()
        elif reason=='insufficient_measured_observations':
            manifest['support_counts_per_view']=observed.sum(0).tolist()
        return state,manifest
