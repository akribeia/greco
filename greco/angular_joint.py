"""Rebuild LOO references after each shared rigid angular update (bin8)."""
from greco.volume_storage import float16_data
from greco.project_layout import join as storage_path, ensure_output
from greco.preprocessing_star import read_table

import json
from pathlib import Path
import numpy as np


def rescale_geometry(positions,shifts,source_size,target_size,axis,flip_y=False):
    from .gcc import raw_affine
    factor=target_size/source_size
    delta=(target_size-1)/2-factor*(source_size-1)/2
    if np.ndim(axis):
        angles=np.deg2rad(np.asarray(axis,float));co=np.cos(angles);si=np.sin(angles)
        raw_offset=np.array([factor-1 if flip_y else 0,0.])
        v=np.full(2,delta)-raw_offset
        offsets=np.column_stack((co*v[0]-si*v[1],si*v[0]+co*v[1]))
        return np.asarray(positions)*factor+delta,np.asarray(shifts)*factor+offsets
    a,b,c,d,_=raw_affine(axis,target_size);rotation=np.array([[a,b],[c,d]])
    raw_offset=np.array([factor-1 if flip_y else 0,0.])
    offset=rotation.T@(np.full(2,delta)-raw_offset)
    return np.asarray(positions)*factor+delta,np.asarray(shifts)*factor+offset


def aligned_planes(stack,shifts,frozen,axis,xp):
    from cupyx.scipy.ndimage import map_coordinates
    n=stack.shape[-1];y,x=xp.mgrid[:n,:n].astype(xp.float64);centre=(n-1)/2;a=np.deg2rad(axis)
    for t in range(len(stack)):
        py=y-shifts[t,0]-centre;px=x-shifts[t,1]-centre
        ry=np.cos(a)*py+np.sin(a)*px+centre-frozen[t,0]
        rx=-np.sin(a)*py+np.cos(a)*px+centre-frozen[t,1]
        yield map_coordinates(stack[t],xp.stack((ry,rx)),order=1,mode='constant',cval=0.)


def main(config,output,refine_xyz=False,resume=None,reconstruct=False,xyz_total_limit=8.,max_passes=None,angle_options=None):
    import cupy as cp
    import torch
    import mrcfile
    from greco.web.refinement_worker import load_state,clean_json
    from greco.rigid_angle import freeze_references,RigidImageObjective,safe_pairs
    data=json.loads(config.read_text());ensure_output(output)
    original=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
    pos,shifts,valid,classes,angles,b=load_state(storage_path(Path(data['coarse_run']), 'final_state.npz'),original)
    if b!=8 or data['inputs']['binning']!=8:raise ValueError('Bin8 required')
    target_bin=int((angle_options or {}).get('binning',8));source_bin=b
    with mrcfile.mmap(data['inputs']['stack']) as m:
        source_size=m.data.shape[-1];voxel=tuple(float(getattr(m.voxel_size,k)) for k in ('x','y','z'))
        stack=cp.asarray(m.data,dtype=cp.float32) if target_bin==8 else None
    source_positions=pos.copy();source_shifts=shifts.copy()
    rows=read_table(data['inputs']['frozen_shifts'])
    if len(rows)!=len(angles) or not np.allclose([float(r['angle_deg']) for r in rows],angles,atol=1e-4):raise ValueError('Angle ordering')
    axis=original_axis=float(data['orientation']['rotation_deg'])
    if target_bin==8:
        frozen=np.array([[float(r['dy_raw']),float(r['dx_raw'])] for r in rows])/8
        if data['orientation'].get('flip_y'):stack=stack[:,::-1,:].copy();frozen[:,0]*=-1
    else:
        from greco.web.refinement_worker import rebuild_native
        native_config=dict(data,options=dict(data['options'],output_binning=target_bin))
        stack,frozen,native_pixel=rebuild_native(native_config,angles)
        voxel=(native_pixel,)*3
        pos,shifts=rescale_geometry(pos,shifts,source_size,stack.shape[-1],axis,data['orientation'].get('flip_y',False))
    if stack.ndim!=3 or stack.shape[0]!=len(angles) or stack.shape[1]!=stack.shape[2]:raise ValueError('Invalid geometry stack shape')
    scale=source_bin/target_bin;original_shifts=shifts.copy();original_pos=pos.copy()

    angle_options=angle_options or {}
    axis_free=angle_options.get('refine_axis',True);tilts_free=angle_options.get('refine_tilts',True)
    axis_bound=float(angle_options.get('axis_bound_deg',3. if refine_xyz else 1.)) if axis_free else 0.
    tilt_bound=float(angle_options.get('tilt_bound_deg',2. if refine_xyz else .5)) if tilts_free else 0.
    shift_bound=(8. if refine_xyz else 2.)*scale
    radius=(int(angle_options.get('patch_size',33))-1)//2
    # Fix support for the ENTIRE original bounded search, not differently each pass.
    valid=safe_pairs(pos,angles,axis,shifts,frozen,valid,stack.shape[-1],radius=radius+int(np.ceil(12*scale)) if refine_xyz else radius,
        axis_bound=axis_bound,tilt_bound=tilt_bound,shift_bound=shift_bound)
    first_pass=1;last_case=None
    if resume is not None:
        with np.load(resume,allow_pickle=False) as state:
            if float(state['binning'])!=target_bin or not np.array_equal(state['active_indices'],np.arange(len(angles))):
                raise ValueError('Resume must be a full-range bin8 geometry')
            if state['positions_zyx'].shape!=pos.shape or not np.array_equal(state['valid_views'],valid):
                raise ValueError('Resume landmark/support mismatch')
            pos=np.array(state['positions_zyx'],float);angles=np.array(state['angles_deg'],float)
            shifts=np.array(state['cumulative_shifts_yx'],float);axis=float(state['axis_rotation_deg'])
        if not all(np.isfinite(a).all() for a in (pos,angles,shifts)) or not np.isfinite(axis):raise ValueError('Nonfinite resume')
        if abs(axis-original_axis)>axis_bound or np.max(np.abs(angles-original))>tilt_bound or np.max(np.abs(shifts-original_shifts))>shift_bound:
            raise ValueError('Resume exceeds ORIGINAL geometry limits')
        first_pass=int(resume.parent.name.split('_')[-1])+1
        last_case=resume.parent
    history=[];stable=0;stop='maximum_passes'
    for iteration in range(first_pass,first_pass+(max_passes if max_passes is not None else (12 if refine_xyz else 6))):
        print(f'Alternating angular pass {iteration}',flush=True)
        estimated=int(valid.sum())*((2*radius+1)**3*4+(2*radius+1)**2*24)
        free,_=cp.cuda.runtime.memGetInfo()
        if estimated>free*.65:raise ValueError('Landmark reference bank exceeds GPU memory; choose a smaller patch or a coarser bin.')
        bank=freeze_references(stack,pos,angles,axis,shifts,frozen,valid,radius=radius,fixed_safe=True)
        transfer=lambda x:torch.utils.dlpack.from_dlpack(x)
        obj=RigidImageObjective(transfer(stack),pos,angles,axis,shifts,frozen,bank['k'],bank['t'],transfer(bank['templates']),transfer(bank['derivatives']),refine_xyz=refine_xyz,translation_scale=2*scale)
        bounds=[(max(-1,(-shift_bound-d)/(2*scale)),min(1,(shift_bound-d)/(2*scale))) for d in (shifts-original_shifts).ravel()]
        if axis_free:bounds.append((max(-.5,original_axis-axis_bound-axis),min(.5,original_axis+axis_bound-axis)))
        if tilts_free:bounds.extend((max(-1,(original[t]-tilt_bound-angles[t])/.5),min(1,(original[t]+tilt_bound-angles[t])/.5)) for t in obj.tilt_ids)
        if refine_xyz:bounds.extend([(-1,1)]*pos.size)
        p,result=obj.fit(axis_free,tilts_free,bounds=bounds,maxiter=200 if refine_xyz else 100)
        q=torch.tensor(p,dtype=torch.float64,device='cuda');zero=torch.zeros_like(q)
        exact=obj.exact_templates(bank,angles+result['tilt_changes_deg'])
        with torch.no_grad():
            before=float(obj.correlations(zero,axis_free,tilts_free).mean().cpu())
            after=float(obj.correlations(q,axis_free,tilts_free,exact_templates=exact).mean().cpu())
            held_before=float(obj.correlations(zero,axis_free,tilts_free,heldout=True).mean().cpu())
            held_after=float(obj.correlations(q,axis_free,tilts_free,heldout=True,exact_templates=exact).mean().cpu())
        result.update(pass_number=iteration,exact_score_before=before,exact_score_after=after,heldout_before=held_before,heldout_after=held_after)
        history.append(result)
        if xyz_total_limit is not None and np.max(np.linalg.norm(pos+result['xyz_changes_zyx']-original_pos,axis=1))>xyz_total_limit:
            stop='xyz_trust_region_rejected';break
        if after<before-1e-6:
            stop='exact_projection_rejected';break
        shifts+=result['shift_changes_yx'];angles+=result['tilt_changes_deg'];axis=result['axis_rotation_deg']
        pos+=result['xyz_changes_zyx']
        case=storage_path(output, f'pass_{iteration:02d}');case.mkdir()
        last_case=case
        np.savez_compressed(storage_path(case, 'geometry.npz'),positions_zyx=pos,cumulative_shifts_yx=shifts,valid_views=valid,classes=classes,
            angles_deg=angles,active_indices=np.arange(len(angles)),binning=target_bin,axis_rotation_deg=axis)
        small=abs(result['axis_change_deg'])<.002 and np.max(np.abs(result['tilt_changes_deg']))<.005 and np.max(np.linalg.norm(result['shift_changes_yx'],axis=1))<.01*scale
        if refine_xyz:small=small and np.max(np.linalg.norm(result['xyz_changes_zyx'],axis=1))<.02*scale
        stable=stable+1 if small else 0
        report=dict(history=history,axis_rotation_deg=axis,axis_change_deg=axis-original_axis,tilt_changes_deg=angles-original,
            shift_changes_yx=shifts-original_shifts,xyz_changes_zyx=pos-original_pos,xyz_fixed=not refine_xyz,references_rebuilt_each_pass=True,
            stop_reason='stable' if stable>=2 else stop,
            caution='Alternating conditional image fit, not an independently validated maximum-likelihood reconstruction. Cloud rigid modes excluded per pass; axis common; no independent in-plane rotation per tilt.')
        (storage_path(output, 'report.json')).write_text(json.dumps(clean_json(report),indent=2,allow_nan=False))
        print(json.dumps(clean_json(report)),flush=True)
        del exact,obj,bank
        cp.get_default_memory_pool().free_all_blocks()
        if stable>=2:break
    # Record termination even when a candidate step was rejected. Reconstruct
    # the LAST ACCEPTED geometry, never the rejected candidate.
    report=dict(history=history,resume=str(resume) if resume else None,axis_rotation_deg=axis,
        axis_change_deg=axis-original_axis,tilt_changes_deg=angles-original,shift_changes_yx=shifts-original_shifts,
        xyz_changes_zyx=pos-original_pos,xyz_fixed=not refine_xyz,references_rebuilt_each_pass=True,
        stop_reason='stable' if stable>=2 else stop,xyz_total_limit=xyz_total_limit,
        last_accepted_geometry=str(storage_path(last_case, 'geometry.npz')) if last_case else None)
    report.update(binning=target_bin,source_binning=source_bin,source_size=source_size,
        fit_size=stack.shape[-1],patch_size=2*radius+1,
        source_positions_zyx=source_positions,source_cumulative_shifts_yx=source_shifts)
    if reconstruct:
        if last_case is None:raise ValueError('No accepted geometry to reconstruct')
        from greco.tomography import gpu_reconstruct_wbp
        from greco.web.refinement_worker import bin_plane_device
        print('Saving final accepted geometry and coarse preview',flush=True)
        aligned=storage_path(last_case, f'aligned_bin{target_bin}.mrcs')
        coarse=cp.empty((len(angles),source_size,source_size),cp.float32)
        with mrcfile.new_mmap(aligned,shape=stack.shape,mrc_mode=2) as m:
            m.set_image_stack();m.voxel_size=voxel
            for t,plane in enumerate(aligned_planes(stack,shifts,frozen,axis,cp)):
                m.data[t]=cp.asnumpy(plane)
                coarse[t]=bin_plane_device(plane,int(scale),cp)
        volume=-gpu_reconstruct_wbp(cp.asnumpy(coarse),angles)
        tomo=storage_path(output, 'tomogram_bin8.mrc')
        with mrcfile.new(tomo) as m:m.set_data(float16_data(volume));m.voxel_size=tuple(v*scale for v in voxel)
        report.update(tomogram=str(tomo),aligned_stack=str(aligned),contrast_inverted=True,reconstruction='Ram-Lak WBP preview at bin8',shape=list(volume.shape))
    (storage_path(output, 'report.json')).write_text(json.dumps(clean_json(report),indent=2,allow_nan=False))
    return report
