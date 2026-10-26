"""Joint landmark cameras and XYZ using polishing's rigid GPU objective."""
from greco.project_layout import join as storage_path, ensure_output
from greco.preprocessing_star import read_table
import json, time, hashlib
from pathlib import Path
import numpy as np


def run(data):
    import cupy as cp,mrcfile
    from .web.refinement_worker import load_state, rebuild_native, clean_json
    from .angular_joint import rescale_geometry
    from .landmark_rigid import initial_cameras,make_objective,fit_rigid_motion,updated_geometry
    from .landmark_rigid_gpu import Kernels,ramp_filter,angle_weights
    from .sta.gp3d import correlation_surfaces
    from .helpers import soft_spherical_mask
    from .geometry_support import prune_pairs,accepted_view_order,expand_camera_update
    from .geometry_trial_costs import trial_image_costs
    start=time.perf_counter();out=Path(data['output']);ensure_output(out);options=data['options']
    regeneration=None
    if data.get('regenerate_landmarks') and not data.get('sta_landmarks'):
        from .geometry_landmarks import seed_after_manual
        data,regeneration=seed_after_manual(data,out)
    basis=Path(data.get('landmark_basis') or storage_path(Path(data['coarse_run']), 'final_state.npz'))
    angles=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
    positions,shifts,valid,classes,_,source_bin=load_state(basis,angles,allow_partial=True)
    with np.load(basis) as checkpoint:
        active=np.asarray(checkpoint['active_indices'],int)
    if source_bin!=8:raise ValueError('Start from an accepted bin8 reconstruction.')
    sta=data.get('sta_landmarks')
    if sta:
        positions=np.asarray(sta['positions_zyx'],float);valid=np.asarray(sta['valid_views'],bool)
        if positions.ndim!=2 or positions.shape[1]!=3 or valid.shape!=(len(positions),len(angles)) or not np.isfinite(positions).all():
            raise ValueError('Invalid STA landmark coordinates/support.')
        classes=np.zeros(len(positions),int)
    # STA clouds replace load_state's support; keep rejected tilts excluded.
    valid[:,~np.isin(np.arange(len(angles)),active)]=False
    original_positions=positions.copy();target_bin=options['binning'];axis=float(data['orientation']['rotation_deg'])
    with mrcfile.mmap(data['inputs']['stack']) as m:
        source_n=m.data.shape[-1];source_pixel=float(m.voxel_size.x)
        stack=cp.asarray(m.data,cp.float32) if target_bin==8 else None
    rows=read_table(data['inputs']['frozen_shifts'])
    if len(rows)!=len(angles) or not np.allclose([float(r['angle_deg']) for r in rows],angles,atol=1e-4):raise ValueError('Frozen tilt identity mismatch.')
    order,fit_order,fit_inverse=accepted_view_order([int(r['mdoc_index']) for r in rows],active)
    if len(set(int(r['mdoc_index']) for r in rows))!=len(rows):raise ValueError('Ambiguous acquisition order.')
    if sta and sta['mdoc_indices']!=[int(r['mdoc_index']) for r in rows]:raise ValueError('STA acquisition identities differ.')
    if target_bin==8:
        pixel=source_pixel;frozen=np.array([[float(r['dy_raw']),float(r['dx_raw'])] for r in rows])/8
        if data['orientation'].get('flip_y'):stack=stack[:,::-1,:].copy();frozen[:,0]*=-1
    else:
        config=dict(data,options=dict(options,output_binning=target_bin));stack,frozen,pixel=rebuild_native(config,angles)
        positions,shifts=rescale_geometry(positions,shifts,source_n,stack.shape[-1],axis,data['orientation'].get('flip_y',False))
    n=stack.shape[-1];center=(n-1)/2;xyz=positions[:,::-1]-center
    cameras=initial_cameras(angles,axis);rz=initial_cameras(np.array([0.]),axis)[0]
    translations=-(shifts[:,::-1]@rz[:2,:2].T)-frozen[:,::-1]
    initial_cameras_array=cameras.copy();initial_translations=translations.copy();initial_xyz=xyz.copy()
    if sta and data.get('resume_geometry'):
        # New STA landmarks replace the checkpoint's old cloud. Transfer only
        # its shared camera correction, not its per-landmark XYZ corrections.
        with np.load(data['resume_geometry'],allow_pickle=False) as saved:
            if (not np.array_equal(saved['angles_deg'],angles)
                or not np.array_equal(saved['acquisition_order'],order)
                or not np.allclose(saved['original_cameras_xyz'],initial_cameras_array)
                or bool(saved['raster_flip_y'])!=bool(data['orientation'].get('flip_y',False))
                or not np.isclose(float(saved['fit_size'])*float(saved['pixel_size_A']),n*pixel)):
                raise ValueError('STA camera checkpoint does not match this alignment.')
            cameras=np.array(saved['cameras_xyz'])
            translations=initial_translations+(saved['translations_xy_A']-saved['original_translations_xy_A'])/pixel
    kernels=Kernels(cp);patch=options['patch_size'];half=(patch-1)/2;padding=2;fft_n=patch+1
    maximum=min(options['search_support_A']/pixel,(patch-1)/2-4)
    if maximum<=0:raise ValueError('Insufficient landmark patch support.')
    def centers_for(c,x,t):return np.einsum('tab,kb->kta',c[:,:2],x)+t[None]+center
    centers=centers_for(cameras,xyz,translations)
    # Support is fixed within each fit/comparison. Geometric-only exclusions
    # trigger a fresh reference bank and fit, never a cheaper masked trial.
    support=valid & np.all((centers>=half+maximum+2)&(centers<n-half-maximum-2),axis=2)
    if data.get('resume_geometry') and not sta:
        # Internal diagnostic/continuation entry; validate identity rather than
        # interpreting an arbitrary checkpoint as this tilt series.
        with np.load(data['resume_geometry'],allow_pickle=False) as checkpoint:
            saved={k:np.array(checkpoint[k]) for k in checkpoint.files}
        if (not np.array_equal(saved['angles_deg'],angles)
            or not np.array_equal(saved['source_positions_zyx'],original_positions)
            or not np.allclose(saved['original_cameras_xyz'],initial_cameras_array)
            or not np.array_equal(saved['acquisition_order'],order)
            or bool(saved['raster_flip_y'])!=bool(data['orientation'].get('flip_y',False))):
            raise ValueError('Rigid geometry checkpoint identity mismatch')
        ids=saved['landmark_ids'];support=np.asarray(saved['valid_views'],bool)
        if (ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(valid))
            or support.shape!=(len(ids),len(angles)) or np.any(support & ~valid[ids])
            or np.any(support.sum(1)<3)):
            raise ValueError('Invalid checkpoint landmark support')
        from .geometry_checkpoint import resample_checkpoint
        initial_xyz=initial_xyz[ids]
        xyz,translations=resample_checkpoint(saved,pixel,n,target_bin,initial_xyz,initial_translations)
        cameras=saved['cameras_xyz']
        centers=centers_for(cameras,xyz,translations)
        outside=support & ~np.all((centers>=half+1)&(centers<n-half-1),axis=2)
        if outside.any():
            support,keep,_=prune_pairs(support,outside,active_tilts=active)
            ids=ids[keep];xyz=xyz[keep];initial_xyz=initial_xyz[keep]
    else:
        ids=np.flatnonzero(support.sum(axis=1)>=3);support=support[ids];xyz=xyz[ids];initial_xyz=initial_xyz[ids]
    if not len(ids) or np.any(support.sum(axis=0)[active]<8):raise ValueError('Rigid geometry needs eight safe landmarks per tilt, each with at least three views.')
    mask=cp.asarray(soft_spherical_mask(patch,radius=half-2,edge=3),cp.float32)
    y,x=cp.mgrid[:patch,:patch]-half;window=cp.clip((half-cp.sqrt(x*x+y*y))/3,0,1).astype(cp.float32)
    ky=cp.fft.fftfreq(fft_n)[:,None];kx=cp.fft.rfftfreq(fft_n)[None,:];band=(kx*kx+ky*ky<=.35**2)&(kx*kx+ky*ky>0)
    weights=np.zeros(len(angles));weights[active]=angle_weights(angles[active]);history=[];stable=0;stop='maximum_passes'
    support_events=[];support_refits=0
    def spectra(images):
        v=images-(images*window).sum((-2,-1),keepdims=True)/cp.maximum(window.sum(),1)
        v=v*window
        return cp.fft.rfft2(cp.pad(v,((0,0),(0,1),(0,1))),axes=(-2,-1))
    def reference(filtered,c,target,chosen):
        volume=kernels.local_wbp(filtered[chosen],support[chosen],c,weights,target)*mask
        return volume
    while len(history)<options['max_passes']:
        iteration=len(history)+1
        centers=centers_for(cameras,xyz,translations)
        if not np.all(np.all((centers>=half+1)&(centers<n-half-1),axis=2)[support]):raise ValueError('Accepted geometry lost extraction support.')
        estimated=len(ids)*len(angles)*(patch**2*12+(2*int(np.ceil(maximum*padding))+7)**2*8)
        free,_=cp.cuda.runtime.memGetInfo()
        if estimated>free*.6:raise ValueError('Landmark reference bank exceeds GPU memory. Choose a smaller patch or coarser bin.')
        patches=kernels.patches(stack,centers,patch);filtered=ramp_filter(patches,cameras,cp)
        owners,views=np.where(support);data_fft=spectra(patches[owners,views]);prediction=cp.empty_like(data_fft)
        for target in range(len(angles)):
            selected=np.flatnonzero(views==target)
            for first in range(0,len(selected),8):
                obs=selected[first:first+8];volume=reference(filtered,cameras,target,owners[obs]);prediction[obs]=spectra(kernels.project(volume,cameras[target]))
        denominator=cp.maximum(cp.sum(abs(prediction)**2*band,axis=(-2,-1)),1e-20)
        amplitude=cp.maximum(cp.sum((data_fft*cp.conj(prediction)).real*band,axis=(-2,-1))/denominator,0.)
        prediction*=amplitude[:,None,None]
        noise=cp.maximum(cp.mean(abs(data_fft[:,~band])**2,axis=1),1e-12)
        precision=band[None]/noise[:,None,None]
        surfaces,radius=correlation_surfaces(data_fft,prediction,precision,fft_n,maximum,padding,cp)
        frames=fit_inverse[views]
        objective=make_objective(xyz,cameras[fit_order],owners,frames,surfaces,radius,padding,maximum,cp)
        parameters,diagnostic=fit_rigid_motion(objective,options['optimizer_iterations'])
        # Exactly the polishing global-fraction safeguard, evaluated again with
        # rotated 3D projections rather than the old camera's CC approximation.
        diagnostic['proposal_acceptance_scale']=diagnostic['acceptance_scale']
        trials=[];blocked=[];fractions=(0.,.125,.25,.5,1.)
        for fraction in fractions:
            c,new_xyz,t=updated_geometry(objective,parameters*fraction,cameras[fit_order],xyz,translations[fit_order])
            c,t=expand_camera_update(cameras,translations,fit_order,cp.asnumpy(c),cp.asnumpy(t));new_xyz=cp.asnumpy(new_xyz)
            new_centers=centers_for(c,new_xyz,t)
            outside=support & ~np.all((new_centers>=half+1)&(new_centers<n-half-1),axis=2)
            displacement=new_centers[owners,views]-centers[owners,views]
            search_exceeded=np.any(abs(displacement)>(radius-2)/padding)
            blocked.append(dict(outside=outside,search_exceeded=bool(search_exceeded)))
            trials.append((c,displacement,outside,bool(search_exceeded)))
        batches=[]
        for target in range(len(angles)):
            selected=np.flatnonzero(views==target)
            batches.extend((target,selected[first:first+8]) for first in range(0,len(selected),8))
        candidates=trial_image_costs(trials,batches,
            lambda target,obs:reference(filtered,cameras,target,owners[obs]),
            lambda volume,camera:spectra(kernels.project(volume,camera)),
            data_fft,amplitude,precision,kx,ky,cp)
        index=int(np.argmin(candidates));accepted=fractions[index]
        diagnostic['candidate_support']=[dict(fraction=fraction,
            out_of_field_pairs=int(b['outside'].sum()),
            out_of_field_landmarks=int(b['outside'].any(1).sum()),
            search_limit_exceeded=b['search_exceeded']) for fraction,b in zip(fractions,blocked)]
        # If every positive trial is geometrically/range invalid, a single
        # edge observation must not veto the whole cloud. Choose the smallest
        # field-blocked trial that fits the correlation range, exclude ONLY
        # its geometrically invalid pairs, then rebuild LOO evidence and refit.
        # Both zero and candidate costs will use this SAME new support bank.
        geometric_trial=next((i for i in range(1,len(fractions))
            if blocked[i]['outside'].any() and not blocked[i]['search_exceeded']),None)
        if accepted==0 and geometric_trial is not None and support_refits>=8:
            diagnostic['support_exclusion_blocked']='support_refit_limit'
        if accepted==0 and not np.isfinite(candidates[1:]).any() and geometric_trial is not None and support_refits<8:
            outside=blocked[geometric_trial]['outside']
            try:new_support,keep,removed=prune_pairs(support,outside,active_tilts=active)
            except ValueError:
                diagnostic['support_exclusion_blocked']='insufficient_remaining_support'
            else:
                event=dict(pass_number=iteration,trigger_fraction=fractions[geometric_trial],
                    removed_observations=[dict(landmark_id=int(ids[k]),tilt_number=int(t+1),
                        angle_deg=float(angles[t])) for k,t in np.argwhere(removed)],
                    retired_landmark_ids=ids[~keep].tolist(),observations_before=int(support.sum()),
                    observations_after=int(new_support.sum()),reason='projected_patch_outside_detector',
                    references_rebuilt=True,baseline_and_candidate_same_observations=True)
                support_events.append(event);support_refits+=1
                support=new_support;ids=ids[keep];xyz=xyz[keep];initial_xyz=initial_xyz[keep]
                stable=0
                print('Geometry support refit: '+json.dumps(event),flush=True)
                del objective,surfaces,patches,filtered,data_fft,prediction,parameters
                cp.get_default_memory_pool().free_all_blocks()
                continue
        c,new_xyz,t=updated_geometry(objective,parameters*accepted,cameras[fit_order],xyz,translations[fit_order])
        c,t=expand_camera_update(cameras,translations,fit_order,cp.asnumpy(c),cp.asnumpy(t));new_xyz=cp.asnumpy(new_xyz)
        camera_change=float(np.max(np.linalg.norm(c-cameras,axis=(1,2))));xyz_change=float(np.max(np.linalg.norm(new_xyz-xyz,axis=1)))
        shift_change=float(np.max(np.linalg.norm(t-translations,axis=1)))
        diagnostic.update(pass_number=iteration,exact_cost_before=candidates[0],exact_cost_after=candidates[index],acceptance_scale=accepted,candidate_fractions=list(fractions),candidate_costs=candidates,camera_change=camera_change,xyz_change_A=xyz_change*pixel,translation_change_A=shift_change*pixel,support_refits=support_refits)
        support_refits=0
        history.append(diagnostic);cameras,xyz,translations=c,new_xyz,t
        stable=stable+1 if camera_change<1e-4 and xyz_change*pixel<.05 and shift_change*pixel<.05 else 0
        if accepted==0 and not diagnostic['converged']:stop='no_improving_step';break
        if stable>=2:stop='stable';break
        (storage_path(out, 'progress.json')).write_text(json.dumps(clean_json(dict(phase='Rigid landmark geometry',pass_number=iteration,maximum_passes=options['max_passes'],landmarks=len(ids),diagnostic=diagnostic)),allow_nan=False))
        del objective,surfaces,patches,filtered,data_fft,prediction,parameters
        cp.get_default_memory_pool().free_all_blocks()
    # Preserve the full matrices; three Euler corrections cannot be reduced
    # to the single-axis/single-inclination angle_state representation.
    geometry=storage_path(out, 'rigid_geometry.npz')
    np.savez_compressed(geometry,protocol='greco-landmark-rigid-v1',cameras_xyz=cameras,translations_xy_A=translations*pixel,
        positions_xyz_A=xyz*pixel,landmark_ids=ids,valid_views=support,angles_deg=angles,mdoc_indices=np.array([int(r['mdoc_index']) for r in rows]),
        active_indices=active,acquisition_order=order,anchor_view_index=fit_order[0],pixel_size_A=pixel,fit_size=n,binning=target_bin,
        centered_stack_translations_xy_A=np.zeros_like(translations),
        original_cameras_xyz=initial_cameras_array,original_translations_xy_A=initial_translations*pixel,original_positions_xyz_A=initial_xyz*pixel,
        source_positions_zyx=original_positions,coordinate_convention='centered_world_xyz_and_raw_detector_xy_angstrom',raster_flip_y=data['orientation'].get('flip_y',False))
    aligned=storage_path(out, f'centered_detector_bin{target_bin}.mrcs')
    from .angular_joint import aligned_planes
    with mrcfile.new_mmap(aligned,shape=stack.shape,mrc_mode=2) as m:
        m.set_image_stack();m.voxel_size=pixel
        # Remove detector translations only; retain each tilt's full camera.
        for t,plane in enumerate(aligned_planes(stack,-translations[:,::-1],np.zeros_like(translations),0.,cp)):
            m.data[t]=cp.asnumpy(plane)
    from .landmark_geometry_export import export_outputs
    with np.load(geometry,allow_pickle=False) as saved:
        export=export_outputs(data,{key:np.array(saved[key]) for key in saved.files},out,cp)
    from .web.particle_worker import check_snapshot
    check_snapshot(data.get('snapshot',[]))
    report=dict(protocol='greco-landmark-rigid-v1',geometry_model='Rz_chi_Ry_theta_Rx_phi_per_tilt_plus_xy_and_static_landmark_xyz',optimizer_module='greco.sta.rigid_polishing',
        reference_mode='leave_one_tilt_out',shared_geometry=True,individual_landmark_motion=False,angular_bounds=False,
        active_count=len(active),total_tilts=len(angles),anchor_view_index=int(fit_order[0]),anchor_mdoc_index=int(rows[fit_order[0]]['mdoc_index']),binning=target_bin,options=options,
        history=history,stop_reason=stop,converged=stop=='stable',landmark_count=len(ids),supported_observations=int(support.sum()),
        support_exclusions=support_events,passes=len(history),
        geometry_file=str(geometry),centered_detector_stack=str(aligned),**export,
        runtime_s=time.perf_counter()-start,source_reconstruction=data['coarse_run'])
    relative=cameras@initial_cameras_array.transpose(0,2,1)
    report['max_rotation_change_deg']=float(np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))).max())
    report['max_translation_change_A']=float(np.linalg.norm(translations-initial_translations,axis=1).max()*pixel)
    report['max_xyz_change_A']=float(np.linalg.norm(xyz-initial_xyz,axis=1).max()*pixel)
    if data.get('landmark_basis'):report['landmark_basis']=data['landmark_basis']
    if regeneration:report['landmark_regeneration']=regeneration
    if sta:report['sta_landmarks']=dict(source_star=sta['source_star'],selection=sta['selection'],particle_ids=np.asarray(sta['particle_ids'])[ids].tolist())
    code=Path(__file__).resolve().parent
    report['code_sha256']={name:hashlib.sha256((storage_path(code, name)).read_bytes()).hexdigest() for name in ('preprocessing_star.py','particle_star.py','landmark_rigid.py','landmark_rigid_gpu.py','landmark_rigid_worker.py','geometry_trial_costs.py','geometry_landmarks.py','geometry_support.py','sta/rigid_polishing.py','sta/gp3d.py','sta/gpu_lbfgs.py')}
    if not report['converged']:report['warning']='No convergence recorded; inspect the stop reason and the last accepted geometry.'
    (storage_path(out, 'geometry_report.json')).write_text(json.dumps(clean_json(report),indent=2,allow_nan=False))
    result=dict(report,state='completed',kind='geometry_refinement',display_name='Refine geometry',gpu=True,report=str(storage_path(out, 'geometry_report.json')))
    if not data.get('defer_web_result'):
        (storage_path(out, 'web_result.json')).write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
    from .reconstruction_layout import publish
    publish(out,result)
    return result
