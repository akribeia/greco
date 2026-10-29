"""Batch neighbour-column CC and target-only XY GCC, followed by Greco export."""
from greco.project_layout import join as storage_path, ensure_output, job_work
from greco.preprocessing_star import read_table
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
from .refinement_worker import load_state,clean_json


def finish(data,geometry,out,xp,reports,start):
    """Publish a reviewable alignment first; reconstruct only on final Submit."""
    import mrcfile,hashlib
    from .particle_worker import check_snapshot
    from ..landmark_geometry_export import export_outputs
    from ..curation_2d import diagnose
    action=data['manual_action']
    path=storage_path(out,'rigid_geometry.npz');np.savez_compressed(path,**geometry)
    check_snapshot(data.get('snapshot',[]))
    export=export_outputs(data,geometry,out,xp,review_only=action=='refine')
    active=np.asarray(geometry.get('active_indices',np.arange(len(geometry['angles_deg']))),int)
    with mrcfile.mmap(export['aligned_stack']) as m:
        diagnostic=diagnose(np.array(m.data,np.float32),np.asarray(geometry['angles_deg'])[active])
    for row in diagnostic['rows']:row.update(number=int(active[row['index']])+1,source_index=int(active[row['index']]))
    for pair in diagnostic['pairs']:
        pair.update(left_number=int(active[pair['left_index']])+1,right_number=int(active[pair['right_index']])+1)
    storage_path(out,'transition_diagnostic.json').write_text(json.dumps(clean_json(diagnostic),indent=2,allow_nan=False))
    result=dict(state='completed',kind='geometry_refinement',protocol='greco-manual-xy-v1',manual_curation=True,
        manual_action=action,geometry_file=str(path),source_reconstruction=data['coarse_run'],
        geometry_model='selected_tilt_XY_only',gpu=True,binning=8,target_reports=reports,
        converged=None if action=='reconstruct' else all(r['converged'] for r in reports),passes=sum(len(r['history']) for r in reports),
        disabled_tilts=data.get('options',{}).get('disabled_tilts',[]),
        frozen_angles=True,frozen_xyz=True,frozen_other_tilts=True,runtime_s=time.perf_counter()-start,**export)
    if data.get('landmark_basis'):result['landmark_basis']=data['landmark_basis']
    check_snapshot(data.get('snapshot',[]))
    if action!='refine':
        from .final_denoising import denoise_outputs
        denoise_outputs(data,result,geometry=True)
    code=Path(__file__).resolve().parents[1]
    result['code_sha256']={name:hashlib.sha256(storage_path(code,name).read_bytes()).hexdigest() for name in
        ('preprocessing_star.py','particle_star.py','manual_tilt.py','web/manual_refinement_worker.py','landmark_rigid_gpu.py','registration.py','sta/gp3d.py','sta/gpu_lbfgs.py','landmark_geometry_export.py')}
    storage_path(out,'web_result.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
    from ..reconstruction_layout import publish
    publish(out,result)
    return result


def run(data):
    import cupy as cp,mrcfile
    from ..manual_tilt import validate_edits,choose_reference,display_prior_to_detector,shared_xy_objective,neighbor_column_candidates,image_smoothness_guides,smooth_xy_penalty,validate_gaussian_search,gaussian_search_penalty,gaussian_search_start
    from ..landmark_rigid import initial_cameras
    from ..landmark_rigid_gpu import Kernels,ramp_filter,angle_weights
    from ..geometry_checkpoint import resample_checkpoint
    from ..selection import column_score_volume_gpu
    from ..registration import register_local_models,cosine_disk_mask
    from ..helpers import soft_spherical_mask
    from ..sta.gp3d import correlation_surfaces,keys_sample
    from ..sta.gpu_lbfgs import minimize
    from .particle_worker import check_snapshot
    start=time.perf_counter();check_snapshot(data.get('snapshot',[]))
    out=Path(data['output']);ensure_output(out)
    angles=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
    action=data['manual_action'];options=data.get('options',{})
    edits=[] if action=='reconstruct' or not data.get('manual_edits') else validate_edits(data['manual_edits'],len(angles))
    targets={e['index'] for e in edits}
    axis=float(data['orientation']['rotation_deg']);flip=bool(data['orientation'].get('flip_y'))
    checkpoint=Path(data.get('landmark_basis') or storage_path(Path(data['coarse_run']), 'final_state.npz'))
    positions,shifts,valid,_,_,source_bin=load_state(checkpoint,angles,allow_partial=True)
    with np.load(checkpoint,allow_pickle=False) as saved:active=np.asarray(saved['active_indices'],int)
    if data.get('resume_geometry'):
        with np.load(data['resume_geometry'],allow_pickle=False) as prior:
            previous_active=np.asarray(prior['active_indices'] if 'active_indices' in prior else np.arange(len(angles)),int)
        if (previous_active.ndim!=1 or np.any(np.diff(previous_active)<=0) or np.any(previous_active<0)
                or np.any(previous_active>=len(angles))):
            raise ValueError('Invalid resumed accepted tilt selection.')
        active=previous_active
    disabled=set(options.get('disabled_tilts',[]))
    if not disabled.issubset(set(active.tolist())) or disabled & targets:raise ValueError('Invalid disabled tilt selection.')
    active=np.array([t for t in active if t not in disabled],int)
    accepted_before=active.copy()
    recovering=targets-set(active.tolist())
    if recovering and not options.get('include_discarded',False):raise ValueError('Enable discarded tilts before attempting recovery.')
    if len(set(active.tolist())-targets)<3:raise ValueError('At least three fixed accepted reference tilts are required.')
    excluded=set(range(len(angles)))-set(active.tolist())
    valid[:,list(excluded)]=False
    if source_bin!=8:raise ValueError('Manual refinement requires bin8 geometry.')
    with mrcfile.mmap(data['inputs']['stack']) as m:raw=np.array(m.data,np.float32);pixel=float(m.voxel_size.x)
    if raw.ndim!=3 or raw.shape[0]!=len(angles) or raw.shape[1]!=raw.shape[2] or not np.isfinite(raw).all() or not np.isfinite(pixel) or pixel<=0:
        raise ValueError('Manual refinement requires a finite square bin8 stack with matching tilt identities and pixel size.')
    if flip:raw=raw[:,::-1,:].copy()
    n=raw.shape[-1];center=(n-1)/2
    rows=read_table(data['inputs']['frozen_shifts'])
    if len(rows)!=len(angles) or not np.allclose([float(r['angle_deg']) for r in rows],angles,atol=1e-4):raise ValueError('Frozen angle identity mismatch.')
    if len(set(int(r['mdoc_index']) for r in rows))!=len(rows):raise ValueError('Ambiguous acquisition identity.')
    frozen=np.array([[float(r['dx_raw']),float(r['dy_raw'])] for r in rows])/8
    if flip:frozen[:,1]*=-1
    from ..alignment_geometry import camera_components
    with np.load(storage_path(Path(data['coarse_run']),'final_state.npz'),allow_pickle=False) as source_geometry:
        cameras,translations=camera_components(source_geometry,angles,axis,shifts,frozen[:,::-1])
        from ..alignment_geometry import resolve
        fitted_angles,display_axes=resolve(source_geometry,angles,axis)
        if data.get('resume_geometry'):display_axes=np.full(len(angles),axis)
    rz=initial_cameras([0.],axis)[0]
    base_xyz=positions[:,::-1]-center;ids=np.arange(len(positions))
    geometry=dict(protocol='greco-landmark-rigid-v1',cameras_xyz=cameras,translations_xy_A=translations*pixel,
        positions_xyz_A=base_xyz*pixel,landmark_ids=ids,valid_views=valid,angles_deg=angles,
        original_cameras_xyz=cameras.copy(),original_translations_xy_A=translations*pixel,
        original_positions_xyz_A=base_xyz*pixel,source_positions_zyx=positions,
        acquisition_order=np.argsort([int(r['mdoc_index']) for r in rows]),mdoc_indices=np.array([int(r['mdoc_index']) for r in rows]),
        pixel_size_A=pixel,fit_size=n,binning=8,raster_flip_y=flip,
        coordinate_convention='centered_world_xyz_and_raw_detector_xy_angstrom',centered_stack_translations_xy_A=np.zeros_like(translations))
    geometry['active_indices']=np.array(sorted(set(active.tolist())|targets),int)
    geometry['anchor_view_index']=next(int(t) for t in geometry['acquisition_order'] if t in active)
    if data.get('resume_geometry'):
        with np.load(data['resume_geometry'],allow_pickle=False) as saved:prior={k:np.array(saved[k]) for k in saved.files}
        if (not np.array_equal(prior['angles_deg'],angles) or not np.array_equal(prior['source_positions_zyx'],positions)
            or not np.allclose(prior['original_cameras_xyz'],cameras) or bool(prior['raster_flip_y'])!=flip
            or not np.array_equal(prior['acquisition_order'],geometry['acquisition_order'])):raise ValueError('Manual source checkpoint identity mismatch.')
        ids=np.asarray(prior['landmark_ids'],int)
        xyz,translations=resample_checkpoint(prior,pixel,n,8,base_xyz[ids],translations)
        geometry.update(positions_xyz_A=xyz*pixel,landmark_ids=ids,valid_views=prior['valid_views'],
            original_positions_xyz_A=base_xyz[ids]*pixel)
        cameras=prior['cameras_xyz'].copy();geometry['cameras_xyz']=cameras
    if action=='reconstruct' or not edits:
        geometry['valid_views'][:,list(excluded)]=False
        geometry['translations_xy_A']=translations*pixel
        del raw
        return finish(data,geometry,out,cp,[],start)
    original_t=translations.copy();original_cameras=cameras.copy();original_xyz=geometry['positions_xyz_A'].copy()
    options=data.get('options',{});patch=int(options.get('patch_size',33));half=(patch-1)//2
    max_passes=int(options.get('max_passes',60));entry_radius=float(options.get('entry_radius_px',16))
    normal_radius=float(options.get('search_radius_px',4));strength=float(options.get('smoothness_strength',0.))
    gaussian=validate_gaussian_search(options.get('gaussian_search_prior',False),options.get('gaussian_sigma_px',8.))
    sigma=gaussian['gaussian_sigma_px']
    # Apply every prior before measuring image guidance. None enters the reference model.
    for edit in edits:translations[edit['index']]+=display_prior_to_detector(edit['shift_yx'],float(display_axes[edit['index']]),pixel)/pixel
    stack=cp.asarray(raw);del raw
    kernels=Kernels(cp);weights=angle_weights(fitted_angles);weights[list(targets|excluded)]=0
    fixed=np.array([t for t in active if t not in targets])
    filtered=ramp_filter(stack,cameras,cp)
    from ..tomogram_thickness import stage_bounds
    source_config=storage_path(job_work(Path(data['coarse_run'])), 'config.json')
    source_settings=json.loads(source_config.read_text()) if source_config.is_file() else {}
    core,context=stage_bounds(n,source_settings.get('tomogram_thickness_percent',100.),halo=32)
    free,_=cp.cuda.runtime.memGetInfo()
    estimate=4*(n**3+11*(context[1]-context[0])*n*n)
    if estimate>free*.8:raise ValueError('The bin8 q-min context exceeds GPU memory. Use a GPU with more free memory.')
    volume=cp.zeros((n,n,n),cp.float32)
    for z in range(context[0],context[1],8):
        end=min(z+8,context[1]);volume[z:end]=kernels.global_wbp(filtered,cameras,translations,weights,z,end-z)
    context_score=column_score_volume_gpu(volume[context[0]:context[1]],'bright_structure')
    score=cp.zeros_like(volume);score[core[0]:core[1]]=context_score[core[0]-context[0]:core[1]-context[0]]
    del volume,filtered,context_score
    cp.get_default_memory_pool().free_all_blocks()
    guides={}
    if strength:
        from ..gcc import materialize_raw_stack_gpu
        from ..landmark_geometry_export import display_geometry
        from ..curation_2d import diagnose
        _,display_shifts,_=display_geometry(cameras,translations,axis,n,n,flip)
        diagnostic_indices=geometry['active_indices']
        aligned=materialize_raw_stack_gpu(stack,diagnostic_indices,display_shifts[diagnostic_indices],np.zeros_like(display_shifts[diagnostic_indices]),axis)
        diagnostic=diagnose(cp.asnumpy(aligned),angles[diagnostic_indices]);del aligned
        guides=image_smoothness_guides(diagnostic,diagnostic_indices,targets,translations,axis,pixel)
    caches={};reports=[];mask=cp.asarray(soft_spherical_mask(patch,radius=half-2,edge=3),cp.float32)
    window=cp.asarray(cosine_disk_mask(patch),cp.float32);fft_n=2*(patch+1);padding=2
    kx=cp.fft.rfftfreq(fft_n)[None,:];ky=cp.fft.fftfreq(fft_n)[:,None]
    band=(kx*kx+ky*ky>0)&(kx*kx+ky*ky<.35**2)
    def spectra(images):
        images=(images-(images*window).sum((-2,-1),keepdims=True)/window.sum())*window
        return cp.fft.rfft2(cp.pad(images,((0,0),(0,fft_n-patch),(0,fft_n-patch))))
    for edit in edits:
        # The selected overlay reference only guided the absolute XY placement.
        # Numerical references are fixed neighbours, independent of that UI choice.
        target=edit['index'];reference=choose_reference(angles,target,targets|excluded)
        references=fixed[np.argsort(abs(angles[fixed]-angles[reference]))[:int(options.get('reference_views',11))]]
        if reference not in caches:caches[reference]=neighbor_column_candidates(score,cameras[reference],cell=int(options.get('column_cell',24)),z_bounds=core)
        candidates=caches[reference]
        landmark_file=storage_path(out,f'tilt_{target+1:03d}_landmarks.npz')
        history=[];stable=0;reason='maximum_passes';good=[];entry=np.zeros(2)
        guide=guides.get(target)
        for iteration in range(max_passes):
            # Re-extract and rebuild the local bank at the current geometry.
            # Visibility is reconsidered as the target moves; all selected
            # targets remain absent from every reference cube on every pass.
            centres=np.einsum('tab,kb->kta',cameras[:,:2],candidates)+translations[None]+center
            supported=np.all((centres>=half+entry_radius+2)&(centres<n-half-entry_radius-2),axis=2)
            supported[:,[t for t in range(len(angles)) if t not in references]]=False
            safe=supported[:,reference] & (supported.sum(1)>=3) & np.all((centres[:,target]>=half+entry_radius+2)&(centres[:,target]<n-half-entry_radius-2),axis=1)
            xyz=candidates[safe];supported=supported[safe];centres=centres[safe]
            if len(xyz)<8:raise ValueError(f'Tilt {target+1}: fewer than eight safe neighbour-column landmarks.')
            if supported[:,list(targets|excluded)].any():raise AssertionError('A fitted/rejected tilt entered the reference bank.')
            patches=kernels.patches(stack,centres,patch);filtered=ramp_filter(patches,cameras,cp)
            model=cp.empty((len(xyz),patch,patch),cp.float32)
            for first in range(0,len(xyz),8):
                volumes=kernels.local_wbp(filtered[first:first+8],supported[first:first+8],cameras,weights,target)*mask
                model[first:first+8]=kernels.project(volumes,cameras[target])
            del patches,filtered,volumes
            model_fft=spectra(model)
            def target_patches(width):
                c=np.einsum('ab,kb->ka',cameras[target,:2],xyz)+translations[target]+center
                return kernels.patches(stack[target:target+1],c[:,None],width)[:,0]
            if iteration==0:
                observed=target_patches(patch+2*int(entry_radius))
                votes=register_local_models(cp.asnumpy(model),cp.asnumpy(observed),method='gradient_zncc',search_radius=int(entry_radius))
                good=[v for v in votes if v.valid and not v.boundary]
                entry=np.median([v.correction_yx[::-1] for v in good],axis=0) if len(good)>=8 else np.zeros(2)
            observed_fft=spectra(target_patches(patch))
            # Amplitude/noise are frozen for acceptance comparisons in this pass.
            cross=(observed_fft*cp.conj(model_fft)*band).real.sum((1,2))
            if iteration==0:
                phase=cp.exp(2j*cp.pi*(entry[0]*kx+entry[1]*ky))
                cross=cp.maximum(cross,(observed_fft*phase*cp.conj(model_fft)*band).real.sum((1,2)))
            amplitude=cp.maximum(cross/cp.maximum((abs(model_fft)**2*band).sum((1,2)),1e-12),0)
            if int(cp.count_nonzero(amplitude>0))<8:raise ValueError(f'Tilt {target+1}: fewer than eight informative local references.')
            prediction=model_fft*amplitude[:,None,None]
            noise=cp.maximum((abs(observed_fft[:,~band])**2).mean(1),1e-12)
            radius_px=entry_radius if iteration==0 else normal_radius
            surfaces,radius=correlation_surfaces(observed_fft,prediction,band[None]/noise[:,None,None],fft_n,radius_px,padding,cp)
            likelihood=shared_xy_objective(surfaces,radius,padding,radius_px,cp)
            curvature=0.;desired=np.zeros(2)
            if guide:
                probes=cp.asarray([[0,0],[1,0],[-1,0],[0,1],[0,-1]])*padding+radius
                probe_values=[]
                for point in probes:
                    value,_=keys_sample(surfaces,cp.broadcast_to(point,(len(xyz),2)),cp)
                    probe_values.append(float(value.mean()))
                curvature=max(0.,(sum(probe_values[1:])-4*probe_values[0])/2)
                desired=guide['detector_xy']-translations[target]
            use_gaussian=gaussian['gaussian_search_prior'] and iteration==0
            def objective(parameter):
                value,gradient=likelihood(parameter)
                if guide and curvature>0:
                    h=cp.tanh(parameter)
                    penalty,derivative=smooth_xy_penalty(radius_px*h,desired,strength*guide['quality'],curvature,cp)
                    value+=penalty;gradient+=derivative*radius_px*(1-h*h)
                if use_gaussian:
                    h=cp.tanh(parameter)
                    penalty,derivative=gaussian_search_penalty(radius_px*h,sigma,cp)
                    value+=penalty;gradient+=derivative*radius_px*(1-h*h)
                return value,gradient
            start_xy=cp.asnumpy(gaussian_search_start(surfaces,radius,padding,radius_px,sigma,cp)) if use_gaussian else entry
            initial=np.clip(start_xy/radius_px,-.999,.999) if use_gaussian else np.clip(entry/radius_px,-.9,.9) if iteration==0 else np.zeros(2)
            optimum=minimize(objective,cp.asarray(np.arctanh(initial)),cp,max_iterations=300,gtol=1e-6)
            proposal=cp.asnumpy(radius_px*cp.tanh(optimum.x))
            proposals=[np.zeros(2)]+[proposal*f for f in (1.,.5,.25)]
            costs=[]
            for delta in proposals:
                values,_=keys_sample(surfaces,cp.broadcast_to(cp.asarray(delta)*padding+radius,(len(xyz),2)),cp)
                penalty=smooth_xy_penalty(cp.asarray(delta),desired,strength*guide['quality'],curvature,cp)[0] if guide else 0.
                if use_gaussian:penalty+=gaussian_search_penalty(cp.asarray(delta),sigma,cp)[0]
                costs.append(float(values.mean()+penalty))
            chosen=int(np.argmin(costs));delta=proposals[chosen]
            new_c=np.einsum('ab,kb->ka',cameras[target,:2],xyz)+translations[target]+delta+center
            if not np.all((new_c>=half+1)&(new_c<n-half-1)):raise ValueError('Manual XY proposal exceeds detector support.')
            translations[target]+=delta
            improvement=costs[0]-costs[chosen]
            history.append(dict(pass_number=iteration+1,delta_detector_xy_px=delta.tolist(),objective_before=costs[0],objective_after=costs[chosen],
                optimizer_iterations=optimum.nit,optimizer_success=optimum.success,search_radius_px=radius_px,improvement=improvement,
                column_count=len(xyz),reference_bank_rebuilt=True,smoothness_curvature=curvature,gaussian_search_prior=use_gaussian))
            stable=stable+1 if np.linalg.norm(delta)<.025 else 0
            (storage_path(out, 'progress.json')).write_text(json.dumps(clean_json(dict(phase='Manual XY GCC',tilt_number=target+1,completed_tilts=len(reports),total_tilts=len(edits),pass_number=iteration+1,landmarks=len(xyz)))))
            if stable>=2:reason='stable_xy';break
        np.savez_compressed(landmark_file,positions_xyz_A=xyz*pixel,valid_reference_views=supported,
            target_index=target,reference_index=reference,angles_deg=angles)
        reports.append(dict(tilt_number=target+1,index=target,reference_number=reference+1,reference_views=(references+1).tolist(),
            placement_reference_number=edit['reference']+1 if edit['reference'] is not None else None,
            manual_prior_yx_px=edit['shift_yx'],smoothness_strength=strength,
            gaussian_search_prior=dict(**gaussian,first_wide_pass_only=True,center='manual_placement',objective='mean_column_sse_plus_gaussian_neg2log_prior_v1'),
            image_smoothness_guide=None if guide is None else dict(correction_yx_px=guide['correction_yx_px'],edges=guide['edges'],quality=guide['quality']),column_count=len(xyz),entry_votes=len(good),entry_cc_xy_px=entry.tolist(),
            total_detector_xy_A=((translations[target]-original_t[target])*pixel).tolist(),
            total_display_shift_yx_px=(-(translations[target]-original_t[target])@rz[:2,:2])[::-1].tolist(),
            landmark_file=str(landmark_file),history=history,stop_reason=reason,converged=reason=='stable_xy'))
    untouched=[t for t in range(len(angles)) if t not in targets]
    if not np.array_equal(translations[untouched],original_t[untouched]) or not np.array_equal(cameras,original_cameras) or not np.array_equal(geometry['positions_xyz_A'],original_xyz):raise AssertionError('Manual refinement changed frozen geometry.')
    recovered={report['index'] for report in reports if report['index'] in recovering and report['converged']}
    geometry['active_indices']=np.array(sorted(set(accepted_before.tolist())|recovered),int)
    for report in reports:report.update(was_discarded=report['index'] in recovering,recovered=report['index'] in recovered)
    geometry['valid_views'][:,[t for t in range(len(angles)) if t not in set(geometry['active_indices'])]]=False
    geometry['translations_xy_A']=translations*pixel
    return finish(data,geometry,out,cp,reports,start)



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    data=json.loads(args.config.read_text())
    try:run(data)
    except Exception as error:
        traceback.print_exc();out=Path(data['output']);out.mkdir(parents=True,exist_ok=True)
        (storage_path(out, 'web_result.json')).write_text(json.dumps(dict(state='failed',error=str(error),manual_curation=True)))
        raise SystemExit(1)


if __name__=='__main__':main()
