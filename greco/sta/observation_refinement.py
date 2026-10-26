"""GPU projected-motion and mean-defocus refinement of immutable STA data."""
from greco.project_layout import join as storage_path
from pathlib import Path
import time
import numpy as np
from .dataset import ParticleDataset,ctf_phase
from .fixed_pose import FixedPoseBackend
from .residual_noise import estimate_half_residual_noise
from .corrections import write_corrections


def run(config):
    import mrcfile
    from .auto_refine import alignment_reference
    from ..web.particle_worker import check_snapshot,save_json
    from .polishing import fit_view_shifts,smooth_track
    from .ctf_refinement import defocus_evidence,fit_defocus_offsets
    check_snapshot(config['snapshot']);begin=time.perf_counter();out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    options=config['options'];kind=config['action'];n=int(config['output_box']);pixel=float(config['pixel_size_A'])
    backend=FixedPoseBackend(n,config.get('gpu',True),radius=min(n//2-1,n*pixel/options['resolution_A']))
    xp=backend.xp;dataset=ParticleDataset(config['extraction_manifest'],backend)
    if not np.isclose(dataset.pixel,pixel,rtol=1e-6):raise ValueError('Observation refinement sampling differs from extraction.')
    if len(config['half_maps'])!=2:raise ValueError('Two independent reference half-maps are required.')
    model=options.get('observation_model','ctf_premultiplied') if kind=='polishing' else 'raw'
    dataset.configure_observation_model(model)
    count=len(dataset.particles);halves=np.array([p['grecoRandomSubset']-1 for p in dataset.particles])
    independent=kind=='polishing' and options.get('motion_model')=='independent_ridge'
    previous_residual=np.array([[dataset.observation_corrections.get(key,{}).get('grecoResidualShift'+d+'Angst',0.) for d in 'XY'] for key in dataset.observation_keys])/pixel
    def progress(phase,processed=0,**extra):save_json(storage_path(out, 'progress.json'),dict(phase=phase,processed=processed,total=count,**extra))
    progress('Caching extracted views when GPU memory permits')
    cached=dataset.cache(lambda done,total:progress('Caching extracted views',done))
    refs=[];reference_mask=None
    mask_path=options.get('mask_path')
    if mask_path:
        from .reference_mask import load_mask
        reference_mask=load_mask(mask_path,n,pixel,resample_physical=True)
    for path in config['half_maps']:
        with mrcfile.open(path) as m:
            if m.data.shape!=(n,)*3 or not np.isclose(float(m.voxel_size.x),pixel,rtol=1e-3):raise ValueError('Polishing/CTF reference sampling mismatch.')
            v=np.array(m.data,np.float64)
        if not np.isfinite(v).all():raise ValueError('Non-finite reference half-map.')
        if reference_mask is not None:v*=reference_mask
        refs.append(backend.prepare_map(alignment_reference(xp.asarray(v),backend.radius,xp)))
    progress('Estimating independent half noise')
    if config.get('frozen_noise_path'):
        from .frozen_noise import load_frozen_noise
        variance,noise_info=load_frozen_noise(config['frozen_noise_path'],n,xp)
    else:variance,noise_info=estimate_half_residual_noise(dataset,backend,per_view=options.get('noise_model')=='view_residual')
    noise_info['reference_mask']=mask_path
    np.save(storage_path(out, 'noise_variance.npy'),backend.host(variance));save_json(storage_path(out, 'noise_model.json'),noise_info)
    if kind=='polishing' and options.get('motion_model')=='gp3d':
        from .gp3d_polishing import run_fit
        return run_fit(config,dataset,backend,refs,variance,begin,cached)
    ns=n//2+1;y=xp.fft.fftfreq(n)*n;x=xp.arange(ns)
    radii=xp.sqrt(y[:,None]**2+x[None,:]**2);shell=xp.minimum(xp.rint(radii).astype(xp.int32),n//2)
    active=(radii>0)&(radii<=backend.radius)
    if kind=='polishing':
        measured=np.zeros((int(dataset.offsets[-1]),2));curvature=np.zeros((len(measured),2,2))
    else:
        evidence=xp.zeros((len(dataset.group_keys),5,n,ns),xp.float64)
    for start,stop,f,c in dataset.loaded_batches(512):
        progress('Fitting per-view shifts' if kind=='polishing' else 'Accumulating defocus evidence',start)
        a,b=dataset.offsets[[start,stop]];owner=np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1]))
        for h in (0,1):
            use=halves[owner]==h
            if not use.any():continue
            projection=backend.project(refs[h],dataset.rotations[a:b][use],dataset.inverse_rotations[a:b][use])
            from .polishing_precision import precision as select_precision
            precision=select_precision(variance,h,shell,active,dataset.groups[a:b][use],xp)
            if kind=='polishing':
                shifts,H=fit_view_shifts(f[use],c[use]*projection,precision,n,options['maximum_shift_A']/pixel,options['optimizer_steps'],xp,
                    prior_sigma=options['velocity_sigma_A']/pixel if independent else None,
                    previous=previous_residual[a:b][use] if independent else None)
                ids=np.arange(a,b)[use];measured[ids]=backend.host(shifts);curvature[ids]=backend.host(H)
            else:
                gamma=ctf_phase(dataset.parameters[a:b][use],n,pixel,xp)
                values=defocus_evidence(f[use],projection*dataset.transfer_envelope(a,b)[use],gamma,precision,xp)
                xp.add.at(evidence,xp.asarray(dataset.groups[a:b][use]),values)
    corrections={key:dict(dataset.observation_corrections.get(key,{}),
                          grecoParticleId=key[0],grecoViewId=key[1])
                 for key in dataset.observation_keys}
    if kind=='polishing':
        previous_shifts=np.array([[corrections[key].get('grecoResidualShift'+d+'Angst',0.)+dataset.applied_observation_corrections.get(key,{}).get('grecoResidualShift'+d+'Angst',0.) for d in 'XY'] for key in dataset.observation_keys])/pixel
        increments=np.zeros_like(measured)
        for k,p in enumerate(dataset.particles):
            a,b=dataset.offsets[[k,k+1]];keys=dataset.observation_keys[a:b]
            acquisition=np.array([dataset.views[(p['grecoTomogramId'],view)]['grecoMdocIndex'] for _,view in keys])
            previous=previous_shifts[a:b]
            motion=measured[a:b] if independent else smooth_track(measured[a:b],curvature[a:b],acquisition,options['velocity_sigma_A']/pixel,previous)
            increments[a:b]=np.clip(motion,-options['maximum_shift_A']/pixel,options['maximum_shift_A']/pixel)
        # Validate the smoothed proposal against the actual Fourier objective,
        # not just its quadratic approximation. A whole trajectory shares one
        # acceptance scale; independent per-view rejection would break the prior.
        scales=np.array([0.,.125,.25,.5,1.]);costs=np.zeros((count,len(scales)))
        for k,p in enumerate(dataset.particles):
            a,b=dataset.offsets[[k,k+1]];keys=dataset.observation_keys[a:b]
            previous=previous_shifts[a:b]
            acq=np.array([dataset.views[(p['grecoTomogramId'],v)]['grecoMdocIndex'] for _,v in keys]);order=np.argsort(acq)
            gap=np.maximum(1,np.diff(acq[order]));sigma=options['velocity_sigma_A']/pixel
            for j,scale in enumerate(scales):
                difference=np.diff((previous+scale*increments[a:b])[order],axis=0)
                costs[k,j]=np.sum((previous_residual[a:b]+scale*increments[a:b])**2)/sigma**2 if independent else np.sum(difference**2/(sigma**2*gap[:,None]))
        ky=xp.fft.fftfreq(n)[:,None]*2*xp.pi;kx=xp.arange(ns)[None,:]*2*xp.pi/n
        progress('Checking regularized trajectory likelihood')
        for start,stop,f,c in dataset.loaded_batches(512):
            a,b=dataset.offsets[[start,stop]];owner=np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1]))
            for h in (0,1):
                use=halves[owner]==h
                if not use.any():continue
                prediction=c[use]*backend.project(refs[h],dataset.rotations[a:b][use],dataset.inverse_rotations[a:b][use])
                shift=xp.asarray(increments[a:b][use]);phase=shift[:,0,None,None]*kx+shift[:,1,None,None]*ky
                precision=select_precision(variance,h,shell,active,dataset.groups[a:b][use],xp)
                for j,scale in enumerate(scales):
                    value=xp.sum(abs(f[use]*xp.exp(1j*scale*phase)-prediction)**2*precision,axis=(-2,-1))
                    costs[:,j]+=np.bincount(owner[use],weights=backend.host(value),minlength=count)
        accepted=scales[np.argmin(costs,axis=1)]
        increments*=np.repeat(accepted,np.diff(dataset.offsets))[:,None]
        for i,key in enumerate(dataset.observation_keys):
            corrections[key].update(grecoParticleId=key[0],grecoViewId=key[1])
            for d,shift in zip('XY',increments[i]*pixel):corrections[key]['grecoResidualShift'+d+'Angst']=corrections[key].get('grecoResidualShift'+d+'Angst',0.)+float(shift)
        norms=np.linalg.norm(increments*pixel,axis=1)
        diagnostics=dict(median_shift_A=float(np.median(norms)),maximum_shift_A=float(np.max(norms)),
            rms_shift_A=float(np.sqrt(np.mean(norms**2))),p95_shift_A=float(np.quantile(norms,.95)),
            shift_outlier_fraction=float(np.mean(norms>=.1*pixel)),
            motion_model='independent_2d_zero_prior' if independent else 'projected_2d_Brownian_acquisition_order',tilt_geometry_refined=False,
            accepted_tracks=int(np.count_nonzero(accepted)),regularized_objective_change=float(np.sum(np.min(costs,axis=1)-costs[:,0])))
    else:
        k1=np.zeros(len(dataset.group_keys))
        for group in np.unique(dataset.groups):
            vals=dataset.parameters[dataset.groups==group,3]
            if np.ptp(vals)>1e-12:raise ValueError('Inconsistent wavelength within a tilt group.')
            k1[group]=vals[0]
        if options.get('ctf_model')=='joint':
            from .joint_ctf_refinement import fit_views
            views,diagnostics=fit_views(dataset,backend,evidence,k1,options,out,progress)
            star=storage_path(out, 'particles.star')
            write_corrections(star,dataset.path,dataset.all_particles,list(corrections.values()),'joint_ctf',views)
            # Assert that all fitted optical state survives STAR serialization.
            from .corrections import corrected_views
            saved=corrected_views(star)
            for view in views:
                actual=saved[(view['grecoTomogramId'],view['grecoViewId'])]
                for field in ('grecoCentralDefocusUAngst','grecoCentralDefocusVAngst','grecoDefocusAngleDeg','grecoCtfScaleFactor'):
                    if actual[field]!=view[field]:raise ValueError('Joint CTF STAR state differs from fit')
            result=dict(state='completed',protocol='sta-joint-ctf-v1',particles=str(star),particle_count=len(dataset.all_particles),used_particles=count,
                box_size=n,pixel_size_A=pixel,poses_refined=False,observation_corrections=True,
                gpu=backend.gpu,gpu_particle_cache=cached,options=options,diagnostics=diagnostics,
                reference_job=config.get('reference_job'),extraction_manifest=str(dataset.path),elapsed_seconds=time.perf_counter()-begin)
            check_snapshot(config['snapshot']);save_json(storage_path(out, 'web_result.json'),result);progress('Completed',count)
            return result
        shifts,curves,H=fit_defocus_offsets(evidence,k1,n,pixel,options['maximum_defocus_A'],options['defocus_step_A'],xp)
        shifts=backend.host(shifts);curves=backend.host(curves);np.save(storage_path(out, 'defocus_curves.npy'),curves)
        best=np.argmin(curves,axis=1);observations=np.bincount(dataset.groups,minlength=len(dataset.group_keys))
        boundary=np.flatnonzero((observations>0)&((best==0)|(best==curves.shape[1]-1)))
        for key,group in zip(dataset.observation_keys,dataset.groups):
            corrections[key].update(grecoParticleId=key[0],grecoViewId=key[1],grecoDefocusOffsetAngst=corrections[key].get('grecoDefocusOffsetAngst',0.)+float(shifts[group]))
        diagnostics=dict(defocus_offsets_A=shifts.tolist(),groups=dataset.group_keys,
            astigmatism_refined=False,beam_gradient_preserved=True,
            boundary_groups=boundary.tolist(),boundary_fits_retained_original=True,
            observations_per_group=observations.tolist())
    star=storage_path(out, 'particles.star');write_corrections(star,dataset.path,dataset.all_particles,list(corrections.values()),kind)
    result=dict(state='completed',protocol='sta-observation-refinement-v1',particles=str(star),particle_count=len(dataset.all_particles),used_particles=count,
        box_size=n,pixel_size_A=pixel,poses_refined=False,observation_corrections=True,
        gpu=backend.gpu,gpu_particle_cache=cached,options=options,diagnostics=diagnostics,reference_job=config.get('reference_job'),
        extraction_manifest=str(dataset.path),elapsed_seconds=time.perf_counter()-begin)
    check_snapshot(config['snapshot']);save_json(storage_path(out, 'web_result.json'),result)
    progress('Completed',count);return result
