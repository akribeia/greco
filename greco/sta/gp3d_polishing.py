"""Independent-half, per-tomogram GP3D polishing of immutable raw patches."""
from greco.project_layout import join as storage_path

import time
from pathlib import Path
import numpy as np
from .gp3d import correlation_surfaces,spatial_basis,MotionObjective,fit_motion
from .corrections import write_corrections
from .polishing_precision import precision


def run_fit(config,dataset,backend,refs,variance,begin,cached):
    from ..web.particle_worker import save_json,check_snapshot
    xp=backend.xp;host=backend.host;options=config['options'];out=Path(config['output'])
    n=dataset.n;pixel=dataset.pixel;count=len(dataset.particles);m=int(dataset.offsets[-1])
    halves=np.array([p['grecoRandomSubset']-1 for p in dataset.particles])
    if np.any((halves<0)|(halves>1)):raise ValueError('Invalid GP3D half labels')
    owner=np.repeat(np.arange(count),np.diff(dataset.offsets));padding=2
    maximum=options['maximum_shift_A']/pixel;radius=int(np.ceil(maximum*padding))+3
    # Retain compact evidence on device when it fits; raw images are streamed
    # once to build it, not reread at every optimizer iteration.
    surface_bytes=m*(2*radius+1)**2*8
    device_surfaces=backend.gpu and surface_bytes<.4*xp.cuda.runtime.memGetInfo()[0]
    surface_xp=xp if device_surfaces else np
    surfaces=surface_xp.zeros((m,2*radius+1,2*radius+1),dtype=surface_xp.float64)
    ky=xp.fft.fftfreq(n)*n;kx=xp.arange(n//2+1)
    radii=xp.sqrt(ky[:,None]**2+kx[None,:]**2)
    shell=xp.minimum(xp.rint(radii).astype(xp.int32),n//2)
    active=(radii>0)&(radii<=backend.radius)
    last_progress=[0.]
    def progress(phase,done=0,**extra):
        now=time.perf_counter()
        if 'optimizer_iteration' in extra and now-last_progress[0]<.5:return
        last_progress[0]=now
        save_json(storage_path(out, 'progress.json'),dict(phase=phase,processed=done,total=count,**extra))
    progress('GP3D: accumulating own-half correlation surfaces')
    for start,stop,f,c in dataset.loaded_batches(256):
        a,b=dataset.offsets[[start,stop]]
        for half in (0,1):
            use=halves[owner[a:b]]==half
            if not use.any():continue
            prediction=c[use]*backend.project(refs[half],dataset.rotations[a:b][use],dataset.inverse_rotations[a:b][use])
            weight=precision(variance,half,shell,active,dataset.groups[a:b][use],xp)
            values,r=correlation_surfaces(f[use],prediction,weight,n,maximum,padding,xp)
            if r!=radius:raise ValueError('Correlation grid mismatch')
            surfaces[np.arange(a,b)[use]]=values if device_surfaces else host(values)
        progress('GP3D: accumulating own-half correlation surfaces',stop)
    if options.get('refine_rigid_geometry',False):
        from .rigid_polishing_worker import run_joint
        return run_joint(config,dataset,backend,refs,variance,begin,cached,surfaces,radius,padding,active,shell,progress)
    shifts=np.zeros((m,2));xyz_motion=np.zeros((m,3));fits=[];completed=0;prior_cost={}
    scales=np.array([0.,.125,.25,.5,1.])
    particle_tomos=np.array([p['grecoTomogramId'] for p in dataset.particles])
    xyz=np.array([[p['grecoCoordinate'+d+'Angst'] for d in 'XYZ'] for p in dataset.particles])/pixel
    old_motion=np.array([[dataset.observation_corrections.get(key,{}).get('grecoGpMotion'+d+'Angst',0.)
                          for d in 'XYZ'] for key in dataset.observation_keys])/pixel
    for tid in dict.fromkeys(particle_tomos):
        group_ids=[i for i,key in enumerate(dataset.group_keys) if key[0]==tid]
        rows=[dataset.views[dataset.group_keys[i]] for i in group_ids]
        # Pre-exposure is the strongest available acquisition ordering; mdoc
        # ZValue is the fallback, never angularly sorted view ID.
        have_dose=all('grecoPreExposureElectronPerAngst2' in row for row in rows)
        order=np.argsort([row['grecoPreExposureElectronPerAngst2'] if have_dose else row['grecoMdocIndex'] for row in rows])
        ids=np.array(group_ids)[order];ordered=[rows[i] for i in order]
        times=np.array([row['grecoPreExposureElectronPerAngst2'] if have_dose else row['grecoMdocIndex'] for row in ordered],float)
        if np.any(np.diff(times)<=0):raise ValueError('Ambiguous GP3D acquisition order')
        dose=float(np.median([row.get('grecoExposureElectronPerAngst2',1.) for row in rows]))
        if options.get('gp3d_dose_scaling',True) and (not have_dose or dose<=0):
            raise ValueError('Dose-scaled GP3D needs explicit positive exposure metadata')
        gaps=np.diff(times)/dose if have_dose else np.diff(times)
        frame_lookup={group:i for i,group in enumerate(ids)}
        scale=dose if options.get('gp3d_dose_scaling',True) else 1.
        for half in (0,1):
            particles=np.flatnonzero((particle_tomos==tid)&(halves==half))
            if not len(particles):continue
            use=np.isin(owner,particles);obs=np.flatnonzero(use)
            local_lookup=np.full(count,-1,int);local_lookup[particles]=np.arange(len(particles))
            owners=local_lookup[owner[obs]]
            frames=np.array([frame_lookup[g] for g in dataset.groups[obs]])
            basis,eigenvalues=spatial_basis(xyz[particles],scale*options['velocity_sigma_A']/pixel,
                scale*options.get('divergence_sigma_A',5000.)/pixel,options.get('gp3d_max_modes',-1),xp)
            # Repeated polishing penalizes the cumulative GP velocity, not
            # merely the latest increment. Interpolate only missing state;
            # missing observations still have no data term in the optimizer.
            old=np.zeros((len(particles),len(ids),3))
            for k in range(len(particles)):
                selected=owners==k
                sort=np.argsort(frames[selected]);ordered_frames=frames[selected][sort]
                for d in range(3):old[k,:,d]=np.interp(np.arange(len(ids)),ordered_frames,old_motion[obs[selected],d][sort])
            velocity=np.diff(old,axis=1)/np.sqrt(gaps)[None,:,None]
            # Eigenbasis columns are orthogonal: exact pseudoinverse without
            # two dense SVDs per fit (also valid for deliberately truncated GP).
            inverse_basis=basis.T/eigenvalues[:,None]
            previous=(inverse_basis@xp.asarray(velocity).reshape(len(particles),-1)).reshape(len(eigenvalues),len(gaps),3).transpose(1,0,2)
            objective=MotionObjective(basis,dataset.view_rotations[obs],owners,frames,
                xp.asarray(surfaces[obs]),radius,padding,gaps,maximum,xp,previous)
            name=dataset.tomos[tid]['grecoTomogramName']
            label=f'GP3D: {name} half {half+1}'
            start=time.perf_counter()
            delta,track,diagnostic=fit_motion(objective,options.get('gp3d_iterations',1000),
                lambda iteration:progress(label,completed,optimizer_iteration=iteration))
            shifts[obs]=delta;xyz_motion[obs]=track[owners,frames]
            proposed_velocity=xp.asarray(np.diff(track,axis=1)/np.sqrt(gaps)[None,:,None])
            proposed_coeff=(inverse_basis@proposed_velocity.reshape(len(particles),-1)).reshape(len(eigenvalues),len(gaps),3).transpose(1,0,2)
            prior_cost[(tid,half)]=np.array([float(xp.sum((previous+scale*proposed_coeff)**2)-xp.sum(previous**2)) for scale in scales])
            diagnostic.update(tomogram=name,half=half+1,seconds=time.perf_counter()-start,
                              frame_dose=dose,minimum_eigenvalue=float(eigenvalues[-1]))
            fits.append(diagnostic);completed+=len(particles)
            save_json(storage_path(out, 'gp3d_fits.json'),fits);progress(label,completed)
            del objective,basis,previous,inverse_basis
    # Validate compact-surface proposals against the actual Fourier SSE.
    # Use one common scale per tomogram/half to preserve GP coupling.
    data_cost={}
    ky=xp.fft.fftfreq(n)[:,None]*2*xp.pi;kx=xp.arange(n//2+1)[None,:]*2*xp.pi/n
    progress('GP3D: checking exact Fourier likelihood')
    for start,stop,f,c in dataset.loaded_batches(256):
        a,b=dataset.offsets[[start,stop]]
        for half in (0,1):
            for tid in dict.fromkeys(particle_tomos[owner[a:b]]):
                use=(halves[owner[a:b]]==half)&(particle_tomos[owner[a:b]]==tid)
                if not use.any():continue
                prediction=c[use]*backend.project(refs[half],dataset.rotations[a:b][use],dataset.inverse_rotations[a:b][use])
                delta=xp.asarray(shifts[a:b][use]);phase=delta[:,0,None,None]*kx+delta[:,1,None,None]*ky
                # Constant terms cancel exactly; avoid catastrophic cancellation
                # when comparing tiny shifts to huge noisy observation power.
                cross=f[use]*xp.conj(prediction);weight=precision(variance,half,shell,active,dataset.groups[a:b][use],xp)
                costs=data_cost.setdefault((tid,half),np.zeros(len(scales)))
                for j,scale in enumerate(scales):
                    costs[j]+=float(-2*xp.sum((cross*(xp.exp(1j*scale*phase)-1)).real*weight))
    # Re-evaluate the exact joint posterior, including the cumulative velocity
    # prior at every tested scale (not a per-view or data-only veto).
    acceptance={key:float(scales[np.argmin(values+prior_cost[key])]) for key,values in data_cost.items()}
    factors=np.array([acceptance[(particle_tomos[k],halves[k])] for k in owner])
    shifts*=factors[:,None];xyz_motion*=factors[:,None]
    corrections=[]
    for i,key in enumerate(dataset.observation_keys):
        row=dict(dataset.observation_corrections.get(key,{}),grecoParticleId=key[0],grecoViewId=key[1])
        for d,value in zip('XY',shifts[i]*pixel):row['grecoResidualShift'+d+'Angst']=row.get('grecoResidualShift'+d+'Angst',0.)+float(value)
        for d,value in zip('XYZ',(old_motion[i]+xyz_motion[i])*pixel):row['grecoGpMotion'+d+'Angst']=float(value)
        corrections.append(row)
    np.savez(storage_path(out, 'gp3d_motion.npz'),observation_keys=dataset.observation_keys,
             incremental_xyz_A=xyz_motion*pixel,incremental_xy_A=shifts*pixel)
    norms=np.linalg.norm(shifts*pixel,axis=1)
    diagnostics=dict(motion_model='spatial_GP3D_exponential',tilt_geometry_refined=False,
        halves_independent=True,median_shift_A=float(np.median(norms)),maximum_shift_A=float(norms.max()),
        rms_shift_A=float(np.sqrt(np.mean(norms**2))),p95_shift_A=float(np.quantile(norms,.95)),
        shift_outlier_fraction=float(np.mean(norms>=.1*pixel)),fits=fits,
        regularized_objective_change=float(sum(np.min(values+prior_cost[key]) for key,values in data_cost.items())),
        exact_likelihood_acceptance=[dict(tomogram=dataset.tomos[tid]['grecoTomogramName'],half=int(h)+1,
                                        scale=value) for (tid,h),value in acceptance.items()])
    star=storage_path(out, 'particles.star');write_corrections(star,dataset.path,dataset.all_particles,corrections,'polishing_gp3d')
    diagnostics['gpu_optimizer']=bool(backend.gpu)
    diagnostics['gpu_correlation_cache']=bool(device_surfaces)
    diagnostics['support_policy']='per-observation interpolation-margin barrier; no global trajectory rescaling'
    result=dict(state='completed',protocol='sta-gp3d-polishing-v2',particles=str(star),
        particle_count=len(dataset.all_particles),used_particles=count,box_size=n,pixel_size_A=pixel,
        poses_refined=False,observation_corrections=True,gpu=backend.gpu,
        gpu_particle_cache=cached,options=options,diagnostics=diagnostics,
        reference_job=config.get('reference_job'),extraction_manifest=str(dataset.path),
        elapsed_seconds=time.perf_counter()-begin)
    check_snapshot(config['snapshot']);save_json(storage_path(out, 'web_result.json'),result);progress('Completed',count)
    return result
