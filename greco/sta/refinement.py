"""Local six-DOF STA refinement, independent halves, GPU particle hot path.

Uses Greco local-particle-refinement's weighted Fourier L2/MAP formulation,
not the exhaustive reference_refine orientation-marginalisation driver.
"""
from greco.project_layout import join as storage_path, ensure_output
from pathlib import Path
import time
import warnings
import hashlib
import numpy as np
from .fixed_pose import FixedPoseBackend
from .dataset import ParticleDataset
from .local_pose import LocalPoseObjective,optimize_local_search,pose_geometry
from .auto_refine import AutoRefine,alignment_reference
from .pose_posterior import local_posterior,compose_pose
from .trust_region import optimize_trust_region
from .signal_prior import fsc_signal_prior
from .fsc import masked_fsc
from .observation_model import backprojection_statistics
from .reference_storage import half_reference
from .storage import write_mrc


def fsc_curve(half1,half2,pixel,xp=np):
    n=half1.shape[0];a=xp.fft.rfftn(half1);b=xp.fft.rfftn(half2)
    freq=xp.fft.fftfreq(n)*n;x=xp.arange(n//2+1)
    shell=xp.rint(xp.sqrt(freq[:,None,None]**2+freq[None,:,None]**2+x[None,None,:]**2)).astype(int)
    mult=xp.ones(n//2+1)*2;mult[0]=1;mult[-1]=1
    valid=shell<=n//2;ids=shell[valid];weight=xp.broadcast_to(mult,a.shape)[valid]
    cross=xp.bincount(ids,weights=(a*xp.conj(b)).real[valid]*weight,minlength=n//2+1)
    aa=xp.bincount(ids,weights=abs(a)[valid]**2*weight,minlength=n//2+1)
    bb=xp.bincount(ids,weights=abs(b)[valid]**2*weight,minlength=n//2+1)
    values=cross/xp.maximum(xp.sqrt(aa*bb),1e-30)
    if xp is not np:values=xp.asnumpy(values)
    return [dict(shell=i,resolution_A=n*pixel/i if i else None,fsc=float(v)) for i,v in enumerate(values)]


def diagnostic_resolution(curve,threshold=.143):
    for previous,current in zip(curve[1:-1],curve[2:]):
        if previous['fsc']>=threshold and current['fsc']<threshold:
            fraction=(threshold-previous['fsc'])/(current['fsc']-previous['fsc'])
            shell=previous['shell']+fraction
            return previous['shell']*previous['resolution_A']/shell
    return None


def loaded_batches(dataset,max_observations):
    if hasattr(dataset,'loaded_batches'):
        yield from dataset.loaded_batches(max_observations)
    else:
        for start,stop in dataset.batches(max_observations):yield start,stop,*dataset.load(start,stop)


def refine(config):
    if config['options'].get('symmetry','C1')!='C1':
        raise ValueError('STA supports asymmetric particles only; point-group symmetry is not supported.')
    if config['options'].get('refine_per_tilt_shifts',False) or config['options'].get('refine_ctf',False):
        from .alternating_refinement import refine_alternating
        return refine_alternating(config,_refine_once)
    return _refine_once(config)


def _refine_once(config):
    import mrcfile
    from scipy.spatial.transform import Rotation
    from ..particle_star import write_loop
    from ..web.particle_worker import check_snapshot,save_json
    from greco._numerics.core.projector import initial_lowpass_filter_reference
    from greco._numerics.core.tait_bryan import angles_to_matrix3
    begin=time.perf_counter();output=Path(config['output']);ensure_output(output)
    last_progress=[0.]
    check_snapshot(config['snapshot']);options=config['options']
    sources=[Path(__file__),Path(__file__).with_name('trust_region.py'),Path(__file__).with_name('local_pose.py'),
             Path(__file__).with_name('auto_refine.py'),Path(__file__).with_name('signal_prior.py'),Path(__file__).with_name('fsc.py'),Path(__file__).with_name('observation_model.py'),
             Path(__file__).with_name('reference_storage.py'),Path(__file__).with_name('storage.py'),
             Path(__file__).with_name('dataset.py'),Path(__file__).with_name('transfer_envelope.py'),Path(__file__).with_name('fixed_pose.py'),Path(__file__).with_name('gpu_maps.py'),
             Path(__file__).with_name('residual_noise.py'),Path(__file__).with_name('memory_refinement.py'),storage_path(Path(__file__).parent.parent, 'particle_ctf.py')]
    save_json(storage_path(output, 'protocol.json'),dict(protocol='sta-local-trust-v1' if options.get('alignment_mode')=='trust_region' else 'sta-local-search-v1',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        extraction_manifest=str(config['extraction_manifest']),options=options))
    n=int(config['output_box']);pixel=float(config['pixel_size_A']);iterations=int(options['iterations'])
    backend=FixedPoseBackend(n,config.get('gpu',True),radius=min(n//2-1,n*pixel/float(options['resolution_A'])))
    automatic=bool(options.get('auto_refine',False))
    use_masked_fsc=automatic and bool(options.get('mask_corrected_fsc',False))
    local_search=options.get('alignment_mode','trust_region')=='local_search'
    trust_region=options.get('alignment_mode')=='trust_region'
    if not (local_search or trust_region):raise ValueError('Unsupported STA alignment mode.')
    incremental=True
    fsc_prior=options.get('signal_prior','fsc')=='fsc' and automatic
    batch_observations=int(options.get('batch_observations',2048 if trust_region else 128))
    if batch_observations<1:raise ValueError('Observation batch size must be positive.')
    marginalize=bool(options.get('local_marginalization',False)) and local_search
    update_noise=bool(options.get('update_noise',False))
    reference_precision=options.get('reference_precision','float16')
    if reference_precision not in ('float16','float64'):raise ValueError('Unknown reference precision.')
    if not backend.gpu or not trust_region or update_noise:reference_precision='float64'
    from .memory_refinement import memory_policy,AccumulatorSpool,release_unused
    policy=memory_policy(backend.shape,backend.xp.cuda.runtime.memGetInfo()[0]) if backend.gpu else dict(staged=False)
    if config.get('force_staged_accumulators',False):policy.update(staged=True,forced_for_validation=True)
    staged=policy['staged']
    backend.staged_accumulators=staged;backend.memory_output=output
    if staged:batch_observations=min(batch_observations,512)
    xp=backend.xp;dataset=ParticleDataset(config['extraction_manifest'],backend);count=len(dataset.particles)
    requested_batch_observations=batch_observations
    batch_observations=dataset.observation_batch_limit(batch_observations)
    save_json(storage_path(output, 'memory_policy.json'),dict(policy,box_size=n,
        requested_batch_observations=requested_batch_observations,batch_observations=batch_observations,
        observation_workspace_bytes_per_pixel=128,observation_workspace_max_bytes=1024**3))
    if 'observation_model' in options:
        dataset.configure_observation_model(options['observation_model'])
    premultiplied=bool(getattr(dataset,'ctf_premultiplied',False))
    if not np.isclose(pixel,dataset.pixel):raise ValueError('Refinement/extraction sampling differs.')
    halves=np.array([int(p['grecoRandomSubset'])-1 for p in dataset.particles])
    if any((halves==h).sum()<2 for h in (0,1)) or np.any((halves<0)|(halves>1)):
        raise ValueError('Each independent half requires at least two supported particles.')
    parameters=np.zeros((count,6));angle_range=np.deg2rad(options['angular_range_deg']);shift_range=options['shift_range_A']/pixel
    if shift_range>n/8:raise ValueError('Local shifts must remain within one eighth of the extracted box.')
    def progress(phase,iteration=0,processed=0):
        now=time.monotonic()
        if processed and processed<count and now-last_progress[0]<.5:return
        last_progress[0]=now
        save_json(storage_path(output, 'progress.json'),dict(phase=phase,iteration=iteration,iterations=iterations,processed=processed,total=count))
    progress('Caching particle Fourier data');cached=False if staged else dataset.cache(lambda done,total:progress('Caching particle Fourier data',processed=done))
    progress('Estimating initial noise spectra')
    y=np.where(np.arange(n)<=n//2,np.arange(n),np.arange(n)-n)
    radii=np.sqrt(y[:,None]**2+np.arange(n//2+1)[None,:]**2);shell=xp.asarray(np.minimum(np.rint(radii).astype(int),n//2))
    active=xp.asarray((radii<=backend.radius)&(radii>0));ns=n//2+1;ng=len(dataset.group_keys)
    reconstruction_active=xp.asarray((radii<=n//2-1)&(radii>0)) if automatic else active
    # Estimate frozen per-half, per-view shell variances. Never share noise or maps.
    power=xp.zeros((2,ng,ns),xp.float64);samples=xp.zeros((2,ng),xp.float64)
    shell_counts=xp.bincount(shell[reconstruction_active],minlength=ns)
    initial_batches=() if options.get('noise_model')=='half_residual' else loaded_batches(dataset,batch_observations)
    for start,stop,f,c in initial_batches:
        a,b=dataset.offsets[start:stop+1][[0,-1]]
        owner=np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1]));groups=dataset.groups[a:b]
        for h in (0,1):
            use=halves[owner]==h
            if not use.any():continue
            g=xp.asarray(groups[use]);values=.5*abs(f[use])**2
            bins=g[:,None]*ns+shell[reconstruction_active][None,:]
            power[h]+=xp.bincount(bins.ravel(),weights=values[:,reconstruction_active].ravel(),minlength=ng*ns).reshape(ng,ns)
            samples[h]+=xp.bincount(g,minlength=ng)
    sigma=power/xp.maximum(samples[:,:,None]*shell_counts,1)
    floors=xp.asarray([max(float(xp.mean(sigma[h,:,1:])),1e-20)*1e-6 for h in (0,1)])
    sigma=xp.maximum(sigma,floors[:,None,None])
    noise_diagnostics={}
    if options.get('noise_model')=='half_residual':
        if update_noise:raise ValueError('Frozen half-residual noise cannot update noise.')
        from .residual_noise import estimate_half_residual_noise
        progress('Estimating independent half residual noise')
        if config.get('frozen_noise_path'):
            from .frozen_noise import load_frozen_noise
            variance,noise_diagnostics=load_frozen_noise(config['frozen_noise_path'],n,xp)
        else:
            variance,noise_diagnostics=estimate_half_residual_noise(dataset,backend,batch_observations=batch_observations)
        if staged:release_unused(backend)
        sigma=xp.broadcast_to(variance[:,None,:],(2,ng,ns)).copy()
        np.save(storage_path(output, 'noise_variance.npy'),backend.host(variance))
        save_json(storage_path(output, 'noise_model.json'),noise_diagnostics)
    if options.get('noise_model')=='unit':
        # Diagnostic control: preserve the unweighted normal equations.
        # Not a production replacement for an estimated noise spectrum.
        if update_noise:raise ValueError('Unit-noise control cannot update noise.')
        sigma=xp.ones_like(sigma)
    elif options.get('noise_model')=='provided':
        if update_noise:raise ValueError('Provided-noise control cannot update noise.')
        supplied=np.asarray(dataset.noise_variance,float)
        if supplied.shape not in ((2,ns),(2,ng,ns)) or not np.isfinite(supplied).all() or np.any(supplied<=0):
            raise ValueError('Expected independent positive half noise spectra on the extraction grid.')
        sigma=xp.broadcast_to(xp.asarray(supplied)[:,None,:] if supplied.ndim==2 else xp.asarray(supplied),(2,ng,ns)).copy()
    z,yy,x=np.ogrid[:n,:n,:n];r=np.sqrt((z-n/2)**2+(yy-n/2)**2+(x-n/2)**2)
    radius=float(options.get('particle_diameter_A') or .9*n*pixel)/(2*pixel);edge=min(5.,radius/4)
    mask=np.where(r<radius-edge,1.,np.where(r<radius,.5+.5*np.cos(np.pi*(r-radius+edge)/edge),0.))
    if options.get('mask_path'):
        from .reference_mask import load_mask
        mask=load_mask(options['mask_path'],n,pixel,resample_physical=True)
    mask=xp.asarray(mask)
    inv_base=backend.host(dataset.inverse_rotations)
    posterior_poses=None;posterior_weights=None;noise_references=None
    validation_maps=[]
    prior_diagnostics={}
    fsc_diagnostics={}
    def corrected_batch(start,stop,poses=None):
        a,b=dataset.offsets[start:stop+1][[0,-1]];owner=np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1]))
        inverse,_,shift=pose_geometry(parameters[start:stop] if poses is None else poses,inv_base[a:b],dataset.view_rotations[a:b],owner-start)
        detector=np.einsum('bij,bj->bi',dataset.view_rotations[a:b],shift)
        phase=xp.exp(2j*np.pi/n*(xp.asarray(detector[:,0,None,None])*xp.arange(n//2+1)[None,None,:]+xp.asarray(detector[:,1,None,None])*xp.asarray(y)[None,:,None]))
        return a,b,owner,inverse,phase
    def maximize(previous=None,power_spectra=None):
        nonlocal validation_maps,sigma,prior_diagnostics,fsc_diagnostics
        alignment_radius=backend.radius
        if automatic:backend.radius=n//2-1
        acc=[backend.zeros(),backend.zeros()]
        residual_power=xp.zeros_like(sigma);residual_samples=xp.zeros_like(samples)
        noise_active=active & xp.asarray(radii<=max(1,alignment_radius-1)) if automatic else active
        for start,stop,f,c in loaded_batches(dataset,batch_observations):
            nodes=1 if posterior_weights is None else posterior_weights.shape[1]
            for node in range(nodes):
                poses=None if posterior_poses is None else posterior_poses[start:stop,node]
                a,b,owner,inverse,phase=corrected_batch(start,stop,poses)
                mass=np.ones(len(owner)) if posterior_weights is None else posterior_weights[owner,node]
                for h in (0,1):
                    use=(halves[owner]==h)&(mass>0)
                    if not use.any():continue
                    g=xp.asarray(dataset.groups[a:b][use]);q=xp.asarray(mass[use])
                    precision=reconstruction_active[None]/sigma[h,g][:,shell]
                    data,weight=backprojection_statistics(f[use]*phase[use],c[use],precision*q[:,None,None],premultiplied)
                    backend.accumulate(acc[h],data,weight,np.transpose(inverse[use],(0,2,1)),xp.asarray(inverse[use]))
                    if update_noise and noise_references is not None:
                        prediction=backend.project(noise_references[h],np.transpose(inverse[use],(0,2,1)),xp.asarray(inverse[use]))
                        residual=.5*abs(f[use]*phase[use]-c[use]*prediction)**2
                        bins=g[:,None]*ns+shell[noise_active][None,:]
                        residual_power[h]+=xp.bincount(bins.ravel(),weights=(residual[:,noise_active]*q[:,None]).ravel(),minlength=ng*ns).reshape(ng,ns)
                        residual_samples[h]+=xp.bincount(g,weights=q,minlength=ng)
        if update_noise and noise_references is not None:
            counts=xp.bincount(shell[noise_active],minlength=ns)
            denominator=residual_samples[:,:,None]*counts
            estimate=residual_power/xp.maximum(denominator,1)
            # Damped residual update; unseen shells retain the previous noise.
            sigma=xp.where(denominator>0,.5*sigma+.5*xp.maximum(estimate,floors[:,None,None]),sigma)
            # Only precision changed. Cached Fourier observations and CTF stay
            # valid; each objective refreshes its precision before the next pass.
        spool=None
        if staged:
            progress('Staging full-precision half accumulators')
            spool=AccumulatorSpool(output,backend)
            for h in (0,1):spool.store(h,acc[h])
            acc.clear();release_unused(backend)
        def reconstruct_half(h,tau=None):
            if spool is None:return backend.reconstruct(acc[h],tau,float(options['regularization']),device=backend.gpu)
            with spool.acquire(h) as accumulator:
                return backend.reconstruct(accumulator,tau,float(options['regularization']),device=backend.gpu)
        def prior_and_map(h,curve):
            if spool is None:
                tau,map_noise=fsc_signal_prior(backend,acc[h][2],curve)
                return reconstruct_half(h,tau),tau,map_noise
            with spool.acquire(h) as accumulator:
                tau,map_noise=fsc_signal_prior(backend,accumulator[2],curve)
                volume=backend.reconstruct(accumulator,tau,float(options['regularization']),device=backend.gpu)
                return volume,tau,map_noise
        maps=[]
        validation_maps=[]
        if fsc_prior:
            validation_maps=[reconstruct_half(h) for h in (0,1)]
            if use_masked_fsc:
                fsc_diagnostics=masked_fsc(*validation_maps,mask,pixel,xp)
                curve=fsc_diagnostics['corrected']
            else:curve=fsc_curve(*validation_maps,pixel,xp)
            prior_diagnostics={}
            for h in (0,1):
                volume,tau,map_noise=prior_and_map(h,curve)
                maps.append(volume)
                prior_diagnostics[str(h+1)]=dict(tau2=tau.tolist(),map_noise_sigma2=map_noise.tolist())
            backend.radius=alignment_radius
            if spool is not None:spool.close()
            return maps
        for h in (0,1):
            tau=None
            if previous is not None:
                if power_spectra is None:
                    backend.prepare_map(previous[h]);tau=np.maximum(backend.projector.power_spectrum,1e-30)
                else:tau=power_spectra[h]
            if automatic:
                # Evaluate unseen frequencies without a prior derived from a
                # low-pass reference (which otherwise suppresses them to zero).
                validation=reconstruct_half(h)
                validation_maps.append(validation)
                if tau is not None:
                    backend.prepare_map(validation*mask)
                    measured=np.maximum(backend.projector.power_spectrum.copy(),1e-30)
                    tau=np.array(tau,copy=True)
                    unseen=np.arange(len(tau))>alignment_radius
                    tau[unseen]=measured[unseen]
            volume=validation if automatic and tau is None else reconstruct_half(h,tau)
            maps.append(volume if automatic else volume*mask)
        backend.radius=alignment_radius
        if spool is not None:spool.close()
        return maps
    progress('Initializing independent half references')
    if config.get('continuation_half_maps'):
        maps=[]
        for path in config['continuation_half_maps']:
            with mrcfile.open(path) as m:
                if m.data.shape!=(n,)*3 or not np.isclose(float(m.voxel_size.x),pixel,rtol=1e-3):
                    raise ValueError('Continuation reference sampling mismatch.')
                v=np.array(m.data,copy=True)
            if not np.isfinite(v).all():raise ValueError('Non-finite continuation half-map.')
            maps.append(xp.asarray(v))
        if len(maps)!=2:raise ValueError('Two independent continuation half-maps required.')
    else:maps=maximize()
    if not config.get('postconvergence_continuation',False):
        maps=[xp.asarray(initial_lowpass_filter_reference(backend.host(v),n,pixel,min(max(60.,options['resolution_A']),n*pixel/4))) for v in maps]
    # Compact Fourier likelihood buffers fit much more easily than full planes.
    # Keep each optimizer batch unchanged so its stopping behaviour is preserved.
    compact_count=int(backend.host(active).sum())
    # 8-byte data + 4-byte CTF + 8-byte precision, small tile reductions
    # and geometry overhead. The old full-frequency gradient buffer is gone.
    likelihood_cache_fraction=.4 if trust_region else .25
    cache_objectives=backend.gpu and not staged and int(dataset.offsets[-1])*compact_count*24<xp.cuda.runtime.memGetInfo()[0]*likelihood_cache_fraction
    objective_cache={}
    controller=AutoRefine(backend.radius,n//2-1,2*radius,pixel) if automatic else None
    if automatic and config.get('postconvergence_continuation',False):
        # Continue at demonstrated bandwidth, without replaying coarse search.
        # Movement/FSC checks can still expand the range when necessary.
        controller.search_factor=.5
    history=[]
    for iteration in range(1,iterations+1):
        check_snapshot(config['snapshot']);tic=time.perf_counter();before=parameters.copy()
        refs=[];power_spectra=[]
        for volume in maps:
            reference=volume*mask if automatic else volume
            if automatic:
                reference=alignment_reference(reference,backend.radius,xp)
            prepared=backend.prepare_map(reference)
            cropped=backend.crop_map(prepared,backend.radius) if trust_region else prepared
            refs.append(half_reference(cropped) if reference_precision=='float16' else cropped)
            del cropped,prepared
            power_spectra.append(np.maximum(backend.projector.power_spectrum.copy(),1e-30))
        diagnostics=[];optimizer_status=np.full(count,-99,np.int32)
        if marginalize:
            posterior_poses=np.empty((count,13,6));posterior_weights=np.empty((count,13))
        if update_noise:
            # Reuse resident references; omit the cosine taper from noise fitting.
            noise_references=refs
        alignment_batches=(((start,stop,None) for start,stop in dataset.batches(batch_observations)) if cache_objectives
                           else ((start,stop,(f,c)) for start,stop,f,c in loaded_batches(dataset,batch_observations)))
        for start,stop,batch_data in alignment_batches:
            progress('Local XYZ and orientation refinement',iteration,start)
            a,b=dataset.offsets[start:stop+1][[0,-1]]
            owner=np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1]))
            for h in (0,1):
                ids=np.flatnonzero(halves[start:stop]==h)+start
                if not len(ids):continue
                use=halves[owner]==h;local_owner=np.searchsorted(ids,owner[use])
                objective=objective_cache.get((start,h))
                if objective is None:
                    if batch_data is None:batch_data=dataset.load(start,stop)
                    f,c=batch_data
                    precision=active[None]/sigma[h,xp.asarray(dataset.groups[a:b][use])][:,shell]
                    objective=LocalPoseObjective(backend,refs[h],f[use],c[use],precision,inv_base[a:b][use],dataset.view_rotations[a:b][use],local_owner,angle_range/2,shift_range/2)
                    if cache_objectives:objective_cache[start,h]=objective
                else:
                    objective.set_reference(refs[h]);objective.evaluations=0
                    objective.update_group_noise(sigma[h],dataset.groups[a:b][use],shell)
                objective.surface_radius_pixels=radius
                if automatic:
                    objective.step_limits=np.array([angle_range]*3+[shift_range]*3)*controller.search_factor
                current=parameters[ids]
                objective.base=pose_geometry(current,inv_base[a:b][use],dataset.view_rotations[a:b][use],local_owner)[0]
                objective.shift_origin=current[local_owner,3:]
                if trust_region:
                    if not automatic:objective.step_limits=np.array([angle_range]*3+[shift_range]*3)
                    step,diag=optimize_trust_region(objective,int(options['optimizer_iterations']))
                else:
                    step,diag=optimize_local_search(objective,int(options['optimizer_iterations']),bool(options.get('local_stability_rescue',True)))
                optimizer_status[ids]=diag.get('optimizer_statuses',[-99]*len(ids))
                # Compose rotations, do not add Euler angles. XYZ increments
                # are in the fixed specimen frame, as in applyChange.
                old_rotation=Rotation.from_euler('ZYX',current[:,:3][:,::-1])
                increment=Rotation.from_euler('ZYX',step[:,:3][:,::-1])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fitted=np.column_stack(((old_rotation*increment).as_euler('ZYX')[:,::-1],current[:,3:]+step[:,3:]))
                # Finite extracted images cannot support arbitrary drift.
                unsupported=np.any(abs(fitted[:,3:])>n/8,axis=1)
                fitted[unsupported]=current[unsupported]
                if unsupported.any():
                    step[unsupported]=0
                    diag['final_cost']=float(objective.per_particle(step)[:,0].sum())
                diag['support_rejected_particles']=int(unsupported.sum())
                if marginalize:
                    local,mass=local_posterior(objective,step,radius,getattr(objective,'step_limits',None))
                    for node in range(13):posterior_poses[ids,node]=compose_pose(current,local[:,node])
                    outside=np.any(abs(posterior_poses[ids,:,3:])>n/8,axis=2)
                    mass[outside]=0
                    # Rejected optimizer steps fall back to the supported input pose.
                    empty=mass.sum(axis=1)==0
                    mass[empty,0]=1;posterior_poses[ids[empty],0]=current[empty]
                    mass/=mass.sum(axis=1,keepdims=True);posterior_weights[ids]=mass
                parameters[ids]=fitted;diagnostics.append(diag)
        align_seconds=time.perf_counter()-tic;tic=time.perf_counter();progress('Independent half-map MAP reconstruction',iteration,count)
        maps=maximize(maps,power_spectra);map_seconds=time.perf_counter()-tic
        if not all(bool(xp.isfinite(v).all()) for v in maps):raise ValueError('Non-finite refinement map.')
        # Publish immutable files first, then one atomic manifest. Readers never
        # observe a partially written map or a mixture of two iterations.
        checkpoint={}
        for name,volume in [('half1',maps[0]),('half2',maps[1])]:
            target=storage_path(output, f'iter_{iteration:03d}_{name}.mrc')
            temporary=target.with_suffix('.mrc.tmp')
            write_mrc(temporary,volume,pixel,xp=xp,overwrite=True)
            temporary.replace(target);checkpoint[name]=str(target)
        change=parameters-before
        relative=Rotation.from_euler('ZYX',parameters[:,:3][:,::-1]).inv()*Rotation.from_euler('ZYX',before[:,:3][:,::-1])
        angle_rms=float(np.rad2deg(np.sqrt(np.mean(relative.magnitude()**2))))
        shift_rms=float(np.sqrt(np.mean(np.sum(change[:,3:]**2,axis=1)))*pixel)
        shift_lengths=np.linalg.norm(change[:,3:],axis=1)*pixel
        angle_lengths=np.rad2deg(relative.magnitude())
        motion_by_status={}
        for code,label in ((-1,'failed'),(0,'converged'),(1,'iteration_limit'),(2,'stalled'),(3,'plateau'),(4,'numerical_plateau')):
            selected=optimizer_status==code
            if selected.any():motion_by_status[label]=dict(count=int(selected.sum()),
                shift_rms_A=float(np.sqrt(np.mean(shift_lengths[selected]**2))),
                shift_median_A=float(np.median(shift_lengths[selected])),shift_p95_A=float(np.quantile(shift_lengths[selected],.95)),
                angle_rms_deg=float(np.sqrt(np.mean(angle_lengths[selected]**2))))
        if incremental:
            target=storage_path(output, f'iter_{iteration:03d}_pose_diagnostics.npz');temporary=target.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,particle_id=np.array([p['grecoParticleId'] for p in dataset.particles]),
                    half=halves+1,optimizer_status=optimizer_status,parameters_before=before,parameters_after=parameters,
                    shift_change_A=shift_lengths,angle_change_deg=angle_lengths)
            temporary.replace(target)
        entry=dict(iteration=iteration,alignment_seconds=align_seconds,reconstruction_seconds=map_seconds,
            alignment_resolution_A=n*pixel/backend.radius,angle_rms_change_deg=angle_rms,shift_rms_change_A=shift_rms,
            median_shift_change_A=float(np.median(np.linalg.norm(change[:,3:],axis=1))*pixel),
            median_angle_change_deg=float(np.rad2deg(np.median(relative.magnitude()))),
            initial_cost=sum(d['initial_cost'] for d in diagnostics),final_cost=sum(d['final_cost'] for d in diagnostics),
            evaluations=sum(d['evaluations'] for d in diagnostics),rejected_particles=sum(d['rejected_particles'] for d in diagnostics),
            local_optimizer_converged_particles=sum(d.get('optimizer_converged_particles',0) for d in diagnostics),
            local_optimizer_plateau_particles=sum(d.get('optimizer_plateau_particles',0) for d in diagnostics),
            local_optimizer_numerical_plateau_particles=sum(d.get('optimizer_numerical_plateau_particles',0) for d in diagnostics),
            local_optimizer_motion_by_status=motion_by_status,
            local_optimizer_limited_particles=sum(d.get('optimizer_limited_particles',0) for d in diagnostics),
            local_optimizer_failed_particles=sum(d.get('optimizer_failed_particles',0) for d in diagnostics),
            local_optimizer_native_failed_particles=sum(d.get('optimizer_native_failed_particles',0) for d in diagnostics),
            local_optimizer_stationary_particles=sum(d.get('optimizer_stationary_particles',0) for d in diagnostics),
            local_optimizer_stalled_particles=sum(d.get('optimizer_stalled_particles',0) for d in diagnostics),
            neighborhood_bound_particles=sum(d.get('optimizer_neighborhood_bound_particles',0) for d in diagnostics),
            local_marginalization=marginalize,noise_updated=update_noise,
            posterior_effective_poses=float(np.mean(1/np.sum(posterior_weights**2,axis=1))) if marginalize else 1.,
            local_poll_evaluations=sum(d.get('optimizer_poll_evaluations',0) for d in diagnostics),
            local_poll_motion_pixels=.025 if local_search else None,
            local_optimizer_status_counts={str(int(code)):int(total) for code,total in zip(*np.unique(
                [code for d in diagnostics for code in d.get('optimizer_statuses',[])],return_counts=True))},
            alignment_mode=options.get('alignment_mode','trust_region'),
            signal_prior='fsc_from_accumulated_precision' if fsc_prior else 'reference_power',
            trust_accepted_steps=sum(d.get('trust_accepted_steps',0) for d in diagnostics),
            trust_rejected_steps=sum(d.get('trust_rejected_steps',0) for d in diagnostics),
            trust_median_condition=float(np.median([d['trust_median_condition'] for d in diagnostics])) if trust_region else None,
            reference_grid_size=int(refs[0].shape[0]),batch_observations=batch_observations,
            angular_search_range_deg=float(options['angular_range_deg'])*(controller.search_factor if automatic else 1.),
            shift_search_range_A=float(options['shift_range_A'])*(controller.search_factor if automatic else 1.),
            support_rejected_particles=sum(d.get('support_rejected_particles',0) for d in diagnostics),
            bound_particles=0 if incremental else int(np.any(np.isclose(abs(parameters),np.array([angle_range]*3+[shift_range]*3),atol=1e-6),axis=1).sum()))
        if automatic:
            curve=fsc_diagnostics['corrected'] if use_masked_fsc and fsc_prior else fsc_curve(*validation_maps,pixel,xp)
            if use_masked_fsc:
                if not fsc_prior:fsc_diagnostics=masked_fsc(*validation_maps,mask,pixel,xp)
                curve=fsc_diagnostics['corrected']
            control=controller.update(curve,angle_rms,shift_rms,(entry['support_rejected_particles']+entry['rejected_particles']+entry['local_optimizer_failed_particles']+entry['neighborhood_bound_particles'])/count if incremental else entry['bound_particles']/count)
            entry['auto_refine']=control
            entry['diagnostic_resolution_A']=diagnostic_resolution(curve)
            entry['fsc_mode']='phase_randomization_corrected' if use_masked_fsc else 'unmasked'
            if use_masked_fsc:entry['unmasked_resolution_A']=diagnostic_resolution(fsc_diagnostics['unmasked'])
            save_json(storage_path(output, 'fsc.json'),dict(curve=curve,mask_corrected=use_masked_fsc,unmasked=not use_masked_fsc,diagnostic_only=True,map_type='unregularized validation halves',mask_diagnostics=fsc_diagnostics if use_masked_fsc else None))
            if controller.radius>backend.radius+.01:
                objective_cache.clear();backend.radius=controller.radius
                active=xp.asarray((radii<=backend.radius)&(radii>0))
                compact_count=int(backend.host(active).sum())
                cache_objectives=backend.gpu and not staged and int(dataset.offsets[-1])*compact_count*24<xp.cuda.runtime.memGetInfo()[0]*likelihood_cache_fraction
        history.append(entry);save_json(storage_path(output, 'iterations.json'),history);print(entry,flush=True)
        if fsc_prior:save_json(storage_path(output, 'signal_prior.json'),dict(iteration=iteration,halves=prior_diagnostics,source='unregularized FSC and each half accumulated precision'))
        save_json(storage_path(output, 'latest_iteration.json'),dict(iteration=iteration,volumes=checkpoint,box_size=n,pixel_size_A=pixel,metrics=entry))
        if automatic and control['converged']:break
    volumes={}
    for name,volume in [('half1',maps[0]),('half2',maps[1]),('merged',.5*(maps[0]+maps[1]))]:
        target=storage_path(output, name+'.mrc')
        write_mrc(target,volume,pixel,xp=xp,overwrite=True)
        volumes[name]=str(target)
    final_curve=fsc_diagnostics['corrected'] if use_masked_fsc else fsc_curve(*(validation_maps if automatic else maps),pixel,xp)
    save_json(storage_path(output, 'fsc.json'),dict(curve=final_curve,mask_corrected=use_masked_fsc,unmasked=automatic and not use_masked_fsc,diagnostic_only=True,
        map_type='unregularized validation halves' if automatic else 'output halves',mask_diagnostics=fsc_diagnostics if use_masked_fsc else None))
    unfiltered_volumes={}
    if automatic:
        for h,volume in enumerate(validation_maps):
            path=storage_path(output, f'half{h+1}_unfiltered.mrc')
            write_mrc(path,volume,pixel,xp=xp,overwrite=True)
            unfiltered_volumes[f'half{h+1}']=str(path)
    fitted={p['grecoParticleId']:parameters[i] for i,p in enumerate(dataset.particles)};records=[]
    for particle in dataset.all_particles:
        row=dict(particle)
        if 'grecoImageStack' in row:row['grecoSourceImageStack']=row.pop('grecoImageStack')
        p=fitted.get(row['grecoParticleId']);row['grecoRefined']=int(p is not None)
        if p is not None:
            initial=Rotation.from_euler('ZYZ',[row['grecoAngle'+a] for a in ('Rot','Tilt','Psi')],degrees=True).as_matrix().T
            pose=initial@angles_to_matrix3(*p[:3])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');euler=Rotation.from_matrix(pose.T).as_euler('ZYZ',degrees=True)
            for a,v,delta in zip('XYZ',p[3:]*pixel,p[3:]*pixel):row['grecoCoordinate'+a+'Angst']+=v;row['grecoShift'+a+'Angst']=float(delta)
            for a,v in zip(('Rot','Tilt','Psi'),euler):row['grecoAngle'+a]=float(v)
        else:
            for a in 'XYZ':row['grecoShift'+a+'Angst']=0.
        records.append(row)
    star=storage_path(output, 'particles.star');temporary=storage_path(output, 'particles.star.tmp')
    with temporary.open('w') as stream:
        write_loop(stream,'general',['grecoFormatName','grecoFormatVersion','grecoSourceParticleStar','grecoPosesRefined'],[('greco-refinement-v1',1,str(dataset.path.resolve()),1)])
        write_loop(stream,'particles',list(records[0]),[tuple(row[c] for c in records[0]) for row in records])
    temporary.replace(star)
    result=dict(state='completed',particles=str(star),volumes=volumes,particle_count=len(dataset.all_particles),used_particles=count,
        box_size=n,pixel_size_A=pixel,iterations=len(history),maximum_iterations=iterations,
        convergence_state=('converged' if controller.stable>=3 else 'iteration_limit') if automatic else 'fixed_iterations',
        auto_refine=automatic,poses_refined=True,gpu_local_alignment=backend.gpu,
        alignment_mode=options.get('alignment_mode','trust_region'),reference_precision=reference_precision,volume_storage='float16',
        signal_prior='fsc_from_accumulated_precision' if fsc_prior else 'reference_power',
        local_marginalization=marginalize,noise_updated=update_noise,
        observation_model='ctf_premultiplied' if premultiplied else 'raw',
        particle_gradient_model=getattr(dataset,'particle_gradient_model','adapter'),
        dose_weighting=getattr(dataset,'dose_weighting',False),tilt_scale_weighting=getattr(dataset,'tilt_scale_weighting',False),
        noise_model=options.get('noise_model','view_power'),noise_diagnostics=noise_diagnostics,
        diagnostic_resolution_A=diagnostic_resolution(final_curve),unfiltered_volumes=unfiltered_volumes,
        fsc_mode='phase_randomization_corrected' if use_masked_fsc else 'unmasked' if automatic else 'output halves',
        gpu_likelihood_cache=bool(cache_objectives),likelihood_frequency_count=compact_count,
        gpu_backprojection=backend.gpu,gpu_particle_cache=cached,independent_half_refinement=not bool(dataset.general.get('grecoSharedReference',0)),
        map_reconstruction_backend='CuPy Greco skip-gridding MAP' if backend.gpu else 'greco._numerics CPU MAP',
        ctf_frozen_at_extraction=False,ctf_frozen_during_refinement=True,
        algorithm=('Local six-DOF Fourier likelihood; Gauss–Newton trust region; FSC signal prior' if trust_region else 'Local six-DOF Fourier likelihood; native libLBFGS; 13-node local pose posterior' if marginalize else 'Greco local six-DOF weighted Fourier likelihood; native independent libLBFGS / More-Thuente'),
        elapsed_seconds=time.perf_counter()-begin,options=options,half_counts=[int((halves==h).sum()) for h in (0,1)])
    check_snapshot(config['snapshot']);save_json(storage_path(output, 'web_result.json'),result);return result
