"""Fixed-pose, CTF-aware Gaussian-mixture EM for Greco tilt particles.

Greco Fourier projection, scatter and MAP reconstruction conventions; a new
driver for Greco's explicit per-view data, not a complete reference_refine port.
"""
from greco.project_layout import join as storage_path, ensure_output
from pathlib import Path
import json
import time
import hashlib
import numpy as np
from .fixed_pose import FixedPoseBackend,posterior
from .dataset import ParticleDataset
from .observation_model import backprojection_statistics


def initial_responsibilities(count,classes,seed):
    """Small balanced symmetry breaking, not near-hard random labels."""
    rng=np.random.default_rng(seed)
    labels=np.empty(count,np.int32)
    labels[rng.permutation(count)]=np.arange(count)%classes
    return .9/classes+.1*np.eye(classes)[labels]


def classify(config):
    import mrcfile
    from ..particle_star import write_loop
    from ..web.particle_worker import check_snapshot,save_json
    from greco._numerics.core.projector import initial_lowpass_filter_reference
    output=Path(config['output']);ensure_output(output)
    check_snapshot(config['snapshot']);begin=time.perf_counter()
    n=int(config['output_box']);pixel=float(config['pixel_size_A'])
    options=config['options'];k=int(options['classes']);iterations=int(options['iterations'])
    sources=[Path(__file__),Path(__file__).with_name('dataset.py'),Path(__file__).with_name('residual_noise.py'),
             storage_path(Path(__file__).parent.parent, 'particle_ctf.py'),Path(__file__).with_name('observation_model.py'),Path(__file__).with_name('transfer_envelope.py')]
    save_json(storage_path(output, 'protocol.json'),dict(protocol='sta-fixed-pose-native-v3',options=options,
        likelihood_domain='raw',initialization='soft-balanced-10-percent',
        extraction_manifest=str(config['extraction_manifest']),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    resolution=float(options['resolution_A']);regularization=float(options['regularization'])
    backend=FixedPoseBackend(n,config.get('gpu',True),radius=min(n//2-1,n*pixel/resolution))
    xp=backend.xp;dataset=ParticleDataset(config['extraction_manifest'],backend)
    # Extracted MRCS are raw. Evaluate BOTH E and M steps in that domain:
    # |F-T*P(V)|²/sigma² -> numerator F*T/sigma², weight T²/sigma².
    # The former premultiplied E step used |T*F-T²*P(V)|² with radial
    # noise, but its M step used F*T and T². Those are different objectives.
    # Using the available raw observations avoids dividing by CTF zeros and
    # estimates noise in the very same domain as the likelihood.
    if hasattr(dataset,'configure_observation_model'):
        dataset.configure_observation_model('raw')
    premultiplied=bool(getattr(dataset,'ctf_premultiplied',False))
    count=len(dataset.particles)
    if count<k*2:raise ValueError('At least two supported particles per initial class are required.')
    if not np.isclose(pixel,dataset.pixel):raise ValueError('Classification/extraction sampling differs.')
    # Memory guard includes K accumulation grids (three float64 channels each).
    if backend.gpu and k*np.prod(backend.shape)*24>xp.cuda.runtime.memGetInfo()[0]*.35:
        raise ValueError('Too many class grids for this box/GPU. Extract at a larger binning.')
    timings={};history=[];last_progress=[0.]
    def progress(phase,processed=0,total=count,iteration=0):
        now=time.monotonic()
        if processed and processed<total and now-last_progress[0]<.5:return
        last_progress[0]=now
        save_json(storage_path(output, 'progress.json'),dict(phase=phase,processed=processed,total=total,iteration=iteration,iterations=iterations,classes=k))
    tic=time.perf_counter()
    cached=dataset.cache(lambda done,total:progress('Caching particle Fourier data on GPU',done,total))
    timings['cache_seconds']=time.perf_counter()-tic
    y=np.where(np.arange(n)<=n//2,np.arange(n),np.arange(n)-n)
    radii=np.sqrt(y[:,None]**2+np.arange(n//2+1)[None,:]**2)
    shell=xp.asarray(np.floor(radii+.5).astype(np.int32));ns=n//2+1
    active=xp.asarray((radii<=backend.radius)&(radii>0))
    shell=xp.minimum(shell,ns-1)
    group_count=len(dataset.group_keys)
    noise_groups=dataset.groups
    frozen_noise=options.get('noise_model')=='half_residual'
    noise_diagnostics={}
    if frozen_noise:
        halves=np.array([p['grecoRandomSubset']-1 for p in dataset.particles])
        owner=np.repeat(np.arange(count),np.diff(dataset.offsets))
        noise_groups=dataset.groups+group_count*halves[owner]
        group_count*=2
    shell_count=xp.bincount(shell[active],minlength=ns).astype(xp.float64)
    def noise_sum(target,values,groups):
        bins=xp.asarray(groups)[:,None]*ns+shell[active][None,:]
        target+=xp.bincount(bins.ravel(),weights=values[:,active].ravel(),minlength=group_count*ns).reshape(group_count,ns)
    # Initial sigma^2 is variance of a real/imaginary component, per tilt/shell.
    power=xp.zeros((group_count,ns),xp.float64);samples=xp.zeros(group_count,xp.float64)
    initial_batches=() if frozen_noise else dataset.loaded_batches()
    for start,stop,f,_ in initial_batches:
        a,b=dataset.offsets[start],dataset.offsets[stop]
        noise_sum(power,.5*(f.real.astype(xp.float64)**2+f.imag.astype(xp.float64)**2),noise_groups[a:b])
        samples+=xp.bincount(xp.asarray(noise_groups[a:b]),minlength=group_count)
    denom=samples[:,None]*shell_count[None,:]
    sigma=power/xp.maximum(denom,1.)
    selected=sigma[:,backend.radius/2<np.arange(ns)]
    floor=max(float(xp.mean(selected)) if selected.size else float(xp.mean(sigma)),1e-20)*1e-6
    sigma=xp.maximum(sigma,floor)
    if frozen_noise:
        from .residual_noise import estimate_half_residual_noise
        progress('Estimating independent half residual noise')
        variance,noise_diagnostics=estimate_half_residual_noise(dataset,backend)
        sigma=xp.repeat(variance,len(dataset.group_keys),axis=0)
        np.save(storage_path(output, 'noise_variance.npy'),backend.host(variance))
        save_json(storage_path(output, 'noise_model.json'),noise_diagnostics)
    if options.get('mask_path'):
        from .reference_mask import load_mask
        mask=load_mask(options['mask_path'],n,pixel,resample_physical=True)
    else:
        # Fixed spherical soft solvent mask, no centroid or pose adjustment.
        z,yy,x=np.ogrid[:n,:n,:n]
        distance=np.sqrt((z-n/2)**2+(yy-n/2)**2+(x-n/2)**2)
        diameter=float(options.get('particle_diameter_A') or .9*n*pixel)
        radius=diameter/(2*pixel);edge=min(5.,radius/4)
        mask=np.where(distance<radius-edge,1.,np.where(distance<radius,.5+.5*np.cos(np.pi*(distance-(radius-edge))/edge),0.))
    q=xp.asarray(initial_responsibilities(count,k,int(options.get('seed',0))))
    priors=xp.full(k,1/k,xp.float64)

    def maximize(q, previous=None,power_spectra=None):
        accumulators=[backend.zeros() for _ in range(k)]
        for start,stop,f,c in dataset.loaded_batches():
            a,b=dataset.offsets[start],dataset.offsets[stop]
            owner=xp.asarray(np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1])))
            inv=active[None,:,:]/sigma[xp.asarray(noise_groups[a:b])][:,shell]
            base,weight=backprojection_statistics(f.astype(xp.complex128),c.astype(xp.float64),inv,premultiplied)
            for cl in range(k):
                resp=q[owner,cl,None,None]
                inverse=getattr(dataset,'inverse_rotations',None)
                backend.accumulate(accumulators[cl],base*resp,weight*resp,dataset.rotations[a:b],None if inverse is None else inverse[a:b])
        maps=[]
        for cl,acc in enumerate(accumulators):
            tau=None
            if previous is not None:
                if power_spectra is None:
                    backend.prepare_map(previous[cl]);tau=np.maximum(backend.projector.power_spectrum,1e-30)
                else:tau=power_spectra[cl]
            volume=backend.reconstruct(acc,tau,regularization)*mask
            if not np.isfinite(volume).all():raise ValueError('Non-finite class reconstruction.')
            maps.append(volume)
        return maps

    def expectation(maps, update_noise=True):
        refs=[];power_spectra=[]
        for v in maps:
            refs.append(backend.prepare_map(v))
            power_spectra.append(np.maximum(backend.projector.power_spectrum.copy(),1e-30))
        costs=xp.zeros((count,k),xp.float64)
        new_power=xp.zeros_like(power)
        # This is small P×K metadata; image buffers remain bounded/resident.
        for start,stop,f,c in dataset.loaded_batches():
            a,b=dataset.offsets[start],dataset.offsets[stop]
            groups=noise_groups[a:b]
            inv=active[None,:,:]/sigma[xp.asarray(groups)][:,shell]
            normalization=xp.sum(xp.where(active[None,:,:],xp.log(2*np.pi*sigma[xp.asarray(groups)][:,shell]),0.),axis=(1,2))
            owner=xp.asarray(np.repeat(np.arange(stop-start),np.diff(dataset.offsets[start:stop+1])))
            residuals=[]
            for cl,ref in enumerate(refs):
                inverse=getattr(dataset,'inverse_rotations',None)
                delta=f-backend.project(ref,dataset.rotations[a:b],None if inverse is None else inverse[a:b])*c
                residual=delta.real**2+delta.imag**2
                per_view=.5*xp.sum(residual*inv,axis=(1,2))
                # Gaussian normalization is class-independent but required to
                # compare likelihoods when the estimated noise changes.
                per_view+=normalization
                costs[start:stop,cl]=xp.bincount(owner,weights=per_view,minlength=stop-start)
                if update_noise:residuals.append(residual)
            batch_q,_=posterior(costs[start:stop],priors,xp)
            if update_noise:
                expected=sum(residuals[cl]*batch_q[owner,cl,None,None] for cl in range(k))*.5
                noise_sum(new_power,expected,groups)
        responsibilities,evidence=posterior(costs,priors,xp)
        updated=xp.maximum(new_power/xp.maximum(denom,1.),floor) if update_noise else sigma
        return responsibilities,float(xp.sum(evidence)),updated,power_spectra

    tic=time.perf_counter();progress('Initializing random balanced class subsets')
    maps=maximize(q)
    maps=[initial_lowpass_filter_reference(v,n,pixel,min(max(60.,resolution),n*pixel/4)) for v in maps]
    timings['initialization_seconds']=time.perf_counter()-tic
    for iteration in range(1,iterations+1):
        check_snapshot(config['snapshot']);tic=time.perf_counter()
        progress('Class likelihoods (fixed poses)',iteration=iteration)
        updated,evidence,new_sigma,power_spectra=expectation(maps,not frozen_noise)
        change=float(xp.mean(xp.abs(updated-q)));q=updated
        masses=backend.host(q.sum(axis=0));priors=xp.maximum(q.mean(axis=0),1e-12);priors/=priors.sum()
        sigma=new_sigma
        e_seconds=time.perf_counter()-tic;tic=time.perf_counter()
        progress('CTF/noise-weighted class MAP reconstruction',iteration=iteration)
        maps=maximize(q,maps,power_spectra)
        m_seconds=time.perf_counter()-tic
        entry=dict(iteration=iteration,log_evidence=evidence,mean_probability_change=change,
                   effective_counts=masses.tolist(),expectation_seconds=e_seconds,maximization_seconds=m_seconds)
        history.append(entry);save_json(storage_path(output, 'iterations.json'),history)
        print(json.dumps(entry),flush=True)
    # Export posteriors evaluated against the actual final maps, not stale maps.
    progress('Final class probabilities',iteration=iterations)
    q,evidence,_,_=expectation(maps,False);q=backend.host(q)
    assignments=q.argmax(axis=1);probabilities={p['grecoParticleId']:q[i] for i,p in enumerate(dataset.particles)}
    class_records=[]
    for cl,volume in enumerate(maps):
        target=storage_path(output, f'class_{cl+1:03d}.mrc')
        with mrcfile.new(target,overwrite=True) as m:m.set_data(np.asarray(volume,np.float32));m.voxel_size=pixel
        class_records.append(dict(id=cl+1,volume=str(target),particle_count=int((assignments==cl).sum()),effective_count=float(q[:,cl].sum())))
    # The extraction is immutable: classifications reference it explicitly.
    # Do not copy/rebase MRCS paths outside the image_path ownership boundary.
    star=storage_path(output, 'particles.star');temporary=storage_path(output, 'particles.star.tmp')
    with temporary.open('w') as stream:
        write_loop(stream,'general',['grecoFormatName','grecoFormatVersion','grecoSourceParticleStar',
            'grecoClassCount','grecoPosesRefined'],[('greco-classification-v1',1,str(Path(dataset.path).resolve()),k,0)])
        records=[]
        for p in dataset.all_particles:
            row=dict(p)
            if 'grecoImageStack' in row:row['grecoSourceImageStack']=row.pop('grecoImageStack')
            prob=probabilities.get(row['grecoParticleId'])
            row.update(grecoClassNumber=int(np.argmax(prob))+1 if prob is not None else 0,
                       grecoMaxClassProbability=float(np.max(prob)) if prob is not None else None)
            for cl in range(k):row[f'grecoClassProbability{cl+1}']=float(prob[cl]) if prob is not None else None
            records.append(row)
        write_loop(stream,'particles',list(records[0]),[tuple(row.values()) for row in records])
    temporary.replace(star)
    save_json(storage_path(output, 'model.json'),dict(likelihood_domain='raw',initialization='soft-balanced-10-percent',class_priors=backend.host(priors).tolist(),noise_sigma2=backend.host(sigma).tolist(),noise_groups=dataset.group_keys,
        noise_group_layout='half_then_view' if frozen_noise else 'view',noise_diagnostics=noise_diagnostics,options=options,iterations=history))
    result=dict(state='completed',classes=class_records,particles=str(star),particle_count=len(dataset.all_particles),
        used_particles=count,box_size=n,pixel_size_A=pixel,iterations=iterations,
        poses_refined=False,gpu_likelihoods=backend.gpu,gpu_backprojection=backend.gpu,
        observation_model='ctf_premultiplied' if premultiplied else 'raw',
        requested_observation_model=options.get('observation_model','raw'),
        initialization='soft-balanced-10-percent',
        particle_gradient_model=getattr(dataset,'particle_gradient_model','adapter'),
        dose_weighting=getattr(dataset,'dose_weighting',False),tilt_scale_weighting=getattr(dataset,'tilt_scale_weighting',False),
        noise_model=options.get('noise_model','view_residual_em'),noise_diagnostics=noise_diagnostics,
        gpu_particle_cache=cached,map_reconstruction_backend='CuPy Greco skip-gridding MAP' if backend.gpu else 'greco._numerics CPU MAP',
        algorithm='Fixed-pose CTF-aware Gaussian mixture EM; Greco projector/backprojector/MAP conventions',
        extraction_manifest=str(dataset.path),final_log_evidence=evidence,
        elapsed_seconds=time.perf_counter()-begin,timings=timings,independent_half_refinement=False)
    check_snapshot(config['snapshot']);save_json(storage_path(output, 'web_result.json'),result)
    return result
