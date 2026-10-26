"""Joint rigid/XYZ/GP3D polishing, exporting cumulative native STAR states."""
from greco.project_layout import join as storage_path
from pathlib import Path
import time
import numpy as np
from .gp3d import spatial_basis,MotionObjective
from .rigid_polishing import RigidMotionObjective,fit_rigid_motion
from .corrections import write_corrections
from .polishing_precision import precision


def run_joint(config,dataset,backend,refs,variance,begin,cached,surfaces,
              radius,padding,active,shell,progress):
    from ..web.particle_worker import save_json,check_snapshot
    from .dataset import ctf_batch
    xp=backend.xp;host=backend.host;options=config['options'];out=Path(config['output'])
    pixel=dataset.pixel;n=dataset.n;count=len(dataset.particles)
    owner=np.repeat(np.arange(count),np.diff(dataset.offsets))
    tomos=np.array([p['grecoTomogramId'] for p in dataset.particles])
    halves=np.array([p['grecoRandomSubset']-1 for p in dataset.particles])
    xyz=np.array([[p['grecoCoordinate'+d+'Angst'] for d in 'XYZ'] for p in dataset.particles])/pixel
    old=np.array([[dataset.observation_corrections.get(key,{}).get('grecoGpMotion'+d+'Angst',0.)
                   for d in 'XYZ'] for key in dataset.observation_keys])/pixel
    beam_hands=np.array([v.get('grecoDefocusHand',0)*(-1 if v.get('grecoRasterFlipY',False) else 1)
                        if 'grecoCentralDefocusUAngst' in v else 0. for v in dataset.views.values()])
    proposals={};fits=[];processed=0
    for tid in dict.fromkeys(tomos):
        groups=np.array([g for g,key in enumerate(dataset.group_keys) if key[0]==tid])
        rows=[dataset.views[dataset.group_keys[g]] for g in groups]
        have_dose=all('grecoPreExposureElectronPerAngst2' in r for r in rows)
        order=np.argsort([r['grecoPreExposureElectronPerAngst2'] if have_dose else r['grecoMdocIndex'] for r in rows])
        groups=groups[order];rows=[rows[i] for i in order]
        times=np.array([r['grecoPreExposureElectronPerAngst2'] if have_dose else r['grecoMdocIndex'] for r in rows])
        if np.any(np.diff(times)<=0):raise ValueError('Ambiguous acquisition order')
        dose=float(np.median([r.get('grecoExposureElectronPerAngst2',1.) for r in rows]))
        if options.get('gp3d_dose_scaling',True) and (not have_dose or dose<=0):raise ValueError('GP3D needs positive acquisition dose')
        gaps=np.diff(times)/dose if have_dose else np.diff(times)
        factor=dose if options.get('gp3d_dose_scaling',True) else 1.
        lookup={g:f for f,g in enumerate(groups)};blocks=[];positions=[];cameras=[];members=[]
        for half in (0,1):
            particles=np.flatnonzero((tomos==tid)&(halves==half))
            if not len(particles):continue
            obs=np.flatnonzero(np.isin(owner,particles));local=np.full(count,-1);local[particles]=np.arange(len(particles))
            owners=local[owner[obs]];frames=np.array([lookup[g] for g in dataset.groups[obs]])
            basis,eigenvalues=spatial_basis(xyz[particles],factor*options['velocity_sigma_A']/pixel,
                factor*options.get('divergence_sigma_A',5000.)/pixel,options.get('gp3d_max_modes',-1),xp)
            old_track=np.zeros((len(particles),len(groups),3))
            for p in range(len(particles)):
                mask=owners==p;sort=np.argsort(frames[mask]);f=frames[mask][sort]
                for d in range(3):old_track[p,:,d]=np.interp(np.arange(len(groups)),f,old[obs[mask],d][sort])
            velocity=np.diff(old_track,axis=1)/np.sqrt(gaps)[None,:,None]
            previous=((basis.T/eigenvalues[:,None])@xp.asarray(velocity).reshape(len(particles),-1)).reshape(len(eigenvalues),len(gaps),3).transpose(1,0,2)
            blocks.append(MotionObjective(basis,dataset.view_rotations[obs],owners,frames,
                xp.asarray(surfaces[obs]),radius,padding,gaps,options['maximum_shift_A']/pixel,xp,previous))
            positions.append(xyz[particles]);cameras.append(dataset.view_rotations[obs]);members.append((particles,obs))
        objective=RigidMotionObjective(blocks,positions,cameras,xp)
        name=dataset.tomos[tid]['grecoTomogramName'];label='Rigid + GP3D: '+name;start=time.perf_counter()
        parameters,diagnostic=fit_rigid_motion(objective,options.get('gp3d_iterations',10000),
            lambda iteration:progress(label,processed,optimizer_iteration=iteration))
        frame,q,_=objective.decode(parameters)
        static=np.zeros((count,3));dynamic=np.zeros((len(owner),3));shift=np.zeros((len(owner),2))
        for i,(particles,obs) in enumerate(members):
            b=blocks[i];part=parameters[objective.offsets[i]:objective.offsets[i+1]]
            track,_=b.trajectory(part);base=part[:3*b.pc].reshape(b.pc,3)
            static[particles]=host(base)
            dynamic[obs]=host(track[b.owners,b.frames]-base[b.owners])
            shift[obs]=host(objective.projected(parameters,i)[0])
        # Prior changes for whole-tomogram proposal fractions; static XYZ and
        # rigid frames are not incorrectly penalized as GP velocities.
        fractions=np.array([0.,.125,.25,.5,1.]);prior=np.zeros(len(fractions))
        for i,b in enumerate(blocks):
            _,coeff=b.trajectory(parameters[objective.offsets[i]:objective.offsets[i+1]])
            for j,f in enumerate(fractions):prior[j]+=float(xp.sum((b.previous+f*coeff)**2)-xp.sum(b.previous**2))
        proposals[tid]=dict(groups=groups,frame=host(frame),static=static,dynamic=dynamic,
            prior=prior,cost=np.zeros(len(fractions)),fractions=fractions)
        diagnostic.update(tomogram=name,seconds=time.perf_counter()-start,
            rigid_geometry_shared_between_halves=True,individual_motion_half_independent=True)
        fits.append(diagnostic);processed+=sum(len(a) for a,_ in members)
        save_json(storage_path(out, 'gp3d_fits.json'),fits);progress(label,processed)
        del objective,blocks,parameters,basis,previous
    # Compact CC is the Greco alignment objective. Validate exported state
    # against the *new* projected orientation and updated local beam-depth CTF,
    # not only shifted predictions made with the previous camera geometry.
    progress('Rigid + GP3D: checking updated Fourier likelihood')
    from .rigid_polishing import rotations_and_derivatives
    ky=xp.fft.fftfreq(n)[:,None]*2*xp.pi;kx=xp.arange(n//2+1)[None,:]*2*xp.pi/n
    for start,stop,data,ctf in dataset.loaded_batches(128):
        a,b=dataset.offsets[[start,stop]]
        for tid in dict.fromkeys(tomos[owner[a:b]]):
            proposal=proposals[tid];lookup={g:i for i,g in enumerate(proposal['groups'])}
            for h in (0,1):
                use=(tomos[owner[a:b]]==tid)&(halves[owner[a:b]]==h)
                if not use.any():continue
                obs=np.arange(a,b)[use];particles=owner[obs];frames=np.array([lookup[g] for g in dataset.groups[obs]])
                weight=precision(variance,h,shell,active,dataset.groups[obs],xp)
                base_cameras=xp.asarray(dataset.view_rotations[obs]);base_rotations=xp.asarray(dataset.rotations[obs])
                base_xyz=xp.asarray(xyz[particles]);static=xp.asarray(proposal['static'][particles]);dynamic=xp.asarray(proposal['dynamic'][obs])
                frame_parameters=xp.asarray(proposal['frame']);frame_indices=xp.asarray(frames)
                hand=xp.asarray(beam_hands[dataset.groups[obs]])
                old_transfer=ctf_batch(dataset.parameters[obs],n,pixel,xp)*dataset.transfer_envelope(a,b)[use]
                original=ctf[use]*backend.project(refs[h],dataset.rotations[obs],dataset.inverse_rotations[obs])
                for j,fraction in enumerate(proposal['fractions'][1:],start=1):
                    frame=frame_parameters*fraction;q,_=rotations_and_derivatives(frame[:,:3],xp)
                    rotation=q[frame_indices]@base_cameras
                    position=base_xyz+fraction*(static+dynamic)
                    delta=xp.einsum('mij,mj->mi',rotation[:,:2],position)-xp.einsum('mij,mj->mi',base_cameras[:,:2],base_xyz)+frame[frame_indices,3:]
                    params=xp.asarray(dataset.parameters[obs]).copy()
                    depth_change=(xp.sum(rotation[:,2]*(base_xyz+fraction*static),axis=1)-xp.sum(base_cameras[:,2]*base_xyz,axis=1))*hand*pixel
                    params[:,:2]+=depth_change[:,None]
                    new_ctf=ctf_batch(params,n,pixel,xp)*dataset.transfer_envelope(a,b)[use]
                    # The cached observation is raw * OLD transfer. Its forward
                    # model is therefore OLD * NEW transfer, not just NEW and
                    # not NEW squared. Never divide data at CTF zeros.
                    if dataset.ctf_premultiplied:new_ctf*=old_transfer
                    new_rotation=q[frame_indices]@base_rotations
                    prediction=new_ctf*backend.project(refs[h],new_rotation,new_rotation.transpose(0,2,1))
                    phase=delta[:,0,None,None]*kx+delta[:,1,None,None]*ky
                    # SSE difference without the huge constant |data|^2.
                    change=abs(prediction)**2-abs(original)**2-2*(data[use]*(xp.exp(1j*phase)*xp.conj(prediction)-xp.conj(original))).real
                    proposal['cost'][j]+=float(xp.sum(change*weight))
        progress('Rigid + GP3D: checking updated Fourier likelihood',stop)
    particles=[dict(p) for p in dataset.all_particles];by_id={p['grecoParticleId']:p for p in particles}
    corrections={key:dict(dataset.observation_corrections.get(key,{}),grecoParticleId=key[0],grecoViewId=key[1]) for key in dataset.observation_keys}
    views=[];accepted=[];total_change=0.;all_shift=np.zeros((len(owner),2))
    for tid,proposal in proposals.items():
        posterior=proposal['cost']+proposal['prior'];index=int(np.argmin(posterior));fraction=float(proposal['fractions'][index])
        total_change+=float(posterior[index]);frame=proposal['frame']*fraction;q,_=rotations_and_derivatives(frame[:,:3],np)
        lookup={g:f for f,g in enumerate(proposal['groups'])}
        for g,f in lookup.items():
            key=dataset.group_keys[g];view=dataset.views[key];old_camera=np.array([[view[f'grecoProjection{i}{j}'] for j in range(3)] for i in range(3)])
            new=q[f]@old_camera
            record=dict(grecoTomogramId=key[0],grecoViewId=key[1])
            record.update({f'grecoProjection{i}{j}':float(new[i,j]) for i in range(3) for j in range(3)})
            for d,value in zip('XY',frame[f,3:]*pixel):record['grecoGeometryShift'+d+'Angst']=view.get('grecoGeometryShift'+d+'Angst',0.)+float(value)
            views.append(record)
        for k in np.flatnonzero(tomos==tid):
            for d,value in zip('XYZ',fraction*proposal['static'][k]*pixel):by_id[dataset.particles[k]['grecoParticleId']]['grecoCoordinate'+d+'Angst']+=float(value)
        for observation in np.flatnonzero(tomos[owner]==tid):
            key=dataset.observation_keys[observation];f=lookup[dataset.groups[observation]];camera=q[f]@dataset.view_rotations[observation]
            dynamic=fraction*proposal['dynamic'][observation]
            for d,value in zip('XY',camera[:2]@dynamic*pixel):corrections[key]['grecoResidualShift'+d+'Angst']=corrections[key].get('grecoResidualShift'+d+'Angst',0.)+float(value)
            for d,value in zip('XYZ',(old[observation]+dynamic)*pixel):corrections[key]['grecoGpMotion'+d+'Angst']=float(value)
            k=owner[observation];position=xyz[k]+fraction*proposal['static'][k]+dynamic
            all_shift[observation]=camera[:2]@position-dataset.view_rotations[observation,:2]@xyz[k]+frame[f,3:]
        accepted.append(dict(tomogram=dataset.tomos[tid]['grecoTomogramName'],scale=fraction,
            posterior_change=float(posterior[index]),tested_data_changes=proposal['cost'].tolist(),tested_prior_changes=proposal['prior'].tolist(),
            frame_angles_deg=np.rad2deg(frame[:,:3]).tolist(),frame_shifts_A=(frame[:,3:]*pixel).tolist()))
    norms=np.linalg.norm(all_shift*pixel,axis=1)
    star=storage_path(out, 'particles.star');write_corrections(star,dataset.path,particles,list(corrections.values()),'polishing_rigid_gp3d',views)
    diagnostics=dict(motion_model='rigid_frames_static_xyz_half_independent_GP3D',tilt_geometry_refined=True,
        nonrigid_image_deformation=False,halves_independent=False,own_half_references=True,rigid_geometry_shared_between_halves=True,
        gauge='first acquisition frame anchored',fits=fits,gpu_optimizer=bool(backend.gpu),
        median_shift_A=float(np.median(norms)),maximum_shift_A=float(norms.max()),rms_shift_A=float(np.sqrt(np.mean(norms**2))),
        p95_shift_A=float(np.quantile(norms,.95)),regularized_objective_change=total_change,exact_likelihood_acceptance=accepted)
    result=dict(state='completed',protocol='sta-rigid-gp3d-polishing-v1',particles=str(star),
        particle_count=len(particles),used_particles=count,box_size=n,pixel_size_A=pixel,
        poses_refined=True,observation_corrections=True,gpu=backend.gpu,
        gpu_particle_cache=cached,options=options,diagnostics=diagnostics,reference_job=config.get('reference_job'),
        extraction_manifest=str(dataset.path),elapsed_seconds=time.perf_counter()-begin)
    check_snapshot(config['snapshot']);save_json(storage_path(out, 'web_result.json'),result);progress('Completed',count)
    return result
