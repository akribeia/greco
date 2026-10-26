"""Shared STA job submission, extraction building and result inspection."""
from greco.project_layout import join as storage_path, ensure_output, job_work, preprocessing_work, owned as layout_owned
from greco.preprocessing_star import read_table
import json
from pathlib import Path
from .projects import InputError
from .preprocessing import refresh,render_script,command_result
from .metadata_cache import read_json,read_scientific_metadata
from .assemblies import PARTICLE_KINDS

ACTIVE=('submitted','running','submitting','unknown')
from ..runtime import ROOT
REFERENCE=storage_path(ROOT, 'greco')


def refinement_live_files(output, service=None):
    """Resolve only known alternating blocks, including already running jobs."""
    import re
    output=Path(output);block=output;previous=output
    try:root_progress=read_json(service,storage_path(output, 'progress.json'))
    except (ValueError,OSError):root_progress={}
    phase=str(root_progress.get('phase',''))
    if phase=='Initial pose convergence':block=previous=storage_path(output, 'alternating/initial')
    match=re.fullmatch(r'(Per-tilt shift cycle|Pose convergence after shift cycle) ([1-9]|[12][0-9]|30)',phase)
    if match:
        cycle=int(match[2]);folder=storage_path(output, f'alternating/cycle_{cycle:02d}')
        block=storage_path(folder, 'shifts' if match[1]=='Per-tilt shift cycle' else 'poses')
        previous=storage_path(output, 'alternating/initial') if cycle==1 else storage_path(output, f'alternating/cycle_{cycle-1:02d}/poses')
        if cycle==1 and (storage_path(output, 'alternating/ctf_before_poses/latest_iteration.json')).exists():
            previous=storage_path(output, 'alternating/ctf_before_poses')
    ctf=re.fullmatch(r'CTF refinement (before|after)',phase)
    if ctf:
        label=ctf[1];poses=storage_path(output, f'alternating/ctf_{label}_poses')
        block=poses if (storage_path(poses, 'progress.json')).exists() else storage_path(output, f'alternating/ctf_{label}')
        candidates=sorted((storage_path(output, 'alternating')).glob('cycle_*/poses/**/latest_iteration.json'))
        previous=(candidates[-1].parent.parent if candidates[-1].parent.name=='optimisation' else candidates[-1].parent) if label=='after' and candidates else storage_path(output, 'alternating/initial')
    latest=storage_path(block, 'latest_iteration.json')
    if not latest.is_file():latest=storage_path(previous, 'latest_iteration.json')
    try:progress=read_json(service,storage_path(block, 'progress.json'))
    except (ValueError,OSError):progress=root_progress.copy()
    if block!=output:
        progress['phase']=phase+' · '+str(progress.get('phase','Starting'))
    return latest,progress


def assembly(data,identifier):
    result=next((a for a in data.get('particle_assemblies',[]) if a['id']==identifier),None)
    if result is None:raise InputError('Particle assembly not found.')
    return result


def update(service,record,live_states=None,kind='extraction'):
    original=record.get(kind,{})
    if original and not original.get('work'):return dict(original)
    job=refresh(service,{'preprocessing':original},live_states)
    if job.get('work') and job['state'] in ACTIVE:
        from .sta_dependencies import pending_particle_counts
        counts=pending_particle_counts(service,job['work'])
        if counts is not None:job.update(counts)
        progress=storage_path(storage_path(Path(job['work']), 'output'), 'progress.json')
        if progress.exists():
            try:job['progress']=read_json(service,progress)
            except (ValueError,OSError):pass
        if kind=='polishing':
            fits_path=storage_path(Path(job['work']), 'output/gp3d_fits.json')
            try:
                from ..sta_layout import read_metadata
                fits=read_scientific_metadata(service,fits_path)
                job['diagnostics']={**job.get('diagnostics',{}),'fits':fits,
                    'gpu_optimizer':bool(fits) and all(f.get('optimizer')=='device_lbfgs' for f in fits)}
            except (ValueError,OSError):pass
    if kind=='refinement3d' and job.get('work'):
        if job['state'] in ACTIVE:
            latest,live_progress=refinement_live_files(storage_path(Path(job['work']), 'output'),service)
        else:
            latest,live_progress=storage_path(Path(job['work']),'result.json'),{}
        if job['state'] in ACTIVE:job['progress']=live_progress
        total=live_progress.get('total')
        if job.get('used_particles') is None and type(total) is int and total>0:
            job['used_particles']=total
        if not latest.is_file() and job.get('state')=='completed':latest=storage_path(Path(job['work']),'result.json')
        if latest.is_file():
            try:
                preview=read_json(service,latest)
                job['latest_iteration']=int(preview['iteration'])
                job['box_size']=int(preview['box_size']);job['pixel_size_A']=float(preview['pixel_size_A'])
                job['refinement_metrics']=preview.get('metrics',{})
                job['preview_label']='Half 1 + Half 2 average' if 'merged' in preview.get('volumes',{}) else 'Half 1'
            except (ValueError,OSError,KeyError,TypeError):pass
    if kind=='postprocessing' and job.get('state')=='completed' and job.get('work'):
        
        fsc=storage_path(Path(job['work']),'fsc.json')
        if fsc.suffix=='.star' and fsc.is_file():job['postprocess']=read_scientific_metadata(service,fsc)
    return job


def submit(service,project_id,identifier,*,kind='extraction',options=None,build_only=False):
    options=dict(options) if isinstance(options,dict) else options
    queue_only=options.pop('queue_only',False) if isinstance(options,dict) else False
    if type(queue_only) is not bool:raise InputError('Invalid queue option.')
    if kind!='extraction':raise InputError('Invalid assembly task.')
    from .assemblies import extraction_sampling
    with service.lock:
        root=service._root(project_id);data=service._read(root);record=assembly(data,identifier)
        slurm=data.get('slurm',{})
        if not slurm.get('ready'):raise InputError('Save cluster settings first.')
        from .sta_dependencies import maybe_defer
        deferred=maybe_defer(service,project_id,root,data,record,kind,options,queue_only)
        if deferred is not None:return deferred
        if not (storage_path(REFERENCE, '_numerics/tomo/extraction.py')).is_file():raise InputError('Greco reconstruction reference is unavailable.')
        if kind=='extraction' and any(m.get('picking_dependency') for m in record['tomograms']):
            from .pending_extraction import submit_pending
            return submit_pending(service,project_id,root,data,record,options)
        class_source=None
        if options is not None:
            from .assemblies import BINNINGS
            if kind!='extraction' or not isinstance(options,dict) or set(options)-{'binning','from','classes'}:raise InputError('Unknown extraction parameters.')
            source=options.get('from','assembly:'+record['id'])
            if source!='assembly:'+record['id']:
                if not isinstance(source,str) or not source.startswith(('classification:','refinement3d:','polishing:','ctf_refinement:')):raise InputError('This coordinate source is unavailable.')
                source_kind=source.split(':',1)[0]
                try:job=volume_job(service,project_id,identifier,source_kind,Path(source.split(':',1)[-1]).name)
                except InputError as error:raise InputError('This coordinate source is unavailable.') from error
                if job['state']!='completed' or source!=source_kind+':'+str(job.get('work')):
                    raise InputError('This coordinate source is unavailable. Select a completed Class3D from this assembly.')
                classes=options.get('classes')
                if source_kind=='classification':
                    available={c['id'] for c in job['classes']}
                    if not isinstance(classes,list) or not classes or any(type(c) is not int or c not in available for c in classes):
                        raise InputError('Select at least one valid class.')
                elif classes:raise InputError('Classes require a Class3D source.')
                class_source=dict(kind=source_kind,work=job['work'],classes=sorted(set(classes or [])),particles=job['particles'])
            elif options.get('classes'):raise InputError('Classes require a Class3D source.')
            binning=options.get('binning',4)
            if isinstance(binning,bool) or not isinstance(binning,int) or binning not in BINNINGS:raise InputError('Choose binning 1, 2, 3, 4, 6, 8 or 16.')
            native=float(record.get('native_pixel_size_A',record['pixel_size_A']/record['binning']))
            record.update(binning=binning,pixel_size_A=native*binning,native_pixel_size_A=native,sampling_configured=True)
            record.pop('output_box_size',None);record.pop('output_pixel_size_A',None)
        sampling=extraction_sampling(record)
        if kind=='extraction':record.update(output_box_size=sampling['box_size'],output_pixel_size_A=sampling['pixel_size_A'],sampling_configured=True)
        pixel=sampling['pixel_size_A'];native=sampling['native_pixel_size_A'];size=sampling['box_size']
        if record['particle_count']<1:raise InputError('Not enough particles for extraction.')
        snapshot=[];members=[]
        rows={r['id']:r for r in data['series']}
        def checked(path):
            p=service.path(str(path))
            if not p.is_file() or not layout_owned(root,p):raise InputError('Assembly inputs must exist inside this project.')
            stat=p.stat();snapshot.append(dict(path=str(p),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
            return p
        selected_by_uid={};geometry_by_uid={}
        if class_source:
            from ..particle_star import rows as star_rows
            from .assemblies import assembly_directory
            class_path=checked(class_source['particles'])
            if class_path!=storage_path(Path(class_source['work']), 'output/particles.star') or not class_path.is_relative_to(assembly_directory(root,record)):
                raise InputError('Invalid classification source.')
            table=list(star_rows(class_path))
            general=next((r for b,r in table if b=='general'),{})
            refined=class_source['kind']!='classification'
            expected='greco-observation-refinement-v1' if class_source['kind'] in ('polishing','ctf_refinement') else 'greco-refinement-v1' if refined else 'greco-classification-v1'
            if general.get('grecoFormatName')!=expected or general.get('grecoPosesRefined')!=int(refined):raise InputError('Unsupported source coordinates.')
            corrections={};view_indices={}
            original=checked(general['grecoSourceParticleStar'])
            original_general=next((r for b,r in star_rows(original) if b=='general'),{})
            if refined or original_general.get('grecoFormatName') in ('greco-star-mrcs-v1','greco-refinement-v1','greco-observation-refinement-v1'):
                from ..sta.corrections import reextraction_tables
                original,chain,_,corrections=reextraction_tables(class_path)
                for path in [original,*chain]:checked(path)
                view_indices={(v['grecoTomogramId'],v['grecoViewId']):v['grecoMdocIndex'] for b,v in star_rows(original) if b=='views'}
            tomos={r['grecoTomogramId']:r['grecoTomogramUid'] for b,r in star_rows(original) if b=='tomograms'}
            from ..sta.corrections import corrected_views
            geometry={(r['grecoTomogramId'],r['grecoViewId']):r for b,r in star_rows(original) if b=='views' and 'grecoGeometryShiftXAngst' in r}
            geometry.update(corrected_views(class_path))
            for key,value in geometry.items():
                geometry_by_uid.setdefault(tomos[key[0]],[]).append(dict(value,mdoc_index=view_indices[key]))
            corrections_by_pid={}
            for (pid,view),value in corrections.items():corrections_by_pid.setdefault(pid,[]).append((view,value))
            for block,p in table:
                if block=='particles' and (refined or p['grecoClassNumber'] in class_source['classes']):
                    if corrections:
                        p=dict(p,_observation_corrections=[dict(value,mdoc_index=view_indices[(p['grecoTomogramId'],view)]) for view,value in corrections_by_pid.get(p['grecoParticleId'],[])])
                    full_pose=all(k in p for k in ['grecoRandomSubset']+['grecoAngle'+a for a in ('Rot','Tilt','Psi')])
                    selected_by_uid.setdefault(tomos[p['grecoTomogramId']],[]).append(p if refined or full_pose else [p['grecoCoordinate'+a+'Angst'] for a in 'XYZ'])
            if not selected_by_uid:raise InputError('The selected classes contain no particles.')
        for member in record['tomograms']:
            if class_source and member['series_id'] not in selected_by_uid:continue
            row=rows.get(member['series_id'])
            if row is None:raise InputError('An assembly tomogram was removed.')
            for source_kind in ('preprocessing','reconstruction','picking','refinement','angle_refinement'):
                if row.get(source_kind) and refresh(service,{'preprocessing':row[source_kind]})['state'] in ACTIVE:
                    raise InputError(member['name']+': wait for active jobs before averaging.')
            coordinates=checked(member['star']);stat=coordinates.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(member['star_size'],member['star_mtime_ns']):
                raise InputError(member['name']+': picks changed. Create a new assembly.')
            volume=checked(member['volume']);work=job_work(volume)
            source=checked(storage_path(work, 'config.json'));result_path=checked(storage_path(work, 'output/web_result.json'))
            result=json.loads(result_path.read_text());transform=result.get('internal_to_export_zyx')
            rigid_geometry=None
            if result.get('kind')=='geometry_refinement':
                settings=json.loads(source.read_text())
                # The refined volume owns full cameras; its coarse run owns
                # the original tilt identities and cumulative 2D shifts.
                coarse=checked(storage_path(Path(settings['coarse_run']), 'final_state.npz')).parent
                checked(storage_path(coarse, 'summary.json'))
                from .picking_source import volume_context
                _,_,result,context=volume_context(work)
                if Path(context['picking_source_run'])!=coarse:
                    raise InputError('Refined tomogram lineage does not match its alignment.')
                rigid_geometry=checked(context['rigid_geometry'])
                if transform is None:raise InputError('Tomogram export geometry is missing. Reconstruct the tomogram again.')
                state=checked(storage_path(coarse, 'final_state.npz'))
            else:
                if transform is None:raise InputError('Tomogram export geometry is missing. Reconstruct the tomogram again.')
                state=checked(storage_path(work, 'output/alignment/final_state.npz'))
            star=checked(storage_path(Path(member['picking_work']), 'output/particles.star'))
            from ..picker.io import read_star
            import math
            _,particles=read_star(star)
            required=['grecoCoordinate'+a+'Angst' for a in 'XYZ']+['grecoAngle'+a for a in ('Rot','Tilt','Psi')]
            if len(particles)!=member['particle_count'] or any(
                any(not isinstance(p.get(key),(int,float)) or not math.isfinite(p[key]) for key in required) for p in particles):
                raise InputError(member['name']+': picking coordinates/orientations are incomplete.')
            recon=json.loads(source.read_text());prep=preprocessing_work(Path(recon['inputs']['stack']))
            checked(storage_path(prep, 'config.json'));checked(recon['inputs']['stack']);frozen=checked(recon['inputs']['frozen_shifts'])
            if (storage_path(prep, 'input.mdoc')).exists():checked(storage_path(prep, 'input.mdoc'))
            
            tilts=read_table(frozen)
            for tilt in tilts:
                folder=storage_path(storage_path(prep, 'output/per_tilt'), f"tilt_{int(tilt['mdoc_index']):04d}")
                checked(storage_path(folder, 'ctf.json'));checked(storage_path(folder, 'shifts.npz'))
                if (storage_path(folder, 'ctf_geometry.json')).exists():checked(storage_path(folder, 'ctf_geometry.json'))
                if (storage_path(folder, 'movie.json')).exists():checked(storage_path(folder, 'movie.json'))
                for saved in folder.glob('mc_bin*.mrc'):checked(saved)
            members.append(dict(member,reconstruction_config=str(source),final_state=str(state),
                star=str(star),preprocessing_work=str(prep),export_transform=transform,tilt_count=len(tilts)))
            if rigid_geometry is not None:members[-1]['rigid_geometry']=str(rigid_geometry)
            if class_source:
                selected=selected_by_uid[member['series_id']]
                members[-1].update(source_particle_count=member['particle_count'],particle_count=len(selected))
                members[-1]['refined_particles' if isinstance(selected[0],dict) else 'selected_coordinates_A']=selected
                if member['series_id'] in geometry_by_uid:members[-1]['refined_views']=geometry_by_uid[member['series_id']]
        config=dict(reference_root=str(REFERENCE),tomograms=members,
            output_box=size,pixel_size_A=pixel,native_pixel_size_A=native,snapshot=snapshot,gpu=True,
            action='extract',
            assembly_id=record['id'],assembly_sampling=dict(sampling,native_box_size=record['box_size'],
                binning=record['binning'],requested_extent_A=record['dimensions_A'],additional_binning=1))
        if kind=='extraction':config['coordinate_source']=dict(kind='assembly',assembly_id=record['id'],name=record['name'])
        if class_source:config['coordinate_source']=class_source
        if build_only:return config
        return enqueue(service,project_id,root,data,record,config,kind,queue_only=queue_only)


def config_uses_work(config,work):
    """Match exact paths under an input job, not coincidental name substrings."""
    if isinstance(config,dict):return any(config_uses_work(v,work) for v in config.values())
    if isinstance(config,(list,tuple)):return any(config_uses_work(v,work) for v in config)
    if isinstance(config,str):
        return config==str(work) or config.startswith(str(work)+'/')
    return False


def source_dependencies(service,record,config,host,*,validated_sources=None):
    config={k:v for k,v in config.items() if k not in ('output','project_snapshot')}
    candidates={}
    for task in PARTICLE_KINDS:
        if record.get(task):candidates[record[task]['work']]=(task,record[task])
    for job in record.get('job_history',[]):
        if job.get('work') and job.get('kind'):candidates.setdefault(job['work'],(job['kind'],job))
    dependencies=[]
    for work,(task,job) in candidates.items():
        if not config_uses_work(config,work):continue
        current=(validated_sources or {}).get(work)
        if current is None:current=update(service,{task:job},kind=task)
        if current.get('state')=='completed':continue
        if current.get('state') not in ACTIVE or not current.get('job_id'):
            raise InputError('The selected source job is not completed or submitted.')
        if current.get('host','')!=host:raise InputError('Source dependencies must use the same submission host.')
        dependencies.append(current['job_id'])
    return tuple(dict.fromkeys(dependencies))


def enqueue(service,project_id,root,data,record,config,kind,*,dependencies=(),queue_only=False,validated_sources=None):
    from .assemblies import PARTICLE_KINDS
    if kind not in PARTICLE_KINDS:raise InputError('Invalid STA job type.')
    if getattr(service,'build_only',False):return config
    from .queue import dispatch
    from .assemblies import allocate_job_directory
    if not queue_only:
        host=data['slurm'].get('submission_host','')
        dependencies=tuple(dict.fromkeys([*dependencies,*source_dependencies(service,record,config,host,validated_sources=validated_sources)]))
    work,number=allocate_job_directory(root,record,kind)
    if service.path(str(work),exists=False)!=work:raise InputError('Invalid particle job directory.')
    ensure_output(storage_path(work, 'output'));config=dict(config,job_kind=kind,output=str(storage_path(work, 'output')))
    from ..sta_layout import freeze_mask
    config=freeze_mask(config)
    path=storage_path(work, 'config.json');path.write_text(json.dumps(config,indent=2))
    slurm=data['slurm']
    script=render_script(slurm['template'],work,path,dict(cpus=4,gpus=1),'greco.web.particle_worker')
    if kind=='postprocessing' and config['options'].get('backend','cpu')=='cpu':
        import re
        script=re.sub(r'^#SBATCH\s+--(?:gres|gpus|gpus-per-node|gpus-per-task)(?:[=\s].*)?\n?', '',script,flags=re.M)
    (storage_path(work, 'job.sbatch')).write_text(script)
    job=dict(state='submitted',work=str(work),host=slurm.get('submission_host',''),box_size=config['output_box'],pixel_size_A=config['pixel_size_A'])
    job.update(number=number,name=work.name,kind=kind)
    for key in ('particle_count','used_particles'):
        if key in config:job[key]=config[key]
    if 'coordinate_source' in config:job['coordinate_source']=config['coordinate_source']
    if 'particle_source' in config:job['particle_source']=config['particle_source']
    if kind in ('classification','refinement3d','polishing','ctf_refinement','postprocessing'):job['options']=config['options']
    try:dispatch(job,queue_only,command_result,dependencies=dependencies)
    except InputError as error:job.update(state='failed',error=str(error))
    previous=record.get(kind)
    if previous and previous.get('work'):
        history=record.setdefault('job_history',[])
        if not any(item.get('work')==previous['work'] for item in history):
            history.append(dict(previous,kind=kind))
    record[kind]=job
    if kind=='extraction':record['extracted']=False
    service._write(root,data)
    return service._view(project_id)


def status(service,project_id,identifier,*,kind='extraction'):
    with service.lock:
        root=service._root(project_id);data=service._read(root);record=assembly(data,identifier)
        job=update(service,record,kind=kind)
        if record.get(kind) and job!=record[kind]:
            record[kind]=job
            if kind=='extraction':record['extracted']=job['state']=='completed'
            service._write(root,data)
        return job


def volume_job(service,project_id,identifier,kind,job_name=None):
    if not job_name:return status(service,project_id,identifier,kind=kind)
    from .assemblies import assembly_directory
    with service.lock:
        root=service._root(project_id);record=assembly(service._read(root),identifier)
        candidates=[record.get(kind,{})]+[j for j in record.get('job_history',[]) if j.get('kind')==kind]
        job=next((j for j in candidates if Path(j.get('work','')).name==job_name),None)
        if job is None:raise InputError('Unknown assembly job.')
        work=service.path(job['work'])
        if not work.is_relative_to(assembly_directory(root,record)) or work.is_symlink():raise InputError('Invalid assembly job.')
        return job


def plane(service,project_id,identifier,index,axis='z',job_name=None,kind='refinement3d'):
    import io
    from .sta_display import volume_image
    if kind not in ('refinement3d','postprocessing'):raise InputError('Invalid volume job kind.')
    job=volume_job(service,project_id,identifier,kind,job_name)
    output=storage_path(Path(job['work']), 'output')
    if kind=='refinement3d' and job['state']!='completed':
        latest,_=refinement_live_files(output,service)
        try:preview=read_json(service,latest);iteration=int(preview['iteration'])
        except (OSError,ValueError,KeyError,TypeError):raise InputError('No complete refinement iteration available yet.')
        if not 1<=iteration<=100:raise InputError('Invalid refinement iteration.')
        preview_kind='half1'
        expected=storage_path(output, f'iter_{iteration:03d}_{preview_kind}.mrc')
        path=service.path(preview['volumes'][preview_kind])
    else:
        if job['state']!='completed':raise InputError('STA volume is not complete.')
        label='merged'
        expected=storage_path(output, label+'.mrc');path=service.path(job['volumes'][label])
        if kind=='postprocessing':expected=storage_path(output, 'volumes/map_masked.mrc' if job.get('postprocess',{}).get('mask_corrected') else 'volumes/map.mrc')
    if path!=expected or not layout_owned(service._root(project_id),path):
        raise InputError('Invalid STA volume path.')
    image=volume_image(path,index,axis=axis)
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()
