"""Persistent linear refinement/curation steps beneath Reconstruction."""
from greco.project_layout import join as storage_path, preprocessing_work, series_row, owned as layout_owned

import json,uuid,re
from pathlib import Path
from .projects import InputError
from .preprocessing import refresh,render_script,command_result
KINDS=('geometry_refinement','manual_curation')

def ensure(row):
    changed=False
    if 'processing_steps' not in row:
        row['processing_steps']=[]
        changed=True
    used={s['number'] for s in row['processing_steps'] if 'number' in s}
    number=max(used|{1})
    for step in row['processing_steps']:
        if 'number' not in step:
            number+=1;step['number']=number;changed=True
    if row.get('last_refinement_number')!=number:
        row['last_refinement_number']=number;changed=True
    return changed

def find(row,identifier):
    ensure(row)
    matches=[s for s in row['processing_steps'] if s['id']==identifier]
    if len(matches)>1:raise InputError('Duplicate pipeline step identity.')
    step=matches[0] if matches else None
    if step is None:raise InputError('Unknown pipeline step.')
    if step['kind'] not in KINDS or (step['id'] not in ('geometry','curation') and not re.fullmatch(r'[a-f0-9]{8}',step['id'])):
        raise InputError('Invalid pipeline step identity.')
    return step

def job_for(row,step):
    return row.get('geometry_refinement',{}) if step['id']=='geometry' else step.get('job',{})

def work_for(root, sid, step):
    from greco.project_layout import work_path
    return storage_path(work_path(root,series_row(root,sid),'reconstruction'),'refinement',f"{step['number']:03d}")

def previous_geometry(row,step):
    items=row['processing_steps'];index=items.index(step)
    return next((s for s in reversed(items[:index]) if s['kind']=='geometry_refinement' or (s['kind']=='manual_curation' and s.get('job'))),None)

def curation_source(row,step):
    if step['kind']=='manual_curation' and step.get('job'):return 'curation_step_'+step['id']
    prior=previous_geometry(row,step)
    return 'reconstruction' if prior is None else 'curation_step_'+prior['id'] if prior['kind']=='manual_curation' else 'geometry' if prior['id']=='geometry' else 'geometry_step_'+prior['id']

def create(service,pid,sid,kind,*,reconstruction_id=None):
    if kind not in KINDS:raise InputError('Unknown pipeline step type.')
    with service.lock:
        root=service._root(pid);data=service._read(root)
        if reconstruction_id is not None and reconstruction_id!=data.get('active_reconstruction_assembly'):
            raise InputError('Reconstruction changed; reopen its card before adding a step.')
        from .reconstruction_assemblies import current
        record=current(data)
        if record and record['mode']!='standard':
            raise InputError('Geometry refinement and manual curation are available only in Simple reconstruction.')
        row=next((r for r in data['series'] if r['id']==sid),None)
        if row is None:raise InputError('Unknown tilt series.')
        ensure(row);row['last_refinement_number']+=1
        row['processing_steps'].append(dict(id=uuid.uuid4().hex[:8],kind=kind,number=row['last_refinement_number']))
        service._write(root,data);return service._view(pid)

def submit_all(service,pid,series_ids,options,*,step_id=None):
    """Submit missing first geometry steps, each with its own persistent job."""
    from .refinement import validate_geometry_options
    from .projects import active_series_ids
    options=validate_geometry_options(options)
    if (not isinstance(series_ids,list) or not series_ids or
            any(not isinstance(sid,str) for sid in series_ids) or len(set(series_ids))!=len(series_ids)):
        raise InputError('Select distinct tilt series.')
    with service.lock:
        root=service._root(pid);data=service._read(root)
        if not data.get('slurm',{}).get('ready'):raise InputError('Save cluster settings first.')
        rows={row['id']:row for row in data['series']}
        identifiers=active_series_ids(rows,series_ids,bulk=True)
        if step_id:
            anchor=next(((row,step) for row in data['series'] for step in row.get('processing_steps',[]) if step['id']==step_id),None)
            if anchor is None:raise InputError('Unknown pipeline step.')
            row,step=anchor
            if row['processing_steps'][0] is not step or step['kind']!='geometry_refinement':
                raise InputError('Submit all is only available for the first geometry step, directly after Reconstruct.')
        plans=[];errors=[];changed=False
        for sid in identifiers:
            row=rows[sid];ensure(row)
            step=row['processing_steps'][0] if row['processing_steps'] else None
            if step and step['kind']!='geometry_refinement':
                errors.append(row['name']+': submit geometry after Manual Curation individually.');continue
            if step and job_for(row,step):continue
            if not step and row.get('geometry_refinement'):continue
            parent=refresh(service,{'preprocessing':row.get('reconstruction',{})})
            if parent['state'] not in ('completed','partial','submitted','running'):
                errors.append(row['name']+': submit or complete Reconstruct first.');continue
            if step is None:
                step=dict(id=uuid.uuid4().hex[:8],kind='geometry_refinement',number=2)
                row['processing_steps'].append(step);ensure(row);changed=True
            plans.append((sid,step['id']))
        if changed:service._write(root,data)
        for sid,identifier in plans:
            try:submit(service,pid,sid,identifier,options)
            except (InputError,OSError,ValueError,KeyError) as error:
                errors.append(rows[sid]['name']+': '+str(error))
        result=service._view(pid);result['submission_errors']=errors
        return result

def submit(service,pid,sid,step_id,options):
    from .refinement import validate_geometry_options
    from .queue import dispatch
    options=validate_geometry_options(options)
    with service.lock:
        root=service._root(pid);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==sid),None)
        if row is None:raise InputError('Unknown tilt series.')
        step=find(row,step_id)
        if step['kind']!='geometry_refinement':raise InputError('Not a geometry step.')
        if step_id=='geometry':
            # Original first step keeps its historical namespace and UI data.
            from .refinement import submit as first_submit
            return first_submit(service,pid,[sid],options,kind='geometry_refinement')
        if job_for(row,step):raise InputError('This step already has a job. Delete it before creating a replacement.')
        prior=previous_geometry(row,step)
        parent=refresh(service,{'preprocessing':job_for(row,prior) if prior else row.get('reconstruction',{})})
        if parent['state'] not in ('completed','partial','submitted','running'):raise InputError('Submit or complete the preceding geometry first.')
        slurm=data.get('slurm',{})
        if not slurm.get('ready'):raise InputError('Save cluster settings first.')
        parent_work=service.path(parent['work'])
        if not layout_owned(root,parent_work):raise InputError('Invalid pipeline source.')
        source=json.loads((storage_path(parent_work, 'config.json')).read_text())
        if prior:
            config={k:source[k] for k in ('coarse_run','preprocessing_work','inputs','orientation','main_revision')}
            if parent.get('landmark_basis') or source.get('landmark_basis'):config['landmark_basis']=parent.get('landmark_basis') or source['landmark_basis']
            config['regenerate_landmarks']=prior['kind']=='manual_curation'
            config['resume_geometry']=str(storage_path(Path(source.get('output',storage_path(parent_work, 'output'))), 'rigid_geometry.npz'))
        else:
            config=dict(coarse_run=str(storage_path(parent_work, 'alignment')),inputs=source['inputs'],orientation=source['orientation'],
                main_revision=source['main_revision'],preprocessing_work=str(preprocessing_work(Path(source['inputs']['stack']))))
        if source.get('inputs',{}).get('binning')!=8:raise InputError('Geometry requires a bin8 source reconstruction.')
        pending=parent['state'] not in ('completed','partial');dependencies=[]
        if pending:
            if not parent.get('job_id') or parent.get('host','')!=slurm.get('submission_host',''):raise InputError('Pipeline dependency needs a Slurm ID on the same host.')
            dependencies=[parent['job_id']]
        else:
            required=Path(config['resume_geometry']) if prior else storage_path(Path(config['coarse_run']), 'final_state.npz')
            service.path(str(required))
            if not prior:
                import numpy as np
                from .refinement_worker import load_state
                load_state(required,np.loadtxt(config['inputs']['rawtlt'],ndmin=1),allow_partial=True)
        work=work_for(root,sid,step)
        from .final_denoising import prepare_model_reuse
        reuse=prepare_model_reuse(root,row,options,config['inputs'],options['final_binning'])
        if work.exists():raise InputError('Pipeline output directory already exists; refusing to overwrite it.')
        work.mkdir(parents=True)
        config.update(options=options,output=str(storage_path(work, 'output')),final_binning=options['final_binning'],
            pipeline_step_id=step_id,parent_step_id=row['processing_steps'][row['processing_steps'].index(step)-1]['id'] if row['processing_steps'].index(step) else None)
        paths=[storage_path(Path(config['coarse_run']), 'final_state.npz'),*[Path(config['inputs'][k]) for k in ('stack','rawtlt','frozen_shifts')],
               storage_path(Path(config['preprocessing_work']), 'config.json'),storage_path(Path(config['preprocessing_work']), 'input.mdoc'),storage_path(parent_work, 'config.json')]
        if prior and not pending:paths.append(Path(config['resume_geometry']))
        if config.get('landmark_basis') and not pending:paths.append(Path(config['landmark_basis']))
        config['snapshot']=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths if p.is_file()]
        config['snapshot'].extend(reuse.pop('tinycare_model_snapshot',[]));config.update(reuse)
        path=storage_path(work, 'config.json');path.write_text(json.dumps(config,indent=2))
        (storage_path(work, 'job.sbatch')).write_text(render_script(slurm['template'],work,path,dict(cpus=4,gpus=1),'greco.web.geometry_worker'))
        job=dict(state='submitted',work=str(work),host=slurm.get('submission_host',''),options=options,step_id=step_id)
        try:dispatch(job,False,command_result,dependencies=dependencies)
        except InputError as error:job.update(state='failed',error=str(error))
        step['job']=job;data['geometry_refinement_options']=options
        service._write(root,data);return service._view(pid)

def status(service,pid,sid,step_id):
    with service.lock:
        root=service._root(pid);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==sid),None)
        if row is None:raise InputError('Unknown tilt series.')
        step=find(row,step_id);job=refresh(service,{'preprocessing':job_for(row,step)})
        if job_for(row,step) and job_for(row,step)!=job:
            if step_id=='geometry':row['geometry_refinement']=job
            else:step['job']=job
            service._write(root,data)
        return dict(job,step_id=step_id)

def volume_png(service,pid,sid,step_id,index,axis='z',method='original',**contrast):
    from .reconstruction import volume_plane_image
    import io
    job=status(service,pid,sid,step_id)
    if job.get('state')!='completed':raise InputError('Geometry is not complete.')
    path=service.path(job.get('volumes',{}).get(method,job.get('tomogram') if method=='original' else ''))
    from ..reconstruction_layout import owned_volume
    if method not in ('original','tinycare') or not owned_volume(path,job['work']):raise InputError('Invalid geometry volume.')
    image=volume_plane_image(path,index,axis,denoised=method=='tinycare',**contrast);stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()

def delete(service,pid,sid,step_id,confirmed):
    if confirmed is not True:raise InputError('Confirm permanent step deletion.')
    with service.lock:
        root=service._root(pid);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==sid),None)
        if row is None:raise InputError('Unknown tilt series.')
        step=find(row,step_id)
        if job_for(row,step):
            from .preprocessing_deletion import delete as delete_job
            return delete_job(service,pid,sid,job_for(row,step)['work'],True,kind=step['kind'],step_id=step_id)
        work=work_for(root,sid,step)
        if work.exists() or work.is_symlink():
            from .preprocessing_deletion import delete as delete_job
            return delete_job(service,pid,sid,str(work),True,kind=step['kind'],step_id=step_id,unsubmitted=True)
        row['processing_steps'].remove(step)
        if row.get('picking_step_id')==step_id:row.update(picking_source=None,picking_step_id=None)
        service._write(root,data);return service._view(pid)
