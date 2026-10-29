"""Persistent queue, separate from the currently viewable job results."""
from greco.project_layout import join as storage_path, work_path as layout_work_path, owned as layout_owned
import json

import shutil
from pathlib import Path
from .projects import InputError
from .assemblies import PARTICLE_KINDS
KINDS=('preprocessing','reconstruction','angle_refinement','geometry_refinement','refinement','picking')
ACTIVE=('submitting','submitted','running','unknown')

def needs_submission(job):
    """Bulk submissions retry terminal failures, never completed/active jobs."""
    if job.get('state') in ('completed',*ACTIVE,'queued'):return False
    return job.get('state') in ('pending','failed','cancelled') or not job.get('work') and not job.get('job_id')

def _work_path(root, row, kind, queued=False):
    return layout_work_path(root, row, kind, queued)

def _reset_work(service, root, path, kind=None):
    if path.is_symlink() or path.parent.is_symlink():
        raise InputError('Symlinked job directories cannot be overwritten.')
    resolved = service.path(str(path), exists=False)
    if resolved != path or not layout_owned(root, resolved):
        raise InputError('Invalid reusable job directory.')
    if path.exists():
        if path.name == 'reconstruction' or kind=='reconstruction':
            refinements = storage_path(path, 'refinement')
            if refinements.exists() or refinements.is_symlink():
                if refinements.is_symlink() or not refinements.is_dir() or any((child.is_symlink() or not child.is_dir() for child in refinements.rglob('*'))):
                    raise InputError('Delete reconstruction refinements before replacing their source.')
        shutil.rmtree(path)
    path.mkdir(parents=True)
    storage_path(path, 'logs').mkdir(exist_ok=True)
    return path

def guard_preprocessing_dependencies(data,series_id,*,allow_queued_dependencies=False):
    """A reconstruction retains its preprocessing inputs after completion too."""
    row=next((r for r in data['series'] if r['id']==series_id),None)
    if row is None:raise InputError('Unknown tilt series.')
    owners=[]
    for record in data.get('reconstruction_assemblies',[]):
        saved=row if record['id']==data.get('active_reconstruction_assembly') else record.get('series',{}).get(series_id,{})
        existing=saved.get('reconstruction',{}).get('work')
        queued=saved.get('queued_reconstruction') or saved.get('queued_jobs',{}).get('reconstruction',{})
        awaiting_preprocessing=False
        if allow_queued_dependencies and not existing and queued.get('state')=='queued' and queued.get('work'):
            try:
                config=json.loads((Path(queued['work'])/'config.json').read_text())
                awaiting_preprocessing=config.get('preprocessing_dependency') is True and config.get('inputs') is None
            except (OSError,ValueError):pass
        if existing or (queued.get('work') and not awaiting_preprocessing):owners.append(record['name'])
    if not data.get('reconstruction_assemblies') and row.get('reconstruction',{}).get('work'):owners.append('Reconstruction')
    if owners:raise InputError(row['name']+': preprocessing is used by '+', '.join(owners)+'. Delete the dependent reconstruction jobs before rerunning preprocessing.')


def prepare_work(service, root, row, kind, queue_only=False, *, allowed_dependencies=()):
    if not isinstance(queue_only, bool):
        raise InputError('Invalid queue option.')
    if kind in ('preprocessing','ctf_redo'):
        guard_preprocessing_dependencies(service._read(root),row['id'],
            allow_queued_dependencies=kind=='preprocessing' and not queue_only and bool(row.get('queued_jobs',{}).get('preprocessing')))
    if not queue_only and kind=='reconstruction':
        from greco.project_layout import config_paths
        from .particle_jobs import config_uses_work
        target=_work_path(root,row,kind)
        for config_path in config_paths(root,kinds=('picking',)):
            from .metadata_cache import read_json
            try:config=read_json(service,config_path)
            except (ValueError,OSError) as error:raise InputError('Cannot verify picking dependencies.') from error
            if config_uses_work({k:v for k,v in config.items() if k not in ('series','project_snapshot','source_result')},target):
                raise InputError(row['name']+': picking uses this reconstruction. Create another reconstruction or delete the dependent picking first.')
    if not queue_only:
        from .preprocessing import refresh
        data = service._read(root)
        if kind != 'picking':
            from .pick_assemblies import inactive_rows
            for (other, _) in inactive_rows(data):
                if other['id'] == row['id'] and other.get('picking') and (refresh(service, {'preprocessing': other['picking']})['state'] in ACTIVE):
                    raise InputError(row['name'] + ': wait for active picking before overwriting its inputs.')
        for assembly in data.get('particle_assemblies', []):
            if kind in ('geometry_refinement',):
                continue
            if any((member['series_id'] == row['id'] for member in assembly.get('tomograms', []))):
                if kind == 'picking' and (not any((member.get('picking_work') == str(_work_path(root, row, kind)) for member in assembly.get('tomograms', [])))):
                    continue
                from .particle_jobs import update
                if any((update(service, assembly, kind=k)['state'] in ACTIVE for k in PARTICLE_KINDS)):
                    raise InputError(row['name'] + ': wait for particle assembly extraction/reconstruction before overwriting its inputs.')
                if any((update(service, {j['kind']: j}, kind=j['kind'])['state'] in ACTIVE for j in assembly.get('job_history', []) if j.get('kind'))):
                    raise InputError(row['name'] + ': wait for earlier assembly jobs before overwriting their inputs.')
        if kind in ('preprocessing', 'reconstruction', 'ctf_redo'):
            for step in row.get('processing_steps', []):
                if step.get('job') and refresh(service, {'preprocessing': step['job']})['state'] in ACTIVE:
                    raise InputError(row['name'] + ': wait for active pipeline refinement before overwriting its inputs.')
        for task in (*KINDS, 'handedness_test', 'ctf_redo'):
            if task in allowed_dependencies:
                continue
            if row.get(task) and refresh(service, {'preprocessing': row[task]})['state'] in ACTIVE:
                raise InputError(row['name'] + ': wait for active ' + task + ' before overwriting a job.')
    path = _work_path(root, row, kind, queue_only)
    return _reset_work(service, root, path,kind)

def store_job(row,kind,job,queue_only):
    if queue_only:row.setdefault('queued_jobs',{})[kind]=job
    else:
        row.get('queued_jobs',{}).pop(kind,None)
        row.setdefault(kind+'_history',[]).append(dict(job))
        row[kind]=job
        if kind=='picking':row.pop('pick_count',None)

def _remove_submitted_staging(service,root,row,kind,source):
    """Remove only the exact staging directory, never the submitted work."""
    expected=_work_path(root,row,kind,queued=True)
    if source!=expected or source.is_symlink() or source.parent.is_symlink():
        raise InputError('Refusing to remove an invalid queued directory.')
    if service.path(str(source),exists=False)!=expected:
        raise InputError('Refusing to remove an invalid queued directory.')
    if source.exists():shutil.rmtree(source)

def clean(service,project_id):
    with service.lock:
        root=service._root(project_id);data=service._read(root);count=0
        for assembly in data.get('pick_assemblies',[]):
            if assembly['id']==data.get('active_pick_assembly'):continue
            for saved in assembly.get('series',{}).values():
                if saved.pop('queued_picking',None):count+=1
        for record in data.get('reconstruction_assemblies',[]):
            if record['id']==data.get('active_reconstruction_assembly'):continue
            for saved in record.get('series',{}).values():
                if saved.pop('queued_reconstruction',None):count+=1
        for row in data['series']:
            count+=len(row.pop('queued_jobs',{}))
        for record in data.get('particle_assemblies',[]):
            for job in [*record.get('job_history',[]),*[record.get(k,{}) for k in PARTICLE_KINDS]]:
                if job.get('state')=='queued':job['state']='discarded';count+=1
        service._write(root,data)
        result=service._view(project_id);result['cleared_count']=count
        return result

def dispatch(job,queue_only,runner,*,dependencies=()):
    if not isinstance(queue_only,bool):raise InputError('Invalid queue option.')
    if queue_only:
        job['state']='queued'
        return
    command=['sbatch','--parsable']
    if dependencies:
        from .slurm_arrays import valid_job_id
        if any(not valid_job_id(identifier) for identifier in dependencies):raise InputError('Invalid Slurm dependency.')
        command.append('--dependency=afterok:'+':'.join(str(identifier) for identifier in dependencies))
        job['dependencies']=list(dependencies)
    command.append(str(storage_path(Path(job['work']), 'job.sbatch')))
    response=runner(command,job.get('host',''))
    identifier=response.split(';')[0].strip()
    if not identifier.isdigit():raise InputError('Unexpected Slurm response: '+response)
    job.update(state='submitted',job_id=identifier)

def submit_all(service,project_id):
    from .preprocessing import command_result
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        errors=[];submitted=0
        from .pick_assemblies import inactive_rows,save_proxy
        from . import reconstruction_assemblies as reconstructions
        reconstructions.capture(data)
        contexts=[(row,None,tuple(k for k in KINDS if k!='picking'),None) for row in data['series']]
        contexts.extend((row,saved,('reconstruction',),reconstructions.save_proxy) for row,saved in reconstructions.inactive_rows(data))
        contexts.extend((row,None,('picking',),None) for row in data['series'])
        contexts.extend((row,saved,('picking',),save_proxy) for row,saved in inactive_rows(data))
        for row,saved,kinds,save_context in contexts:
            if saved is not None:
                parent=next(r for r in data['series'] if r['id']==row['id'])
                for key in ('preprocessing','ctf_redo','handedness_test'):
                    row.pop(key,None)
                    if key in parent:row[key]=parent[key]
            for kind in kinds:
                pending=row.get('queued_jobs',{})
                job=pending.get(kind) or row.get(kind,{})
                if job.get('state')!='queued':continue
                promoted=False
                try:
                    source=service.path(job['work'])
                    if not layout_owned(root,source):raise InputError('Invalid queued job path.')
                    service.path(str(storage_path(source, 'job.sbatch')))
                    # Check staged files before replacing existing results.
                    config=json.loads((storage_path(source, 'config.json')).read_text()) if kind in pending else None
                    script=(storage_path(source, 'job.sbatch')).read_text()
                    dependencies=[];allowed=[]
                    if config and config.get('reconstruction_dependency'):
                        from .preprocessing import refresh
                        reconstruction_id=config.get('reconstruction_assembly_id')
                        if not reconstruction_id:
                            picker=next((p for p in data.get('pick_assemblies',[]) if p['id']==row.get('pick_assembly_id')),{})
                            reconstruction_id=picker.get('reconstruction_assembly_id',data.get('active_reconstruction_assembly'))
                        parent_row=reconstructions.source_row(data,row,reconstruction_id)
                        parent_job=refresh(service,{'preprocessing':parent_row.get('reconstruction',{})})
                        if parent_job['state'] not in ('completed','submitted','running'):
                            raise InputError('Reconstruction must be completed or successfully submitted before picking.')
                        parent=service.path(parent_job['work'])
                        if not layout_owned(root,parent):raise InputError('Invalid reconstruction dependency path.')
                        config['reconstruction_work']=str(parent)
                        if parent_job['state']!='completed':
                            if not parent_job.get('job_id') or parent_job.get('host','')!=job.get('host',''):
                                raise InputError('Reconstruction dependency needs a Slurm ID on the same submission host.')
                            dependencies=[parent_job['job_id']];allowed=['reconstruction']
                            if row.get('preprocessing',{}).get('state') in ACTIVE:allowed.append('preprocessing')
                    if config and config.get('preprocessing_dependency'):
                        from .preprocessing import refresh
                        prep=refresh(service,row)
                        if prep['state'] not in ('completed','submitted','running'):
                            raise InputError('Preprocessing must be completed or successfully submitted before reconstruction.')
                        parent=service.path(prep['work'])
                        if not layout_owned(root,parent):raise InputError('Invalid preprocessing dependency path.')
                        binning=prep.get('output_binning',prep.get('options',{}).get('output_binning',8))
                        output=storage_path(parent, 'output')
                        config['inputs']=dict(stack=str(storage_path(output, f'ctf/reliability_bin{binning}.mrcs')),
                            rawtlt=str(storage_path(output, f'greco_raw_bin{binning}.rawtlt')),
                            frozen_shifts=str(storage_path(output, f'alignment/tilts.star')),binning=binning)
                        if prep['state']!='completed':
                            if not prep.get('job_id') or prep.get('host','')!=job.get('host',''):
                                raise InputError('Preprocessing dependency needs a Slurm ID on the same submission host.')
                            dependencies=[prep['job_id']];allowed=['preprocessing']
                        else:
                            for key in ('stack','rawtlt','frozen_shifts'):service.path(config['inputs'][key])
                    if kind in pending:
                        target=prepare_work(service,root,row,kind,allowed_dependencies=allowed)
                        config_text=json.dumps(config).replace(str(source),str(target))
                        (storage_path(target, 'config.json')).write_text(config_text)
                        (storage_path(target, 'job.sbatch')).write_text(script.replace(str(source),str(target)))
                        code_record=source/'provenance/code.json'
                        if code_record.is_file():
                            (target/'provenance').mkdir(exist_ok=True)
                            shutil.copy2(code_record,target/'provenance/code.json')
                        job=dict(job,work=str(target));pending.pop(kind)
                        promoted=True
                    job['state']='submitting'
                    row[kind]=job
                    if kind=='picking':row.pop('pick_count',None)
                    if saved is not None:save_context(row,saved)
                    service._write(root,data)
                    dispatch(job,False,command_result,dependencies=dependencies)
                    submitted+=1
                except (InputError,OSError,ValueError) as error:
                    errors.append(row['name']+' · '+kind+': '+str(error))
                    if not promoted and kind in pending:continue
                    job.update(state='failed',error=str(error))
                if promoted:row.setdefault(kind+'_history',[]).append(dict(job))
                elif row.get(kind+'_history'):row[kind+'_history'][-1].update(job)
                if saved is not None:save_context(row,saved)
                service._write(root,data)
                # Persist the accepted Slurm ID first. Cleanup failure must never
                # mark a successfully submitted job as failed or resubmit it.
                if promoted and job.get('state')=='submitted' and job.get('job_id'):
                    try:_remove_submitted_staging(service,root,row,kind,source)
                    except (InputError,OSError) as error:
                        errors.append(row['name']+' · '+kind+': submitted, but queued directory cleanup failed: '+str(error))
        from .particle_worker import check_snapshot
        for record in data.get('particle_assemblies',[]):
            jobs={}
            for kind in PARTICLE_KINDS:
                if record.get(kind):jobs[record[kind]['work']]=record[kind]
            for job in record.get('job_history',[]):jobs.setdefault(job.get('work'),job)
            active=[j for j in jobs.values() if j.get('state') in ACTIVE and j.get('job_id')]
            for job in sorted(jobs.values(),key=lambda j:int(j.get('number',0))):
                if job.get('state')!='queued':continue
                try:
                    work=service.path(job['work']);config=json.loads((storage_path(work, 'config.json')).read_text())
                    from .assemblies import assembly_directory
                    if not work.is_relative_to(assembly_directory(root,record)) or config.get('assembly_id')!=record['id']:raise InputError('Invalid queued assembly job.')
                    check_snapshot(config.get('snapshot',[]))
                    from .particle_jobs import source_dependencies
                    dependencies=source_dependencies(service,record,config,job.get('host',''))
                    job['state']='submitting';service._write(root,data)
                    dispatch(job,False,command_result,dependencies=dependencies)
                    active=[job];submitted+=1
                except (InputError,OSError,ValueError,RuntimeError) as error:
                    job.update(state='failed',error=str(error));errors.append(record['name']+': '+str(error))
                    service._write(root,data)
                    break
                service._write(root,data)
        result=service._view(project_id)
        result.update(submission_errors=errors,submitted_count=submitted)
        return result
