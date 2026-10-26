"""Delete a single owned STA job, with source and active-job protection."""
from greco.project_layout import join as storage_path
import json
import re
import shutil
import getpass

from .projects import InputError
from .assemblies import assembly_directory,restore_job_history,PARTICLE_KINDS
from .particle_jobs import assembly, ACTIVE, config_uses_work
from .preprocessing import command_result

KINDS=PARTICLE_KINDS


def delete(service,project_id,assembly_id,name,confirmed=False):
    if confirmed is not True:raise InputError('Confirm permanent job deletion.')
    if not isinstance(name,str) or not re.fullmatch(r'[0-9]{3,}',name):
        raise InputError('Invalid particle job name.')
    with service.lock:
        root=service._root(project_id);data=service._read(root);record=assembly(data,assembly_id)
        restore_job_history(root,record)
        parent=assembly_directory(root,record)
        candidates=list(parent.glob('*/'+name))
        if len(candidates)!=1:raise InputError('Particle job not found or ambiguous.')
        target=candidates[0]
        if parent.is_symlink() or target.is_symlink() or service.path(str(target))!=target or not target.is_dir():
            raise InputError('Invalid owned particle job directory.')
        jobs={}
        for kind in KINDS:
            if record.get(kind):jobs[record[kind]['work']]=(kind,record[kind])
        for job in record.get('job_history',[]):
            if job.get('work') and job.get('kind'):jobs.setdefault(job['work'],(job['kind'],job))
        if str(target) not in jobs:raise InputError('Particle job not found.')
        kind,job=jobs[str(target)]
        for other in parent.glob('*/*'):
            if other==target or not other.is_dir() or other.is_symlink():continue
            if not other.name.isdigit():continue
            try:config=json.loads((storage_path(other, 'config.json')).read_text())
            except (OSError,ValueError) as error:raise InputError('Cannot verify other job dependencies; deletion blocked.') from error
            if config_uses_work(config,target):
                raise InputError(other.name+' uses this job. Delete dependent jobs first.')
        # Check dependencies before scancel: deleting an input must not silently
        # cancel it underneath a dependent job. Never trust an output completion
        # marker alone: the Slurm allocation may still be cleaning up.
        identifier=job.get('job_id');host=job.get('host','')
        if identifier:
            identifier=str(identifier)
            if not re.fullmatch(r'\d+(?:_\d+)?',identifier):raise InputError('Invalid Slurm job ID; deletion blocked.')
            user=getpass.getuser()
            live=command_result(['squeue','-h','-j',identifier,'-o','%u|%T'],host).strip()
            if live:
                if any(line.split('|')[0].strip()!=user for line in live.splitlines()):
                    raise InputError('Slurm job owner does not match the server user; deletion blocked.')
                if not job.get('delete_requested'):
                    command_result(['scancel','--user',user,identifier],host)
                    job['delete_requested']=True
                    service._write(root,data)
                # Return without touching files while RUNNING/COMPLETING, then
                # the browser retries. No long wait under the project lock.
                live=command_result(['squeue','-h','-j',identifier,'-o','%u|%T'],host).strip()
                if live:
                    result=service._view(project_id)
                    result['deletion_pending']=True
                    return result
        elif job.get('state') in ACTIVE:
            raise InputError('The active job has no Slurm ID; cancellation cannot be verified.')
        # Explicit target only: never delete an assembly or any source data.
        shutil.rmtree(target)
        record['job_history']=[j for j in record.get('job_history',[]) if j.get('work')!=str(target)]
        for task in KINDS:
            if record.get(task,{}).get('work')==str(target):record.pop(task,None)
            if task not in record:
                older=[j for j in record.get('job_history',[]) if j.get('kind')==task]
                if older:
                    latest=max(older,key=lambda j:int(j.get('number',0)))
                    record[task]=dict(latest)
                    record['job_history']=[j for j in record['job_history'] if j.get('work')!=latest['work']]
        record['extracted']=record.get('extraction',{}).get('state')=='completed'
        service._write(root,data)
        return service._view(project_id)
