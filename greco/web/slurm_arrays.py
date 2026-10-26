"""Stable project indices and real Slurm arrays with per-series outputs."""
from greco.project_layout import join as storage_path
import json
import re
import shlex
from pathlib import Path
from .projects import InputError

def valid_job_id(value):
    return bool(re.fullmatch('[1-9]\\d*(?:_\\d+)?', str(value)))

def ensure_indices(data):
    used=set();changed=False
    for row in data['series']:
        index=row.get('slurm_array_index')
        if index is None:continue
        if type(index) is not int or index<1 or index in used:raise InputError('Invalid project Slurm array indices.')
        used.add(index)
    for position,row in enumerate(data['series'],1):
        if row.get('slurm_array_index') is not None:continue
        index=position if position not in used else max(used,default=0)+1
        row['slurm_array_index']=index;used.add(index);changed=True
    return changed

def array_expression(indices):
    values=sorted(indices)
    if not values or len(set(values))!=len(values) or any(type(i) is not int or i<1 for i in values):
        raise InputError('Invalid Slurm array task indices.')
    ranges=[];start=previous=values[0]
    for index in values[1:]+[None]:
        if index is not None and index==previous+1:previous=index;continue
        ranges.append(str(start) if start==previous else f'{start}-{previous}')
        start=previous=index
    return ','.join(ranges)

def dependency_signature(dependencies,index):
    corr=[];ok=[]
    for identifier in dependencies:
        if not valid_job_id(identifier):raise InputError('Invalid Slurm dependency.')
        parts=str(identifier).split('_')
        if len(parts)==2 and int(parts[1])==index:corr.append(parts[0])
        else:ok.append(str(identifier))
    clauses=[]
    if corr:clauses.append('aftercorr:'+':'.join(sorted(set(corr))))
    if ok:clauses.append('afterok:'+':'.join(sorted(set(ok))))
    return ','.join(clauses)

def _header(script):
    return [line for line in script.splitlines() if line.startswith('#SBATCH')]

def _resource_signature(script, work):
    """Ignore task labels while retaining all scheduling directives."""
    headers = []
    for line in _header(script):
        try:
            options = shlex.split(line[len('#SBATCH'):], comments=True)
        except ValueError as error:
            raise InputError('Invalid Slurm directive.') from error
        retained = []
        index = 0
        while index < len(options):
            option = options[index]
            if option in ('--job-name', '-J'):
                if index + 1 >= len(options):
                    raise InputError('Missing Slurm job name.')
                index += 2
            elif option.startswith('--job-name=') or option.startswith('-J'):
                index += 1
            else:
                retained.append(option.replace(str(work), '{{TASK_WORK}}'))
                index += 1
        if retained:
            headers.append(tuple(retained))
    return tuple(headers)

def live_states_for(jobs,runner):
    """Expand pending array task IDs in one scheduler query per host."""
    hosts={};states={}
    for job in jobs:
        if job.get('job_id') and job.get('state') not in ('completed','failed','queued'):
            hosts.setdefault(job.get('host',''),set()).add(str(job['job_id']))
    try:
        for host,identifiers in hosts.items():
            command=['squeue','-h','-j',','.join(sorted(identifiers)),'-o','%i|%T','--array']
            for line in runner(command,host).splitlines():
                identifier,separator,state=line.strip().partition('|')
                if separator and identifier in identifiers:states[(host,identifier)]=state.strip()
    except InputError:return None
    return states

def submit_arrays(service, root, data, kind, entries, runner):
    """entries=(row,job,dependencies); persist every accepted array immediately.

    Split by resource headers and dependency family: ready rows do not wait for
    unrelated unfinished parent jobs. No array concurrency cap is introduced.
    """
    from .queue import store_job
    ensure_indices(data)
    groups = {}
    errors = []
    for (row, job, dependencies) in entries:
        index = row['slurm_array_index']
        work = Path(job['work'])
        script = storage_path(work, 'job.sbatch').read_text()
        dependency = dependency_signature(dependencies, index)
        resources = _resource_signature(script, work)
        key = (job.get('host', ''), dependency, resources)
        job.update(state='submitting', dependencies=list(dependencies))
        store_job(row, kind, job, False)
        groups.setdefault(key, []).append((row, job, script))
    service._write(root, data)
    for ((host, dependency, _), group) in groups.items():
        container = root / '.slurm' / 'arrays'
        if container.is_symlink():
            raise InputError('Invalid Slurm array directory.')
        number = 1 + max([int(p.name.split('_')[0]) for p in container.glob('*') if p.name.split('_')[0].isdigit()] + [0])
        batch = container / (f'{number:06d}_' + kind)
        batch.mkdir(parents=True)
        indices = [row['slurm_array_index'] for (row, _, _) in group]
        expression = array_expression(indices)
        lines = ['#!/usr/bin/env bash', *_header(group[0][2]), 'set -euo pipefail', 'case "${SLURM_ARRAY_TASK_ID:?Missing array task ID}" in']
        mapping = {}
        for (row, job, script) in group:
            work = Path(job['work'])
            index = row['slurm_array_index']
            first = script.splitlines()[0] if script else ''
            interpreter = shlex.split(first[2:]) if first.startswith('#!') else ['bash']
            lines += [f'{index})', f"exec {shlex.join([*interpreter, str(storage_path(work, 'job.sbatch'))])} > {shlex.quote(str(storage_path(work, 'slurm.out')))} 2> {shlex.quote(str(storage_path(work, 'slurm.err')))}", ';;']
            mapping[str(index)] = dict(series_id=row['id'], name=row['name'], work=str(work), dependencies=job['dependencies'])
        lines += ['*) echo "Unknown array task ID" >&2; exit 64 ;;', 'esac', '']
        storage_path(batch, 'job.sbatch').write_text('\n'.join(lines))
        storage_path(batch, 'tasks.json').write_text(json.dumps(mapping, indent=2))
        command = ['sbatch', '--parsable', '--array=' + expression, '--job-name=greco_' + kind, '--output=' + str(storage_path(batch, 'slurm-%A_%a.out')), '--error=' + str(storage_path(batch, 'slurm-%A_%a.err'))]
        if dependency:
            command.append('--dependency=' + dependency)
        command.append(str(storage_path(batch, 'job.sbatch')))
        try:
            response = runner(command, host)
            base = response.split(';')[0].strip()
            if not re.fullmatch('[1-9]\\d*', base):
                raise InputError('Unexpected Slurm array response: ' + response)
            for (row, job, _) in group:
                index = row['slurm_array_index']
                job.update(state='submitted', job_id=f'{base}_{index}', array_job_id=base, array_task_id=index, array_indices=expression, array_work=str(batch), dependency_expression=dependency)
        except InputError as error:
            for (row, job, _) in group:
                job.update(state='failed', error=str(error))
                errors.append(row['name'] + ': ' + str(error))
        for (row, job, _) in group:
            row[kind + '_history'][-1].update(job)
        service._write(root, data)
    return errors
