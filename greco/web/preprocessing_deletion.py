"""Delete only an owned preprocessing job after Slurm has stopped."""
from greco.project_layout import config_paths
import getpass
import json
import re
import shutil
from pathlib import Path
from .metadata_cache import read_json
from .projects import InputError
from .queue import _work_path, ACTIVE
from .pick_assemblies import _live_job_owner_state
from .preprocessing import command_result
from .particle_jobs import config_uses_work

def delete(service, project_id, series_id, work, confirmed=False, *, kind='preprocessing', step_id=None, unsubmitted=False):
    if kind not in ('preprocessing', 'reconstruction', 'geometry_refinement', 'manual_curation'):
        raise InputError('Invalid processing job kind.')
    if confirmed is not True:
        raise InputError('Confirm permanent job deletion.')
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        row = next((r for r in data['series'] if r['id'] == series_id), None)
        if row is None:
            raise InputError('Unknown tilt series.')
        job = row.get(kind, {})
        target = None if kind == 'manual_curation' else _work_path(root, row, kind)
        step = None
        if step_id:
            from .reconstruction_steps import find, job_for, work_for
            step = find(row, step_id)
            if kind not in ('geometry_refinement', 'manual_curation') or step['kind'] != kind:
                raise InputError('Invalid step deletion kind.')
            job = job_for(row, step)
            target = work_for(root, series_id, step)
        owned = work == str(target)
        if kind == 'manual_curation' and target and work:
            from .manual_refinement import valid_work
            owned = valid_work(root, target, Path(work))
        if unsubmitted and (step is None or job):
            raise InputError('Processing job changed; reload before deleting.')
        if not work or (not unsubmitted and work != job.get('work')) or (not owned):
            raise InputError('Processing job changed; reload before deleting.')
        if target.is_symlink() or target.parent.is_symlink() or service.path(str(target), exists=False) != target:
            raise InputError('Invalid owned processing directory.')
        targets = {target}
        for candidate in targets:
            if candidate.is_symlink() or candidate.parent.is_symlink() or service.path(str(candidate), exists=False) != candidate:
                raise InputError('Invalid owned processing directory.')
        if unsubmitted:
            commands=command_result(['squeue','-h','-u',getpass.getuser(),'-o','%o'],data.get('slurm',{}).get('submission_host',''))
            if any(Path(command.strip()).is_relative_to(target) for command in commands.splitlines() if command.strip()):
                raise InputError('An active Slurm job uses this directory; wait for it to finish before deleting.')
        for config_path in config_paths(root):
            if any((config_path.is_relative_to(t) for t in targets)):
                if not (kind == 'reconstruction' and config_path.is_relative_to(target / 'refinement')):
                    continue
            try:
                config = read_json(service,config_path)
            except (OSError, ValueError) as error:
                raise InputError('Cannot verify job dependencies; deletion blocked.') from error
            inputs = {k: v for (k, v) in config.items() if k not in ('series', 'project_snapshot', 'source_result')}
            if any((config_uses_work(inputs, t) for t in targets)):
                dependency = str(config_path.parent.relative_to(root))
                raise InputError('A downstream job uses this job: ' + dependency + '. Delete dependent jobs first.')
        jobs = [job]
        if kind == 'preprocessing':
            for key in ('preprocessing', 'ctf_redo', 'handedness_test'):
                jobs.extend([row.get(key, {})] + row.get(key + '_history', []))
        selected = {}
        for record in jobs:
            if not record.get('work') or not any((Path(record['work']) == t or (kind == 'preprocessing' and Path(record['work']).is_relative_to(t)) for t in targets)):
                continue
            identifier = record.get('job_id')
            host = record.get('host', '')
            if identifier:
                identifier = str(identifier)
                if not re.fullmatch('\\d+(?:_\\d+)?', identifier):
                    raise InputError('Invalid Slurm job ID; deletion blocked.')
                selected[host, identifier] = record
            elif record.get('state') in ACTIVE:
                raise InputError('The active job has no Slurm ID; cancellation cannot be verified.')
        user = getpass.getuser()
        live_jobs = []
        for ((host, identifier), record) in selected.items():
            live = _live_job_owner_state(identifier, host)
            if live:
                if any((line.split('|')[0].strip() != user for line in live.splitlines())):
                    raise InputError('Slurm job owner does not match the server user.')
                live_jobs.append((host, identifier, record))
        pending = False
        for (host, identifier, record) in live_jobs:
            if not record.get('delete_requested'):
                try:
                    command_result(['scancel', '--user', user, identifier], host)
                except InputError as error:
                    if 'Invalid job id specified' not in str(error) or _live_job_owner_state(identifier, host):
                        raise
                record['delete_requested'] = True
            if _live_job_owner_state(identifier, host):
                pending = True
        if pending:
            service._write(root, data)
            result = service._view(project_id)
            result['deletion_pending'] = True
            return result
        for candidate in sorted(targets, key=lambda p: p.name == 'preprocessing'):
            if candidate.exists():
                shutil.rmtree(candidate)
        if kind == 'preprocessing':
            for key in ('preprocessing', 'ctf_redo', 'handedness_test'):
                row.pop(key, None)
                row[key + '_history'] = [r for r in row.get(key + '_history', []) if r.get('work') and (not any((Path(r['work']) == t or Path(r['work']).is_relative_to(t) for t in targets)))]
        if step is not None:
            row['processing_steps'].remove(step)
            if step_id == 'geometry':
                row.pop(kind, None)
            if row.get('picking_step_id') == step_id or (step_id == 'geometry' and row.get('picking_source') == 'geometry_refinement' and (not row.get('picking_step_id'))):
                row.update(picking_source=None, picking_step_id=None)
        else:
            row.pop(kind, None)
            if kind == 'preprocessing' and row.get('ctf_redo', {}).get('work') == work:
                row.pop('ctf_redo', None)
        row[kind + '_history'] = [j for j in row.get(kind + '_history', []) if j.get('work') != work]
        service._write(root, data)
        return service._view(project_id)
