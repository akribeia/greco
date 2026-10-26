"""Persistent picking experiments; selected results remain compatible with STA."""
from greco.project_layout import record_directory
from copy import deepcopy
import uuid
import re
import shutil
import getpass

from pathlib import Path
from .projects import InputError
from .preprocessing import command_result
ASSETS=('picker_template','picker_mask','picking_options')
FIELDS=('picking','picking_history','pick_count')

def _live_job_owner_state(identifier,host):
    """A purged Slurm job is absent, not a failed deletion permission check."""
    try:
        return command_result(['squeue','-h','-j',identifier,'-o','%u|%T'],host).strip()
    except InputError as error:
        if 'slurm_load_jobs error: Invalid job id specified' in str(error):return ''
        raise

def current(data):
    return next((a for a in data.get('pick_assemblies',[]) if a['id']==data.get('active_pick_assembly')),None)

def capture(data):
    assembly=current(data)
    if assembly is None:return
    for key in ASSETS:
        if key in data:assembly[key]=deepcopy(data[key])
        else:assembly.pop(key,None)
    for row in data['series']:
        saved=assembly.setdefault('series',{}).setdefault(row['id'],{})
        for key in FIELDS:
            if key in row:saved[key]=deepcopy(row[key])
            else:saved.pop(key,None)
        queued=row.get('queued_jobs',{}).get('picking')
        if queued:saved['queued_picking']=deepcopy(queued)
        else:saved.pop('queued_picking',None)

def activate(data, identifier):
    capture(data)
    assembly = next((a for a in data.get('pick_assemblies', []) if a['id'] == identifier), None)
    if assembly is None:
        raise InputError('Unknown pick assembly.')
    data['active_pick_assembly'] = identifier
    for key in ASSETS:
        data.pop(key, None)
        if key in assembly:
            data[key] = deepcopy(assembly[key])
    for row in data['series']:
        saved = assembly.get('series', {}).get(row['id'], {})
        for key in FIELDS:
            row.pop(key, None)
            if key in saved:
                row[key] = deepcopy(saved[key])
        row.get('queued_jobs', {}).pop('picking', None)
        if saved.get('queued_picking'):
            row.setdefault('queued_jobs', {})['picking'] = deepcopy(saved['queued_picking'])
        row['pick_assembly_id'] = identifier


def create(service, project_id, name=None):
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        capture(data)
        assemblies = data.setdefault('pick_assemblies', [])
        identifier = uuid.uuid4().hex[:8]
        while any((a['id'] == identifier for a in assemblies)):
            identifier = uuid.uuid4().hex[:8]
        name = (name or f'Pick assembly {len(assemblies) + 1:03d}').strip()
        if not name or len(name) > 100:
            raise InputError('Enter a pick assembly name (1–100 characters).')
        assemblies.append(dict(id=identifier, name=name, series={}))
        activate(data, identifier)
        service._write(root, data)
        return service._view(project_id)

def select(service,project_id,identifier):
    with service.lock:
        root=service._root(project_id);data=service._read(root);activate(data,identifier)
        service._write(root,data);return service._view(project_id)

def delete(service, project_id, identifier, confirmed=False):
    """Remove owned picking outputs after checking dependants and Slurm."""
    if confirmed is not True:
        raise InputError('Confirm permanent pick assembly deletion.')
    if not isinstance(identifier, str) or not re.fullmatch('[a-f0-9]{8}', identifier):
        raise InputError('Invalid pick assembly ID.')
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        capture(data)
        record = next((a for a in data.get('pick_assemblies', []) if a['id'] == identifier), None)
        if record is None:
            raise InputError('Pick assembly not found.')
        from .queue import _work_path, ACTIVE
        targets = set()
        jobs = []
        rows = {r['id']: r for r in data['series']}
        for (series_id, saved) in record.get('series', {}).items():
            if series_id not in rows:
                raise InputError('Cannot verify the pick assembly tomogram identity.')
            row = dict(rows[series_id], pick_assembly_id=identifier)
            expected = {_work_path(root, row, 'picking', q) for q in (False, True)}
            targets.update(expected)
            for job in [saved.get('picking'), saved.get('queued_picking'), *saved.get('picking_history', [])]:
                if not job:
                    continue
                jobs.append(job)
                if not job.get('work'):
                    raise InputError('Cannot verify picking job ownership.')
                path = Path(job['work'])
                if path not in expected:raise InputError('Invalid owned picking directory.')
                targets.add(path)
        for target in targets:
            if target.is_symlink() or target.parent.is_symlink() or service.path(str(target), exists=False) != target:
                raise InputError('Refusing to delete a symlinked or invalid picking directory.')
            if target.exists() and (not target.is_dir()):
                raise InputError('Picking output is not a directory.')

        def uses_target(value):
            if not value:
                return False
            path = Path(value)
            return any((path == t or path.is_relative_to(t) for t in targets))
        for assembly in data.get('particle_assemblies', []):
            if assembly.get('pick_assembly_id') == identifier or any((uses_target(m.get('picking_work')) or uses_target(m.get('star')) for m in assembly.get('tomograms', []))):
                raise InputError(assembly['name'] + ' uses these picks. Delete the dependent particle assembly first.')
        for other in data.get('pick_assemblies', []):
            if other['id'] == identifier:
                continue
            for saved in other.get('series', {}).values():
                if any((uses_target(j.get('work')) for j in [saved.get('picking'), saved.get('queued_picking'), *saved.get('picking_history', [])] if j)):
                    raise InputError(other['name'] + ' shares these picking files; deletion blocked.')
        category = record_directory(root, record, 'picking')
        if category:
            if category.is_symlink() or service.path(str(category), exists=False) != category:
                raise InputError('Invalid picking category directory.')
            for other in data.get('pick_assemblies', []):
                if other['id'] == identifier:
                    continue
                for key in ('picker_template', 'picker_mask'):
                    value = other.get(key, {}).get('path')
                    if value and Path(value).is_relative_to(category):
                        raise InputError(other['name'] + ' uses a template or mask from this category.')
        user = getpass.getuser()
        live_jobs = {}
        for job in jobs:
            jid = job.get('job_id')
            host = job.get('host', '')
            if not jid:
                if job.get('state') in ACTIVE:
                    raise InputError('Active picking has no Slurm ID; cancellation cannot be verified.')
                continue
            jid = str(jid)
            if not re.fullmatch('\\d+(?:_\\d+)?', jid):
                raise InputError('Invalid Slurm job ID.')
            if (host, jid) in live_jobs:
                continue
            live = _live_job_owner_state(jid, host)
            if live and any((line.split('|')[0].strip() != user for line in live.splitlines())):
                raise InputError('Slurm job owner does not match the server user.')
            live_jobs[host, jid] = bool(live)
        requested = record.setdefault('delete_requested_jobs', [])
        for ((host, jid), live) in live_jobs.items():
            key = host + '|' + jid
            if live and key not in requested:
                try:
                    command_result(['scancel', '--user', user, jid], host)
                except InputError as error:
                    if 'Invalid job id specified' not in str(error) or _live_job_owner_state(jid, host):
                        raise
                requested.append(key)
        if any(live_jobs.values()):
            service._write(root, data)
            pending = False
            for ((host, jid), live) in live_jobs.items():
                if not live:
                    continue
                remaining = _live_job_owner_state(jid, host)
                if remaining and any((line.split('|')[0].strip() != user for line in remaining.splitlines())):
                    raise InputError('Slurm job owner does not match the server user.')
                pending = bool(remaining) or pending
            if pending:
                result = service._view(project_id)
                result['deletion_pending'] = True
                return result
        for target in sorted(targets, key=str):
            if target.exists():
                shutil.rmtree(target)
            if target.parent.is_dir():
                try:
                    target.parent.rmdir()
                except OSError:
                    pass
        if category and category.exists():
            shutil.rmtree(category)
        data['pick_assemblies'] = [a for a in data['pick_assemblies'] if a['id'] != identifier]
        if data.get('active_pick_assembly') == identifier:
            data.pop('active_pick_assembly', None)
            for key in ASSETS:
                data.pop(key, None)
            for row in data['series']:
                for key in (*FIELDS, 'pick_assembly_id'):
                    row.pop(key, None)
                row.get('queued_jobs', {}).pop('picking', None)
            if data['pick_assemblies']:
                activate(data, data['pick_assemblies'][0]['id'])
        service._write(root, data)
        return service._view(project_id)

def inactive_rows(data):
    """Mutable jobs for background refresh/queue dispatch in other assemblies."""
    for assembly in data.get('pick_assemblies', []):
        if assembly['id'] == data.get('active_pick_assembly'):
            continue
        for row in data['series']:
            saved = assembly.setdefault('series', {}).setdefault(row['id'], {})
            proxy = dict(row, pick_assembly_id=assembly['id'])
            for key in FIELDS:
                proxy.pop(key, None)
                if key in saved:
                    proxy[key] = saved[key]
            proxy['queued_jobs'] = {}
            if saved.get('queued_picking'):
                proxy['queued_jobs']['picking'] = saved['queued_picking']
            yield (proxy, saved)

def save_proxy(proxy,saved):
    for key in FIELDS:
        if key in proxy:saved[key]=deepcopy(proxy[key])
        else:saved.pop(key,None)
    queued=proxy.get('queued_jobs',{}).get('picking')
    if queued:saved['queued_picking']=deepcopy(queued)
    else:saved.pop('queued_picking',None)
