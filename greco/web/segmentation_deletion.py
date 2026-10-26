"""Delete an owned segmentation workspace after dependencies and Slurm checks."""
import getpass
import json
from pathlib import Path
import re
import shutil

from greco.project_layout import config_paths, record_directory
from .particle_jobs import config_uses_work
from .pick_assemblies import _live_job_owner_state
from .preprocessing import command_result
from .projects import InputError
from .queue import ACTIVE
from .segmentation import find


def delete(service, project_id, identifier, confirmed=False):
    if confirmed is not True:
        raise InputError('Confirm permanent segmentation deletion.')
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        record = find(data, identifier)
        target = record_directory(root, record, 'segmentation')
        if (target.parent != root / 'segmentation' or target.is_symlink()
                or service.path(str(target), exists=False) != target):
            raise InputError('Invalid owned segmentation directory.')
        if target.exists() and not target.is_dir():
            raise InputError('Segmentation output is not a directory.')
        for other in data.get('segmentations', []):
            if other['id'] == identifier:
                continue
            if record_directory(root, other, 'segmentation') == target:
                raise InputError('Another segmentation shares this directory.')
            if config_uses_work(other, target):
                raise InputError(other['name'] + ' uses this segmentation. Delete its dependent jobs first.')
        for path in config_paths(root):
            if path.is_relative_to(target):
                continue
            try:
                config = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise InputError('Cannot verify job dependencies; deletion blocked.') from error
            if not isinstance(config, dict):
                raise InputError('Cannot verify job dependencies; deletion blocked.')
            inputs = {k: v for k, v in config.items()
                      if k not in ('series', 'project_snapshot', 'source_result')}
            if config_uses_work(inputs, target):
                raise InputError('Another job uses this segmentation. Delete its dependent jobs first.')
        jobs = []
        for member in record['tomograms']:
            jobs.extend([*member['steps'], *member.get('cleared_steps', [])])
        user = getpass.getuser()
        live_jobs = {}
        for job in jobs:
            work = job.get('work')
            if work:
                path = Path(work)
                if not path.is_relative_to(target) or service.path(work, exists=False) != path:
                    raise InputError('Invalid owned segmentation job directory.')
            elif job.get('job_id') or job.get('state') in ACTIVE:
                raise InputError('Cannot verify the segmentation job directory.')
            jid = job.get('job_id')
            host = job.get('host', '')
            if not jid:
                if job.get('state') in ACTIVE:
                    raise InputError('Active segmentation has no Slurm ID; cancellation cannot be verified.')
                continue
            jid = str(jid)
            if not re.fullmatch(r'\d+(?:_\d+)?', jid):
                raise InputError('Invalid Slurm job ID.')
            if (host, jid) in live_jobs:
                continue
            live = _live_job_owner_state(jid, host)
            if live and any(line.split('|')[0].strip() != user for line in live.splitlines()):
                raise InputError('Slurm job owner does not match the server user.')
            live_jobs[host, jid] = bool(live)
        requested = record.setdefault('delete_requested_jobs', [])
        for (host, jid), live in live_jobs.items():
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
            for (host, jid), live in live_jobs.items():
                if not live:
                    continue
                remaining = _live_job_owner_state(jid, host)
                if remaining and any(line.split('|')[0].strip() != user for line in remaining.splitlines()):
                    raise InputError('Slurm job owner does not match the server user.')
                pending = bool(remaining) or pending
            if pending:
                result = service._view(project_id)
                result['deletion_pending'] = True
                return result
        # rmtree unlinks internal symlinks without following their shared inputs.
        # Keep the manifest entry if cleanup fails, so deletion can be retried.
        if target.exists():
            shutil.rmtree(target)
        data['segmentations'] = [r for r in data['segmentations'] if r['id'] != identifier]
        service._write(root, data)
        return service._view(project_id)
