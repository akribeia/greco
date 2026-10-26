"""Read an owned job's original settings without copying outputs or snapshots."""
from greco.project_layout import join as storage_path
import json
import re
from pathlib import Path
from .assemblies import PARTICLE_KINDS,restore_job_history
from .projects import InputError
from .particle_jobs import volume_job,assembly


def preset(service, project_id, assembly_id, name):
    if not isinstance(name, str) or not re.fullmatch(r'[0-9]{3,}', name):
        raise InputError('Invalid particle job name.')
    with service.lock:
        root=service._root(project_id);record=assembly(service._read(root),assembly_id)
        restore_job_history(root,record)
        candidates=[(kind,record[kind]) for kind in PARTICLE_KINDS if record.get(kind)]
        candidates.extend((job.get('kind'),job) for job in record.get('job_history',[]))
        kinds={kind for kind,job in candidates if kind in PARTICLE_KINDS and Path(job.get('work','')).name==name}
        if len(kinds)!=1:raise InputError('Particle job not found or ambiguous.')
        kind=kinds.pop()
        job = volume_job(service, project_id, assembly_id, kind, name)
        path = storage_path(Path(job['work']), 'config.json')
        if path.is_symlink():raise InputError('Invalid job configuration.')
        try:config = json.loads(path.read_text())
        except (OSError, ValueError) as error:raise InputError('Cannot read original job settings.') from error
        if config.get('assembly_id') != assembly_id:raise InputError('Invalid job configuration ownership.')
        if config.get('action')=='sta_after_dependencies':
            options=dict(config['deferred_options'])
            if kind=='refinement3d':
                if options.get('symmetry','C1')!='C1':
                    raise InputError('This job requested particle symmetry, which is no longer supported.')
                options.pop('symmetry',None)
            return dict(kind=kind,options=options,name=name)
        options = dict(config.get('options') or {}) if kind != 'extraction' else {}
        source = config.get('coordinate_source' if kind == 'extraction' else 'particle_source', {})
        if kind == 'extraction':
            options['binning'] = config.get('assembly_sampling', {}).get('binning', 8)
            if source.get('kind') == 'classification':options['classes'] = list(source.get('classes', []))
        if kind=='refinement3d':
            if options.get('symmetry','C1')!='C1':
                raise InputError('This job used particle symmetry, which is no longer supported. Create a new asymmetric refinement.')
            options.pop('symmetry',None)
        if source.get('work'):
            options['from'] = source['kind'] + ':' + source['work']
        elif kind == 'extraction':options['from'] = 'assembly:' + assembly_id
        else:raise InputError('Original job source is missing; cannot clone safely.')
        if kind in ('polishing', 'ctf_refinement'):
            if not config.get('reference_job'):raise InputError('Original reference job is missing.')
            options['reference_from'] = 'refinement3d:' + config['reference_job']
        return dict(kind=kind, options=options, name=name)
