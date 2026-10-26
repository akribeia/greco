"""Immutable batch manual XY attempts, Slurm submission and step output lineage."""
from greco.project_layout import join as storage_path, preprocessing_work, config_paths, owned as layout_owned
import json,re,uuid
from pathlib import Path
from .projects import InputError
from .preprocessing import refresh,render_script,command_result

def effective_job(row,step):
    from .reconstruction_steps import job_for
    job=job_for(row,step)
    if job.get('state')=='completed':return job
    return next((j for j in reversed(step.get('manual_history',[])) if j.get('state')=='completed'),{})

def previous_output(row,step):
    from .reconstruction_steps import previous_geometry,job_for
    own=effective_job(row,step)
    if own:return step,own
    prior=previous_geometry(row,step)
    return (prior,job_for(row,prior)) if prior else (None,row.get('reconstruction',{}))

def valid_work(root, base, work):
    work = Path(work)
    return work == base or (work.parent == base / 'provenance/actions' and bool(re.fullmatch('[a-f0-9]{32}', work.name)))

def submit_export(service,pid,sid,step_id,source_identity,options=None):
    return submit(service,pid,sid,step_id,None,source_identity,options,action='reconstruct')

def submit(service, pid, sid, step_id, edits, source_identity, options=None, *, action='refine'):
    if action not in ('refine', 'reconstruct'):
        raise InputError('Invalid manual curation action.')
    from .final_denoising import validate_denoising_options, prepare_model_reuse
    if options is not None and (not isinstance(options, dict)):
        raise InputError('Invalid manual curation options.')
    options = dict(options or {})
    final_binning = options.pop('final_binning', 8)
    if type(final_binning) is not int or final_binning not in (1, 2, 4, 8, 16):
        raise InputError('Final binning must be 1, 2, 4, 8 or 16.')
    disabled = options.pop('disabled_tilts', [])
    if not isinstance(disabled, list) or any((type(t) is not int for t in disabled)) or len(set(disabled)) != len(disabled):
        raise InputError('Invalid disabled tilt selection.')
    include_discarded = options.pop('include_discarded', False)
    if type(include_discarded) is not bool:
        raise InputError('Discarded tilts must be a checkbox.')
    from .alignment_options import validate as validate_alignment
    smoothness = validate_alignment(options.pop('smoothness_strength', 0.0))['smoothness_strength']
    from ..manual_tilt import validate_gaussian_search
    try:
        gaussian = validate_gaussian_search(options.pop('gaussian_search_prior', False), options.pop('gaussian_sigma_px', 8.0))
    except ValueError as error:
        raise InputError(str(error)) from None
    denoising = validate_denoising_options(options)
    from .reconstruction_steps import find, work_for
    from .queue import dispatch, ACTIVE
    from ..manual_tilt import validate_edits
    from .particle_jobs import config_uses_work
    with service.lock:
        root = service._root(pid)
        data = service._read(root)
        row = next((r for r in data['series'] if r['id'] == sid), None)
        if row is None:
            raise InputError('Unknown tilt series.')
        step = find(row, step_id)
        if step['kind'] != 'manual_curation':
            raise InputError('Not a Manual Curation step.')
        latest = refresh(service, {'preprocessing': step.get('job', {})})
        if latest.get('state') in ACTIVE:
            raise InputError('Wait for the current manual refinement to finish.')
        if step.get('job'):
            step['job'] = latest
        from .manual_curation import sources, identity
        from .reconstruction_steps import curation_source
        (_, available) = sources(service, pid, sid)
        current = available.get(curation_source(row, step))
        if current is None or not isinstance(source_identity, str) or identity(current)[0] != source_identity:
            raise InputError('The input alignment changed. Reload the diagnostic and place the manual priors again.')
        base = work_for(root, sid, step)
        for path in config_paths(root):
            if path.is_relative_to(base):
                continue
            if config_uses_work(json.loads(path.read_text()), base):
                raise InputError('A downstream job already uses this curation. Add another Manual Curation step.')
        (prior, parent) = previous_output(row, step)
        parent = refresh(service, {'preprocessing': parent})
        if parent.get('state') not in ('completed', 'partial'):
            raise InputError('Complete the preceding step first.')
        parent_work = service.path(parent['work'])
        if not layout_owned(root, parent_work):
            raise InputError('Invalid manual refinement input.')
        source = json.loads(storage_path(parent_work, 'config.json').read_text())
        inputs = source['inputs']
        angles = service.path(inputs['rawtlt'])
        import numpy as np
        values = np.loadtxt(angles, ndmin=1)
        from .manual_curation import source_angles
        (_, accepted_indices) = source_angles(current)
        discarded_indices = np.array([t for t in range(len(values)) if t not in set(accepted_indices.tolist())], int)
        plane_indices = np.concatenate((accepted_indices, discarded_indices)) if include_discarded else accepted_indices
        if any((t not in set(accepted_indices.tolist()) for t in disabled)):
            raise InputError('Only accepted tilts can be deactivated.')
        try:
            edits = validate_edits(edits, len(plane_indices)) if action == 'refine' and edits else []
        except ValueError as error:
            raise InputError(str(error)) from None
        if action == 'refine' and (not edits) and (not disabled):
            raise InputError('Select tilts to refine or deactivate.')
        if action == 'reconstruct' and disabled:
            raise InputError('Submit the tilt selection with Submit fixes first.')
        if any((int(plane_indices[edit['index']]) in disabled for edit in edits)):
            raise InputError('A disabled tilt cannot be refined.')
        accepted_targets = sum((edit['index'] < len(accepted_indices) for edit in edits))
        if len(accepted_indices) - len(disabled) - accepted_targets < 3:
            raise InputError('Leave at least three fixed accepted reference tilts.')
        if any((edit['reference'] is not None and (edit['reference'] >= len(accepted_indices) or int(plane_indices[edit['reference']]) in disabled) for edit in edits)):
            raise InputError('Choose an accepted tilt as the fixed reference.')
        edits = [dict(edit, index=int(plane_indices[edit['index']]), reference=int(plane_indices[edit['reference']]) if edit['reference'] is not None else None) for edit in edits]
        if inputs.get('binning') != 8:
            raise InputError('Manual curation currently uses bin8 inputs.')
        slurm = data.get('slurm', {})
        if not slurm.get('ready'):
            raise InputError('Save cluster settings first.')
        if prior:
            config = {key: source[key] for key in ('coarse_run', 'inputs', 'orientation', 'preprocessing_work', 'main_revision')}
            report = json.loads(storage_path(parent_work, 'output/web_result.json').read_text())
            if report.get('landmark_basis') or source.get('landmark_basis'):
                config['landmark_basis'] = report.get('landmark_basis') or source['landmark_basis']
            geometry = service.path(report['geometry_file'])
            if not geometry.is_relative_to(storage_path(parent_work, 'output')):
                raise InputError('Invalid parent geometry checkpoint.')
            config['resume_geometry'] = str(geometry)
        else:
            config = dict(coarse_run=str(storage_path(parent_work, 'alignment')), inputs=inputs, orientation=source['orientation'], preprocessing_work=str(preprocessing_work(Path(inputs['stack']))), main_revision=source['main_revision'])
        if base.is_symlink() or base.parent.is_symlink():
            raise InputError('Invalid curation output directory.')
        reuse = prepare_model_reuse(root, row, denoising, inputs, final_binning) if action == 'reconstruct' else {}
        actions = 'provenance/actions'
        work = base / actions / uuid.uuid4().hex
        work.mkdir(parents=True)
        config.update(output=str(storage_path(work, 'output')), manual_edits=edits, pipeline_step_id=step_id, final_binning=final_binning, manual_action=action, source_work=str(parent_work), options=dict(binning=8, patch_size=33, max_passes=60, search_radius_px=4, entry_radius_px=16, reference_views=11, column_cell=24, smoothness_strength=smoothness, include_discarded=include_discarded, disabled_tilts=disabled, final_binning=final_binning, **gaussian, **denoising))
        paths = [storage_path(parent_work, 'config.json'), storage_path(parent_work, 'output/web_result.json'), storage_path(Path(config['coarse_run']), 'final_state.npz'), *[Path(inputs[key]) for key in ('stack', 'rawtlt', 'frozen_shifts')]]
        if config.get('resume_geometry'):
            paths.append(Path(config['resume_geometry']))
        if config.get('landmark_basis'):
            paths.append(Path(config['landmark_basis']))
        config['snapshot'] = [dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths if p.is_file()]
        config['snapshot'].extend(reuse.pop('tinycare_model_snapshot', []))
        config.update(reuse)
        path = storage_path(work, 'config.json')
        path.write_text(json.dumps(config, indent=2))
        storage_path(work, 'job.sbatch').write_text(render_script(slurm['template'], work, path, dict(cpus=4, gpus=1), 'greco.web.manual_refinement_worker'))
        job = dict(state='submitted', work=str(work), host=slurm.get('submission_host', ''), step_id=step_id, edits=edits, options=config['options'], manual_action=action)
        try:
            dispatch(job, False, command_result)
        except InputError as error:
            job.update(state='failed', error=str(error))
        if step.get('job'):
            step.setdefault('manual_history', []).append(step['job'])
        step['job'] = job
        service._write(root, data)
        return service._view(pid)

def status(service,pid,sid,step_id):
    from .reconstruction_steps import find,work_for
    with service.lock:
        root=service._root(pid);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==sid),None)
        if row is None:raise InputError('Unknown tilt series.')
        step=find(row,step_id)
        if step['kind']!='manual_curation':raise InputError('Not a Manual Curation step.')
        original=step.get('job',{})
        if original.get('work'):
            work=service.path(original['work'],exists=False)
            if not valid_work(root,work_for(root,sid,step),work):raise InputError('Invalid manual refinement directory.')
        job=refresh(service,{'preprocessing':original})
        if original and original!=job:step['job']=job;service._write(root,data)
        if job.get('work') and job.get('state') in ('submitted','running','unknown'):
            progress=storage_path(Path(job['work']), 'output/progress.json')
            if progress.is_file():
                try:job=dict(job,progress=json.loads(progress.read_text()))
                except (OSError,ValueError):pass
        return dict(job,step_id=step_id)
