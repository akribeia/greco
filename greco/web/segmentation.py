"""Independent persistent segmentation workspaces; source jobs are immutable."""
from greco.project_layout import preprocessing_work, join as storage_path, record_directory, series_row, series_directory, owned as layout_owned
from greco.preprocessing_star import read_table
from .. import runtime
import json,re,uuid
from pathlib import Path
from .projects import InputError,active_series_ids
from .preprocessing import refresh,render_script,command_result
from .queue import dispatch
BINNINGS=(1,2,4,8,16)
KINDS={'geometry_refinement':'Refine geometry','denoise':'Denoise tomogram',
       'membranes':'Segment membranes','insert_particle':'Insert particle'}

def find(data,identifier):
    result=next((a for a in data.get('segmentations',[]) if a['id']==identifier),None)
    if result is None:raise InputError('Unknown segmentation.')
    return result

def rename_layer(service,pid,identifier,sid,number,name):
    """Rename a particle layer without changing its job identity or artifacts."""
    if not isinstance(name,str) or not name.strip() or len(name.strip())>100 or any(ord(c)<32 for c in name):
        raise InputError('Enter a layer name (1–100 characters, one line).')
    if type(number) is not int:raise InputError('Invalid layer number.')
    name=name.strip()
    with service.lock:
        root=service._root(pid);data=service._read(root);record=find(data,identifier)
        member=next((t for t in record['tomograms'] if t['series_id']==sid),None)
        if member is None:raise InputError('Unknown segmentation tomogram.')
        layer=next((s for s in member['steps'] if s['number']==number and s['kind']=='insert_particle'),None)
        if layer is None:raise InputError('Unknown particle layer.')
        layer['name']=name
        service._write(root,data)
        return dict(number=number,name=name)


def next_number(member):
    return 1+max((s['number'] for s in [*member['steps'],*member.get('cleared_steps',[])]),default=0)

def work_directory(root,record,member,kind):
    series=series_directory(root,series_row(root,member['series_id'])).name
    base=record_directory(root,record,'segmentation')/'series'/series
    layer_number=None
    if kind=='insert_particle':
        layer_number=1+max((s.get('layer_number',0) for s in [*member['steps'],*member.get('cleared_steps',[])] if s['kind']=='insert_particle'),default=0)
        work=base/'layers'/f'{layer_number:03d}'
        while work.exists():
            layer_number+=1;work=base/'layers'/f'{layer_number:03d}'
    else:work=base/('refinement' if kind=='geometry_refinement' else kind)
    return work,layer_number


def clear_step(service,pid,identifier,sid,number,confirmed=False):
    if confirmed is not True:raise InputError('Confirm clearing this step.')
    with service.lock:
        root=service._root(pid);data=service._read(root);record=find(data,identifier)
        member=next((t for t in record['tomograms'] if t['series_id']==sid),None)
        if member is None:raise InputError('Unknown segmentation tomogram.')
        step=next((s for s in member['steps'] if type(number) is int and s['number']==number),None)
        if step is None or step['kind']=='reconstruction':raise InputError('Only downstream steps can be cleared.')
        removed=[s for s in member['steps'] if s['number']==number or (step['kind'] not in ('insert_particle','membranes') and s['number']>number)]
        for s in removed:
            if s.get('work'):s.update(refresh(service,{'preprocessing':s}))
            if s.get('state') in ('submitted','running','unknown'):raise InputError('Wait for dependent jobs to finish before clearing.')
        member.setdefault('cleared_steps',[]).extend(dict(s,previous_state=s.get('state'),state='cleared') for s in removed)
        removed_numbers={s['number'] for s in removed}
        member['steps']=[s for s in member['steps'] if s['number'] not in removed_numbers]
        service._write(root,data)
        return service._view(pid)

def create(service, pid, raw):
    if not isinstance(raw, dict) or set(raw) - {'name', 'series_ids', 'binning'}:
        raise InputError('Unknown segmentation options.')
    name = raw.get('name', '')
    ids = raw.get('series_ids', [])
    binning = raw.get('binning', 8)
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise InputError('Enter a segmentation name (1–100 characters).')
    if type(binning) is not int or binning not in BINNINGS:
        raise InputError('Invalid segmentation binning.')
    if not isinstance(ids, list) or not ids or any((not isinstance(i, str) for i in ids)) or (len(set(ids)) != len(ids)):
        raise InputError('Select tomograms without duplicates.')
    with service.lock:
        root = service._root(pid)
        data = service._read(root)
        ids = active_series_ids({r['id']: r for r in data['series']}, ids)
        if not data.get('slurm', {}).get('ready'):
            raise InputError('Configure cluster Settings first.')
        members = []
        from .picking_source import source_job, volume_context
        for sid in ids:
            row = next((r for r in data['series'] if r['id'] == sid), None)
            if row is None:
                raise InputError('Unknown tilt series.')
            job = refresh(service, {'preprocessing': source_job(row)})
            if job.get('state') != 'completed':
                raise InputError(row['name'] + ': complete the selected reconstruction first.')
            work = service.path(job['work'])
            if not layout_owned(root, work):
                raise InputError('Invalid source reconstruction.')
            (_, _, _, provenance) = volume_context(work)
            members.append(dict(series_id=sid, name=row['name'], source_work=str(work), source_geometry=provenance.get('rigid_geometry'), steps=[]))
        identifier = uuid.uuid4().hex[:8]
        while any(a['id'] == identifier for a in data.get('segmentations', [])):
            identifier = uuid.uuid4().hex[:8]
        record = dict(id=identifier, name=name.strip(), binning=binning, tomograms=members)
        base = record_directory(root, record, 'segmentation').name
        suffix = 2
        used = {record_directory(root, a, 'segmentation') for a in data.get('segmentations', [])}
        while record_directory(root, record, 'segmentation') in used or record_directory(root, record, 'segmentation').exists():
            record['storage_name'] = f'{base}_{suffix}'
            suffix += 1
        data.setdefault('segmentations', []).append(record)
        service._write(root, data)
        for member in members:
            try:
                _submit(service, root, data, record, member, 'reconstruction', {})
            except (InputError, OSError, ValueError) as error:
                if not member['steps']:
                    member['steps'].append(dict(number=1, kind='reconstruction', state='failed', error=str(error)))
                else:
                    member['steps'][-1].update(state='failed', error=str(error))
        service._write(root, data)
        return service._view(pid)

def add_step(service,pid,identifier,sid,kind,options=None):
    if kind not in KINDS:raise InputError('Unknown segmentation step.')
    options={} if options is None else options
    allowed={'sta_source'} if kind=='geometry_refinement' else {'sta_source','color'} if kind=='insert_particle' else {'color'} if kind=='membranes' else {'method','model_from'} if kind=='denoise' else set()
    if not isinstance(options,dict) or set(options)-allowed:raise InputError('Unknown step options.')
    if kind=='denoise':
        if options.get('method','tinycare')!='tinycare':raise InputError('Segmentation supports TinyCARE only.')
        options=dict(options,method='tinycare')
    if kind=='denoise' and options.get('model_from') and options.get('method')!='tinycare':raise InputError('Only TinyCARE accepts a reusable model.')
    if kind=='membranes':
        options=dict(options,color=options.get('color','#f472b6'))
        if not isinstance(options['color'],str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',options['color']):raise InputError('Choose a valid membrane color.')
    if kind=='insert_particle' and (not isinstance(options.get('color'),str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',options['color'])):raise InputError('Choose a valid particle color.')
    with service.lock:
        root=service._root(pid);data=service._read(root);record=find(data,identifier)
        member=next((t for t in record['tomograms'] if t['series_id']==sid),None)
        if member is None:raise InputError('Tomogram does not belong to this segmentation.')
        _submit(service,root,data,record,member,kind,options)
        service._write(root,data);return service._view(pid)

def reset_workspace(service,root,data,work):
    """Do not replace a cleared singleton while a current job still uses it."""
    from .queue import _reset_work
    from .particle_jobs import config_uses_work
    if work.exists():
        for record in data.get('segmentations',[]):
            for member in record['tomograms']:
                for step in member['steps']:
                    if not step.get('work') or Path(step['work'])==work:continue
                    config=storage_path(Path(step['work']),'config.json')
                    if config.is_file() and config_uses_work(json.loads(config.read_text()),work):
                        raise InputError('Another segmentation job still uses this output. Clear its dependent jobs before replacing it.')
    return _reset_work(service,root,work)


def _submit(service, root, data, record, member, kind, options):
    if record.get('delete_requested_jobs'):
        raise InputError('This segmentation is being deleted. Wait for deletion to finish.')
    if kind!='insert_particle' and any(s['kind']==kind and s.get('work') for s in member['steps']):
        raise InputError('Clear the existing '+KINDS.get(kind,'reconstruction')+' job before submitting another.')
    from .particle_worker import check_snapshot
    labels = dict(reconstruction='Reconstruct', geometry_refinement='RefineGeometry', denoise='Denoise', insert_particle='InsertParticle', membranes='SegmentMembranes')
    number = next_number(member)
    parent = None
    for step in reversed(member['steps']):
        if step.get('work') and step['kind'] in ('reconstruction', 'geometry_refinement', 'denoise'):
            parent = refresh(service, {'preprocessing': step})
            step.update(parent)
            break
    if parent is not None and parent.get('state') != 'completed':
        raise InputError('Complete the preceding volume step first.')
    if kind=='membranes' and (not parent or parent.get('kind')!='denoise' or not parent.get('volumes',{}).get('original')):
        raise InputError('Complete TinyCARE denoising for the current geometry before segmenting membranes.')
    work,layer_number=work_directory(root,record,member,kind)
    if not re.fullmatch('[A-Za-z0-9_-]+', member['series_id']):
        raise InputError('Invalid tilt series identity.')
    source = service.path(member['source_work'])
    from .picking_source import volume_context
    (source_config, _, _, provenance) = volume_context(source)
    coarse = Path(provenance['picking_source_run'])
    geometry = (parent or {}).get('geometry_file') or member.get('source_geometry')
    parent_volume = (parent or {}).get('volumes', {}).get('original')
    inputs = [storage_path(source, 'config.json'), storage_path(source, 'output/web_result.json'), storage_path(coarse, 'final_state.npz'), *[Path(source_config['inputs'][k]) for k in ('stack', 'rawtlt', 'frozen_shifts')]]
    if geometry:
        inputs.append(service.path(str(geometry)))
    if parent_volume:
        inputs.append(service.path(str(parent_volume)))
    if kind == 'membranes':
        from .segmentation_membranes import runtime_paths
        (membrane_python, membrane_checkpoint) = runtime_paths()
        if not parent_volume:
            raise InputError('Complete the preceding tomogram first.')
        if not membrane_python.is_file():
            raise InputError('Membrane Torch environment is unavailable.')
        if not membrane_checkpoint.is_file():
            raise InputError('Membrane checkpoint is unavailable.')
        inputs.append(membrane_checkpoint)
    tinycare = kind == 'denoise' and options.get('method') == 'tinycare'
    model_from = None
    if tinycare:
        if not parent_volume:
            raise InputError('Complete the preceding tomogram first.')
        prep = preprocessing_work(Path(source_config['inputs']['stack']))
        settings = storage_path(prep, 'config.json')
        inputs.append(settings)
        settings_data = json.loads(settings.read_text())
        inputs.extend((Path(t['movie_path']) for t in settings_data['series']['tilts']))
        if settings_data['series'].get('gain_path'):
            inputs.append(Path(settings_data['series']['gain_path']))
        
        tilts = read_table(storage_path(prep, 'ctf/ctf.star'))
        inputs.append(storage_path(prep, 'ctf/ctf.star'))
        for tilt in tilts:
            folder = storage_path(storage_path(prep, 'per_tilt'), f"tilt_{int(tilt['z_value']):04d}")
            inputs.extend((storage_path(folder, name) for name in ('shifts.npz', 'ctf.json')))
            inputs.extend((p for p in [storage_path(folder, 'ctf_geometry.json'), storage_path(folder, 'detector_correction.npz')] if p.exists()))
        if options.get('model_from'):
            candidates = [s for a in data.get('segmentations', []) for t in a['tomograms'] for s in t['steps'] if s.get('work') == options['model_from'] and s.get('options', {}).get('method') == 'tinycare']
            if len(candidates) != 1:
                raise InputError('Choose an owned completed TinyCARE job.')
            job = refresh(service, {'preprocessing': candidates[0]})
            if job.get('state') != 'completed' or not job.get('tinycare_model'):
                raise InputError('Complete the selected TinyCARE model first.')
            model_from = service.path(job['tinycare_model'])
            if not layout_owned(root, model_from):
                raise InputError('Invalid TinyCARE model path.')
            import mrcfile
            from .._tinycare.config import model_metadata
            with mrcfile.open(parent_volume, header_only=True) as m:
                pixel = float(m.voxel_size.x)
            try:
                (norm, checkpoint) = model_metadata(model_from, pixel)
            except ValueError as error:
                raise InputError(str(error)) from error
            inputs.extend([storage_path(model_from, 'config.json'), storage_path(model_from, 'norm.json'), checkpoint])
    sta_landmarks = None
    if kind in ('geometry_refinement', 'insert_particle'):
        from .segmentation_sta import load_source, insertion_source
        (sta_landmarks, sta_snapshot) = (insertion_source if kind == 'insert_particle' else load_source)(service, data['id'], data, member, coarse, options.get('sta_source'))
    else:
        sta_snapshot = []
    snapshot = [dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in inputs]
    snapshot.extend(sta_snapshot)
    check_snapshot(snapshot)
    tinycare_python = runtime.python('tinycare')
    if tinycare and (not tinycare_python.is_file()):
        raise InputError('TinyCARE GPU environment is unavailable.')
    config = dict(action=kind, binning=record['binning'], source_work=str(source), coarse_run=str(coarse), source_config=source_config, geometry_file=str(geometry) if geometry else None, parent_volume=parent_volume, options=options, snapshot=snapshot, output=str(storage_path(work, 'output')))
    if tinycare:
        config.update(tinycare_python=str(tinycare_python), tinycare_model_from=str(model_from) if model_from else None)
    if kind == 'membranes':
        config.update(membrane_python=str(membrane_python), membrane_checkpoint=str(membrane_checkpoint))
    if sta_landmarks is not None:
        config['sta_particles' if kind == 'insert_particle' else 'sta_landmarks'] = sta_landmarks
    reset_workspace(service,root,data,work)
    path = storage_path(work, 'config.json')
    path.write_text(json.dumps(config, indent=2))
    slurm = data['slurm']
    storage_path(work, 'job.sbatch').write_text(render_script(slurm['template'], work, path, dict(cpus=4, gpus=1), 'greco.web.segmentation_worker'))
    step = dict(number=number, kind=kind, state='submitted', work=str(work), host=slurm.get('submission_host', ''), options=options)
    if layer_number is not None:step['layer_number']=layer_number
    else:member['steps']=[s for s in member['steps'] if s['kind']!=kind or s.get('work')]
    member['steps'].append(step)
    try:
        dispatch(step, False, command_result)
    finally:
        service._write(root, data)

def status(service,pid):
    with service.lock:
        root=service._root(pid);data=service._read(root)
        live={};hosts={}
        for record in data.get('segmentations',[]):
            for member in record['tomograms']:
                for step in member['steps']:
                    if step.get('job_id') and step.get('state') in ('submitted','running','unknown'):
                        hosts.setdefault(step.get('host',''),set()).add(str(step['job_id']))
        for host,ids in hosts.items():
            try:
                text=command_result(['squeue','-h','-j',','.join(sorted(ids)),'-o','%i|%T'],host)
                for line in text.splitlines():
                    fields=line.strip().split('|')
                    if len(fields)==2:live[host,fields[0]]=fields[1]
            except InputError:live=None;break
        for record in data.get('segmentations',[]):
            for member in record['tomograms']:
                for step in member['steps']:
                    if step.get('work'):
                        step.update(refresh(service,{'preprocessing':step},live_states=live))
                        if step.get('state') in ('submitted','running','unknown'):
                            try:step['progress']=json.loads((storage_path(Path(step['work']), 'output/progress.json')).read_text())
                            except (OSError,ValueError):pass
                        else:step.pop('progress',None)
        service._write(root,data)
        return service._view(pid)

def plane(service, pid, identifier, sid, number, index, axis='z', segment=False, *, autocontrast=True, contrast=1, brightness=0):
    from .reconstruction import volume_plane_image
    import io
    with service.lock:
        root = service._root(pid)
        record = find(service._read(root), identifier)
        member = next((t for t in record['tomograms'] if t['series_id'] == sid), None)
        if member is None:
            raise InputError('Unknown segmentation tomogram.')
        step = next((s for s in member['steps'] if s['number'] == number), None)
        if step is None or step.get('state') != 'completed':
            raise InputError('Complete this volume first.')
        path = service.path(step['volumes']['original'])
        if not path.is_relative_to(record_directory(root, record, 'segmentation')):
            raise InputError('Invalid segmentation volume.')
        denoised = step['kind']=='denoise'
        layers = [dict(s) for s in member['steps'] if s.get('state') == 'completed' and (s.get('particle_layer') or s.get('membrane_layer'))]
    image = volume_plane_image(path, index, axis, autocontrast=autocontrast, contrast=contrast, brightness=brightness, denoised=denoised)
    if segment:
        import mrcfile, numpy as np
        from PIL import Image
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
        for layer in layers:
            layer_path = service.path(layer.get('particle_layer') or layer['membrane_layer'])
            if not layer_path.is_relative_to(record_directory(root, record, 'segmentation')):
                raise InputError('Invalid particle layer.')
            with mrcfile.mmap(layer_path) as m:
                if tuple(m.data.shape) != tuple(step['shape']):
                    raise InputError('Particle layer grid differs from the tomogram.')
                density = np.array(np.take(m.data, index, axis={'z': 0, 'y': 1, 'x': 2}[axis]))
            color = layer.get('color', layer.get('options', {}).get('color', '#f472b6' if layer.get('membrane_layer') else '#38bdf8'))
            if not re.fullmatch('#[0-9a-fA-F]{6}', color):
                raise InputError('Invalid layer color.')
            color = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
            alpha = (0.9 * np.sqrt(np.clip(density, 0, 1)))[..., None]
            rgb = rgb * (1 - alpha) + color * alpha
        image = Image.fromarray(np.clip(rgb, 0, 255).astype('uint8'))
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()
