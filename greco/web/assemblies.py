"""Persist STA particle assembly definitions without extracting particle data."""
from greco.project_layout import join as storage_path, record_directory, STA_CATEGORIES, owned as layout_owned
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from .projects import InputError
BINNINGS = (1, 2, 3, 4, 6, 8, 16)
PARTICLE_KINDS=('extraction','classification','refinement3d','polishing','ctf_refinement','postprocessing')
JOB_LABELS=dict(extraction='Extract',classification='Class3D',refinement3d='Refine3D',polishing='Polishing',ctf_refinement='CTFRefine',postprocessing='PostProcess')

def default_mask_path(record):
    """Actual picking mask, already resampled/flipped, never a display sphere."""
    from pathlib import Path
    if record.get('default_mask_path'):return record['default_mask_path']
    for member in record.get('tomograms',[]):
        if member.get('picking_work'):
            path=storage_path(Path(member['picking_work']), 'output/mask.mrc')
            if path.is_file():return str(path)
    return None

def restore_job_history(root, record):
    """Recover older owned jobs whose latest-per-kind manifest pointers were replaced."""
    import json
    parent = assembly_directory(root, record)
    if not parent.is_dir() or parent.is_symlink():
        return
    history = record.get('job_history', [])
    known = {j.get('work') for j in history}
    for kind in PARTICLE_KINDS:
        job=record.get(kind)
        if job and job.get('work'):known.add(job['work'])
    candidates = parent.glob('*/*')
    for work in sorted(candidates):
        if not work.name.isdigit() or str(work) in known or work.is_symlink() or (not work.is_dir()):
            continue
        try:
            config = json.loads(storage_path(work, 'config.json').read_text())
            if config.get('assembly_id') != record['id']:
                continue
            result = json.loads(storage_path(work, 'output/web_result.json').read_text()) if storage_path(work, 'output/web_result.json').is_file() else {}
        except (OSError, ValueError):
            continue
        kind = config.get('job_kind', next((k for (k, v) in STA_CATEGORIES.items() if v == work.parent.name), None))
        if kind not in PARTICLE_KINDS:
            continue
        history.append(dict(result, work=str(work), number=int(work.name), name=work.name, kind=kind, state=result.get('state', 'unknown'), box_size=config.get('output_box'), pixel_size_A=config.get('pixel_size_A')))
    if history:
        record['job_history'] = history

def assembly_directory(root, record):
    """Resolve an owned directory without changing stable assembly IDs."""
    return record_directory(root, record, 'sta')

def allocate_job_directory(root, record, kind):
    parent = assembly_directory(root, record)
    parent.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name) for p in parent.glob('*/*') if p.is_dir() and p.name.isdigit()]
    number = max(numbers + [0]) + 1
    path = storage_path(storage_path(parent, STA_CATEGORIES[kind]), f'{number:03d}')
    if path.exists():
        raise InputError('Duplicate STA job number.')
    path.mkdir(parents=True)
    storage_path(path, 'logs').mkdir()
    record['last_job_number'] = number
    return (path, number)

def extraction_sampling(record):
    """Resolve the assembly's output grid, never bin that grid again.

    Assembly definitions retain their physical extent and stored output pixel
    size. Explicit output fields are authoritative but must be consistent.
    """
    from ..particle_geometry import output_box
    try:
        pixel=float(record.get('output_pixel_size_A',record['pixel_size_A']))
        extent=float(record['dimensions_A'])
        native=float(record.get('native_pixel_size_A',pixel/record['binning']))
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        raise InputError('The assembly sampling is incomplete.') from None
    if not all(math.isfinite(v) and v>0 for v in (pixel,extent,native)):
        raise InputError('Invalid assembly sampling.')
    # extent / output sampling is ALREADY the output box; no /binning here.
    expected=output_box(extent/pixel,1)
    size=record.get('output_box_size',expected)
    if isinstance(size,bool) or not isinstance(size,int) or size!=expected:
        raise InputError('Assembly output box is inconsistent with its extent and sampling.')
    if not math.isclose(pixel,float(record['pixel_size_A']),rel_tol=1e-6):
        raise InputError('Assembly output sampling is inconsistent.')
    return dict(box_size=size,pixel_size_A=pixel,native_pixel_size_A=native)

def delete(service, project_id, assembly_id, confirmed=False):
    """Delete the definition and its exact assembly-owned output directory.

    Never derive deletion targets from arbitrary paths in the manifest:
    tomograms and picking outputs are shared inputs, not assembly assets.
    """
    if confirmed is not True:
        raise InputError('Confirm the particle assembly deletion.')
    if not isinstance(assembly_id,str) or not re.fullmatch(r'[a-f0-9]{32}',assembly_id):
        raise InputError('Invalid particle assembly ID.')
    with service.lock:
        root=service._root(project_id)
        data=service._read(root)
        assemblies=data.get('particle_assemblies',[])
        if not any(row['id']==assembly_id for row in assemblies):
            raise InputError('Particle assembly not found.')
        from .particle_jobs import update, ACTIVE
        record=next(row for row in assemblies if row['id']==assembly_id)
        target=assembly_directory(root, record)
        if target.is_symlink() or service.path(str(target),exists=False)!=target:
            raise InputError('Refusing to delete a symlinked or invalid assembly directory.')
        if target.exists() and not target.is_dir():
            raise InputError('The assembly output path is not a directory.')
        if any(update(service,record,kind=k)['state'] in ACTIVE for k in PARTICLE_KINDS):
            raise InputError('Wait for active STA jobs to finish before deleting the assembly.')
        if any(update(service,{j['kind']:j},kind=j['kind'])['state'] in ACTIVE
               for j in record.get('job_history',[]) if j.get('kind')):
            raise InputError('Wait for earlier assembly jobs to finish before deleting the assembly.')
        # Keep the manifest entry if cleanup fails, allowing the user to retry.
        # rmtree does not follow symlinks inside the verified target directory.
        if target.exists():shutil.rmtree(target)
        data['particle_assemblies']=[row for row in assemblies if row['id']!=assembly_id]
        service._write(root,data)
        return service._view(project_id)

def create(service, project_id, raw):
    if not isinstance(raw, dict) or set(raw) - {'name', 'series_ids', 'box_size', 'binning', 'pick_assembly_id'}:
        raise InputError('Unknown particle assembly parameters.')
    name = raw.get('name', '')
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise InputError('Enter an assembly name (1–100 characters).')
    box = raw.get('box_size')
    binning = raw.get('binning', 8)
    if isinstance(binning, bool) or not isinstance(binning, int) or binning not in BINNINGS:
        raise InputError('Choose binning 1, 2, 3, 4, 6, 8 or 16.')
    if isinstance(box, bool) or not isinstance(box, int) or (not 8 <= box <= 2048):
        raise InputError('Box size must be an integer from 8 to 2048 pixels.')
    ids = raw.get('series_ids')
    if not isinstance(ids, list) or not ids or (not all((isinstance(i, str) for i in ids))) or (len(set(ids)) != len(ids)):
        raise InputError('Select at least one tomogram, without duplicates.')
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        pick_id = raw.get('pick_assembly_id', data.get('active_pick_assembly'))
        pick_source = next((a for a in data.get('pick_assemblies', []) if a['id'] == pick_id), None)
        if pick_id and pick_source is None:
            raise InputError('Unknown pick assembly.')
        rows = {row['id']: row for row in data['series']}
        from .projects import active_series_ids
        ids = active_series_ids(rows, ids)
        members = []
        native_pixel = None
        for identifier in ids:
            row = rows.get(identifier)
            if row is None:
                raise InputError('Unknown tomogram.')
            job = pick_source.get('series', {}).get(identifier, {}).get('picking', {}) if pick_source else row.get('picking', {})
            count = job.get('pick_count', 0)
            from .slurm_arrays import valid_job_id
            pending = job.get('state') in ('submitted', 'running') and valid_job_id(job.get('job_id'))
            if not pending and (job.get('state') != 'completed' or not isinstance(count, int) or count <= 0):
                raise InputError(row['name'] + ': complete picking with at least one particle first.')
            try:
                pixel = float(row.get('metadata', {}).get('pixel_size_A', data.get('metadata', {}).get('pixel_size_A')))
            except (TypeError, ValueError):
                raise InputError('Native pixel size is missing.') from None
            if not math.isfinite(pixel) or pixel <= 0:
                raise InputError('Native pixel size must be positive.')
            if native_pixel is not None and (not math.isclose(pixel, native_pixel, rel_tol=1e-06)):
                raise InputError('Selected tomograms must share the same native pixel size.')
            native_pixel = pixel
            if pending:
                work = service.path(job['work'])
                if not layout_owned(root, work):
                    raise InputError('Invalid picking dependency.')
                members.append(dict(series_id=identifier, name=row['name'], particle_count=0, picking_work=str(work), picking_dependency=True, picking_job_id=job['job_id'], picking_host=job.get('host', '')))
                continue
            coordinates = service.path(job.get('star'))
            volume = service.path(job.get('volume'))
            if not coordinates.is_file() or not volume.is_file() or (not layout_owned(root, coordinates)) or (not layout_owned(root, volume)):
                raise InputError(row['name'] + ': picking outputs are unavailable in this project.')
            stat = coordinates.stat()
            members.append(dict(series_id=identifier, name=row['name'], particle_count=count, star=str(coordinates), volume=str(volume), picking_work=job.get('work'), star_size=stat.st_size, star_mtime_ns=stat.st_mtime_ns))
        record = dict(id=uuid.uuid4().hex, name=name.strip(), state='configured', created_at=datetime.now(timezone.utc).isoformat(), box_size=box, binning=binning, native_pixel_size_A=native_pixel, pixel_size_A=native_pixel * binning, box_sampling='native', dimensions_A=box * native_pixel, particle_count=sum((m['particle_count'] for m in members)), tomograms=members, extracted=False)
        if pick_source:
            record.update(pick_assembly_id=pick_id, pick_assembly_name=pick_source['name'])
        inherited_mask = default_mask_path(record)
        if inherited_mask:
            record['default_mask_path'] = inherited_mask
        record.update(sampling_version=1, sampling_configured='binning' in raw)
        if record['sampling_configured']:
            sampling = extraction_sampling(record)
            record.update(output_box_size=sampling['box_size'], output_pixel_size_A=sampling['pixel_size_A'])
        data.setdefault('particle_assemblies', []).append(record)
        service._write(root, data)
        return service._view(project_id)
