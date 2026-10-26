"""Versioned CTF-only jobs; never overwrite the previous preprocessing."""
from greco.project_layout import project_root, publish_ctf, join as storage_path, owned as layout_owned
import json
from pathlib import Path
from .projects import InputError
from .preprocessing import refresh,render_script,command_result,validate_options
from .queue import ACTIVE,prepare_work


def apply_completed(row,job):
    if job.get('state')!='completed' or job.get('applied'):return False
    if row.get('preprocessing',{}).get('work')!=job.get('source_work'):
        job.update(state='failed',error='Preprocessing changed during CTF recalculation; previous results were not replaced.')
        return True
    if project_root(Path(job['work'])):
        return publish_ctf(row,job)
    row.setdefault('preprocessing_history',[]).append(dict(row['preprocessing']))
    row['preprocessing']=dict(job,applied=True,kind='ctf_redo')
    job['applied']=True
    return True


def submit(service,project_id,series_id,raw=None):
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        if not row.get('enabled',True):raise InputError('Enable this tilt series first.')
        current=refresh(service,{'preprocessing':row.get('ctf_redo',{})})
        if current['state'] in ACTIVE:raise InputError('CTF recalculation is already active.')
        prep=refresh(service,row)
        if prep['state']!='completed':raise InputError('Complete preprocessing first.')
        source=service.path(prep['work'])
        if not layout_owned(root,source):raise InputError('Invalid preprocessing source.')
        if raw is not None and not isinstance(raw,dict):raise InputError('Invalid CTF options.')
        from .preprocessing import DEFAULTS
        if set(raw or {})-set(DEFAULTS):raise InputError('Unknown CTF options.')
        ctf_options={k:v for k,v in (raw or {}).items() if k.startswith('ctf_') or k=='df_hand'}
        options=validate_options(dict(prep.get('options',{}),**ctf_options))
        # MC/sampling parameters belong to the existing images, not the form.
        old=prep.get('options',{})
        for key in options:
            if not key.startswith('ctf_') and key not in ('df_hand','cs_mm','amp_contrast','cpus','gpus'):
                options[key]=old.get(key,options[key])
        options['cs_mm']=data['metadata']['cs_mm']
        options['amp_contrast']=data['metadata']['amplitude_contrast']
        options['ctf_local_refine']=True
        options['gpus']=1
        orientation=data.get('orientation')
        if orientation:
            from .preprocessing import ctf_axis_from_orientation
            options['ctf_spatial_axis_deg']=ctf_axis_from_orientation(orientation)
        elif not raw or 'ctf_spatial_axis_deg' not in raw:
            # An implicit zero is not an explicitly calibrated raw axis.
            options['ctf_spatial_axis_deg']=None
        from .preprocessing import select_ctf_handedness
        fixed_hand,hand_source=select_ctf_handedness(options,raw,orientation)
        from .ctf_view import tilt_records
        records=tilt_records(storage_path(source, 'output'),options['output_binning'])
        if not records:raise InputError('No saved tilt records.')
        for record in records:
            folder=storage_path(storage_path(storage_path(source, 'output'), 'per_tilt'), f"tilt_{int(record['mdoc_index']):04d}")
            for name in ('ctf.json','ctf_tiles.npy','ctf_tiles_metadata.npz','shifts.npz'):
                if not (storage_path(folder, name)).is_file():raise InputError('Cached CTF tiles or frame shifts are missing; repeat preprocessing.')
            fitted=json.loads((storage_path(folder, 'ctf.json')).read_text())
            if abs(float(fitted['pixel_size_A'])-data['metadata']['pixel_size_A']/(options['eer_upscaling'] if row.get('input_format')=='EER' else 1))>1e-6:
                raise InputError('Pixel size changed; repeat preprocessing rather than reuse cached CTF tiles.')
            if options['ctf_tile_size']!=old.get('ctf_tile_size',512):
                raise InputError('Redo CTF reuses the saved tile size. Repeat preprocessing to change it.')
        first=json.loads((storage_path(storage_path(storage_path(storage_path(source, 'output'), 'per_tilt'), f"tilt_{int(records[0]['mdoc_index']):04d}"), 'ctf.json')).read_text())
        pixel=float(first['pixel_size_A'])
        low=options['ctf_res_low_A'] or 18*pixel
        high=options['ctf_res_high_A'] or max(2*pixel/.75,3.5)
        if not low>high>=2*pixel:raise InputError('CTF fit requires low resolution > high resolution ≥ Nyquist.')
        slurm=data.get('slurm',{})
        if not slurm.get('ready'):raise InputError('Save cluster settings first.')
        original=json.loads((storage_path(source, 'config.json')).read_text())
        work=prepare_work(service,root,row,'ctf_redo')
        from ..ctf_coordinates import PROTOCOL
        payload=dict(original,source_work=str(source),source_result=prep,orientation=orientation,
                     options=options,output=str(storage_path(work, 'output')),metadata=data['metadata'],ctf_axis_protocol=PROTOCOL,
                     ctf_fixed_hand=fixed_hand,ctf_hand_source=hand_source)
        (storage_path(work, 'config.json')).write_text(json.dumps(payload,indent=2))
        (storage_path(work, 'job.sbatch')).write_text(render_script(slurm['template'],work,storage_path(work, 'config.json'),options,'greco.web.ctf_redo_worker'))
        job=dict(state='submitted',work=str(work),host=slurm.get('submission_host',''),
                 source_work=str(source),options=options,kind='ctf_redo',ctf_axis_protocol=PROTOCOL,
                 ctf_fixed_hand=fixed_hand,ctf_hand_source=hand_source)
        try:
            identifier=command_result(['sbatch','--parsable',str(storage_path(work, 'job.sbatch'))],job['host']).split(';')[0]
            if not identifier.isdigit():raise InputError('Unexpected Slurm response.')
            job['job_id']=identifier
        except InputError as error:job.update(state='failed',error=str(error))
        if row.get('ctf_redo'):row.setdefault('ctf_redo_history',[]).append(row['ctf_redo'])
        row['ctf_redo']=job;service._write(root,data)
        return job


def status(service,project_id,series_id):
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        job=refresh(service,{'preprocessing':row.get('ctf_redo',{})})
        if job.get('state') in ACTIVE and job.get('work'):
            try:job['progress']=json.loads((storage_path(Path(job['work']), 'progress.json')).read_text())
            except (OSError,ValueError):pass
        changed=apply_completed(row,job)
        if row.get('ctf_redo') and (changed or row['ctf_redo']!=job):
            row['ctf_redo']=job;service._write(root,data)
        return job
