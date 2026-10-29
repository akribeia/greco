"""Validated Slurm submission and read-only MRC visualization."""
from greco.project_layout import join as storage_path, owned as layout_owned
import io
import json
import math

from pathlib import Path
import re
import shlex
import subprocess

from .projects import InputError, validate_metadata

DEFAULTS = dict(mc_b_factor=500., mc_iter=15, mc_tol=.1, mc_search_radius=64,
                flip_y=False, transpose=False, ctf_regularization=.05,
                ctf_tile_size=512, ctf_spatial=True, ctf_spatial_axis_deg=None,
                df_hand=1, gpus=1, cpus=8, eer_group=15, eer_threads=8, eer_upscaling=1,
                cs_mm=2.7, amp_contrast=.1, output_binning=8,output_bin1_sta=True,ctf_local_refine=True,
                ctf_res_low_A=None,ctf_res_high_A=None)


def ctf_axis_from_orientation(orientation):
    """Gradient-direction angle in raw array (x right, y down) coordinates.

    The operator has made the physical tilt axis vertical by flipping Y
    first and rotating counter-clockwise. The defocus gradient is transverse
    to that axis, i.e. horizontal in the oriented view. Inverting the same
    transform gives this raw gradient direction. With the default unflipped
    view, the displayed rotation and MDOC TiltAxisAngle have this same value.
    """
    rotation = float(orientation['rotation_deg'])
    return -rotation if orientation.get('flip_y',False) else rotation


def select_ctf_handedness(options,raw,orientation):
    """An explicit job sign wins over the project's saved calibration."""
    if raw is not None and 'df_hand' in raw:
        return True,'explicit'
    if orientation:
        options['df_hand']=orientation['df_hand']
        from ..ctf_local_refinement import current_hand_test
        if current_hand_test(orientation):return True,'tested_orientation'
    return False,'auto'


def validate_options(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise InputError('Unknown preprocessing parameters.')
    values = dict(DEFAULTS, **raw)
    for key in ('ctf_res_low_A','ctf_res_high_A'):
        if values[key] in (None,''):
            values[key]=None
            continue
        if isinstance(values[key],bool):
            raise InputError('Invalid CTF fit resolution.')
        try:value=float(values[key])
        except (ValueError,TypeError):raise InputError('Invalid CTF fit resolution.') from None
        if not math.isfinite(value) or not 0<value<=10000:
            raise InputError('CTF fit resolution must be positive and finite, in Å.')
        values[key]=value
    if values['ctf_res_low_A'] is not None and values['ctf_res_high_A'] is not None and values['ctf_res_low_A']<=values['ctf_res_high_A']:
        raise InputError('Low resolution must be larger than high resolution in Å.')
    limits = {'mc_b_factor':(0,10000), 'mc_iter':(1,1000), 'mc_tol':(.0001,10),
              'mc_search_radius':(1,256), 'ctf_regularization':(.00001,10),
              'ctf_tile_size':(64,2048), 'ctf_spatial_axis_deg':(-360,360),
              'gpus':(1,8), 'cpus':(1,128), 'df_hand':(-1,1),
              'eer_group':(1,10000), 'eer_threads':(1,64), 'eer_upscaling':(1,2),
              'cs_mm':(.01,10), 'amp_contrast':(.0001,.9999),'output_binning':(1,16)}
    integers = {'mc_iter','mc_search_radius','ctf_tile_size','gpus','cpus','df_hand','eer_group','eer_threads','eer_upscaling','output_binning'}
    for key, (lo, hi) in limits.items():
        value = values[key]
        if key=='ctf_spatial_axis_deg' and value in (None,''):
            values[key]=None
            continue
        if isinstance(value, bool):
            raise InputError('Invalid parameter: ' + key)
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise InputError('Invalid parameter: ' + key) from None
        if not math.isfinite(value) or not lo <= value <= hi or (key in integers and value != int(value)):
            raise InputError('Parameter out of range: ' + key)
        values[key] = int(value) if key in integers else value
    if values['df_hand'] not in (-1,1):
        raise InputError('Defocus handedness must be -1 or 1.')
    if values['output_binning'] != 8:
        raise InputError('Preprocessing requires fixed output binning 8.')
    for key in ('flip_y','transpose','ctf_spatial','ctf_local_refine','output_bin1_sta'):
        if not isinstance(values[key], bool):
            raise InputError('Expected a checkbox value: ' + key)
    if values['flip_y'] and values['transpose']:
        raise InputError('Choose flip Y or transpose, not both.')
    if values['gpus']*values['eer_threads'] > values['cpus']:
        raise InputError('CPU cores must cover GPUs × EER decoder threads.')
    if values['ctf_tile_size'] % 2:
        raise InputError('CTF tile size must be even.')
    return values


def command_result(argv, host=''):
    if host:
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(argv)]
    try:
        return subprocess.run(argv, text=True, capture_output=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, 'stderr', '') or str(error)
        raise InputError('Slurm command failed: ' + detail[-2000:]) from None


def preflight(service, row, options):
    if row.get('input_kind') == 'tilt_stack' or row.get('input_format') not in ('TIFF','EER'):
        raise InputError(row['name'] + ': this native MC adapter accepts TIFF/EER movies. Already-summed MRC stacks need a separate adapter; no job was submitted.')
    import tifffile
    validate_metadata(row['metadata'])
    pixel=row['metadata']['pixel_size_A']/(options['eer_upscaling'] if row['input_format']=='EER' else 1)
    low=options['ctf_res_low_A'] if options['ctf_res_low_A'] is not None else 18*pixel
    high=options['ctf_res_high_A'] if options['ctf_res_high_A'] is not None else max(2*pixel/.75,3.5)
    if not low>high>=2*pixel:
        raise InputError('CTF fit requires low resolution > high resolution ≥ Nyquist (2 × bin1 pixel size), in Å.')
    for tilt in row['tilts']:
        movie = service.path(tilt['movie_path'])
        if movie.suffix.lower()=='.eer':
            # Read only the TIFF container header on the web server. The native
            # decoder executes in the submitted worker, not in the HTTP thread.
            with tifffile.TiffFile(movie) as handle:
                page=handle.pages[0]
                raw_frames=len(handle.pages)
                if int(page.compression) not in (65000,65001):
                    raise InputError('Unrecognized EER compression.')
                shape=(raw_frames//options['eer_group'],*page.shape)
        else:
            with tifffile.TiffFile(movie) as handle:
                shape = handle.series[0].shape
        if len(shape) != 3 or shape[0] < 2 or min(shape[-2:]) < options['ctf_tile_size']:
            raise InputError(row['name'] + ': expected multi-frame movies large enough for the CTF tiles; found ' + str(shape))
    source = service.path(row['mdoc_path'])
    if [source.stat().st_size,source.stat().st_mtime_ns] != row['mdoc_fingerprint']:
        raise InputError(row['name'] + ': mdoc changed since import; import it again.')
    gain = service.gain(row.get('gain_path'))
    if gain:
        from ..movie_io import load_gain
        try:load_gain(gain,flip_y=row.get('gain_flip_y',False))
        except (OSError,ValueError) as error:raise InputError('Invalid gain reference: '+str(error)) from error


def render_script(template, work, config, options, worker_module='greco.web.preprocess_worker'):
    if re.search(r'^#SBATCH\s+--(?:array|wrap)\b',template,re.M):
        raise InputError('Remove array/wrap directives: Greco assigns the array indices and worker commands.')
    (work/'logs').mkdir(parents=True,exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    from ..runtime import python, worker_environment
    configured = worker_environment()
    names = ('PYTHONPATH', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
             'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'CUDA_PATH',
             'CUDA_HOME', 'PATH', 'LD_LIBRARY_PATH', 'GRECO_RUNTIME_ROOT',
             'GRECO_ASSETS_ROOT', 'GRECO_MAIN_PYTHON')
    env = 'unset PYTHONHOME\n' + '\n'.join(
        'export ' + name + '=' + shlex.quote(configured[name])
        for name in names if name in configured)
    prefix=('greco_seg_' if worker_module.endswith('segmentation_worker') else
            'greco_geom_' if worker_module.endswith('geometry_worker') else
            'greco_angle_' if worker_module.endswith('angle_worker') else
            'greco_sta_' if worker_module.endswith('particle_worker') else
            'greco_pick_' if worker_module.endswith('picking_worker') else
            'greco_hand_' if worker_module.endswith('handedness_worker') else
            'greco_ctf_' if worker_module.endswith('ctf_redo_worker') else
            'greco_refine_' if worker_module.endswith('refinement_worker') else
            'greco_recon_' if worker_module.endswith('reconstruction_worker') else 'greco_mc_')
    executable=str(python('main'))
    if worker_module.endswith(('angle_worker','picking_worker')):
        # Image refinement and picking use the configured vision interpreter.
        torch_python=python('vision')
        if worker_module.endswith('picking_worker') and not torch_python.is_file():
            raise InputError('The CUDA PyTorch environment for Greco Picker is unavailable.')
        if torch_python.is_file():executable=str(torch_python)
    replacements = dict(job_name=prefix + work.name[:10], log_path=shlex.quote(str(storage_path(work, 'slurm.out'))),
                        error_path=shlex.quote(str(storage_path(work, 'slurm.err'))), nodes='1', ntasks='1',
                        cpus_per_task=str(options['cpus']), environment=env, work_dir=str(repo),
                        command=shlex.join([executable,'-m','greco.worker_runner','--module',worker_module,'--config',str(config)]))
    unknown = set(re.findall(r'{{(\w+)}}', template)) - set(replacements)
    if unknown:
        raise InputError('Unresolved Slurm settings: ' + ', '.join(sorted(unknown)))
    script = re.sub(r'{{(\w+)}}', lambda match: replacements[match[1]], template)
    if not re.search(r'^#SBATCH\s+--(?:gres|gpus|gpus-per-node|gpus-per-task)\b', script, re.M):
        lines = script.splitlines()
        lines.insert(1, '#SBATCH --gres=gpu:' + str(options['gpus']))
        script = '\n'.join(lines) + '\n'
    from ..provenance import record_code
    record_code(config,worker_module,'submitted')
    return script


def submit(service, project_id, series_ids, raw,*,queue_only=False,only_missing=False):
    from .queue import dispatch,prepare_work,store_job
    options = validate_options(raw)
    if not isinstance(series_ids,list) or not series_ids or len(series_ids)!=len(set(series_ids)):
        raise InputError('Select one or more distinct tilt series.')
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        metadata=data.get('metadata',{})
        options=validate_options(dict(options,
            cs_mm=metadata.get('cs_mm',options['cs_mm']),
            amp_contrast=metadata.get('amplitude_contrast',options['amp_contrast'])))
        orientation = data.get('orientation')
        if orientation:
            orientation = dict(orientation,ctf_axis_deg=ctf_axis_from_orientation(orientation))
            options.update(ctf_spatial_axis_deg=orientation['ctf_axis_deg'])
        fixed_hand,hand_source=select_ctf_handedness(options,raw,orientation)
        slurm = data.get('slurm', {})
        if not slurm.get('ready'):
            raise InputError('Save a valid cluster template in Settings first.')
        rows = {row['id']:row for row in data['series']}
        if any(identifier not in rows for identifier in series_ids):
            raise InputError('Unknown tilt series.')
        from .projects import active_series_ids
        series_ids=active_series_ids(rows,series_ids,bulk=only_missing)
        if only_missing:
            from .queue import needs_submission
            series_ids=[i for i in series_ids if needs_submission(rows[i].get('preprocessing',{}))]
            if not series_ids:
                result=service._view(project_id);result['submission_errors']=[];return result
        for identifier in series_ids:
            row = rows[identifier]
            from .queue import guard_preprocessing_dependencies
            guard_preprocessing_dependencies(data,identifier)
            status = refresh(service, row)
            if status.get('state') in ('queued','submitting','submitted','running','unknown'):
                raise InputError(row['name'] + ': a preprocessing job is already active.')
            preflight(service,row,options)
        data['preprocessing_options'] = options
        errors = [];array_entries=[]
        for identifier in series_ids:
            row = rows[identifier]
            work = prepare_work(service,root,row,'preprocessing',queue_only)
            work = service.path(str(work))
            config = storage_path(work, 'config.json')
            from ..ctf_coordinates import PROTOCOL
            payload = dict(series=row,options=options,orientation=orientation,output=str(storage_path(work, 'output')),ctf_axis_protocol=PROTOCOL,
                           ctf_fixed_hand=fixed_hand,ctf_hand_source=hand_source)
            config.write_text(json.dumps(payload,indent=2))
            script = render_script(slurm['template'],work,config,options)
            (storage_path(work, 'job.sbatch')).write_text(script)
            job = dict(state='submitted',work=str(work),host=slurm.get('submission_host',''),options=options,ctf_axis_protocol=PROTOCOL,
                       ctf_fixed_hand=fixed_hand,ctf_hand_source=hand_source)
            if only_missing and not queue_only:
                array_entries.append((row,job,()));continue
            try:
                dispatch(job,queue_only,command_result)
            except InputError as error:
                job.update(state='failed',error=str(error))
                errors.append(row['name'] + ': ' + str(error))
            store_job(row,'preprocessing',job,queue_only)
            service._write(root,data)
        if array_entries:
            from .slurm_arrays import submit_arrays
            errors+=submit_arrays(service,root,data,'preprocessing',array_entries,command_result)
        from ..ctf_coordinates import PROTOCOL
        data['ctf_axis_protocol']=PROTOCOL
        service._write(root,data)
        result = service._view(project_id)
        result['submission_errors'] = errors
        return result


def preprocessing_state(job, redo):
    """A replacement CTF is part of preprocessing until it finishes."""
    job=dict(job)
    for key in ('ctf_redo_active','ctf_redo_job_id'):
        job.pop(key,None)
    if redo.get('state') in ('queued','submitting','submitted','running','unknown'):
        job.update(state=redo['state'],ctf_redo_active=True,ctf_redo_job_id=redo.get('job_id'))
    return job


def refresh(service, row, live_states=None):
    job = dict(row.get('preprocessing',{}))
    if not job:
        return dict(state='pending')
    work = service.path(job['work'],exists=False)
    if not work.exists():
        if job.get('state') in ('submitted','running','submitting','unknown'):return job
        return dict(job,state='failed',error='Job directory is missing; submit again to regenerate it.')
    done = storage_path(storage_path(work, 'output'), 'web_result.json')
    if done.exists():
        from .metadata_cache import read_json
        result = read_json(service, done)
        if (result.get('state')=='partial' and not result.get('denoising_errors') and 'active_count' in result
                and 'total_tilts' in result and 'last_gcc_stop' in result and result.get('volumes',{}).get('original')):
            result=dict(result,state='completed')
            result.pop('error',None);job.pop('error',None)
            result['warning']=f"Tomogram uses {result['active_count']}/{result['total_tilts']} accepted tilts."
        if result.get('denoising_errors') and result.get('state')=='completed':
            result=dict(result,state='partial',warning='Reconstruction available; requested TinyCARE failed.')
        if result.get('state')=='completed' and 'error' not in result:
            job.pop('error',None);job.pop('status_error',None)
        job.update(result)
    elif job.get('job_id') and job.get('state') not in ('completed','failed'):
        try:
            if live_states is None:
                live = command_result(['squeue','-h','-j',str(job['job_id']),'-o','%T'],job.get('host','')).strip()
            else:
                live = live_states.get((job.get('host',''),str(job['job_id'])), '')
            if live:
                state = live.splitlines()[0].split('|')[0].split()[0]
            else:
                from .slurm_arrays import accounting_state
                state=accounting_state(job['job_id'],command_result,job.get('host',''),live_states)
            if state=='CANCELLED':
                job.update(state='pending',last_job_id=job.pop('job_id'),last_slurm_state=state)
                job.pop('error',None);job.pop('status_error',None)
            elif state in ('FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','COMPLETED'):
                job.update(state='failed',error='Slurm: ' + state + '; no complete output. See slurm.err.')
            elif state in ('RUNNING','COMPLETING'):
                job['state']='running'
            elif state in ('PENDING','CONFIGURING','REQUEUED'):
                job['state']='submitted'
            else:
                job['state']='unknown'
        except InputError as error:
            job.update(state='unknown',status_error=str(error))
    redo=row.get('ctf_redo',{})
    if redo.get('state') in ('queued','submitting','submitted','running','unknown'):
        redo=refresh(service,{'preprocessing':redo},live_states)
    return preprocessing_state(job,redo)


def project_status(service, project_id):
    """Refresh all task states with one live scheduler query per host."""
    kinds = ('preprocessing','reconstruction','refinement','angle_refinement','geometry_refinement','picking','handedness_test','ctf_redo')
    from .assemblies import PARTICLE_KINDS
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        hosts = {}
        from .pick_assemblies import inactive_rows,save_proxy
        other_picks=list(inactive_rows(data))
        from .reconstruction_assemblies import inactive_jobs
        other_reconstructions=list(inactive_jobs(data))
        for owner,key in other_reconstructions:
            job=owner[key]
            if job.get('job_id') and job.get('state') in ('submitted','running','submitting','unknown'):
                hosts.setdefault(job.get('host',''),set()).add(str(job['job_id']))
        for row,_ in other_picks:
            job=row.get('picking',{})
            if job.get('job_id') and job.get('state') in ('submitted','running','submitting','unknown'):
                hosts.setdefault(job.get('host',''),set()).add(str(job['job_id']))
        for row in data['series']:
            for kind in kinds:
                job = row.get(kind, {})
                if job.get('job_id') and job.get('state') in ('submitted','running','submitting','unknown'):
                    hosts.setdefault(job.get('host',''),set()).add(str(job['job_id']))
            for step in row.get('processing_steps',[]):
                job=step.get('job',{})
                if job.get('job_id') and job.get('state') in ('submitted','running','submitting','unknown'):
                    hosts.setdefault(job.get('host',''),set()).add(str(job['job_id']))
        live = {}
        for record in data.get('particle_assemblies',[]):
            for kind in PARTICLE_KINDS:
                job=record.get(kind,{})
                if job.get('job_id') and job.get('state') in ('submitted','running','submitting','unknown'):
                    hosts.setdefault(job.get('host',''),set()).add(str(job['job_id']))
            for job in record.get('job_history',[]):
                if job.get('job_id') and job.get('state') in ('submitted','running','submitting','unknown'):
                    hosts.setdefault(job.get('host',''),set()).add(str(job['job_id']))
        for host, ids in hosts.items():
            try:
                command=['squeue','-h','-j',','.join(sorted(ids)),'-o','%i|%T']
                if any('_' in identifier for identifier in ids):command.append('--array')
                output = command_result(command,host)
                for line in output.splitlines():
                    identifier, separator, state = line.strip().partition('|')
                    if separator and identifier in ids:
                        live[(host,identifier)] = state.strip()
            except InputError:
                # Do not replace known states when the scheduler is unreachable.
                return service._view(project_id)
        changed = False
        for owner,key in other_reconstructions:
            if owner[key].get('state') in ('submitted','running','submitting','unknown'):
                updated=refresh(service,{'preprocessing':owner[key]},live)
                if updated!=owner[key]:owner[key]=updated;changed=True
        for row,saved in other_picks:
            if row.get('picking') and row['picking'].get('state') in ('submitted','running','submitting','unknown'):
                updated=refresh(service,{'preprocessing':row['picking']},live)
                if updated!=row['picking']:
                    row['picking']=updated
                    if updated.get('state')=='completed':row['pick_count']=updated.get('pick_count',0)
                    save_proxy(row,saved);changed=True
        for row in data['series']:
            for step in row.get('processing_steps',[]):
                if step.get('job',{}).get('state') in ('submitted','running','submitting','unknown'):
                    updated=refresh(service,{'preprocessing':step['job']},live)
                    if updated!=step['job']:step['job']=updated;changed=True
            for kind in kinds:
                if row.get(kind) and row[kind].get('state') in ('submitted','running','submitting','unknown'):
                    updated = refresh(service, {'preprocessing':row[kind]}, live)
                    if updated != row[kind]:
                        row[kind] = updated
                        changed = True
                    if kind == 'picking' and updated.get('state') == 'completed' and 'pick_count' in updated:
                        if row.get('pick_count') != updated['pick_count']:
                            row['pick_count'] = updated['pick_count']
                            changed = True
            if row.get('ctf_redo'):
                from .ctf_redo import apply_completed
                if apply_completed(row,row['ctf_redo']):changed=True
        from .particle_jobs import update
        for record in data.get('particle_assemblies',[]):
            for kind in PARTICLE_KINDS:
                if record.get(kind,{}).get('state') in ('submitted','running','submitting','unknown'):
                    updated=update(service,record,live,kind=kind)
                    if updated!=record[kind]:
                        record[kind]=updated;changed=True
                    if kind=='extraction' and record.get('extracted')!=(updated['state']=='completed'):
                        record['extracted']=updated['state']=='completed';changed=True
            for index,job in enumerate(record.get('job_history',[])):
                kind=job.get('kind')
                if kind not in PARTICLE_KINDS or job.get('state') not in ('submitted','running','submitting','unknown'):continue
                updated=update(service,{kind:job},live,kind=kind)
                if updated!=job:record['job_history'][index]=updated;changed=True
        if changed:
            service._write(root,data)
        return service._view(project_id)


def status(service, project_id, series_id):
    with service.lock:
        root=service._root(project_id)
        data=service._read(root)
        row=next((row for row in data['series'] if row['id']==series_id),None)
        if row is None:
            raise InputError('Unknown tilt series.')
        if row.get('ctf_redo'):
            # Publish a completed replacement before exposing completed CTF.
            from .ctf_redo import status as redo_status
            redo_status(service,project_id,series_id)
            data=service._read(root)
            row=next(row for row in data['series'] if row['id']==series_id)
        job=refresh(service,row)
        if row.get('preprocessing') and job != row['preprocessing']:
            row['preprocessing']=job
            service._write(root,data)
        return dict(job,tilt_display_version=2)


def plane_png(service, project_id, series_id, index, rotation_deg=None,flip_y=False,
              landscape=False,df_hand=None,axis_deg=None,preview_geometry=False):
    import mrcfile
    import numpy as np
    from PIL import Image
    if rotation_deg is None:
        rotation_deg = service.get(project_id).get('orientation', {}).get('rotation_deg', 0)
    try:
        rotation_deg = float(rotation_deg)
    except (ValueError, TypeError):
        raise InputError('Invalid preview rotation.') from None
    if not math.isfinite(rotation_deg) or abs(rotation_deg) > 360:
        raise InputError('Preview rotation out of range.')
    job=status(service,project_id,series_id)
    if job.get('state')!='completed':
        raise InputError('Preprocessing has not completed.')
    stack=service.path(job['stack'])
    if not layout_owned(service._root(project_id),stack):
        raise InputError('Output must be inside the project jobs folder.')
    with mrcfile.mmap(stack,mode='r',permissive=False) as handle:
        data = handle.data if handle.data.ndim==3 else handle.data[None]
        if not 0<=index<len(data):
            raise InputError('Tilt index out of range.')
        plane=np.array(data[index],dtype=np.float32)
    valid=plane[np.isfinite(plane)]
    if not len(valid):
        raise InputError('Output plane has no finite pixels.')
    lo,hi=np.percentile(valid,[1,99])
    image=np.clip((np.nan_to_num(plane,nan=lo)-lo)/max(float(hi-lo),1e-12),0,1)
    from .tilt_display import orient_image
    background=float(np.clip((valid.mean()-lo)/max(float(hi-lo),1e-12),0,1))
    if landscape:
        from .ctf_view import ctf_png
        # Attach the saved detector field to the unrotated image first.
        # The resulting RGB raster has exactly one display transform.
        colour=Image.open(io.BytesIO(ctf_png(service,project_id,series_id,index,'landscape',df_hand=df_hand,axis_deg=axis_deg,preview_geometry=preview_geometry))).convert('RGB')
        image=.5*image[...,None]+.5*np.asarray(colour,dtype=np.float32)/255
        background=.5*background+.5*np.array([230,230,235],dtype=np.float32)/255
    image=orient_image(image,rotation_deg,flip_y,background)
    rendered=Image.fromarray((np.clip(image,0,1)*255).astype('uint8'))
    stream=io.BytesIO()
    rendered.save(stream,format='PNG')
    return stream.getvalue()
