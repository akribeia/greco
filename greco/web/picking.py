"""Template import, GPU submission and persisted Greco Picker results."""
from greco.project_layout import picker_assets, join as storage_path, owned as layout_owned
import hashlib
import json
import math
from pathlib import Path
import uuid
from .projects import InputError
from .preprocessing import refresh,render_script,command_result
from ..picker.prepare import inspect_template,inspect_mask
MAX_TEMPLATE_BYTES=128*1024*1024
MAX_SERVER_TEMPLATE_BYTES=2*1024**3
DEFAULTS=dict(template_pixel_size_A=None,flip_z=False,invert_contrast=False,
    particle_diameter_A=300.,template_mask_diameter_A=None,angular_search_deg=10.,max_particles=1000,
    candidate_count=2000,high_pass_A=400.,low_pass_A=None,matching_mode='exhaustive',threshold_factor=1.)

def check_particle_dependants(data,series_ids):
    """Never replace picking files already referenced by a particle assembly."""
    selected=set(series_ids)
    rows={r['id']:r for r in data['series']}
    for assembly in data.get('particle_assemblies',[]):
        for member in assembly.get('tomograms',[]):
            sid=member.get('series_id')
            if sid not in selected:continue
            same_assembly=assembly.get('pick_assembly_id')==data.get('active_pick_assembly')
            job=rows[sid].get('picking',{})
            work=job.get('work')
            shared=bool(work and any(value and (Path(value)==Path(work) or Path(value).is_relative_to(Path(work)))
                for value in (member.get('picking_work'),member.get('star'))))
            if same_assembly or shared:
                raise InputError(rows[sid]['name']+': picks are used by '+assembly.get('name','a particle assembly')+'. Create a new pick assembly to run another picking.')

def upload_template(service, project_id, filename, payload):
    if not payload or len(payload) > MAX_TEMPLATE_BYTES:
        raise InputError('Template exceeds the 128 MB upload limit.')
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        folder = picker_assets(root, data)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ('.upload-' + uuid.uuid4().hex + '.mrc')
        path.write_bytes(payload)
        try:
            metadata = inspect_template(path)
        except (ValueError, OSError) as error:
            path.unlink()
            raise InputError(str(error)) from None
        target = folder / 'template.mrc'
        path.replace(target)
        path = target
        template = dict(metadata, path=str(path), name=str(filename).replace('\\', '/').rsplit('/', 1)[-1], sha256=hashlib.sha256(payload).hexdigest())
        data.setdefault('picker_template_history', []).append(template)
        data['picker_template'] = template
        service._write(root, data)
        return template

def select_template(service, project_id, source_path, *, asset='template', assembly_id=None):
    """Snapshot a server-side template without transferring it via the browser."""
    source = service.path(source_path)
    if asset not in ('template', 'mask'):
        raise InputError('Invalid picker asset.')
    if not source.is_file() or source.suffix.lower() not in ('.mrc', '.map', '.mrcs'):
        raise InputError('Select an existing MRC template file on the server.')
    if not 0 < source.stat().st_size <= MAX_SERVER_TEMPLATE_BYTES:
        raise InputError('Server template or mask must be between 1 byte and 2 GiB.')
    with service.lock:
        root = service._root(project_id)
        data = service._read(root)
        if assembly_id and data.get('active_pick_assembly') != assembly_id:
            raise InputError('Pick assembly changed; reopen its form.')
        folder = picker_assets(root, data)
        folder.mkdir(parents=True, exist_ok=True)
        if not folder.resolve().is_relative_to(root):
            raise InputError('Invalid template folder.')
        path = folder / ('.upload-' + uuid.uuid4().hex + '.mrc')
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open('rb') as incoming, path.open('xb') as target:
                for chunk in iter(lambda : incoming.read(1024 * 1024), b''):
                    size += len(chunk)
                    if size > MAX_SERVER_TEMPLATE_BYTES:
                        raise InputError('Server template or mask exceeds the 2 GiB limit.')
                    target.write(chunk)
                    digest.update(chunk)
            metadata = (inspect_template if asset == 'template' else inspect_mask)(path)
        except (ValueError, OSError, InputError) as error:
            path.unlink(missing_ok=True)
            raise InputError(str(error)) from None
        target = folder / (asset + '.mrc')
        path.replace(target)
        path = target
        template = dict(metadata, path=str(path), source_path=str(source), name=source.name, sha256=digest.hexdigest(), source='server')
        data.setdefault('picker_' + asset + '_history', []).append(template)
        data['picker_' + asset] = template
        service._write(root, data)
        return template

def validate_options(raw):
    if not isinstance(raw,dict) or set(raw)-set(DEFAULTS):raise InputError('Unknown picker parameters.')
    options=dict(DEFAULTS,**raw)
    if options['matching_mode'] not in ('fast','exhaustive'):
        raise InputError('Picker mode must be Fast or Exhaustive.')
    for key in ('flip_z','invert_contrast'):
        if not isinstance(options[key],bool):raise InputError('Invalid picker checkbox.')
    for key in ('template_pixel_size_A','particle_diameter_A','template_mask_diameter_A','angular_search_deg','max_particles','candidate_count','high_pass_A','low_pass_A','threshold_factor'):
        value=options[key]
        if key in ('high_pass_A','low_pass_A','template_mask_diameter_A') and value in ('',None):options[key]=None;continue
        if isinstance(value,bool):raise InputError('Invalid picker value: '+key)
        try:value=float(value)
        except (ValueError,TypeError):raise InputError('Enter a valid '+key) from None
        if not math.isfinite(value) or value<=0:raise InputError('Picker values must be finite and positive: '+key)
        options[key]=value
    if not 5<=options['angular_search_deg']<=30:raise InputError('Angular sampling must be between 5 and 30 degrees.')
    for key in ('max_particles','candidate_count'):
        if options[key]!=int(options[key]) or options[key]>10000:raise InputError('Particle/candidate counts must be integers from 1 to 10000.')
        options[key]=int(options[key])
    if options['matching_mode']=='fast' and options['max_particles']>options['candidate_count']:raise InputError('Max particles cannot exceed candidate count.')
    if options['low_pass_A'] and options['high_pass_A'] and options['low_pass_A']>=options['high_pass_A']:
        raise InputError('High-pass wavelength must exceed low-pass wavelength.')
    return options

def validate_asset_path(root, data, path, asset, assembly_id=None):
    """Accept only the selected category's owned snapshot ."""
    context = dict(data)
    if assembly_id:
        context['active_pick_assembly'] = assembly_id
    folder = picker_assets(root, context).resolve()
    path = Path(path).resolve()
    valid = path == folder / (asset + '.mrc')
    if not valid:
        raise InputError('Invalid picker ' + asset + ' path.')

def asset_png(service,project_id,asset,assembly_id=None,index=None,axis='z',full=False):
    """Small, on-demand central slice of the immutable input snapshot."""
    import io,mrcfile
    from .reconstruction import volume_plane_image
    if asset not in ('template','mask'):raise InputError('Invalid picker asset.')
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        record=data
        if assembly_id:
            record=next((a for a in data.get('pick_assemblies',[]) if a['id']==assembly_id),None)
            if record is None:raise InputError('Unknown pick assembly.')
        metadata=record.get('picker_'+asset)
        if not metadata:raise InputError('No '+asset+' selected.')
        path=service.path(metadata['path'])
        validate_asset_path(root,data,path,asset,assembly_id)
    if axis not in ('x','y','z'):raise InputError('Unknown slice axis.')
    with mrcfile.open(path,header_only=True) as handle:
        size=int(getattr(handle.header,{'x':'nx','y':'ny','z':'nz'}[axis]))
    if index is None:index=size//2
    if not 0<=index<size:raise InputError('Slice index out of range.')
    if asset=='mask':
        import numpy as np
        from PIL import Image
        with mrcfile.mmap(path,mode='r') as handle:
            plane=np.take(handle.data,index,axis={'z':0,'y':1,'x':2}[axis])
            image=Image.fromarray(np.rint(255*np.clip(plane,0,1)).astype(np.uint8))
    else:image=volume_plane_image(path,index,axis)
    if not full:image.thumbnail((180,180))
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()

def submit(service,project_id,series_ids,raw,*,queue_only=False,assembly_id=None,only_missing=False):
    from .queue import dispatch,prepare_work,store_job
    from .picking_source import selected,source_job,volume_context
    options=validate_options(raw)
    if not isinstance(series_ids,list) or not series_ids or len(set(series_ids))!=len(series_ids):raise InputError('Select distinct tilt series.')
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        if assembly_id and data.get('active_pick_assembly')!=assembly_id:raise InputError('Pick assembly changed; reopen its form.')
        slurm=data.get('slurm',{})
        if not slurm.get('ready'):raise InputError('Save cluster settings first.')
        template=data.get('picker_template')
        if not template:raise InputError('Select a template first.')
        template_path=service.path(template['path'])
        validate_asset_path(root,data,template_path,'template')
        inspect_template(template_path)
        mask=data.get('picker_mask')
        if not mask:raise InputError('Select a template mask MRC first.')
        mask_path=service.path(mask['path'])
        validate_asset_path(root,data,mask_path,'mask')
        mask_metadata=inspect_mask(mask_path)
        if mask_metadata['shape']!=template['shape']:raise InputError('Mask must have the same box size as the template.')
        if mask_metadata['pixel_size_A'] and template.get('pixel_size_A') and not math.isclose(mask_metadata['pixel_size_A'],template['pixel_size_A'],rel_tol=1e-4):
            raise InputError('Mask and template must have the same pixel size.')
        options.pop('template_mask_diameter_A',None)
        rows={row['id']:row for row in data['series']}
        if any(i not in rows for i in series_ids):raise InputError('Unknown tilt series.')
        from .projects import active_series_ids
        series_ids=active_series_ids(rows,series_ids,bulk=only_missing)
        inputs={};parents={}
        import mrcfile
        import numpy as np
        if only_missing:
            from .queue import needs_submission
            series_ids=[i for i in series_ids if needs_submission(rows[i].get('picking',{}))]
            if not series_ids:
                result=service._view(project_id);result['submission_errors']=[];return result
        check_particle_dependants(data,series_ids)
        from .slurm_arrays import live_states_for
        live=live_states_for([source_job(rows[i]) for i in series_ids],command_result) if only_missing and not queue_only else None
        for identifier in series_ids:
            row=rows[identifier]
            active=refresh(service,{'preprocessing':row.get('picking',{})})
            if active['state'] in ('queued','submitting','submitted','running','unknown'):raise InputError(row['name']+': picking already active.')
            source_kind=selected(row)
            parent_row={'preprocessing':source_job(row)}
            reconstruction=refresh(service,parent_row,live) if live is not None else refresh(service,parent_row)
            if (queue_only and row.get('queued_jobs',{}).get(source_kind)) or reconstruction['state'] in ('submitted','running'):
                parent=row.get('queued_jobs',{}).get(source_kind) if queue_only else reconstruction
                parent=parent or reconstruction
                work=service.path(parent['work'])
                if not layout_owned(root,work):raise InputError('Invalid reconstruction dependency path.')
                if not queue_only and (not parent.get('job_id') or parent.get('host','')!=slurm.get('submission_host','')):raise InputError('Reconstruction dependency needs a Slurm ID on the same host.')
                inputs[identifier]=dict(reconstruction_dependency=True,reconstruction_work=str(work),picking_source_kind=source_kind);parents[identifier]=parent
                continue
            if reconstruction['state']!='completed':raise InputError(row['name']+': complete reconstruction first.')
            work=service.path(reconstruction['work'])
            if not layout_owned(root,work):raise InputError('Invalid reconstruction output.')
            volume=service.path(reconstruction['volumes']['original'])
            if not volume.is_relative_to(storage_path(work, 'output')):raise InputError('Invalid tomogram path.')
            with mrcfile.open(volume,header_only=True) as handle:
                shape=[int(handle.header.nz),int(handle.header.ny),int(handle.header.nx)]
                voxel=float(handle.voxel_size.x)
                if not np.isfinite(voxel) or voxel<=0:raise InputError('Tomogram pixel size is missing.')
                if not np.allclose([handle.voxel_size.y,handle.voxel_size.z],voxel):raise InputError('Tomogram voxels must be isotropic.')
                if (int(handle.header.mapc),int(handle.header.mapr),int(handle.header.maps))!=(1,2,3):raise InputError('Tomogram must use standard XYZ MRC axes.')
            size=int(round(template['shape'][0]*options['template_pixel_size_A']/voxel))
            if size<4 or size>128 or size>=min(shape):raise InputError('Template size after resampling must be 4–128 voxels and fit inside the tomogram.')
            if options['particle_diameter_A']/voxel>=min(shape):raise InputError('Particle diameter exceeds tomogram dimensions.')
            if options['low_pass_A'] is not None and options['low_pass_A']<2*voxel:raise InputError('Low-pass resolution cannot exceed the tomogram Nyquist limit.')
            # Use the angles that actually contributed to THIS reconstruction.
            try:source_config,summary,result,provenance=volume_context(work)
            except (ValueError,OSError,KeyError) as error:raise InputError(str(error)) from error
            if provenance['picking_source_kind']!=source_kind:raise InputError('Selected picking source does not match its output.')
            rawtlt=service.path(source_config['inputs']['rawtlt'])
            all_angles=np.loadtxt(rawtlt,ndmin=1)
            indices=np.asarray(summary['active_indices'])
            if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=len(all_angles)):
                raise InputError('Invalid reconstruction active indices.')
            angles=all_angles[indices]
            if len(angles)<2 or not np.isfinite(angles).all():raise InputError('Invalid reconstructed tilt angles.')
            inputs[identifier]=dict(volume=str(volume),voxel_size_A=voxel,tilt_angles_deg=angles.tolist(),**provenance)
            if result.get('ctf_template_metadata') is not None:
                inputs[identifier].update(tilt_series_metadata=result['ctf_template_metadata'],per_tilt_weighting=True)
        data['picking_options']=options;errors=[];array_entries=[]
        for identifier in series_ids:
            row=rows[identifier];work=prepare_work(service,root,row,'picking',queue_only,allowed_dependencies=('preprocessing',selected(row)) if identifier in parents else ())
            config=dict(inputs[identifier],template=template,mask=mask,options=options,series_id=identifier,series_name=row['name'],pick_assembly_id=data.get('active_pick_assembly'))
            path=storage_path(work, 'config.json');path.write_text(json.dumps(config,indent=2))
            resources=dict(cpus=data.get('preprocessing_options',{}).get('cpus',4),gpus=1)
            (storage_path(work, 'job.sbatch')).write_text(render_script(slurm['template'],work,path,resources,'greco.web.picking_worker'))
            job=dict(state='submitted',work=str(work),host=slurm.get('submission_host',''),options=options,template=template,mask=mask,volume=inputs[identifier].get('volume'))
            job['pick_assembly_id']=data.get('active_pick_assembly')
            job['picking_source_kind']=inputs[identifier]['picking_source_kind']
            job['picking_step_id']=row.get('picking_step_id')
            if only_missing and not queue_only:
                array_entries.append((row,job,[parents[identifier]['job_id']] if identifier in parents else ()));continue
            try:
                dispatch(job,queue_only,command_result,dependencies=[parents[identifier]['job_id']] if identifier in parents and not queue_only else ())
            except InputError as error:job.update(state='failed',error=str(error));errors.append(row['name']+': '+str(error))
            store_job(row,'picking',job,queue_only);service._write(root,data)
        if array_entries:
            from .slurm_arrays import submit_arrays
            errors+=submit_arrays(service,root,data,'picking',array_entries,command_result)
        project=service._view(project_id);project['submission_errors']=errors;return project

def status(service,project_id,series_id=None,assembly_id=None):
    with service.lock:
        root=service._root(project_id);data=service._read(root);jobs={};changed=False
        other=None
        if assembly_id and assembly_id!=data.get('active_pick_assembly'):
            other=next((a for a in data.get('pick_assemblies',[]) if a['id']==assembly_id),None)
            if other is None:raise InputError('Unknown pick assembly.')
        if series_id and not any(row['id']==series_id for row in data['series']):raise InputError('Unknown tilt series.')
        for row in data['series']:
            if series_id and row['id']!=series_id:continue
            if other is not None:
                identifier=row['id'];row=other.setdefault('series',{}).setdefault(identifier,{})
                row['id']=identifier
            job=refresh(service,{'preprocessing':row.get('picking',{})});jobs[row['id']]=job
            if row.get('picking') and row['picking']!=job:row['picking']=job;changed=True
            if job['state']=='completed' and row.get('pick_count')!=job['pick_count']:
                row['pick_count']=job['pick_count'];changed=True
        if changed:service._write(root,data)
        return jobs

def examples_png(service,project_id,series_id,assembly_id=None):
    job=status(service,project_id,series_id,assembly_id)[series_id]
    if job['state']!='completed':raise InputError('Picking has not completed.')
    root=service._root(project_id)
    path=service.path(job['examples'])
    if not layout_owned(root,path) or path!=storage_path(storage_path(Path(job['work']), 'output'), 'examples.png'):raise InputError('Invalid particle examples path.')
    return path.read_bytes()

def _viewer_job(service, project_id, series_id, assembly_id=None):
    job = status(service, project_id, series_id, assembly_id)[series_id]
    if job['state'] != 'completed':
        raise InputError('Picking has not completed.')
    root = service._root(project_id)
    volume = service.path(job['volume'])
    coordinates = service.path(job['star'])
    work = service.path(job['work'])
    job_root = Path(root) / 'picking'
    if not layout_owned(root, volume) or not work.is_relative_to(job_root) or coordinates != storage_path(storage_path(work, 'output'), 'particles.star'):
        raise InputError('Invalid picking viewer paths.')
    return (job, volume, coordinates)

def viewer_info(service,project_id,series_id,assembly_id=None):
    import mrcfile
    job,volume,coordinates=_viewer_job(service,project_id,series_id,assembly_id)
    from ..picking_star import coordinates_xyz
    picks=coordinates_xyz(coordinates,volume).tolist()
    with mrcfile.open(volume,header_only=True) as handle:
        shape=[int(handle.header.nz),int(handle.header.ny),int(handle.header.nx)]
        voxel=float(handle.voxel_size.x)
    return dict(shape=shape,pick_count=job['pick_count'],picks_xyz=picks,voxel_size_A=voxel,
                radius_px=float(job['options']['particle_diameter_A'])/(2*voxel),
                volumes={key:str(path) for key,path in _viewer_volumes(service,volume).items()})

def _viewer_volumes(service, volume):
    """Only use denoised variants belonging to the reconstruction actually picked."""
    paths = {'original': volume}
    for method in ('tinycare',):
        import re
        name=re.fullmatch(r'(.+_r[0-9]+)(_bin[0-9]+\.mrc)',volume.name)
        if name is None:continue
        candidate=volume.with_name(name[1]+'_'+method+name[2])
        if not candidate.is_file():
            continue
        candidate = service.path(str(candidate))
        if candidate.parent != volume.parent:
            continue
        import mrcfile, numpy as np
        with mrcfile.open(volume, header_only=True) as source, mrcfile.open(candidate, header_only=True) as target:
            if tuple((int(source.header[k]) for k in ('nx', 'ny', 'nz'))) != tuple((int(target.header[k]) for k in ('nx', 'ny', 'nz'))):
                continue
            if not np.allclose(list(source.voxel_size.tolist()), list(target.voxel_size.tolist())):
                continue
            if (int(target.header.mapc), int(target.header.mapr), int(target.header.maps)) != (1, 2, 3):
                continue
        paths[method] = candidate
    return paths

def viewer_png(service,project_id,series_id,index,axis='z',background=.2,autocontrast=False,contrast=1,method='original',brightness=0,assembly_id=None):
    import io,mrcfile
    import numpy as np
    from PIL import Image
    from .reconstruction import volume_plane_image
    job,volume,coordinates=_viewer_job(service,project_id,series_id,assembly_id)
    if method!='original':
        path=_viewer_volumes(service,volume).get(method)
        if path is None:raise InputError('This volume variant was not generated for the picked tomogram.')
        volume=path
    background=float(background)
    if not np.isfinite(background) or not 0<=background<=1:raise InputError('Background must be between 0 and 1.')
    image=np.array(volume_plane_image(volume,index,axis,autocontrast,contrast,brightness,denoised=method=='tinycare'))
    if background==1:
        stream=io.BytesIO();Image.fromarray(image).save(stream,format='PNG');return stream.getvalue()
    from ..picking_star import coordinates_xyz
    xyz=coordinates_xyz(coordinates,volume)
    with mrcfile.open(volume,header_only=True) as handle:voxel=float(handle.voxel_size.x)
    radius=float(job['options']['particle_diameter_A'])/(2*voxel)
    if not np.isfinite(radius) or radius<=0:raise InputError('Invalid particle mask radius.')
    normal,horizontal,vertical={'z':(2,0,1),'y':(1,0,2),'x':(0,1,2)}[axis]
    mask=np.zeros(image.shape,bool)
    for point in xyz:
        if not np.isfinite(point).all():continue
        squared=radius**2-(point[normal]-index)**2
        if squared<0:continue
        r=np.sqrt(squared);x=point[horizontal];y=point[vertical]
        x0=max(0,int(np.floor(x-r)));x1=min(image.shape[1],int(np.ceil(x+r))+1)
        y0=max(0,int(np.floor(y-r)));y1=min(image.shape[0],int(np.ceil(y+r))+1)
        if x0>=x1 or y0>=y1:continue
        yy,xx=np.ogrid[y0:y1,x0:x1]
        mask[y0:y1,x0:x1]|=(xx-x)**2+(yy-y)**2<=squared
    if len(xyz):image=np.where(mask,image,np.rint(image*background)).astype(np.uint8)
    stream=io.BytesIO();Image.fromarray(image).save(stream,format='PNG');return stream.getvalue()
