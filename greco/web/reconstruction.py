"""Project-scoped Greco Optimizer reconstruction jobs and read-only tomogram slices."""
from greco.project_layout import join as storage_path, owned as layout_owned

import io
import json
from pathlib import Path

from .projects import InputError
from .preprocessing import refresh,render_script,command_result

METHODS=('tinycare',)

# Jobs record the shared application source identity.


def capabilities():
    return dict(optimizer='greco_optimizer',optimizer_name='Greco Optimizer',optimizer_version=1,
                entry_search_sigma_default_px=8.,integrated_smoothness=True,final_binning=True,rescue_radius_px=16,linear_refinement_steps=True,advanced_reconstruction=True)


def main_revision():
    from ..provenance import source_identity
    return source_identity()['source_sha256']


def submit(service,project_id,series_ids,denoisers,optimizer='greco_optimizer',entry_search_sigma_px=8.,*,queue_only=False,only_missing=False,tomogram_thickness_percent=50.,smoothness_strength=None,rescue_tilts=True,final_binning=None,advanced_options=None):
    from .queue import dispatch,prepare_work,store_job
    from greco.tomogram_thickness import validate_percent
    try:tomogram_thickness_percent=validate_percent(tomogram_thickness_percent)
    except ValueError as error:raise InputError(str(error)) from None
    if optimizer!='greco_optimizer':
        raise InputError('Unknown reconstruction optimizer.')
    if final_binning is not None and (type(final_binning) is not int or final_binning not in (1,2,4,8,16)):
        raise InputError('Final binning must be 1, 2, 4, 8 or 16.')
    from ..optimizer.advanced_options import resolve
    try:advanced=resolve(advanced_options)
    except ValueError as error:raise InputError(str(error)) from None
    if advanced['advanced_mode']:rescue_tilts=True
    alignment=None
    if smoothness_strength is None:smoothness_strength=0.
    if smoothness_strength is not None:
        from .alignment_options import validate
        alignment=validate(smoothness_strength,rescue_tilts)
    if any(advanced[k] for k in ('refine_common_axis','refine_per_view_axis','refine_tilt_angles')) and alignment['smoothness_strength']!=0:
        raise InputError('Smoothness must be zero when refining the tilt axis.')
    from greco.optimizer.entry import validate_sigma
    try:entry_search_sigma_px=validate_sigma(entry_search_sigma_px)
    except (ValueError,TypeError,OverflowError):
        raise InputError('Entry sigma must be finite and positive.') from None
    if not isinstance(denoisers,list) or len(set(denoisers))!=len(denoisers) or any(m not in METHODS for m in denoisers):
        raise InputError('Unknown TinyCARE denoising selection.')
    if not isinstance(series_ids,list) or not series_ids or len(set(series_ids))!=len(series_ids):
        raise InputError('Select distinct tilt series.')
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        from .reconstruction_assemblies import current
        record=current(data)
        if 'reconstruction_assemblies' in data and record is None:raise InputError('Create a reconstruction first.')
        if record and (record['mode']=='advanced')!=advanced['advanced_mode']:raise InputError('The reconstruction mode is fixed; create another reconstruction to change it.')
        slurm=data.get('slurm',{})
        if not slurm.get('ready'):raise InputError('Save cluster settings first.')
        orientation=data.get('orientation')
        if not orientation:raise InputError('Apply the tilt angle / Flip Y to the entire project before reconstructing.')
        orientation=dict(orientation)
        if advanced['advanced_mode']:
            orientation['df_hand']=advanced['defocus_handedness']
            if advanced['flip_tomogram_z']:
                orientation['rotation_deg']=(float(orientation['rotation_deg'])+360.)%360.-180.
        rows={row['id']:row for row in data['series']}
        if any(i not in rows for i in series_ids):raise InputError('Unknown tilt series.')
        from .projects import active_series_ids
        series_ids=active_series_ids(rows,series_ids,bulk=only_missing)
        revision=main_revision()
        inputs={};deferred=set();parents={}
        if only_missing:
            from .queue import needs_submission
            series_ids=[i for i in series_ids if needs_submission(rows[i].get('reconstruction',{}))]
            if not series_ids:
                result=service._view(project_id);result['submission_errors']=[];return result
        from .slurm_arrays import live_states_for
        live=live_states_for([rows[i].get('preprocessing',{}) for i in series_ids],command_result) if only_missing and not queue_only else None
        for identifier in series_ids:
            row=rows[identifier]
            active=refresh(service,{'preprocessing':row.get('reconstruction',{})})
            if active['state'] in ('queued','submitting','submitted','running','unknown'):raise InputError(row['name']+': reconstruction already active.')
            prep=refresh(service,row,live) if live is not None else refresh(service,row)
            if prep.get('ctf_redo_active'):raise InputError(row['name']+': wait for Redo CTF to finish before reconstruction.')
            queued_prep=row.get('queued_jobs',{}).get('preprocessing')
            if (queue_only and queued_prep) or (prep['state'] in ('submitted','running') and prep.get('job_id')):
                planned=queued_prep or prep
                binning=planned.get('options',{}).get('output_binning',8)
                if advanced['advanced_mode'] and binning!=8:raise InputError('Advanced mode requires preprocessing at bin8.')
                if 4096*planned.get('options',{}).get('eer_upscaling',1)/binning>1024:
                    raise InputError('This output binning would create an oversized cubic tomogram. Preprocess at bin8 or a coarser bin first.')
                if row.get('found',row.get('expected',7))<7:raise InputError('Greco Optimizer requires at least seven tilt planes.')
                parent=service.path(planned['work'])
                if not layout_owned(root,parent):raise InputError('Invalid preprocessing dependency path.')
                if not queue_only and planned.get('host','')!=slurm.get('submission_host',''):raise InputError('Preprocessing dependency must use the same Slurm host.')
                output=storage_path(parent, 'output')
                inputs[identifier]=dict(stack=str(storage_path(output, f'ctf/reliability_bin{binning}.mrcs')),rawtlt=str(storage_path(output, f'greco_raw_bin{binning}.rawtlt')),frozen_shifts=str(storage_path(output, f'alignment/tilts.star')),binning=binning)
                parents[identifier]=planned;deferred.add(identifier)
                if queue_only:inputs[identifier]=None
                continue
            if prep['state']!='completed':raise InputError(row['name']+': complete preprocessing first.')
            work=service.path(prep['work'])
            if not layout_owned(root,work):raise InputError('Invalid preprocessing output.')
            binning=prep.get('output_binning',prep.get('options',{}).get('output_binning',8))
            if advanced['advanced_mode'] and binning!=8:raise InputError('Advanced mode requires preprocessing at bin8.')
            output=storage_path(work, 'output')
            paths=dict(stack=storage_path(output, f'ctf/reliability_bin{binning}.mrcs'),
                       rawtlt=storage_path(output, f'greco_raw_bin{binning}.rawtlt'),
                       frozen_shifts=storage_path(output, f'alignment/tilts.star'))
            for path in paths.values():service.path(str(path))
            import mrcfile
            with mrcfile.mmap(paths['stack'],mode='r') as handle:
                if handle.data.ndim!=3 or handle.data.shape[0]<7 or handle.data.shape[1]!=handle.data.shape[2]:
                    raise InputError('Greco Optimizer requires at least seven square tilt planes.')
                if handle.data.shape[1]>1024:
                    raise InputError('This output binning would create an oversized cubic tomogram. Preprocess at bin8 or a coarser bin first.')
            inputs[identifier]=dict(**{key:str(value) for key,value in paths.items()},binning=binning)
        errors=[];array_entries=[];data['reconstruction_denoisers']=denoisers
        data['reconstruction_advanced_options']=advanced['selected']
        data['reconstruction_optimizer']=optimizer
        data['reconstruction_entry_sigma_px']=entry_search_sigma_px
        data['reconstruction_thickness_percent']=tomogram_thickness_percent
        data['reconstruction_final_binning']=final_binning
        if alignment is not None:data['reconstruction_alignment_options']=alignment
        for identifier in series_ids:
            row=rows[identifier];work=prepare_work(service,root,row,'reconstruction',queue_only,allowed_dependencies=('preprocessing',) if identifier in deferred else ())
            work=service.path(str(work))
            options=dict(cpus=data.get('preprocessing_options',{}).get('cpus',4),gpus=1)
            config=dict(advanced_options=advanced,inputs=inputs[identifier],orientation=orientation,output=str(storage_path(work, 'output')),
                        tomogram_thickness_percent=tomogram_thickness_percent,final_binning=final_binning,
                        preprocessing_dependency=identifier in deferred,
                        final_reconstruction_protocol='native-wiener-sz-no-reliability-canonical-z-v1',
                        final_wbp='standard',
                        main_revision=revision,denoisers=denoisers,optimizer=optimizer,
                        optimizer_version=1,
                        entry_search_sigma_px=entry_search_sigma_px,
                        alignment_code_origin='bundled production engine',
                        protocol="Greco Optimizer")
            if alignment is not None:
                config['alignment_options']=alignment
            path=storage_path(work, 'config.json');path.write_text(json.dumps(config,indent=2))
            (storage_path(work, 'job.sbatch')).write_text(render_script(slurm['template'],work,path,options,'greco.web.reconstruction_worker'))
            job=dict(advanced_options=advanced,state='submitted',work=str(work),host=slurm.get('submission_host',''),main_revision=revision,orientation=orientation,denoisers=denoisers,optimizer=optimizer,
                     tomogram_thickness_percent=tomogram_thickness_percent,final_binning=final_binning,
                     optimizer_version=1,
                     entry_search_sigma_px=entry_search_sigma_px)
            if alignment is not None:job['alignment_options']=alignment
            if only_missing and not queue_only:
                array_entries.append((row,job,[parents[identifier]['job_id']] if identifier in parents else ()));continue
            try:
                dispatch(job,queue_only,command_result,dependencies=[parents[identifier]['job_id']] if identifier in parents and not queue_only else ())
            except InputError as error:
                job.update(state='failed',error=str(error));errors.append(row['name']+': '+str(error))
            store_job(row,'reconstruction',job,queue_only)
            service._write(root,data)
        if array_entries:
            from .slurm_arrays import submit_arrays
            errors+=submit_arrays(service,root,data,'reconstruction',array_entries,command_result)
        result=service._view(project_id);result['submission_errors']=errors;return result


def landmark_preview(service,root,job):
    """Describe the latest complete landmark projections in tilt-angle order."""
    import re
    import mrcfile
    import numpy as np
    from .metadata_cache import read_json
    if not job.get('work') or job.get('state') not in ('completed','partial'):return None
    work=service.path(job['work'])
    checkpoint=work/'alignment/state.npz'
    if not layout_owned(root,checkpoint.resolve()) or not checkpoint.is_file():return None
    def read_angles(path):
        with np.load(path,allow_pickle=False) as state:return state.get('refined_angles_deg',state['angles_deg']).tolist()
    def read_shape(path):
        with mrcfile.open(path,header_only=True) as handle:
            return [int(handle.header.nz),int(handle.header.ny),int(handle.header.nx)]
    cache=getattr(service,'metadata_cache',None)
    def read(path,loader):return cache.read(path,loader) if cache is not None else loader(path)
    try:angles=np.asarray(read(checkpoint,read_angles),dtype=float)
    except (OSError,ValueError,KeyError):return None
    if angles.ndim!=1 or not np.isfinite(angles).all():return None
    candidates=[]
    fine=work/'registration/fine_scale'
    if fine.is_dir():candidates.append((2,0,fine))
    for directory in (work/'registration').glob('*'):
        match=re.fullmatch(r'(extra_round|stage)_(\d+)',directory.name)
        if match:candidates.append((match[1]=='extra_round',int(match[2]),directory))
    for _,_,directory in sorted(candidates,reverse=True):
        stack=directory/'landmarks_projection.mrcs';geometry=directory/'output_geometry.json'
        if not all(layout_owned(root,p.resolve()) and p.is_file() for p in (stack,geometry)):continue
        try:
            indices=read_json(service,geometry)['active_indices']
            shape=read_shape(stack)
            if len(indices)!=shape[0] or min(shape)<=0 or len(set(indices))!=len(indices):continue
            if any(type(i) is not int or not 0<=i<len(angles) for i in indices):continue
            order=sorted(range(len(indices)),key=lambda i:angles[indices[i]])
            stamp=stack.stat()
            return dict(path=str(stack),shape=shape,stage=directory.name,
                        angles=[float(angles[indices[i]]) for i in order],planes=order,
                        tilt_numbers=[indices[i]+1 for i in order],
                        revision=[stamp.st_size,stamp.st_mtime_ns])
        except (OSError,ValueError,KeyError,TypeError):continue
    return None


def refined_angle_table(service, root, job):
    """Read source-indexed fitted angles from this reconstruction's checkpoint."""
    import numpy as np
    if not job.get('work'):return []
    work=Path(job['work'])
    if not layout_owned(root,work):return []
    paths=[storage_path(work,'alignment/final_state.npz'),work/'alignment/final_state.npz']
    path=next((p for p in paths if p.is_file()),None)
    if path is None:return []
    def read(path):
        with np.load(path,allow_pickle=False) as state:
            if 'refined_angles_deg' not in state or 'axis_rotation_deg' not in state:return []
            nominal=np.asarray(state['angles_deg'],float)
            refined=np.asarray(state['refined_angles_deg'],float)
            axes=np.asarray(state['axis_rotation_deg'],float)
            if axes.ndim==0:axes=np.full(nominal.shape,float(axes))
            if nominal.ndim!=1 or nominal.shape!=refined.shape or nominal.shape!=axes.shape:return []
            if not all(np.isfinite(a).all() for a in (nominal,refined,axes)):return []
            active=set(np.asarray(state['active_indices'],int).tolist())
            return [dict(tilt=int(i)+1,alpha_nominal=float(nominal[i]),alpha_refined=float(refined[i]),beta_refined=float(axes[i]),included=i in active) for i in np.argsort(nominal)]
    try:
        from .metadata_cache import MetadataCache
        cache=getattr(service,'refined_angle_cache',None)
        if cache is None:
            cache=MetadataCache(limit=64)
            service.refined_angle_cache=cache
        return cache.read(path,read)
    except (OSError,ValueError,KeyError):return []



def refined_tilt_preview(root,job):
    import numpy as np
    work=Path(job.get('work',''))
    if not job.get('work') or not layout_owned(root,work):return None
    stack=storage_path(work,'alignment/aligned_bin8.mrcs')
    checkpoint=storage_path(work,'alignment/final_state.npz')
    if not stack.is_file() or not checkpoint.is_file():return None
    with np.load(checkpoint,allow_pickle=False) as state:
        active=np.asarray(state['active_indices'],int)
        angles=np.asarray(state.get('refined_angles_deg',state['angles_deg']),float)[active]
        order=np.argsort(angles)
        return dict(angles=angles[order].tolist(),planes=order.tolist(),
                    source_indices=active[order].tolist(),revision=checkpoint.stat().st_mtime_ns,
                    final_ctf=storage_path(work/'alignment','ctf_refinement.json').is_file())


def refined_tilt_png(service,project_id,series_id,index,mode=None):
    root=service._root(project_id)
    data=service._read(root)
    row=next((r for r in data['series'] if r['id']==series_id),None)
    if row is None:raise InputError('Unknown tilt series.')
    job=row.get('reconstruction',{})
    preview=refined_tilt_preview(root,job)
    if preview is None:raise InputError('Refined tilt series is not available yet.')
    if not 0<=index<len(preview['planes']):raise InputError('Tilt index out of range.')
    work=Path(job['work']);stack=storage_path(work,'alignment/aligned_bin8.mrcs')
    image=_refined_tilt_image(service,stack,index,preview)
    if mode:
        from .reconstruction_ctf import render
        return render(work,preview['source_indices'][index],mode,image)
    return image


def _refined_tilt_image(service,stack,index,preview):
    import numpy as np
    from PIL import Image
    from .curation_display import display_plane
    plane=display_plane(service,{'stack':str(stack)},preview['planes'][index],source_index=preview['source_indices'][index])
    if plane is None:raise InputError('Refined display geometry is unavailable.')
    values=plane[np.isfinite(plane)]
    if not values.size:raise InputError('No finite pixels in refined tilt.')
    lo,hi=np.percentile(values,[1,99])
    pixels=np.clip((np.nan_to_num(plane,nan=lo)-lo)/max(float(hi-lo),1e-12),0,1)
    stream=io.BytesIO();Image.fromarray((pixels*255).astype('uint8')).save(stream,format='PNG')
    return stream.getvalue()


def status(service,project_id,series_id):
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        job=refresh(service,{'preprocessing':row.get('reconstruction',{})})
        if row.get('reconstruction') and row['reconstruction']!=job:
            row['reconstruction']=job;service._write(root,data)
        # A completed rigid geometry preview belongs to this exact reconstruction.
        # Keep the derived variant out of the persisted original reconstruction.
        job=dict(job,volumes=dict(job.get('volumes',{})))
        job['volumes'].pop('geometry',None)
        for method in METHODS:job['volumes'].pop('geometry_'+method,None)
        job['volumes'].pop('rescue',None)
        geo=refresh(service,{'preprocessing':row.get('geometry_refinement',{})})
        if geo.get('state')=='completed' and geo.get('tomogram') and geo.get('source_reconstruction'):
            expected=storage_path(Path(job.get('work','')), 'alignment')
            sources={expected.resolve()}
            if Path(geo['source_reconstruction']).resolve() in sources:
                path=service.path(geo['tomogram'])
                if path==storage_path(service.path(geo['work']), 'output/tomogram_bin8.mrc'):
                    job['volumes']['geometry']=str(path)
                    job['volume_shapes']={'geometry':geo['preview_shape']}
                    for method in METHODS:
                        candidate=geo.get('volumes',{}).get(method)
                        if candidate and Path(candidate)==storage_path(service.path(geo['work']), f'output/tomogram_{method}.mrc') and Path(candidate).is_file():
                            job['volumes']['geometry_'+method]=str(service.path(candidate))
                            job['volume_shapes']['geometry_'+method]=geo['preview_shape']
        job['landmarks']=landmark_preview(service,root,job)
        job['refined_angles']=refined_angle_table(service,root,job)
        job['tilt_series']=refined_tilt_preview(root,job)
        return job


def landmark_intensity_window(service,path,automatic=False):
    """Use one stack-wide range; masked zero background is not image signal."""
    import mrcfile
    import numpy as np
    def measure(path):
        with mrcfile.mmap(path,mode='r') as handle:
            minimum=float(handle.header.dmin);maximum=float(handle.header.dmax)
            pixels=handle.data.reshape(-1)
            sample=np.asarray(pixels[::max(1,(pixels.size+262143)//262144)],dtype=float)
            sample=sample[np.isfinite(sample)]
            if not sample.size:raise InputError('No finite landmark pixels.')
            if not np.isfinite([minimum,maximum]).all() or maximum<=minimum:
                minimum,maximum=map(float,(sample.min(),sample.max()))
            signal=sample[sample!=0]
            low,high=np.percentile(signal if signal.size else sample,[.01,99.99])
            if high<=low:low,high=minimum,maximum
            else:
                # Leave headroom around rare landmark peaks instead of stretching
                # the supported density to pure black and white.
                margin=.1*(high-low)
                low=max(minimum,low-margin);high=min(maximum,high+margin)
            return ((minimum,maximum),(float(low),float(high)))
    cache=getattr(service,'metadata_cache',None)
    windows=cache.read(path,measure) if cache is not None else measure(path)
    return windows[bool(automatic)]


def volume_png(service,project_id,series_id,index,method='original',axis='z',autocontrast=False,contrast=1,brightness=0):
    
    
    
    job=status(service,project_id,series_id)
    if job['state'] not in ('completed','partial'):raise InputError('No reconstructed tomogram yet.')
    if axis not in ('z','y','x'):raise InputError('Unknown slice axis.')
    if method=='landmarks':
        preview=job.get('landmarks')
        if not preview:raise InputError('No landmark projections are available.')
        if axis!='z':raise InputError('Landmarks are indexed by tilt, not volume axis.')
        if not 0<=index<len(preview['planes']):raise InputError('Tilt index out of range.')
        path=preview['path'];index=preview['planes'][index]
    else:path=job.get('volumes',{}).get(method)
    if not path:raise InputError('This volume variant was not generated.')
    path=service.path(path)
    if not layout_owned(service._root(project_id),path):raise InputError('Invalid volume path.')
    window=landmark_intensity_window(service,path,autocontrast) if method=='landmarks' else None
    image=volume_plane_image(path,index,axis,autocontrast,contrast,brightness,intensity_window=window,denoised='tinycare' in method)
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def volume_plane_image(path,index,axis='z',autocontrast=False,contrast=1,brightness=0,intensity_window=None,*,denoised=False):
    """Read one MRC slice and apply the shared viewer contrast window."""
    import mrcfile
    import numpy as np
    from PIL import Image
    if axis not in ('z','y','x'):raise InputError('Unknown slice axis.')
    contrast=float(contrast)
    if not np.isfinite(contrast) or not .25<=contrast<=4:raise InputError('Contrast must be between 0.25 and 4.')
    brightness=float(brightness)
    if not np.isfinite(brightness) or not -1<=brightness<=1:raise InputError('Brightness must be between -1 and 1.')
    with mrcfile.mmap(path,mode='r') as handle:
        data=handle.data;dimension={'z':0,'y':1,'x':2}[axis]
        if not 0<=index<data.shape[dimension]:raise InputError('Slice index out of range.')
        plane=np.array(np.take(data,index,axis=dimension),dtype=np.float32)
    finite=plane[np.isfinite(plane)]
    if not finite.size:raise InputError('No finite volume pixels.')
    if intensity_window is None:lo,hi=np.percentile(finite,[1,99])
    else:
        lo,hi=map(float,intensity_window)
        if not np.isfinite([lo,hi]).all() or hi<lo:raise InputError('Invalid volume intensity window.')
    if autocontrast and intensity_window is None:
        if denoised:
            # A gentle window keeps TinyCARE tails and leaves grayscale headroom.
            lo,hi=np.percentile(finite,[.5,99.5])
            margin=.05*float(hi-lo)
            lo,hi=lo-margin,hi+margin
        else:
            # Robust window from actual float MRC pixels, not the clipped PNG.
            center=float(np.median(finite))
            sigma=1.4826*float(np.median(np.abs(finite-center)))
            if sigma>0:
                lo,hi=center-2*sigma,center+2*sigma
    center=(float(lo)+float(hi))/2-brightness*max(float(hi-lo),1e-12)
    half=max(float(hi-lo),1e-12)/(2*contrast)
    lo,hi=center-half,center+half
    image=(255*np.clip((np.nan_to_num(plane,nan=lo)-lo)/(hi-lo),0,1)).astype('uint8')
    return Image.fromarray(image)
