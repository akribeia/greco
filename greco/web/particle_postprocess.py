"""Owned, immutable Greco Python postprocessing jobs for refined half maps."""
from greco.project_layout import join as storage_path
import math
from pathlib import Path
from . import particle_jobs as jobs
from .assemblies import assembly_directory
from .projects import InputError


def validate_options(raw):
    allowed={'backend','from','mask_path','bfactor_mode','bfactor_A2','autob_lowres_A','autob_highres_A','low_pass_A','interpolate'}
    if not isinstance(raw,dict) or set(raw)-allowed:raise InputError('Unknown PostProcess options.')
    values=dict(backend='gpu',mask_path=None,bfactor_mode='auto',bfactor_A2=-20.,autob_lowres_A=10.,autob_highres_A=0.,low_pass_A=-1.,interpolate=False)
    values.update({k:v for k,v in raw.items() if k!='from'})
    if values['backend'] not in ('cpu','gpu'):raise InputError('Invalid PostProcess backend.')
    if values['bfactor_mode'] not in ('manual','auto','none'):raise InputError('Invalid sharpening mode.')
    if type(values['interpolate']) is not bool:raise InputError('Invalid FSC interpolation option.')
    if values['mask_path'] is not None and (not isinstance(values['mask_path'],str) or not values['mask_path'].strip()):raise InputError('Invalid mask path.')
    for key in ('bfactor_A2','autob_lowres_A','autob_highres_A','low_pass_A'):
        v=values[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise InputError('Invalid '+key+'.')
    if abs(values['bfactor_A2'])>10000 or values['autob_lowres_A']<=0 or values['autob_highres_A']<0 or values['low_pass_A']<-1:
        raise InputError('Invalid sharpening or filtering range.')
    if values['autob_highres_A'] and values['autob_highres_A']>=values['autob_lowres_A']:raise InputError('B-factor fit high resolution must be smaller than its low resolution.')
    return values


def half_map_inputs(service,root,record,job):
    import mrcfile
    import numpy as np
    work=service.path(job['work'])
    if not work.is_relative_to(assembly_directory(root,record)) or work.is_symlink():raise InputError('Invalid PostProcess source.')
    maps=job.get('unfiltered_volumes',{})
    paths=[];shape=None;pixel=None
    for h in (1,2):
        expected=storage_path(storage_path(work, 'output'), f'half{h}_unfiltered.mrc')
        path=service.path(maps.get(f'half{h}',''))
        if path!=expected or expected.is_symlink():raise InputError('Select a Refine3D with unfiltered half maps.')
        with mrcfile.mmap(path,mode='r') as m:
            sampling=np.array([float(m.voxel_size[a]) for a in 'xyz'])
            if m.data.ndim!=3 or len(set(m.data.shape))!=1 or not np.isfinite(sampling).all() or np.any(sampling<=0) or not np.allclose(sampling,sampling[0]):
                raise InputError('Half maps must be cubic with positive isotropic sampling.')
            if shape is not None and (shape!=m.data.shape or not np.allclose(sampling,pixel,rtol=1e-6)):raise InputError('Half-map dimensions or sampling differ.')
            shape=m.data.shape;pixel=float(sampling[0])
        paths.append(str(path))
    return paths,shape[0],pixel


def submit(service,project_id,identifier,options=None):
    raw=dict(options) if isinstance(options,dict) else options
    if raw is None:raw={}
    if not isinstance(raw,dict):raise InputError('Invalid PostProcess options.')
    queue_only=raw.pop('queue_only',False)
    if type(queue_only) is not bool:raise InputError('Invalid queue option.')
    params=validate_options(raw)
    with service.lock:
        root=service._root(project_id);data=service._read(root);record=jobs.assembly(data,identifier)
        if not data.get('slurm',{}).get('ready'):raise InputError('Configure cluster Settings first.')
        from .sta_dependencies import maybe_defer
        deferred=maybe_defer(service,project_id,root,data,record,'postprocessing',raw,queue_only)
        if deferred is not None:return deferred
        if not (storage_path(jobs.REFERENCE, '_numerics/apps/postprocess.py')).is_file():raise InputError('Greco Python postprocess is unavailable.')
        latest=jobs.update(service,record,kind='refinement3d')
        selected=raw.get('from','refinement3d:'+str(latest.get('work','')))
        if not isinstance(selected,str) or not selected.startswith('refinement3d:'):raise InputError('Select a completed Refine3D as From.')
        source=jobs.volume_job(service,project_id,identifier,'refinement3d',Path(selected.split(':',1)[1]).name)
        source=jobs.update(service,{'refinement3d':source},kind='refinement3d')
        if selected!='refinement3d:'+source['work'] or source['state']!='completed':raise InputError('Select a completed Refine3D as From.')
        maps,box,pixel=half_map_inputs(service,root,record,source)
        inputs=list(maps)
        if params['mask_path']:
            from ..sta.reference_mask import load_mask
            path=service.path(params['mask_path'])
            try:load_mask(path,box,pixel,resample_physical=True)
            except (ValueError,OSError) as error:raise InputError(str(error)) from error
            params['mask_path']=str(path);inputs.append(str(path))
        snapshot=[]
        for name in inputs:
            stat=Path(name).stat();snapshot.append(dict(path=name,size=stat.st_size,mtime_ns=stat.st_mtime_ns))
        config=dict(action='postprocess',reference_root=str(jobs.REFERENCE),assembly_id=identifier,
            half_maps=maps,snapshot=snapshot,output_box=box,pixel_size_A=pixel,options=params,
            particle_source=dict(kind='refinement3d',work=source['work']),
            particle_count=source.get('particle_count'),used_particles=source.get('used_particles'))
        return jobs.enqueue(service,project_id,root,data,record,config,'postprocessing',queue_only=queue_only)


FILES={'map':'volumes/map.mrc','masked_map':'volumes/map_masked.mrc','star':'analysis.star','report':'report.pdf'}


def artifact(service,project_id,identifier,job_name,name):
    if name not in FILES:raise InputError('Unknown PostProcess file.')
    job=jobs.volume_job(service,project_id,identifier,'postprocessing',job_name)
    if job['state']!='completed':raise InputError('PostProcess is not complete.')
    path=storage_path(storage_path(Path(job['work']), 'output'), FILES[name])
    if path.is_symlink() or service.path(str(path))!=path:raise InputError('Invalid PostProcess file.')
    return path
