"""Submit fixed-pose GPU Class3D from completed Greco extractions."""
from greco.project_layout import join as storage_path, owned as layout_owned
import math
from pathlib import Path
from .projects import InputError
from . import particle_jobs as jobs
from .assemblies import assembly_directory
from ..particle_star import header


def validate_options(raw, pixel, box):
    if not isinstance(raw,dict) or set(raw)-{'from','classes','iterations','regularization','resolution_A','particle_diameter_A','seed','observation_model','noise_model','mask_path'}:
        raise InputError('Unknown Class3D options.')
    values=dict(mask_path=None,classes=3,iterations=25,regularization=4.,resolution_A=2*pixel,seed=0,particle_diameter_A=None,observation_model='raw',noise_model='half_residual')
    values.update({k:v for k,v in raw.items() if k!='from'})
    if values['observation_model'] not in ('raw','ctf_premultiplied'):raise InputError('Invalid observation model.')
    if values['noise_model'] not in ('half_residual','view_residual_em'):raise InputError('Invalid noise model.')
    for key,lo,hi in [('classes',2,32),('iterations',1,200),('seed',0,2**31-1)]:
        v=values[key]
        if isinstance(v,bool) or not isinstance(v,int) or not lo<=v<=hi:raise InputError(f'Invalid {key}.')
    for key,lo,hi in [('regularization',.1,100.),('resolution_A',2*pixel,box*pixel/4)]:
        v=values[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise InputError(f'Invalid {key}.')
    if values['mask_path'] is not None and (not isinstance(values['mask_path'],str) or not values['mask_path'].strip()):raise InputError('Invalid mask path.')
    diameter=values['particle_diameter_A']
    if diameter is not None and (isinstance(diameter,bool) or not isinstance(diameter,(int,float)) or not math.isfinite(diameter) or not 4*pixel<=diameter<=box*pixel):
        raise InputError('Particle diameter must fit in the extracted box.')
    return values


def submit(service,project_id,identifier,options=None):
    options=dict(options) if isinstance(options,dict) else options
    queue_only=options.pop('queue_only',False) if isinstance(options,dict) else False
    if type(queue_only) is not bool:raise InputError('Invalid queue option.')
    with service.lock:
        root=service._root(project_id);data=service._read(root);record=jobs.assembly(data,identifier)
        if not data.get('slurm',{}).get('ready'):raise InputError('Configure cluster Settings first.')
        from .sta_dependencies import maybe_defer
        deferred=maybe_defer(service,project_id,root,data,record,'classification',options,queue_only)
        if deferred is not None:return deferred
        extraction=jobs.update(service,record,kind='extraction')
        if extraction['state']!='completed' and not (isinstance(options,dict) and str(options.get('from','')).startswith(('extraction:','refinement3d:','polishing:','ctf_refinement:'))):raise InputError('Complete particle extraction first.')
        raw=options or {};source='extraction:'+str(extraction.get('work'))
        if not isinstance(raw,dict):raise InputError('Select a completed extraction as From.')
        selected=raw.get('from',source)
        source_kind='extraction'
        if isinstance(selected,str) and selected.startswith(('refinement3d:','polishing:','ctf_refinement:')):
            from .particle_observation_refinement import completed_source
            source_kind,extraction,_=completed_source(service,project_id,identifier,selected)
        elif selected!=source:
            if not isinstance(selected,str) or not selected.startswith('extraction:'):raise InputError('Select a completed extraction as From.')
            try:extraction=jobs.volume_job(service,project_id,identifier,'extraction',Path(selected.split(':',1)[1]).name)
            except InputError as error:raise InputError('Select a completed extraction as From.') from error
            if extraction['state']!='completed' or selected!='extraction:'+str(extraction['work']):raise InputError('Select a completed extraction as From.')
        work=service.path(extraction['work']);manifest=service.path(extraction.get('particles'))
        if not work.is_relative_to(assembly_directory(root,record)) or manifest!=storage_path(work, 'output/particles.star'):
            raise InputError('Class3D requires this assembly’s Greco STAR/MRCS extraction.')
        general,_=header(manifest);n=int(general['grecoBoxSize']);pixel=float(general['grecoPixelSizeAngst'])
        if 'mask_path' not in raw:
            from .assemblies import default_mask_path
            raw=dict(raw,mask_path=default_mask_path(record))
        params=validate_options(raw,pixel,n)
        mask_snapshot=[]
        if params['mask_path']:
            from ..sta.reference_mask import load_mask
            path=service.path(params['mask_path'])
            try:load_mask(path,n,pixel,resample_physical=True)
            except (ValueError,OSError) as error:raise InputError(str(error)) from error
            params['mask_path']=str(path)
            stat=path.stat();mask_snapshot=[dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)]
        if general['grecoSupportedParticleCount']<2*params['classes']:raise InputError('Not enough supported particles for these classes.')
        from .particle_observation_refinement import input_snapshot
        snapshot=mask_snapshot+input_snapshot(service,root,record,manifest)
        config=dict(action='classify',reference_root=str(jobs.REFERENCE),extraction_manifest=str(manifest),
            snapshot=snapshot,gpu=True,assembly_id=identifier,output_box=n,pixel_size_A=pixel,options=params,
            particle_source=dict(kind=source_kind,assembly_id=identifier,work=str(work),particles=str(manifest)))
        if selected==source:record['extraction']=extraction
        return jobs.enqueue(service,project_id,root,data,record,config,'classification',queue_only=queue_only)


def plane(service,project_id,identifier,index,class_id,axis='z',job_name=None):
    import io
    from .sta_display import volume_image
    job=jobs.volume_job(service,project_id,identifier,'classification',job_name)
    if job['state']!='completed':raise InputError('Class3D is not complete.')
    record=next((c for c in job['classes'] if c['id']==class_id),None)
    if record is None:raise InputError('Unknown class.')
    path=service.path(record['volume'])
    if path!=storage_path(storage_path(Path(job['work']), 'output'), f'class_{class_id:03d}.mrc') or not layout_owned(service._root(project_id),path):
        raise InputError('Invalid class volume path.')
    image=volume_image(path,index,axis=axis,zero_centered=True)
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()
