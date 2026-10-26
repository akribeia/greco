"""Submit bounded local GPU STA Refine3D from immutable extraction images."""
from greco.project_layout import join as storage_path
import math
from pathlib import Path
from . import particle_jobs as jobs
from .projects import InputError
from .assemblies import assembly_directory
from ..particle_star import header, rows


def validate_options(raw,pixel,box):
    allowed={'from','iterations','regularization','resolution_A','angular_range_deg','shift_range_A','optimizer_iterations','particle_diameter_A','mask_path','auto_refine','alignment_mode','local_stability_rescue','local_marginalization','update_noise','signal_prior','observation_model','noise_model','mask_corrected_fsc','refine_per_tilt_shifts'}
    allowed.add('refine_ctf')
    if not isinstance(raw,dict) or set(raw)-allowed:raise InputError('Unknown Refine3D options.')
    values=dict(iterations=100,regularization=4.,resolution_A=min(max(12.,2*pixel),box*pixel/4),
        angular_range_deg=3.,shift_range_A=min(2*pixel,box*pixel/8),optimizer_iterations=30,particle_diameter_A=None,mask_path=None,auto_refine=True,alignment_mode='trust_region',local_stability_rescue=False,local_marginalization=False,update_noise=False,signal_prior='fsc',observation_model='ctf_premultiplied',noise_model='half_residual',mask_corrected_fsc=True,refine_per_tilt_shifts=False)
    values['refine_ctf']=False
    values.update({k:v for k,v in raw.items() if k!='from'})
    if values['observation_model'] not in ('raw','ctf_premultiplied'):raise InputError('Invalid observation model.')
    if values['noise_model'] not in ('half_residual','view_power'):raise InputError('Invalid noise model.')
    if values['noise_model']=='half_residual' and values['update_noise']:raise InputError('Half-residual noise is frozen.')
    if type(values['refine_per_tilt_shifts']) is not bool:raise InputError('Invalid refine_per_tilt_shifts.')
    if type(values['refine_ctf']) is not bool:raise InputError('Invalid refine_ctf.')
    if values['refine_ctf'] and values['noise_model']!='half_residual':raise InputError('CTF alternation requires frozen half-residual noise.')
    if type(values['mask_corrected_fsc']) is not bool:raise InputError('Invalid mask_corrected_fsc.')
    if values['alignment_mode'] not in ('trust_region','local_search'):raise InputError('Invalid alignment mode.')
    if values['signal_prior'] not in ('fsc','reference_power'):raise InputError('Invalid signal prior.')
    if values['alignment_mode']=='trust_region' and values['local_marginalization']:raise InputError('Local marginalization requires local_search mode.')
    if type(values['auto_refine']) is not bool:raise InputError('Invalid auto_refine.')
    if type(values['local_stability_rescue']) is not bool:raise InputError('Invalid local_stability_rescue.')
    for key in ('local_marginalization','update_noise'):
        if type(values[key]) is not bool:raise InputError('Invalid '+key+'.')
    if values['mask_path'] is not None and (not isinstance(values['mask_path'],str) or not values['mask_path'].strip()):raise InputError('Invalid reference mask path.')
    for key,lo,hi in [('iterations',1,100),('optimizer_iterations',1,1000)]:
        v=values[key]
        if type(v) is not int or not lo<=v<=hi:raise InputError('Invalid '+key+'.')
    for key,lo,hi in [('regularization',.1,100.),('resolution_A',2*pixel,box*pixel/4),('angular_range_deg',.1,45.),('shift_range_A',.01,box*pixel/8)]:
        v=values[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise InputError('Invalid '+key+'.')
    d=values['particle_diameter_A']
    if d is not None and (isinstance(d,bool) or not isinstance(d,(int,float)) or not math.isfinite(d) or not 4*pixel<=d<=box*pixel):raise InputError('Particle diameter must fit in the box.')
    return values


def submit(service,project_id,identifier,options=None):
    options=dict(options) if isinstance(options,dict) else options
    queue_only=options.pop('queue_only',False) if isinstance(options,dict) else False
    if type(queue_only) is not bool:raise InputError('Invalid queue option.')
    with service.lock:
        root=service._root(project_id);data=service._read(root);record=jobs.assembly(data,identifier)
        if not data.get('slurm',{}).get('ready'):raise InputError('Configure cluster Settings first.')
        from .sta_dependencies import maybe_defer
        deferred=maybe_defer(service,project_id,root,data,record,'refinement3d',options,queue_only)
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
        work=service.path(extraction['work']);manifest=service.path(extraction['particles'])
        if not work.is_relative_to(assembly_directory(root,record)) or manifest!=storage_path(work, 'output/particles.star'):raise InputError('Invalid extraction source.')
        general,_=header(manifest);box=int(general['grecoBoxSize']);pixel=float(general['grecoPixelSizeAngst'])
        if 'mask_path' not in raw:
            from .assemblies import default_mask_path
            raw=dict(raw,mask_path=default_mask_path(record))
        params=validate_options(raw,pixel,box);halves=[0,0];particle_count=0
        mask_snapshot=[]
        if params['mask_path']:
            from ..sta.reference_mask import load_mask
            path=service.path(params['mask_path'])
            try:load_mask(path,box,pixel,resample_physical=True)
            except (ValueError,OSError) as error:raise InputError(str(error)) from error
            params['mask_path']=str(path)
            stat=path.stat();mask_snapshot=[dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)]
        for block,row in rows(manifest):
            if block=='observations':break
            if block=='particles':particle_count+=1
            if block=='particles' and row['grecoUsable']==1:
                h=row['grecoRandomSubset']
                if h not in (1,2):raise InputError('Invalid extraction half-set.')
                halves[h-1]+=1
        if min(halves)<2:raise InputError('Refine3D requires at least two supported particles in each half.')
        from .particle_observation_refinement import input_snapshot
        snapshot=list(mask_snapshot)+input_snapshot(service,root,record,manifest)
        config=dict(action='refine3d',reference_root=str(jobs.REFERENCE),extraction_manifest=str(manifest),snapshot=snapshot,
            gpu=True,assembly_id=identifier,output_box=box,pixel_size_A=pixel,options=params,
            particle_count=particle_count,used_particles=sum(halves),
            particle_source=dict(kind=source_kind,assembly_id=identifier,work=str(work),particles=str(manifest)))
        if selected==source:record['extraction']=extraction
        return jobs.enqueue(service,project_id,root,data,record,config,'refinement3d',queue_only=queue_only)
