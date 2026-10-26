"""Owned immutable GPU polishing / mean-defocus jobs, using own-half maps."""
from greco.project_layout import join as storage_path
import math
from pathlib import Path
from .projects import InputError
from .assemblies import assembly_directory
from . import particle_jobs as jobs
from ..particle_star import header,image_sources,rows
from ..sta.corrections import source_chain

SOURCE_KINDS=('extraction','refinement3d','polishing','ctf_refinement')


def completed_source(service,project_id,identifier,selection):
    if not isinstance(selection,str) or ':' not in selection:raise InputError('Select a completed particle source.')
    kind,work=selection.split(':',1)
    if kind not in SOURCE_KINDS:raise InputError('Invalid particle source kind.')
    job=jobs.volume_job(service,project_id,identifier,kind,Path(work).name)
    job=jobs.update(service,{kind:job},kind=kind)
    if job['state']!='completed' or selection!=kind+':'+str(job['work']):raise InputError('Complete the selected particle source first.')
    manifest=service.path(job['particles'])
    if manifest!=storage_path(service.path(job['work']), 'output/particles.star'):raise InputError('Invalid particle source STAR.')
    return kind,job,manifest


def input_snapshot(service,root,record,manifest,extra=()):
    base,chain=source_chain(manifest);owned=assembly_directory(root,record);result=[]
    for path in dict.fromkeys([base,*chain,*image_sources(manifest),*extra]):
        path=service.path(str(path))
        if not path.is_file() or not path.is_relative_to(owned):raise InputError('Particle inputs must belong to this assembly.')
        stat=path.stat();result.append(dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return result


def validate_options(raw,kind,pixel,box):
    common={'from','reference_from','resolution_A','mask_path','noise_model'}
    allowed=common|({'maximum_shift_A','velocity_sigma_A','optimizer_steps','motion_model','divergence_sigma_A','gp3d_iterations','gp3d_max_modes','gp3d_dose_scaling','refine_rigid_geometry','observation_model'} if kind=='polishing' else {'maximum_defocus_A','defocus_step_A','ctf_model'})
    if not isinstance(raw,dict) or set(raw)-allowed:raise InputError('Unknown observation refinement options.')
    result=dict(resolution_A=min(box*pixel,max(2*pixel,12.)))
    result.update(dict(maximum_shift_A=min((5 if raw.get('motion_model')=='gp3d' else 2)*pixel,box*pixel/8),velocity_sigma_A=.2,optimizer_steps=10,
        motion_model=raw.get('motion_model','projected_2d'),divergence_sigma_A=5000.,gp3d_iterations=10000,gp3d_max_modes=-1,gp3d_dose_scaling=raw.get('motion_model')=='gp3d') if kind=='polishing' else dict(maximum_defocus_A=500.,defocus_step_A=25.))
    if kind=='polishing' and result['motion_model']!='gp3d':result['velocity_sigma_A']=pixel/2
    result.update({k:v for k,v in raw.items() if k not in ('from','reference_from')})
    limits=dict(resolution_A=(2*pixel,box*pixel),maximum_shift_A=(.001,box*pixel/8),velocity_sigma_A=(.001,box*pixel),maximum_defocus_A=(1.,5000.),defocus_step_A=(1.,1000.),optimizer_steps=(1,50),divergence_sigma_A=(.001,1e6),gp3d_iterations=(1,10000),gp3d_max_modes=(-1,10000))
    for key,value in result.items():
        if key=='ctf_model':
            if value not in ('joint','mean'):raise InputError('Invalid CTF refinement model.')
            continue
        if key=='mask_path':
            if value is not None and (not isinstance(value,str) or not value.strip()):raise InputError('Invalid reference mask path.')
            continue
        if key in ('observation_model','noise_model'):
            choices=('raw','ctf_premultiplied') if key=='observation_model' else ('half_residual','view_residual')
            if value not in choices:raise InputError('Invalid polishing '+key+'.')
            continue
        if key=='motion_model':
            if value not in ('gp3d','projected_2d','independent_ridge'):raise InputError('Invalid motion model.')
            continue
        if key in ('gp3d_dose_scaling','refine_rigid_geometry'):
            if type(value) is not bool:raise InputError('Invalid GP dose scaling.')
            continue
        lo,hi=limits[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:raise InputError('Invalid '+key+'.')
    if kind=='polishing' and type(result['optimizer_steps']) is not int:raise InputError('Optimizer steps must be an integer.')
    if kind=='polishing':
        if result.get('refine_rigid_geometry') and result['motion_model']!='gp3d':raise InputError('Rigid joint polishing requires GP3D.')
        for key in ('gp3d_iterations','gp3d_max_modes'):
            if type(result[key]) is not int or (key=='gp3d_max_modes' and result[key]==0):raise InputError('Invalid '+key+'.')
    return result


def submit(service,project_id,identifier,kind,options=None):
    if kind not in ('polishing','ctf_refinement'):raise InputError('Invalid observation refinement job.')
    raw=dict(options) if isinstance(options,dict) else options
    queue_only=raw.pop('queue_only',False) if isinstance(raw,dict) else False
    if type(queue_only) is not bool:raise InputError('Invalid queue option.')
    with service.lock:
        root=service._root(project_id);data=service._read(root);record=jobs.assembly(data,identifier)
        if not data.get('slurm',{}).get('ready'):raise InputError('Configure cluster Settings first.')
        from .sta_dependencies import maybe_defer
        deferred=maybe_defer(service,project_id,root,data,record,kind,raw,queue_only)
        if deferred is not None:return deferred
        if not isinstance(raw,dict):raise InputError('Select a completed Refine3D reference.')
        source_kind,source,manifest=completed_source(service,project_id,identifier,raw.get('from'))
        reference_kind,reference,reference_star=completed_source(service,project_id,identifier,raw.get('reference_from',raw.get('from')))
        if reference_kind!='refinement3d':raise InputError('Reference must be a completed Refine3D with independent half-maps.')
        # Same immutable extraction is stronger than equal particle count.
        if source_chain(manifest)[0]!=source_chain(reference_star)[0]:raise InputError('Source and reference must use the same extraction and half assignments.')
        general,_=header(manifest);box=int(general['grecoBoxSize']);pixel=float(general['grecoPixelSizeAngst'])
        # New GUI jobs inherit the measured reference bandwidth, not a manual
        # Nyquist target. Keep explicit values compatible with existing jobs.
        if 'resolution_A' not in raw:
            resolution=reference.get('diagnostic_resolution_A')
            if isinstance(resolution,(int,float)) and not isinstance(resolution,bool) and math.isfinite(resolution) and resolution>0:
                raw['resolution_A']=min(box*pixel,max(2*pixel,resolution))
        params=validate_options(raw,kind,pixel,box)
        maps=reference.get('unfiltered_volumes') or reference.get('volumes',{})
        half_maps=[maps.get('half1'),maps.get('half2')]
        if not all(half_maps):raise InputError('Independent reference half-maps are missing.')
        snapshot=input_snapshot(service,root,record,manifest,[reference_star,*half_maps])
        if params.get('mask_path'):
            from ..sta.reference_mask import load_mask
            path=service.path(params['mask_path'])
            try:load_mask(path,box,pixel,resample_physical=True)
            except (ValueError,OSError) as error:raise InputError(str(error)) from error
            params['mask_path']=str(path)
            stat=path.stat();snapshot.append(dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
        config=dict(action=kind,reference_root=str(jobs.REFERENCE),extraction_manifest=str(manifest),snapshot=snapshot,
            gpu=True,assembly_id=identifier,output_box=box,pixel_size_A=pixel,options=params,half_maps=half_maps,
            reference_job=reference['work'],particle_source=dict(kind=source_kind,work=source['work'],particles=str(manifest)))
        return jobs.enqueue(service,project_id,root,data,record,config,kind,queue_only=queue_only)


def report_file(service,project_id,identifier,kind,name):
    """Serve a report only from a completed, owned observation-correction job."""
    from .assemblies import assembly_directory
    if kind not in ('polishing','ctf_refinement'):raise InputError('Invalid report kind.')
    job=jobs.volume_job(service,project_id,identifier,kind,name)
    if job.get('state')!='completed':raise InputError('Complete the job before downloading its report.')
    work=Path(job['work']);path=work/'report.pdf'
    if path.is_symlink():raise InputError('Invalid report path.')
    if not path.is_file():
        with service.lock:
            root=service._root(project_id);record=jobs.assembly(service._read(root),identifier)
            directory=assembly_directory(root,record).resolve()
        current=Path(job['particles']).resolve();seen=set()
        if current!=(work/'particles.star').resolve():raise InputError('Invalid report source.')
        for _ in range(32):
            if current in seen or not current.is_relative_to(directory):raise InputError('Invalid report source chain.')
            seen.add(current);general=next((r for block,r in rows(current) if block=='general'),{})
            source=general.get('grecoSourceParticleStar')
            if not source:break
            parent=current.parent;current=Path(source)
            if not current.is_absolute():current=parent/current
            current=current.resolve()
        else:raise InputError('Report source chain is too deep.')
        from ..sta.observation_report import publish
        publish(work)
    if service.path(str(path))!=path:raise InputError('Invalid report path.')
    return path
