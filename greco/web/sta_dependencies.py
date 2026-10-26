"""Submit future STA outputs with afterok; resolve via ordinary builders later."""
from greco.project_layout import join as storage_path
import copy,json
from pathlib import Path
from .projects import InputError,ProjectService

SOURCE_KINDS=dict(extraction=('classification','refinement3d','polishing','ctf_refinement'),
    classification=('extraction','refinement3d','polishing','ctf_refinement'),
    refinement3d=('extraction','refinement3d','polishing','ctf_refinement'),
    polishing=('extraction','refinement3d','polishing','ctf_refinement'),
    ctf_refinement=('extraction','refinement3d','polishing','ctf_refinement'),
    postprocessing=('refinement3d',))


def maybe_defer(service,pid,root,data,record,kind,raw,queue_only=False):
    if getattr(service,'build_only',False):return None
    from . import particle_jobs as jobs
    from .slurm_arrays import valid_job_id
    if raw is None:raw={}
    if not isinstance(raw,dict):return None
    raw=dict(raw)
    default_kind={'postprocessing':'refinement3d'}.get(kind,'extraction')
    default=('assembly:'+record['id'] if kind=='extraction'
             else default_kind+':'+str(record.get(default_kind,{}).get('work','')))
    selected=raw.get('from',default)
    if 'from' not in raw and not default.split(':',1)[1]:return None
    selections=[selected]
    if kind in ('polishing','ctf_refinement'):selections.append(raw.get('reference_from',selected))
    sources=[];pending=False
    for i,selection in enumerate(selections):
        if kind=='extraction' and selection=='assembly:'+record['id']:return None
        if not isinstance(selection,str) or ':' not in selection:return None
        source_kind,work=selection.split(':',1)
        allowed=('refinement3d',) if i==1 else SOURCE_KINDS[kind]
        if source_kind not in allowed:return None
        try:source=jobs.volume_job(service,pid,record['id'],source_kind,Path(work).name)
        except InputError:return None  # Preserve the ordinary builder's validation/error.
        if selection!=source_kind+':'+str(source.get('work')):raise InputError('Source must belong to this particle assembly.')
        source=jobs.update(service,{source_kind:source},kind=source_kind)
        if source.get('state')!='completed':
            if source.get('state') not in ('submitted','running','unknown') or not valid_job_id(source.get('job_id')):
                raise InputError('The selected source is neither completed nor submitted to Slurm.')
            if source.get('host','')!=data['slurm'].get('submission_host',''):raise InputError('Dependencies must use the same Slurm host.')
            pending=True
        sources.append((source_kind,source))
    if not pending:return None
    source_kind,source=sources[0];box=int(source['box_size']);pixel=float(source['pixel_size_A'])
    if kind in ('classification','refinement3d'):
        from . import particle_classification,particle_refinement
        params=(particle_classification if kind=='classification' else particle_refinement).validate_options(raw,pixel,box)
    elif kind in ('polishing','ctf_refinement'):
        from .particle_observation_refinement import validate_options
        params=validate_options(raw,kind,pixel,box)
    elif kind=='postprocessing':
        from .particle_postprocess import validate_options
        params=validate_options(raw)
    elif kind=='extraction':
        from .assemblies import BINNINGS,extraction_sampling
        if set(raw)-{'from','classes','binning'}:raise InputError('Unknown extraction parameters.')
        q=raw.get('binning',4)
        if type(q) is not int or q not in BINNINGS:raise InputError('Invalid extraction binning.')
        params=dict(raw,binning=q)
        if source_kind=='classification':
            classes=raw.get('classes');total=source.get('options',{}).get('classes',0)
            if not isinstance(classes,list) or not classes or any(type(c) is not int or not 1<=c<=total for c in classes):raise InputError('Choose valid class numbers.')
        elif raw.get('classes'):raise InputError('Classes require a Class3D source.')
        extraction_record=dict(record,binning=q,pixel_size_A=float(record['native_pixel_size_A'])*q)
        extraction_record.pop('output_box_size',None);extraction_record.pop('output_pixel_size_A',None)
        sampling=extraction_sampling(extraction_record)
        box=sampling['box_size'];pixel=sampling['pixel_size_A']
    else:raise InputError('Unsupported STA job type.')
    raw['from']=selected
    # The snapshot is metadata only. Future stars/maps are deliberately absent
    # from snapshots until the parent has completed successfully.
    config=dict(action='sta_after_dependencies',deferred_kind=kind,deferred_options=raw,
        project_snapshot=copy.deepcopy(data),project_root=str(root),project_id=pid,
        assembly_id=record['id'],reference_root=str(jobs.REFERENCE),output_box=box,pixel_size_A=pixel,
        options=params,particle_count=source.get('particle_count',record.get('particle_count')),
        used_particles=source.get('used_particles'),
        particle_source=dict(kind=source_kind,work=source['work']))
    if len(sources)>1:config['reference_job']=sources[1][1]['work']
    return jobs.enqueue(service,pid,root,data,record,config,kind,queue_only=queue_only,
        validated_sources={job['work']:job for _,job in sources})


def pending_particle_counts(service, work, seen=None):
    """Resolve deferred counts from their frozen source, never the assembly total."""
    from .metadata_cache import read_json
    work=Path(work)
    seen=set() if seen is None else seen
    identity=str(work.resolve())
    if identity in seen:return dict(particle_count=None,used_particles=None)
    seen.add(identity)
    def load(path):
        try:return read_json(service,path)
        except (ValueError,OSError):return {}
    config=load(work/'config.json')
    if config.get('action')!='sta_after_dependencies':return None
    resolved=load(storage_path(work,'resolved_config.json'))
    if resolved:
        return {key:resolved.get(key) for key in ('particle_count','used_particles')}
    source=config.get('particle_source',{})
    if not source.get('work'):return dict(particle_count=None,used_particles=None)
    parent=Path(source['work'])
    result=load(parent/'result.json')
    if config.get('deferred_kind')=='extraction' and source.get('kind')=='classification':
        selected=config.get('deferred_options',{}).get('classes',[])
        classes=result.get('classes',[]) if result.get('state')=='completed' else []
        counts={c['id']:c.get('particle_count') for c in classes}
        total=sum(counts[c] for c in selected) if selected and all(type(counts.get(c)) is int for c in selected) else None
        return dict(particle_count=total,used_particles=total)
    if result.get('state')=='completed':
        return {key:result.get(key) for key in ('particle_count','used_particles')}
    inherited=pending_particle_counts(service,parent,seen)
    if inherited is not None:return inherited
    parent_config=load(storage_path(parent,'resolved_config.json')) or load(parent/'config.json')
    return {key:parent_config.get(key) for key in ('particle_count','used_particles')}


def resolve(config):
    from . import particle_jobs as jobs
    root=Path(config['project_root']).resolve();data=copy.deepcopy(config['project_snapshot'])
    class SnapshotService(ProjectService):
        build_only=True
        def _root(self,pid):
            if pid!=config['project_id']:raise InputError('Invalid deferred project.')
            return root
        def _read(self,path):return data
        def _write(self,*args):raise RuntimeError('Deferred STA worker cannot modify project metadata.')
    service=SnapshotService(root)
    record=jobs.assembly(data,config['assembly_id'])
    # Refresh frozen records from THEIR result files, not today's GUI selection.
    for kind in SOURCE_KINDS:
        if record.get(kind):record[kind]=jobs.update(service,record,kind=kind)
    for index,job in enumerate(record.get('job_history',[])):
        if job.get('kind'):record['job_history'][index]=dict(jobs.update(service,{job['kind']:job},kind=job['kind']),kind=job['kind'])
    kind=config['deferred_kind'];options=config['deferred_options']
    if kind=='extraction':
        resolved=jobs.submit(service,config['project_id'],record['id'],kind=kind,options=options,build_only=True)
    else:
        from . import particle_refinement,particle_classification,particle_observation_refinement,particle_postprocess
        module=dict(refinement3d=particle_refinement,classification=particle_classification,
            polishing=particle_observation_refinement,ctf_refinement=particle_observation_refinement,
            postprocessing=particle_postprocess)[kind]
        if kind in ('polishing','ctf_refinement'):resolved=module.submit(service,config['project_id'],record['id'],kind,options)
        else:resolved=module.submit(service,config['project_id'],record['id'],options)
    resolved['output']=config['output']
    if resolved['action'] in ('sta_after_dependencies','extract_after_picking'):raise InputError('Parent jobs have not produced their outputs.')
    (storage_path(Path(config['output']), 'resolved_config.json')).write_text(json.dumps(resolved,indent=2))
    return resolved
