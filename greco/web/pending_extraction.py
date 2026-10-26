"""Freeze a pending pick source; resolve its outputs only after Slurm afterok."""
from greco.project_layout import join as storage_path, owned as layout_owned
import copy
import json
from pathlib import Path
from .projects import InputError, ProjectService


def refresh_members(service,root,record):
    """Publish counts when the assembly's exact pending pick outputs become ready."""
    changed=False
    for member in record.get('tomograms',[]):
        if not member.get('picking_dependency'):continue
        try:
            work=service.path(member['picking_work'])
            if not layout_owned(root,work):continue
            from .metadata_cache import read_json
            result=read_json(service,storage_path(work, 'output/web_result.json'))
            count=result.get('pick_count')
            if result.get('state')!='completed' or type(count) is not int or count<=0:continue
            coordinates=service.path(result['star']);volume=service.path(result['volume'])
            if not coordinates.is_file() or not volume.is_file() or not coordinates.is_relative_to(storage_path(work, 'output')) or not layout_owned(root,volume):continue
            stat=coordinates.stat()
            member.update(particle_count=count,star=str(coordinates),volume=str(volume),star_size=stat.st_size,star_mtime_ns=stat.st_mtime_ns)
            member.pop('picking_dependency');changed=True
        except (InputError,OSError,ValueError,KeyError,TypeError):continue
    if changed:record['particle_count']=sum(m['particle_count'] for m in record['tomograms'])
    return changed


def submit_pending(service,project_id,root,data,record,options):
    from .assemblies import BINNINGS,extraction_sampling
    from .particle_jobs import enqueue
    options=dict(options or {})
    if set(options)-{'from','binning'} or options.get('from','assembly:'+record['id'])!='assembly:'+record['id']:
        raise InputError('Pending picking requires the particle assembly coordinate source.')
    binning=options.get('binning',record['binning'])
    if type(binning) is not int or binning not in BINNINGS:raise InputError('Invalid extraction binning.')
    record.update(binning=binning,pixel_size_A=record['native_pixel_size_A']*binning,sampling_configured=True)
    record.pop('output_box_size',None);record.pop('output_pixel_size_A',None)
    sampling=extraction_sampling(record)
    dependencies=[]
    for member in record['tomograms']:
        if not member.get('picking_dependency'):continue
        if member['picking_host']!=data['slurm'].get('submission_host',''):raise InputError('Picking dependency must use the same Slurm host.')
        dependencies.append(member['picking_job_id'])
    config=dict(action='extract_after_picking',assembly_id=record['id'],output_box=sampling['box_size'],pixel_size_A=sampling['pixel_size_A'],
        native_pixel_size_A=record['native_pixel_size_A'],project_root=str(root),project_id=project_id,
        project_snapshot=copy.deepcopy(data),extraction_options=options)
    return enqueue(service,project_id,root,data,record,config,'extraction',dependencies=tuple(dict.fromkeys(dependencies)))


def resolve(config):
    """Use the ordinary validated extraction builder, without writing a project."""
    from .particle_jobs import submit
    from .preprocessing import refresh
    root=Path(config['project_root']).resolve();data=copy.deepcopy(config['project_snapshot'])
    class SnapshotService(ProjectService):
        def _root(self,identifier):
            if identifier!=config['project_id']:raise InputError('Invalid deferred project.')
            return root
        def _read(self,path):return data
        def _write(self,*args):raise RuntimeError('Deferred worker cannot modify project metadata.')
    service=SnapshotService(root)
    record=next(a for a in data['particle_assemblies'] if a['id']==config['assembly_id'])
    for member in record['tomograms']:
        if not member.get('picking_dependency'):continue
        work=service.path(member['picking_work'])
        if not layout_owned(root,work):raise InputError('Invalid deferred picking path.')
        result=json.loads((storage_path(work, 'output/web_result.json')).read_text())
        count=result.get('pick_count')
        if result.get('state')!='completed' or type(count) is not int or count<=0:raise InputError(member['name']+': picking did not produce particles.')
        coordinates=service.path(result['star']);volume=service.path(result['volume'])
        if not coordinates.is_relative_to(storage_path(work, 'output')) or not layout_owned(root,volume):raise InputError('Invalid deferred picking outputs.')
        stat=coordinates.stat()
        member.update(particle_count=count,star=str(coordinates),volume=str(volume),star_size=stat.st_size,star_mtime_ns=stat.st_mtime_ns)
        member.pop('picking_dependency')
    record['particle_count']=sum(m['particle_count'] for m in record['tomograms'])
    for row in data['series']:
        for kind in ('preprocessing','reconstruction','picking','refinement','angle_refinement'):
            if row.get(kind):row[kind]=refresh(service,{'preprocessing':row[kind]})
    resolved=submit(service,config['project_id'],record['id'],kind='extraction',options=config['extraction_options'],build_only=True)
    resolved['output']=config['output']
    (storage_path(Path(config['output']).parent, 'resolved_config.json')).write_text(json.dumps(resolved,indent=2))
    return resolved
