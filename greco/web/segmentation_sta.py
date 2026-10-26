"""Map owned STA coordinates back to their original alignment world frame."""
from greco.project_layout import job_work, join as storage_path
from greco.preprocessing_star import read_table
import json
from pathlib import Path
import numpy as np
from .projects import InputError


def geometry_view_support(valid, coarse, angles):
    """Require STA observations only on the reconstruction's accepted tilts."""
    angles = np.asarray(angles, float)
    valid = np.array(valid, dtype=bool, copy=True)
    with np.load(storage_path(Path(coarse), 'final_state.npz'), allow_pickle=False) as state:
        active = np.asarray(state['active_indices'])
        saved_angles = np.asarray(state['angles_deg'], float)
    if (active.ndim != 1 or not np.issubdtype(active.dtype, np.integer)
            or len(active) < 3 or np.any(active < 0) or np.any(active >= len(angles))
            or np.any(np.diff(active) <= 0)):
        raise InputError('Invalid accepted tilt selection for STA geometry.')
    if (saved_angles.shape != angles.shape or not np.isfinite(saved_angles).all()
            or not np.allclose(saved_angles, angles, atol=1e-4, rtol=0)):
        raise InputError('STA geometry accepted tilt identities differ from segmentation.')
    if valid.ndim != 2 or valid.shape[1] != len(angles):
        raise InputError('STA geometry observation dimensions differ from the tilt series.')
    valid[:, ~np.isin(np.arange(len(angles)), active)] = False
    counts = valid.sum(axis=0)
    bad = active[counts[active] < 8]
    if len(bad):
        detail = ', '.join(f'{angles[i]:g}°: {counts[i]} particles' for i in bad[:3])
        raise InputError('STA geometry needs at least eight visible particles per accepted tilt (' + detail + ').')
    return valid


def load_source(service,pid,data,member,coarse,selection,require_geometry=True):
    from . import particle_jobs as jobs
    from .assemblies import assembly_directory
    from ..particle_star import rows
    from ..sta.corrections import correction_tables
    if not isinstance(selection,str) or len(selection.split(':'))!=3:
        raise InputError('Select a Refine3D or Polishing STA job.')
    aid,kind,name=selection.split(':')
    if kind not in ('refinement3d','polishing'):raise InputError('Select Refine3D or Polishing.')
    assembly=jobs.assembly(data,aid);owned=assembly_directory(service._root(pid),assembly)
    job=jobs.volume_job(service,pid,aid,kind,name)
    job=jobs.update(service,{kind:job},kind=kind)
    if job.get('state')!='completed':raise InputError('Complete the selected STA job first.')
    star=service.path(job['particles'])
    if star!=storage_path(Path(job['work']), 'output/particles.star'):raise InputError('Invalid STA output.')
    base,chain,updates,_=correction_tables(star)
    paths=[base,*chain]
    for path in paths:
        if not path.is_relative_to(owned):raise InputError('STA source belongs to another assembly.')
    config_path=storage_path(job_work(base), 'resolved_config.json')
    if not config_path.exists():config_path=storage_path(job_work(base), 'config.json')
    extraction=json.loads(config_path.read_text());paths.append(config_path)
    source=next((t for t in extraction.get('tomograms',[]) if t['series_id']==member['series_id']),None)
    if source is None or Path(source['final_state']).resolve()!=(storage_path(Path(coarse), 'final_state.npz')).resolve():
        raise InputError('STA particles must originate from this tomogram and alignment.')
    source_config=Path(source['reconstruction_config']);paths.append(source_config)
    source_settings=json.loads(source_config.read_text())
    # A shared coarse lineage alone is insufficient if the display raster changed.
    from .picking_source import volume_context
    target_settings,_,_,_=volume_context(member['source_work'])
    if source_settings['orientation']!=target_settings['orientation'] or source_settings['inputs']['stack']!=target_settings['inputs']['stack']:
        raise InputError('STA particle raster does not match segmentation geometry.')
    tables=list(rows(base))
    tomo=next((r for b,r in tables if b=='tomograms' and r['grecoTomogramUid']==member['series_id']),None)
    if tomo is None:raise InputError('No STA particles for this tomogram.')
    tid=tomo['grecoTomogramId'];pixel=float(tomo['grecoTomogramPixelSizeAngst'])
    shape=np.array([tomo['grecoTomogramSize'+d] for d in 'XYZ'],float)
    transform=np.asarray(source['export_transform'],float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():raise InputError('Invalid STA export transform.')
    # Extraction metadata uses its alignment grid, whereas its frozen input
    # transform belongs to the picked volume (possibly at another binning).
    import mrcfile
    from ..particle_geometry import alignment_pick_grid
    paths.append(Path(source['volume']))
    with mrcfile.open(source['volume'],header_only=True) as volume:
        picked_shape=np.array([volume.header.nx,volume.header.ny,volume.header.nz],int)
        picked_pixel=float(volume.voxel_size.x)
    _,mapped_shape,transform=alignment_pick_grid(np.empty((0,3)),picked_shape,
        picked_pixel,int(shape[0]),pixel,transform)
    if not np.array_equal(mapped_shape,shape):raise InputError('STA tomogram grid does not match its picking source.')
    inverse=np.linalg.inv(transform)
    
    tilts=read_table(target_settings['inputs']['frozen_shifts'])
    mdoc=[int(r['mdoc_index']) for r in tilts]
    if len(set(mdoc))!=len(mdoc):raise InputError('Duplicate tilt identities.')
    lookup={v:i for i,v in enumerate(mdoc)}
    views={r['grecoViewId']:r for b,r in tables if b=='views' and r['grecoTomogramId']==tid}
    mapped={}
    for vid,v in views.items():
        index=lookup.get(int(v['grecoMdocIndex']))
        if index is None or not np.isclose(float(v['grecoTiltAngleDeg']),float(tilts[index]['angle_deg']),atol=1e-4):
            raise InputError('STA tilt identities differ from segmentation.')
        mapped[vid]=index
    particles=[];ids=[];poses=[]
    for block,row in tables:
        if block!='particles' or row['grecoTomogramId']!=tid:continue
        p=dict(row,**updates.get(row['grecoParticleId'],{}))
        if not p.get('grecoUsable',1):continue
        coord=np.array([p['grecoCoordinate'+d+'Angst'] for d in 'XYZ'],float)/pixel+(shape-1)/2
        particles.append(inverse[:3,:3]@coord[::-1]+inverse[:3,3]);ids.append(p['grecoParticleId'])
        poses.append(p)
    positions=np.asarray(particles,float).reshape(-1,3);valid=np.zeros((len(ids),len(mdoc)),bool)
    index={pid:k for k,pid in enumerate(ids)}
    for block,r in tables:
        if block=='observations' and r['grecoParticleId'] in index and r['grecoViewId'] in mapped:
            valid[index[r['grecoParticleId']],mapped[r['grecoViewId']]]=bool(r.get('grecoVisible',0))
    if not np.isfinite(positions).all() or not len(ids):raise InputError('No usable finite STA coordinates.')
    if require_geometry:
        paths.append(storage_path(Path(coarse), 'final_state.npz'))
        valid=geometry_view_support(valid,coarse,[float(r['angle_deg']) for r in tilts])
    snapshot=[]
    for path in dict.fromkeys(paths):
        path=service.path(str(path));stat=path.stat()
        snapshot.append(dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return dict(positions_zyx=positions.tolist(),valid_views=valid.tolist(),particle_ids=ids,
                source_star=str(star),selection=selection,mdoc_indices=mdoc,
                particles=poses,tomogram_pixel_size_A=pixel,tomogram_shape_xyz=shape.tolist(),
                export_transform=transform.tolist()),snapshot


def insertion_source(service,pid,data,member,coarse,selection):
    from . import particle_jobs as jobs
    if not isinstance(selection,str) or len(selection.split(':'))!=3:raise InputError('Select a completed PostProcess or Refine3D job.')
    aid,kind,name=selection.split(':')
    if kind not in ('postprocessing','refinement3d'):raise InputError('Insertion requires PostProcess or Refine3D, not Polishing.')
    job=jobs.volume_job(service,pid,aid,kind,name)
    job=jobs.update(service,{kind:job},kind=kind)
    if job.get('state')!='completed':raise InputError('Complete the selected STA job first.')
    work=Path(job['work']);config_path=storage_path(work, 'resolved_config.json')
    if not config_path.exists():config_path=storage_path(work, 'config.json')
    config=json.loads(config_path.read_text())
    coordinate_selection=selection
    if kind=='postprocessing':
        source=config.get('particle_source',{})
        if source.get('kind')!='refinement3d' or not source.get('work'):raise InputError('PostProcess has no recorded Refine3D source.')
        parent=jobs.volume_job(service,pid,aid,'refinement3d',Path(source['work']).name)
        if parent.get('work')!=source['work']:raise InputError('Invalid PostProcess coordinate source.')
        coordinate_selection=aid+':refinement3d:'+Path(source['work']).name
    cloud,snapshot=load_source(service,pid,data,member,coarse,coordinate_selection,require_geometry=False)
    volumes=job.get('volumes',{})
    maps=[volumes['merged']] if volumes.get('merged') else [volumes[k] for k in ('half1','half2') if volumes.get(k)]
    if len(maps) not in (1,2):raise InputError('The STA job has no refined or postprocessed reference map.')
    for p in cloud['particles']:
        if not all(isinstance(p.get('grecoAngle'+a),(int,float)) and np.isfinite(p['grecoAngle'+a]) for a in ('Rot','Tilt','Psi')):
            raise InputError('Refined particle Euler angles are missing.')
    owned=Path(job['work']).parent;paths=[config_path,*map(Path,maps)]
    mask=config.get('options',{}).get('mask_path')
    if mask:paths.append(Path(mask))
    for path in paths:
        path=service.path(str(path))
        if path!=Path(mask or '') and not path.is_relative_to(owned):raise InputError('STA map belongs to another assembly.')
        stat=path.stat();snapshot.append(dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    cloud.update(reference_maps=maps,mask_path=mask,selection=selection,coordinate_source=coordinate_selection)
    return cloud,snapshot
