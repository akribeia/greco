"""Final fitted landmark coordinates in the optimizer's centred d,q,h frame."""
from pathlib import Path
import json
import numpy as np
from greco.project_layout import join,owned
from .projects import InputError


def cloud(service,project_id,series_id,assembly_id=None,*,apply_leveling=True):
    import mrcfile
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        if assembly_id is not None and assembly_id!=data.get('active_reconstruction_assembly'):
            raise InputError('Reconstruction changed; reopen its card.')
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        job=row.get('reconstruction',{})
        if job.get('state') not in ('completed','partial'):raise InputError('Complete reconstruction to view the final landmarks.')
        work=Path(job['work']);checkpoint=join(work,'alignment/final_state.npz')
        if not owned(root,checkpoint.resolve()):raise InputError('Invalid landmark geometry path.')
        with mrcfile.open(join(work,'alignment/aligned_bin8.mrcs'),header_only=True) as handle:size=int(handle.header.nx)
        with np.load(checkpoint,allow_pickle=False) as state:
            positions=np.asarray(state['positions_zyx'],float)
            valid=np.asarray(state['valid_views'],bool)
            binning=int(state['binning'])
        if positions.ndim!=2 or positions.shape[1]!=3 or valid.ndim!=2 or len(valid)!=len(positions):
            raise InputError('Invalid final landmark geometry.')
        supports=valid.sum(axis=1);keep=np.isfinite(positions).all(axis=1)&(supports>=4)
        points=positions[keep][:,[2,1,0]]-(size-1)/2
        config=json.loads((work/'config.json').read_text())
        depth=size*float(config.get('tomogram_thickness_percent',50.))/100
        result=dict(points=[dict(id=int(index)+1,d=float(p[0]),q=float(p[1]),h=float(p[2]),views=int(count))
                            for index,p,count in zip(np.flatnonzero(keep),points,supports[keep])],
                    dimensions_dqh=[size,size,depth],binning=binning,
                    coordinate_frame='centred_optimizer_dqh',revision=checkpoint.stat().st_mtime_ns)
        return leveled_display(result,job) if apply_leveling else result


def leveled_display(data,job):
    """Rotate display coordinates and density offsets, leaving saved geometry intact."""
    report=job.get('tomogram_leveling')
    if not report:return data
    rotation=np.asarray(report['rotation_zyx'],float)[::-1,::-1]
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all():raise InputError('Invalid leveling rotation.')
    for point in data.get('points',data.get('meshes',[])):
        point['d'],point['q'],point['h']=map(float,rotation@np.array([point['d'],point['q'],point['h']]))
        if 'vertices' in point:point['vertices']=(np.asarray(point['vertices'])@rotation.T).tolist()
    data.update(coordinate_frame='centred_leveled_dqh',leveled=True,tilt_axis_dqh=(rotation@np.array([0.,1.,0.])).tolist())
    return data
