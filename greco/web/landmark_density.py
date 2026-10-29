"""Read-only density previews at fitted landmarks, in the same d,q,h frame."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from greco.project_layout import join,owned
from .landmark_cloud import cloud,leveled_display
from .projects import InputError

MAX_LANDMARKS=128
PATCH_RADIUS=16


def _stamp(path):
    stat=path.stat()
    return str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns


def _spread(points,limit):
    """Deterministic farthest-point subset, preserving the displayed coordinates."""
    if len(points)<=limit:return points
    xyz=np.array([[p[k] for k in ('d','q','h')] for p in points])
    scale=np.maximum(np.ptp(xyz,axis=0),1.)
    xyz=xyz/scale;distance=np.full(len(points),np.inf)
    index=int(np.argmax([p['views'] for p in points]));selected=[]
    for _ in range(limit):
        selected.append(index)
        distance=np.minimum(distance,np.sum((xyz-xyz[index])**2,axis=1))
        distance[selected]=-1.;index=int(np.argmax(distance))
    return [points[i] for i in selected]


def _surface(patch):
    """A display isosurface of the central connected density, of either polarity."""
    from scipy.ndimage import gaussian_filter,label
    from skimage.measure import marching_cubes
    patch=np.asarray(patch,np.float32)
    if not np.isfinite(patch).all():return None
    smooth=gaussian_filter(patch,1.)
    background=float(np.median(smooth));smooth=smooth-background
    sigma=float(np.std(smooth))
    if sigma<=1e-12:return None
    r=PATCH_RADIUS;z,y,x=np.ogrid[-r:r+1,-r:r+1,-r:r+1];radius=z*z+y*y+x*x
    central=np.where(radius<=36,smooth,0.)
    peak=float(central.flat[int(np.argmax(np.abs(central)))])
    if abs(peak)<sigma:return None
    signal=smooth*(1 if peak>=0 else -1)
    threshold=max(1.5*sigma,.35*abs(peak))
    support=(signal>threshold)&(radius<=12**2)
    labels,count=label(support)
    if not count:return None
    central_labels=np.unique(labels[(radius<=36)&support]);central_labels=central_labels[central_labels>0]
    if not len(central_labels):return None
    sizes=np.bincount(labels.ravel());chosen=int(central_labels[np.argmax(sizes[central_labels])])
    if sizes[chosen]<8:return None
    # Smooth the binary support only for the small display surface.
    display=gaussian_filter((labels==chosen).astype(np.float32),.7)
    if display[::2,::2,::2].max()<=.35:return None
    vertices,faces,_,_=marching_cubes(display,level=.35,step_size=2,allow_degenerate=False)
    return np.round(vertices[:,[2,1,0]]-r,2).tolist(),faces.tolist()


@lru_cache(maxsize=4)
def _meshes(volume_key,checkpoint_key,points_json,size,ratio,transform_tuple):
    import mrcfile
    transform=np.array(transform_tuple).reshape(4,4)
    points=json.loads(points_json);eligible=[];offset=np.arange(-PATCH_RADIUS,PATCH_RADIUS+1)
    grid=np.array(np.meshgrid(offset,offset,offset,indexing='ij'))*ratio
    with mrcfile.mmap(volume_key[0],mode='r') as handle:
        shape=np.array(handle.data.shape)
        def coordinates(point):
            internal=np.array([point['h'],point['q'],point['d']])+(size-1)/2
            centre=(internal+.5)*ratio-.5
            positions=centre[:,None,None,None]+grid
            return np.rint(np.einsum('ij,jabc->iabc',transform[:3,:3],positions)+transform[:3,3,None,None,None]).astype(int)
        for point in points:
            coords=coordinates(point)
            if all(c.min()>=0 and c.max()<n for c,n in zip(coords,shape)):eligible.append(point)
        meshes=[]
        for point in _spread(eligible,MAX_LANDMARKS):
            # Gather a small, bin-8-spaced preview; never load the full tomogram.
            patch=np.asarray(handle.data[tuple(coordinates(point))],np.float32)
            surface=_surface(patch)
            if surface is not None:
                vertices,faces=surface;meshes.append(dict(point,vertices=vertices,faces=faces))
    return meshes,len(eligible)


def densities(service,project_id,series_id,assembly_id=None):
    import mrcfile
    with service.lock:
        data=cloud(service,project_id,series_id,assembly_id,apply_leveling=False)
        root=service._root(project_id);project=service._read(root)
        row=next(r for r in project['series'] if r['id']==series_id)
        job=row['reconstruction'];work=Path(job['work'])
        original=job.get('volumes',{}).get('original')
        if not original:raise InputError('Original tomogram is not available.')
        volume=Path(original).resolve()
        if not owned(root,volume) or not volume.is_file():raise InputError('Invalid landmark density volume.')
        transform=np.asarray(job.get('internal_to_export_zyx'),float)
        if transform.shape!=(4,4) or not np.isfinite(transform).all():
            raise InputError('Tomogram export coordinates are not available for this reconstruction.')
        aligned=join(work,'alignment/aligned_bin8.mrcs')
        with mrcfile.open(aligned,header_only=True) as h:input_pixel=float(h.voxel_size.x)
        with mrcfile.open(volume,header_only=True) as h:output_pixel=float(h.voxel_size.x)
        if not np.isfinite([input_pixel,output_pixel]).all() or min(input_pixel,output_pixel)<=0:raise InputError('Missing tomogram pixel size.')
        ratio=input_pixel/output_pixel
        volume_key=_stamp(volume);checkpoint_key=_stamp(join(work,'alignment/final_state.npz'))
        points_json=json.dumps(data['points'],separators=(',',':'))
        size=int(data['dimensions_dqh'][0])
    meshes,eligible=_meshes(volume_key,checkpoint_key,points_json,size,ratio,tuple(transform.ravel()))
    from copy import deepcopy
    return leveled_display(dict(meshes=deepcopy(meshes),dimensions_dqh=data['dimensions_dqh'],binning=data['binning'],
        total_count=len(data['points']),eligible_count=eligible,displayed_count=len(meshes),
        source='original_tomogram',sampling='bin8_spaced_nearest_voxel_smoothed_display',
        revision=[data['revision'],volume_key[-1]]),job)
