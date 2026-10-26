"""Oriented STA density insertion into canonical exported tomogram grids."""
from greco.project_layout import join as storage_path
from pathlib import Path
import numpy as np


def insertion_geometry(cloud,shape,pixel):
    from .final_reconstruction import export_transform
    n=shape[2];source_n=cloud['tomogram_shape_xyz'][0]
    physical=(np.asarray(cloud['positions_zyx'])-(source_n-1)/2)*cloud['tomogram_pixel_size_A']
    internal=physical/pixel+(n-1)/2
    transform=np.asarray(export_transform(n,(n-shape[0])//2),float)
    centers=internal@transform[:3,:3].T+transform[:3,3]
    inverse=np.linalg.inv(np.asarray(cloud['export_transform'],float))
    basis=transform[:3,:3][::-1,::-1]@inverse[:3,:3][::-1,::-1]
    return centers,basis


def insert_particles(data,output,use_gpu=True):
    import mrcfile
    from scipy.ndimage import gaussian_filter,affine_transform as cpu_affine
    from scipy.spatial.transform import Rotation
    cloud=data['sta_particles'];output=Path(output)
    with mrcfile.mmap(data['parent_volume']) as m:shape=tuple(m.data.shape);pixel=float(m.voxel_size.x)
    volumes=[];reference_pixel=None
    for path in cloud['reference_maps']:
        with mrcfile.mmap(path) as m:
            v=np.array(m.data,np.float32);p=float(m.voxel_size.x)
        if v.ndim!=3 or len(set(v.shape))!=1 or p<=0 or not np.isfinite(v).all():raise ValueError('Invalid STA reference map.')
        if reference_pixel is not None and (v.shape!=volumes[0].shape or not np.isclose(p,reference_pixel)):raise ValueError('STA reference half-map grids differ.')
        reference_pixel=p;volumes.append(v)
    template=sum(volumes)/len(volumes);del volumes
    mask_path=cloud.get('mask_path')
    if mask_path:
        from .sta.reference_mask import load_mask
        mask=load_mask(mask_path,template.shape[0],reference_pixel,resample_physical=True)
    else:
        from .helpers import soft_spherical_mask
        mask=soft_spherical_mask(template.shape[0],radius=.45*template.shape[0],edge=max(2,.05*template.shape[0]))
    template=gaussian_filter(template,max(.0,.35*pixel/reference_pixel))
    inside=np.asarray(mask)>.5
    if not inside.any():raise ValueError('Empty insertion mask.')
    values=template[inside];lo,hi=np.quantile(values,[.01,.99])
    if -lo>hi:template=-template
    background=float(np.median(template[~inside])) if (~inside).any() else float(np.median(template))
    signal=template-background;scale=float(np.quantile(signal[inside],.995))
    if scale<=0:raise ValueError('STA reference contains no positive density.')
    template=(np.clip((signal/scale-.1)/.9,0,1)*mask).astype(np.float32)
    if use_gpu:
        import cupy as cp
        from cupyx.scipy.ndimage import affine_transform
        reference=cp.asarray(template);host=cp.asnumpy;device=cp.asarray
    else:affine_transform=cpu_affine;reference=template;host=np.asarray;device=np.asarray
    centers,basis=insertion_geometry(cloud,shape,pixel)
    target=storage_path(output, 'particles.mrc');inserted=0
    with mrcfile.new_mmap(target,shape=shape,mrc_mode=2,fill=0,overwrite=False) as m:
        m.voxel_size=pixel
        for k,(center,p) in enumerate(zip(centers,cloud['particles'])):
            # Dataset uses this same passive ZYZ transpose: reference -> world.
            pose=Rotation.from_euler('ZYZ',[p['grecoAngle'+a] for a in ('Rot','Tilt','Psi')],degrees=True).as_matrix().T
            rotation=basis@pose
            extent=(abs(rotation)@((np.array(template.shape[::-1])-1)/2))*reference_pixel/pixel
            start=np.maximum(0,np.floor(center-extent[::-1]-1).astype(int));end=np.minimum(shape,np.ceil(center+extent[::-1]+2).astype(int))
            if np.any(end<=start):continue
            matrix=rotation.T[::-1,::-1]*pixel/reference_pixel
            offset=(np.array(template.shape)-1)/2+matrix@(start-center)
            patch=host(affine_transform(reference,device(matrix),offset=device(offset),output_shape=tuple(int(s) for s in end-start),order=1,mode='constant',cval=0,prefilter=False))
            slices=tuple(slice(a,b) for a,b in zip(start,end))
            m.data[slices]=np.maximum(m.data[slices],patch)
            inserted+=int(np.any(patch>0))
            if k%50==0:print(f'Insert particles: {k+1}/{len(centers)}',flush=True)
        m.update_header_stats()
    if inserted==0:raise ValueError('No particles were inserted into the tomogram. Check the STA coordinates and reference density.')
    return dict(particle_layer=str(target),shape=list(shape),voxel_size_A=pixel,
        particle_count=len(centers),inserted_particles=inserted,color=data['options']['color'],
        sta_source=cloud['selection'],parent_volume=data['parent_volume'],
        insertion_protocol='sta_passive_zyz_reference_to_export_world_v1',density_contour_fraction=.1)
