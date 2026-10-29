"""Unweighted MC sums on the exact saved Greco display grid (CPU, one tilt).

Registration stacks and diagnostic measurements are never overwritten. Camera
inclinations are projection geometry, not extra image rotations: use the same
axis-aligned display convention as landmark_geometry_export.
"""
from greco.project_layout import join as storage_path
from greco.preprocessing_star import read_table

import json
from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import map_coordinates


def _stamp(path):
    stat=path.stat()
    return str(path),stat.st_size,stat.st_mtime_ns


def display_plane(service,item,index,*,source_index=None):
    aligned=service.path(item['stack'])
    # Resolve the owning reconstruction or geometry configuration.
    work=None
    for parent in aligned.parents:
        candidate=storage_path(parent, 'config.json')
        if candidate.is_file() and json.loads(candidate.read_text()).get('inputs',{}).get('stack'):
            work=parent;break
    if work is None:return None  # No owning reconstruction configuration.
    config_path=storage_path(work, 'config.json')
    config=json.loads(config_path.read_text())
    inputs=config.get('inputs',{})
    if not inputs.get('stack'):return None
    binning=int(inputs.get('binning',8))
    raw=service.path(str(storage_path(Path(inputs['stack']).parent, f'greco_raw_bin{binning}.mrcs')))
    checkpoint=storage_path(aligned.parent, 'rigid_geometry.npz')
    geometry=checkpoint.is_file()
    if geometry:
        with np.load(checkpoint,allow_pickle=False) as saved:geometry='cameras_xyz' in saved
    if not geometry:checkpoint=storage_path(aligned.parent, 'final_state.npz')
    checkpoint=service.path(str(checkpoint))
    frozen=service.path(inputs['frozen_shifts'])
    angles=service.path(inputs['rawtlt'])
    return _render(_stamp(raw),_stamp(checkpoint),_stamp(config_path),
                   _stamp(frozen),_stamp(angles),_stamp(aligned),int(index),geometry,source_index)


@lru_cache(maxsize=16)
def _render(raw_key,checkpoint_key,config_key,frozen_key,angles_key,aligned_key,index,geometry,source_index=None):
    from ..gcc import raw_affine
    from ..landmark_geometry_export import display_geometry
    config=json.loads(Path(config_key[0]).read_text())
    orientation=config['orientation'];axis=float(orientation['rotation_deg'])
    flip=bool(orientation.get('flip_y',False))
    angles=np.loadtxt(angles_key[0],ndmin=1)
    rows=read_table(frozen_key[0])
    if len(rows)!=len(angles) or not np.allclose([float(r['angle_deg']) for r in rows],angles,atol=1e-4):
        raise ValueError('Raw display tilt identities do not match the saved geometry.')
    with np.load(checkpoint_key[0],allow_pickle=False) as saved, mrcfile.mmap(raw_key[0],mode='r') as m:
        if len(m.data)!=len(angles) or m.data.shape[1]!=m.data.shape[2]:
            raise ValueError('Raw display stack shape does not match tilt identities.')
        n=m.data.shape[-1]
        if geometry:
            if not np.allclose(saved['angles_deg'],angles,atol=1e-4):
                raise ValueError('Refined display angle identities do not match.')
            active=np.asarray(saved['active_indices'] if 'active_indices' in saved else np.arange(len(angles)),int)
            if source_index is None:
                if not 0<=index<len(active):raise ValueError('Tilt index out of range.')
                source_index=int(active[index])
            _,shifts,_=display_geometry(saved['cameras_xyz'],
                saved['translations_xy_A']/float(saved['pixel_size_A']),axis,
                int(saved['fit_size']),n,flip)
            shift=shifts[source_index];frozen=np.zeros(2)
        else:
            active=np.asarray(saved['active_indices'],int)
            if source_index is None:
                if not 0<=index<len(active):raise ValueError('Tilt index out of range.')
                source_index=int(active[index])
            if not np.allclose(saved['angles_deg'],angles,atol=1e-4):
                raise ValueError('Reconstruction display angle identities do not match.')
            from ..alignment_geometry import resolve
            axis=float(resolve(saved,angles,axis)[1][source_index])
            shift=saved['cumulative_shifts_yx'][source_index]
            row=rows[source_index];binning=int(config['inputs'].get('binning',8))
            frozen=np.array([float(row['dy_raw']),float(row['dx_raw'])])/binning
            if flip:frozen[0]*=-1
        if not 0<=source_index<len(m.data):raise ValueError('Tilt index out of range.')
        plane=np.array(m.data[source_index],np.float32)
    if flip:plane=plane[::-1]
    a,b,c,d,center=raw_affine(axis,n)
    yy,xx=np.mgrid[:n,:n].astype(np.float32)
    py=yy-center-shift[0];px=xx-center-shift[1]
    ry=a*py+b*px+center-frozen[0];rx=c*py+d*px+center-frozen[1]
    result=map_coordinates(plane,np.array([ry,rx]),order=1,mode='constant',cval=float(plane.mean()),prefilter=False)
    result.setflags(write=False)
    return result
