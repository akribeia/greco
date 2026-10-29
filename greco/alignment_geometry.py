"""Resolve acquisition identities and fitted raster geometry separately."""
import numpy as np


def resolve(state,expected_angles,default_axis):
    expected=np.asarray(expected_angles,float)
    source=np.asarray(state['angles_deg'],float)
    if source.shape!=expected.shape or not np.allclose(source,expected,rtol=0,atol=1e-4):
        raise ValueError('Alignment source tilt identities differ.')
    angles=np.asarray(state.get('refined_angles_deg',source),float)
    axes=np.asarray(state.get('axis_rotation_deg',default_axis),float)
    if axes.ndim==0:axes=np.full(len(expected),float(axes))
    if angles.shape!=expected.shape or axes.shape!=expected.shape or not np.isfinite(angles).all() or not np.isfinite(axes).all():
        raise ValueError('Invalid fitted per-view geometry.')
    return angles,axes


def scalar_or_array(axes):
    axes=np.asarray(axes,float)
    return float(axes.flat[0]) if axes.ndim==0 or np.all(axes==axes.flat[0]) else axes


def camera_components(state,expected_angles,default_axis,shifts_yx,frozen_yx):
    from .landmark_rigid import initial_cameras
    angles,axes=resolve(state,expected_angles,default_axis)
    axis=scalar_or_array(axes)
    cameras=initial_cameras(angles,axis)
    if np.ndim(axis)==0:
        rz=initial_cameras([0.],axis)[0]
        translations=-(np.asarray(shifts_yx)[:,::-1]@rz[:2,:2].T)-np.asarray(frozen_yx)[:,::-1]
    else:
        rz=initial_cameras(np.zeros(len(angles)),axes)
        translations=-np.einsum('tab,tb->ta',rz[:,:2,:2],np.asarray(shifts_yx)[:,::-1])-np.asarray(frozen_yx)[:,::-1]
    return cameras,translations
