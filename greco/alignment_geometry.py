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
    cameras=raw_cameras(state,expected_angles,default_axis)
    if np.ndim(axis)==0:
        rz=initial_cameras([0.],axis)[0]
        translations=-(np.asarray(shifts_yx)[:,::-1]@rz[:2,:2].T)-np.asarray(frozen_yx)[:,::-1]
    else:
        rz=initial_cameras(np.zeros(len(angles)),axes)
        translations=-np.einsum('tab,tb->ta',rz[:,:2,:2],np.asarray(shifts_yx)[:,::-1])-np.asarray(frozen_yx)[:,::-1]
    return cameras,translations


def inclinations(state,expected_angles):
    expected=np.asarray(expected_angles,float)
    gamma=np.asarray(state.get('axis_inclination_deg',np.zeros_like(expected)),float)
    if gamma.shape!=expected.shape or not np.isfinite(gamma).all() or np.any(abs(gamma)>=89):
        raise ValueError('Invalid fitted per-view axis inclinations.')
    return gamma


def aligned_cameras(state,expected_angles,default_axis):
    """Full cameras after the saved in-plane raster rotation is removed."""
    from .landmark_rigid import initial_cameras
    angles,_=resolve(state,expected_angles,default_axis)
    base=initial_cameras(angles,0.)
    if 'axis_inclination_deg' not in state:return base
    gamma=np.deg2rad(inclinations(state,expected_angles));rx=np.broadcast_to(np.eye(3),base.shape).copy()
    rx[:,1,1]=rx[:,2,2]=np.cos(gamma);rx[:,1,2]=-np.sin(gamma);rx[:,2,1]=np.sin(gamma)
    return rx@base@rx.transpose(0,2,1)


def raw_cameras(state,expected_angles,default_axis):
    from .landmark_rigid import initial_cameras
    angles,axes=resolve(state,expected_angles,default_axis)
    if 'axis_inclination_deg' not in state:return initial_cameras(angles,scalar_or_array(axes))
    return initial_cameras(np.zeros(len(angles)),scalar_or_array(axes))@aligned_cameras(state,expected_angles,default_axis)


def defocus_geometry(state,expected_angles,default_axis,flip_y=False):
    """Equivalent plane normal for spatial CTF, including experimental gamma."""
    angles,axes=resolve(state,expected_angles,default_axis)
    if 'axis_inclination_deg' not in state:return angles,(-axes if flip_y else axes)
    normal=raw_cameras(state,expected_angles,default_axis)[:,:,2].copy()
    if flip_y:normal[:,1]*=-1
    length=np.linalg.norm(normal[:,:2],axis=1);sign=np.where(angles<0,-1.,1.)
    effective=sign*np.rad2deg(np.arctan2(length,normal[:,2]))
    beta=np.rad2deg(np.arctan2(-normal[:,1]*sign,-normal[:,0]*sign))
    reference=-axes if flip_y else axes
    beta+=360*np.rint((reference-beta)/360)
    beta=np.where(length>1e-12,beta,reference)
    return effective,beta
