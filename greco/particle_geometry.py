"""Particle extraction geometry and beam-defocus helpers.

Used by STA extraction and refinement; this module does not define a job type.
"""
import numpy as np


def alignment_pick_grid(xyz, shape_xyz, voxel, coarse_size, coarse_pixel,
                        export_transform_zyx):
    """Express full-field exported picks on the alignment pixel grid.

    Both grids use voxel centres. Conjugating the export transform preserves
    reflections and cropped Z origins without scaling the camera rotations.
    """
    shape=np.asarray(shape_xyz,int)
    transform=np.asarray(export_transform_zyx,float)
    if (shape.shape!=(3,) or np.any(shape<1) or not np.isfinite([voxel,coarse_pixel]).all()
            or min(voxel,coarse_pixel)<=0 or coarse_size<1
            or transform.shape!=(4,4) or not np.isfinite(transform).all()):
        raise ValueError('Invalid picking or alignment grid.')
    if not np.allclose(shape[:2]*voxel,coarse_size*coarse_pixel,rtol=1e-5,atol=1e-3):
        raise ValueError('Picking volume and alignment fields of view differ; cropped XY input is not supported.')
    if np.isclose(voxel,coarse_pixel) and np.all(shape[:2]==coarse_size):
        return xyz,shape,transform
    ratio=voxel/coarse_pixel
    aligned_shape=np.maximum(1,np.rint(shape*ratio).astype(int))
    aligned_shape[:2]=coarse_size
    exported=np.eye(4)*ratio;exported[3,3]=1
    exported[:3,3]=(aligned_shape[::-1]-1)/2-ratio*(shape[::-1]-1)/2
    internal=np.eye(4)*ratio;internal[3,3]=1
    internal[:3,3]=(coarse_size-1)/2-ratio*(shape[0]-1)/2
    mapped=(np.asarray(xyz)-(shape-1)/2)*ratio+(aligned_shape-1)/2
    return mapped,aligned_shape,exported@transform@np.linalg.inv(internal)


def output_box(native_box, binning):
    size=2*int(np.floor(float(native_box)/(2*binning)+.5))
    if size<8 or size>512:
        raise ValueError('The binned reconstruction box must be 8–512 pixels. Adjust the native box or binning.')
    return size


def raw_particle_geometry(xyz, shape_xyz, export_transform_zyx, angles, cumulative,
                          frozen, axis_deg, flip_y, coarse_size, *, cameras=None):
    """Map exported pick pixels back to the unchanged raw MC raster.

    Returns centres (particle,tilt,xy) in coarse detector pixels and one
    proper 3-D rotation per tilt. The third row is its oriented beam normal.
    """
    from .selection import project_points_aligned_axis0
    from .gcc import raw_sample_bases
    xyz=np.asarray(xyz,float)
    transform=np.asarray(export_transform_zyx,float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Missing export coordinate transform')
    inverse=np.linalg.inv(transform)
    def project(points):
        zyx=points[:,::-1]
        internal=(inverse[:3,:3]@zyx.T).T+inverse[:3,3]
        if cameras is None:
            aligned=np.stack([project_points_aligned_axis0(internal,a,coarse_size) for a in angles],axis=1)
        else:
            aligned=np.einsum('tij,nj->nti',np.asarray(cameras)[:,[1,0]],internal[:,::-1]-(coarse_size-1)/2)+(coarse_size-1)/2
        by,bx=raw_sample_bases(aligned,cumulative,frozen,axis_deg,coarse_size)
        if flip_y:by=coarse_size-1-by
        return np.stack((bx,by),axis=-1)
    bases=np.vstack((np.zeros((1,3)),np.eye(3)))
    p=project(bases)
    rows=(p[1:]-p[0]).transpose(1,2,0)
    rotations=np.concatenate((rows,np.cross(rows[:,0],rows[:,1])[:,None,:]),axis=1)
    np.testing.assert_allclose(rotations@rotations.transpose(0,2,1),np.broadcast_to(np.eye(3),rotations.shape),atol=1e-10)
    if np.any(np.linalg.det(rotations)<.999):raise ValueError('Invalid projection orientation')
    return project(xyz),rotations


def refined_raw_particle_geometry(xyz, export_transform_zyx, centres, rotations,
                                  geometry, angles, mdoc_indices, coarse_pixel,
                                  coarse_size, flip_y):
    """Apply a landmark camera fit in its world frame to exported picks.

    Replay changes relative to its saved original cameras/translations. This
    preserves the established detector-centre and binning conventions, and is
    exactly the original extraction when the fit has zero corrections.
    """
    if (str(geometry.get('protocol',''))!='greco-landmark-rigid-v1'
            or str(geometry.get('coordinate_convention',''))!='centered_world_xyz_and_raw_detector_xy_angstrom'
            or bool(geometry['raster_flip_y'])!=bool(flip_y)):
        raise ValueError('Unsupported refined landmark geometry convention.')
    ids=np.asarray(geometry['mdoc_indices'],int)
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or set(ids)!=set(mdoc_indices):
        raise ValueError('Refined landmark acquisition identities differ from extraction.')
    lookup={int(index):t for t,index in enumerate(ids)}
    order=np.array([lookup[int(index)] for index in mdoc_indices])
    if not np.allclose(np.asarray(geometry['angles_deg'])[order],angles,rtol=0,atol=1e-4):
        raise ValueError('Refined landmark angles differ from extraction.')
    cameras=np.asarray(geometry['cameras_xyz'],float)[order]
    original=np.asarray(geometry['original_cameras_xyz'],float)[order]
    translations=np.asarray(geometry['translations_xy_A'],float)[order]
    baseline=np.asarray(geometry['original_translations_xy_A'],float)[order]
    t=len(order)
    if (cameras.shape!=(t,3,3) or original.shape!=(t,3,3)
            or translations.shape!=(t,2) or baseline.shape!=(t,2)
            or not all(np.isfinite(a).all() for a in (cameras,original,translations,baseline))):
        raise ValueError('Invalid refined landmark cameras or translations.')
    for matrices in (cameras,original):
        if (not np.allclose(matrices@matrices.transpose(0,2,1),np.eye(3),atol=1e-8,rtol=0)
                or not np.allclose(np.linalg.det(matrices),1.,atol=1e-8,rtol=0)):
            raise ValueError('Refined landmark cameras must be proper rotations.')
    inverse=np.linalg.inv(np.asarray(export_transform_zyx,float))
    world_basis=inverse[:3,:3][::-1,::-1]
    detector_sign=np.array([1.,-1. if flip_y else 1.])
    expected=detector_sign[None,:,None]*np.einsum('tab,bc->tac',original[:,:2],world_basis)
    if not np.allclose(expected,rotations[:,:2],atol=1e-8,rtol=0):
        raise ValueError('Refined landmark cameras do not match the original detector geometry.')
    zyx=np.asarray(xyz,float)[:,::-1]
    internal=(zyx@inverse[:3,:3].T+inverse[:3,3])[:,::-1]
    world_A=(internal-(coarse_size-1)/2)*coarse_pixel
    delta=np.einsum('tab,kb->kta',(cameras-original)[:,:2],world_A)
    delta+=translations[None]-baseline[None]
    result_centres=centres+delta*detector_sign/coarse_pixel
    rows=detector_sign[None,:,None]*np.einsum('tab,bc->tac',cameras[:,:2],world_basis)
    result_rotations=np.concatenate((rows,np.cross(rows[:,0],rows[:,1])[:,None]),axis=1)
    if not np.isfinite(result_centres).all():raise ValueError('Non-finite refined particle centres.')
    return result_centres,result_rotations



# Same interpolation arithmetic as the single-view reference kernel. A thread
# owns its output voxel throughout a batch, so no atomics or changed reduction
# order are needed. Each half receives observations in their original order.




def native_beam_defocus_delta(rotation, centred_xyz, pixel_size_A, flip_y, hand):
    """Defocus offset in Angstrom from the alignment's physical beam depth.

    XYZ and pixel_size_A must share a sampling. CTF image-plane alpha/beta
    offsets are not alignment rotations and must not be mixed into this model.
    Raster flip reverses the exported beam convention, not the fitted hand.
    Accept a point or a batch of points; no detector-plane inversion is needed.
    """
    from .particle_ctf import native_beam_defocus_delta as delta
    return delta(rotation,centred_xyz,pixel_size_A,flip_y,hand)


def particle_defocus(ctf, geometry, raw_xy_native, native_shape, rotation,
                     centred_xyz, target_pixel, flip_y, hand):
    """Central fitted defocus plus the alignment's physical particle depth.

    Rotation maps exported tomogram XYZ to the unchanged MC detector. The
    correction is a beam depth, not simply the tomogram Z coordinate.
    """
    from .ctf_estimator import TiltDefocusGeometry
    if not geometry or geometry.get('particle_gradient_model','native_beam') == 'native_beam':
        delta = native_beam_defocus_delta(rotation, centred_xyz, target_pixel,
                    flip_y, (geometry or {}).get('handedness', hand))
        return ctf['defocus_min_A']+delta, ctf['defocus_max_A']+delta
    mean=.5*(ctf['defocus_min_A']+ctf['defocus_max_A'])
    if geometry:
        g=TiltDefocusGeometry(tilt_deg=geometry.get('tilt_deg',ctf['angle_deg']),
            tilt_axis_deg=geometry.get('tilt_axis_deg',0.),pixel_size_A=ctf['pixel_size_A'],
            handedness=geometry.get('handedness',hand),
            alpha_offset_deg=geometry.get('alpha_offset_deg',0),beta_offset_deg=geometry.get('beta_offset_deg',0),
            gradient_A_per_pixel=geometry.get('gradient_A_per_pixel'))
        origin=(np.asarray(native_shape,dtype=float)-1)/2 if geometry.get("gradient_A_per_pixel") is not None or geometry.get("coordinate_origin")=="image_pixel_centres" else np.asarray(native_shape,dtype=float)/2
        mean=float(g.local_mean_defocus_A(mean,raw_xy_native[0]-origin[1],raw_xy_native[1]-origin[0]))
    # Compare the point with the z=0 point at exactly the same projection.
    r=np.asarray(centred_xyz,float);R=np.asarray(rotation,float)
    surface=np.linalg.solve(R[:2,:2],R[:2]@r)
    depth=R[2]@r-R[2,:2]@surface
    # Plane and particle height must use the same fitted physical hand.
    effective_hand=geometry.get('handedness',hand) if geometry else hand
    mean+=effective_hand*(-1 if flip_y else 1)*target_pixel*depth
    delta=mean-.5*(ctf['defocus_min_A']+ctf['defocus_max_A'])
    return ctf['defocus_min_A']+delta,ctf['defocus_max_A']+delta
