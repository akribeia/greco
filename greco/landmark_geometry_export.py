"""Rigid geometry export in Greco's raster and canonical volume conventions.

The fitted cameras/XYZ remain in the internal world frame. Display rotation,
canonical Z reflection and volume cropping are EXPORT operations only.
"""
from greco.volume_storage import float16_data
from greco.project_layout import join as storage_path, ensure_output, job_work
import json
from pathlib import Path
import numpy as np


def wiener_at_sampling(path,shape,pixel,xp):
    """Reuse a final stack only on the same physical detector grid.

    A finer final reconstruction can be Fourier-binned to the geometry preview.
    A coarser one must be replayed from movies, never upsampled as observations.
    """
    import mrcfile
    from .fourier_binning import fourier_bin_2d,fourier_bin_2d_gpu
    with mrcfile.mmap(path) as m:
        sampling=np.asarray(m.voxel_size.tolist(),float)
        if not np.isfinite(sampling).all() or (sampling<=0).any() or not np.allclose(sampling,sampling[0]):
            raise ValueError('Final Wiener stack has invalid or anisotropic sampling.')
        if m.data.ndim!=3 or m.data.shape[0]!=shape[0] or not np.allclose(
                np.asarray(m.data.shape[-2:])*sampling[0],np.asarray(shape[-2:])*pixel,rtol=1e-5):
            raise ValueError('Final Wiener stack tilt count or physical field of view differs from geometry inputs.')
        ratio=pixel/float(sampling[0])
        if ratio<1-1e-5:return None
        factor=int(round(ratio))
        if not np.isclose(ratio,factor,rtol=1e-5) or tuple(n//factor for n in m.data.shape[-2:])!=tuple(shape[-2:]):
            raise ValueError('Final Wiener sampling cannot be binned onto the geometry input grid.')
        if factor==1:return np.array(m.data,np.float32)
        result=np.empty(shape,np.float32)
        for i,plane in enumerate(m.data):
            result[i]=(fourier_bin_2d(plane,factor) if xp is np else
                       xp.asnumpy(fourier_bin_2d_gpu(xp.asarray(plane),factor)))
        return result


def display_geometry(cameras,translations,axis,fit_size,source_size,flip_y=False):
    from .landmark_rigid import initial_cameras
    from .angular_joint import rescale_geometry
    rz=initial_cameras(np.array([0.]),axis)[0]
    # t_raw = -R_axis * shift_aligned; frozen MC shifts are already in t_raw.
    shifts=(-np.asarray(translations)@rz[:2,:2])[:,::-1]
    _,coarse=rescale_geometry(np.zeros((1,3)),shifts,fit_size,source_size,axis,flip_y)
    return shifts,coarse,np.einsum('ab,tbc->tac',rz.T,np.asarray(cameras))


def export_outputs(data,geometry,output,xp,*,review_only=False):
    import mrcfile
    from .gcc import materialize_raw_stack_gpu
    from .rigid_backprojection import RigidCameraWBP,ramp_filter
    from .landmark_rigid import initial_cameras
    from .tomogram_thickness import slab_bounds
    from .final_reconstruction import export_transform
    output=Path(output);coarse=Path(data['coarse_run']);orientation=data['orientation']
    with mrcfile.mmap(data['inputs']['stack']) as m:
        raw=np.array(m.data,np.float32);pixel=float(m.voxel_size.x)
    if orientation.get('flip_y'):raw=raw[:,::-1,:].copy()
    n=raw.shape[-1];fit_size=int(geometry['fit_size']);fit_pixel=float(geometry['pixel_size_A'])
    active=np.asarray(geometry.get('active_indices',np.arange(len(raw))),int)
    if active.ndim!=1 or len(active)<3 or np.any(active<0) or np.any(active>=len(raw)) or np.any(np.diff(active)<=0):
        raise ValueError('Invalid geometry export tilt selection.')
    translations=np.asarray(geometry['translations_xy_A'])/fit_pixel
    fine,shifts,cameras=display_geometry(geometry['cameras_xyz'],translations,
        orientation['rotation_deg'],fit_size,n,orientation.get('flip_y',False))
    aligned=materialize_raw_stack_gpu(xp.asarray(raw),active,shifts[active],
        np.zeros_like(shifts[active]),orientation['rotation_deg'])
    path=storage_path(output, 'aligned_bin8.mrcs')
    with mrcfile.new(path,overwrite=True) as m:
        m.set_data(xp.asnumpy(aligned));m.set_image_stack();m.voxel_size=pixel
    if review_only:
        return dict(aligned_stack=str(path),active_indices=active.tolist(),full_range=len(active)==len(raw),
            export_protocol='main-raster-canonical-z-v1',output_raster_convention='MAIN_axis_aligned')
    # Use the SAME final Wiener (no reliability) observations as Reconstruction.
    physical=storage_path(coarse.parent, 'final_wiener_rawsource.mrcs')
    source_config=storage_path(job_work(coarse), 'config.json')
    settings=json.loads(source_config.read_text()) if source_config.is_file() else {}
    source_binning=int(data['inputs'].get('binning',8))
    final_binning=data.get('final_binning') or source_binning
    if type(final_binning) is not int or final_binning not in (1,2,4,8,16):raise ValueError('Invalid final reconstruction binning.')
    if n*source_binning%final_binning:raise ValueError('Final binning does not divide the recorded field of view.')
    target_n=n*source_binning//final_binning
    target_pixel=pixel*final_binning/source_binning
    target_shape=(len(raw),target_n,target_n)
    final_raw=wiener_at_sampling(physical,target_shape,target_pixel,xp) if physical.is_file() else None
    if final_raw is None:
        from .final_reconstruction import final_stack
        replay=storage_path(output,'final_observations');ensure_output(replay)
        config=dict(settings,inputs=dict(data['inputs'],binning=final_binning))
        final_raw,_=final_stack(config,coarse,replay,return_raw=True)
        if final_raw.shape!=target_shape:raise ValueError('Replayed Wiener stack differs from final output grid.')
    n=target_n;pixel=target_pixel
    _,shifts,cameras=display_geometry(geometry['cameras_xyz'],translations,
        orientation['rotation_deg'],fit_size,n,orientation.get('flip_y',False))
    if orientation.get('flip_y'):final_raw=final_raw[:,::-1,:].copy()
    preview=materialize_raw_stack_gpu(xp.asarray(final_raw),active,shifts[active],
        np.zeros_like(shifts[active]),orientation['rotation_deg'])
    sign=1.;filter_source='native_wiener_no_reliability'
    thickness=settings.get('tomogram_thickness_percent',50.)
    z0,z1=slab_bounds(n,thickness)
    filtered=ramp_filter(preview,cameras[active],xp)
    reconstructor=RigidCameraWBP(filtered,cameras[active],initial_cameras(np.asarray(geometry['angles_deg'])[active],0.),xp)
    tomo=storage_path(output, 'tomogram_bin8.mrc')
    # Canonical Z reflection, then central crop: same convention as Greco.
    with mrcfile.new_mmap(tomo,shape=(z1-z0,n,n),mrc_mode=12,overwrite=True) as m:
        m.voxel_size=pixel
        for first in range(z0,z1,8):
            last=min(first+8,z1)
            slab=reconstructor.slab(n-last,last-first)
            m.data[first-z0:last-z0]=float16_data(sign*xp.asnumpy(slab[::-1]))
    return dict(aligned_stack=str(path),tomogram=str(tomo),preview_shape=[z1-z0,n,n],
        final_binning=final_binning,voxel_size_A=pixel,
        active_indices=active.tolist(),full_range=len(active)==len(raw),
        volumes=dict(original=str(tomo)),output_contrast_convention='dark_particles',
        output_raster_convention='MAIN_axis_aligned',export_z_reflected=True,
        export_z_crop_bounds=[z0,z1],tomogram_thickness_percent=thickness,
        internal_to_export_zyx=export_transform(n,z0),internal_volume_shape_zyx=[n,n,n],
        final_observation_source=filter_source,export_protocol='main-raster-canonical-z-v1',
        final_wbp='main_discrete_ram_lak_full_cameras',final_wbp_padding=True,
        final_projection_background='per_projection_mean_before_zero_padding_v1',
        final_wbp_normalization='pi/(2*number_of_tilts)')
