"""Final Wiener/WBP volumes at independent segmentation sampling."""
from greco.volume_storage import float16_data
from greco.project_layout import join as storage_path, ensure_output, job_work, preprocessing_work
from greco.preprocessing_star import read_table
import argparse, json, time, traceback
from pathlib import Path
import numpy as np


def scale_shifts(shifts,source_size,target_size,factor,axis,flip):
    from ..gcc import raw_affine
    if np.ndim(axis):
        from ..angular_joint import rescale_geometry
        return rescale_geometry(np.zeros((1,3)),shifts,source_size,target_size,axis,flip)[1]
    a,b,c,d,_=raw_affine(axis,target_size);rotation=np.array([[a,b],[c,d]])
    delta=(target_size-1)/2-factor*(source_size-1)/2
    raw_offset=np.array([factor-1 if flip else 0.,0.])
    return np.asarray(shifts)*factor+rotation.T@(np.full(2,delta)-raw_offset)


def check_tilt_angles(saved, expected, message):
    """Compare ordered identities allowing decimal rounding in angle exports."""
    saved=np.asarray(saved,dtype=float);expected=np.asarray(expected,dtype=float)
    if saved.ndim!=1 or expected.ndim!=1 or not len(saved) or saved.shape!=expected.shape or not np.allclose(saved,expected,rtol=0,atol=1e-4):
        raise ValueError(message)


def reconstruct(data,output,raw_override=None,suffix=''):
    import cupy as cp,mrcfile
    from ..fourier_binning import fourier_bin_2d_gpu
    from ..gcc import materialize_raw_stack_gpu
    from ..landmark_rigid import initial_cameras
    from ..rigid_backprojection import RigidCameraWBP,ramp_filter
    from ..tomogram_thickness import slab_bounds
    source=data['source_config'];coarse=Path(data['coarse_run']);b=int(source['inputs'].get('binning',8));q=data['binning']
    angles=np.loadtxt(source['inputs']['rawtlt'],ndmin=1)
    with np.load(storage_path(coarse, 'final_state.npz')) as state:
        active=np.asarray(state['active_indices'],int)
        cumulative=state['cumulative_shifts_yx'][active]
        check_tilt_angles(state['angles_deg'],angles,'Segmentation angle identities differ.')
        from ..alignment_geometry import resolve,scalar_or_array
        fitted_angles,fitted_axes=resolve(state,angles,source['orientation']['rotation_deg'])
        from ..alignment_geometry import aligned_cameras
        full_cameras=aligned_cameras(state,angles,source['orientation']['rotation_deg']) if 'axis_inclination_deg' in state else None
        reference_angles=np.asarray(state.get('gamma_reference_angles_deg',fitted_angles),float)
    if not data.get('geometry_file'):
        rows=read_table(source['inputs']['frozen_shifts'])
        check_tilt_angles([float(r['angle_deg']) for r in rows],angles,'Frozen shift identities differ.')
    # Use the coarse job's frozen final-filter configuration and thickness.
    original=json.loads((storage_path(job_work(coarse), 'config.json')).read_text())
    orientation=source['orientation'];axis=orientation['rotation_deg'];flip=bool(orientation.get('flip_y'))
    cached=storage_path(coarse.parent, 'final_wiener_rawsource.mrcs')
    with mrcfile.mmap(source['inputs']['stack']) as m:source_size=m.data.shape[-1]
    ctf_metadata=None
    cached_binning=original.get('final_binning') or b
    if raw_override is not None:
        raw,pixel=raw_override
    elif cached.is_file() and q>=cached_binning and q%cached_binning==0:
        with mrcfile.mmap(cached) as m:
            pixel=float(m.voxel_size.x)*q/cached_binning
            planes=[cp.asnumpy(fourier_bin_2d_gpu(cp.asarray(plane),q//cached_binning)) if q!=cached_binning else np.array(plane,np.float32) for plane in m.data]
        raw=np.stack(planes);del planes
    else:
        from ..final_reconstruction import final_stack
        config=dict(original,inputs=dict(original['inputs'],binning=q))
        raw,ctf_metadata=final_stack(config,coarse,output,return_raw=True)
        with mrcfile.mmap(storage_path(output, 'final_wiener_rawsource.mrcs')) as m:pixel=float(m.voxel_size.x)
    if flip:raw=raw[:,::-1].copy()
    n=raw.shape[-1]
    if data.get('geometry_file'):
        with np.load(data['geometry_file']) as geometry:
            check_tilt_angles(geometry['angles_deg'],angles,'Refined camera identities differ.')
            rz=initial_cameras([0],axis)[0]
            shifts=(-geometry['translations_xy_A']/float(geometry['pixel_size_A'])@rz[:2,:2])[:,::-1]
            shifts=scale_shifts(shifts,int(geometry['fit_size']),n,float(geometry['pixel_size_A'])/pixel,axis,flip)[active]
            cameras=np.einsum('ab,tbc->tac',rz.T,geometry['cameras_xyz'])[active]
        frozen=np.zeros_like(shifts)
    else:
        angles=fitted_angles;axis=scalar_or_array(fitted_axes[active])
        shifts=scale_shifts(cumulative,source_size,n,b/q,axis,flip)
        frozen=np.array([[float(r['dy_raw']),float(r['dx_raw'])] for r in rows])/q
        if flip:frozen[:,0]*=-1
        frozen=frozen[active];cameras=initial_cameras(angles[active],0) if full_cameras is None else full_cameras[active]
    images=materialize_raw_stack_gpu(cp.asarray(raw),active,shifts,frozen,axis);del raw
    percent=original.get('tomogram_thickness_percent',50.);z0,z1=slab_bounds(n,percent)
    from ..optimizer.leveling import resolve as resolve_leveling
    leveling_rotation,leveling_transform,leveling_report=resolve_leveling(source,coarse,n,z0)
    # Filter in the detector geometry; rotate only the reconstruction frame.
    filtered=ramp_filter(images,cameras,cp);del images
    cameras=cameras@leveling_rotation[::-1,::-1].T
    wbp=RigidCameraWBP(filtered,cameras,initial_cameras((reference_angles if full_cameras is not None and not data.get('geometry_file') else angles)[active],0),cp)
    percent=original.get('tomogram_thickness_percent',50.);z0,z1=slab_bounds(n,percent)
    target=storage_path(output, 'tomogram'+suffix+'.mrc')
    with mrcfile.new_mmap(target,shape=(z1-z0,n,n),mrc_mode=12,overwrite=False) as m:
        m.voxel_size=pixel
        for first in range(z0,z1,8):
            end=min(first+8,z1);m.data[first-z0:end-z0]=float16_data(cp.asnumpy(wbp.slab(n-end,end-first)[::-1]))
            print(f'Segmentation WBP: {end-z0}/{z1-z0} slices',flush=True)
        m.update_header_stats()
    if ctf_metadata is not None:
        ctf_metadata=dict(ctf_metadata,ctf_data=[ctf_metadata['ctf_data'][int(i)] for i in active])
    return dict(tomogram_leveling=leveling_report,internal_to_export_zyx=leveling_transform.tolist(),ctf_template_metadata=ctf_metadata,volumes=dict(original=str(target)),shape=[z1-z0,n,n],voxel_size_A=pixel,
        geometry_file=data.get('geometry_file'),binning=q,source_work=data['source_work'],
        reconstruction_protocol='native_wiener_no_reliability_MAIN_WBP_canonical_z',
        final_projection_background='per_projection_mean_before_zero_padding_v1',
        tomogram_thickness_percent=percent,output_contrast_convention='dark_particles')


def run(path):
    from .particle_worker import check_snapshot
    data=json.loads(Path(path).read_text());output=Path(data['output']);ensure_output(output)
    start=time.monotonic()
    try:
        check_snapshot(data['snapshot'])
        if data['action']=='insert_particle':
            from ..particle_insertion import insert_particles
            result=insert_particles(data,output,use_gpu=True)
        elif data['action']=='membranes':
            from .segmentation_membranes import run as membranes
            result=membranes(data,output)
        elif data['action']=='denoise':
            if data['options'].get('method','tinycare')!='tinycare':
                raise ValueError('Only TinyCARE denoising is supported.')
            from .segmentation_tinycare import run as tinycare
            result=tinycare(data,output)
        else:
            if data['action']=='geometry_refinement':
                from .refinement import validate_geometry_options
                from ..landmark_rigid_worker import run as refine_geometry
                fit=dict(data['source_config'],coarse_run=data['coarse_run'],output=str(storage_path(output, 'geometry')),
                    options=validate_geometry_options({}),preprocessing_work=str(preprocessing_work(Path(data['source_config']['inputs']['stack']))))
                if data.get('geometry_file'):fit['resume_geometry']=data['geometry_file']
                if data.get('sta_landmarks'):fit['sta_landmarks']=data['sta_landmarks']
                fitted=refine_geometry(fit);data['geometry_file']=fitted['geometry_file']
            result=reconstruct(data,output)
            if data.get('sta_landmarks'):
                result['sta_source']=data['sta_landmarks']['selection']
                result['sta_particle_count']=len(data['sta_landmarks']['particle_ids'])
        check_snapshot(data['snapshot']);result.update(state='completed',runtime_s=time.monotonic()-start)
    except Exception as error:
        traceback.print_exc();result=dict(state='failed',error=str(error))
    temporary=storage_path(output, 'web_result.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(storage_path(output, 'web_result.json'))
    if result['state']=='failed':raise RuntimeError(result['error'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);run(p.parse_args().config)
