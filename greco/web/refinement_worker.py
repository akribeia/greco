"""Replay raw MC sources at final sampling, then refine device-resident patches and models."""
from greco.project_layout import join as storage_path, ensure_output
from greco.preprocessing_star import read_table
import argparse

import json
from pathlib import Path
import traceback
import numpy as np
from ..refine_landmarks import lift_geometry


def clean_json(value):
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def load_state(path,expected_angles,*,allow_partial=False):
    with np.load(path) as d:
        positions=np.array(d['positions_zyx'],float);cumulative=np.array(d['cumulative_shifts_yx'],float)
        valid=np.array(d['valid_views'],bool);classes=np.array(d['classes']);angles=np.array(d['angles_deg'],float)
        active=np.array(d['active_indices'],int);b=float(d['binning'])
    if active.ndim!=1 or len(active)<3 or np.any(active<0) or np.any(active>=len(angles)) or np.any(np.diff(active)<=0):
        raise ValueError('Invalid accepted tilt selection.')
    if not allow_partial and (len(active)!=len(angles) or not np.array_equal(active,np.arange(len(angles)))):
        raise ValueError('Final-resolution refinement requires a full-range accepted state, not an attempted/partial stage.')
    if allow_partial and len(active)!=len(angles) and valid.shape==(len(positions),len(active)):
        accepted=valid;valid=np.zeros((len(positions),len(angles)),bool);valid[:,active]=accepted
    if allow_partial and valid.shape==(len(positions),len(angles)):
        valid[:,~np.isin(np.arange(len(angles)),active)]=False
    if valid.shape!=(len(positions),len(angles)) or cumulative.shape!=(len(angles),2) or classes.shape!=(len(positions),):
        raise ValueError('Accepted state has inconsistent track/tilt dimensions.')
    if positions.shape!=(len(positions),3) or not np.isfinite(positions).all() or not np.isfinite(cumulative).all():
        raise ValueError('Invalid accepted geometry.')
    if angles.shape!=np.shape(expected_angles) or not np.allclose(angles,expected_angles,atol=1e-4):
        raise ValueError('Accepted tilt identities/order do not match preprocessing angles.')
    return positions,cumulative,valid,classes,angles,b


def load_angle_geometry(path,positions,valid,angles,binning,axis,*,return_positions=False):
    """Accept only a calibration of this exact accepted landmark geometry."""
    with np.load(path) as d:
        if not np.array_equal(d['active_indices'],np.arange(len(angles))) or float(d['binning'])!=binning:
            raise ValueError('Angle calibration range/binning mismatch.')
        if not np.allclose(d['original_angles_deg'],angles,atol=1e-6) or not np.isclose(float(d['original_axis_deg']),axis,atol=1e-6):
            raise ValueError('Angle calibration original geometry mismatch.')
        origin_pos=d['original_positions_zyx'] if 'original_positions_zyx' in d else d['positions_zyx']
        origin_valid=d['original_valid_views'] if 'original_valid_views' in d else d['valid_views']
        if not np.array_equal(origin_pos,positions) or not np.array_equal(origin_valid,valid):
            raise ValueError('Angle calibration landmark geometry mismatch.')
        new_positions=np.array(d['positions_zyx'],float)
        new_valid=np.array(d['valid_views'],bool)
        if new_positions.shape!=positions.shape or not np.isfinite(new_positions).all() or new_valid.shape!=valid.shape or np.any(new_valid & ~valid):
            raise ValueError('Invalid refined landmark geometry/support.')
        new_angles=np.array(d['angles_deg'],float);shifts=np.array(d['cumulative_shifts_yx'],float)
        new_axis=float(d['axis_rotation_deg'])
    if new_angles.shape!=angles.shape or shifts.shape!=(len(angles),2) or not all(np.isfinite(v).all() for v in (new_angles,shifts,[new_axis])):
        raise ValueError('Invalid calibrated angles/shifts.')
    if np.any(np.abs(new_angles)>=89) or not np.all(np.diff(new_angles)*np.diff(angles)>0):
        raise ValueError('Invalid calibrated tilt order/range.')
    return (new_angles,shifts,new_axis,new_positions) if return_positions else (new_angles,shifts,new_axis)


def bin_plane_device(plane,factor,xp):
    """Fourier resample on device, including even-grid Nyquist folding."""
    if factor==1:return plane
    if any(n%factor for n in plane.shape):raise ValueError('Image dimensions must divide output binning.')
    shape=tuple(n//factor for n in plane.shape)
    indices=[]
    for n,source in zip(shape,plane.shape):
        i=xp.arange(n);signed=xp.where(i<=n//2,i,i-n)
        if n%2==0:signed[n//2]=-n//2
        indices.append(signed%source)
    y,x=indices;spectrum=xp.fft.fft2(plane)
    crop=spectrum[y[:,None],x[None,:]].copy()
    if shape[0]%2==0:crop[shape[0]//2,:]+=spectrum[shape[0]//2,x]
    if shape[1]%2==0:crop[:,shape[1]//2]+=spectrum[y,shape[1]//2]
    if all(n%2==0 for n in shape):crop[shape[0]//2,shape[1]//2]+=spectrum[shape[0]//2,shape[1]//2]
    yp=(-xp.arange(shape[0]))%shape[0];xp_=(-xp.arange(shape[1]))%shape[1]
    crop=.5*(crop+crop[yp[:,None],xp_[None,:]].conj())/(factor*factor)
    result=xp.fft.ifft2(crop).real
    result+=plane.mean()-result.mean()
    return result.astype(xp.float32)


def rebuild_native(config,angles):
    """Replay saved MC shifts and saved CTF fits; no new MC/CTF estimation."""
    import cupy as cp
    from types import SimpleNamespace
    from .. import movie_replay as p
    prep=Path(config['preprocessing_work']);source=json.loads((storage_path(prep, 'config.json')).read_text())
    options=source['options'];series=source['series']
    mdoc=p._movie_mdoc(SimpleNamespace(mdoc=storage_path(prep, 'input.mdoc'),eer_upscaling=options.get('eer_upscaling',1)))
    rows=read_table(config['inputs']['frozen_shifts'])
    if len(rows)!=len(angles) or not np.allclose([float(r['angle_deg']) for r in rows],angles,atol=1e-4):
        raise ValueError('Frozen shifts are not in accepted angle order.')
    gain=p._load_gain(series.get('gain_path'),flip_y=series.get('gain_flip_y',False))
    if gain is not None and options.get('eer_upscaling',1)==2 and all(Path(t.movie).suffix.lower()=='.eer' for t in mdoc.tilts) and gain.shape==(4096,4096):
        gain=np.repeat(np.repeat(gain,2,axis=0),2,axis=1)
    dev=p._Dev(True);stack=None
    target_bin=config['options'].get('output_binning',1)
    frozen=np.array([[float(r['dy_raw']),float(r['dx_raw'])] for r in rows])
    for k,row in enumerate(rows):
        index=int(row['mdoc_index']);tilt=mdoc.tilts[index]
        if abs(tilt.angle_deg-angles[k])>1e-4:raise ValueError('MDOC index/angle mismatch.')
        directory=storage_path(storage_path(storage_path(prep, 'output'), 'per_tilt'), f'tilt_{index:04d}')
        ctf=json.loads((storage_path(directory, 'ctf.json')).read_text())
        if abs(ctf['angle_deg']-angles[k])>1e-4:raise ValueError('CTF index/angle mismatch.')
        frames,_=p.load_movie(storage_path(storage_path(prep, 'movies'), tilt.movie),tilt.n_frames,
            eer_group=options.get('eer_group',15),eer_threads=options.get('eer_threads',1),
            eer_upscaling=options.get('eer_upscaling',1))
        frames=p._prepare_frames(frames,gain,options.get('flip_y',False),options.get('transpose',False))
        with np.load(storage_path(directory, 'shifts.npz')) as shifts:motion=np.array(shifts['full_shifts'])
        if motion.shape!=(len(frames),2):raise ValueError('Saved MC shifts and decoded movie groups differ.')
        frame_device=cp.asarray(frames);del frames
        from ..detector_defects import apply_correction
        apply_correction(frame_device,p._load_detector_correction(directory))
        plane=p._rebuild_aligned_sum(dev,frame_device,motion);del frame_device
        params=p.CTFParameters(voltage_kV=mdoc.voltage_kv,cs_mm=options['cs_mm'],
            amplitude_contrast=options['amp_contrast'],defocus_u_A=ctf['defocus_min_A'],
            defocus_v_A=ctf['defocus_max_A'],astigmatism_angle_deg=ctf['astig_azimuth_raw_deg'],phase_shift_rad=ctf['phase_rad'])
        if options.get('ctf_spatial',False):
            geometry=json.loads((storage_path(directory, 'ctf_geometry.json')).read_text()) if (storage_path(directory, 'ctf_geometry.json')).exists() else {}
            corrected=p.correct_tilt_ctf_spatial(cp.asnumpy(plane),pixel_size_A=mdoc.pixel_size_A,
                tilt_deg=tilt.angle_deg,tilt_axis_deg=geometry.get('tilt_axis_deg',options.get('ctf_spatial_axis_deg')),
                defocus_u_A=params.defocus_u_A,defocus_v_A=params.defocus_v_A,
                astigmatism_angle_deg=params.astigmatism_angle_deg,voltage_kV=params.voltage_kV,
                cs_mm=params.cs_mm,amplitude_contrast=params.amplitude_contrast,phase_shift_rad=params.phase_shift_rad,
                regularization=options['ctf_regularization'],handedness=geometry.get('handedness',options['df_hand']),
                alpha_offset_deg=geometry.get('alpha_offset_deg',0),beta_offset_deg=geometry.get('beta_offset_deg',0),
                gradient_A_per_pixel=geometry.get('gradient_A_per_pixel'),
                coordinate_origin=geometry.get('coordinate_origin'),gpu=True)
            plane=cp.asarray(corrected);del corrected
        else:
            inverse=p.regularized_ctf_inverse(p.ctf_transfer(plane.shape,mdoc.pixel_size_A,params),options['ctf_regularization'])
            plane=cp.fft.ifft2(cp.fft.fft2(plane)*cp.fft.ifftshift(cp.asarray(inverse))).real.astype(cp.float32)
        plane=bin_plane_device(plane,target_bin,cp)
        if config['orientation'].get('flip_y',False):plane=plane[::-1]
        if stack is None:stack=cp.empty((len(angles),*plane.shape),cp.float32)
        if plane.shape!=stack.shape[1:] or not bool(cp.isfinite(plane).all()):raise ValueError('Invalid native plane.')
        stack[k]=plane
        print(f'Native source {k+1}/{len(rows)} rebuilt from saved MC/CTF',flush=True)
        del plane
        cp.get_default_memory_pool().free_all_blocks()
    if config['orientation'].get('flip_y',False):frozen[:,0]*=-1
    return stack,frozen/target_bin,mdoc.pixel_size_A*target_bin


def save_landmarks(stack,frozen,orientation,state,size,pixel,output,batch_size=8):
    """Save final local full-support cubes as contiguous Z blocks in one MRC."""
    import cupy as cp
    import mrcfile
    from ..helpers import projected_centres
    from ..gcc import raw_sample_bases,raw_patch_stacks_gpu
    from ..tomography import gpu_reconstruct_wbp_batch_device
    valid=state['valid_views'];positions=state['positions_zyx'];angles=state['angles_deg']
    centres=projected_centres(positions,angles,stack.shape[-1])
    by,bx=raw_sample_bases(centres,state['cumulative_shifts_yx'],frozen,orientation['rotation_deg'],stack.shape[-1])
    ids=np.flatnonzero(valid.sum(axis=1)>=2)
    if not len(ids):raise ValueError('No supported final landmarks to save.')
    path=storage_path(output, 'landmarks.mrc');records=[]
    with mrcfile.new_mmap(path,shape=(len(ids),size,size,size),mrc_mode=2) as handle:
        handle.voxel_size=pixel
        for start in range(0,len(ids),batch_size):
            selected=ids[start:start+batch_size]
            patches,inside=raw_patch_stacks_gpu(stack,by[selected],bx[selected],valid[selected],size//2,orientation['rotation_deg'])
            support=valid[selected]&inside
            if np.any(support.sum(axis=1)<2):raise ValueError('Final landmark lost raw support.')
            cubes=-gpu_reconstruct_wbp_batch_device(patches,angles,valid_bt=support,y_chunk=8)
            handle.data[start:start+len(selected)]=cp.asnumpy(cubes)
            records.extend(dict(landmark_id=int(i),z_start=(start+j)*size,z_stop=(start+j+1)*size,
                position_zyx=positions[i].tolist(),tilt_indices=np.flatnonzero(support[j]).tolist()) for j,i in enumerate(selected))
        handle.update_header_stats()
    (storage_path(output, 'landmarks.json')).write_text(json.dumps(dict(cube_size=size,voxel_size_A=pixel,
        layout='MRC2014 volume stack (landmark,z,y,x); consecutive Z blocks are independent cubes, NOT a global tomogram',
        contrast_inverted=True,landmarks=records),indent=2))
    return str(path)


def run(config):
    data=json.loads(config.read_text());output=Path(data['output']);ensure_output(output)
    try:
        import cupy as cp
        import mrcfile
        from ..refine_landmarks import refine_landmarks
        from .refinement import validate_options
        options=validate_options(data['options']);size=options['cube_size'];q=options['output_binning']
        angles=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
        pos,shifts,valid,classes,angles,b=load_state(storage_path(Path(data['coarse_run']), 'final_state.npz'),angles)
        original_angles=angles.copy();orientation=dict(data['orientation'])
        if data.get('angle_geometry'):
            angles,shifts,orientation['rotation_deg'],pos=load_angle_geometry(data['angle_geometry'],pos,valid,angles,b,orientation['rotation_deg'],return_positions=True)
        if b!=data['inputs']['binning']:raise ValueError('Accepted state binning differs from source preprocessing.')
        if b%q:raise ValueError('Output binning must divide input binning.')
        scale=b//q
        with mrcfile.mmap(data['inputs']['stack'],mode='r') as m:
            coarse_size=m.data.shape[-1];expected_pixel=float(m.voxel_size.x)/scale
        stack,frozen,pixel=rebuild_native(data,original_angles)
        if not np.isclose(pixel,expected_pixel,rtol=1e-5):raise ValueError('Native pixel size does not match accepted coarse source.')
        pos,shifts=lift_geometry(pos,shifts,scale,coarse_size,stack.shape[-1],orientation['rotation_deg'],orientation.get('flip_y',False))
        cp.get_default_memory_pool().free_all_blocks()
        free=cp.cuda.runtime.memGetInfo()[0]
        # Persistent sums + patches + bounded transfer cache; transient batch
        # includes projection coordinate arrays and complex FFT workspaces.
        persistent=len(pos)*(size**3*4+len(angles)*size**2*4)
        batch=min(8,max(1,int((free*.60-persistent)/(size**3*96))))
        if persistent+batch*size**3*96>free*.65:raise ValueError('Insufficient GPU RAM for this cube size/landmark count; choose a smaller cube.')
        bands=[];band=.8/scale
        while band<.8:bands.append(band);band=min(.8,band*2)
        bands.append(.8)
        print(f'Final-resolution refinement {size}³, {len(pos)} landmarks, GPU batch {batch}, bands {bands}',flush=True)
        final,manifest=refine_landmarks(stack,angles,dict(positions_zyx=pos,valid_views=valid,classes=classes),shifts,
            patch_radius=size//2,search_radius=8//q,raw_source=True,frozen_shifts_yx=frozen,
            axis_deg=orientation['rotation_deg'],batch_size=batch,solver_backend='gpu_lsmr32',
            robust_scale=.2*scale,lowpass_schedule=bands,stop_shift=.10/q,stop_shift_p90=.25/q,plateau_eps=.05/q)
        state_path=storage_path(output, f'refine_bin{q}.npz');shifts_path=storage_path(output, f'shifts_bin{q}.csv');report_path=storage_path(output, f'refine_bin{q}_report.json')
        np.savez_compressed(state_path,**final,binning=q,input_binning=b,coordinate_pixel_size_A=pixel,
            original_angles_deg=original_angles,axis_rotation_deg=orientation['rotation_deg'])
        np.savetxt(shifts_path,np.column_stack([np.arange(len(angles)),angles,shifts,final['cumulative_shifts_yx']]),delimiter=',',
            header='plane,angle_deg,dy_initial,dx_initial,dy_refined,dx_refined',comments='')
        import hashlib
        code_root=Path(__file__).resolve().parents[1]
        manifest['code_sha256']={name:hashlib.sha256((code_root/name).read_bytes()).hexdigest()
            for name in ('refine_landmarks.py','gcc.py','solver.py','local_loo_gpu.py','web/refinement_worker.py')}
        manifest.update(input_binning=b,output_binning=q,coordinate_units=f'bin{q} pixels',cube_size=size,
                        batch_size=batch,source_orientation=data['orientation'],effective_orientation=orientation,
                        angle_geometry=data.get('angle_geometry'),angles_fixed_during_refinement=True,
                        main_revision=data['main_revision'])
        report_path.write_text(json.dumps(clean_json(manifest),indent=2,allow_nan=False))
        good=manifest['stop_reason'] in ('converged','plateau') and manifest['final_lowpass_fraction']==bands[-1]
        result=dict(state='completed' if good else 'partial',stop_reason=manifest['stop_reason'],passes=manifest['passes'],
            runtime_s=manifest['runtime_s'],cube_size=size,output_binning=q,coordinate_units=f'bin{q} pixels',main_revision=data['main_revision'],
            state_file=str(state_path),shifts_file=str(shifts_path),report=str(report_path))
        if not good:result['error']='Refinement stopped before completing the native-frequency protocol: '+manifest['stop_reason']
        if options['save_landmarks']:result['landmarks_file']=save_landmarks(stack,frozen,orientation,final,size,pixel,output,batch)
    except Exception as error:
        traceback.print_exc();result=dict(state='failed',error=str(error))
    temporary=storage_path(output, '.web_result.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(storage_path(output, 'web_result.json'))
    return 1 if result['state']=='failed' else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().config))
