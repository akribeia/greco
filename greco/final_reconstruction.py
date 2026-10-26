"""Final-volume observation chain, independent of alignment preconditioning.

Reuse saved MC shifts and detector corrections; never re-fit geometry.
Final CTF uses native Wiener SZ, without even/odd reliability. Alignments
remain in the historical internal frame; export explicitly reflects Z.
"""
from greco.project_layout import join as storage_path, preprocessing_work
from greco.preprocessing_star import read_table

import json
from pathlib import Path
import numpy as np

PROTOCOL='native-wiener-sz-no-reliability-canonical-z-v1'


def final_stack(data, alignment, output, *, return_raw=False, split_halves=False):
    import cupy as cp
    import mrcfile
    from .movie_io import load_movie
    from .motion import _Dev
    from .preprocess import _load_gain,_prepare_frames,_load_detector_correction,_rebuild_aligned_sum
    from .detector_defects import apply_correction
    from .ctf_spatial import correct_tilt_ctf_spatial
    from .fourier_binning import fourier_bin_2d_gpu
    from .gcc import materialize_raw_stack_gpu

    prep=preprocessing_work(Path(data['inputs']['stack']))
    settings=json.loads((storage_path(preprocessing_work(prep), 'config.json')).read_text())
    opts=settings['options'];series=settings['series'];binning=int(data['inputs']['binning'])
    rows=read_table(storage_path(prep,'ctf/ctf.star'))
    angles=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
    np.testing.assert_allclose([float(r['tilt_deg']) for r in rows],angles,atol=1e-4)
    tilts={int(t['z']):t for t in series['tilts']}
    gain=_load_gain(series.get('gain_path'),flip_y=series.get('gain_flip_y',False));dev=_Dev(True)
    if split_halves and not return_raw:raise ValueError('Half stacks require raw-raster output.')
    planes={name:[] for name in (('even','odd') if split_halves else ('full',))};ctfs=[]
    for i,row in enumerate(rows):
        z=int(row['z_value']);tilt=tilts[z];folder=storage_path(storage_path(prep, 'per_tilt'), f'tilt_{z:04d}')
        frames,_=load_movie(Path(tilt['movie_path']),tilt['n_frames'],
            eer_group=opts['eer_group'],eer_threads=opts['eer_threads'],eer_upscaling=opts['eer_upscaling'])
        frames=cp.asarray(_prepare_frames(frames,gain,opts['flip_y'],opts['transpose']))
        apply_correction(frames,_load_detector_correction(folder))
        with np.load(storage_path(folder, 'shifts.npz')) as saved:
            if split_halves:
                count=len(frames)
                if count<2:raise ValueError('TinyCARE needs at least two decoded frames per tilt.')
                images={name:_rebuild_aligned_sum(dev,frames,saved['full_shifts'],parity=parity)*(count/len(range(parity,count,2)))
                        for name,parity in (('even',1),('odd',0))}
            else:images={'full':_rebuild_aligned_sum(dev,frames,saved['full_shifts'])}
        del frames
        ctf=json.loads((storage_path(folder, 'ctf.json')).read_text())
        path=storage_path(folder, 'ctf_geometry.json');geometry=json.loads(path.read_text()) if path.exists() else {}
        physical_tilt=float(row['tilt_deg']);spatial=opts.get('ctf_spatial',True)
        # A global correction still uses the physical-tilt lowpass, without
        # imposing a geometric defocus gradient when the user disabled it.
        lowpass=15. if spatial else 15.*1.001/(np.cos(physical_tilt*.01745)+.001)
        correction=dict(pixel_size_A=ctf['pixel_size_A'],
            tilt_deg=geometry.get('tilt_deg',ctf['angle_deg']) if spatial else 0.,
            tilt_axis_deg=geometry.get('tilt_axis_deg',opts['ctf_spatial_axis_deg']),
            defocus_u_A=ctf['defocus_min_A'],defocus_v_A=ctf['defocus_max_A'],
            astigmatism_angle_deg=ctf['astig_azimuth_raw_deg'],voltage_kV=ctf['voltage_kv'],
            cs_mm=ctf['cs_mm'],amplitude_contrast=ctf['amp_contrast'],phase_shift_rad=ctf['phase_rad'],
            handedness=geometry.get('handedness',opts['df_hand']),
            alpha_offset_deg=geometry.get('alpha_offset_deg',0.) if spatial else 0.,
            beta_offset_deg=geometry.get('beta_offset_deg',0.) if spatial else 0.,
            gradient_A_per_pixel=geometry.get('gradient_A_per_pixel') if spatial else None,
            coordinate_origin=geometry.get('coordinate_origin'),
            gpu=True,filter_mode='wiener_sz',wiener_lowpass=lowpass)
        for name,image in images.items():
            corrected=correct_tilt_ctf_spatial(image,**correction)
            plane=cp.asnumpy(fourier_bin_2d_gpu(cp.asarray(corrected),binning))
            if not np.isfinite(plane).all():raise ValueError('Nonfinite final CTF plane')
            planes[name].append(plane)
        ctfs.append(dict(defocus=(ctf['defocus_min_A']+ctf['defocus_max_A'])*.5e-10,
            voltage=ctf['voltage_kv']*1000.,spherical_aberration=ctf['cs_mm']*.001,
            amplitude_contrast=ctf['amp_contrast'],phase_shift_deg=np.rad2deg(ctf['phase_rad']),flip_phase=True))
        del images,image,corrected
        print(f'Final native Wiener CTF {i+1}/{len(rows)}: {physical_tilt:.2f} deg',flush=True)
    stacks={name:np.stack(values) for name,values in planes.items()};del planes
    for name,stack in stacks.items():
        raw_path=storage_path(output, 'final_wiener_rawsource.mrcs' if name=='full' else f'final_wiener_{name}_rawsource.mrcs')
        with mrcfile.new(raw_path,overwrite=False) as h:
            h.set_data(stack);h.voxel_size=ctf['pixel_size_A']*binning
    if split_halves:return stacks,dict(pixel_size_A=ctf['pixel_size_A']*binning,split='decoded_movie_frame_parity_full_shifts_full_CTF_full_scale')
    stack=stacks['full'];del stacks
    if return_raw:
        return stack,dict(ctf_data=ctfs,defocus_handedness=0)
    orientation=data['orientation']
    if orientation.get('flip_y',False):stack=np.ascontiguousarray(stack[:,::-1,:])
    frozen_rows=read_table(data['inputs']['frozen_shifts'])
    np.testing.assert_allclose([float(r['angle_deg']) for r in frozen_rows],angles,atol=1e-4)
    frozen=np.array([[float(r['dy_raw']),float(r['dx_raw'])] for r in frozen_rows])/binning
    if orientation.get('flip_y',False):frozen[:,0]*=-1
    with np.load(storage_path(alignment, 'final_state.npz')) as state:
        active=state['active_indices'];cumulative=state['cumulative_shifts_yx'][active]
    aligned=materialize_raw_stack_gpu(cp.asarray(stack),active,cumulative,frozen[active],orientation['rotation_deg']).get()
    with mrcfile.new(storage_path(output, 'final_wiener_aligned.mrcs'),overwrite=False) as h:
        h.set_data(aligned);h.voxel_size=ctf['pixel_size_A']*binning
    metadata=dict(ctf_data=[ctfs[int(i)] for i in active],defocus_handedness=0)
    return aligned,metadata


def canonical_export(volume):
    """Dark particles, canonical Z. Internal landmark coordinates stay intact."""
    return np.ascontiguousarray(volume[::-1],dtype=np.float32)


def export_transform(depth,z_start=0):
    return [[-1,0,0,int(depth)-1-int(z_start)],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
