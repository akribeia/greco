"""Job-local spatial CTF re-estimation after angular alignment."""
import hashlib,json
from pathlib import Path
import numpy as np
from ..project_layout import join as storage_path, preprocessing_work
from ..preprocessing_star import read_table


def reestimate(data, alignment):
    """Fit saved native tile spectra; keep preprocessing products immutable."""
    if alignment is None:
        raise ValueError('Final CTF re-estimation requires fitted alignment geometry.')
    alignment=Path(alignment)
    target=storage_path(alignment,'ctf_refinement.json')
    checkpoint=storage_path(alignment,'final_state.npz')
    signature=dict(geometry_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        orientation=data['orientation'],defocus_handedness=data.get('advanced_options',{}).get('defocus_handedness'))
    if target.exists():
        try:cached=json.loads(target.read_text())
        except (ValueError,OSError):cached={}
        if cached.get('source_signature')==signature:return cached
    from ..alignment_geometry import resolve
    from ..ctf_estimator import TiltCTF
    from ..ctf_local_refinement import refine_local_series
    from .flat_ctf import fit_flat
    prep=preprocessing_work(Path(data['inputs']['stack']))
    settings=json.loads(storage_path(prep,'config.json').read_text())
    options=settings['options']
    rows=read_table(storage_path(prep,'ctf/ctf.star'))
    expected=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
    np.testing.assert_allclose([float(r['tilt_deg']) for r in rows],expected,atol=1e-4)
    with np.load(checkpoint,allow_pickle=False) as state:
        angles,axes=resolve(state,expected,data['orientation']['rotation_deg'])
        active=np.asarray(state['active_indices'],int)
    # Invert the same Flip-Y-first raster orientation used for alignment.
    if data['orientation'].get('flip_y',False):axes=-axes
    folders=[];initial=[];templates=[]
    for index in active:
        row=rows[int(index)]
        folder=storage_path(storage_path(prep,'per_tilt'),f"tilt_{int(row['z_value']):04d}")
        for name in ('ctf_tiles.npy','ctf_tiles_metadata.npz'):
            if not storage_path(folder,name).is_file():
                raise ValueError('CTF re-estimation requires saved native tile spectra; rerun preprocessing with tile spectra retained.')
        ctf=fit_flat(folder,json.loads(storage_path(folder,'ctf.json').read_text()),options)
        folders.append(folder);templates.append(ctf)
        initial.append(TiltCTF(float(angles[index]),ctf['defocus_min_A'],ctf['defocus_max_A'],
            ctf['astig_azimuth_raw_deg'],float(np.rad2deg(ctf['phase_rad'])),
            ctf.get('defocus_score',0.),ctf.get('ctf_resolution_A',0.)))
    if not initial:raise ValueError('Final CTF re-estimation requires accepted views.')
    first=templates[0]
    fitted,model=refine_local_series(folders,initial,pixel_A=first['pixel_size_A'],
        voltage_kV=first['voltage_kv'],cs_mm=first['cs_mm'],amp_contrast=first['amp_contrast'],
        axis_deg=float(axes[active[0]]),axes_deg=axes[active],
        tile_size=options.get('ctf_tile_size',512),gpu=True,
        res_low_A=options.get('ctf_res_low_A'),res_high_A=options.get('ctf_res_high_A'),
        forced_hand=data.get('advanced_options',{}).get('defocus_handedness'),write_outputs=False)
    views={}
    for index,ctf,result in zip(active,templates,fitted):
        index=int(index)
        ctf=dict(ctf,angle_deg=float(angles[index]),defocus_A=.5*(float(result.df_min_A)+float(result.df_max_A)),defocus_min_A=float(result.df_min_A),
            defocus_max_A=float(result.df_max_A),astig_azimuth_raw_deg=float(result.astig_azimuth_deg),
            phase_rad=float(result.phase_rad),defocus_score=float(result.score),
            ctf_resolution_A=float(result.resolution_A))
        geometry=dict(model,tilt_axis_deg=float(axes[index]),tilt_deg=float(angles[index]))
        views[str(int(rows[index]['z_value']))]=dict(ctf=ctf,geometry=geometry)
    record=dict(protocol='final-geometry-spatial-ctf-v1',source='native_tile_spectra',source_signature=signature,
        active_indices=active.tolist(),views=views)
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(target)
    return record
