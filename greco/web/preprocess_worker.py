"""Slurm worker: native MC/CTF/2D preprocessing and aligned raw bin8 viewer product."""
from greco.project_layout import join as storage_path, ensure_output
from greco.preprocessing_star import read_table
import argparse

import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


def prepare_mdoc(series, work):
    import tifffile
    from ..eer import eer_info
    source=Path(series['mdoc_path']).read_text()
    # Keep per-tilt acquisition metadata, but override shared user settings and
    # resolve movies explicitly. Originals are never edited.
    metadata=series['metadata']
    source=re.sub(r'^\s*(?:Voltage|PixelSpacing)\s*=.*$', '', source, flags=re.M)
    source=f"Voltage = {metadata['voltage_kv']}\nPixelSpacing = {metadata['pixel_size_A']}\n"+source
    pieces=re.split(r'(\[\s*ZValue\s*=\s*\d+\s*\])',source)
    records={tilt['z']:tilt for tilt in series['tilts']}
    inputs=storage_path(work, 'movies')
    inputs.mkdir(exist_ok=True)
    for i in range(1,len(pieces),2):
        z=int(re.search(r'\d+',pieces[i])[0])
        tilt=records[z]
        movie=Path(tilt['movie_path'])
        name=f"tilt_{z:04d}"+movie.suffix
        (storage_path(inputs, name)).symlink_to(movie)
        if movie.suffix.lower()=='.eer':
            nframes=eer_info(movie)[0]
        else:
            with tifffile.TiffFile(movie) as handle:
                nframes=handle.series[0].shape[0]
        body=re.sub(r'^\s*(?:SubFramePath|NumSubFrames)\s*=.*$', '',pieces[i+1],flags=re.M)
        pieces[i+1]=f'\nSubFramePath = {name}\nNumSubFrames = {nframes}\n'+body
    mdoc=storage_path(work, 'input.mdoc')
    mdoc.write_text(''.join(pieces))
    return mdoc,inputs


def write_aligned_view(output, binning=8):
    import mrcfile
    import numpy as np
    from scipy.ndimage import shift
    raw=storage_path(output, f'greco_raw_bin{binning}.mrcs')
    shifts=read_table(storage_path(output,f'alignment/tilts.star'))
    destination=storage_path(output, f'greco_2d_aligned_bin{binning}.mrcs')
    temporary=storage_path(output, f'.greco_2d_aligned_bin{binning}.tmp')
    with mrcfile.mmap(raw,mode='r') as source:
        if len(source.data)!=len(shifts):
            raise ValueError('Stack and frozen shifts do not match.')
        with mrcfile.new_mmap(temporary,shape=source.data.shape,mrc_mode=2,overwrite=True) as target:
            for i,row in enumerate(shifts):
                plane=np.asarray(source.data[i],np.float32)
                # align2d returns shifts TO APPLY, in full-resolution pixels.
                target.data[i]=shift(plane,(float(row['dy_raw'])/binning,float(row['dx_raw'])/binning),
                                     order=1,mode='constant',cval=float(plane.mean()),prefilter=False)
            target.voxel_size=source.voxel_size
            target.update_header_stats()
    temporary.replace(destination)
    angles=[float(value) for value in (storage_path(output, f'greco_raw_bin{binning}.rawtlt')).read_text().split()]
    if len(angles)!=len(shifts):
        raise ValueError('Stack and angle count do not match.')
    return dict(state='completed',stack=str(destination),angles=angles,output_binning=binning,
                processing=f'MC + unweighted sum + Fourier bin{binning} + 2D shifts measured on local-mean input; no CTF or axis rotation baked in')


def run(config):
    data=json.loads(config.read_text())
    series,options=data['series'],data['options']
    output=Path(data['output'])
    ensure_output(output)
    try:
        from ..ctf_coordinates import PROTOCOL
        if data.get('ctf_axis_protocol')!=PROTOCOL:
            raise ValueError('The web server uses an old CTF coordinate protocol. Restart Greco and submit again; no motion correction was started.')
        if options.get('output_binning',8) != 8:
            raise ValueError('Preprocessing requires fixed output binning 8. Submit again.')
        mdoc,movies=prepare_mdoc(series,config.parent)
        command=[sys.executable,'-m','greco.preprocess','--mdoc',str(mdoc),
                 '--movies-dir',str(movies),'--output',str(output),
                 '--binning','8','--raw-source','--raw-only',
                 '--dose-per-tilt',str(series['metadata']['dose_per_tilt']),
                 '--axis-deg',str((data.get('orientation') or {}).get('rotation_deg',series['metadata']['tilt_axis_deg'])),
                 '--cs-mm',str(options['cs_mm']), '--amp-contrast',str(options['amp_contrast'])]
        if options.get('output_bin1_sta',True):command.append('--output-bin1-sta')
        for key in ('mc_b_factor','mc_iter','mc_tol','mc_search_radius','ctf_regularization',
                    'ctf_tile_size','gpus','eer_group','eer_threads','eer_upscaling','df_hand'):
            command.extend(['--'+key.replace('_','-'),str(options[key])])
        for key in ('ctf_res_low_A','ctf_res_high_A'):
            if options.get(key) is not None:
                command.extend(['--'+key.replace('_','-'),str(options[key])])
        for key in ('flip_y','transpose','ctf_spatial','ctf_local_refine'):
            if options.get(key,False):
                command.append('--'+key.replace('_','-'))
        from ..ctf_local_refinement import fixed_handedness
        if fixed_handedness(data) is not None:
            command.append('--ctf-fixed-hand')
        if (options['ctf_spatial'] or options.get('ctf_local_refine',False)) and options.get('ctf_spatial_axis_deg') is not None:
            command.extend(['--ctf-spatial-axis-deg',str(options['ctf_spatial_axis_deg'])])
        if series.get('gain_path'):
            command.extend(['--gain',series['gain_path']])
            if series.get('gain_flip_y',False):command.append('--gain-flip-y')
        subprocess.run(command,check=True)
        result=write_aligned_view(output,8)
        result['ctf_axis_protocol']=PROTOCOL
        manifest=json.loads((storage_path(output, 'manifest.json')).read_text())
        resolved=manifest.get('ctf_spatial_axis_deg')
        if resolved is None and not options['ctf_spatial'] and not options.get('ctf_local_refine',False):resolved=0.
        result['options']=dict(options,ctf_spatial_axis_deg=resolved)
        data['ctf_axis_requested']=options.get('ctf_spatial_axis_deg')
        data['options']=result['options']
        temporary=config.with_name('.config-ctf.tmp')
        temporary.write_text(json.dumps(data,indent=2));temporary.replace(config)
        from .ctf_view import write_spectrum_stack
        result['ctf_spectra_stack']=write_spectrum_stack(output,8)
        geometry_files=list((storage_path(output, 'per_tilt')).glob('tilt_*/ctf_geometry.json'))
        if geometry_files:
            result['ctf_local_geometry']=json.loads(geometry_files[0].read_text())
    except Exception as error:
        traceback.print_exc()
        result=dict(state='failed',error=str(error))
    temporary=storage_path(output, '.web_result.tmp')
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(storage_path(output, 'web_result.json'))
    return 0 if result['state']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().config))
