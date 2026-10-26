"""Refit cached native tile spectra and rebuild CTF products with saved MC shifts."""
from greco.project_layout import join as storage_path, ensure_output
from greco.preprocessing_star import write_table
import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
from types import SimpleNamespace


def clone_cached(source,work):
    """Share immutable arrays by inode; keep all mutable CTF files independent."""
    def copy_file(src,dst):
        path=Path(src)
        immutable=(path.suffix in ('.mrc','.mrcs','.npz') or path.name=='ctf_tiles.npy')
        if immutable:
            try:os.link(src,dst);return dst
            except OSError:pass
        return shutil.copy2(src,dst)
    def ignore(folder,names):
        return [name for name in names if name in ('web_result.json','ctf_spectrum.npy','ctf_geometry.json','ctf_local_samples.npz','ctf.star')
                or name.startswith('greco_ctf') or (name.startswith('reliability_bin') and Path(name).suffix in ('.mrcs','.rawtlt')) or name.startswith('.')]
    for name in ('motion','alignment'):
        shutil.copytree(storage_path(source,name),storage_path(work,name),copy_function=copy_file)
    shutil.copytree(source/'ctf',work/'ctf',copy_function=copy_file,
        ignore=lambda folder,names:ignore(folder,names)+(['handedness'] if 'handedness' in names else []))
    for name in ('manifest.json','ctf/astigmatism.json','alignment2d.json'):
        src=storage_path(source,name);dst=storage_path(work,name)
        if src.is_file():
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    ensure_output(work)
    shutil.copy2(storage_path(source, 'input.mdoc'),storage_path(work, 'input.mdoc'))
    (storage_path(work, 'movies')).mkdir(parents=True,exist_ok=True)
    for movie in (storage_path(source, 'movies')).iterdir():
        (storage_path(storage_path(work, 'movies'), movie.name)).symlink_to(movie.resolve(strict=True))


def write_ctf_products(output,binning,pixel_A,records,products):
    """Publish the same reconstruction inputs as initial preprocessing."""
    import mrcfile
    for name,stack in products.items():
        filename=(f'ctf/reliability_bin{binning}.mrcs' if name=='reliability'
                  else f'greco_ctf_{name}_bin{binning}_rawsource.mrcs')
        path=storage_path(output,filename)
        with mrcfile.new(path,overwrite=False) as handle:
            handle.set_data(stack);handle.voxel_size=(pixel_A*binning,)*3;handle.update_header_stats()
        path.with_suffix('.rawtlt').write_text(''.join(f"{float(r['angle_deg']):.6f}\n" for r in records))


def run(config):
    import mrcfile
    import numpy as np
    from ..ctf_estimator import _CTFEstimator,refine_series_bad_tilts
    from ..ctf_local_refinement import refine_local_series
    from ..preprocess import _stage_b_tilt
    from ..movie_replay import _movie_mdoc,_load_gain
    from .ctf_view import tilt_records,write_spectrum_stack
    started=time.monotonic();work=config.parent;output=storage_path(work, 'output')
    def progress(phase,**extra):
        temporary=storage_path(work, 'progress.tmp')
        temporary.write_text(json.dumps(dict(phase=phase,**extra)))
        temporary.replace(storage_path(work, 'progress.json'))
    try:
        data=json.loads(config.read_text());source=Path(data['source_work']);options=data['options']
        from ..ctf_coordinates import PROTOCOL,detector_axis
        if data.get('ctf_axis_protocol')!=PROTOCOL:
            raise ValueError('Restart Greco before submitting Redo CTF: the coordinate protocol changed.')
        if source.resolve()==work.resolve():raise ValueError('CTF revision must not overwrite its source.')
        progress('Preparing cached images')
        clone_cached(source,work)
        binning=options['output_binning'];records=tilt_records(output,binning)
        folders=[storage_path(storage_path(output, 'per_tilt'), f"tilt_{int(r['mdoc_index']):04d}") for r in records]
        old=[json.loads((storage_path(folder, 'ctf.json')).read_text()) for folder in folders]
        pixel=float(old[0]['pixel_size_A'])
        acquisition=_movie_mdoc(SimpleNamespace(mdoc=storage_path(work, 'input.mdoc'),eer_upscaling=options['eer_upscaling']))
        raw_axis=detector_axis(acquisition,options.get('ctf_spatial_axis_deg'),
                               flip_y=options['flip_y'],transpose=options['transpose'])
        params=dict(pixel_A=pixel,voltage_kV=data['metadata']['voltage_kv'],
                    cs_mm=options['cs_mm'],amp_contrast=options['amp_contrast'],
                    tile_size=options['ctf_tile_size'],gpu=True,
                    res_low_A=options['ctf_res_low_A'],res_high_A=options['ctf_res_high_A'])
        est=_CTFEstimator(**params);initial=[];spectra=[]
        for i,(folder,ctf) in enumerate(zip(folders,old)):
            progress('Estimating CTF',tilt=i+1,total=len(folders))
            tiles=np.load(storage_path(folder, 'ctf_tiles.npy'),mmap_mode='r',allow_pickle=False)
            if tiles.shape[1]!=options['ctf_tile_size']:raise ValueError('Cached CTF tile size changed.')
            clean=est.clean(est.dev.asarray(tiles.mean(axis=0,dtype=np.float32)))
            fit=est.do_tilt(clean,float(ctf['angle_deg']),(0.,0.))
            initial.append(fit);spectra.append(est.dev.asnumpy(clean))
        refined=initial
        if len(initial)>=7:
            refined,_=refine_series_bad_tilts(spectra,initial,**params,angles_deg=[r.tilt_deg for r in initial])
        progress('Refining local CTF')
        from ..ctf_local_refinement import fixed_handedness
        forced=fixed_handedness(data)
        refined,geometry=refine_local_series(folders,refined,**params,axis_deg=raw_axis,forced_hand=forced)
        raw_axis=geometry['tilt_axis_deg']
        for folder,previous,before,after in zip(folders,old,initial,refined):
            previous.update(voltage_kv=params['voltage_kV'],cs_mm=params['cs_mm'],amp_contrast=params['amp_contrast'],
                fit_res_low_A=est.res_lo_A,fit_res_high_A=est.res_hi_A,
                initial_defocus_min_A=before.df_min_A,initial_defocus_max_A=before.df_max_A,
                initial_astig_azimuth_raw_deg=before.astig_azimuth_deg,initial_defocus_score=before.score,
                defocus_A=after.df_mean_A,defocus_min_A=after.df_min_A,defocus_max_A=after.df_max_A,
                astig_azimuth_raw_deg=after.astig_azimuth_deg,phase_rad=after.phase_rad,
                defocus_score=after.score,ctf_resolution_A=after.resolution_A,series_refined=True)
            (storage_path(folder, 'ctf.json')).write_text(json.dumps(previous,indent=2,allow_nan=False))
        # Keep the accepted 2D alignment, detector calibration and MC trajectories.
        manifest=json.loads((storage_path(output, 'manifest.json')).read_text())
        args=SimpleNamespace(**options)
        args.output=output;args.mdoc=storage_path(work, 'input.mdoc');args.movies_dir=storage_path(work, 'movies')
        args.binning=binning;args.axis_deg=manifest.get('axis_deg',0.)
        args._gain=_load_gain(data['series'].get('gain_path'),flip_y=data['series'].get('gain_flip_y',False))
        args.bin1_reliability='off';args.ctf_spatial_axis_deg=geometry['tilt_axis_deg'];args.df_hand=geometry['handedness']
        mdoc=_movie_mdoc(args)
        mdoc.voltage_kv=params['voltage_kV']
        if args._gain is not None and options['eer_upscaling']==2 and args._gain.shape==(4096,4096):
            args._gain=np.repeat(np.repeat(args._gain,2,axis=0),2,axis=1)
        with mrcfile.mmap(storage_path(output, f'greco_raw_bin{binning}.mrcs'),mode='r') as handle:shape=handle.data.shape
        corrected=np.empty(shape,np.float32)
        half_names=['even','odd'] if any((storage_path(storage_path(source, 'output'), f'greco_ctf_{name}_bin{binning}_rawsource.mrcs')).exists() for name in ('even','odd')) else []
        halves={name:np.empty(shape,np.float32) for name in half_names} or None
        rows=[];lock=threading.Lock()
        for i,record in enumerate(records):
            progress('Rebuilding CTF-corrected images',tilt=i+1,total=len(records))
            _stage_b_tilt(args,int(record['mdoc_index']),mdoc,{},
                (float(record['dy_raw']),float(record['dx_raw'])),None,lock,rows,
                out_raw8=corrected,raw_only=True,slot=i,out_ctf_halves=halves)
        products={'reliability':corrected,**(halves or {})}
        write_ctf_products(output,binning,pixel,records,products)
        write_table(storage_path(output,'ctf/ctf.star'),rows,'ctf')
        def rebase(value):
            if isinstance(value,str):return value.replace(str(storage_path(source, 'output')),str(output))
            if isinstance(value,dict):return {k:rebase(v) for k,v in value.items()}
            if isinstance(value,list):return [rebase(v) for v in value]
            return value
        manifest=rebase(manifest)
        manifest.update(ctf_local_geometry=geometry,ctf_spatial_axis_deg=raw_axis,
            df_hand=geometry['handedness'],cs_mm=params['cs_mm'],amp_contrast=params['amp_contrast'],
            ctf_spatial=options['ctf_spatial'],ctf_regularization=options['ctf_regularization'],
            ctf_redo_source=str(source),ctf_redo_seconds=time.monotonic()-started,
            motion_correction_repeated=False,alignment2d_repeated=False)
        (storage_path(output, 'manifest.json')).write_text(json.dumps(manifest,indent=2,allow_nan=False))
        result=rebase({k:v for k,v in data['source_result'].items()
                       if k in ('stack','angles','output_binning','processing','tilt_display_version')})
        result.update(state='completed',work=str(work),ctf_local_geometry=geometry,
            ctf_revision=work.name,ctf_spectra_stack=write_spectrum_stack(output,binning),
            ctf_redo_seconds=time.monotonic()-started,motion_correction_repeated=False,
            alignment2d_repeated=False,options=dict(options,ctf_spatial_axis_deg=raw_axis),ctf_axis_protocol=PROTOCOL)
        data['ctf_axis_requested']=options.get('ctf_spatial_axis_deg');data['options']=result['options']
        configuration=config.with_name('.config-ctf.tmp')
        configuration.write_text(json.dumps(data,indent=2));configuration.replace(config)
        # Completion is published only after all CTF-dependent products exist.
        temporary=storage_path(output, 'web_result.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False))
        temporary.replace(storage_path(output, 'web_result.json'))
        progress('Completed')
        return 0
    except Exception as error:
        traceback.print_exc();ensure_output(output)
        temporary=storage_path(output, 'web_result.tmp')
        temporary.write_text(json.dumps(dict(state='failed',error=str(error))))
        temporary.replace(storage_path(output, 'web_result.json'))
        return 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().config))
