"""Read cached bin1 tile spectra, test both signs, never mutate preprocessing."""
from greco.project_layout import join as storage_path, ensure_output
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
from ..ctf_estimator import TiltCTF
from ..ctf_local_refinement import test_defocus_hand,resolve_raster_axis,DEFOCUS_CONVENTION


def run(config):
    output=storage_path(config.parent, 'output');ensure_output(output)
    try:
        data=json.loads(config.read_text());folders=[Path(p) for p in data['folders']]
        records=[json.loads((storage_path(p, 'ctf.json')).read_text()) for p in folders]
        first=records[0]
        # Test the model actually fitted to these movie pixels. Display
        # rotation cannot redefine its detector axis.
        geometry=storage_path(folders[0], 'ctf_geometry.json')
        saved=json.loads(geometry.read_text()) if geometry.exists() else {}
        axis=float(saved.get('tilt_axis_deg',data['axis_deg']))
        initial=[TiltCTF(float(r['angle_deg']),float(r.get('initial_defocus_min_A',r['defocus_min_A'])),
            float(r.get('initial_defocus_max_A',r['defocus_max_A'])),
            float(r.get('initial_astig_azimuth_raw_deg',r['astig_azimuth_raw_deg'])),
            float(np.rad2deg(r['phase_rad'])),float(r['defocus_score']),float(r['ctf_resolution_A'])) for r in records]
        mapping=saved.get('diagnostics',{}).get('raster_mapping')
        if mapping is None:
            # Old fits predate movie-raster calibration. Diagnose the same
            # mapping that the next Redo CTF will use, without rewriting it.
            axis,mapping=resolve_raster_axis(folders,initial,axis_deg=axis,
                pixel_A=first['pixel_size_A'],voltage_kV=first['voltage_kv'],
                cs_mm=first['cs_mm'],amp_contrast=first['amp_contrast'],
                tile_size=np.load(storage_path(folders[0], 'ctf_tiles.npy'),mmap_mode='r').shape[1],gpu=True,
                res_low_A=first.get('fit_res_low_A'),res_high_A=first.get('fit_res_high_A'))
        result=test_defocus_hand(folders,initial,pixel_A=first['pixel_size_A'],voltage_kV=first['voltage_kv'],
            cs_mm=first['cs_mm'],amp_contrast=first['amp_contrast'],axis_deg=axis,
            tile_size=np.load(storage_path(folders[0], 'ctf_tiles.npy'),mmap_mode='r').shape[1],gpu=True,
            res_low_A=first.get('fit_res_low_A'),res_high_A=first.get('fit_res_high_A'))
        if result.get('defocus_convention')!=DEFOCUS_CONVENTION:
            raise ValueError('Handedness fitter returned an unsupported physical-defocus convention.')
        result.update(state='completed',orientation=data['orientation'],raster_mapping=mapping)
        temp=storage_path(output, 'web_result.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(storage_path(output, 'web_result.json'))
    except Exception as error:
        (storage_path(output, 'web_result.json')).write_text(json.dumps(dict(state='failed',error=str(error))))
        traceback.print_exc();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
