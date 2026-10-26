"""Greco half-map postprocessing and publication of STA results."""
from greco.project_layout import join as storage_path, ensure_output

import math
import os
import time
import tempfile
from contextlib import nullcontext
from pathlib import Path


def run(config):
    started=time.perf_counter()
    import mrcfile
    import numpy as np
    from .reference_mask import load_mask
    from ..web.particle_worker import check_snapshot,save_json
    output=Path(config['output']);ensure_output(output)
    check_snapshot(config['snapshot'])
    save_json(storage_path(output, 'progress.json'),dict(phase='Greco half-map postprocessing'))
    from ..runtime import ROOT
    reference=storage_path(ROOT, 'greco')
    if not (storage_path(reference, '_numerics/apps/postprocess.py')).is_file():raise ValueError('Greco Python postprocess is unavailable.')
    from greco._numerics.apps import postprocess as port
    Postprocessor,_build_parser=port.Postprocessor,port._build_parser
    options=config['options'];n=config['output_box'];pixel=config['pixel_size_A']
    args=['--i',config['half_maps'][0],'--i2',config['half_maps'][1],'--o',str(output),
          '--angpix','-1','--random_seed','0','--low_pass',str(options['low_pass_A'])]
    if options['mask_path']:
        mask=load_mask(options['mask_path'],n,pixel,resample_physical=True)
        path=storage_path(output, 'mask_physical.mrc')
        if path.resolve()!=Path(options['mask_path']).resolve():
            with mrcfile.new(path) as m:m.set_data(mask.astype(np.float32));m.voxel_size=pixel
        args+=['--mask',str(path)]
    if options['bfactor_mode']=='auto':args+=['--auto_bfac','--autob_lowres',str(options['autob_lowres_A']),'--autob_highres',str(options['autob_highres_A'])]
    elif options['bfactor_mode']=='manual':args+=['--adhoc_bfac',str(options['bfactor_A2'])]
    if options['interpolate']:args+=['--interpolate']
    # Isolate the port's phase cache per job, including concurrent Slurm jobs.
    cache=tempfile.TemporaryDirectory(prefix='greco-postprocess-')
    old_cache=os.environ.get('GRECO_CACHE');os.environ['GRECO_CACHE']=cache.name
    try:
        pp=Postprocessor(_build_parser().parse_args(args))
        context=nullcontext()
        if options.get('backend','cpu')=='gpu':
            from .postprocess_gpu import accelerated
            context=accelerated(port)
        with context:
            if pp.run()!=0:raise RuntimeError('Greco Python postprocess failed.')
    finally:
        if old_cache is None:os.environ.pop('GRECO_CACHE',None)
        else:os.environ['GRECO_CACHE']=old_cache
        cache.cleanup()
    check_snapshot(config['snapshot'])
    def number(v):return float(v) if math.isfinite(float(v)) else None
    curves={}
    for name,attr in [('corrected','fsc_true'),('unmasked','fsc_unmasked'),('masked','fsc_masked'),('randomized_mask','fsc_random_masked')]:
        values=getattr(pp,attr,None)
        if values is not None:curves[name]=[dict(shell=i,resolution_A=n*pixel/i if i else None,fsc=number(v)) for i,v in enumerate(values)]
    report=dict(resolution_A=number(pp.global_resol),interpolated_resolution_A=number(pp.fract_resol) if options['interpolate'] else None,
        bfactor_A2=number(pp.global_bfactor),mask_path=options['mask_path'],mask_corrected=bool(getattr(pp,'do_mask',bool(options['mask_path']))),
        evaluator='Greco PostProcess',backend=options.get('backend','cpu'),elapsed_seconds=time.perf_counter()-started,curves=curves)
    from .postprocess_report import update_summary
    update_summary(output/'analysis.star',report)
    save_json(storage_path(output, 'protocol.json'),dict(protocol='greco-postprocess-v1',arguments=args,reference_root=str(reference),half_maps=config['half_maps'],options=options))
    result=dict(state='completed',box_size=n,pixel_size_A=pixel,postprocess=report,options=options,
        particle_count=config.get('particle_count'),used_particles=config.get('used_particles'),
        volumes=dict(merged=str(storage_path(output, 'volumes/map_masked.mrc')) if pp.do_mask else str(storage_path(output, 'volumes/map.mrc')),
                     postprocessed=str(storage_path(output, 'volumes/map.mrc')),masked=str(storage_path(output, 'volumes/map_masked.mrc')) if pp.do_mask else None))
    if not pp.do_mask:result['volumes'].pop('masked',None)
    result['artifacts']=dict(analysis=str(output/'analysis.star'),report=str(output/'report.pdf'))
    save_json(storage_path(output, 'web_result.json'),result)
    scratch=output/'optimisation'
    if scratch.is_dir() and not any(scratch.iterdir()):scratch.rmdir()
    return result
