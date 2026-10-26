"""Frozen-half reconstruction and TinyCARE in the shared GPU environment."""
from greco.project_layout import join as storage_path, job_work
import json,subprocess,time
from pathlib import Path


def environment(python=None):
    from ..runtime import worker_environment
    return worker_environment('tinycare')


def _run(data,output):
    from .._tinycare.device import check_device
    check_device()
    import cupy as cp,mrcfile
    from ..final_reconstruction import final_stack
    from .segmentation_worker import reconstruct
    from .._tinycare.config import write_json
    output=Path(output);start=time.monotonic()
    def progress(phase):
        write_json(storage_path(output, 'progress.json'),dict(phase=phase))
    progress('TinyCARE: replaying saved MC/CTF into even/odd halves')
    original=json.loads((storage_path(job_work(Path(data['coarse_run'])), 'config.json')).read_text())
    source=storage_path(output, 'half_sources');source.mkdir()
    halves,metadata=final_stack(dict(original,inputs=dict(original['inputs'],binning=data['binning'])),
                               Path(data['coarse_run']),source,return_raw=True,split_halves=True)
    results={}
    for name in ('even','odd'):
        progress('TinyCARE: reconstructing '+name+' with the selected geometry')
        results[name]=reconstruct(data,output,raw_override=(halves.pop(name),metadata['pixel_size_A']),suffix='_'+name)
    del halves
    cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    reconstruction_seconds=time.monotonic()-start
    config=dict(even=results['even']['volumes']['original'],odd=results['odd']['volumes']['original'],output=str(output),model_from=data.get('tinycare_model_from'))
    path=storage_path(output, 'tinycare_config.json');write_json(path,config)
    subprocess.run([data['tinycare_python'],'-m','greco.web.tinycare_worker','--config',str(path)],env=environment(),check=True)
    diagnostic=json.loads((storage_path(output, 'tinycare.json')).read_text())
    with mrcfile.mmap(diagnostic['output']) as m:shape=list(m.data.shape);pixel=float(m.voxel_size.x)
    with mrcfile.mmap(data['parent_volume']) as m:
        if list(m.data.shape)!=shape or abs(float(m.voxel_size.x)-pixel)>1e-4:raise ValueError('TinyCARE output differs from the preceding tomogram grid.')
    diagnostic.update(intermediate_files_retained=False,reconstruction_seconds=reconstruction_seconds)
    write_json(storage_path(output,'tinycare.json'),diagnostic)
    return dict(volumes=dict(original=diagnostic['output']),shape=shape,voxel_size_A=pixel,
        geometry_file=data.get('geometry_file'),binning=data['binning'],method='tinycare',tinycare_model=diagnostic['model_dir'],
        tinycare=diagnostic,reconstruction_seconds=reconstruction_seconds,half_volumes_retained=False,
        output_contrast_convention='dark_particles',registration_unchanged=True)


def run(data,output):
    from .tinycare_cleanup import cleanup_intermediates
    try:
        return _run(data,output)
    finally:
        cleanup_intermediates(output)
