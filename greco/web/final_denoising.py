"""Optional TinyCARE on final exports; never changes registration inputs."""
from greco.project_layout import join as storage_path, ensure_output, job_work
import argparse
import json
from pathlib import Path
import subprocess
from .. import runtime
from .._tinycare.config import write_json

def validate_denoising_options(raw):
    from .projects import InputError
    if not isinstance(raw,dict) or set(raw)-{'denoisers','reuse_tinycare_model'}:
        raise InputError('Unknown denoising options.')
    selected=raw.get('denoisers',[]);reuse=raw.get('reuse_tinycare_model',False)
    if not isinstance(selected,list) or selected not in ([],['tinycare']):
        raise InputError('Unknown TinyCARE denoising selection.')
    if not isinstance(reuse,bool):raise InputError('Reuse TinyCARE model must be a checkbox.')
    if reuse and not selected:raise InputError('Select TinyCARE before reusing its model.')
    return dict(denoisers=selected,reuse_tinycare_model=reuse)

def prepare_model_reuse(root, row, options, inputs, final_binning=8):
    """Resolve only completed models owned by this series, before submission."""
    if not options.get('reuse_tinycare_model'):
        return {}
    import mrcfile
    from ..project_layout import owned
    from .._tinycare.config import model_metadata
    from .projects import InputError
    jobs = []
    for step in reversed(row.get('processing_steps', [])):
        jobs.append(step.get('job', {}))
        jobs.extend(reversed(step.get('manual_history', [])))
    jobs.extend([row.get('geometry_refinement', {}), row.get('reconstruction', {})])
    jobs.extend(reversed(row.get('reconstruction_history', [])))
    owners = [Path(j['work']).resolve() for j in jobs if j.get('work') and owned(root, Path(j['work']))]
    with mrcfile.mmap(inputs['stack']) as m:
        pixel = float(m.voxel_size.x) * final_binning / inputs.get('binning', 8)
    for job in jobs:
        if job.get('state') not in ('completed', 'partial') or not job.get('work'):
            continue
        work = Path(job['work'])
        if not owned(root, work):
            continue
        result = job
        report = storage_path(work, 'output/web_result.json')
        if report.is_file():
            result = json.loads(report.read_text())
        model = result.get('tinycare', {}).get('model_dir')
        if not model:
            continue
        model = Path(model).resolve()
        if not any((model.is_relative_to(owner) for owner in owners)):
            continue
        try:
            (_, weights) = model_metadata(model, pixel)
        except ValueError:
            continue
        paths = [model / 'norm.json', model / 'config.json', weights]
        return dict(tinycare_model_from=str(model), tinycare_model_snapshot=[dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths])
    raise InputError('No compatible TinyCARE model exists for this tilt series and output sampling. Run TinyCARE without Reuse first.')

def denoise_outputs(data, result, geometry=False):
    selected = data.get('options', {}).get('denoisers', []) if geometry else data.get('denoisers', [])
    result['denoising_errors'] = {}
    if not selected:
        return
    if selected != ['tinycare']:
        raise ValueError('Only TinyCARE final-volume denoising is supported.')
    output = Path(data['output'])
    coarse = Path(data['coarse_run']) if geometry else storage_path(output, 'alignment')
    source = json.loads(storage_path(job_work(coarse), 'config.json').read_text())
    original = result['tomogram'] if geometry else result['volumes']['original']
    config = dict(coarse_run=str(coarse), source_config=source, source_work=str(job_work(coarse)), binning=data.get('final_binning') or 8 if geometry else data.get('final_binning') or source['inputs']['binning'], parent_volume=original, geometry_file=result.get('geometry_file') if geometry else None, tinycare_python=str(runtime.python('tinycare')), output=str(storage_path(output, 'tinycare')))
    if data.get('options', {}).get('reuse_tinycare_model'):
        if not data.get('tinycare_model_from'):
            raise ValueError('No same-series TinyCARE model was selected.')
        config['tinycare_model_from'] = data['tinycare_model_from']
        from .particle_worker import check_snapshot
        check_snapshot([s for s in data.get('snapshot', []) if Path(s['path']).is_relative_to(Path(data['tinycare_model_from']))])
    path = storage_path(output, 'tinycare_export_config.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path, config)
    try:
        import sys
        if 'cupy' in sys.modules:
            sys.modules['cupy'].get_default_memory_pool().free_all_blocks()
        if 'torch' in sys.modules:
            sys.modules['torch'].cuda.empty_cache()
        subprocess.run([str(runtime.python('tinycare')), '-m', 'greco.web.final_denoising', '--config', str(path)], env=runtime.worker_environment('tinycare'), check=True)
        report = json.loads(storage_path(Path(config['output']), 'tinycare.json').read_text())
        target = storage_path(output, 'tomogram_tinycare.mrc')
        target.parent.mkdir(parents=True,exist_ok=True)
        Path(report['output']).replace(target)
        report['output'] = str(target)
        write_json(storage_path(Path(config['output']), 'tinycare.json'), report)
        result['volumes']['tinycare'] = str(target)
        result['tinycare'] = report
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        detail = Path(config['output']) / 'error.json'
        try:
            message = json.loads(detail.read_text())['error']
        except (OSError, ValueError, KeyError):
            message = str(error)
        result['denoising_errors']['tinycare'] = message
        result['state'] = 'partial'
        result['warning'] = 'Reconstruction available; requested TinyCARE failed.'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    data = json.loads(parser.parse_args().config.read_text())
    output = Path(data['output'])
    ensure_output(output)
    from .segmentation_tinycare import run
    detail = output / 'error.json'
    detail.unlink(missing_ok=True)
    try:
        run(data, output)
    except Exception as error:
        write_json(detail, dict(error=str(error)))
        raise
if __name__ == '__main__':
    main()
