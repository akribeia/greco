"""Current STA artifact layout and metadata publication."""
import json

from pathlib import Path
STAR_FILES={'iterations.json':'iterations/iterations.star','fsc.json':'optimisation/fsc.star',
 'ctf_fits.json':'optimisation/ctf_fits.star','gp3d_fits.json':'optimisation/motion_fits.star',
 'alternating_iterations.json':'iterations/alternating.star','signal_prior.json':'optimisation/signal_prior.star',
 'model.json':'optimisation/model.star'}
NUMERICS={'noise_model.json','memory_policy.json'}


def enabled(root):
    """All recognized projects use the current STA artifact layout."""
    return (Path(root)/'project.json').is_file()


def artifact_relative(work,rest):
    if not rest:return rest
    # Nested alternating passes have the same artifact organisation.
    prefix=[]
    if rest[0]=='alternating' and len(rest)>=3:
        length=3 if rest[1].startswith('cycle_') and len(rest)>=4 else 2
        prefix=rest[:length];rest=rest[length:]
        if not rest:return prefix
    name=rest[0]
    if name.startswith('tomo_'):return prefix+['particles',*rest]
    if name.startswith('iter_') or name.startswith('iteration_'):return prefix+['iterations',*rest]
    if work.parent.name=='postprocessing' and name=='fsc.json':return prefix+['analysis.star',*rest[1:]]
    if name in STAR_FILES:return prefix+STAR_FILES[name].split('/')+rest[1:]
    if name in NUMERICS:return prefix+['provenance',name]+rest[1:]
    if name in ('protocol.json','job.sbatch','resolved_config.json'):return prefix+['provenance',*rest]
    if name in ('progress.json','latest_iteration.json'):return prefix+['optimisation',*rest]
    if name=='mask_physical.mrc':return prefix+['inputs','mask.mrc',*rest[1:]]
    if name=='merged.mrc':return prefix+['volumes','map.mrc',*rest[1:]]
    if name.endswith('.mrc') and (name.startswith('half') or name.startswith('class_')):return prefix+['volumes',*rest]
    if name=='phase_cache' or name.endswith(('.npy','.npz')):
        return prefix+['optimisation',*rest]
    return prefix+rest


def read_metadata(path):
    path=Path(path)
    if path.name=='analysis.star':
        from .sta.postprocess_report import read_analysis
        return read_analysis(path)
    if path.suffix=='.star':
        from .sta_metadata import read
        return read(path)
    # Numerical policy/diagnostics share one provenance file.
    if path.name in NUMERICS and path.parent.name=='provenance':
        return json.loads((path.parent/'numerics.json').read_text())[path.stem]
    return json.loads(path.read_text())


def write_metadata(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.star':
        from .sta_metadata import write
        return write(path,value)
    from .project_layout import project_root,join
    root=project_root(path)
    active=root is not None and enabled(root) and path.is_relative_to(root/'sta')
    if active and path.name=='result.json' and isinstance(value,dict) and value.get('state')=='completed':
        value=dict(value)
        work=path.parent;latest=join(work,'latest_iteration.json')
        resolved=join(work,'resolved_config.json');config_path=work/'config.json'
        if resolved.is_file() and config_path.is_file():
            submitted=json.loads(config_path.read_text())
            if submitted.get('action')=='sta_after_dependencies':
                actual=json.loads(resolved.read_text())
                actual.update(job_kind=submitted.get('job_kind'),mask_source_path=submitted.get('mask_source_path'),
                    submission=dict(action='sta_after_dependencies',options=submitted.get('deferred_options',{})))
                write_metadata(config_path,actual);resolved.unlink()
        if latest.is_file():
            preview=json.loads(latest.read_text())
            value.update(iteration=preview['iteration'],metrics=preview.get('metrics',{}))
        elif value.get('gpu_local_alignment') and isinstance(value.get('iterations'),int):
            history=join(work,'iterations.json')
            metrics=read_metadata(history)[-1] if history.is_file() else {}
            value.update(iteration=value['iterations'],metrics=metrics)
        if isinstance(value.get('postprocess'),dict):
            report=dict(value['postprocess']);report.pop('curves',None);report['fsc_file']=str(join(work,'fsc.json'));value['postprocess']=report
    if active and path.name=='progress.json' and isinstance(value,dict) and value.get('phase')=='Completed':
        result=join(path.parent.parent,'result.json')
        if result.is_file() and json.loads(result.read_text()).get('state')=='completed':
            path.unlink(missing_ok=True);return
    if path.name in NUMERICS and path.parent.name=='provenance':
        target=path.parent/'numerics.json'
        try:merged=json.loads(target.read_text())
        except FileNotFoundError:merged={}
        merged[path.stem]=value;path=target;value=merged
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)
    if active and path.name=='result.json' and isinstance(value,dict) and value.get('state')=='completed':
        for name in ('progress.json','latest_iteration.json'):join(path.parent,name).unlink(missing_ok=True)


def freeze_mask(config):
    """Keep the exact physically adapted mask with its immutable job."""
    from .project_layout import project_root
    if not config.get('output'):return config
    output=Path(config['output']);root=project_root(output)
    if root is None or not enabled(root):return config
    options=config.get('options',{});source=options.get('mask_path')
    if not source:return config
    target=output/'inputs/mask.mrc';target.parent.mkdir(parents=True,exist_ok=True)
    if Path(source).resolve()!=target.resolve():
        import numpy as np,mrcfile
        from .sta.reference_mask import load_mask
        mask=load_mask(source,config['output_box'],config['pixel_size_A'],resample_physical=True)
        if target.exists():
            with mrcfile.open(target) as m:
                if m.data.shape!=mask.shape or not np.allclose(m.data,mask,atol=1e-6,rtol=1e-6) or not np.isclose(float(m.voxel_size.x),config['pixel_size_A']):raise ValueError('Existing job mask does not match the frozen input.')
        else:
            temporary=target.with_name('mask.tmp.mrc')
            with mrcfile.new(temporary,overwrite=True) as m:m.set_data(mask.astype(np.float32));m.voxel_size=config['pixel_size_A']
            temporary.replace(target)
        config=dict(config,options=dict(options,mask_path=str(target)),mask_source_path=str(source))
        if 'deferred_options' in config:config['deferred_options']=dict(config['deferred_options'],mask_path=str(target))
        snapshot=[s for s in config.get('snapshot',[]) if Path(s['path']).resolve()!=Path(source).resolve()]
        stat=target.stat();config['snapshot']=[*snapshot,dict(path=str(target),size=stat.st_size,mtime_ns=stat.st_mtime_ns)]
    return config
