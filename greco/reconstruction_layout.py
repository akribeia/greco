"""Definitive reconstruction artifacts, scientific exports and publication."""
from pathlib import Path
import json,re



def artifact_relative(work,parts):
    """Publish worker and pinned-engine artifacts directly in their final folders."""
    from .project_layout import _binning
    p=list(parts)
    if not p:return p
    names={'rigid_geometry.npz':['alignment','state.npz'],
           'landmark_basis.npz':['alignment','landmark_basis.npz'],
           'landmark_seed_geometry.npz':['alignment','landmark_seed_geometry.npz'],
           'landmark_regeneration.json':['registration','landmark_regeneration.json'],
           'geometry_report.json':['logs','geometry.json'],
           'transition_diagnostic.json':['registration','transition.json'],
           'tinycare_export_config.json':['tinycare','config.json']}
    if p[0] in names:p=[*names[p[0]],*p[1:]]
    elif re.fullmatch(r'(registration_)?aligned_bin\d+\.mrcs',p[0]):p=['alignment','greco_'+p[0],*p[1:]]
    elif p[0].startswith(('centered_detector_bin','final_wiener','reflected_rawsource')) or p[0]=='frozen_shifts.csv':p=['alignment',*p]
    elif re.fullmatch(r'tilt_\d+_landmarks\.npz',p[0]):p=['registration','landmarks',*p]
    elif p[0] in ('progress.json','progress.tmp','command.json'):p=['logs',*p]
    elif p[0] in ('tomogram.mrc','tomogram_bin8.mrc','tomogram_tinycare.mrc'):
        series=Path(work).parts[Path(work).parts.index('series')+1]
        number=Path(work).parts[Path(work).parts.index('refinement')+1] if 'refinement' in Path(work).parts else '001'
        method='_tinycare' if 'tinycare' in p[0] else ''
        p=['volumes',f'{series}_r{number}{method}_bin{_binning(Path(work))}.mrc',*p[1:]]
    if p[0]=='alignment' and len(p)>1:
        name=p[1]
        if name in ('final_state.npz','rigid_geometry.npz'):p[1]='state.npz'
        elif name=='code':p=['provenance','code',*p[2:]]
        elif name=='config.json':p=['provenance','alignment_config.json',*p[2:]]
        elif name=='selected_landmarks_final.npz':p=['provenance','selected_landmarks.npz',*p[2:]]
        elif name=='summary.json':p=['logs','alignment.json',*p[2:]]
        elif name in ('trajectory.json','shifts.csv') or name.startswith(('stage_','extra_round_')):p=['registration',*p[1:]]
        elif re.fullmatch(r'(registration_)?aligned_bin\d+\.mrcs',name):p[1]='greco_'+name
    if p[0]=='tinycare' and len(p)>1:
        if p[1]=='tinycare.json':p[1]='result.json'
        elif p[1]=='tinycare_model':p[1]='model'
        elif p[1]=='tinycare_config.json':p[1]='training_config.json'
        elif p[1]=='progress.json':p=['logs','tinycare_progress.json',*p[2:]]
    return p


def enabled(path):
    from .project_layout import project_root
    return project_root(path) is not None


def organize_engine_output(work):
    """The pinned engine writes its historic names; publish them before export."""
    from .project_layout import job_work
    work=job_work(work)
    if not enabled(work):return
    for path in sorted((work/'alignment').iterdir()):
        from .project_layout import join
        target=join(work,*path.relative_to(work).parts)
        if target==path:continue
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Reconstruction destination already exists: '+str(target))
        path.rename(target)


def checkpoint_active(state):
    """Old rigid checkpoints imply the full range; Progressive alignment always stores selection."""
    import numpy as np
    if 'active_indices' in state:return np.asarray(state['active_indices'],int)
    if 'cameras_xyz' not in state:raise ValueError('Alignment checkpoint lacks accepted tilt selection.')
    count=len(state['angles_deg'])
    if np.shape(state['cameras_xyz'])!=(count,3,3) or np.shape(state['valid_views'])[1]!=count:
        raise ValueError('Rigid checkpoint does not contain a complete tilt grid.')
    return np.arange(count,dtype=int)


def write_tables(work):
    """Readable STAR exports; NPZ remains the authoritative continuation state."""
    import numpy as np
    from .particle_star import write_loop
    work=Path(work);path=work/'alignment/state.npz'
    if not path.is_file():return
    with np.load(path,allow_pickle=False) as state:d={key:state[key] for key in state.files}
    angles=d['angles_deg'];active=set(checkpoint_active(d).tolist());binning=int(d['binning'])
    rigid='cameras_xyz' in d
    columns=['grecoTiltIndex','grecoTiltAngleDeg','grecoTiltActive','grecoBinning']
    if rigid:
        columns+=['grecoTranslationXAngst','grecoTranslationYAngst']+['grecoCamera'+a+b for a in 'XYZ' for b in 'XYZ']
        records=[(i,float(a),i in active,binning,*map(float,d['translations_xy_A'][i]),*map(float,d['cameras_xyz'][i].ravel())) for i,a in enumerate(angles)]
    else:
        columns+=['grecoShiftXPixel','grecoShiftYPixel']
        records=[(i,float(a),i in active,binning,float(d['cumulative_shifts_yx'][i,1]),float(d['cumulative_shifts_yx'][i,0])) for i,a in enumerate(angles)]
    with (path.parent/'tilts.star').open('w') as stream:
        write_loop(stream,'general',['grecoFormatName','grecoCoordinateConvention'],[('greco-reconstruction-v1','centered_world_xyz_and_raw_detector_xy_angstrom' if rigid else 'MAIN_aligned_raster_shifts_pixels_at_binning')])
        write_loop(stream,'tilts',columns,records)
    positions=d['positions_xyz_A'] if rigid else d['positions_zyx'][:,::-1]
    ids=d['landmark_ids'] if rigid else np.arange(len(positions))
    valid=np.asarray(d['valid_views'],bool)
    if valid.shape[1]!=len(angles):
        expanded=np.zeros((len(positions),len(angles)),bool);expanded[:,checkpoint_active(d)]=valid;valid=expanded
    labels=['grecoLandmarkId']+(['grecoCoordinateXAngst','grecoCoordinateYAngst','grecoCoordinateZAngst'] if rigid else ['grecoCoordinateXPixel','grecoCoordinateYPixel','grecoCoordinateZPixel'])+['grecoBinning']
    with (path.parent/'landmarks.star').open('w') as stream:
        write_loop(stream,'general',['grecoFormatName','grecoCoordinateConvention'],[('greco-reconstruction-v1','centered_world_xyz_angstrom' if rigid else 'uncentered_source_raster_xyz_pixels_at_binning')])
        write_loop(stream,'landmarks',labels,[(int(k),*map(float,p),binning) for k,p in zip(ids,positions)])
        write_loop(stream,'support',['grecoLandmarkId','grecoTiltIndex'],((int(ids[k]),int(t)) for k,t in zip(*np.nonzero(valid)) if int(t) in active))
    config=work/'config.json'
    if config.is_file():
        data=json.loads(config.read_text());inputs=work/'inputs';inputs.mkdir(exist_ok=True)
        (inputs/'manifest.json').write_text(json.dumps({k:data[k] for k in ('inputs','source_work','coarse_run','resume_geometry','landmark_basis','snapshot') if k in data},indent=2))
        if data.get('manual_action')=='refine':
            from .particle_star import write_loop
            edits=work/'edits.star'
            with edits.open('w') as stream:
                write_loop(stream,'fixes',['grecoTiltIndex','grecoReferenceTiltNumber','grecoShiftXPixel','grecoShiftYPixel','grecoBinning'],
                    [(e['index'],None if e.get('reference') is None else e['reference']+1,float(e['shift_yx'][1]),float(e['shift_yx'][0]),8) for e in data.get('manual_edits',[])])
                write_loop(stream,'disabled',['grecoTiltIndex'],[(i,) for i in data.get('options',{}).get('disabled_tilts',[])])


def publish(work,result=None):
    """Publish readable tables and visible links for immutable manual attempts."""
    work=Path(work)
    if not enabled(work):return
    if result is not None:publish_raw_display(work,result)
    write_tables(work)
    if work.parent.name!='actions' or work.parent.parent.name!='provenance':return
    base=work.parent.parent.parent
    config=json.loads((work/'config.json').read_text())
    if config.get('manual_action')=='refine':
        links={base/'curation/fixes':work,base/'curation/edits.star':work/'edits.star'}
    else:
        names=('config.json','result.json','job.sbatch','inputs','alignment','volumes','tinycare','logs','registration')
        links={base/name:work/name for name in names if (work/name).exists()}
    for link,target in links.items():
        link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():link.unlink()
        if link.exists():raise ValueError('Manual publication would replace existing data: '+str(link))
        import os
        link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=target.is_dir())


def identity_path(path):
    """Keep existing manual drafts valid across a rename, never across data edits."""
    from .project_layout import project_root
    root=project_root(path)
    aliases=root/'.reconstruction-path-aliases.json' if root else None
    if aliases and aliases.is_file():
        reverse=json.loads(aliases.read_text())
        for current,original in sorted(reverse.items(),key=lambda item:len(item[0]),reverse=True):
            if str(path)==current or str(path).startswith(current+'/'):return original+str(path)[len(current):]
    return str(path)


def publish_raw_display(work,result):
    """Export unweighted MC sums; retain registration observations for diagnostics."""
    import os
    import mrcfile,numpy as np
    from .project_layout import join,project_root
    from .web.curation_display import _render,_stamp
    work=Path(work);target=join(work,'output/aligned_bin8.mrcs')
    # The initial engine uses alignment/aligned_bin8 too.
    registration=join(work,'registration_aligned_bin8.mrcs')
    if result.get('aligned_stack_source')=='motion_corrected_unweighted':return
    if not target.is_file():return
    config_path=work/'config.json';config=json.loads(config_path.read_text());inputs=config['inputs']
    binning=int(inputs.get('binning',8))
    raw=join(Path(inputs['stack']).parent,f'greco_raw_bin{binning}.mrcs')
    checkpoint=work/'alignment/state.npz'
    if not raw.is_file() or not checkpoint.is_file():raise ValueError('Raw MC stack/checkpoint missing for aligned display export.')
    with np.load(checkpoint,allow_pickle=False) as saved:
        active=checkpoint_active(saved);rigid='cameras_xyz' in saved
    with mrcfile.mmap(raw) as m:
        n=m.data.shape[-1];pixel=float(m.voxel_size.x)
    temporary=target.with_name('.aligned-display.tmp.mrcs')
    try:
        with mrcfile.new_mmap(temporary,shape=(len(active),n,n),mrc_mode=2,overwrite=False) as m:
            m.set_image_stack();m.voxel_size=pixel
            keys=(_stamp(raw),_stamp(checkpoint),_stamp(config_path),_stamp(Path(inputs['frozen_shifts'])),_stamp(Path(inputs['rawtlt'])),_stamp(target))
            for index in range(len(active)):
                m.data[index]=_render(*keys,index,rigid)
            m.update_header_stats()
        if registration.exists():raise ValueError('Registration stack already exists: '+str(registration))
        registration.parent.mkdir(parents=True,exist_ok=True)
        target.rename(registration)
        try:os.replace(temporary,target)
        except Exception:registration.rename(target);raise
    finally:temporary.unlink(missing_ok=True)
    # Draft identities follow the original unchanged registration observations.
    alias_file=project_root(work)/'.reconstruction-path-aliases.json'
    aliases=json.loads(alias_file.read_text()) if alias_file.exists() else {}
    aliases[str(registration)]=identity_path(target)
    alias_file.write_text(json.dumps(aliases,indent=2))
    result.update(aligned_stack=str(target),registration_stack=str(registration),aligned_stack_source='motion_corrected_unweighted')
    report=work/'result.json'
    if report.is_file():report.write_text(json.dumps(result,indent=2))








