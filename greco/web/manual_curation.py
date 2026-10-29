"""Lightweight CPU diagnostics, background execution and persistent cache."""
from greco.project_layout import curation_directory, join as storage_path, owned as layout_owned
import hashlib,io,json,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .projects import InputError
from ..curation_2d import PROTOCOL,THRESHOLDS

_POOL=ThreadPoolExecutor(max_workers=1,thread_name_prefix='greco-curation')
_TASKS={};_LOCK=threading.Lock()


def sources(service,pid,sid):
    with service.lock:
        root=service._root(pid);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==sid),None)
        if row is None:raise InputError('Unknown tilt series.')
        recon=row.get('reconstruction',{});geometry=row.get('geometry_refinement',{})
    result={}
    if recon.get('work'):
        work=service.path(recon['work'],exists=False)
        if not layout_owned(root,work):raise InputError('Invalid reconstruction path.')
        config=storage_path(work, 'config.json');stack=storage_path(work, 'output/alignment/aligned_bin8.mrcs')
        if config.is_file() and stack.is_file():
            settings=json.loads(config.read_text());angles=service.path(settings['inputs']['rawtlt'])
            result['reconstruction']=dict(stack=str(stack),angles=str(angles),label='Reconstruction')
            selection=storage_path(stack.parent,'final_state.npz')
            if selection.is_file():result['reconstruction']['selection']=str(selection)
        if result and geometry.get('work'):
            geo=service.path(geometry['work'],exists=False)
            if not layout_owned(root,geo):raise InputError('Invalid geometry path.')
            report=storage_path(geo, 'output/web_result.json')
            if report.is_file():
                saved=json.loads(report.read_text())
                accepted_sources={(storage_path(work, 'alignment')).resolve()}
                if saved.get('state')=='completed' and saved.get('export_protocol')=='main-raster-canonical-z-v1' and Path(saved.get('source_reconstruction','')).resolve() in accepted_sources:
                    path=service.path(saved['aligned_stack'])
                    if path!=storage_path(geo, 'output/aligned_bin8.mrcs'):raise InputError('Invalid geometry display stack.')
                    result['geometry']=dict(stack=str(path),angles=str(angles),label='Refined geometry')
    from .reconstruction_steps import ensure,job_for,work_for
    ensure(row)
    for number,step in enumerate(row['processing_steps'],2):
        if step['id']=='geometry':continue
        manual=step['kind']=='manual_curation'
        job=job_for(row,step)
        if not job.get('work'):continue
        geo=service.path(job['work'],exists=False)
        base=work_for(root,sid,step)
        if manual:
            from .manual_refinement import valid_work
            if not valid_work(root,base,geo):raise InputError('Invalid manual refinement directory.')
        elif geo!=base:raise InputError('Invalid pipeline geometry directory.')
        report=storage_path(geo, 'output/web_result.json');config=storage_path(geo, 'config.json')
        if manual and (not report.is_file() or json.loads(report.read_text()).get('state')!='completed'):
            from .manual_refinement import effective_job
            previous=effective_job(row,step)
            if previous:
                geo=service.path(previous['work'])
                if not valid_work(root,base,geo):raise InputError('Invalid prior manual refinement directory.')
                report=storage_path(geo, 'output/web_result.json');config=storage_path(geo, 'config.json')
            else:
                from .reconstruction_steps import previous_geometry
                prior=previous_geometry(row,step)
                key='reconstruction' if prior is None else 'curation_step_'+prior['id'] if prior['kind']=='manual_curation' else 'geometry' if prior['id']=='geometry' else 'geometry_step_'+prior['id']
                if key in result:result['curation_step_'+step['id']]=result[key]
                continue
        if not report.is_file() or not config.is_file():continue
        saved=json.loads(report.read_text());settings=json.loads(config.read_text())
        if saved.get('state')=='completed' and saved.get('export_protocol')=='main-raster-canonical-z-v1' and saved.get('source_reconstruction')==settings.get('coarse_run'):
            path=service.path(saved['aligned_stack'])
            if path!=storage_path(geo, 'output/aligned_bin8.mrcs'):raise InputError('Invalid geometry display stack.')
            key=('curation_step_' if manual else 'geometry_step_')+step['id']
            result[key]=dict(stack=str(path),angles=str(service.path(settings['inputs']['rawtlt'])),label=f'Manual Curation {number:03d}' if manual else f'Refined geometry {number:03d}')
            if saved.get('geometry_file') and Path(saved['geometry_file']).is_file():result[key]['selection']=saved['geometry_file']
            if manual and (storage_path(geo, 'output/transition_diagnostic.json')).is_file():result[key]['diagnostic']=str(storage_path(geo, 'output/transition_diagnostic.json'))
        elif manual:
            # Keep inspecting the immutable previous input while the new job runs.
            from .reconstruction_steps import previous_geometry
            prior=previous_geometry(row,step)
            key='reconstruction' if prior is None else 'curation_step_'+prior['id'] if prior['kind']=='manual_curation' else 'geometry' if prior['id']=='geometry' else 'geometry_step_'+prior['id']
            if key in result:result['curation_step_'+step['id']]=result[key]
    for item in result.values():
        from ..project_layout import job_work
        owner=job_work(Path(item['stack']))
        report=storage_path(owner,'output/web_result.json')
        if report.is_file():
            registration=json.loads(report.read_text()).get('registration_stack')
            if registration:
                path=service.path(registration)
                if path!=storage_path(owner,'registration_aligned_bin8.mrcs'):raise InputError('Invalid registration diagnostic stack.')
                item['registration_stack']=str(path)
    return root,result


def source_angles(item):
    """Angles in aligned-stack plane order, including partial Greco results."""
    angles=np.loadtxt(item['angles'],ndmin=1)
    indices=np.arange(len(angles))
    if item.get('selection'):
        with np.load(item['selection'],allow_pickle=False) as saved:
            indices=np.asarray(saved['active_indices'] if 'active_indices' in saved else np.arange(len(angles)),int)
            if (indices.ndim!=1 or len(set(indices.tolist()))!=len(indices)
                    or np.any(indices<0) or np.any(indices>=len(angles))):
                raise ValueError('Invalid aligned-stack tilt selection.')
            if not np.allclose(saved['angles_deg'],angles,atol=1e-4):
                raise ValueError('Aligned-stack tilt selection belongs to different angles.')
        angles=angles[indices]
    return angles,indices


def identity(item):
    files=[]
    for key in ('stack','angles',*(('selection',) if item.get('selection') else ())):
        p=Path(item.get('registration_stack',item[key]) if key=='stack' else item[key]);s=p.stat()
        # File identity survives renames, but replacement or editing invalidates it.
        files.append(dict(role=key,device=s.st_dev,inode=s.st_ino,
                          size=s.st_size,mtime_ns=s.st_mtime_ns))
    value=dict(protocol=PROTOCOL,thresholds=THRESHOLDS,files=files)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),value


def status(service,pid,sid,source='reconstruction'):
    root,available=sources(service,pid,sid)
    if source not in available:
        return dict(state='pending',sources={k:v['label'] for k,v in available.items()},source=source,rows=[])
    item=available[source];key,fingerprint=identity(item)
    target=storage_path(curation_directory(root,sid), source+'_'+key[:16]+'.json')
    angles,indices=source_angles(item)
    result=dict(state='pending',source=source,input_identity=key,sources={k:v['label'] for k,v in available.items()},
        rows=[dict(index=i,number=int(indices[i])+1,source_index=int(indices[i]),angle_deg=float(a),color='gray',reason='2D diagnostic not calculated yet.') for i,a in enumerate(angles)])
    all_angles=np.loadtxt(item['angles'],ndmin=1)
    discarded=[t for t in range(len(all_angles)) if t not in set(indices.tolist())]
    result['discarded_rows']=[dict(index=len(indices)+i,number=t+1,source_index=t,angle_deg=float(all_angles[t]),
        color='gray',discarded=True,reason='Discarded by alignment; not included in reconstruction.') for i,t in enumerate(discarded)]
    diagnostic=Path(item.get('diagnostic',target))
    if diagnostic.is_file():result.update(json.loads(diagnostic.read_text()),state='completed',report=str(diagnostic),source=source)
    else:
        with _LOCK:future=_TASKS.get(str(target))
        if future:
            if future.done():
                error=future.exception()
                if error:result.update(state='failed',error=str(error))
            else:result['state']='running'
    return result


def run_test(service,pid,sid,source='reconstruction'):
    root,available=sources(service,pid,sid)
    if source not in available:raise InputError('No aligned stack for this source yet.')
    if not isinstance(sid,str) or not sid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sid):raise InputError('Invalid tilt series identity.')
    item=available[source];key,fingerprint=identity(item)
    parent=curation_directory(root,sid)
    if parent.is_symlink() or parent.parent.is_symlink():raise InputError('Invalid curation directory.')
    target=storage_path(parent, source+'_'+key[:16]+'.json')
    def task():
        import mrcfile
        from ..curation_2d import diagnose
        with mrcfile.mmap(item.get('registration_stack',item['stack'])) as m:images=np.array(m.data,np.float32)
        angles,indices=source_angles(item);report=diagnose(images,angles)
        for row in report['rows']:
            row.update(number=int(indices[row['index']])+1,source_index=int(indices[row['index']]))
        for pair in report['pairs']:
            pair.update(left_number=int(indices[pair['left_index']])+1,right_number=int(indices[pair['right_index']])+1)
        if identity(item)[0]!=key:raise ValueError('Aligned stack changed during the test; run it again.')
        report.update(source=source,fingerprint=fingerprint)
        parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(target)
        return report
    with _LOCK:
        previous=_TASKS.get(str(target))
        if not target.is_file() and (previous is None or previous.done()):_TASKS[str(target)]=_POOL.submit(task)
    return status(service,pid,sid,source)


def plane_png(service,pid,sid,index,source='reconstruction',*,discarded=False):
    import mrcfile
    from PIL import Image
    _,available=sources(service,pid,sid)
    if source not in available:raise InputError('No aligned stack yet.')
    if discarded:
        _,active=source_angles(available[source])
        angles=np.loadtxt(available[source]['angles'],ndmin=1)
        if not 0<=index<len(angles) or index in active:raise InputError('Not a discarded tilt.')
    else:
        with mrcfile.mmap(available[source]['stack']) as m:
            if not 0<=index<len(m.data):raise InputError('Tilt index out of range.')
    from .curation_display import display_plane
    plane=display_plane(service,available[source],index,source_index=index if discarded else None)
    if discarded and plane is None:raise InputError('No raw observations for this discarded tilt.')
    if plane is None:
        with mrcfile.mmap(available[source]['stack']) as m:plane=np.array(m.data[index],np.float32)
    finite=plane[np.isfinite(plane)]
    if not finite.size:raise InputError('No finite image pixels.')
    lo,hi=np.percentile(finite,[1,99]);scaled=np.clip((np.nan_to_num(plane,nan=lo)-lo)/max(float(hi-lo),1e-12),0,1)
    stream=io.BytesIO();Image.fromarray((scaled*255).astype(np.uint8)).save(stream,format='PNG');return stream.getvalue()
