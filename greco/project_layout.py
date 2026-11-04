"""Definitive project storage and artifact paths.

Numerical workers pass artifact names to join(); it publishes them directly
inside the single project layout without selecting alternate storage formats.
"""
import json
import os
import re
import unicodedata
from pathlib import Path
STA_CATEGORIES={'extraction':'extraction',
 'classification':'classification','refinement3d':'refinement','polishing':'polishing',
 'ctf_refinement':'ctf_refinement','postprocessing':'postprocessing'}
ROOTS=('preprocessing','reconstruction','picking','sta','segmentation')
_ROOT_CACHE=set()

def safe_name(value):
    text=unicodedata.normalize('NFC',str(value)).strip()
    text=re.sub(r'[^\w. -]+','_',text,flags=re.UNICODE).replace(' ','_')
    text=text.strip('._')
    return text[:100] or 'Unnamed'

def assign_names(records):
    used=set()
    for record in records:
        name=record.get('storage_name')
        if name:
            if name!=safe_name(name) or name.casefold() in used:raise ValueError('Invalid or duplicate storage name.')
            used.add(name.casefold())
    for record in records:
        if record.get('storage_name'):continue
        base=safe_name(record.get('name','Unnamed'));name=base;n=2
        while name.casefold() in used:
            suffix='_'+str(n);name=base[:100-len(suffix)]+suffix;n+=1
        record['storage_name']=name;used.add(name.casefold())

def project_root(path):
    path=Path(path)
    for cached in _ROOT_CACHE:
        if path==cached or path.is_relative_to(cached):return cached
    for parent in (path,*path.parents):
        if (parent/'project.json').is_file():
            _ROOT_CACHE.add(parent)
            return parent
    return None





def owned(root, path):
    root = Path(root)
    path = Path(path)
    return any((path.is_relative_to(root / name) for name in ROOTS)) and path != root

def series_name(row):
    return row.get('storage_name',safe_name(row.get('name',row['id'])))

def record_directory(root,record,kind):
    return Path(root)/kind/record.get('storage_name',safe_name(record['name']))

def series_row(root,sid):
    data=json.loads((Path(root)/'project.json').read_text())
    return next(r for r in data['series'] if r['id']==sid)

def work_path(root,row,kind,queued=False):
    root=Path(root);series=series_name(row)
    if kind=='picking':
        data=json.loads((root/'project.json').read_text())
        record=next((r for r in data.get('pick_assemblies',[]) if r['id']==row.get('pick_assembly_id')),None)
        if record is None:raise ValueError('Select a picking category first.')
        result=record_directory(root,record,'picking')/'series'/series
    elif kind in ('preprocessing','ctf_redo','handedness_test'):
        data=json.loads((root/'project.json').read_text())
        record=next((r for r in data.get('preprocessing_assemblies',[]) if r['id']==row.get('preprocessing_assembly_id')),None)
        if record is None:raise ValueError('Create a preprocessing collection first.')
        result=record_directory(root,record,'preprocessing')/series
        if kind=='ctf_redo':result=result/'.ctf-pending'
        elif kind=='handedness_test':result=result/'ctf/handedness'
    elif kind=='reconstruction':
        data=json.loads((root/'project.json').read_text()) if (root/'project.json').exists() else {}
        record=next((r for r in data.get('reconstruction_assemblies',[]) if r['id']==row.get('reconstruction_assembly_id')),None)
        if row.get('reconstruction_assembly_id') and record is None:raise ValueError('Unknown reconstruction identity.')
        if record is None:raise ValueError('Create a reconstruction first.')
        result=record_directory(root,record,'reconstruction')/series
    else:
        from .web.reconstruction_steps import ensure
        ensure(row)
        step=next((s for s in row['processing_steps'] if s['kind']==kind),None)
        if step is None:raise ValueError('Create a reconstruction refinement step first.')
        result=work_path(root,row,'reconstruction')/'refinement'/f"{step['number']:03d}"
    return result.with_name(result.name+'.queued') if queued else result

def job_work(path):
    """Find the job owning an artifact or an immutable action."""
    path=Path(path);root=project_root(path)
    if root:
        work,_,_=_context(path,root)
        if work is not None:return work
    for parent in (path,*path.parents):
        if parent.name!='alignment' and (parent/'config.json').is_file():return parent
    raise ValueError(f'No job configuration owns {path}')

def preprocessing_work(path):
    path=Path(path);root=project_root(path)
    if root:
        work,_,kind=_context(path,root)
        if kind=='preprocessing':return work
    return job_work(path)

def _context(path, root):
    parts = path.relative_to(root).parts
    if len(parts)>=2 and parts[0]=='preprocessing':
        length=3
        if len(parts)<length:return None,[],None
        work=root.joinpath(*parts[:length]);rest=list(parts[length:])
        if rest and rest[0]=='.ctf-pending':return work/rest[0],rest[1:],'preprocessing'
        if rest[:2]==['ctf','handedness']:return work/'ctf/handedness',rest[2:],'handedness'
        return work,rest,'preprocessing'
    if len(parts)>=3 and parts[0]=='reconstruction':
        length=5 if len(parts)>=5 and parts[3]=='refinement' else 3
        if len(parts)>=length+3 and parts[length:length+2]==('provenance','actions'):length+=3
        return (root.joinpath(*parts[:length]),list(parts[length:]),'reconstruction')
    if len(parts) >= 4 and parts[0] == 'sta':
        return (root.joinpath(*parts[:4]), list(parts[4:]), 'sta')
    if len(parts) >= 4 and parts[0] == 'picking':
        return (root.joinpath(*parts[:4]), list(parts[4:]), 'picking')
    if len(parts) >= 4 and parts[0] == 'segmentation':
        length=5 if len(parts)>=5 and parts[3]=='layers' else 4
        return (root.joinpath(*parts[:length]),list(parts[length:]),'segmentation')
    return (None, [], None)

def _binning(work):
    try:
        data=json.loads((work/'config.json').read_text())
        return data.get('final_binning') or (8 if 'coarse_run' in data else data.get('binning') or data.get('options',{}).get('binning') or data.get('inputs',{}).get('binning',8))
    except (OSError,ValueError):return 8

def join(base,*pieces):
    path=Path(base).joinpath(*pieces);root=project_root(path)
    if root is None:return path
    work,rest,kind=_context(path,root)
    if work is None:return path
    if rest and rest[0]=='output':rest=rest[1:]
    if not rest:return work
    if rest[0]=='web_result.json':rest[0]='result.json'
    if rest[0] in ('slurm.out','slurm.err'):
        rest=['logs',rest[0] if kind in ('reconstruction','picking') else 'stdout.log' if rest[0]=='slurm.out' else 'stderr.log',*rest[1:]]
    if kind=='preprocessing':
        if len(rest)==2 and rest[0] in ('motion','alignment','ctf') and rest[1].startswith('greco_'):rest=[rest[1]]
        if len(rest)>=4 and rest[:2]==['ctf','tilts']:
            name=rest[3]
            if name in ('ctf_tiles.npy','ctf_tiles_metadata.npz'):rest=['ctf','spectra',*rest[2:]]
            elif not name.startswith('ctf') and name!='initial_ctf.json':rest=['motion','tilts',*rest[2:]]
        if rest[0]=='per_tilt':
            category='ctf'
            if len(rest)>=3:
                name=rest[2]
                if name in ('ctf_tiles.npy','ctf_tiles_metadata.npz'):category='spectra'
                elif not name.startswith('ctf') and name!='initial_ctf.json':category='motion'
            rest=['ctf','spectra',*rest[1:]] if category=='spectra' else [category,'tilts',*rest[1:]]
        elif rest[0].startswith('greco_'):
            name=rest[0]
            rest=['ctf' if name.startswith('greco_ctf') else 'alignment' if name.startswith('greco_2d_aligned') else 'motion',*rest]
        else:
            names={'input.mdoc':['inputs','input.mdoc'],'manifest.json':['inputs','manifest.json'],
                   'movies':['inputs','movies'],'alignment2d.json':['logs','alignment.json'],
                   'progress.json':['logs','progress.json'],'progress.tmp':['logs','progress.tmp']}
            if rest[0] in names:rest=[*names[rest[0]],*rest[1:]]
    elif kind=='sta':
        from .sta_layout import enabled,artifact_relative
        if enabled(root):rest=artifact_relative(work,rest)
    elif kind=='picking':
        from .picking_layout import artifact_relative
        rest=artifact_relative(rest)
    elif kind=='reconstruction':
        from .reconstruction_layout import artifact_relative
        rest=artifact_relative(work,rest)
    if kind=='preprocessing' and rest:
        from .preprocessing_layout import stack_name
        stack=stack_name(rest[-1])
        if stack and (len(rest)==1 or rest[0] in ('motion','alignment','ctf','stacks')):rest=['stacks',stack]
    return work.joinpath(*rest)

def ensure_output(path,**kwargs):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);root=project_root(path)
    if root is None:return path
    work,_,kind=_context(path,root)
    if work is None:return path
    (work/'logs').mkdir(parents=True,exist_ok=True)
    if kind=='preprocessing':
        for name in ('motion','motion/tilts','ctf/spectra','ctf/tilts','alignment','stacks','inputs/movies'):
            (work/name).mkdir(parents=True,exist_ok=True)
        try:data=json.loads((work/'config.json').read_text())
        except (OSError,ValueError):data={}
        for tilt in data.get('series',{}).get('tilts',[]):
            for name in ('motion/tilts','ctf/tilts','ctf/spectra'):
                (work/name/f"tilt_{int(tilt['z']):04d}").mkdir(parents=True,exist_ok=True)
    elif kind=='sta':
        from .sta_layout import enabled
        if enabled(root):
            names=['optimisation','provenance','inputs']
            if work.parent.name in ('refinement','classification'):names+=['iterations','volumes']
            if work.parent.name=='postprocessing':names+=['volumes']
            if work.parent.name=='extraction':names+=['particles']
            for name in names:(path/name).mkdir(parents=True,exist_ok=True)
    elif kind=='picking':
        for name in ('inputs','matching','diagnostics'):(work/name).mkdir(parents=True,exist_ok=True)
    elif kind=='reconstruction':
        for name in ('alignment','inputs','registration','provenance','tinycare','volumes'):
            (work/name).mkdir(parents=True,exist_ok=True)
    return path

def config_paths(root, *, kinds=ROOTS):
    """Configurations owned by managed jobs, excluding copied series siblings."""
    root = Path(root)
    for name in kinds:
        if name not in ROOTS:raise ValueError('Unknown job category.')
        parent = root / name
        if parent.is_dir():
            if name=='picking':
                # Job definitions and the matching input record live at fixed depths.
                yield from parent.glob('*/series/*/config.json')
                yield from parent.glob('*/series/*/matching/config.json')
            else:
                # Scientific payloads cannot own jobs. Avoid walking particle
                # stacks, volume exports and native movie/spectrum trees.
                payloads={'particles','volumes','stacks','spectra','movies'}
                for folder,directories,files in os.walk(parent,followlinks=False):
                    directories[:]=[d for d in directories if d not in payloads]
                    if 'config.json' in files:yield Path(folder)/'config.json'

def initialize(root,data,*,previous=None):
    """Create category metadata and input references; never copy input images."""
    root=Path(root)
    previous=previous or {}
    def publish(path,value,old=None):
        if value!=old:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,indent=2))
    for key in ('series','preprocessing_assemblies','reconstruction_assemblies','pick_assemblies','particle_assemblies','segmentations'):
        assign_names(data.get(key,[]))
    for row in data['series']:
        from .web.reconstruction_steps import ensure
        ensure(row)
    inputs=root/'inputs'
    def inputs_manifest(value):
        return {'series':[dict(id=r['id'],name=r['name'],mdoc_path=r.get('mdoc_path'),gain_path=r.get('gain_path'),gain_flip_y=r.get('gain_flip_y',False),movies=[t.get('movie_path') for t in r.get('tilts',[])]) for r in value.get('series',[])]}
    publish(inputs/'manifest.json',inputs_manifest(data),inputs_manifest(previous) if previous else None)
    for key,folder,filename in [('preprocessing_assemblies','preprocessing','category.json'),('reconstruction_assemblies','reconstruction','category.json'),('pick_assemblies','picking','category.json'),('particle_assemblies','sta','assembly.json'),('segmentations','segmentation','category.json')]:
        old_records={record['id']:record for record in previous.get(key,[])}
        for record in data.get(key,[]):
            directory=record_directory(root,record,folder)
            publish(directory/filename,record,old_records.get(record['id']))

def publish_ctf(row, job):
    """Promote a successful staged CTF calculation, keeping the shared MC data."""
    import os, shutil, uuid
    work = Path(job['work'])
    source = Path(job['source_work'])
    root = project_root(work)
    if root is None:
        return False
    if work != source / '.ctf-pending' or work.is_symlink():
        raise ValueError('Invalid pending CTF folder.')
    result = json.loads((work / 'result.json').read_text())
    if result.get('state') != 'completed':
        raise ValueError('Only a successful CTF result may replace the current fit.')
    promotions=[]
    for name in ('ctf','stacks'):
        current=source/name;pending=work/name;old=source/('.'+name+'-previous')
        if old.exists():raise ValueError('An interrupted CTF promotion needs recovery.')
        if not pending.is_dir():raise ValueError('Pending CTF result lacks '+name+'.')
        promotions.append((current,pending,old))
    current,pending,_=promotions[0]
    moved_hand=(current/'handedness').exists()
    if moved_hand:os.replace(current/'handedness',pending/'handedness')
    promoted=[]
    try:
        for current,pending,old in promotions:
            os.replace(current,old)
            try:os.replace(pending,current)
            except BaseException:os.replace(old,current);raise
            promoted.append((current,pending,old))
    except BaseException:
        for current,pending,old in reversed(promoted):
            os.replace(current,pending);os.replace(old,current)
        current,pending,_=promotions[0]
        if moved_hand and (pending/'handedness').exists():os.replace(pending/'handedness',current/'handedness')
        raise
    current=source/'ctf'

    def rebase(v):
        if isinstance(v, str):
            return v.replace(str(work), str(source))
        if isinstance(v, dict):
            return {k: rebase(x) for (k, x) in v.items()}
        if isinstance(v, list):
            return [rebase(x) for x in v]
        return v
    config = json.loads((work / 'config.json').read_text())
    original = json.loads((source / 'config.json').read_text())
    original.update(options=config['options'], orientation=config.get('orientation'))
    (source / 'config.json').write_text(json.dumps(original, indent=2))
    (current / 'config.json').write_text(json.dumps(rebase(config), indent=2))
    for name in ('job.sbatch', 'logs', 'provenance'):
        p = work / name
        if p.exists():
            target = current / name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            os.replace(p, target)
    result = rebase(result)
    result['work'] = str(source)
    result['ctf_revision'] = uuid.uuid4().hex
    result['source_work'] = str(source)
    (source / 'result.json').write_text(json.dumps(result, indent=2))
    manifest = join(work, 'manifest.json')
    if manifest.is_file():
        join(source, 'manifest.json').write_text(json.dumps(rebase(json.loads(manifest.read_text())), indent=2))
    job.update(rebase(job), work=str(source), applied=True, ctf_revision=result['ctf_revision'])
    row['preprocessing'] = {**job, **result, 'work': str(source), 'kind': 'ctf_redo', 'applied': True}
    row['ctf_redo'] = dict(job)
    row['preprocessing_history'] = []
    row['ctf_redo_history'] = []
    for _,_,old in promotions:shutil.rmtree(old)
    shutil.rmtree(work)
    return True

def curation_directory(root, sid):
    base = work_path(root,series_row(root,sid),'reconstruction')
    return base / 'curation/cache'

def picker_assets(root, data):
    record = next((r for r in data.get('pick_assemblies', []) if r['id'] == data.get('active_pick_assembly')), None)
    if record is None:
        raise ValueError('Create a picking category before selecting its template.')
    return record_directory(root, record, 'picking') / 'assets'


