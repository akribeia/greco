"""Named reconstruction experiments with independent settings and outputs."""
from copy import deepcopy
import uuid
from .metadata_cache import read_json
from .projects import InputError

FIELDS=('reconstruction','reconstruction_history','processing_steps','geometry_refinement','angle_refinement','refinement','picking_source','picking_step_id','last_refinement_number')
SETTINGS=('reconstruction_denoisers','reconstruction_alignment_options','reconstruction_entry_sigma_px','reconstruction_thickness_percent','reconstruction_final_binning','reconstruction_advanced_options')


def current(data):
    return next((r for r in data.get('reconstruction_assemblies',[]) if r['id']==data.get('active_reconstruction_assembly')),None)


def capture(data):
    record=current(data)
    if record is None:return
    for key in SETTINGS:
        if key in data:record[key]=deepcopy(data[key])
        else:record.pop(key,None)
    for row in data['series']:
        saved=record.setdefault('series',{}).setdefault(row['id'],{})
        for key in FIELDS:
            if key in row:saved[key]=deepcopy(row[key])
            else:saved.pop(key,None)
        queued=row.get('queued_jobs',{}).get('reconstruction')
        if queued:saved['queued_reconstruction']=deepcopy(queued)
        else:saved.pop('queued_reconstruction',None)


def ensure(data):
    if 'reconstruction_assemblies' in data:return False
    data['reconstruction_assemblies']=[]
    if not any(row.get('reconstruction',{}).get('work') for row in data['series']):return True
    advanced=bool(data.get('reconstruction_advanced_options',{}).get('enabled'))
    record=dict(id=uuid.uuid4().hex[:8],name='Reconstruction 001',mode='advanced' if advanced else 'standard',legacy=True,series={})
    data['reconstruction_assemblies'].append(record);data['active_reconstruction_assembly']=record['id']
    for row in data['series']:row['reconstruction_assembly_id']=record['id']
    capture(data)
    for picker in data.get('pick_assemblies',[]):picker.setdefault('reconstruction_assembly_id',record['id'])
    return True


def activate(data,identifier):
    capture(data)
    record=next((r for r in data.get('reconstruction_assemblies',[]) if r['id']==identifier),None)
    if record is None:raise InputError('Unknown reconstruction.')
    data['active_reconstruction_assembly']=identifier
    for key in SETTINGS:
        data.pop(key,None)
        if key in record:data[key]=deepcopy(record[key])
    options=data.setdefault('reconstruction_advanced_options',{})
    options['enabled']=record['mode']=='advanced'
    for row in data['series']:
        saved=record.get('series',{}).get(row['id'],{})
        for key in FIELDS:
            row.pop(key,None)
            if key in saved:row[key]=deepcopy(saved[key])
        from .reconstruction_steps import ensure as ensure_steps
        ensure_steps(row)
        row['reconstruction_assembly_id']=identifier
        row.get('queued_jobs',{}).pop('reconstruction',None)
        if saved.get('queued_reconstruction'):row.setdefault('queued_jobs',{})['reconstruction']=deepcopy(saved['queued_reconstruction'])


def create(service,project_id,name,mode):
    if not isinstance(name,str) or not 1<=len(name.strip())<=100:raise InputError('Enter a reconstruction name (1–100 characters).')
    if mode not in ('standard','advanced'):raise InputError('Choose Simple or Advanced.')
    with service.lock:
        root=service._root(project_id);data=service._read(root);ensure(data);capture(data)
        if any(r['name'].casefold()==name.strip().casefold() for r in data['reconstruction_assemblies']):raise InputError('A reconstruction already has this name.')
        data.update(layout_version=3,reconstruction_layout_version=3)
        identifier=uuid.uuid4().hex[:8]
        record=dict(id=identifier,name=name.strip(),mode=mode,series={})
        for key in SETTINGS:
            if key in data and key!='reconstruction_advanced_options':record[key]=deepcopy(data[key])
        if mode=='advanced':
            record['reconstruction_denoisers']=['tinycare']
            record['reconstruction_alignment_options']=dict(record.get('reconstruction_alignment_options',{}),smoothness_strength=0.,rescue_tilts=True)
        record['reconstruction_advanced_options']=dict(enabled=mode=='advanced',defocus_handedness=data.get('orientation',{}).get('df_hand',1))
        data['reconstruction_assemblies'].append(record);activate(data,identifier)
        service._write(root,data);return service._view(project_id)


def select(service,project_id,identifier):
    with service.lock:
        root=service._root(project_id);data=service._read(root);created=ensure(data)
        if data.get('active_reconstruction_assembly')==identifier:return service._view(project_id)
        activate(data,identifier)
        service._write(root,data,publish=created);return service._view(project_id)


def delete(service,project_id,identifier,confirmed=False):
    """Delete a whole named reconstruction only after checking all dependants."""
    import json,shutil
    from pathlib import Path
    from greco.project_layout import record_directory,config_paths
    from .particle_jobs import config_uses_work
    from .preprocessing import refresh
    if confirmed is not True:raise InputError('Confirm permanent reconstruction deletion.')
    with service.lock:
        root=service._root(project_id);data=service._read(root);capture(data)
        record=next((r for r in data.get('reconstruction_assemblies',[]) if r['id']==identifier),None)
        if record is None:raise InputError('Unknown reconstruction.')
        if record.get('legacy'):raise InputError('Migrate this reconstruction before deleting the whole collection.')
        for picker in data.get('pick_assemblies',[]):
            if picker.get('reconstruction_assembly_id')==identifier:
                raise InputError('Picking assembly '+picker['name']+' uses this reconstruction. Delete it first.')
        target=record_directory(root,record,'reconstruction')
        if target.parent!=root/'reconstruction' or target.is_symlink() or target.parent.is_symlink() or target.resolve()!=target:
            raise InputError('Invalid reconstruction directory.')
        def jobs(value):
            if isinstance(value,dict):
                if value.get('work') and value.get('state'):yield value
                for child in value.values():yield from jobs(child)
            elif isinstance(value,list):
                for child in value:yield from jobs(child)
        checked=set()
        for job in jobs([record.get('series',{}),data.get('series',[])]):
            work=job['work']
            if not Path(work).is_relative_to(target) or work in checked:continue
            checked.add(work)
            if job.get('state')=='queued':raise InputError('Remove queued jobs before deleting this reconstruction.')
            if refresh(service,{'preprocessing':job}).get('state') in ('submitting','submitted','running','unknown'):
                raise InputError('Wait for active reconstruction jobs to finish before deleting the collection.')
        for item in config_paths(root):
            if item.is_relative_to(target):continue
            try:config=read_json(service,item)
            except (ValueError,OSError) as error:raise InputError('Cannot verify reconstruction dependencies.') from error
            inputs={k:v for k,v in config.items() if k not in ('series','project_snapshot','source_result')}
            if config_uses_work(inputs,target):raise InputError('A downstream job uses this reconstruction. Delete dependent jobs first.')
        for segmentation in data.get('segmentations',[]):
            if segmentation.get('reconstruction_assembly_id')==identifier or config_uses_work(segmentation,target):raise InputError('A segmentation uses this reconstruction. Delete it first.')
        trash=target.with_name('.'+target.name+'.deleting-'+uuid.uuid4().hex[:8])
        moved=target.exists()
        if moved:target.rename(trash)
        try:
            data['reconstruction_assemblies']=[r for r in data['reconstruction_assemblies'] if r['id']!=identifier]
            if data.get('active_reconstruction_assembly')==identifier:
                data.pop('active_reconstruction_assembly',None)
                for key in SETTINGS:data.pop(key,None)
                for row in data['series']:
                    for key in FIELDS:row.pop(key,None)
                    row.pop('reconstruction_assembly_id',None)
                    row.get('queued_jobs',{}).pop('reconstruction',None)
                if data['reconstruction_assemblies']:activate(data,data['reconstruction_assemblies'][0]['id'])
            service._write(root,data)
        except BaseException:
            if moved:trash.rename(target)
            raise
        if moved:shutil.rmtree(trash)
        return service._view(project_id)


def inactive_jobs(data):
    """Mutable job slots belonging to reconstruction cards other than the active one."""
    for record in data.get('reconstruction_assemblies',[]):
        if record['id']==data.get('active_reconstruction_assembly'):continue
        for saved in record.get('series',{}).values():
            for key in ('reconstruction','geometry_refinement','angle_refinement','refinement'):
                if saved.get(key):yield saved,key
            for step in saved.get('processing_steps',[]):
                if step.get('job'):yield step,'job'


def has_active_jobs(data):
    busy={'queued','submitting','submitted','running','unknown'}
    for row in data['series']:
        if row.get('queued_jobs',{}).get('reconstruction'):return True
        if any(row.get(k,{}).get('state') in busy for k in ('reconstruction','geometry_refinement','angle_refinement','refinement')):return True
        if any(s.get('job',{}).get('state') in busy for s in row.get('processing_steps',[])):return True
    if any(owner[key].get('state') in busy for owner,key in inactive_jobs(data)):return True
    return any(saved.get('queued_reconstruction') for r in data.get('reconstruction_assemblies',[]) for saved in r.get('series',{}).values())


def source_row(data,row,identifier):
    """Read one reconstruction without changing the selected card."""
    if identifier==data.get('active_reconstruction_assembly'):return row
    record=next((r for r in data.get('reconstruction_assemblies',[]) if r['id']==identifier),None)
    if record is None:raise InputError('Unknown reconstruction.')
    return {**{k:v for k,v in row.items() if k not in FIELDS},
            **record.get('series',{}).get(row['id'],{}),
            'reconstruction_assembly_id':identifier}


def inactive_rows(data):
    """Independent queue contexts for reconstruction cards not selected in the UI."""
    for record in data.get('reconstruction_assemblies',[]):
        if record['id']==data.get('active_reconstruction_assembly'):continue
        for row in data['series']:
            saved=record.setdefault('series',{}).setdefault(row['id'],{})
            proxy=source_row(data,row,record['id'])
            proxy['queued_jobs']={}
            if saved.get('queued_reconstruction'):proxy['queued_jobs']['reconstruction']=saved['queued_reconstruction']
            yield proxy,saved


def save_proxy(proxy,saved):
    for key in FIELDS:
        if key in proxy:saved[key]=deepcopy(proxy[key])
        else:saved.pop(key,None)
    queued=proxy.get('queued_jobs',{}).get('reconstruction')
    if queued:saved['queued_reconstruction']=deepcopy(queued)
    else:saved.pop('queued_reconstruction',None)
