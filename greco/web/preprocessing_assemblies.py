"""Named preprocessing collections and explicit reconstruction dependencies."""
from copy import deepcopy
from pathlib import Path
import uuid
from .projects import InputError

FIELDS=('preprocessing','preprocessing_history','ctf_redo','ctf_redo_history','handedness_test','handedness_test_history')
SETTINGS=('preprocessing_options','orientation','orientation_convention','ctf_axis_protocol')


def current(data):
    return next((r for r in data.get('preprocessing_assemblies',[]) if r['id']==data.get('active_preprocessing_assembly')),None)


def capture(data):
    record=current(data)
    if record is None:return
    for key in SETTINGS:
        if key in data:record[key]=deepcopy(data[key])
    for row in data['series']:
        saved=record.setdefault('series',{}).setdefault(row['id'],{})
        for key in FIELDS:
            if key in row:saved[key]=deepcopy(row[key])
            else:saved.pop(key,None)
        queued=row.get('queued_jobs',{}).get('preprocessing')
        if queued:saved['queued_preprocessing']=deepcopy(queued)
        else:saved.pop('queued_preprocessing',None)


def activate(data,identifier):
    capture(data)
    record=next((r for r in data.get('preprocessing_assemblies',[]) if r['id']==identifier),None)
    if record is None:raise InputError('Unknown preprocessing collection.')
    data['active_preprocessing_assembly']=identifier
    for key in SETTINGS:
        data.pop(key,None)
        if key in record:data[key]=deepcopy(record[key])
    for row in data['series']:
        saved=record.get('series',{}).get(row['id'],{})
        for key in FIELDS:
            row.pop(key,None)
            if key in saved:row[key]=deepcopy(saved[key])
        row['preprocessing_assembly_id']=identifier
        row.get('queued_jobs',{}).pop('preprocessing',None)
        if saved.get('queued_preprocessing'):row.setdefault('queued_jobs',{})['preprocessing']=deepcopy(saved['queued_preprocessing'])


def create(service,project_id,name):
    if not isinstance(name,str) or not 1<=len(name.strip())<=100:raise InputError('Enter a preprocessing name (1–100 characters).')
    with service.lock:
        root=service._root(project_id);data=service._read(root);capture(data)
        if data.get('preprocessing_layout_version')!=3:raise InputError('This project is waiting for preprocessing layout migration.')
        if any(r['name'].casefold()==name.strip().casefold() for r in data.get('preprocessing_assemblies',[])):raise InputError('A preprocessing already has this name.')
        record=dict(id=uuid.uuid4().hex[:8],name=name.strip(),series={})
        for key in SETTINGS:
            if key in data:record[key]=deepcopy(data[key])
        data.setdefault('preprocessing_assemblies',[]).append(record)
        activate(data,record['id']);service._write(root,data);return service._view(project_id)


def select(service,project_id,identifier):
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        activate(data,identifier);service._write(root,data);return service._view(project_id)


def delete(service,project_id,identifier,confirmed=False):
    if confirmed is not True:raise InputError('Confirm permanent preprocessing deletion.')
    import shutil
    from .preprocessing import refresh
    from ..project_layout import record_directory
    with service.lock:
        root=service._root(project_id);data=service._read(root);capture(data)
        record=next((r for r in data.get('preprocessing_assemblies',[]) if r['id']==identifier),None)
        if record is None:raise InputError('Unknown preprocessing collection.')
        users=[r['name'] for r in data.get('reconstruction_assemblies',[]) if r.get('preprocessing_assembly_id')==identifier]
        if users:raise InputError('Used by reconstruction: '+', '.join(users)+'. Delete dependent reconstructions first.')
        for saved in record.get('series',{}).values():
            for key in ('preprocessing','ctf_redo','handedness_test','queued_preprocessing'):
                if saved.get(key) and refresh(service,{'preprocessing':saved[key]}).get('state') in ('submitted','running','submitting','unknown','queued'):
                    raise InputError('Wait for preprocessing jobs to finish or delete them first.')
        target=record_directory(root,record,'preprocessing')
        if target.is_symlink() or target.parent!=root/'preprocessing':raise InputError('Invalid preprocessing directory.')
        if target.exists():shutil.rmtree(target)
        data['preprocessing_assemblies'].remove(record)
        if data.get('active_preprocessing_assembly')==identifier:
            for row in data['series']:
                for key in FIELDS:row.pop(key,None)
                row.pop('preprocessing_assembly_id',None)
            data.pop('active_preprocessing_assembly',None)
            if data['preprocessing_assemblies']:activate(data,data['preprocessing_assemblies'][0]['id'])
        service._write(root,data);return service._view(project_id)


def bind_reconstruction(service,project_id,identifier,preprocessing_id):
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        from .reconstruction_assemblies import capture as capture_reconstruction
        capture_reconstruction(data)
        record=next((r for r in data.get('reconstruction_assemblies',[]) if r['id']==identifier),None)
        if record is None:raise InputError('Unknown reconstruction.')
        if not any(r['id']==preprocessing_id for r in data.get('preprocessing_assemblies',[])):raise InputError('Select a preprocessing collection.')
        if record.get('preprocessing_assembly_id')!=preprocessing_id:
            if any(saved.get('reconstruction') or saved.get('queued_reconstruction') for saved in record.get('series',{}).values()):raise InputError('Create a new reconstruction to change its preprocessing inputs.')
            record['preprocessing_assembly_id']=preprocessing_id
        activate(data,preprocessing_id);service._write(root,data);return service._view(project_id)
