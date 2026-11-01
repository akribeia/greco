"""User view selection, fixed before a reconstruction job is submitted."""
from .projects import InputError


def save(service,project_id,series_id,excluded):
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        from .reconstruction_assemblies import current
        record=current(data)
        if not record or record['mode']!='advanced':raise InputError('View exclusion requires Advanced reconstruction.')
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        if row.get('reconstruction_views_locked') or row.get('reconstruction',{}).get('work') or row.get('queued_jobs',{}).get('reconstruction'):raise InputError('View selection is locked after submission.')
        from .preprocessing import status
        prep=status(service,project_id,series_id)
        if prep.get('state')!='completed':raise InputError('Complete preprocessing before selecting views.')
        count=len(prep.get('angles',[]))
        if not isinstance(excluded,list) or any(type(i) is not int or not 0<=i<count for i in excluded) or len(set(excluded))!=len(excluded):raise InputError('Invalid excluded view indices.')
        if count-len(excluded)<7:raise InputError('Keep at least seven views.')
        # status() may have refreshed preprocessing metadata.
        data=service._read(root);row=next(r for r in data['series'] if r['id']==series_id)
        row['excluded_reconstruction_views']=sorted(excluded)
        service._write(root,data);return dict(excluded_views=sorted(excluded))
