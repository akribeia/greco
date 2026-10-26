"""Exclusive per-series choice; submitted picking inputs remain immutable."""
from greco.project_layout import project_root, join as storage_path, owned as layout_owned
import json
from pathlib import Path
from .projects import InputError
from .preprocessing import refresh

KINDS=('reconstruction','geometry_refinement','manual_curation')


def manual_parent(row):
    from .reconstruction_steps import find,previous_geometry
    step=find(row,row.get('picking_step_id'))
    if step['kind']!='manual_curation':raise InputError('Invalid manual curation selection.')
    return previous_geometry(row,step)


def selected(row):
    kind=row.get('picking_source') or 'reconstruction'
    if kind=='manual_curation':
        from .reconstruction_steps import find
        step=find(row,row.get('picking_step_id'))
        return 'geometry_refinement' if step.get('job') or manual_parent(row) else 'reconstruction'
    if kind not in KINDS:raise InputError('Invalid picking source.')
    return kind


def source_job(row):
    if row.get('picking_source')=='manual_curation':
        from .reconstruction_steps import job_for,find
        step=find(row,row.get('picking_step_id'))
        if step.get('job'):
            if step['job'].get('manual_action')=='refine':return dict(state='pending')
            return step['job']
        prior=manual_parent(row)
        return job_for(row,prior) if prior else row.get('reconstruction',{})
    kind=selected(row)
    if kind=='geometry_refinement' and row.get('picking_step_id'):
        from .reconstruction_steps import find,job_for
        step=find(row,row['picking_step_id'])
        if step['kind']!='geometry_refinement':raise InputError('The selected step does not produce a picking volume.')
        return job_for(row,step)
    return row.get(kind,{})


def set_source(service,pid,sid,kind,*,step_id=None):
    if kind is None:raise InputError('Select another output instead of clearing the picking source.')
    if kind not in KINDS:raise InputError('Invalid picking source.')
    if step_id and kind not in ('geometry_refinement','manual_curation'):raise InputError('Only pipeline steps accept a step identity.')
    with service.lock:
        root=service._root(pid);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==sid),None)
        if row is None:raise InputError('Unknown tilt series.')
        if kind is not None:
            choice=dict(row,picking_source=kind,picking_step_id=step_id)
            job=refresh(service,{'preprocessing':source_job(choice)})
            if job['state'] not in ('completed','submitted','running','queued'):
                raise InputError('Submit or complete the selected volume first.')
            work=service.path(job['work'],exists=False)
            if not layout_owned(root,work):raise InputError('Invalid source directory.')
        row['picking_source']=kind
        row['picking_step_id']=step_id
        service._write(root,data)
        return service._view(pid)


def volume_context(work):
    """Resolve geometry previews through their own recorded coarse lineage."""
    work=Path(work).resolve()
    config=json.loads((storage_path(work, 'config.json')).read_text())
    result_path=storage_path(work, 'output/web_result.json')
    result=json.loads(result_path.read_text()) if result_path.is_file() else {}
    coarse=storage_path(work, 'alignment');kind='reconstruction'
    if result.get('kind')=='geometry_refinement':
        kind='geometry_refinement';coarse=Path(config['coarse_run']).resolve()
        root=project_root(work)
        owner=root.joinpath(*work.relative_to(root).parts[:3])
        if work.parent!=owner/'refinement' or not work.name.isdigit():
            raise ValueError('Invalid refinement volume directory.')
        if not coarse.is_relative_to(owner) or result.get('source_reconstruction')!=str(coarse):
            raise ValueError('Geometry preview does not match its recorded reconstruction.')
        parent=json.loads((storage_path(coarse.parent, 'web_result.json')).read_text())
        if parent.get('ctf_template_metadata') is not None:
            result=dict(result,ctf_template_metadata=parent['ctf_template_metadata'])
    summary=json.loads((storage_path(coarse, 'summary.json')).read_text())
    return config,summary,result,dict(picking_source_kind=kind,picking_source_work=str(work),
        picking_source_run=str(coarse),rigid_geometry=result.get('geometry_file'))
