"""Read-only per-view correction history from the immutable STA source chain."""
from greco.project_layout import join as storage_path, job_work
from collections import defaultdict
from pathlib import Path
import math
from statistics import median
from .projects import InputError
from . import particle_jobs as jobs
from .assemblies import assembly_directory
from ..particle_star import rows
from ..sta.corrections import source_chain


def history(service,project_id,identifier,kind,name):
    if kind not in ('ctf_refinement','polishing'):raise InputError('Invalid correction history kind.')
    job=jobs.volume_job(service,project_id,identifier,kind,name)
    if job.get('state')!='completed':raise InputError('Correction history requires a completed job.')
    with service.lock:
        root=service._root(project_id);record=jobs.assembly(service._read(root),identifier)
        directory=assembly_directory(root,record).resolve()
    manifest=service.path(job['particles']).resolve()
    if manifest != storage_path(Path(job['work']),'output/particles.star').resolve() or not manifest.is_relative_to(directory):
        raise InputError('Invalid correction history source.')
    # Check every link before the generic source-chain reader opens it.
    current=manifest;seen=set()
    for _ in range(32):
        if current in seen or not current.is_relative_to(directory):raise InputError('Invalid correction history chain.')
        seen.add(current)
        general=next((r for b,r in rows(current) if b=='general'),{})
        source=general.get('grecoSourceParticleStar')
        if not source:break
        parent=current.parent
        current=Path(source)
        if not current.is_absolute():current=parent/current
        current=current.resolve()
    else:raise InputError('Correction history chain is too deep.')
    base,chain=source_chain(manifest)
    views={};tomos={};owners={}
    for block,row in rows(base):
        if block=='tomograms':tomos[row['grecoTomogramId']]=row['grecoTomogramName']
        elif block=='views':views[(row['grecoTomogramId'],row['grecoViewId'])]=row
        elif block=='particles':owners[row['grecoParticleId']]=row['grecoTomogramId']
        elif block=='observations':break
    observations={};geometry={};snapshots=[]
    for source in reversed(chain):
        correction_kind=None
        for block,row in rows(source):
            if block=='general':correction_kind=row.get('grecoCorrectionKind')
            elif block=='observations':observations.setdefault((row['grecoParticleId'],row['grecoViewId']),{}).update(row)
            elif block=='view_corrections':geometry.setdefault((row['grecoTomogramId'],row['grecoViewId']),{}).update({k:v for k,v in row.items() if v is not None})
        compatible={'polishing':{'polishing','polishing_gp3d','polishing_rigid_gp3d'},'ctf_refinement':{'ctf_refinement','joint_ctf'}}
        if correction_kind not in compatible[kind]:continue
        grouped=defaultdict(list)
        for (pid,view),row in observations.items():
            if pid in owners:grouped[(owners[pid],view)].append(row)
        series=[]
        for tid,label in tomos.items():
            points=[]
            for key,view in sorted(views.items(),key=lambda item:item[1].get('grecoTiltAngleDeg',item[0][1])):
                if key[0]!=tid:continue
                group=grouped[key];update=geometry.get(key,{})
                point=dict(view=int(view.get('grecoTiltIndex',key[1]))+1,angle=view.get('grecoTiltAngleDeg',0.))
                if kind=='ctf_refinement':
                    delta=sum(float(update.get('grecoCentralDefocus'+d+'Angst',view.get('grecoCentralDefocus'+d+'Angst',0.)))-float(view.get('grecoCentralDefocus'+d+'Angst',0.)) for d in 'UV')/2
                    point['defocus']=delta+(median(float(r.get('grecoDefocusOffsetAngst',0.)) for r in group) if group else 0.)
                else:
                    for d in 'XY':
                        point[d.lower()]=median(float(r.get('grecoResidualShift'+d+'Angst',0.)) for r in group) if group else 0.
                        point['rigid_'+d.lower()]=float(update.get('grecoGeometryShift'+d+'Angst',view.get('grecoGeometryShift'+d+'Angst',0.)))-float(view.get('grecoGeometryShift'+d+'Angst',0.))
                if not all(math.isfinite(v) for v in point.values()):raise InputError('Non-finite view history.')
                points.append(point)
            series.append(dict(id=tid,name=label,points=points))
        snapshots.append(dict(job=job_work(source).name,series=series))
    return dict(kind=kind,baseline=job_work(base).name,units='Å',snapshots=snapshots)
