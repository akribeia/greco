"""Project-scoped read-only GPU handedness diagnostics."""
from greco.project_layout import join as storage_path, owned as layout_owned
import json
import math
from pathlib import Path
from .queue import prepare_work
from .projects import InputError
from .preprocessing import refresh, render_script, command_result, ctf_axis_from_orientation
from ..ctf_local_refinement import DEFOCUS_CONVENTION


def submit(service, project_id, series_id, rotation_deg, flip_y):
    if isinstance(rotation_deg,bool) or not isinstance(flip_y,bool):raise InputError('Invalid orientation.')
    rotation=float(rotation_deg)
    if not math.isfinite(rotation) or abs(rotation)>360:raise InputError('Invalid rotation.')
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        slurm=data.get('slurm',{})
        if not slurm.get('ready'):raise InputError('Save cluster settings first.')
        current=refresh(service,{'preprocessing':row.get('handedness_test',{})})
        if current['state'] in ('submitted','running','unknown'):raise InputError('Handedness test already active.')
        prep=refresh(service,row)
        if prep['state']!='completed':raise InputError('Complete preprocessing with Local CTF refinement first.')
        source=service.path(prep['work'])
        if not layout_owned(root,source):raise InputError('Invalid preprocessing output.')
        from .ctf_view import tilt_records
        output=storage_path(source, 'output');binning=prep.get('output_binning',prep.get('options',{}).get('output_binning',8))
        records=tilt_records(output,binning)
        folders=[]
        for record in records:
            folder=storage_path(storage_path(output, 'per_tilt'), f"tilt_{int(record['mdoc_index']):04d}")
            for name in ('ctf.json','ctf_tiles.npy','ctf_tiles_metadata.npz'):
                if not (storage_path(folder, name)).is_file():raise InputError('Bin1 tile spectra are missing. Repeat preprocessing with Local CTF refinement enabled.')
            folders.append(str(folder))
        work=prepare_work(service,root,row,'handedness_test')
        orientation=dict(rotation_deg=rotation,flip_y=flip_y)
        config=dict(folders=folders,orientation=orientation,axis_deg=ctf_axis_from_orientation(orientation),defocus_convention=DEFOCUS_CONVENTION)
        path=storage_path(work, 'config.json');path.write_text(json.dumps(config,indent=2))
        options=dict(cpus=data.get('preprocessing_options',{}).get('cpus',4),gpus=1)
        (storage_path(work, 'job.sbatch')).write_text(render_script(slurm['template'],work,path,options,'greco.web.handedness_worker'))
        job=dict(state='submitted',work=str(work),host=slurm.get('submission_host',''),orientation=orientation,source_work=str(source))
        try:
            job_id=command_result(['sbatch','--parsable',str(storage_path(work, 'job.sbatch'))],job['host']).split(';')[0]
            if not job_id.isdigit():raise InputError('Unexpected Slurm response.')
            job['job_id']=job_id
        except InputError as error:job.update(state='failed',error=str(error))
        row.setdefault('handedness_test_history',[]).append(job);row['handedness_test']=job
        service._write(root,data)
        return job


def status(service,project_id,series_id):
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        job=refresh(service,{'preprocessing':row.get('handedness_test',{})})
        convention=job.get('defocus_convention')
        if job.get('state')=='completed' and job.get('work'):
            # refresh merges persisted metadata; only the actual result can
            # certify which convention the worker used.
            try:
                result=json.loads((storage_path(storage_path(Path(job['work']), 'output'), 'web_result.json')).read_text())
                convention=result.get('defocus_convention')
            except (OSError,ValueError):
                convention=None
        if job.get('state')=='completed' and convention!=DEFOCUS_CONVENTION:
            job=dict(job,state='outdated',conclusive=False,error='This handedness result predates the physical-defocus fix. Run Test defocus handedness again, then apply it to the project.')
        if job.get('state')=='completed' and job.get('source_work')!=row.get('preprocessing',{}).get('work'):
            job=dict(job,state='outdated',conclusive=False,error='CTF preprocessing changed. Run Test defocus handedness again on the current CTF.')
        if row.get('handedness_test') and row['handedness_test']!=job:
            row['handedness_test']=job;service._write(root,data)
        return job
