"""Submit patch-only native-resolution refinement, never a global WBP."""
from greco.project_layout import join as storage_path, preprocessing_work, owned as layout_owned
from greco.preprocessing_star import read_table
import json
from pathlib import Path
import math
from .projects import InputError
from .preprocessing import refresh, render_script, command_result


def validate_options(raw):
    if not isinstance(raw,dict) or set(raw)-{'cube_size','save_landmarks','output_binning'}:
        raise InputError('Unknown refinement options.')
    size=raw.get('cube_size',127)
    if isinstance(size,bool) or not isinstance(size,int) or not 33<=size<=257 or size%2!=1:
        raise InputError('Cube size must be an odd integer from 33 to 257 (e.g. 127).')
    save=raw.get('save_landmarks',False)
    if not isinstance(save,bool):raise InputError('Save landmarks must be boolean.')
    binning=raw.get('output_binning',1)
    if isinstance(binning,bool) or not isinstance(binning,int) or binning not in (1,2):
        raise InputError('Output binning must be 1 or 2.')
    return dict(cube_size=size,save_landmarks=save,output_binning=binning)


def validate_angle_options(raw):
    if not isinstance(raw,dict) or set(raw)-{'refine_axis','refine_tilts','axis_bound_deg','tilt_bound_deg','max_passes','binning','patch_size'}:
        raise InputError('Unknown angle refinement options.')
    options=dict(refine_axis=raw.get('refine_axis',True),refine_tilts=raw.get('refine_tilts',True),
        axis_bound_deg=raw.get('axis_bound_deg',3.),tilt_bound_deg=raw.get('tilt_bound_deg',2.),max_passes=raw.get('max_passes',25),
        binning=raw.get('binning',8),patch_size=raw.get('patch_size',33))
    if type(options['binning']) is not int or options['binning'] not in (1,2,4,8):raise InputError('Geometry binning must be 1, 2, 4 or 8.')
    if type(options['patch_size']) is not int or options['patch_size'] not in (33,49,65):raise InputError('Landmark patch size must be 33, 49 or 65.')
    if isinstance(options['max_passes'],bool) or not isinstance(options['max_passes'],int) or not 1<=options['max_passes']<=100:
        raise InputError('Maximum passes must be an integer from 1 to 100.')
    if any(not isinstance(options[k],bool) for k in ('refine_axis','refine_tilts')):
        raise InputError('Angle refinement switches must be boolean.')
    if not (options['refine_axis'] or options['refine_tilts']):raise InputError('Select axis and/or tilt angle refinement.')
    for key,limit in [('axis_bound_deg',5),('tilt_bound_deg',3)]:
        value=options[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not .1<=value<=limit:
            raise InputError(f'{key} must be between 0.1 and {limit} degrees.')
        options[key]=float(value)
    return options


def validate_geometry_options(raw):
    defaults=dict(binning=8,final_binning=8,patch_size=33,max_passes=10,optimizer_iterations=1000,search_support_A=16.,denoisers=[],reuse_tinycare_model=False)
    if not isinstance(raw,dict) or set(raw)-set(defaults):raise InputError('Unknown rigid geometry options.')
    values=dict(defaults,**raw)
    from .final_denoising import validate_denoising_options
    values.update(validate_denoising_options({k:values[k] for k in ('denoisers','reuse_tinycare_model')}))
    methods=values['denoisers']
    if not isinstance(methods,list) or any(not isinstance(m,str) or m not in ('tinycare',) for m in methods) or len(set(methods))!=len(methods):
        raise InputError('Unknown TinyCARE denoising selection.')
    if type(values['binning']) is not int or values['binning'] not in (1,2,4,8):raise InputError('Geometry binning must be 1, 2, 4 or 8.')
    if type(values['final_binning']) is not int or values['final_binning'] not in (1,2,4,8,16):raise InputError('Final binning must be 1, 2, 4, 8 or 16.')
    values['patch_size']=1+256//values['binning']
    for key,limit in [('max_passes',100),('optimizer_iterations',10000)]:
        if type(values[key]) is not int or not 1<=values[key]<=limit:raise InputError('Invalid '+key+'.')
    v=values['search_support_A']
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not .1<=v<=500:raise InputError('Invalid search support.')
    return values


def submit(service,project_id,series_ids,raw,*,kind='refinement',queue_only=False,only_missing=False,source_kind='reconstruction'):
    from .queue import dispatch,prepare_work,store_job
    if kind not in ('refinement','angle_refinement','geometry_refinement'):raise InputError('Invalid refinement task.')
    if source_kind!='reconstruction':raise InputError('Invalid geometry source.')
    angular=kind in ('angle_refinement','geometry_refinement')
    options=validate_geometry_options(raw) if kind=='geometry_refinement' else validate_angle_options(raw) if angular else validate_options(raw)
    if not isinstance(series_ids,list) or not series_ids or len(set(series_ids))!=len(series_ids):
        raise InputError('Select distinct tilt series.')
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        slurm=data.get('slurm',{})
        if not slurm.get('ready'):raise InputError('Save cluster settings first.')
        rows={r['id']:r for r in data['series']}
        plans={};dependencies={};allowed={};array_entries=[]
        from .projects import active_series_ids
        series_ids=active_series_ids(rows,series_ids,bulk=only_missing)
        if only_missing:
            from .queue import needs_submission
            series_ids=[i for i in series_ids if needs_submission(rows[i].get(kind,{}))]
            if not series_ids:
                result=service._view(project_id);result['submission_errors']=[];return result
        for identifier in series_ids:
            if identifier not in rows:raise InputError('Unknown tilt series.')
            row=rows[identifier]
            running=refresh(service,{'preprocessing':row.get(kind,{})})
            if running['state'] in ('queued','submitting','submitted','running','unknown'):raise InputError(row['name']+': refinement already active.')
            recon=refresh(service,{'preprocessing':row.get(source_kind,{})})
            if recon['state'] in ('submitted','running') and recon.get('job_id'):
                if recon.get('host','')!=slurm.get('submission_host',''):raise InputError('Reconstruction dependency must use the same Slurm host.')
                work=service.path(recon['work'])
                if not layout_owned(root,work):raise InputError('Invalid reconstruction dependency.')
                source=json.loads((storage_path(work, 'config.json')).read_text())
                binning=source['inputs']['binning']
                if angular and binning!=8:raise InputError('Angle refinement requires bin8.')
                if not angular and binning % options['output_binning']:raise InputError('Output binning must divide the preprocessing binning.')
                plans[identifier]=dict(coarse_run=str(storage_path(work, 'alignment')),preprocessing_work=str(preprocessing_work(Path(source['inputs']['stack']))),inputs=source['inputs'],orientation=source['orientation'],main_revision=source['main_revision'])
                dependencies[identifier]=[recon['job_id']];allowed[identifier]=['preprocessing','reconstruction']
                if not angular and row.get('angle_refinement'):
                    calibration=refresh(service,{'preprocessing':row['angle_refinement']})
                    if calibration['state'] in ('submitted','running') and calibration.get('job_id'):
                        if calibration.get('host','')!=slurm.get('submission_host',''):raise InputError('Angle dependency must use the same Slurm host.')
                        angle_work=service.path(calibration['work'])
                        if not layout_owned(root,angle_work):raise InputError('Invalid angle dependency.')
                        plans[identifier]['angle_geometry']=str(storage_path(angle_work, 'output/angle_state.npz'))
                        dependencies[identifier].append(calibration['job_id']);allowed[identifier].append('angle_refinement')
                continue
            if recon['state'] not in (('completed','partial') if kind=='geometry_refinement' else ('completed',)):raise InputError(row['name']+': complete reconstruction first.')
            work=service.path(recon['work'])
            if not layout_owned(root,work):raise InputError('Invalid reconstruction path.')
            config=service.path(str(storage_path(work, 'config.json')))
            source=json.loads(config.read_text())
            if angular and source['inputs']['binning']!=8:
                raise InputError('Angle refinement requires an accepted bin8 reconstruction.')
            if not angular and source['inputs']['binning'] % options['output_binning']:
                raise InputError('Output binning must divide the preprocessing binning.')
            coarse=service.path(str(storage_path(storage_path(work, 'output'), 'alignment')))
            service.path(str(storage_path(coarse, 'final_state.npz')))
            import numpy as np
            from .refinement_worker import load_state
            expected_angles=np.loadtxt(service.path(source['inputs']['rawtlt']),ndmin=1)
            try:load_state(storage_path(coarse, 'final_state.npz'),expected_angles,allow_partial=kind=='geometry_refinement')
            except (ValueError,KeyError) as error:raise InputError('Invalid accepted state: '+str(error)) from error
            prep=service.path(str(preprocessing_work(Path(source['inputs']['stack']))))
            if not layout_owned(root,prep):raise InputError('Invalid preprocessing path.')
            service.path(str(storage_path(prep, 'config.json')))
            service.path(str(storage_path(prep, 'input.mdoc')))
            service.path(source['inputs']['frozen_shifts'])
            
            records=read_table(source['inputs']['frozen_shifts'])
            if len(records)!=len(expected_angles) or any('mdoc_index' not in r for r in records):
                raise InputError('Native replay requires the original frozen-shift CSV with MDOC indices.')
            plans[identifier]=dict(coarse_run=str(coarse),preprocessing_work=str(prep),
                inputs=source['inputs'],orientation=source['orientation'],main_revision=source['main_revision'])
            if not angular and row.get('angle_refinement'):
                calibration=refresh(service,{'preprocessing':row['angle_refinement']})
                if calibration['state'] in ('queued','submitting','unknown'):
                    raise InputError('Angle refinement must be successfully submitted before final-resolution refinement.')
                if calibration['state'] in ('completed','submitted','running'):
                    angle_work=service.path(calibration['work'])
                    if not layout_owned(root,angle_work):raise InputError('Invalid angle job path.')
                    previous=json.loads((storage_path(angle_work, 'config.json')).read_text())
                    for field in ('coarse_run','inputs','orientation','main_revision'):
                        if previous[field]!=plans[identifier][field]:
                            raise InputError('Angle refinement belongs to a different reconstruction/configuration; refine angles again.')
                    pending=calibration['state']!='completed'
                    if pending:
                        if not calibration.get('job_id') or calibration.get('host','')!=slurm.get('submission_host',''):raise InputError('Angle dependency needs a Slurm ID on the same host.')
                        dependencies[identifier]=[calibration['job_id']];allowed[identifier]=['angle_refinement']
                    geometry=service.path(str(storage_path(storage_path(angle_work, 'output'), 'angle_state.npz')),exists=not pending)
                    plans[identifier]['angle_geometry']=str(geometry)
        errors=[];data[kind+'_options']=options
        for identifier in series_ids:
            row=rows[identifier];reuse={}
            if kind=='geometry_refinement':
                from .final_denoising import prepare_model_reuse
                reuse=prepare_model_reuse(root,row,options,plans[identifier]['inputs'],options['final_binning'])
            work=prepare_work(service,root,row,kind,queue_only,allowed_dependencies=allowed.get(identifier,()))
            config=dict(**plans[identifier],options=options,output=str(storage_path(work, 'output')),source_kind=source_kind)
            config.update(reuse)
            if kind=='geometry_refinement':config['final_binning']=options['final_binning']
            if kind=='geometry_refinement' and identifier not in dependencies:
                sources=[storage_path(Path(config['coarse_run']), 'final_state.npz'),*[Path(config['inputs'][name]) for name in ('stack','rawtlt','frozen_shifts')],storage_path(Path(config['preprocessing_work']), 'config.json'),storage_path(Path(config['preprocessing_work']), 'input.mdoc')]
                config['snapshot']=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sources]
            config.setdefault('snapshot',[]).extend(config.pop('tinycare_model_snapshot',[]))
            path=storage_path(work, 'config.json');path.write_text(json.dumps(config,indent=2))
            resources=dict(cpus=data.get('preprocessing_options',{}).get('cpus',4),gpus=1)
            module='greco.web.geometry_worker' if kind=='geometry_refinement' else 'greco.web.angle_worker' if angular else 'greco.web.refinement_worker'
            (storage_path(work, 'job.sbatch')).write_text(render_script(slurm['template'],work,path,resources,module))
            job=dict(state='submitted',work=str(work),host=slurm.get('submission_host',''),options=options,source_kind=source_kind)
            if only_missing and not queue_only:
                array_entries.append((row,job,dependencies.get(identifier,())));continue
            try:
                dispatch(job,queue_only,command_result,dependencies=dependencies.get(identifier,()))
            except InputError as error:
                job.update(state='failed',error=str(error));errors.append(row['name']+': '+str(error))
            store_job(row,kind,job,queue_only)
            service._write(root,data)
        if array_entries:
            from .slurm_arrays import submit_arrays
            errors+=submit_arrays(service,root,data,kind,array_entries,command_result)
        result=service._view(project_id);result['submission_errors']=errors;return result


def status(service,project_id,series_id,*,kind='refinement'):
    if kind not in ('refinement','angle_refinement','geometry_refinement'):raise InputError('Invalid refinement task.')
    with service.lock:
        root=service._root(project_id);data=service._read(root)
        row=next((r for r in data['series'] if r['id']==series_id),None)
        if row is None:raise InputError('Unknown tilt series.')
        job=refresh(service,{'preprocessing':row.get(kind,{})})
        if row.get(kind) and row[kind]!=job:
            row[kind]=job;service._write(root,data)
        return job
