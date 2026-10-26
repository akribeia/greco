from greco.sta_layout import read_metadata,write_metadata,NUMERICS
"""Optional post-convergence alternating pose / independent-view MAP refinement."""
from greco.project_layout import join as storage_path, ensure_output
from pathlib import Path
import shutil
import time
import math
import json
import hashlib
import numpy as np


def fsc_values(result):
    try:
        rows=read_metadata(storage_path(Path(result['particles']).parent, 'fsc.json'))['curve']
        values=np.array([r['fsc'] for r in rows],float)
        if len(values)<=1 or not np.isfinite(values).all():return None
        # Use the supported band, including its first FSC crossing, just as
        # AutoRefine does. Unresolved Nyquist noise is not alignment progress.
        crossing=next((i for i in range(1,len(values)) if values[i]<.143),len(values)-1)
        values=values[:crossing+1]
        return values if len(values)>1 and np.isfinite(values).all() else None
    except (OSError,ValueError,KeyError,TypeError):return None


def refine_alternating(config,pose_refine):
    from .observation_refinement import run
    from ..web.particle_worker import save_json,check_snapshot
    begin=time.perf_counter();output=Path(config['output']);ensure_output(output)
    options=dict(config['options']);options.pop('refine_per_tilt_shifts',None)
    ctf_enabled=bool(options.pop('refine_ctf',False))
    motion_enabled=bool(config['options'].get('refine_per_tilt_shifts',False))
    if not options.get('auto_refine',True):raise ValueError('Per-tilt shifts require automatic pose convergence.')
    history=[];stable=0
    save_json(storage_path(output, 'progress.json'),dict(phase='Initial pose convergence',processed=0,total=0))
    current=pose_refine(dict(config,output=str(storage_path(output, 'alternating/initial')),options=options))
    noise_path=storage_path(Path(current['particles']).parent, 'noise_variance.npy')
    shared=dict(frozen_noise_path=str(noise_path)) if noise_path.exists() and options.get('noise_model')=='half_residual' else {}
    ctf_history=[]
    ctf_shared={}
    def ctf_pass(label,reference):
        resolution=reference.get('diagnostic_resolution_A')
        if not isinstance(resolution,(int,float)) or not math.isfinite(resolution) or resolution<=0:
            raise ValueError('A finite FSC resolution is required for CTF refinement.')
        save_json(storage_path(output, 'progress.json'),dict(phase=f'CTF refinement {label}',processed=0,total=reference['used_particles']))
        correction=run(dict(config,**ctf_shared,action='ctf_refinement',output=str(storage_path(output, f'alternating/ctf_{label}')),
            extraction_manifest=reference['particles'],half_maps=[reference['unfiltered_volumes'][h] for h in ('half1','half2')],
            options=dict(resolution_A=max(2*config['pixel_size_A'],resolution),maximum_defocus_A=500.,defocus_step_A=25.)))
        ctf_history.append(dict(phase=label,diagnostics=correction['diagnostics']))
        ctf_noise=storage_path(output, f'alternating/ctf_{label}/noise_variance.npy')
        if ctf_noise.exists():ctf_shared['frozen_noise_path']=str(ctf_noise)
        # Rebuild final maps and reconverge poses against the changed transfer
        # function. Do not publish old maps with a new CTF STAR.
        return pose_refine(dict(config,**shared,output=str(storage_path(output, f'alternating/ctf_{label}_poses')),
            extraction_manifest=correction['particles'],options=dict(options,resolution_A=max(2*config['pixel_size_A'],resolution)),
            postconvergence_continuation=True,continuation_half_maps=[reference['volumes'][h] for h in ('half1','half2')]))
    if ctf_enabled and current['convergence_state']=='converged':current=ctf_pass('before',current)
    curves=[fsc_values(current)]
    # Safety cap, not a claim of convergence. All blocks remain independently
    # inspectable and immutable; no intermediate completed result at job root.
    initial_converged=current['convergence_state']=='converged'
    for cycle in range(1,31) if motion_enabled and initial_converged else ():
        resolution=current.get('diagnostic_resolution_A')
        if not isinstance(resolution,(int,float)) or not math.isfinite(resolution) or resolution<=0:
            raise ValueError('A finite reference FSC resolution is required for per-tilt shifts.')
        save_json(storage_path(output, 'progress.json'),dict(phase=f'Per-tilt shift cycle {cycle}',processed=0,total=current['used_particles']))
        folder=storage_path(output, f'alternating/cycle_{cycle:02d}')
        motion_shared=shared if options.get('observation_model','ctf_premultiplied')=='ctf_premultiplied' else {}
        motion=run(dict(config,**motion_shared,action='polishing',output=str(storage_path(folder, 'shifts')),extraction_manifest=current['particles'],
            half_maps=[current['unfiltered_volumes'][h] for h in ('half1','half2')],
            options=dict(resolution_A=max(2*config['pixel_size_A'],resolution),maximum_shift_A=2*config['pixel_size_A'],
                velocity_sigma_A=config['pixel_size_A']/2,optimizer_steps=30,motion_model='independent_ridge')))
        save_json(storage_path(output, 'progress.json'),dict(phase=f'Pose convergence after shift cycle {cycle}',processed=0,total=current['used_particles']))
        continuation=dict(options,iterations=1,resolution_A=max(2*config['pixel_size_A'],resolution))
        updated=pose_refine(dict(config,**shared,output=str(storage_path(folder, 'poses')),extraction_manifest=motion['particles'],
            options=continuation,postconvergence_continuation=True,
            continuation_half_maps=[current['volumes'][h] for h in ('half1','half2')]))
        after=updated.get('diagnostic_resolution_A')
        curves=(curves+[fsc_values(updated)])[-4:]
        window_change=None
        if all(v is not None for v in curves):
            common=min(map(len,curves));window_change=float(np.max(np.ptp(np.stack([v[:common] for v in curves]),axis=0)))
        d=motion['diagnostics'];pixel=config['pixel_size_A']
        metrics=read_metadata(storage_path(Path(updated['particles']).parent, 'iterations.json'))[-1] if (storage_path(Path(updated['particles']).parent, 'iterations.json')).exists() else {}
        control=metrics.get('auto_refine',{})
        unreliable=sum(metrics.get(key,0) for key in ('support_rejected_particles','rejected_particles',
            'local_optimizer_failed_particles','neighborhood_bound_particles'))
        pose_quiet=(control.get('movement_pixels',float('inf'))<control.get('tolerance_pixels',0.)
            and unreliable/max(updated['used_particles'],1)<.05)
        # A one-pass block intentionally cannot reach the inner three-pass
        # stop. Joint convergence is measured across alternating cycles.
        quiet=((pose_quiet or updated['convergence_state']=='converged') and after is not None and abs(after-resolution)<.02
            and window_change is not None and window_change<=.005
            and d.get('rms_shift_A',float('inf'))<.05*pixel
            and d.get('p95_shift_A',float('inf'))<.1*pixel
            and d.get('shift_outlier_fraction',1.)<.05)
        stable=stable+1 if quiet else 0
        history.append(dict(cycle=cycle,before_resolution_A=resolution,after_resolution_A=after,
            motion=motion['diagnostics'],stable_cycles=stable,pose_convergence=updated['convergence_state']))
        history[-1]['fsc_window_change']=window_change
        history[-1]['fsc_window_band']='common supported band through first 0.143 crossing'
        save_json(storage_path(output, 'alternating_iterations.json'),history);current=updated
        if stable>=3:break
    joint_state='converged' if stable>=3 else 'alternating_limit' if history else current['convergence_state']
    if ctf_enabled and joint_state=='converged':
        current=ctf_pass('after',current)
        if current['convergence_state']!='converged':joint_state=current['convergence_state']
    # Publish only the final maps under the usual viewer paths. STAR keeps its
    # absolute immutable source chain, including cumulative view corrections.
    for field in ('volumes','unfiltered_volumes'):
        for name,source in current.get(field,{}).items():
            target=storage_path(output, Path(source).name);temporary=target.with_suffix('.tmp')
            shutil.copyfile(source,temporary);temporary.replace(target);current[field][name]=str(target)
    for name in ('particles.star','fsc.json','iterations.json','signal_prior.json','noise_model.json','noise_variance.npy','protocol.json'):
        source=storage_path(Path(current['particles']).parent, name)
        if name in NUMERICS and source.parent.name=='provenance':
            bundle=source.parent/'numerics.json'
            if bundle.is_file() and source.stem in json.loads(bundle.read_text()):write_metadata(storage_path(output,name),read_metadata(source))
        elif source.exists():
            target=storage_path(output,name);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.tmp');shutil.copyfile(source,temporary);temporary.replace(target)
    current['particles']=str(storage_path(output, 'particles.star'))
    protocol_path=storage_path(output, 'protocol.json')
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    protocol.update(protocol='sta-pose-ctf-motion-alternation-v2',options=config['options'],
        extraction_manifest=config.get('extraction_manifest'),single_pose_pass_per_motion_cycle=True,
        ctf_refinement_passes=ctf_history,frozen_noise_by_observation_domain=True)
    protocol.setdefault('code_sha256',{}).update({name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('alternating_refinement.py','observation_refinement.py','polishing.py','ctf_refinement.py','frozen_noise.py')})
    save_json(protocol_path,protocol)
    current.update(options=config['options'],per_tilt_shift_cycles=history,
        convergence_state=joint_state,ctf_refinement_passes=ctf_history,
        elapsed_seconds=time.perf_counter()-begin)
    check_snapshot(config['snapshot']);save_json(storage_path(output, 'web_result.json'),current)
    save_json(storage_path(output, 'progress.json'),dict(phase='Completed',processed=current['used_particles'],total=current['used_particles']))
    return current
