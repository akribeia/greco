# Greco optimizer — implementation of the GRECO algorithm
# GRECO: Geometric Registration of Electron Cryotomography Observations
# Copyright (C) 2026 Carlos Fernández Rodríguez
#
# This file is part of Greco.
#
# Greco is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3 only.
#
# Greco is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Greco. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Production Advanced-mode orchestration on the configured bin8 raw-image grid."""
import argparse,json,time
from pathlib import Path
import numpy as np
from . import progressive_alignment as progressive
from .bootstrap import stage_column_directions
from .column_bootstrap import detect_column_landmarks
from .advanced_options import checked_resolved
from .advanced_refinement import AdvancedRefiner
from ..gcc import materialize_raw_stack_gpu
from ..tomography import gpu_reconstruct_wbp
from ..tomogram_thickness import reconstruct_stage


def run(original,angles,output,*,options,frozen_shifts_yx,axis_deg,voxel,
        thickness=50.,entry_sigma=8.,rescue=True,smoothness=0.,stage_failure_limit=None):
    import cupy as cp
    options=checked_resolved(options)
    if not options['advanced_mode']:raise ValueError('Advanced-mode runner requires Advanced mode')
    if smoothness and any(options[k] for k in ('refine_common_axis','refine_per_view_axis','refine_tilt_angles')):raise ValueError('Smoothness must be zero for angular refinement')
    output=Path(output)
    if (output/'config.json').exists() or (output/'final_state.npz').exists():
        raise ValueError('Advanced alignment output already contains a run.')
    source_angles=np.asarray(angles,float);order=np.argsort(source_angles,kind='stable')
    from .stage_continuation import StageContinuation
    if stage_failure_limit is None:stage_failure_limit=options['consecutive_stage_failure_limit']
    if type(stage_failure_limit) is not int or stage_failure_limit<0:raise ValueError('Invalid stage failure limit')
    stage_acceptor=StageContinuation(stage_failure_limit) if stage_failure_limit else None
    started=time.monotonic();rescue_state={"adaptive":True}
    if smoothness:
        from .guidance import guided_gcc
        from ..gcc import run_gcc
        guided=guided_gcc(run_gcc,smoothness,rescue,rescue_state=rescue_state)
        def runner(*args,**kwargs):
            return guided(*args,**kwargs,max_passes=options['progressive_max_passes'],stop_shift=1.,stop_shift_p90=2.5,
                          measurement_rms_ratio=1.02,normal_radius_only=False)
    else:
        runner=AdvancedRefiner(mode='common' if options['refine_common_axis'] else 'fixed',max_passes=options['progressive_max_passes'],rescue_enabled=rescue,rescue_state=rescue_state)
    trajectory=progressive.run(original,source_angles,output,raw_source=True,
        frozen_shifts_yx=frozen_shifts_yx,axis_deg=axis_deg,voxel=voxel,binning=8,
        extra_rounds=5,exact_extra_rounds=True,landmark_support='accepted',cc_method='gradient',
        solver_backend='gpu_schur',resample='exact',entry_search_sigma_px=entry_sigma,
        tomogram_thickness_percent=thickness,gcc_runner=runner,stage_acceptor=stage_acceptor)
    summary=json.loads((output/'summary.json').read_text())
    if stage_acceptor is not None and stage_acceptor.stopped:
        raise RuntimeError(f'Advanced continuation stopped after {stage_acceptor.failures} consecutive nonconverged stages: '+str(summary['last_gcc_stop']))
    rounds=[r for r in trajectory if r['stage']=='extra_round']
    if len(rounds)!=5 or not all(r['accepted'] for r in rounds):
        raise RuntimeError(f"Advanced common phase stopped with {summary['last_gcc_stop']}; "
            f"accepted {len(summary['active_indices'])}/{len(source_angles)} tilts, "
            f"completed {sum(bool(r['accepted']) for r in rounds)}/5 accepted common rounds")
    with np.load(output/'final_state.npz',allow_pickle=False) as d:state={k:np.array(d[k]) for k in d.files}
    active=np.asarray(state['active_indices'],int);local=np.arange(len(active))
    raw=np.asarray(original,np.float32)[order];frozen=np.asarray(frozen_shifts_yx,float)[order]/8
    raw_dev=cp.asarray(raw);current_angles=state['angles_deg'][active].copy()
    shifts=state['cumulative_shifts_yx'][active].copy()
    axes=np.broadcast_to(state.get('axis_rotation_deg',axis_deg),current_angles.shape).copy()
    phases=[];outer=5
    tilt_mode='tilt' if options['refine_per_view_axis'] else 'tilt_only'
    for mode,count in [('roll',options['roll_rounds']),(tilt_mode,options['inclination_rounds'])]:
        if not count:continue
        refine=AdvancedRefiner(mode=mode,max_passes=options['angular_max_passes'],rescue_enabled=rescue,rescue_state=rescue_state)
        records=[]
        for index in range(count):
            directory=output/f'extra_round_{outer:02d}';directory.mkdir();stamp=time.monotonic()
            stack=materialize_raw_stack_gpu(raw_dev,active,shifts,frozen[active],axes).get()
            volume=reconstruct_stage(gpu_reconstruct_wbp,stack,current_angles,thickness)
            landmarks,selection=detect_column_landmarks(stack,current_angles,volume,
                stage_column_directions(current_angles,local,index),selector='bright_structure',
                column_cell=32,column_width=30,cumulative_shifts_yx=shifts,
                frozen_shifts_yx=frozen[active],axis_deg=axes,thickness_percent=thickness)
            np.savez_compressed(directory/'selected_landmarks.npz',**landmarks)
            progressive._write_json(directory/'selection.json',selection)
            progressive._write_landmark_projections(directory,volume,landmarks,current_angles,voxel)
            candidate,manifest=refine(raw[active],current_angles,landmarks,shifts,
                source='raw',frozen_shifts_yx=frozen[active],axis_deg=axes,
                round_index=index,round_count=count)
            np.savez_compressed(directory/'final_state.npz',**candidate)
            progressive._write_json(directory/'gcc_manifest.json',manifest)
            decision=stage_acceptor.assess(manifest,candidate) if stage_acceptor else {}
            accepted=decision['accepted'] if decision else manifest['stop_reason']=='converged'
            record=dict(stage='extra_round',round=outer,phase=mode,phase_round=index,
                active_count=len(active),accepted=accepted,selected_count=selection['selected_count'],
                gcc_stop=manifest['stop_reason'],gcc_passes=manifest['passes'],
                axis_rotation_deg=float(np.mean(candidate['axis_rotation_deg'])),runtime_s=time.monotonic()-stamp)
            record.update(decision)
            progressive._write_json(directory/'metrics.json',record)
            progressive._write_json(directory/'acceptance.json',dict(accepted=accepted,reason=manifest['stop_reason'],**{k:v for k,v in decision.items() if k!='accepted'}))
            progressive._write_json(directory/'output_geometry.json',dict(active_indices=order[active].tolist(),
                angles_deg=candidate['angles_deg'].tolist(),axis_rotation_deg=candidate['axis_rotation_deg'].tolist()))
            trajectory.append(record);progressive._write_json(output/'trajectory.json',trajectory)
            if not accepted:raise RuntimeError(f'Advanced {mode} round {index+1} did not converge: '+manifest['stop_reason'])
            state.update(positions_zyx=candidate['positions_zyx'],valid_views=candidate['valid_views'],classes=candidate['classes'])
            shifts=candidate['cumulative_shifts_yx'].copy();axes=candidate['axis_rotation_deg'].copy();current_angles=candidate['angles_deg'].copy()
            records.append(record);outer+=1
        phases.append(dict(mode=mode,rounds=records))
    # The displayed stack must use the last accepted per-view geometry.
    final_stack=materialize_raw_stack_gpu(raw_dev,active,shifts,frozen[active],axes).get()
    # The checkpoint below stores accepted views in source order. Keep the
    # preview planes in that same order, including unsorted input angles.
    progressive._write_mrcs(output/'aligned_bin8.mrcs',final_stack[np.argsort(order[active])],float(voxel))
    # Export a source-indexed checkpoint. Acquisition angles remain identities;
    # the refined angles and raster rotations are separate numerical geometry.
    full_shifts=state['cumulative_shifts_yx'].copy();full_shifts[active]=shifts
    full_angles=np.asarray(source_angles[order]).copy();full_angles[active]=current_angles
    full_axes=np.full(len(source_angles),float(summary.get('final_axis_rotation_deg',axis_deg)));full_axes[active]=axes
    inverse=np.argsort(order)
    source_active=order[active];column_order=np.argsort(source_active);source_active=source_active[column_order]
    state['valid_views']=state['valid_views'][:,column_order]
    state.update(cumulative_shifts_yx=full_shifts[inverse],angles_deg=source_angles.copy(),
        refined_angles_deg=full_angles[inverse],axis_rotation_deg=full_axes[inverse],
        active_indices=source_active,source_order=order,binning=np.int64(8),advanced_protocol=np.array(options['protocol']))
    state['geometry_converged']=np.bool_(trajectory[-1].get('gcc_stop')=='converged')
    np.savez_compressed(output/'final_state.npz',**state)
    # Refresh readable shifts after the final phase, in acquisition order.
    with (output/'shifts.csv').open('w') as stream:
        stream.write('stack_index,angle_deg,dy_binned,dx_binned,dy_fullres,dx_fullres\n')
        for i,(angle,shift) in enumerate(zip(source_angles,state['cumulative_shifts_yx'])):
            stream.write(f'{i},{angle:.6f},{shift[0]:.6f},{shift[1]:.6f},{shift[0]*8:.6f},{shift[1]*8:.6f}\n')
    # Stage output geometries refer to the same source identities as final state.
    for directory in sorted(output.glob('stage_*'))+sorted(output.glob('extra_round_*'))[:5]:
        path=directory/'output_geometry.json'
        if path.exists():
            d=json.loads(path.read_text());d['active_indices']=order[np.asarray(d['active_indices'],int)].tolist()
            progressive._write_json(path,d)
    summary.update(geometry_converged=(trajectory[-1].get('gcc_stop')=='converged'),
        consecutive_stage_failures=stage_acceptor.failures if stage_acceptor else 0,
        active_indices=source_active.tolist(),active_angles_deg=current_angles[column_order].tolist(),
        input_angles_deg=source_angles.tolist(),source_order=order.tolist(),
        excluded_signal_indices=order[np.asarray(summary['excluded_signal_indices'],int)].tolist(),
        advanced_options=options,extra_rounds_run=outer,phases=phases,
        final_axis_rotation_deg=float(axes.mean()),axis_rotation_deg=full_axes[inverse].tolist(),
        protocol=options['protocol'],total_runtime_s=time.monotonic()-started)
    progressive._write_json(output/'summary.json',summary)
    return summary


def main():
    import mrcfile
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    data=json.loads(parser.parse_args().config.read_text())
    with mrcfile.mmap(data['stack']) as m:raw=np.array(m.data,np.float32);voxel=float(m.voxel_size.x)
    angles=np.loadtxt(data['angles'],ndmin=1)
    rows=np.loadtxt(data['frozen_shifts'],delimiter=',',skiprows=1,ndmin=2)
    if len(rows)!=len(angles) or not np.allclose(rows[:,1],angles,atol=1e-4):raise ValueError('Frozen shift identities differ')
    frozen=rows[:,2:]
    run(raw,angles,data['output'],options=data['options'],frozen_shifts_yx=frozen,
        axis_deg=data['axis_deg'],voxel=voxel,thickness=data['thickness'],
        entry_sigma=data['entry_sigma'],rescue=data['rescue'],smoothness=data['smoothness'])

if __name__=='__main__':main()
