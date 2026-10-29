"""Finer-grid refinement of the existing landmark graph, without reselection."""
from copy import deepcopy
from pathlib import Path
import json,time
import numpy as np
from ..project_layout import join


def refinement_mode(options):
    if options['refine_tilt_angles']:
        return 'tilt' if options['refine_per_view_axis'] else 'tilt_common' if options['refine_common_axis'] else 'tilt_only'
    return 'roll' if options['refine_per_view_axis'] else 'common' if options['refine_common_axis'] else 'fixed'


def run(data,work,coarse_source):
    import cupy as cp,mrcfile
    from ..final_reconstruction import final_stack
    from ..alignment_geometry import resolve
    from ..angular_joint import rescale_geometry
    from ..preprocessing_star import read_table
    from .advanced_refinement import AdvancedRefiner
    from ..gcc import materialize_raw_stack_gpu
    from ..tomogram_thickness import reconstruct_stage
    from ..tomography import gpu_reconstruct_wbp
    from .progressive_alignment import _write_landmark_projections, _write_mrcs
    work=Path(work);options=data['advanced_options'];binning=options['fine_scale_binning'];rounds=options['fine_scale_rounds']
    folder=work/'registration/fine_scale';folder.mkdir(parents=True,exist_ok=False)
    checkpoint=join(work,'alignment/final_state.npz')
    with np.load(checkpoint,allow_pickle=False) as saved:coarse={key:np.array(saved[key]) for key in saved.files}
    with mrcfile.open(coarse_source,header_only=True) as handle:old_size=int(handle.header.nx);voxel=float(handle.voxel_size.x)
    active=np.asarray(coarse['active_indices'],int);identities=np.asarray(coarse['angles_deg'],float)
    angles,axes=resolve(coarse,identities,data['orientation']['rotation_deg'])
    np.savez_compressed(folder/'input_state_bin8.npz',**coarse)
    fine_data=deepcopy(data);fine_data['inputs']['binning']=binning
    print(f'Fine-scale refinement: preparing bin {binning}, {rounds} passes, same landmark graph.',flush=True)
    alignment=work/'alignment'
    print('Fine-scale refinement: re-estimating CTF with the bin8 geometry.',flush=True)
    original,_=final_stack(fine_data,alignment,folder,return_raw=True,alignment_input=True)
    (folder/'input_ctf.json').write_text(join(alignment,'ctf_refinement.json').read_text())
    frozen_rows=read_table(data['inputs']['frozen_shifts'])
    frozen=np.array([[float(row['dy_raw']),float(row['dx_raw'])] for row in frozen_rows],float)
    if data['orientation'].get('flip_y',False):original=np.ascontiguousarray(original[:,::-1]);frozen[:,0]*=-1
    new_size=original.shape[-1];ratio=8/binning
    if new_size!=old_size*ratio:raise ValueError('Fine-scale grid differs from the native centred binning grid.')
    positions,shifts=rescale_geometry(coarse['positions_zyx'],coarse['cumulative_shifts_yx'][active],old_size,new_size,axes[active],bool(data['orientation'].get('flip_y',False)))
    tracks=dict(positions_zyx=positions,valid_views=coarse['valid_views'].copy(),classes=coarse['classes'].copy())
    current_angles=angles[active];current_axes=axes[active]
    patch_radius=16*int(ratio);mode=refinement_mode(options)
    refiner=AdvancedRefiner(mode=mode,max_passes=options['angular_max_passes'],rescue_enabled=True,patch_radius=patch_radius)
    selected_raw=cp.asarray(original[active]);del original
    history=[];started=time.monotonic()
    for index in range(rounds):
        candidate,manifest=refiner(selected_raw,current_angles,tracks,shifts,source='raw',frozen_shifts_yx=frozen[active]/binning,axis_deg=current_axes,round_index=index,round_count=rounds)
        np.savez_compressed(folder/f'pass_{index+1:03d}.npz',**candidate)
        # Manifests contain numpy arrays; keep a JSON-readable diagnostic copy.
        (folder/f'pass_{index+1:03d}.json').write_text(json.dumps(manifest,default=lambda x:x.tolist() if hasattr(x,'tolist') else x,indent=2))
        if manifest['stop_reason']!='converged':raise RuntimeError(f'Fine-scale pass {index+1} did not converge: '+manifest['stop_reason'])
        tracks={key:candidate[key] for key in tracks}
        shifts=candidate['cumulative_shifts_yx'];current_angles=candidate['angles_deg']
        current_axes=np.broadcast_to(candidate['axis_rotation_deg'],current_angles.shape).copy()
        history.append(dict(pass_number=index+1,iterations=manifest['passes'],runtime_s=manifest['runtime_s']))
        print(f'Fine-scale pass {index+1}/{rounds}: converged at bin {binning}.',flush=True)
    fine=dict(candidate,binning=np.int64(binning),active_indices=active,source_angles_deg=identities)
    np.savez_compressed(folder/'final_state.npz',**fine)
    positions8,shifts8=rescale_geometry(tracks['positions_zyx'],shifts,new_size,old_size,current_axes,bool(data['orientation'].get('flip_y',False)))
    coarse['positions_zyx']=positions8
    coarse['valid_views']=tracks['valid_views'];coarse['classes']=tracks['classes']
    coarse['cumulative_shifts_yx'][active]=shifts8
    full_angles=angles.copy();full_angles[active]=current_angles
    full_axes=axes.copy();full_axes[active]=current_axes
    coarse.update(refined_angles_deg=full_angles,axis_rotation_deg=full_axes,geometry_converged=np.bool_(True),fine_scale_binning=np.int64(binning),fine_scale_rounds=np.int64(rounds))
    temporary=checkpoint.with_name('.fine-final-state.npz');np.savez_compressed(temporary,**coarse);temporary.replace(checkpoint)
    summary_path=join(work,'alignment/summary.json');summary=json.loads(summary_path.read_text())
    summary.update(fine_scale_refinement=dict(binning=binning,passes=history,patch_size=2*patch_radius+1,mode=mode,runtime_s=time.monotonic()-started),geometry_converged=True,active_angles_deg=current_angles.tolist(),axis_rotation_deg=full_axes.tolist(),final_axis_rotation_deg=float(current_axes.mean()))
    summary_path.write_text(json.dumps(summary,indent=2))
    # Diagnostic landmarks must follow the final fine-scale geometry too.
    with mrcfile.mmap(coarse_source) as handle:raw8=cp.asarray(handle.data)
    aligned=materialize_raw_stack_gpu(raw8,active,coarse['cumulative_shifts_yx'][active],frozen[active]/8,current_axes).get()
    _write_mrcs(join(work,'alignment/aligned_bin8.mrcs'),aligned,voxel)
    volume=reconstruct_stage(gpu_reconstruct_wbp,aligned,current_angles,data['tomogram_thickness_percent'])
    _write_landmark_projections(folder,volume,dict(positions_zyx=coarse['positions_zyx'],valid_views=coarse['valid_views']),current_angles,voxel)
    (folder/'output_geometry.json').write_text(json.dumps(dict(active_indices=active.tolist(),angles_deg=current_angles.tolist(),axis_rotation_deg=current_axes.tolist())))
    from .final_ctf import reestimate
    print('Fine-scale refinement: re-estimating CTF with the final geometry.',flush=True)
    reestimate(data,alignment)
    return summary['fine_scale_refinement']
