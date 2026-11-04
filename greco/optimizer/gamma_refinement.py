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

"""Experimental final per-view refinement; fixed XYZ and total camera-change prior."""
from copy import deepcopy
from pathlib import Path
import json,time
import numpy as np
from ..project_layout import join
from ..alignment_geometry import resolve
from .gamma_geometry import cameras,project,parameter_scales,fit_view


def run(data,work):
    import cupy as cp,mrcfile
    from ..final_reconstruction import final_stack
    from ..preprocessing_star import read_table
    from .gamma_imaging import Engine
    work=Path(work);options=data['advanced_options']
    folder=work/'registration/gamma';folder.mkdir(parents=True,exist_ok=False)
    checkpoint=join(work,'alignment/final_state.npz')
    with np.load(checkpoint,allow_pickle=False) as archive:state={k:np.array(archive[k]) for k in archive.files}
    identities=np.asarray(state['angles_deg'],float);active=np.asarray(state['active_indices'],int)
    alpha,axes=resolve(state,identities,data['orientation']['rotation_deg'])
    source_alpha=alpha[active].copy();source_axes=axes[active].copy();source_shifts=state['cumulative_shifts_yx'][active].copy()
    np.savez_compressed(folder/'input_state.npz',**state)
    gamma_data=deepcopy(data);gamma_data['inputs']['binning']=8
    print('Gamma refinement (experimental): preparing final-CTF bin8 observations.',flush=True)
    observations=folder/'observations';observations.mkdir()
    raw,_=final_stack(gamma_data,work/'alignment',observations,return_raw=True)
    (folder/'input_ctf.json').write_text(join(work,'alignment/ctf_refinement.json').read_text())
    with mrcfile.open(join(observations,'final_wiener_rawsource.mrcs'),header_only=True) as handle:voxel=float(handle.voxel_size.x)
    n=raw.shape[-1]
    with mrcfile.open(data['inputs']['stack'],header_only=True) as handle:
        if int(handle.header.nx)!=n:raise ValueError('Gamma refinement requires the existing bin8 landmark grid.')
    rows=read_table(data['inputs']['frozen_shifts'])
    frozen=np.asarray([[float(row['dy_raw']),float(row['dx_raw'])] for row in rows])/8
    if data['orientation'].get('flip_y',False):raw=np.ascontiguousarray(raw[:,::-1]);frozen[:,0]*=-1
    fixed=np.asarray(state['positions_zyx'],float)[:,::-1]-(n-1)/2
    valid=np.asarray(state['valid_views'],bool)
    if valid.shape!=(len(fixed),len(active)):raise ValueError('Gamma landmark support does not match accepted tilts.')
    good=np.isfinite(fixed).all(1)&(valid.sum(1)>=4);positions=fixed[good];support=valid[good]
    if not len(positions):raise ValueError('Gamma refinement requires supported landmarks.')
    scales,radius=parameter_scales(positions);reference=cameras(source_alpha)
    source=cp.asarray(raw);del raw
    selected=source[active];current=np.zeros((len(active),5));engine=Engine();history=[]
    started=time.monotonic();order=np.argsort(identities[active],kind='stable');rounds=options['gamma_rounds'];weight=options['gamma_weight']
    for rnd in range(1,rounds+1):
        stamp=time.monotonic();target=cameras(source_alpha+current[:,2],current[:,3],current[:,4])
        votes,measurement=engine.measure(selected,positions,reference,source_shifts,frozen[active],source_axes,support,np.zeros(len(active)),batch=32,target_cameras=target,target_shifts=source_shifts+current[:,:2])
        votes[engine.last_boundary_mask]=np.nan;proposed=current.copy();fractions=np.ones(len(active));skipped=[]
        for tilt in range(len(active)):
            eligible=np.isfinite(votes[:,tilt]).all(1)
            if eligible.sum()<8:skipped.append(int(active[tilt]));continue
            proposed[tilt],_=fit_view(positions[eligible],source_alpha[tilt],current[tilt],votes[eligible,tilt],weight,scales)
            before=project(positions[eligible],target[tilt:tilt+1])[:,0]-current[tilt,:2]
            for trial in range(25):
                u=current[tilt]+fractions[tilt]*(proposed[tilt]-current[tilt])
                after=project(positions[eligible],cameras(source_alpha[tilt]+u[2],u[3],u[4])[None])[:,0]-u[:2]
                if np.linalg.norm(after-before,axis=1).max()<=4:break
                fractions[tilt]*=.5
            else:raise RuntimeError('No safe gamma image-space step.')
        for trial in range(25):
            candidate=current+fractions[:,None]*(proposed-current);bad=np.flatnonzero(np.diff((source_alpha+candidate[:,2])[order])<=0)
            if not len(bad):break
            fractions[np.unique(np.r_[order[bad],order[bad+1]])]*=.5
        else:raise RuntimeError('Gamma refinement would cross tilt angles.')
        nextcam=cameras(source_alpha+candidate[:,2],candidate[:,3],candidate[:,4])
        flow=np.linalg.norm((project(positions,nextcam)-(source_shifts+candidate[:,:2])[None])-(project(positions,target)-(source_shifts+current[:,:2])[None]),axis=2)
        current=candidate;mask=np.isfinite(votes).all(2)
        record=dict(round=rnd,measurement=measurement,skipped_source_indices=skipped,median_image_step=float(np.median(flow[mask])) if mask.any() else 0.,p90_image_step=float(np.quantile(flow[mask],.9)) if mask.any() else 0.,gamma_min=float(current[:,4].min()),gamma_max=float(current[:,4].max()),seconds=time.monotonic()-stamp)
        history.append(record);(folder/'history.json').write_text(json.dumps(history,indent=2))
        np.savez_compressed(folder/f'pass_{rnd:03d}.npz',parameters=current,source_indices=active)
        print(f'Gamma refinement: pass {rnd}/{rounds}, median image step {record["median_image_step"]:.5f} px.',flush=True)
    # Absorb the fitted detector roll into the saved raster orientation without
    # changing raw-detector translations. Gamma remains a full 3D camera term.
    b=np.deg2rad(current[:,3]);sy,sx=(source_shifts+current[:,:2]).T
    state['cumulative_shifts_yx'][active]=np.column_stack((np.cos(b)*sy-np.sin(b)*sx,np.sin(b)*sy+np.cos(b)*sx))
    alpha[active]=source_alpha+current[:,2];axes[active]=source_axes+current[:,3]
    gamma=np.zeros(len(identities));gamma[active]=current[:,4]
    reference_angles=np.asarray(state.get('refined_angles_deg',identities),float).copy()
    state.update(refined_angles_deg=alpha,axis_rotation_deg=axes,axis_inclination_deg=gamma,gamma_reference_angles_deg=reference_angles)
    report=dict(protocol='fixed-xyz-minimum-camera-change-v1',experimental=True,convergence='fixed_round_count',rounds=rounds,weight=weight,scales=scales.tolist(),landmark_radius_px=radius,landmarks_fixed=True,landmark_count=len(positions),runtime_s=time.monotonic()-started,history=history,skipped_source_indices=sorted(set(i for h in history for i in h['skipped_source_indices'])))
    # Generate complete final-geometry diagnostics before publishing the state.
    _diagnostics(folder,source,active,frozen,voxel,state)
    temporary=checkpoint.with_name('.gamma-final-state.npz');np.savez_compressed(temporary,**state);temporary.replace(checkpoint)
    with join(work,'alignment/shifts.csv').open('w') as f:
        f.write('stack_index,angle_deg,dy_binned,dx_binned,dy_fullres,dx_fullres\n')
        for i,(angle,shift) in enumerate(zip(identities,state['cumulative_shifts_yx'])):
            f.write(f'{i},{angle:.6f},{shift[0]:.6f},{shift[1]:.6f},{shift[0]*8:.6f},{shift[1]*8:.6f}\n')
    summary_path=join(work,'alignment/summary.json');summary=json.loads(summary_path.read_text());summary.update(gamma_refinement=report,axis_inclination_deg=gamma.tolist(),active_angles_deg=alpha[active].tolist(),axis_rotation_deg=axes.tolist(),final_axis_rotation_deg=float(axes[active].mean()))
    summary_path.write_text(json.dumps(summary,indent=2));(folder/'result.json').write_text(json.dumps(report,indent=2))
    from .final_ctf import reestimate
    print('Gamma refinement: re-estimating final CTF with full camera geometry.',flush=True)
    reestimate(data,work/'alignment')
    return report


def _diagnostics(folder,source,active,frozen,voxel,state):
    import cupy as cp,mrcfile
    from ..gcc import materialize_raw_stack_gpu
    from ..alignment_geometry import aligned_cameras
    from ..rigid_backprojection import ramp_filter
    from ..landmark_rigid_gpu import Kernels
    from ..selection import _landmark_sphere_mask_gpu
    from .progressive_alignment import _write_mrcs
    n=source.shape[-1];shifts=state['cumulative_shifts_yx'][active];axes=state['axis_rotation_deg'][active]
    aligned=materialize_raw_stack_gpu(source,active,shifts,frozen[active],axes)
    _write_mrcs(join(folder.parent.parent,'alignment/aligned_bin8.mrcs'),cp.asnumpy(aligned),voxel)
    C=aligned_cameras(state,state['angles_deg'],0.)[active];filtered=ramp_filter(aligned,C,cp);del aligned
    kernels=Kernels(cp);volume=cp.empty((n,n,n),cp.float32);weights=np.full(len(active),np.pi/(2*len(active)))
    for first in range(0,n,8):volume[first:first+8]=kernels.global_wbp(filtered,C,np.zeros((len(active),2)),weights,first,min(8,n-first))
    del filtered
    with mrcfile.new_mmap(folder/'landmarks_projection.mrcs',shape=(len(active),n,n),mrc_mode=2,overwrite=False) as m:
        m.voxel_size=voxel;m.set_image_stack()
        for t in range(len(active)):
            points=np.asarray(state['positions_zyx'])[state['valid_views'][:,t]]
            mask=_landmark_sphere_mask_gpu(volume,points,32)
            if mask is None:raise ValueError('Gamma landmarks are outside the reconstruction grid.')
            m.data[t]=cp.asnumpy(kernels.project((volume*mask)[None],C[t])[0])
        m.update_header_stats()
    (folder/'output_geometry.json').write_text(json.dumps(dict(active_indices=active.tolist(),angles_deg=state['refined_angles_deg'][active].tolist(),axis_rotation_deg=axes.tolist(),axis_inclination_deg=state['axis_inclination_deg'][active].tolist()),indent=2))
