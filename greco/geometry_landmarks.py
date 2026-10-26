"""Fresh bin8 landmark basis from an accepted manual-correction checkpoint."""
import json
from pathlib import Path
import numpy as np
from .project_layout import join as storage_path,job_work
from .preprocessing_star import read_table


def seed_after_manual(data,out):
    import cupy as cp,mrcfile
    from .web.refinement_worker import load_state,clean_json
    from .landmark_rigid import initial_cameras
    from .landmark_rigid_gpu import Kernels,ramp_filter,angle_weights
    from .selection import column_score_volume_gpu
    from .manual_tilt import neighbor_column_candidates
    from .geometry_support import accepted_view_order
    from .tomogram_thickness import stage_bounds
    angles=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
    source_basis=Path(data.get('landmark_basis') or storage_path(Path(data['coarse_run']),'final_state.npz'))
    positions,shifts,_,_,_,binning=load_state(source_basis,angles,allow_partial=True)
    if binning!=8:raise ValueError('Fresh landmarks require a bin8 source basis.')
    parent=Path(data['resume_geometry'])
    with np.load(parent,allow_pickle=False) as archive:saved={k:np.array(archive[k]) for k in archive.files}
    active=np.asarray(saved.get('active_indices',np.arange(len(angles))),int)
    rows=read_table(data['inputs']['frozen_shifts']);mdoc=np.array([int(r['mdoc_index']) for r in rows])
    order,fit_order,_=accepted_view_order(mdoc,active)
    axis=float(data['orientation']['rotation_deg']);flip=bool(data['orientation'].get('flip_y',False))
    nominal=initial_cameras(angles,axis)
    if (not np.array_equal(saved['angles_deg'],angles) or not np.array_equal(saved['source_positions_zyx'],positions)
        or not np.array_equal(saved['acquisition_order'],order) or not np.allclose(saved['original_cameras_xyz'],nominal)
        or bool(saved['raster_flip_y'])!=flip):raise ValueError('Manual checkpoint does not match the source landmark basis.')
    with mrcfile.mmap(data['inputs']['stack']) as handle:
        pixel=float(handle.voxel_size.x);n=handle.data.shape[-1]
        if handle.data.shape!=(len(angles),n,n):raise ValueError('Fresh landmarks require the full square bin8 stack.')
        images=np.array(handle.data[active],np.float32)
    if (int(saved['binning'])!=8 or int(saved['fit_size'])!=n or not np.isclose(float(saved['pixel_size_A']),pixel)):
        raise ValueError('Manual checkpoint and bin8 input sampling differ.')
    cameras=np.asarray(saved['cameras_xyz']);translations=np.asarray(saved['translations_xy_A'])/pixel
    if not np.isfinite(cameras).all() or not np.isfinite(translations).all():raise ValueError('Invalid manual camera geometry.')
    if flip:images=images[:,::-1,:].copy()
    settings_path=storage_path(job_work(Path(data['coarse_run'])),'config.json')
    settings=json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    core,context=stage_bounds(n,settings.get('tomogram_thickness_percent',100.),halo=32)
    free,_=cp.cuda.runtime.memGetInfo()
    if 4*(n**3+11*(context[1]-context[0])*n*n)>free*.8:raise ValueError('Fresh landmark selection exceeds available GPU memory.')
    progress=storage_path(out,'progress.json')
    progress.write_text(json.dumps({'phase':'Rebuilding landmarks from manual geometry','accepted_tilts':len(active),'total_tilts':len(angles)}))
    kernels=Kernels(cp);weights=angle_weights(angles[active]);stack=cp.asarray(images);del images
    filtered=ramp_filter(stack,cameras[active],cp);del stack
    volume=cp.zeros((n,n,n),cp.float32)
    for z in range(context[0],context[1],8):
        end=min(z+8,context[1]);volume[z:end]=kernels.global_wbp(filtered,cameras[active],translations[active],weights,z,end-z)
    del filtered
    context_score=column_score_volume_gpu(volume[context[0]:context[1]],'bright_structure')
    score=cp.zeros_like(volume);score[core[0]:core[1]]=context_score[core[0]-context[0]:core[1]-context[0]]
    del volume,context_score
    directions=np.unique(active[np.rint(np.linspace(0,len(active)-1,min(5,len(active)))).astype(int)])
    clouds=[neighbor_column_candidates(score,cameras[t],cell=24,z_bounds=core) for t in directions]
    del score;cp.get_default_memory_pool().free_all_blocks()
    points=np.concatenate(clouds)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():raise ValueError('No fresh landmarks in the corrected reconstruction.')
    center=(n-1)/2
    _,unique=np.unique(np.rint(points+center).astype(np.int64),axis=0,return_index=True);points=points[np.sort(unique)]
    centers=np.einsum('tab,kb->kta',cameras[:,:2],points)+translations[None]+center
    options=data['options'];margin=((options['patch_size']-1)/2+2)*options['binning']/8+options['search_support_A']/pixel
    valid=np.all((centers>=margin)&(centers<n-margin),axis=2)
    valid[:,~np.isin(np.arange(len(angles)),active)]=False
    keep=valid.sum(1)>=3;points=points[keep];valid=valid[keep]
    if not len(points) or np.any(valid.sum(0)[active]<8):raise ValueError('Fresh landmarks need eight safe observations per accepted tilt and three views per landmark.')
    source_positions=points[:,::-1]+center
    basis=storage_path(out,'landmark_basis.npz')
    np.savez_compressed(basis,positions_zyx=source_positions,cumulative_shifts_yx=shifts,valid_views=valid,
        classes=np.zeros(len(points),int),angles_deg=angles,active_indices=active,binning=8)
    seed=dict(saved,positions_xyz_A=points*pixel,original_positions_xyz_A=points*pixel,
        source_positions_zyx=source_positions,landmark_ids=np.arange(len(points)),valid_views=valid,
        active_indices=active,anchor_view_index=fit_order[0])
    seed_path=storage_path(out,'landmark_seed_geometry.npz');np.savez_compressed(seed_path,**seed)
    report=dict(protocol='fresh_columns_after_manual_v1',source_manual_geometry=str(parent),source_landmark_basis=str(source_basis),
        landmark_count=len(points),active_indices=active.tolist(),direction_indices=directions.tolist(),
        supported_observations=int(valid.sum()),basis=str(basis),seed_geometry=str(seed_path),selection_binning=8)
    storage_path(out,'landmark_regeneration.json').write_text(json.dumps(clean_json(report),indent=2))
    return dict(data,landmark_basis=str(basis),resume_geometry=str(seed_path)),report
