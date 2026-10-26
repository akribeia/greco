"""Real-space MRCS extraction with streaming Greco STAR metadata."""
from greco.project_layout import join as storage_path
import time
from pathlib import Path
import numpy as np
from ..particle_patches import crop_batch,integer_origins,batch_size
from ..particle_star import write_loop,FORMAT
from ..sta.storage import write_mrc


def defocus_batch(fits,geometries,native_centres,native_shapes,rotations,xyz_centred,pixel,flip,hand):
    """Vectorise particle/view defocus; eliminate per-observation small solves."""
    count,views=native_centres.shape[:2];result=np.empty((count,views,2),float)
    for t,(fit,g,R) in enumerate(zip(fits,geometries,rotations)):
        if not g or g.get('particle_gradient_model','native_beam') == 'native_beam':
            from ..particle_geometry import native_beam_defocus_delta
            delta=native_beam_defocus_delta(R,xyz_centred,pixel,flip,(g or {}).get('handedness',hand))
            result[:,t,0]=fit['defocus_min_A']+delta
            result[:,t,1]=fit['defocus_max_A']+delta
            continue
        mean=.5*(fit['defocus_min_A']+fit['defocus_max_A'])
        if g:
            from ..ctf_estimator import TiltDefocusGeometry
            geometry=TiltDefocusGeometry(g.get('tilt_deg',fit['angle_deg']),g.get('tilt_axis_deg',0.),fit['pixel_size_A'],g.get('handedness',hand),g.get('alpha_offset_deg',0),g.get('beta_offset_deg',0),g.get('gradient_A_per_pixel'))
            origin=(np.asarray(native_shapes[t],dtype=float)-1)/2 if g.get("gradient_A_per_pixel") is not None or g.get("coordinate_origin")=="image_pixel_centres" else np.asarray(native_shapes[t],dtype=float)/2
            mean=geometry.local_mean_defocus_A(mean,native_centres[:,t,0]-origin[1],native_centres[:,t,1]-origin[0])
        surface=xyz_centred@np.linalg.solve(R[:2,:2],R[:2]).T
        depth=xyz_centred@R[2]-surface@R[2,:2]
        effective_hand=g.get('handedness',hand) if g else hand
        delta=mean+effective_hand*(-1 if flip else 1)*pixel*depth-.5*(fit['defocus_min_A']+fit['defocus_max_A'])
        result[:,t,0]=fit['defocus_min_A']+delta;result[:,t,1]=fit['defocus_max_A']+delta
    return result


def extract_tomogram(output,index,member,stack,centres,native_centres,native_shapes,
                     fits,geometries,R,xyz,shape,coarse_pixel,orientation,angles,active,rows,
                     particles,n,pixel,halves,processed,total,gpu,previous_used=0):
    
    from .particle_worker import save_json
    start=time.perf_counter();count=len(particles);views=len(angles)
    directory=storage_path(output, f'tomo_{index+1:04d}')
    from ..project_layout import project_root,safe_name
    from ..sta_layout import enabled
    root=project_root(output)
    if root and enabled(root):directory=Path(output)/'particles'/safe_name(member['name'])
    directory.mkdir(parents=True)
    origins,offsets=integer_origins(centres,n)
    rounded=origins+n//2
    visible=(rounded[...,0]>=n//2)&(rounded[...,1]>=n//2)&(rounded[...,0]+n//2<=stack.shape[2])&(rounded[...,1]+n//2<=stack.shape[1])
    usable=visible.sum(axis=1)>=2
    uv=defocus_batch(fits,geometries,native_centres,native_shapes,R,
        (xyz-(shape-1)/2)*coarse_pixel/pixel,pixel,orientation.get('flip_y',False),orientation.get('df_hand',-1))
    crop_seconds=0.;write_seconds=0.
    device=stack
    if gpu:
        import cupy as cp
        device=cp.asarray(stack,dtype=cp.float32)
    paths=[];image_indices=np.cumsum(visible,axis=1)*visible
    # A completely out-of-frame particle has no image file (but retains its
    # metadata/identity). Visible views are compact and indexed from one.
    for first in range(0,count,batch_size(views,n)):
        last=min(count,first+batch_size(views,n));tic=time.perf_counter()
        raw=crop_batch(device,centres[first:last],visible[first:last],n,gpu,storage_dtype=np.float16)
        crop_seconds+=time.perf_counter()-tic;tic=time.perf_counter()
        for local,k in enumerate(range(first,last)):
            valid=raw[local,visible[k]]
            if not len(valid):paths.append(None);continue
            path=storage_path(directory, f'particle_{k+1:06d}.mrcs')
            write_mrc(path,np.ascontiguousarray(valid),pixel,image_stack=True)
            paths.append(str(path.relative_to(output)))
        write_seconds+=time.perf_counter()-tic
        save_json(storage_path(output, 'progress.json'),dict(processed=processed+last,total=total,
            used_particles=previous_used+int(usable[:last].sum()),tomogram=member['name'],phase='extracting MRCS'))
    del device
    result=dict(name=member['name'],particle_count=count,view_count=views,image_stack=None,
        particle_stacks=paths,image_indices=image_indices,
        poses=np.array([[p['grecoAngle'+a] for a in ('Rot','Tilt','Psi')] for p in particles]),
        scores=[p.get('grecoScore') for p in particles],xyz_angstrom=(xyz-(shape-1)/2)*coarse_pixel,
        shape=shape,coarse_pixel=coarse_pixel,series_uid=member.get('series_id',member['name']),
        visible=visible,usable=usable,offsets=offsets,centres=centres,uv=uv,rotations=R,
        fits=fits,geometries=geometries,native_shapes=native_shapes,orientation=orientation,
        angles=angles,active=active,mdoc_indices=[int(r['mdoc_index']) for r in rows],halves=halves,
        first_particle_id=processed+1,timings=dict(total_seconds=time.perf_counter()-start,
            crop_seconds=crop_seconds,write_seconds=write_seconds),gpu_crop=gpu)
    print(f'MRCS {member["name"]}: {count} particles, box {n}, {result["timings"]}',flush=True)
    return result


def write_dataset(output,members,n,pixel,total,used,observations,config):
    path=storage_path(output, 'particles.star');tmp=storage_path(output, 'particles.star.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        f.write('# Greco Particle STAR 1.0 — see docs/greco-particle-star-v1.md\n')
        write_loop(f,'general',['grecoFormatName','grecoFormatVersion','grecoBoxSize','grecoPixelSizeAngst',
            'grecoParticleCount','grecoSupportedParticleCount','grecoObservationCount',
            'grecoCtfApplied','grecoContrastInverted','grecoCircleCropApplied','grecoCircleFalloffPixel',
            'grecoCoordinateConvention','grecoEulerConvention','grecoParticleGradientModel'],
            [(FORMAT,1,n,pixel,total,sum(used),observations,0,0,0,5.,
              'exported_tomogram_center_nminus1_over2_xyz_angstrom','passive_zyz_degrees','native_beam')])
        write_loop(f,'tomograms',['grecoTomogramId','grecoTomogramUid','grecoTomogramName','grecoImageStack',
            'grecoParticleCount','grecoViewCount','grecoTomogramPixelSizeAngst','grecoTomogramSizeX','grecoTomogramSizeY','grecoTomogramSizeZ'],
            [(i+1,m['series_uid'],m['name'],m['image_stack'],m['particle_count'],m['view_count'],m['coarse_pixel'],*[int(s) for s in m['shape']]) for i,m in enumerate(members)])
        view_columns=['grecoTomogramId','grecoViewId','grecoTiltIndex','grecoMdocIndex','grecoTiltAngleDeg']+[f'grecoProjection{i}{j}' for i in range(3) for j in range(3)]+[
            'grecoDefocusAngleDeg','grecoVoltageKV','grecoCsMM','grecoAmplitudeContrast','grecoPhaseShiftDeg',
            'grecoCentralDefocusUAngst','grecoCentralDefocusVAngst','grecoCtfAxisDeg','grecoDefocusHand',
            'grecoCtfAlphaOffsetDeg','grecoCtfBetaOffsetDeg','grecoDetectorPixelSizeAngst','grecoDetectorSizeX','grecoDetectorSizeY','grecoRasterFlipY',
            'grecoPreExposureElectronPerAngst2','grecoExposureElectronPerAngst2','grecoCtfScaleFactor','grecoDoseWeightingModel','grecoDoseSource',
            'grecoGeometryShiftXAngst','grecoGeometryShiftYAngst']
        def view_rows():
            for i,m in enumerate(members):
                for t,fit in enumerate(m['fits']):
                    g=m['geometries'][t];o=m['orientation']
                    dose=m.get('dose_metadata',[None]*m['view_count'])[t]
                    scale=float(np.cos(np.deg2rad(m['angles'][t]))) if dose else 1.
                    if not 0<scale<=1:raise ValueError('Tilt transfer scale requires angles inside (-90,90).')
                    yield (i+1,t+1,int(m['active'][t]),m['mdoc_indices'][t],float(m['angles'][t]),
                        *[float(a) for a in m['rotations'][t].ravel()],fit['astig_azimuth_raw_deg'],fit['voltage_kv'],fit['cs_mm'],fit['amp_contrast'],float(np.rad2deg(fit.get('phase_rad',0))),
                        fit['defocus_min_A'],fit['defocus_max_A'],g.get('tilt_axis_deg',o.get('ctf_axis_deg',0)),g.get('handedness',o.get('df_hand',-1)),g.get('alpha_offset_deg',0),g.get('beta_offset_deg',0),
                        fit['pixel_size_A'],int(m['native_shapes'][t,1]),int(m['native_shapes'][t,0]),int(o.get('flip_y',False)),
                        dose['prior'] if dose else 0.,dose['exposure'] if dose else 0.,scale,
                        'grant_grigorieff_2015' if dose else 'none',dose['source'] if dose else 'unweighted',
                        *[float(v) for v in m.get('geometry_shifts_A',np.zeros((m['view_count'],2)))[t]])
        write_loop(f,'views',view_columns,view_rows())
        def particle_rows():
            for i,m in enumerate(members):
                for k in range(m['particle_count']):
                    score=m['scores'][k]
                    yield (m['first_particle_id']+k,i+1,m['particle_stacks'][k],1,int(m['visible'][k].sum()),int(m['halves'][k])+1,int(m['usable'][k]),
                        *[float(v) for v in m['xyz_angstrom'][k]],*[float(v) for v in m['poses'][k]],float(score) if score is not None else None)
        write_loop(f,'particles',['grecoParticleId','grecoTomogramId','grecoImageStack','grecoImageStartIndex','grecoImageCount','grecoRandomSubset','grecoUsable',
            'grecoCoordinateXAngst','grecoCoordinateYAngst','grecoCoordinateZAngst','grecoAngleRot','grecoAngleTilt','grecoAnglePsi','grecoScore'],particle_rows())
        def observation_rows():
            for m in members:
                for k in range(m['particle_count']):
                    for t in range(m['view_count']):
                        visible=bool(m['visible'][k,t]);usable=bool(m['usable'][k])
                        applied=m.get('applied_shifts_A');delta=m.get('applied_defocus_offsets_A')
                        yield (m['first_particle_id']+k,t+1,int(m['image_indices'][k,t]),int(visible),
                            *[float(v) for v in m['centres'][k,t]],*[float(v) for v in m['offsets'][k,t]],
                            float(m['uv'][k,t,0]) if visible and usable else None,float(m['uv'][k,t,1]) if visible and usable else None,
                            *([float(v) for v in applied[k,t]] if applied is not None else (0.,0.)),float(delta[t]) if delta is not None else 0.)
        write_loop(f,'observations',['grecoParticleId','grecoViewId','grecoImageIndex','grecoVisible',
            'grecoProjectedXPixel','grecoProjectedYPixel','grecoOffsetXPixel','grecoOffsetYPixel','grecoDefocusUAngst','grecoDefocusVAngst',
            'grecoAppliedResidualShiftXAngst','grecoAppliedResidualShiftYAngst','grecoAppliedDefocusOffsetAngst'],observation_rows())
    tmp.replace(path);return path
