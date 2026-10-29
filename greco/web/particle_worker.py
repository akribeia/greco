"""Extract raw tilt patches and dispatch supported STA jobs."""
from greco.project_layout import join as storage_path, ensure_output
from greco.preprocessing_star import read_table
import argparse
import json
from pathlib import Path
import traceback
import time
import numpy as np


def save_json(path,value):
    from ..sta_layout import write_metadata
    write_metadata(path,value)


def check_snapshot(snapshot):
    for item in snapshot:
        stat=Path(item['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(item['size'],item['mtime_ns']):
            raise ValueError('Assembly input changed: '+item['path']+'. Create or submit the assembly again.')


def apply_observation_corrections(native_centres,fits,rows,particles):
    """Bake raw-detector Å offsets into crop centres / shared tilt defocus once."""
    groups={int(row['mdoc_index']):t for t,row in enumerate(rows)};defocus={}
    if len(groups)!=len(rows):raise ValueError('Duplicate acquisition index.')
    fits=[dict(f) for f in fits]
    for k,p in enumerate(particles):
        for correction in p.get('_observation_corrections',[]):
            if correction['mdoc_index'] not in groups:continue
            t=groups[correction['mdoc_index']]
            shift=np.array([correction.get('grecoResidualShift'+d+'Angst',0.) for d in 'XY'])
            delta=correction.get('grecoDefocusOffsetAngst',0.)
            if not np.isfinite(shift).all() or not np.isfinite(delta):raise ValueError('Non-finite observation correction.')
            native_centres[k,t]+=shift/fits[t]['pixel_size_A']
            if t in defocus and not np.isclose(defocus[t],delta,rtol=0,atol=1e-5):raise ValueError('Re-extraction requires a common defocus offset per tilt.')
            defocus[t]=delta
    for t,delta in defocus.items():
        for key in ('defocus_min_A','defocus_max_A'):fits[t][key]+=delta
    return fits


def apply_rigid_geometry(native_centres,rotations,xyz_A,fits,rows,updates):
    """Replay absolute cameras from original MC geometry during re-extraction."""
    groups={int(r['mdoc_index']):t for t,r in enumerate(rows)}
    new=rotations.copy();translations=np.zeros((len(rows),2));seen=set()
    for update in updates:
        index=int(update['mdoc_index'])
        if index in seen:raise ValueError('Duplicate refined acquisition frame')
        seen.add(index)
        if index not in groups:raise ValueError('Unknown refined acquisition frame')
        t=groups[index];camera=np.array([[update[f'grecoProjection{i}{j}'] for j in range(3)] for i in range(3)])
        shift=np.array([update.get('grecoGeometryShift'+d+'Angst',0.) for d in 'XY'])
        if not np.isfinite(camera).all() or not np.isfinite(shift).all() or not np.allclose(camera@camera.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(camera),1.):
            raise ValueError('Invalid refined rigid camera')
        native_centres[:,t]+=(np.asarray(xyz_A)@(camera[:2]-rotations[t,:2]).T+shift)/fits[t]['pixel_size_A']
        new[t]=camera;translations[t]=shift
    return new,translations


def saved_mc_plane(directory,pixel,native_pixel,output_binning=8,*,gpu=True):
    """Prefer an exact saved sum, otherwise decimate the saved native sum."""
    import mrcfile
    factor=int(round(pixel/native_pixel))
    for binning in dict.fromkeys((factor,output_binning,8)):
        path=storage_path(directory, f'mc_bin{binning}.mrc')
        if not path.is_file():continue
        with mrcfile.open(path,header_only=True) as m:matches=np.isclose(float(m.voxel_size.x),pixel)
        if matches:
            with mrcfile.open(path) as m:plane=np.array(m.data,np.float32)
            return plane,np.array(plane.shape)*factor,np.zeros(2),'reused'
    path=storage_path(directory, 'mc_bin1.mrc')
    if not path.is_file():return None
    with mrcfile.open(path) as m:
        if not np.isclose(float(m.voxel_size.x),native_pixel):raise ValueError('Saved native MC sampling differs from its CTF fit.')
        native=np.array(m.data,np.float32)
    native_shape=np.array(native.shape)
    trimmed=np.array([2*(n//(2*factor))*factor for n in native_shape])
    if np.any(trimmed==0):raise ValueError('Saved MC image is too small for output sampling.')
    offset=(native_shape-trimmed)//2
    native=native[offset[0]:offset[0]+trimmed[0],offset[1]:offset[1]+trimmed[1]]
    from .refinement_worker import bin_plane_device
    if gpu:
        import cupy as cp
        plane=cp.asnumpy(bin_plane_device(cp.asarray(native),factor,cp))
    else:plane=np.asarray(bin_plane_device(native,factor,np),np.float32)
    return plane,native_shape,offset,'saved bin1'


def extraction_geometry(path,expected_angles):
    """Select angles and shifts by the same accepted acquisition identities."""
    with np.load(path,allow_pickle=False) as saved:
        angles=np.asarray(saved['angles_deg'],float)
        fitted=np.asarray(saved.get('refined_angles_deg',angles),float)
        active=np.asarray(saved['active_indices'])
        shifts=np.asarray(saved['cumulative_shifts_yx'],float)
    expected=np.asarray(expected_angles,float)
    if (angles.ndim!=1 or angles.shape!=expected.shape
            or not np.allclose(angles,expected,rtol=0,atol=1e-4)):
        raise ValueError('Accepted geometry and tilt order differ.')
    if (active.ndim!=1 or active.dtype.kind not in 'iu' or not len(active)
            or np.any(active<0) or np.any(active>=len(angles)) or np.any(np.diff(active.astype(np.int64))<=0)):
        raise ValueError('Invalid accepted tilt selection for extraction.')
    if shifts.shape!=(len(angles),2) or not np.isfinite(shifts).all():
        raise ValueError('Accepted geometry has inconsistent tilt shifts.')
    active=active.astype(int)
    if fitted.shape!=angles.shape or not np.isfinite(fitted).all():raise ValueError("Invalid refined extraction angles.")
    return fitted[active],active,shifts[active]


def load_mc_stack(prep,rows,pixel,*,ensure_native=False):
    """Reuse raw MC sums if sampling matches; otherwise replay saved shifts.

    Never read the CTF-corrected alignment stack. Only one tomogram's tilt
    stack is held in host RAM, with native movie replay on GPU when needed.
    """
    import mrcfile
    source=json.loads((storage_path(prep, 'config.json')).read_text());options=source['options']
    planes=[];fits=[];geometries=[];native_shapes=[];offsets=[]
    gain=None;mdoc=None
    for row in rows:
        idx=int(row['mdoc_index']);directory=storage_path(storage_path(storage_path(prep, 'output'), 'per_tilt'), f'tilt_{idx:04d}')
        ctf=json.loads((storage_path(directory, 'ctf.json')).read_text())
        geometry_path=storage_path(directory, 'ctf_geometry.json')
        geometry=json.loads(geometry_path.read_text()) if geometry_path.exists() else {}
        fits.append(ctf);geometries.append(geometry)
        ratio=pixel/ctf['pixel_size_A'];factor=int(round(ratio))
        if factor<1 or not np.isclose(factor,ratio):raise ValueError('Requested pixel size must be an integer multiple of the MC sampling.')
        saved=saved_mc_plane(directory,pixel,ctf['pixel_size_A'],options.get('output_binning',8))
        if ensure_native and not (storage_path(directory, 'mc_bin1.mrc')).is_file():saved=None
        source_kind='replayed'
        if saved is not None:
            plane,native_shape,offset,source_kind=saved
            native_shapes.append(native_shape);offsets.append(offset)
        else:
            import cupy as cp
            from types import SimpleNamespace
            from .. import preprocess as p
            from ..detector_defects import apply_correction
            from .refinement_worker import bin_plane_device
            if mdoc is None:
                mdoc=p._movie_mdoc(SimpleNamespace(mdoc=storage_path(prep, 'input.mdoc'),eer_upscaling=options.get('eer_upscaling',1)))
                gain=p._load_gain(source['series'].get('gain_path'),flip_y=source['series'].get('gain_flip_y',False))
                if gain is not None and options.get('eer_upscaling',1)==2 and gain.shape==(4096,4096) and all(Path(t.movie).suffix.lower()=='.eer' for t in mdoc.tilts):
                    gain=np.repeat(np.repeat(gain,2,axis=0),2,axis=1)
            tilt=mdoc.tilts[idx]
            frames,_=p.load_movie(storage_path(storage_path(prep, 'movies'), tilt.movie),tilt.n_frames,
                eer_group=options.get('eer_group',15),eer_threads=options.get('eer_threads',1),
                eer_upscaling=options.get('eer_upscaling',1))
            frames=p._prepare_frames(frames,gain,options.get('flip_y',False),options.get('transpose',False))
            with np.load(storage_path(directory, 'shifts.npz')) as d:shifts=np.array(d['full_shifts'])
            if shifts.shape!=(len(frames),2):raise ValueError('Saved MC shifts and movie groups differ.')
            frames_device=cp.asarray(frames);del frames
            apply_correction(frames_device,p._load_detector_correction(directory))
            native=p._rebuild_aligned_sum(p._Dev(True),frames_device,shifts);del frames_device
            if ensure_native:
                # Cache the unweighted, uncorrected sum BEFORE cropping/binning.
                # MC shifts and detector correction are replayed, never re-fitted.
                target=storage_path(directory, 'mc_bin1.mrc');temporary=storage_path(directory, 'mc_bin1.sta-tmp.mrc')
                with mrcfile.new(temporary,overwrite=True) as m:
                    m.set_data(cp.asnumpy(native).astype(np.float32,copy=False));m.voxel_size=ctf['pixel_size_A']
                temporary.replace(target)
                print(f'Saved native STA sum: {target}',flush=True)
            native_shapes.append(np.array(native.shape))
            trimmed=np.array([2*(n//(2*factor))*factor for n in native.shape])
            offset=(np.array(native.shape)-trimmed)//2;offsets.append(offset)
            native=native[offset[0]:offset[0]+trimmed[0],offset[1]:offset[1]+trimmed[1]]
            plane=cp.asnumpy(bin_plane_device(native,factor,cp));del native
        if plane.ndim!=2 or not np.isfinite(plane).all():raise ValueError('Invalid raw MC sum.')
        planes.append(plane)
        print(f'Raw MC tilt {len(planes)}/{len(rows)} ({source_kind})',flush=True)
    return np.stack(planes),fits,geometries,np.array(native_shapes),np.array(offsets)


def run(config):
    if config.get('action')=='sta_after_dependencies':
        from .sta_dependencies import resolve
        config=resolve(config)
    from ..sta_layout import freeze_mask
    config=freeze_mask(config)
    if config.get('action')=='postprocess':
        from ..sta.postprocess import run as postprocess
        return postprocess(config)
    if config.get('action')=='extract_after_picking':
        from .pending_extraction import resolve
        config=resolve(config)
    if config.get('action') in ('polishing','ctf_refinement'):
        from ..sta.observation_refinement import run as refine_observations
        result=refine_observations(config)
        from ..sta.observation_report import publish
        report=publish(Path(config['output']),config,result)
        result['report_pdf']=report['path']
        save_json(storage_path(Path(config['output']),'web_result.json'),result)
        return result
    if config.get('action')=='refine3d':
        from ..sta.refinement import refine
        return refine(config)
    if config.get('action')=='classify':
        from ..sta.classification import classify
        return classify(config)
    if config.get('action')!='extract':raise ValueError('Unsupported STA action.')
    import mrcfile
    from ..particle_geometry import raw_particle_geometry,alignment_pick_grid
    from ..picker.io import read_star
    from ..runtime import ROOT
    reference=storage_path(ROOT, 'greco')
    if not (storage_path(storage_path(storage_path(reference, '_numerics'), 'tomo'), 'extraction.py')).is_file():raise ValueError('Greco Python reference is unavailable.')
    output=Path(config['output']);ensure_output(output)
    start=time.perf_counter()
    n=config['output_box'];pixel=config['pixel_size_A']
    check_snapshot(config['snapshot'])
    total=sum(m['particle_count'] for m in config['tomograms'])
    if total<1:raise ValueError('Not enough particles for this job.')
    # Balanced, deterministic particle halves; these are not independent refinements.
    halves=np.arange(total)%2;np.random.default_rng(0).shuffle(halves)
    used=[0,0];observations=0;processed=0;reports=[];saved_members=[]
    import shutil
    estimate=sum(m['particle_count']*m.get('tilt_count',len(m.get('angles',[])) or 100)*n*n*2 for m in config['tomograms'])
    if shutil.disk_usage(output).free<estimate*1.1:raise ValueError('Not enough free disk space for particle extraction.')
    for member_index,member in enumerate(config['tomograms']):
        check_snapshot(config['snapshot'])
        recon=json.loads(Path(member['reconstruction_config']).read_text())
        inputs=recon['inputs'];orientation=recon['orientation'];prep=Path(member['preprocessing_work'])
        rows=read_table(inputs['frozen_shifts'])
        all_angles=np.array([float(r['angle_deg']) for r in rows])
        angles,active,cumulative=extraction_geometry(member['final_state'],all_angles)
        rows=[rows[i] for i in active]
        frozen=np.array([[float(r['dy_raw']),float(r['dx_raw'])] for r in rows])
        with mrcfile.open(inputs['stack'],header_only=True) as m:
            coarse_size=int(m.header.nx);coarse_pixel=float(m.voxel_size.x)
        # Frozen shifts are in decoded detector pixels, including EER
        # upscaling. The project pixel size may instead describe 4K sampling.
        frozen/=float(inputs['binning'])
        if orientation.get('flip_y',False):frozen[:,0]*=-1
        with mrcfile.open(member['volume'],header_only=True) as m:
            shape=np.array([m.header.nx,m.header.ny,m.header.nz],int);voxel=float(m.voxel_size.x)
        _,particles=read_star(Path(member['star']))
        if len(particles)!=member.get('source_particle_count',member['particle_count']):raise ValueError('Picking count changed.')
        xyz=np.array([[p['grecoCoordinate'+a+'Angst'] for a in 'XYZ'] for p in particles])/voxel+shape/2
        if 'refined_particles' in member:
            refined=member['refined_particles']
            xyz=np.asarray([[p['grecoCoordinate'+a+'Angst'] for a in 'XYZ'] for p in refined],float)/voxel+(shape-1)/2
            particles=[{**{'grecoAngle'+a:p['grecoAngle'+a] for a in ('Rot','Tilt','Psi')},'grecoScore':p.get('grecoScore')} for p in refined]
            if not np.isfinite(xyz).all():raise ValueError('Invalid refined coordinates.')
            subsets=np.array([p['grecoRandomSubset']-1 for p in refined],int)
            if np.any((subsets<0)|(subsets>1)):raise ValueError('Invalid refined half labels.')
            halves[processed:processed+len(refined)]=subsets
        if 'selected_coordinates_A' in member:
            # Greco STAR uses (n-1)/2; picker coordinates use n/2.
            original=(xyz-(shape-1)/2)*voxel
            indices=[]
            for selected in member['selected_coordinates_A']:
                matches=np.flatnonzero(np.all(np.isclose(original,selected,rtol=0,atol=1e-4),axis=1))
                if len(matches)!=1 or int(matches[0]) in indices:raise ValueError('Classification coordinates do not uniquely match the source picks.')
                indices.append(int(matches[0]))
            particles=[particles[i] for i in indices];xyz=xyz[indices]
        xyz,shape,export_transform=alignment_pick_grid(xyz,shape,voxel,coarse_size,
            coarse_pixel,member['export_transform'])
        from ..alignment_geometry import resolve as resolve_geometry,scalar_or_array
        with np.load(member['final_state'],allow_pickle=False) as geometry_state:
            fitted_axes=resolve_geometry(geometry_state,all_angles,orientation['rotation_deg'])[1][active]
        centres,R=raw_particle_geometry(xyz,shape,export_transform,angles,cumulative,
            frozen,scalar_or_array(fitted_axes),orientation.get('flip_y',False),coarse_size)
        if member.get('rigid_geometry'):
            from ..particle_geometry import refined_raw_particle_geometry
            with np.load(member['rigid_geometry'],allow_pickle=False) as saved:
                centres,R=refined_raw_particle_geometry(xyz,export_transform,centres,R,
                    saved,all_angles[active],[int(r['mdoc_index']) for r in rows],coarse_pixel,
                    coarse_size,orientation.get('flip_y',False))
        save_json(storage_path(output, 'progress.json'),dict(processed=processed,total=total,used_particles=sum(used),
            observations=observations,tomogram=member['name'],phase='loading raw MC tilts'))
        load_start=time.perf_counter()
        stack,fits,geometries,native_shapes,offsets=load_mc_stack(prep,rows,pixel,ensure_native=True)
        load_seconds=time.perf_counter()-load_start
        print(f'Raw MC source {member["name"]}: {load_seconds:.3f}s',flush=True)
        native_factors=np.array([coarse_pixel/c['pixel_size_A'] for c in fits])
        factors=np.array([pixel/c['pixel_size_A'] for c in fits])
        native_centres=centres*native_factors[None,:,None]
        geometry_translations=np.zeros((len(rows),2))
        if member.get('refined_views'):
            R,geometry_translations=apply_rigid_geometry(native_centres,R,(xyz-(shape-1)/2)*coarse_pixel,fits,rows,member['refined_views'])
        original_native_centres=native_centres.copy();original_fits=fits
        if 'refined_particles' in member:
            # Correction shifts are raw detector X/Y in Å, independent of
            # extraction sampling. Bake them into crop centres exactly once.
            fits=apply_observation_corrections(native_centres,fits,rows,member['refined_particles'])
        centres=(native_centres-offsets[None,:,::-1])/factors[None,:,None]
        from .particle_mrcs import extract_tomogram
        saved_member=extract_tomogram(output,member_index,member,stack,centres,native_centres,native_shapes,
            fits,geometries,R,xyz,shape,coarse_pixel,orientation,angles,active,rows,particles,n,pixel,
            halves[processed:processed+len(particles)],processed,total,config.get('gpu',True),sum(used))
        saved_member['applied_shifts_A']=(native_centres-original_native_centres)*np.array([f['pixel_size_A'] for f in fits])[None,:,None]
        saved_member['geometry_shifts_A']=geometry_translations
        from ..sta.transfer_envelope import extraction_doses
        saved_member['dose_metadata']=extraction_doses(prep,rows)
        saved_member['applied_defocus_offsets_A']=np.array([f['defocus_min_A']-old['defocus_min_A'] for f,old in zip(fits,original_fits)])
        saved_member['timings']['source_loading_seconds']=load_seconds
        mask=saved_member['usable']
        for half in (0,1):used[half]+=int(np.sum(mask & (saved_member['halves']==half)))
        observations+=int(saved_member['visible'][mask].sum())
        processed+=len(particles);saved_members.append(saved_member)
        reports.append(dict(name=member['name'],particles=len(particles),used=int(mask.sum()),timings=saved_member['timings']))
        del stack
        continue
    if sum(used)==0:raise ValueError('No particles have two fully supported tilt patches.')
    check_snapshot(config['snapshot'])
    from .particle_mrcs import write_dataset
    from ..particle_star import FORMAT
    manifest_path=write_dataset(output,saved_members,n,pixel,total,used,observations,config)
    result=dict(state='completed',extracted=True,particles=str(manifest_path),format=FORMAT,image_storage='float16',
        particle_count=total,used_particles=sum(used),half_counts=used,observations=observations,
        tomograms=reports,box_size=n,pixel_size_A=pixel,additional_binning=1,
        gpu_crop=config.get('gpu',True),elapsed_seconds=time.perf_counter()-start)
    save_json(storage_path(output, 'web_result.json'),result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    config=json.loads(Path(parser.parse_args().config).read_text())
    try:run(config)
    except Exception as error:
        output=Path(config['output']);ensure_output(output)
        save_json(storage_path(output, 'web_result.json'),dict(state='failed',error=str(error)))
        traceback.print_exc();raise


if __name__=='__main__':main()
