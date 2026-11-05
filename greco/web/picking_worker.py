"""Greco Picker GPU Slurm worker; independent of reconstruction/alignment."""
from greco.project_layout import join as storage_path, ensure_output
import argparse
import json
import logging
import os
from pathlib import Path
import traceback
import numpy as np
import mrcfile
from ..picker.prepare import prepare_template,particle_examples


def reconstruction_inputs(work,template,options):
    """Resolve and validate future outputs only after the Slurm dependency succeeds."""
    result=json.loads((storage_path(storage_path(work, 'output'), 'web_result.json')).read_text())
    if result.get('state')!='completed':raise ValueError('Reconstruction is not complete.')
    volume=Path(result['volumes']['original']).resolve()
    if not volume.is_relative_to((storage_path(work, 'output')).resolve()):raise ValueError('Invalid tomogram path.')
    with mrcfile.open(volume,header_only=True) as handle:
        shape=[int(handle.header.nz),int(handle.header.ny),int(handle.header.nx)]
        voxel=float(handle.voxel_size.x)
        if not np.isfinite(voxel) or voxel<=0:raise ValueError('Tomogram pixel size is missing.')
        if not np.allclose([handle.voxel_size.y,handle.voxel_size.z],voxel):raise ValueError('Tomogram voxels must be isotropic.')
        if (int(handle.header.mapc),int(handle.header.mapr),int(handle.header.maps))!=(1,2,3):raise ValueError('Tomogram must use standard XYZ MRC axes.')
    size=int(round(template['shape'][0]*options['template_pixel_size_A']/voxel))
    if size<4 or size>128 or size>=min(shape):raise ValueError('Template size after resampling must be 4–128 voxels and fit inside the tomogram.')
    if options['particle_diameter_A']/voxel>=min(shape):raise ValueError('Particle diameter exceeds tomogram dimensions.')
    if options['low_pass_A'] is not None and options['low_pass_A']<2*voxel:raise ValueError('Low-pass resolution cannot exceed the tomogram Nyquist limit.')
    from .picking_source import volume_context
    source,summary,result,provenance=volume_context(work)
    angles=np.loadtxt(source['inputs']['rawtlt'],ndmin=1)
    indices=np.asarray(summary['active_indices'])
    if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=len(angles)):raise ValueError('Invalid reconstruction active indices.')
    angles=angles[indices]
    if len(angles)<2 or not np.isfinite(angles).all():raise ValueError('Invalid reconstructed tilt angles.')
    resolved=dict(volume=str(volume),voxel_size_A=voxel,tilt_angles_deg=angles.tolist(),**provenance)
    if result.get('ctf_template_metadata') is not None:
        resolved.update(tilt_series_metadata=result['ctf_template_metadata'],per_tilt_weighting=True)
    return resolved


def run(config_path):
    output=storage_path(config_path.parent, 'output');ensure_output(output)
    try:
        from ..picker.batching import configure_allocator
        configure_allocator()
        from ..picker.matcher import MatchJob,detect_mask_spherical
        from ..picker.geometry import get_angle_list
        from ..picker.fps import get_fps_ids
        from ..picker.protocol import metadata_from_config
        from ..picker.io import read_star,write_star
        config=json.loads(config_path.read_text());options=config['options']
        mode=options.get('matching_mode','fast')
        if mode not in ('fast','exhaustive'):raise ValueError('Invalid picker mode.')
        if config.get('reconstruction_dependency'):
            resolved=reconstruction_inputs(Path(config['reconstruction_work']),config['template'],options)
            if config.get('picking_source_kind') and resolved['picking_source_kind']!=config['picking_source_kind']:
                raise ValueError('Selected picking source changed before execution.')
            config.update(resolved)
            config.pop('reconstruction_dependency',None)
            config.pop('reconstruction_work',None)
            # Save the actual immutable inputs, rather than a future dependency.
            temporary=config_path.with_name('.'+config_path.name+'.resolved.tmp')
            temporary.write_text(json.dumps(config,indent=2,allow_nan=False))
            os.chmod(temporary,config_path.stat().st_mode & 0o777)
            os.replace(temporary,config_path)
        voxel=float(config['voxel_size_A'])
        if not config.get('mask'):raise ValueError('A template mask MRC is required. Select a mask and resubmit.')
        template,mask=prepare_template(config['template']['path'],options['template_pixel_size_A'],voxel,
            options['flip_z'],options['invert_contrast'],mask_path=config['mask']['path'])
        for name,data in (('template.mrc',template),('mask.mrc',mask)):
            with mrcfile.new(storage_path(output, name),overwrite=True) as handle:
                handle.set_data(data);handle.voxel_size=voxel
        angles=get_angle_list(options['angular_search_deg'],sort_angles=True,symmetry=1)
        ids,info=get_fps_ids(angles,min(2592,len(angles))) if mode=='fast' else (None,None)
        spherical=detect_mask_spherical(np.ascontiguousarray(mask.T))
        hybrid=dict(n_candidates=options['candidate_count'],max_climb_steps=3,mu_samples=4096,
            mu_adaptive=False,mu_esc_cons=3.,n_false_positives=1.,n_particles=options['max_particles'],
            particle_diameter=options['particle_diameter_A'],fps=info,
            symmetric_noise=True,threshold_factor=options.get('threshold_factor',1.),candidate_source='signal',candidate_geometric_support=True,
            candidate_nms_radius=options['particle_diameter_A']/(4*voxel))
        metadata=metadata_from_config(config)
        fft_batch=options.get('fft_batch_size')
        if fft_batch is not None and (isinstance(fft_batch,bool) or not isinstance(fft_batch,int) or fft_batch<1):
            raise ValueError('fft_batch_size must be a positive integer.')
        job=MatchJob(tomogram=Path(config['volume']),template=storage_path(output, 'template.mrc'),mask=storage_path(output, 'mask.mrc'),
            output_dir=output,ts_metadata=metadata,voxel_size=voxel,batch_size=fft_batch,
            angle_increment=options['angular_search_deg'],high_pass=options['high_pass_A'],low_pass=options['low_pass_A'],
            noise_correction=True,noise_every=1,mask_is_spherical=spherical,rotate_mask=False,
            orientation_ids=[int(i) for i in ids] if ids is not None else None,
            hybrid_cfg=hybrid if mode=='fast' else None,
            noise_template_iterations=0,symmetric_noise=mode=='exhaustive',
            impulse_spectral_weighting=options.get('impulse_spectral_weighting',False))
        matching_result=job.run()
        if mode=='exhaustive':
            from ..picker.extract import extract_and_write
            _,_,statistics=extract_and_write(output,n_particles=options['max_particles'],
                particle_diameter=options['particle_diameter_A'],n_false_positives=1.,threshold_factor=options.get('threshold_factor',1.))
        else:
            statistics=matching_result['picking_statistics']
        columns,particles=read_star(storage_path(output, 'particles.star'))
        # Stable project tilt-series names, not the generic tomogram filename.
        for row in particles:row['grecoTomogramName']=config['series_name']
        with mrcfile.open(config['volume'],header_only=True) as handle:
            shape_xyz=np.array([handle.header.nx,handle.header.ny,handle.header.nz],float)
        # The port exports centred Angstrom coordinates about shape/2 (not (shape-1)/2).
        xyz=np.array([[row['grecoCoordinate'+a+'Angst'] for a in 'XYZ'] for row in particles],float).reshape(-1,3)/voxel+shape_xyz/2
        from ..picking_star import metadata_for
        write_star(storage_path(output,'particles.star'),columns,[tuple(row[c] for c in columns) for row in particles],metadata=metadata_for(config,shape_xyz))
        from ..picking_star import input_manifest
        manifest=input_manifest(config,output)
        (output/'inputs/manifest.json').write_text(json.dumps(manifest,indent=2))
        (storage_path(output, 'examples.png')).write_bytes(particle_examples(config['volume'],xyz,max(9,int(round(options['particle_diameter_A']/voxel)))))
        result=dict(state='completed',pick_count=len(particles),examples=str(storage_path(output, 'examples.png')),
            star=str(storage_path(output, 'particles.star')),voxel_size_A=voxel,
            template_shape=list(template.shape),volume=config['volume'],options=options,engine='greco-picker',
            mask=config['mask'],mask_is_spherical=spherical,matching_mode=mode,
            impulse_spectral_weighting=job.impulse_spectral_weighting,
            rotate_mask=False,screen_orientations=len(ids) if ids is not None else len(angles),
            noise_model='phase-only-antithetic',noise_template_iterations=0,
            per_tilt_weighting=metadata.per_tilt_weighting,
            ctf_template_weighting=metadata.ctf_data is not None,
            dose_template_weighting=metadata.dose_accumulation is not None,
            protocol=f'greco-{mode}-fixed-mask-antithetic-v1',picking_statistics=statistics)
        for key in ('picking_source_kind','picking_source_work','picking_source_run','rigid_geometry'):
            if key in config:result[key]=config[key]
    except Exception as error:
        traceback.print_exc();result=dict(state='failed',error=str(error))
    temporary=storage_path(output, 'web_result.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(storage_path(output, 'web_result.json'))
    if result['state']=='failed':raise RuntimeError(result['error'])
    return result


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
