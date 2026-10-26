"""Run Greco Optimizer, final reconstruction and optional TinyCARE."""
from greco.volume_storage import float16_data
from greco.project_layout import join as storage_path, ensure_output
from greco.preprocessing_star import read_table
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import traceback



def prepare_raw_source(inputs,orientation,output):
    """Reflect source/frozen shifts together; Greco performs rotation on demand.

    Never pass MDOC-index columns to Greco's positional frozen-shift parser.
    """
    import numpy as np
    import mrcfile
    rows=read_table(inputs['frozen_shifts'])
    angles=np.loadtxt(inputs['rawtlt'],ndmin=1)
    source=Path(inputs['stack'])
    with mrcfile.mmap(source,mode='r') as handle:
        if handle.data.ndim!=3 or len(handle.data)!=len(rows) or len(rows)!=len(angles) or len(rows)<7:
            raise ValueError('Greco requires at least seven matched tilt planes and frozen shifts.')
        if handle.data.shape[1]!=handle.data.shape[2]:raise ValueError('Greco requires square tilt planes.')
        if not np.allclose([float(r['angle_deg']) for r in rows],angles,atol=1e-4):
            raise ValueError('Frozen shifts and tilt angles are not in the same order.')
        voxel=float(handle.voxel_size.x)
        if orientation.get('flip_y',False):
            source=storage_path(output, 'reflected_rawsource.mrcs')
            with mrcfile.new_mmap(source,shape=handle.data.shape,mrc_mode=2) as target:
                for i in range(len(rows)):target.data[i]=handle.data[i,::-1,:]
                target.voxel_size=handle.voxel_size;target.update_header_stats()
    frozen=storage_path(output, 'frozen_shifts.csv')
    with frozen.open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['plane','angle_deg','dy_raw','dx_raw'])
        for i,row in enumerate(rows):
            dy=float(row['dy_raw'])*(-1 if orientation.get('flip_y',False) else 1)
            writer.writerow([i,float(row['angle_deg']),dy,float(row['dx_raw'])])
    return source,frozen,voxel


def run(config):
    import mrcfile
    import numpy as np
    data=json.loads(config.read_text());output=Path(data['output']);ensure_output(output)
    result={}
    try:
        # Deferred inputs only exist after the preprocessing dependency finishes.
        with mrcfile.mmap(data['inputs']['stack'],mode='r') as handle:
            shape=handle.data.shape
            if len(shape)!=3 or shape[0]<7 or shape[1]!=shape[2] or shape[1]>1024:
                raise ValueError('Greco requires at least seven square tilt planes, at most 1024 pixels per side.')
        mode=data.get('optimizer','greco_optimizer')
        from greco.tomogram_thickness import validate_percent,slab_bounds
        thickness=validate_percent(data.get('tomogram_thickness_percent',50.))
        sigma=data.get('entry_search_sigma_px',8.)
        from ..optimizer.entry import validate_sigma
        sigma=validate_sigma(sigma)
        if mode!='greco_optimizer':raise ValueError('Unknown reconstruction optimizer.')
        guidance=None
        if data.get('alignment_options') is not None:
            from .alignment_options import validate
            guidance=validate(**data['alignment_options'])
        source,frozen,voxel=prepare_raw_source(data['inputs'],data['orientation'],output)
        from ..runtime import ROOT,worker_environment
        env=worker_environment()
        alignment=storage_path(output, 'alignment')
        command=[sys.executable,'-m','greco.optimizer.progressive_alignment','--stack',str(source),
                 '--rawtlt',data['inputs']['rawtlt'],'--output',str(alignment),
                 '--binning',str(data['inputs']['binning']),'--raw-source',
                 '--frozen-shifts',str(frozen),'--axis-deg',str(data['orientation']['rotation_deg']),
                 '--landmark-support','accepted','--cc-method','gradient',
                 '--solver-backend','gpu_schur','--resample','exact','--extra-rounds','3',
                 '--entry-search-sigma-px',str(sigma),'--tomogram-thickness-percent',str(thickness)]
        if guidance is not None:
            command[2]='greco.optimizer.guidance'
            env['GRECO_IMAGE_SMOOTHNESS']=str(guidance['smoothness_strength'])
            env['GRECO_RESCUE_ENABLED']='1' if guidance['rescue_tilts'] else '0'
        command.extend(['--global-margin-buffer','0'])
        (storage_path(output, 'command.json')).write_text(json.dumps(command,indent=2))
        subprocess.run(command,env=env,cwd=ROOT,check=True)
        from ..reconstruction_layout import organize_engine_output
        organize_engine_output(output)
        summary=json.loads((storage_path(alignment, 'summary.json')).read_text())
        active=np.asarray(summary['active_indices'],int)
        angles=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)[active]
        final_protocol=data['final_reconstruction_protocol']
        if final_protocol!='native-wiener-sz-no-reliability-canonical-z-v1':
            raise ValueError('Unsupported final reconstruction protocol; create a new reconstruction.')
        physical_metadata=None
        final_binning=data.get('final_binning') or int(data['inputs']['binning'])
        if final_binning!=int(data['inputs']['binning']):
            if final_protocol!='native-wiener-sz-no-reliability-canonical-z-v1':
                raise ValueError('Independent final binning requires native Wiener reconstruction.')
            from .segmentation_worker import reconstruct
            sampled=reconstruct(dict(source_config=data,coarse_run=str(alignment),
                binning=final_binning,source_work=str(config.parent)),output)
            original=Path(sampled['volumes']['original']);voxel=sampled['voxel_size_A']
            full_shape=[sampled['shape'][-1]]*3
            z_start,z_stop=slab_bounds(full_shape[0],thickness)
            physical_metadata=sampled['ctf_template_metadata']
            volume_shape=sampled['shape']
        else:
            from greco.final_reconstruction import final_stack
            stack,physical_metadata=final_stack(data,alignment,output)
            from .. import tomography
            if data.get('final_wbp','standard')!='standard':raise ValueError('Unknown final WBP mode')
            from greco.rigid_backprojection import center_projections
            volume=tomography.gpu_reconstruct_wbp(center_projections(stack,np),angles)
            from greco.final_reconstruction import canonical_export
            volume=canonical_export(volume)
            full_shape=list(volume.shape)
            z_start,z_stop=slab_bounds(int(volume.shape[0]),thickness)
            volume=np.ascontiguousarray(volume[z_start:z_stop])
            original=storage_path(output, 'tomogram.mrc')
            with mrcfile.new(original) as handle:handle.set_data(float16_data(volume));handle.voxel_size=voxel
            volume_shape=list(volume.shape)
        if 'cupy' in sys.modules:
            sys.modules['cupy'].get_default_memory_pool().free_all_blocks()
        result=dict(state='completed',
                    volumes=dict(original=str(original)),shape=volume_shape,final_binning=final_binning,alignment_binning=int(data['inputs']['binning']),
                    voxel_size_A=voxel,contrast_inverted=physical_metadata is None,
                    final_reconstruction_protocol=final_protocol,output_contrast_convention='dark_particles',
                    final_wbp='standard',
                    final_projection_background='per_projection_mean_before_zero_padding_v1',
                    tomogram_thickness_percent=thickness,internal_volume_shape_zyx=full_shape,
                    export_z_crop_bounds=[z_start,z_stop],
                    ctf_template_metadata=physical_metadata,main_revision=data['main_revision'],orientation=data['orientation'],optimizer=mode,optimizer_name='Greco Optimizer',
                    optimizer_version=1,
                    historical_protocol=data.get('historical_protocol'),
                    entry_search_sigma_px=data.get('entry_search_sigma_px'),
                    active_count=len(active),total_tilts=len(np.loadtxt(data['inputs']['rawtlt'],ndmin=1)),
                    excluded_signal_indices=summary.get('excluded_signal_indices',[]),
                    excluded_signal_angles_deg=summary.get('excluded_signal_angles_deg',[]),
                    full_range=summary['full_range'],last_gcc_stop=summary['last_gcc_stop'],denoising_errors={})
        if physical_metadata is not None:
            from greco.final_reconstruction import export_transform
            result.update(final_ctf_filter='wiener_sz',final_reliability=False,
                internal_to_export_zyx=export_transform(full_shape[0],z_start),export_z_reflected=True)
        # Optional final-volume denoising never feeds back into registration.
        from .final_denoising import denoise_outputs
        denoise_outputs(data,result)
        if not summary['full_range']:
            excluded=summary.get('excluded_signal_indices',[])
            if excluded and len(active)+len(excluded)==result['total_tilts']:
                result['warning']=f"Excluded {len(excluded)} projections without usable signal. Tomogram uses {len(active)}/{result['total_tilts']} tilts."
            else:
                result['warning']=f"Greco stopped before the full range: {len(active)}/{result['total_tilts']} accepted tilts. Tomogram uses accepted tilts."
        if data.get('alignment_options') is not None:result['alignment_options']=guidance
        from ..reconstruction_layout import publish
        publish(output,result)
    except Exception as error:
        traceback.print_exc();result=dict(state='failed',error=str(error))
    temporary=storage_path(output, '.web_result.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(storage_path(output, 'web_result.json'))
    return 1 if result['state']=='failed' else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().config))
