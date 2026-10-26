"""Joint bin8 image refinement: shared XYZ, shifts, axis and inclinations."""
from greco.project_layout import join as storage_path, ensure_output
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np


def run(config):
    from .refinement_worker import load_state,clean_json
    from .refinement import validate_angle_options
    from ..angular_joint import main
    data=json.loads(config.read_text());output=Path(data['output'])
    started=time.monotonic()
    try:
        options=validate_angle_options(data['options'])
        report=main(config,output,refine_xyz=True,reconstruct=True,xyz_total_limit=None,
                    max_passes=options['max_passes'],angle_options=options)
        geometry_source=report.get('last_accepted_geometry')
        if not geometry_source:raise ValueError('No accepted joint refinement geometry.')
        original_angles=np.loadtxt(data['inputs']['rawtlt'],ndmin=1)
        original_pos,original_shifts,original_valid,classes,_,b=load_state(
            storage_path(Path(data['coarse_run']), 'final_state.npz'),original_angles)
        with np.load(geometry_source,allow_pickle=False) as state:
            fields={key:np.array(state[key]) for key in state.files}
        fit_bin=float(fields['binning'])
        if fit_bin!=8:
            from ..angular_joint import rescale_geometry
            fields['positions_zyx'],fields['cumulative_shifts_yx']=rescale_geometry(
                fields['positions_zyx'],fields['cumulative_shifts_yx'],report['fit_size'],report['source_size'],
                float(fields['axis_rotation_deg']),data['orientation'].get('flip_y',False))
            fields['binning']=np.array(8.)
        origin_axis=float(data['orientation']['rotation_deg'])
        fields.update(original_angles_deg=original_angles,original_axis_deg=origin_axis,
                      original_positions_zyx=original_pos,original_valid_views=original_valid)
        geometry=storage_path(output, 'angle_state.npz');np.savez_compressed(geometry,**fields)
        angles=fields['angles_deg'];axis=float(fields['axis_rotation_deg'])
        csv_path=storage_path(output, 'angles.csv')
        np.savetxt(csv_path,np.column_stack((np.arange(len(angles)),original_angles,angles,angles-original_angles)),
            delimiter=',',header='plane,original_tilt_deg,refined_tilt_deg,change_deg',comments='')
        report.update(options=options,binning=options['binning'],original_axis_deg=origin_axis,
            original_angles_deg=original_angles,refined_angles_deg=angles,coarse_run=data['coarse_run'],
            source_orientation=data['orientation'],main_revision=data['main_revision'],
            ctf_policy='Saved CTF fits unchanged; refined angles affect registration/reconstruction geometry.',
            runtime_s=time.monotonic()-started,converged=report['stop_reason']=='stable')
        root=Path(__file__).resolve().parents[1]
        report['code_sha256']={name:hashlib.sha256((storage_path(root, name)).read_bytes()).hexdigest()
            for name in ('angular_joint.py','rigid_angle.py','web/angle_worker.py','web/refinement_worker.py','detector_defects.py','movie_replay.py')}
        report_path=storage_path(output, 'angle_report.json')
        report_path.write_text(json.dumps(clean_json(report),indent=2,allow_nan=False))
        result=dict(state='completed',kind='angle_refinement',display_name='Refine geometry',landmark_geometry=True,
            geometry_model='common_axis_per_tilt_inclination_xy_shared_landmark_xyz',
            reference_mode='leave_one_tilt_out',gpu=True,
            landmark_count=len(fields['positions_zyx']),supported_observations=int(fields['valid_views'].sum()),
            binning=options['binning'],geometry_binning=8,geometry_file=str(geometry),fit_geometry_file=str(geometry_source),
            report=str(report_path),angles_file=str(csv_path),axis_rotation_deg=axis,axis_change_deg=axis-origin_axis,
            max_tilt_change_deg=float(np.max(np.abs(angles-original_angles))),
            max_xyz_change=float(np.max(np.linalg.norm(fields['positions_zyx']-original_pos,axis=1))),
            passes=sum('pass_number' in h for h in report['history']),stop_reason=report['stop_reason'],
            runtime_s=report['runtime_s'],xyz_fixed=False,converged=report['converged'],
            tomogram=report.get('tomogram'),aligned_stack=report.get('aligned_stack',str(storage_path(Path(geometry_source).parent, f"aligned_bin{options['binning']}.mrcs"))))
        if not result['converged']:
            result['warning']='Finished without convergence: inspect the stop reason and the last accepted geometry.'
    except Exception as error:
        traceback.print_exc();ensure_output(output)
        result=dict(state='failed',kind='angle_refinement',error=str(error))
    temporary=storage_path(output, '.web_result.tmp')
    temporary.write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
    temporary.replace(storage_path(output, 'web_result.json'))
    return 1 if result['state']=='failed' else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().config))
