"""Slurm entry for polishing-style rigid landmark geometry."""
from greco.project_layout import join as storage_path
import argparse,json,traceback
from pathlib import Path
from .refinement_worker import clean_json


def denoise_outputs(data,result):
    from .final_denoising import denoise_outputs as denoise
    denoise(data,result,geometry=True)


def run(config):
    data=json.loads(Path(config).read_text())
    try:
        from .refinement import validate_geometry_options
        data['options']=validate_geometry_options(data['options'])
        from .particle_worker import check_snapshot
        check_snapshot(data.get('snapshot',[]))
        from ..landmark_rigid_worker import run as fit
        result=fit(dict(data,defer_web_result=True))
        denoise_outputs(data,result)
        from .._tinycare.config import write_json
        write_json(storage_path(Path(data['output']), 'web_result.json'),clean_json(result))
        return 0
    except Exception as e:
        traceback.print_exc();out=Path(data['output']);out.mkdir(parents=True,exist_ok=True)
        (storage_path(out, 'web_result.json')).write_text(json.dumps(clean_json(dict(state='failed',kind='geometry_refinement',error=str(e))),allow_nan=False));return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);raise SystemExit(run(p.parse_args().config))
