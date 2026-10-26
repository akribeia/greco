"""Launch Greco membrane inference in the shared GPU environment."""
from greco.project_layout import join as storage_path
import json

from pathlib import Path
import subprocess


def runtime_paths():
    from ..runtime import python,ASSETS
    return python('membranes'),storage_path(ASSETS, 'membranes/membrain-seg-quantized.pt')


def environment():
    from ..runtime import worker_environment
    return worker_environment('membranes')


def run(data, output):
    output = Path(output)
    config = dict(input=data['parent_volume'], output=str(output), checkpoint=data['membrane_checkpoint'])
    path = storage_path(output, 'membrane_config.json')
    path.write_text(json.dumps(config, indent=2))
    subprocess.run([data['membrane_python'], '-m', 'greco.web.membrane_worker', '--config', str(path)],
                   env=environment(), check=True)
    result = json.loads((storage_path(output, 'membranes.json')).read_text())
    result.update(geometry_file=data.get('geometry_file'), binning=data['binning'],
                  parent_volume=data['parent_volume'], color=data.get('options',{}).get('color','#f472b6'))
    return result
