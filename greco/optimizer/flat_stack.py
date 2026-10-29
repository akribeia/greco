"""Reusable preprocessing product for beta-independent bin-8 alignment."""
import json
from pathlib import Path
import numpy as np
from ..project_layout import join

PROTOCOL='flat-ctf-native-tikhonov-even-odd-bin8-v1'


def signature(output,options):
    output=Path(output)
    files=[join(output,'greco_raw_bin8.mrcs'),join(output,'greco_raw_bin8.rawtlt'),join(output,'alignment/tilts.star')]
    for folder in sorted(join(output,'per_tilt').glob('tilt_*')):
        files.extend(join(folder,name) for name in ('ctf.json','shifts.npz','ctf_tiles.npy','ctf_tiles_metadata.npz','ctf_spectrum.npy','detector_correction.npz') if join(folder,name).is_file())
    return dict(options={key:options.get(key,default) for key,default in (('ctf_regularization',.05),('ctf_tile_size',512))},
                files=[[str(p.relative_to(output)),p.stat().st_size,p.stat().st_mtime_ns] for p in files])


def publish(output,stack,pixel_A,angles,options):
    import mrcfile
    output=Path(output);path=join(output,'ctf/flat_reliability_bin8.mrcs');path.parent.mkdir(parents=True,exist_ok=True)
    data=np.asarray(stack,np.float32);angles=np.asarray(angles,float)
    if data.ndim!=3 or len(data)!=len(angles) or not np.isfinite(data).all():raise ValueError('Invalid flat alignment stack.')
    record=dict(protocol=PROTOCOL,shape=list(data.shape),pixel_size_A=float(pixel_A),angles_deg=angles.tolist(),source=signature(output,options))
    temporary=path.with_name('.'+path.name+'.tmp')
    with mrcfile.new(temporary,overwrite=True) as handle:
        handle.set_data(data);handle.set_image_stack();handle.voxel_size=pixel_A;handle.update_header_stats()
    temporary.replace(path)
    record['stack_stamp']=[path.stat().st_size,path.stat().st_mtime_ns]
    metadata=path.with_suffix('.json');temporary=metadata.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(metadata)
    return path


def reusable(output,options,angles):
    import mrcfile
    output=Path(output);path=join(output,'ctf/flat_reliability_bin8.mrcs')
    try:
        record=json.loads(path.with_suffix('.json').read_text());expected=np.asarray(angles,float)
        saved=np.asarray(record['angles_deg'],float)
        if record['protocol']!=PROTOCOL or saved.shape!=expected.shape or not np.allclose(saved,expected,rtol=0,atol=1e-4):return None
        if record['source']!=signature(output,options):return None
        if record['stack_stamp']!=[path.stat().st_size,path.stat().st_mtime_ns]:return None
        with mrcfile.open(path,header_only=True) as handle:
            if list(map(int,(handle.header.nz,handle.header.ny,handle.header.nx)))!=record['shape']:return None
        return path
    except (OSError,ValueError,KeyError,TypeError):return None
