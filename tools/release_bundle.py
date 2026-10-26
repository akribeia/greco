"""Build a verified installer archive for local use or a tagged GitHub release."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib


def build(source,output,private=False):
    sys.path.insert(0,str(source))
    from greco.updates import version_key,source_files,source_digest,bundled_model
    version=tomllib.loads((source/'pyproject.toml').read_text())['project']['version']
    version_key(version)
    if not private:
        if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
            raise ValueError('Commit your changes before preparing a public release; use --private for a local bundle.')
        tag=subprocess.check_output(['git','describe','--exact-match','--tags','HEAD'],cwd=source,text=True).strip()
        if tag!='v'+version:raise ValueError('Tag HEAD as v'+version+' first.')
    files=list(source_files(source))
    if not private:
        tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=source).decode().split('\0'))
        files=[path for path in files if str(path) in tracked]
    model=bundled_model(source)
    if model is None:raise ValueError('Release needs assets/membranes/membrain-seg-quantized.pt.')
    if not private and str(model.relative_to(source)) not in tracked:
        raise ValueError('Track the verified membrane checkpoint before building the release.')
    output.mkdir(parents=True,exist_ok=True)
    name='greco-'+version+('-private' if private else '')+'.tar.gz'
    target=output/name
    with tempfile.TemporaryDirectory(prefix='.bundle-',dir=output) as temporary:
        stage=Path(temporary);root=stage/('greco-'+version);root.mkdir()
        for path in files:
            destination=root/path;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/path,destination)
        destination=root/model.relative_to(source);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(model,destination);bundled_model(root)
        identity=source_digest(root)
        manifest=dict(format='greco-source-bundle',version=version,private=private,
                      source_sha256=identity,created_at=datetime.now(timezone.utc).isoformat(),
                      dependencies_bundled=False,membrane_model_bundled=True)
        (root/'greco-bundle.json').write_text(json.dumps(manifest,indent=2)+'\n')
        archive=stage/name
        with tarfile.open(archive,'w:gz',compresslevel=6) as stream:
            stream.add(root,arcname=root.name)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.replace(target)
    (output/(name+'.sha256')).write_text(digest+'  '+name+'\n')
    shutil.copy2(source/'install.py',output/'install.py')
    print(target)
    print('SHA-256: '+digest)
    print('Size: '+str(target.stat().st_size)+' bytes')
    print('Local bundle; nothing uploaded.' if private else
          'Upload install.py, '+name+' and '+name+'.sha256 to GitHub release v'+version+'.')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('dist'))
    parser.add_argument('--private',action='store_true',help='Package current local files and model without committing, tagging or uploading')
    args=parser.parse_args()
    try:build(Path(__file__).resolve().parents[1],args.output.resolve(),args.private)
    except (ValueError,OSError,subprocess.SubprocessError) as error:raise SystemExit(str(error))
