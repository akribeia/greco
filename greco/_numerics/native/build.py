"""Build the numerical core: python -m greco._numerics.native.build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def build(output=None):
    from ...runtime import RUNTIME
    source=Path(__file__).with_name('numerics.cpp')
    output=Path(output) if output else source.with_name('greco_numerics.so')
    output.parent.mkdir(parents=True,exist_ok=True)
    compiler=shlex.split(os.environ.get('CXX','c++'))
    flags=['-O3','-std=c++17','-fPIC','-shared','-fvisibility=hidden','-fno-fast-math','-ffp-contract=off','-pthread','-Wl,--as-needed']
    include=os.environ.get('GRECO_FFTW_INCLUDE')
    if include:flags.extend(['-I',include])
    libdir=Path(os.environ.get('GRECO_FFTW_LIBDIR',str(RUNTIME/'native-libs')))
    links=[str(libdir/name) if (libdir/name).is_file() else option for name,option in [('libfftw3.so.3','-lfftw3'),('libfftw3f.so.3','-lfftw3f')]]
    # The source bundle layout. Installed environments can use their loader paths.
    rpath='-Wl,-rpath,$ORIGIN/../../../runtime/native-libs'
    fd,tmp=tempfile.mkstemp(prefix='.greco-numerics-',suffix='.so',dir=output.parent)
    os.close(fd)
    try:
        command=[*compiler,*flags,str(source),*links,'-lm',rpath,'-o',tmp]
        subprocess.run(command,check=True,capture_output=True,text=True)
        os.chmod(tmp,0o755)
        os.replace(tmp,output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    version=subprocess.run([*compiler,'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    metadata={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'compiler':version,'flags':flags,'libraries':links,'binary_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'external_application_linked':False}
    output.with_suffix('.build.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:print(build(args.output))
    except subprocess.CalledProcessError as error:raise SystemExit(error.stderr or str(error))
