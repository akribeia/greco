"""Install a Greco release with Python 3.11."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request,urlopen

LIMIT=256*1024*1024


@contextmanager
def installation_workspace(prefix):
    """Keep all bootstrap build temporaries on the installation filesystem."""
    prefix.mkdir(mode=0o755,parents=True,exist_ok=True)
    prefix.chmod(prefix.stat().st_mode|0o555)
    previous_umask=os.umask(0o022)
    previous={name:os.environ.get(name) for name in ('TMPDIR','TMP','TEMP')}
    previous_tempdir=tempfile.tempdir
    try:
        with tempfile.TemporaryDirectory(prefix='.bootstrap-',dir=prefix) as temporary:
            os.environ.update({name:temporary for name in previous})
            tempfile.tempdir=None
            yield Path(temporary)
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        tempfile.tempdir=previous_tempdir
        os.umask(previous_umask)


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


def extract(path,destination):
    if path.stat().st_size>LIMIT:raise ValueError('Archive exceeds 256 MB.')
    with tarfile.open(path,'r:gz') as archive:
        members=[];total=0
        for member in archive:
            total+=member.size
            if len(members)>=20000 or total>4*LIMIT:raise ValueError('Archive too large.')
            if Path(member.name).is_absolute() or '..' in Path(member.name).parts or not(member.isfile() or member.isdir()):raise ValueError('Unsafe archive.')
            members.append(member)
        for member in members:
            target=destination/member.name
            if member.isdir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    roots=[destination] if (destination/'pyproject.toml').is_file() else list(destination.iterdir())
    if len(roots)!=1 or not (roots[0]/'greco/updates.py').is_file():raise ValueError('Invalid Greco source tree.')
    return roots[0]


def remote_archive(repository,version,destination):
    endpoint='tags/v'+version if version else 'latest'
    headers={'User-Agent':'Greco-installer','Accept':'application/vnd.github+json'}
    with urlopen(Request('https://api.github.com/repos/'+repository+'/releases/'+endpoint,headers=headers),timeout=30) as response:
        data=json.load(response)
    found=str(data['tag_name']).removeprefix('v')
    if data.get('draft') or data.get('prerelease') or not re.fullmatch(r'\d+\.\d+(?:\.\d+)?',found):raise ValueError('Select a published stable release.')
    if version and version!=found:raise ValueError('Release version mismatch.')
    asset=next((a for a in data['assets'] if a['name']=='greco-'+found+'.tar.gz'),None)
    if not asset:raise ValueError('Release has no Greco installation archive.')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',asset.get('digest') or ''):raise ValueError('Archive SHA-256 is missing.')
    url=asset['browser_download_url']
    if not url.startswith('https://github.com/'+repository+'/releases/download/v'+found+'/'):raise ValueError('Unexpected archive URL.')
    size=0
    with urlopen(Request(url,headers=headers),timeout=30) as response,destination.open('xb') as output:
        while chunk:=response.read(1024*1024):
            size+=len(chunk)
            if size>LIMIT:raise ValueError('Archive exceeds 256 MB.')
            output.write(chunk)
    if checksum(destination)!=asset['digest'][7:] or size!=asset['size']:raise ValueError('Archive checksum or size mismatch.')
    return found


def write_launcher(release,base,bindir,repository):
    bindir.mkdir(parents=True,exist_ok=True)
    launcher=bindir/'greco'
    legacy=base/'.greco-launcher'
    marker='export GRECO_INSTALL_ROOT='+shlex.quote(str(base))+'\n'
    if launcher.exists() or launcher.is_symlink():
        managed_link=launcher.is_symlink() and launcher.resolve()==legacy.resolve()
        managed_script=(not launcher.is_symlink() and launcher.is_file()
                        and marker in launcher.read_text() and '-m greco.runtime "$@"' in launcher.read_text())
        if not (managed_link or managed_script):
            raise ValueError('Command already exists and is not this installation\'s launcher: '+str(launcher))
    text='#!/usr/bin/env bash\nset -euo pipefail\n'
    for name,value in dict(GRECO_INSTALL_ROOT=str(base),GRECO_RELEASE_REPOSITORY=repository,
                           GRECO_RUNTIME_ROOT=release['runtime'],GRECO_ASSETS_ROOT=release['assets'],
                           PYTHONPATH=release['root'],PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1').items():
        text+='export '+name+'='+shlex.quote(value)+'\n'
    text+='exec '+shlex.quote(release['python'])+' -m greco.runtime "$@"\n'
    temporary=bindir/('greco-'+os.urandom(6).hex()+'.tmp')
    try:
        with temporary.open('x') as stream:stream.write(text)
        temporary.chmod(0o755);temporary.replace(launcher)
    finally:temporary.unlink(missing_ok=True)
    if legacy.is_file() and not legacy.is_symlink() and marker in legacy.read_text():
        legacy.unlink()
    command=launcher
    rule='='*72
    print('\n'+rule)
    print('  GRECO '+release['version']+' - INSTALLATION COMPLETE')
    print(rule)
    print('\n  START GRECO\n')
    print('    '+shlex.quote(str(command))+' start')
    path_line='export PATH='+shlex.quote(str(bindir))+':"$PATH"'
    print('\n  ADD GRECO TO PATH\n')
    print('    '+path_line)
    print('\n  Individual: add this line to your ~/.bashrc.')
    print("  Shared: add it to the cluster's shared shell profile.")
    print('\n  Then run:\n')
    print('    greco start\n    greco status')
    print('\n  Each user keeps their own settings and socket in ~/.greco.')
    print('\n'+rule+'\n',flush=True)


def main():
    if sys.version_info[:2]!=(3,11):raise SystemExit('Run this installer with Python 3.11.')
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--source',type=Path,help='Install a local source directory without GitHub')
    group.add_argument('--archive',type=Path,help='Install a local release archive without GitHub')
    parser.add_argument('--sha256',help='Expected archive digest (otherwise read <archive>.sha256)')
    parser.add_argument('--prefix',type=Path,default=Path.cwd(),help='Installation directory (default: current working directory)')
    parser.add_argument('--bin-dir',type=Path,help='Command directory (default: <installation>/bin)')
    parser.add_argument('--runtime',type=Path,help='Reuse a compatible existing runtime instead of downloading dependencies')
    parser.add_argument('--version',help='Expected local version, or published release to download')
    parser.add_argument('--repository',default='akribeia/greco')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repository):parser.error('Invalid repository.')
    if args.version and not re.fullmatch(r'\d+\.\d+(?:\.\d+)?',args.version):parser.error('Invalid version.')
    prefix=args.prefix.expanduser().resolve()
    print('Installing Greco in '+str(prefix),flush=True)
    os.environ['GRECO_INSTALL_ROOT']=str(prefix)
    bin_dir=args.bin_dir.expanduser().resolve() if args.bin_dir else prefix/'bin'
    if args.runtime:
        runtime=args.runtime.expanduser().resolve()
        if not (runtime/'gpu/bin/python').is_file():parser.error('--runtime must contain gpu/bin/python.')
        os.environ['GRECO_RUNTIME_ROOT']=str(runtime)
        os.environ['GRECO_MAIN_PYTHON']=str(runtime/'gpu/bin/python')
    source=args.source.expanduser().resolve() if args.source else None
    with installation_workspace(prefix) as work:
        version=args.version
        if not source:
            if args.archive:
                archive=args.archive.expanduser().resolve()
                print('Verifying archive checksum',flush=True)
                digest=args.sha256
                if digest is None:
                    companion=Path(str(archive)+'.sha256')
                    if not companion.is_file():raise ValueError('Copy '+companion.name+' alongside the archive, or provide --sha256.')
                    fields=companion.read_text().split()
                    if not fields:raise ValueError('Archive checksum file is empty.')
                    digest=fields[0]
                if not re.fullmatch(r'[0-9a-fA-F]{64}',digest) or checksum(archive)!=digest.lower():raise ValueError('Local archive checksum mismatch.')
            else:
                archive=work/'release.tar.gz'
                print('Downloading Greco release',flush=True)
                version=remote_archive(args.repository,version,archive)
            print('Extracting Greco source and bundled assets',flush=True)
            source=extract(archive,work/'source')
        if not (source/'greco/updates.py').is_file():raise ValueError('Select a Greco source directory.')
        sys.path.insert(0,str(source))
        os.environ['GRECO_RELEASE_REPOSITORY']=args.repository
        os.environ['GRECO_BOOTSTRAP']='1'
        os.environ['GRECO_INSTALL_PROGRESS']='1'
        from greco.updates import install_local,select_release
        result=install_local(source,lambda message:print(message,flush=True),expected_version=version)
        release=select_release(result['version'])
        write_launcher(release,prefix,bin_dir,args.repository)


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        raise SystemExit('Greco installation failed: '+str(error)+'\nSee install.log in the installation directory.')
