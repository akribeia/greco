"""Verified GitHub releases installed beside running code, never over it."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from . import __version__
from . import runtime
from .user_state import user_directory

REPOSITORY = 'akribeia/greco'
DAY = 86400
MAX_ARCHIVE = 256 * 1024 * 1024


class UpdateError(ValueError):
    pass


def installation_directory():
    """Shared code/dependencies; personal state remains in user_directory()."""
    value=os.environ.get('GRECO_INSTALL_ROOT')
    if value:
        root=Path(value).expanduser().resolve()
    elif runtime.ROOT.parent.name=='releases' and (runtime.ROOT/'.greco-release.json').is_file():
        root=runtime.ROOT.parent.parent
    else:
        return user_directory()
    root.mkdir(mode=0o755,parents=True,exist_ok=True)
    return root


def installation_folder(name):
    path=installation_directory()/name
    path.mkdir(mode=0o755,parents=True,exist_ok=True)
    return path


@contextmanager
def installation_lock():
    root=installation_directory()
    if not os.access(root,os.W_OK):
        raise UpdateError('This installation is read-only for your account. Ask its owner to install updates.')
    hidden=root/'.installation.lock'
    descriptor=os.open(hidden,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'w') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise UpdateError('Another installation operation is in progress.') from error
        yield


def version_key(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d+\.\d+(?:\.\d+)?', value):
        raise UpdateError('Use a stable version such as 1.0 or 1.0.1.')
    return tuple(map(int, value.split('.'))) + (0,) * (3-len(value.split('.')))


def repository():
    value = os.environ.get('GRECO_RELEASE_REPOSITORY', REPOSITORY)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value):
        raise UpdateError('Invalid GitHub release repository.')
    return value


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.' + os.urandom(6).hex())
    try:
        with temporary.open('x') as stream:
            os.chmod(temporary, 0o600)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def update_lock():
    path = user_directory() / 'updates.lock'
    with os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), 'w') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise UpdateError('Another update operation is in progress.') from error
        yield stream.fileno()


def fetch_json(url):
    request = Request(url, headers={'Accept':'application/vnd.github+json', 'User-Agent':'Greco/'+__version__})
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
        if len(payload) > 2 * 1024 * 1024:
            raise UpdateError('Release metadata is too large.')
        return json.loads(payload)
    except HTTPError as error:
        if error.code == 404:
            raise UpdateError('No published release found in '+repository()+'.') from error
        raise UpdateError('GitHub release check failed (HTTP '+str(error.code)+').') from error
    except (URLError, TimeoutError) as error:
        raise UpdateError('Could not reach GitHub. Try again later.') from error


def release_info(version=None):
    repo = repository()
    if version is not None:
        version_key(version)
    endpoint = 'latest' if version is None else 'tags/v'+quote(version, safe='')
    data = fetch_json('https://api.github.com/repos/'+repo+'/releases/'+endpoint)
    if data.get('draft') or data.get('prerelease'):
        raise UpdateError('Only published stable releases can be installed.')
    found = str(data.get('tag_name', '')).removeprefix('v')
    version_key(found)
    if version is not None and version != found:
        raise UpdateError('Release tag does not match the requested version.')
    asset = next((a for a in data.get('assets', []) if a.get('name') == 'greco-'+found+'.tar.gz'), None)
    info = dict(version=found, notes=str(data.get('body') or '')[:50000],
                url='https://github.com/'+repo+'/releases/tag/v'+found)
    if asset:
        url = asset.get('browser_download_url', '')
        digest = asset.get('digest', '')
        if not url.startswith('https://github.com/'+repo+'/releases/download/v'+found+'/'):
            raise UpdateError('Unexpected release download URL.')
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest or ''):
            raise UpdateError('The release archive has no SHA-256 digest.')
        info.update(archive=url, sha256=digest.split(':',1)[1], size=asset.get('size'))
    return info


def environment_key(source):
    project = tomllib.loads((source/'pyproject.toml').read_text())['project']
    locks = {p.name:p.read_text() for p in sorted((source/'runtime-locks').glob('*.txt'))}
    value = dict(python=list(sys.version_info[:2]), dependencies=project.get('dependencies',[]), locks=locks)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:20]


def current_release():
    return dict(version=__version__, root=str(runtime.ROOT.resolve()), runtime=str(runtime.RUNTIME.resolve()),
                python=str(runtime.python().absolute()), assets=str(runtime.ASSETS.resolve()))


def register_current():
    """Remember the CLI's original source tree for explicit rollback from the web."""
    path=user_directory()/'installations.json'
    try:rows=json.loads(path.read_text())
    except (OSError,ValueError):rows=[]
    current=current_release()
    rows=[row for row in rows if row.get('root')!=current['root']]
    atomic_json(path,[current,*rows])


def installed():
    rows = [] if os.environ.get('GRECO_BOOTSTRAP')=='1' else [current_release()]
    try:
        known=[] if os.environ.get('GRECO_BOOTSTRAP')=='1' else json.loads((user_directory()/'installations.json').read_text())
    except (OSError,ValueError):known=[]
    for row in known:
        try:
            version_key(row['version'])
            source=Path(row['root'])
            if (source/'greco/launcher.py').is_file() and (source/'pyproject.toml').is_file() and Path(row['python']).is_file():
                if tomllib.loads((source/'pyproject.toml').read_text())['project']['version']==row['version']:
                    rows.append(row)
        except (OSError,ValueError,KeyError):continue
    directory = installation_directory()/'releases'
    if directory.is_dir():
        for path in directory.iterdir():
            metadata = path/'.greco-release.json'
            if path.is_symlink() or not metadata.is_file():
                continue
            try:
                row = json.loads(metadata.read_text())
                version_key(row['version'])
                if row['version'] != path.name or Path(row['root']).resolve() != path.resolve():
                    continue
                if not (path/'greco/launcher.py').is_file() or not Path(row['python']).is_file():
                    continue
                rows.append(row)
            except (ValueError, KeyError, OSError):
                continue
    rows=list({row['root']:row for row in rows}.values())
    return sorted(rows, key=lambda row:(version_key(row['version']),Path(row['root']).parent==directory), reverse=True)


def select_release(version=None):
    if version is not None:
        version_key(version)
    rows = installed()
    return next((row for row in rows if version is None or row['version']==version), None)


def release_environment(row):
    # No inherited interpreter overrides may silently change the selected release.
    env = {k:v for k,v in os.environ.items()
           if not (k.startswith('GRECO_') and (k.endswith('_PYTHON') or k in ('GRECO_PYTHON','GRECO_CUDA_PATH')))
           and k not in ('PYTHONHOME','CUDA_PATH','CUDA_HOME','LD_LIBRARY_PATH','GRECO_BOOTSTRAP')}
    env.update(PYTHONPATH=row['root'], GRECO_RUNTIME_ROOT=row['runtime'], GRECO_ASSETS_ROOT=row['assets'],
               GRECO_MAIN_PYTHON=row['python'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    root=Path(row['root'])
    if root.parent.name=='releases':env['GRECO_INSTALL_ROOT']=str(root.parent.parent)
    base=Path(row['runtime'])
    overlay=base/'cupy-overlay'
    if overlay.is_dir():env['PYTHONPATH']+=os.pathsep+str(overlay)
    cuda=base/'cuda'
    if cuda.is_dir():env.update(CUDA_PATH=str(cuda),CUDA_HOME=str(cuda))
    libraries=[base/'gpu/lib',base/'native-libs',cuda/'lib64']
    libraries.extend(sorted(base.glob('gpu/lib/python*/site-packages/nvidia/*/lib')))
    env['LD_LIBRARY_PATH']=os.pathsep.join(str(p) for p in libraries if p.is_dir())
    return env


def status(check=False):
    path = user_directory()/'updates.json'
    try:
        cached = json.loads(path.read_text())
    except (OSError,ValueError):
        cached = {}
    stale = cached.get('repository') != repository() or time.time()-cached.get('checked_at',0) >= DAY
    if check or stale:
        try:
            with update_lock():
                try:
                    latest = release_info()
                    cached = dict(repository=repository(), checked_at=time.time(), latest=latest)
                except (UpdateError, OSError, ValueError) as error:
                    cached = dict(repository=repository(), checked_at=time.time(), error=str(error))
                atomic_json(path,cached)
        except UpdateError:
            # An installation owns the lock; serve cached metadata and its progress.
            pass
    rows = installed()
    latest = cached.get('latest')
    return dict(current=__version__, installed=list(dict.fromkeys(r['version'] for r in rows)), latest=latest,
                available=bool(latest and version_key(latest['version'])>version_key(__version__)),
                repository=repository(), checked_at=cached.get('checked_at'), error=cached.get('error'),
                operation=operation_status())


def extract_archive(archive, destination):
    total = 0
    with tarfile.open(archive, 'r:gz') as bundle:
        members=[]
        for member in bundle:
            if len(members)>=20000:
                raise UpdateError('Too many files in release archive.')
            members.append(member)
            path = Path(member.name)
            total += member.size
            if (path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir())
                    or total > MAX_ARCHIVE*4):
                raise UpdateError('Unsafe release archive.')
        # Links, devices and traversal were rejected above; extract regular files only.
        for member in members:
            target = destination/member.name
            if member.isdir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with bundle.extractfile(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src,dst)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    roots = [destination] if (destination/'pyproject.toml').is_file() else list(destination.iterdir())
    if len(roots)!=1 or not (roots[0]/'greco/launcher.py').is_file():
        raise UpdateError('Release must contain one Greco source tree.')
    return roots[0]


def download(info, destination):
    digest = hashlib.sha256()
    size = 0
    with urlopen(Request(info['archive'],headers={'User-Agent':'Greco/'+__version__}),timeout=30) as response, destination.open('xb') as stream:
        if urlsplit(response.url).scheme != 'https':
            raise UpdateError('Release download must use HTTPS.')
        while chunk := response.read(1024*1024):
            size += len(chunk)
            if size > MAX_ARCHIVE:
                raise UpdateError('Release archive exceeds 256 MB.')
            digest.update(chunk); stream.write(chunk)
    if digest.hexdigest()!=info['sha256'] or (info.get('size') is not None and size!=info['size']):
        raise UpdateError('Release archive checksum or size does not match GitHub.')


def environment_matches(source,row):
    requirements=list(tomllib.loads((source/'pyproject.toml').read_text())['project'].get('dependencies',[]))
    for path in sorted((source/'runtime-locks').glob('*.txt')):
        requirements.extend(line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#'))
    code="""import importlib.metadata as m, json, sys
from packaging.requirements import Requirement
for text in json.loads(sys.argv[1]):
 r=Requirement(text)
 if r.marker is None or r.marker.evaluate():
  if m.version(r.name) not in r.specifier: raise SystemExit(1)
"""
    try:
        result=subprocess.run([row['python'],'-c',code,json.dumps(requirements)],
                              env=release_environment(row),cwd=row['root'],
                              capture_output=True,text=True,timeout=30)
        return result.returncode==0
    except (OSError,subprocess.SubprocessError):return False


def installation_command(command, *, log, **kwargs):
    """Keep an installation log and stream foreground installer output."""
    if os.environ.get('GRECO_INSTALL_PROGRESS')!='1':
        return subprocess.run(command,check=True,stdout=log,stderr=log,**kwargs)
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,**kwargs) as process:
        while chunk:=process.stdout.read1(4096):
            log.write(chunk);log.flush()
            sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
        status=process.wait()
    if status:raise subprocess.CalledProcessError(status,command)


@contextmanager
def installation_temporary(prefix):
    """Remove dependency staging and its parent once no temporary work remains."""
    root=installation_folder('tmp')
    try:
        with tempfile.TemporaryDirectory(prefix=prefix,dir=root) as temporary:
            yield temporary
    finally:
        try:root.rmdir()
        except OSError:pass  # Preserve any other temporary work still using it.


def prepare_cudnn_aliases(environment):
    """Keep cuDNN's exact-version dlopen requests inside the pinned runtime.

    NVIDIA wheels ship .so.9 files, but cuDNN first requests .so.9.x.y.
    Without aliases it can select a same-version CUDA 11 component from the
    system before falling back to the wheel's CUDA 12 .so.9 file.
    Only the installer writes these aliases; shared runtime users need no
    write access. Existing native libraries are never overwritten.
    """
    code = ('import json; from importlib.metadata import distribution; '
            'd=distribution("nvidia-cudnn-cu12"); '
            'print(json.dumps(dict(version=d.version,lib=str(d.locate_file("nvidia/cudnn/lib")))))')
    result = subprocess.run([environment['python'], '-I', '-c', code],
                            check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    parts = metadata['version'].split('.')[:3]
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise UpdateError('Invalid installed cuDNN version.')
    version = '.'.join(parts)
    libraries = sorted(Path(metadata['lib']).glob('libcudnn*.so.' + parts[0]))
    if not libraries:
        raise UpdateError('Installed cuDNN wheel libraries are missing.')
    directory = Path(environment['runtime']) / 'native-libs'
    directory.mkdir(exist_ok=True)
    for library in libraries:
        alias = directory / (library.name.split('.so.')[0] + '.so.' + version)
        if alias.is_symlink() and alias.resolve() == library.resolve():
            continue
        if alias.exists() or alias.is_symlink():
            raise UpdateError('Conflicting cuDNN library alias: ' + str(alias))
        alias.symlink_to(os.path.relpath(library, directory))


def prepare_environment(source, progress):
    key = environment_key(source)
    # Existing installations with unchanged requirements share the tested runtime.
    candidates = [current_release(),*installed()]
    for row in candidates:
        if ((Path(row['runtime'])/'gpu/bin/python').is_file()
                and environment_key(Path(row['root']))==key and environment_matches(source,row)):
            return dict(runtime=row['runtime'],python=row['python'],assets=row['assets'])
    directory = installation_folder('environments')/key
    ready = directory/'ready.json'
    if ready.is_file():
        row=json.loads(ready.read_text())
        if environment_matches(source,dict(row,root=str(source))):return row
        raise UpdateError('The installed environment was modified. Restore its pinned dependencies before updating.')
    if directory.exists():
        shutil.rmtree(directory)
    with installation_temporary('pip-') as temporary:
        install_environment=dict(os.environ,TMPDIR=temporary,TMP=temporary,TEMP=temporary,
                                 PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
        progress('Preparing Python and downloading dependencies')
        subprocess.run([sys.executable,'-m','venv',str(directory/'gpu')],check=True,env=install_environment)
        executable = directory/'gpu/bin/python'
        requirements = source/'runtime-locks/gpu.txt'
        if not requirements.is_file():
            raise UpdateError('Release is missing runtime-locks/gpu.txt.')
        dependencies=tomllib.loads((source/'pyproject.toml').read_text())['project'].get('dependencies',[])
        torch=next((value for value in dependencies if value.startswith('torch==')),'')
        cuda=re.search(r'\+(cu[0-9]+)$',torch)
        if not cuda:raise UpdateError('Release must pin a CUDA PyTorch build.')
        index='https://download.pytorch.org/whl/'+cuda.group(1)
        with (installation_directory()/'install.log').open('ab') as log:
            for lock in (requirements,source/'runtime-locks/cupy.txt'):
                progress('Installing dependencies: '+lock.name)
                installation_command([str(executable),'-m','pip','install','--progress-bar','on','--extra-index-url',
                                index,'-r',str(lock)],
                               log=log,env=install_environment)
            progress('Installing Greco Python dependencies')
            installation_command([str(executable),'-m','pip','install','--progress-bar','on','--extra-index-url',index,*dependencies],
                           log=log,env=install_environment)
            progress('Checking installed dependencies')
            installation_command([str(executable),'-m','pip','check'],log=log,env=install_environment)
    assets=runtime.ASSETS.resolve() if runtime.ASSETS.is_dir() and os.environ.get('GRECO_BOOTSTRAP')!='1' else installation_directory()/'assets'
    row = dict(runtime=str(directory),python=str(executable),assets=str(assets))
    atomic_json(ready,row)
    return row


SOURCE_ROOTS=('greco','licenses','runtime-locks')
SOURCE_FILES=('README.md','LICENSE','COPYRIGHT','pyproject.toml','MANIFEST.in','install.py')


def source_files(source):
    for name in SOURCE_FILES:
        if (source/name).is_file():yield Path(name)
    for name in SOURCE_ROOTS:
        for path in sorted((source/name).rglob('*')):
            relative=path.relative_to(source)
            if path.is_symlink():raise UpdateError('Source contains a symbolic link: '+str(relative))
            if not path.is_file() or '__pycache__' in relative.parts:continue
            if path.suffix in ('.so','.pyc') or path.name.endswith('.build.json'):continue
            yield relative


def source_digest(source):
    digest=hashlib.sha256()
    for relative in source_files(source):
        digest.update(str(relative).encode()+b'\0')
        with (source/relative).open('rb') as stream:
            while chunk:=stream.read(1024*1024):digest.update(chunk)
        digest.update(b'\0')
    return digest.hexdigest()


def bundled_model(source):
    path=source/'assets/membranes/membrain-seg-quantized.pt'
    if not path.is_file():return None
    manifest=json.loads((source/'runtime-locks/membranes.json').read_text())
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    if path.stat().st_size!=manifest['bytes'] or digest.hexdigest()!=manifest['sha256']:
        raise UpdateError('Bundled membrane model does not match its manifest.')
    return path


def preflight_native(source, environment):
    """Fail before downloading Python dependencies if source builds cannot work."""
    compilers=['cc','gcc',*shlex.split(os.environ.get('CXX','c++'))[:1]]
    if any(shutil.which(name) is None for name in compilers):
        raise UpdateError('Source installation requires C/C++ build tools (cc, gcc and c++).')
    libdir=Path(os.environ.get('GRECO_FFTW_LIBDIR',str(Path(environment['runtime'])/'native-libs')))
    links=[str(libdir/name) if (libdir/name).is_file() else option for name,option in
           [('libfftw3.so.3','-lfftw3'),('libfftw3f.so.3','-lfftw3f')]]
    includes=['-I',os.environ['GRECO_FFTW_INCLUDE']] if os.environ.get('GRECO_FFTW_INCLUDE') else []
    code='#include <fftw3.h>\n#include <tiffio.h>\nint main(void){fftw_cleanup();fftwf_cleanup();return TIFFGetVersion()==0;}\n'
    with tempfile.TemporaryDirectory(prefix='greco-native-check-') as temporary:
        result=subprocess.run(['cc',*includes,'-x','c','-','-x','none',*links,'-ltiff','-o',str(Path(temporary)/'check')],
                              input=code,capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise UpdateError('Source installation requires FFTW (double/single precision) and libtiff development files. '+result.stderr[-2000:])


def publish_source(source,progress,expected_version=None):
    project=tomllib.loads((source/'pyproject.toml').read_text())['project']
    version=project.get('version');version_key(version)
    if project.get('name')!='greco' or (expected_version is not None and version!=expected_version):
        raise UpdateError('Archive version does not match the requested Greco release.')
    identity=source_digest(source)
    destination=installation_folder('releases')/version
    if destination.exists():
        metadata=destination/'.greco-release.json'
        if metadata.is_file() and json.loads(metadata.read_text()).get('source_sha256')==identity:
            return dict(version=version,installed=True)
        raise UpdateError('Greco '+version+' is already installed with different or unrecorded source. Use a new version; existing releases are never overwritten.')
    model=bundled_model(source)
    progress('Checking native build prerequisites')
    preflight_native(source,current_release())
    environment=prepare_environment(source,progress)
    prepare_cudnn_aliases(environment)
    if model:environment=dict(environment,assets=str(destination/'assets'))
    row=dict(version=version,root=str(destination),source_sha256=identity,**environment)
    progress('Compiling and checking Greco')
    env=release_environment(dict(row,root=str(source)))
    with (installation_directory()/'install.log').open('ab') as log:
        progress('Building numerical helpers')
        installation_command([row['python'],'-m','greco._numerics.native.build'],env=env,cwd=source,log=log)
        progress('Building EER decoder')
        installation_command(['gcc','-O3','-shared','-fPIC','-pthread',str(source/'greco/native/eer_decoder.c'),
                        '-o',str(source/'greco/native/eer_decoder.so'),'-ltiff'],
                       env=env,cwd=source,log=log)
        smoke="import greco; from greco.web.server import make_handler; import torch,cupy; from greco._numerics.native.loader import load; assert load() is not None; from greco.sta.native_lbfgs import library; library(); from greco.eer import _lib; _lib(); assert greco.__version__ == "+repr(version)
        progress('Building pose optimizer and checking Greco imports')
        installation_command([row['python'],'-c',smoke],env=env,cwd=source,log=log)
    atomic_json(source/'.greco-release.json',row)
    (source/'.greco-release.json').chmod(0o644)
    source.rename(destination)
    return dict(version=version,installed=True)


def install_local(source,progress=lambda message:None,expected_version=None):
    source=Path(source).expanduser().resolve()
    if not (source/'pyproject.toml').is_file() or not (source/'greco/launcher.py').is_file():
        raise UpdateError('Select a Greco source directory containing pyproject.toml and greco/.')
    with installation_lock():
        releases=installation_folder('releases')
        with tempfile.TemporaryDirectory(prefix='.install-',dir=releases) as temporary:
            stage=Path(temporary)/'source';stage.mkdir()
            progress('Preparing local Greco source')
            for relative in source_files(source):
                target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source/relative,target)
            model=bundled_model(source)
            if model:
                target=stage/model.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(model,target)
            return publish_source(stage,progress,expected_version)


def install(version=None, progress=lambda message:None, *, lock_fd=None):
    with (update_lock() if lock_fd is None else nullcontext(lock_fd)),installation_lock():
        info=release_info(version)
        if not info.get('archive'):
            raise UpdateError('Publish greco-'+info['version']+'.tar.gz as a release asset first.')
        releases=installation_folder('releases')
        with tempfile.TemporaryDirectory(prefix='.install-',dir=releases) as temporary:
            stage=Path(temporary)
            progress('Downloading Greco '+info['version'])
            download(info,stage/'release.tar.gz')
            source=extract_archive(stage/'release.tar.gz',stage/'source')
            return publish_source(source,progress,info['version'])


def operation_status():
    path = user_directory()/'update-operation.json'
    try:
        row = json.loads(path.read_text())
    except (OSError,ValueError):
        return dict(state='idle')
    if row.get('state')=='running':
        try:
            with update_lock():
                if time.time()-row.get('started_at',0)>30:
                    row.update(state='failed',message='Update process stopped. Retry the installation.')
        except UpdateError:
            pass
    return row


def start_install(version):
    version_key(version)
    with update_lock() as descriptor:
        atomic_json(user_directory()/'update-operation.json',dict(state='running',version=version,message='Starting download',started_at=time.time()))
        with (user_directory()/'install.log').open('ab') as log:
            subprocess.Popen([str(runtime.python()),'-m','greco.updates','_install',version,str(descriptor)],
                             stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                             env=runtime.worker_environment(),cwd=runtime.ROOT,start_new_session=True,pass_fds=(descriptor,))
    return dict(started=True,version=version)


def run_install(version,lock_fd):
    path = user_directory()/'update-operation.json'
    def progress(message):
        atomic_json(path,dict(state='running',version=version,message=message))
    try:
        result = install(version,progress,lock_fd=lock_fd)
        atomic_json(path,dict(state='installed',version=result['version'],message='Installed. Ready to restart.'))
    except Exception as error:
        atomic_json(path,dict(state='failed',version=version,message=str(error)+' See install.log.'))
        raise


def schedule_restart(version):
    from .launcher import instance_directory,read_state
    row=select_release(version)
    if row is None:raise UpdateError('Install this release before restarting.')
    state=read_state(instance_directory())
    if not state or state.get('pid')!=os.getpid():
        raise UpdateError('This server is not managed by greco start. Restart it from the terminal.')
    with (user_directory()/'install.log').open('ab') as log:
        subprocess.Popen([str(runtime.python()),'-m','greco.updates','_restart',version,state['instance']],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                         env=runtime.worker_environment(),cwd=runtime.ROOT,start_new_session=True)
    return dict(restarting=True,version=version)


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='_install':
        try:run_install(sys.argv[2],int(sys.argv[3]))
        finally:os.close(int(sys.argv[3]))
    elif len(sys.argv)==4 and sys.argv[1]=='_restart':
        time.sleep(1)
        raise SystemExit(subprocess.call([str(runtime.python()),'-m','greco.launcher','restart',
                                         '--version',sys.argv[2],'--expected-instance',sys.argv[3]],
                                        env=runtime.worker_environment(),cwd=runtime.ROOT))
    else:
        raise SystemExit('Use greco update.')
