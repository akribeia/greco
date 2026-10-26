"""Identify shared application code without copying source into jobs."""
from datetime import datetime,timezone
from functools import lru_cache
import hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
from .runtime import ROOT

SOURCE_SUFFIXES={'.py','.js','.css','.html','.c','.cpp','.cu','.h','.toml','.sh'}

@lru_cache(maxsize=2048)
def _file_digest(path,device,inode,size,mtime,ctime):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root=None):
    """Hash the installed sources; attach Git metadata only for a real checkout."""
    root=Path(root).resolve() if root is not None else ROOT
    package=root/'greco'
    files=[p for p in package.rglob('*') if p.is_file() and p.suffix in SOURCE_SUFFIXES and '__pycache__' not in p.parts]
    files += [root/name for name in ('pyproject.toml',) if (root/name).is_file()]
    digest=hashlib.sha256()
    for path in sorted(files):
        stat=path.stat()
        value=_file_digest(str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        digest.update(str(path.relative_to(root)).encode()+b'\0'+bytes.fromhex(value))
    commit=None;dirty=None
    if (root/'.git').exists():
        def git(*args):
            result=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,timeout=10)
            return result.stdout.strip() if result.returncode==0 else None
        try:
            head=git('rev-parse','--verify','HEAD')
            if head and re.fullmatch('[0-9a-fA-F]{40,64}',head):commit=head
            status=git('status','--porcelain','--untracked-files=normal')
            if status is not None:dirty=bool(status)
        except (OSError,subprocess.TimeoutExpired):pass
    from . import __version__
    return dict(version=__version__,source_sha256=digest.hexdigest(),source_files=len(files),git_commit=commit,git_dirty=dirty)


def record_code(config,module,phase,*,exit_code=None):
    """Keep submission and execution identities in a small per-job JSON record."""
    config=Path(config)
    target=config.parent/'provenance'/'code.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    try:record=json.loads(target.read_text())
    except FileNotFoundError:record={}
    record.update(software='GRECO',format='greco-job-provenance',schema_version=1)
    entry=dict(source_identity(),recorded_at=datetime.now(timezone.utc).isoformat(),module=module)
    if exit_code is not None:entry['exit_code']=exit_code
    record[phase]=entry
    if 'started' in record:
        for name in ('submitted','finished'):
            if name in record:record[name+'_matches_started']=record[name]['source_sha256']==record['started']['source_sha256']
    fd,name=tempfile.mkstemp(prefix='code-',suffix='.tmp',dir=target.parent)
    try:
        with os.fdopen(fd,'w') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        os.replace(name,target)
    finally:Path(name).unlink(missing_ok=True)
    return record
