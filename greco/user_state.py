"""Private user state shared by Greco instances and browser sessions."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError(f'Expected a directory owned by this user: {path}')
    path.chmod(0o700)
    return path


def user_directory():
    return private_directory(Path(os.environ.get('GRECO_USER_ROOT', '~/.greco')).expanduser().absolute())


class RecentProjects:
    LIMIT = 10

    def __init__(self):
        self.root = user_directory()

    @contextmanager
    def locked(self):
        descriptor = os.open(self.root / 'recent-projects.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _read(self):
        path = self.root / 'recent-projects.json'
        if not path.exists():
            return dict(version=1, projects=[])
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError(f'Invalid recent-project history: {path}')
        data = json.loads(path.read_text())
        if data.get('version') != 1 or not isinstance(data.get('projects'), list):
            raise ValueError('Unsupported recent-project history.')
        rows, seen = [], set()
        for row in data['projects']:
            if isinstance(row, dict) and isinstance(row.get('path'), str) and row['path'] not in seen:
                rows.append(dict(path=row['path'], name=str(row.get('name') or Path(row['path']).name)))
                seen.add(row['path'])
        return dict(data, projects=rows[:self.LIMIT])

    def list(self):
        with self.locked():
            rows = self._read()['projects']
        visible = []
        for row in rows:
            try:
                path = Path(row['path']).resolve()
                if path.is_dir() and os.access(path, os.R_OK | os.X_OK):
                    visible.append(dict(row, path=str(path)))
            except OSError:
                continue
        return visible

    def remember(self, path, name):
        self.add([dict(path=str(Path(path).resolve()), name=name)])

    def directory(self):
        with self.locked():
            directory = self._read().get('last_directory')
        if isinstance(directory, str):
            try:
                path = Path(directory).resolve()
                if path.is_dir() and os.access(path, os.R_OK | os.X_OK):
                    return path
            except OSError:
                pass
        return Path.home()

    def remember_directory(self, path):
        with self.locked():
            data = self._read()
            data['last_directory'] = str(Path(path).resolve())
            self._write(data)

    def _write(self, data):
        temporary = self.root / ('recent-' + secrets.token_hex(8))
        try:
            with temporary.open('x') as stream:
                temporary.chmod(0o600)
                json.dump(data, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.root / 'recent-projects.json')
        finally:
            temporary.unlink(missing_ok=True)

    def add(self, projects, *, append=False):
        """Merge entries in recency order, optionally appending imported projects."""
        with self.locked():
            data = self._read()
            previous = data['projects']
            incoming = [dict(path=str(Path(row['path']).resolve()), name=row['name']) for row in projects]
            seen, rows = set(), []
            for row in (previous + incoming if append else incoming + previous):
                if row['path'] not in seen:
                    seen.add(row['path'])
                    rows.append(row)
            data['projects'] = rows[:self.LIMIT]
            if not append and incoming:
                data['last_directory'] = incoming[0]['path']
            self._write(data)


class SlurmSettings:
    """Per-user cluster settings; atomic publication across server versions."""

    def __init__(self, root):
        self.root = private_directory(Path(root))

    def load(self):
        path = self.root / 'slurm.json'
        if not path.exists():
            return None
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError(f'Invalid Slurm settings: {path}')
        data = json.loads(path.read_text())
        if data.get('version') != 1 or not isinstance(data.get('settings'), dict):
            raise ValueError('Unsupported Slurm settings.')
        return data['settings']

    def save(self, settings, *, only_if_missing=False):
        descriptor = os.open(self.root / 'slurm.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if only_if_missing:
                current = self.load()
                if current is not None:
                    return current
            temporary = self.root / ('slurm-' + secrets.token_hex(8))
            try:
                with temporary.open('x') as stream:
                    temporary.chmod(0o600)
                    json.dump(dict(version=1, settings=settings), stream, indent=2, allow_nan=False)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(self.root / 'slurm.json')
            finally:
                temporary.unlink(missing_ok=True)
        return settings

