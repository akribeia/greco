"""Bounded, copy-isolated metadata cache; changed/atomic-replaced files reload."""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
from threading import RLock


class MetadataCache:
    def __init__(self, limit=512):
        self.limit = limit
        self.entries = OrderedDict()
        self.lock = RLock()
        self.hits = self.misses = 0

    @staticmethod
    def stamp(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def read(self, path, loader, *, decode=deepcopy):
        path = Path(path)
        with self.lock:
            try:
                stamp = self.stamp(path)
            except OSError:
                self.entries.pop(path, None)
                raise
            cached = self.entries.get(path)
            if cached is not None and cached[0] == stamp:
                self.hits += 1
                self.entries.move_to_end(path)
                return cached[2](cached[1])
            value = loader(path)
            self.misses += 1
            self.entries[path] = (stamp, deepcopy(value), decode)
            self.entries.move_to_end(path)
            while len(self.entries) > self.limit:
                self.entries.popitem(last=False)
            return value if decode is deepcopy else decode(value)

    def remember_json(self, path, value):
        path = Path(path)
        with self.lock:
            self.entries[path] = (self.stamp(path), json.dumps(value, allow_nan=False).encode(), json.loads)
            self.entries.move_to_end(path)
            while len(self.entries) > self.limit:
                self.entries.popitem(last=False)

    def previous(self, path):
        """Last read value, used only to avoid rewriting unchanged exports."""
        with self.lock:
            item = self.entries.get(Path(path))
            return item[2](item[1]) if item is not None else None


def read_json(service, path):
    loader = lambda p: json.loads(p.read_text())
    cache = getattr(service, 'metadata_cache', None)
    # Keep immutable JSON bytes and parse in C for each isolated view. Deepcopy
    # of large Python object graphs costs more than parsing their cached JSON.
    return cache.read(path, lambda p: p.read_bytes(), decode=json.loads) if cache is not None else loader(Path(path))


def read_scientific_metadata(service, path):
    from ..sta_layout import read_metadata
    cache = getattr(service, 'metadata_cache', None)
    return cache.read(path, read_metadata) if cache is not None else read_metadata(path)
