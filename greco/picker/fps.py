"""FPS orientation subset: packaged canonical sequence + persistent cache.

The canonical FPS sequence for the 10-degree / symmetry-1 bank
(15552 rotations) is deterministic and a function of the angle list
ONLY (SHA-pinned by the research audit, Round-6F.0).  Rebuilding it
costs ~tens of seconds of CPU, so the production behaviour is:

  * CANONICAL bank (10 deg, symmetry 1): load the packaged id sequence
    (``_data/fps_canonical_15552.npz``) and verify its SHA-256;
  * NON-CANONICAL bank: compute ``geometry.fps_subset(angle_list, k)``
    and cache it in a persistent directory keyed by
    ``sha256(angle-list bytes + algorithm version)``.

Every resolution path records ``fps_k``, ``fps_ids_sha256``,
``angle_list_sha256`` and ``fps_source`` (canonical / cache / computed)
for the job JSON.
"""

from __future__ import annotations
from greco.project_layout import join as storage_path

import hashlib
import os
import pathlib

import numpy as np

from .geometry import fps_subset

__all__ = [
    "FPS_ALGO_VERSION",
    "CANONICAL_ANGLE_LIST_SHA256",
    "get_fps_ids",
    "default_fps_cache_dir",
]

#: algorithm version part of the non-canonical cache key
FPS_ALGO_VERSION = "fps-v1-2026-09"

#: SHA-256 of the canonical angle list (10 deg, sort, symmetry 1;
#: float64 bytes of the 15552 ZXZ-radian tuples)
CANONICAL_ANGLE_LIST_SHA256 = (
    "6c40e9236aab13289c1c1764cf13ca95baa00a5342c9ebe80077442a09e98eb9"
)

#: SHA-256 of the int64 id arrays of the canonical 7776 FPS sequence
# (and of its 2592-id prefix) - pinned by the research audit
_SHA_7776 = "e7d4ba02f66afa3b7bbf4d0a1778047b79189862b804a147196c3e410998fa91"
_SHA_2592 = "831d3da4e70c8285ad8d6a1d326098810b9260e532ad0976fb2b379e80e5c255"

_CANONICAL_PATH = (
    storage_path(storage_path(pathlib.Path(__file__).parent, "_data"), "fps_canonical_15552.npz")
)


def _sha_ids(ids) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes()
    ).hexdigest()


def default_fps_cache_dir() -> pathlib.Path:
    root = os.environ.get("GRECO_PICKER_FPS_CACHE")
    if root:
        p = pathlib.Path(root)
    else:
        p = storage_path(storage_path(storage_path(pathlib.Path.home(), ".cache"), "greco-picker"), "fps")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_canonical() -> np.ndarray:
    """Load the packaged canonical 7776-id sequence, verifying its
    SHA-256 (and the 2592 prefix) before use."""
    d = np.load(_CANONICAL_PATH)
    ids = d["ids"].astype(np.int64)
    if _sha_ids(ids) != _SHA_7776:
        raise RuntimeError(
            f"packaged canonical FPS sequence corrupted "
            f"({_sha_ids(ids)} != {_SHA_7776})"
        )
    if _sha_ids(ids[:2592]) != _SHA_2592:
        raise RuntimeError("packaged canonical FPS 2592 prefix corrupted")
    return ids


def get_fps_ids(angle_list, k: int,
                cache_dir: pathlib.Path | None = None):
    """Resolve the FPS subset ids for ``angle_list`` (size m, k <= m).

    Returns ``(ids, info)`` with ``info`` = {k, ids_sha256,
    angle_list_sha256, source}.  ``source`` is "canonical" (packaged,
    SHA-verified), "cache" (persistent file) or "computed".
    """
    k = int(k)
    angle_arr = np.ascontiguousarray(np.asarray(angle_list, dtype=np.float64))
    m = len(angle_arr)
    if k <= 0 or k > m:
        raise ValueError(f"k={k} out of range for a {m}-rotation bank")
    angle_sha = hashlib.sha256(angle_arr.tobytes()).hexdigest()

    if angle_sha == CANONICAL_ANGLE_LIST_SHA256:
        # the universal constant: never rebuilt, loaded + verified
        ids = _load_canonical()[:k]
        return np.ascontiguousarray(ids, dtype=np.int64), {
            "k": k,
            "ids_sha256": _sha_ids(ids),
            "angle_list_sha256": angle_sha,
            "source": "canonical",
        }

    key = hashlib.sha256(
        angle_arr.tobytes() + b"|" + FPS_ALGO_VERSION.encode()
        + b"|" + str(k).encode()
    ).hexdigest()
    cdir = cache_dir or default_fps_cache_dir()
    path = storage_path(cdir, f"fps_{key}.npz")
    if path.exists():
        d = np.load(path)
        if (str(d["angle_list_sha256"]) != angle_sha
                or int(d["k"]) != k
                or str(d["ids_sha256"]) != _sha_ids(d["ids"])):
            raise RuntimeError(f"FPS cache entry {path} does not match "
                               f"its key (corrupted); delete it")
        ids = d["ids"].astype(np.int64)
        return ids, {
            "k": k,
            "ids_sha256": _sha_ids(ids),
            "angle_list_sha256": angle_sha,
            "source": "cache",
        }
    ids = fps_subset(angle_list, k)
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.ascontiguousarray(ids, dtype=np.int64),
             angle_list_sha256=angle_sha, k=np.int64(k),
             ids_sha256=_sha_ids(ids))
    return np.ascontiguousarray(ids, dtype=np.int64), {
        "k": k,
        "ids_sha256": _sha_ids(ids),
        "angle_list_sha256": angle_sha,
        "source": "computed",
    }
