"""MRC I/O and minimal STAR file writing."""

from __future__ import annotations

import pathlib

import mrcfile
import numpy as np


def read_mrc(path: pathlib.Path, transpose: bool = True) -> np.ndarray:
    """Read an MRC file as float32.

    MRC files store data zyx; like greco_picker_tm this module works in xyz, so the
    array is transposed on read (and back on write) by default.
    """
    with mrcfile.open(str(path), permissive=True) as mrc:
        data = mrc.data.astype(np.float32)
    return np.ascontiguousarray(data.T) if transpose else data


def write_mrc(path: pathlib.Path, data: np.ndarray, voxel_size: float,
              transpose: bool = True):
    path = pathlib.Path(path)
    payload = np.ascontiguousarray(data.astype(np.float32).T) if transpose\
        else data.astype(np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(payload)
        mrc.voxel_size = (voxel_size, voxel_size, voxel_size)


def read_mrc_meta_data(path: pathlib.Path) -> dict:
    with mrcfile.open(str(path), permissive=True, header_only=True) as mrc:
        shape = tuple(int(s) for s in
                      (mrc.header.nx, mrc.header.ny, mrc.header.nz))
        vs = float(mrc.voxel_size.x)
    return {"shape": shape, "voxel_size": vs}


from ..picking_star import COLUMNS as PICK_COLUMNS, read_particles, write_particles


def write_star(path, columns, rows, *, metadata=None):
    return write_particles(path, columns, rows, metadata=metadata)


def read_star(path):
    return read_particles(path)
