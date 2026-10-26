"""Local parallel-beam projection used by the validated bin8 PCC pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates


def project_local_volume_patch(
    volume_zyx: ArrayLike,
    point_xyz: ArrayLike,
    angle_deg: float,
    *,
    observation_radius: int,
    model_radius: int,
    model_sign: float = 1.0,
) -> NDArray[np.float32]:
    """Project a local cube in the aligned, vertical-axis volume frame.

    Volumes use MRC ``(z,y,x)`` storage and points use centred ``(x,y,z)``.
    The detector convention is ``u=x*cos(theta)-z*sin(theta)``.  This is a
    geometry primitive; it performs neither landmark selection nor CTF
    processing.
    """

    volume = np.asarray(volume_zyx, dtype=np.float64)
    point = np.asarray(point_xyz, dtype=np.float64)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("volume must be cubic")
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_xyz must contain three finite values")
    if model_sign not in (-1.0, 1.0):
        raise ValueError("model_sign must be +1 or -1")
    radius = int(observation_radius)
    if radius < 1 or model_radius < 1:
        raise ValueError("observation/model radii must be positive")

    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    line = np.linspace(-float(model_radius), float(model_radius), 2 * int(model_radius) + 1)
    dv, du, ss = np.meshgrid(offsets, offsets, line, indexing="ij")
    angle = np.deg2rad(float(angle_deg))
    cosine, sine = np.cos(angle), np.sin(angle)
    x = point[0] + du * cosine + ss * sine
    z = point[2] - du * sine + ss * cosine
    y = point[1] + dv
    centre = 0.5 * (float(volume.shape[0]) - 1.0)
    samples = map_coordinates(
        volume,
        [z + centre, y + centre, x + centre],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.asarray(float(model_sign) * samples.sum(axis=2), dtype=np.float32)
