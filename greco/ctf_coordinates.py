"""Explicit coordinate contract for the existing detector-frame CTF model."""
import math
import statistics

PROTOCOL='ctf-detector-frame-v1'


def detector_axis(mdoc,explicit=None,*,flip_y=False,transpose=False):
    """Resolve the defocus-gradient direction in the MC detector raster.

    SerialEM TiltAxisAngle is the rotation shown by the GUI. The physical
    tilt axis is perpendicular to this gradient direction. RotationAngle
    describes that physical axis from detector X, so using -RotationAngle
    directly here introduces a quarter-turn error.
    """
    if explicit is None:
        display=getattr(mdoc,'tilt_axis_angle_deg',None)
        if display is not None:
            value=float(display)
            if not math.isfinite(value):raise ValueError('TiltAxisAngle must be finite.')
        else:
            rotations=[float(t.rotation_angle_deg) for t in mdoc.tilts if t.rotation_angle_deg is not None]
            if not rotations or not all(math.isfinite(v) for v in rotations):
                raise ValueError('CTF Auto requires TiltAxisAngle, RotationAngle or an explicit detector gradient axis.')
            if max(rotations)-min(rotations)>.05:
                raise ValueError('RotationAngle is not constant across the tilt series; calibrate the raw CTF axis explicitly.')
            value=-90.-float(statistics.median(rotations))
            value=(value+180.)%360.-180.
        # Auto starts in the acquisition raster; carry it through the same
        # optional MC input reflection/transpose as the images. An explicit
        # calibration is already in the MC raster and is not transformed twice.
        if flip_y and transpose:raise ValueError('Flip Y and transpose are mutually exclusive.')
        if flip_y:value=-value
        elif transpose:value=90-value
        return value
    if isinstance(explicit,bool):raise ValueError('Invalid raw CTF axis.')
    value=float(explicit)
    if not math.isfinite(value) or abs(value)>360:raise ValueError('Invalid raw CTF axis.')
    return value
