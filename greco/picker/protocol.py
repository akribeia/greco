"""Validate picker configuration and build per-tilt physical metadata."""
import numpy as np
from .weights import TiltSeriesMetaData,CtfData


def metadata_from_config(config):
    angles=np.asarray(config['tilt_angles_deg'],dtype=float)
    if angles.ndim!=1 or len(angles)<2 or not np.isfinite(angles).all():
        raise ValueError('Picker needs finite tilt angles')
    # Dose-aware metadata enables tilt weighting only when doses are supplied.
    physical=config.get('tilt_series_metadata',{})
    if not isinstance(physical,dict):raise ValueError('Invalid picker physical metadata')
    allowed={'ctf_data','dose_accumulation','defocus_handedness','level_angle_x','level_angle_y'}
    if set(physical)-allowed:raise ValueError('Unknown picker physical metadata')
    dose=physical.get('dose_accumulation')
    if dose is not None:
        values=np.asarray(dose,dtype=float)
        if values.shape!=angles.shape or not np.isfinite(values).all() or np.any(values<0):
            raise ValueError('Dose must be finite, nonnegative and match tilt order')
        dose=values.tolist()
    ctfs=physical.get('ctf_data')
    if ctfs is not None:
        if not isinstance(ctfs,list) or len(ctfs)!=len(angles):raise ValueError('CTF list must match tilt order')
        parsed=[]
        for row in ctfs:
            if not isinstance(row,dict):raise ValueError('Invalid CTF metadata')
            try:ctf=CtfData(**row)
            except TypeError as error:raise ValueError('Invalid CTF metadata') from error
            if not np.isfinite([ctf.defocus,ctf.voltage,ctf.spherical_aberration,ctf.amplitude_contrast,ctf.phase_shift_deg]).all():
                raise ValueError('CTF metadata must be finite')
            if not 0<ctf.defocus<=.001 or not 1000<=ctf.voltage<=1e7 or not 0<=ctf.spherical_aberration<=.1 or not 0<=ctf.amplitude_contrast<1 or not isinstance(ctf.flip_phase,bool):
                raise ValueError('Invalid CTF physics or units (meters, volts)')
            parsed.append(ctf)
        ctfs=parsed
    hand=physical.get('defocus_handedness',0)
    if isinstance(hand,bool) or hand not in (-1,0,1):raise ValueError('Invalid defocus handedness')
    level=[physical.get('level_angle_x',0.),physical.get('level_angle_y',0.)]
    if not np.isfinite(level).all():raise ValueError('Invalid volume level angles')
    weighted=config.get('per_tilt_weighting',dose is not None)
    if not isinstance(weighted,bool):raise ValueError('Invalid per-tilt weighting flag')
    return TiltSeriesMetaData(angles.tolist(),ctf_data=ctfs,dose_accumulation=dose,
        per_tilt_weighting=weighted,defocus_handedness=hand,
        level_angle_x=float(level[0]),level_angle_y=float(level[1]))
