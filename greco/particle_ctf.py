"""One physical beam-depth convention for STA extraction and STAR consumers."""
import numpy as np


def native_beam_defocus_delta(rotation, centred_xyz, pixel_size_A, flip_y, hand):
    """XYZ in pixels, sampling in A/pixel; return defocus delta in Angstrom."""
    if hand not in (-1,1) or not np.isfinite(pixel_size_A) or pixel_size_A<=0:
        raise ValueError('Invalid beam defocus handedness or sampling.')
    return (np.asarray(centred_xyz, float) @ np.asarray(rotation, float)[2]
            * float(pixel_size_A) * float(hand) * (-1 if flip_y else 1))


def star_particle_defocus(particle, view, observation):
    """Evaluate particle CTF from the central fit and native beam geometry.

    Coordinates in STAR are already Angstrom and centered on the exported
    tomogram. Do not multiply by detector or extraction binning a second time.
    A partially populated native geometry is rejected instead of guessed.
    """
    central=('grecoCentralDefocusUAngst','grecoCentralDefocusVAngst')
    if any(k not in view for k in central):
        raise ValueError('Particle STAR lacks central-fit CTF metadata; extract again.')
    R=np.array([[view[f'grecoProjection{i}{j}'] for j in range(3)] for i in range(3)])
    xyz=np.array([particle[f'grecoCoordinate{d}Angst'] for d in 'XYZ'])
    if view['grecoRasterFlipY'] not in (0,1):raise ValueError('Invalid STAR raster flip.')
    delta=native_beam_defocus_delta(R,xyz,1.,bool(view['grecoRasterFlipY']),view['grecoDefocusHand'])
    result=np.array([view[k]+delta for k in central],float)
    if not np.isfinite(result).all():raise ValueError('Non-finite native particle defocus.')
    return tuple(result)
