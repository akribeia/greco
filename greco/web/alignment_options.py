"""Normal Reconstruction alignment settings; old configs stay historical."""
import math
from .projects import InputError


def validate(smoothness_strength=0.,rescue_tilts=True):
    if isinstance(smoothness_strength,bool) or not isinstance(smoothness_strength,(int,float)) or not math.isfinite(smoothness_strength) or not 0<=smoothness_strength<=5:
        raise InputError('Smoothness must be between 0 and 5.')
    if not isinstance(rescue_tilts,bool):raise InputError('Rescue tilts must be a checkbox.')
    return dict(smoothness_strength=float(smoothness_strength),rescue_tilts=rescue_tilts)
