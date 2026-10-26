"""Materialize cumulative alignment from one immutable source stack."""

import numpy as np
from scipy.ndimage import shift as ndi_shift


def materialize_cumulative_stack(original, cumulative_shifts_yx):
    """Apply total shifts once to ``original``; never resample an aligned stack."""
    original=np.asarray(original,np.float32)
    shifts=np.asarray(cumulative_shifts_yx,float)
    if shifts.shape!=(len(original),2):
        raise ValueError('cumulative shifts must have shape (n_tilts,2)')
    return np.asarray([ndi_shift(image,delta,order=1,mode='constant',
        cval=float(image.mean()),prefilter=False)
        for image,delta in zip(original,shifts,strict=True)],np.float32)
