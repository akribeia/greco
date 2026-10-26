"""Validated reuse of independent half noise within one refinement protocol."""
import numpy as np


def load_frozen_noise(path, n, xp):
    variance = np.load(path, allow_pickle=False)
    if variance.shape != (2, n // 2 + 1) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('Invalid frozen independent-half noise spectra.')
    return xp.asarray(variance), dict(source='protocol_frozen_half_noise', path=str(path))
