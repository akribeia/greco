"""Continue rigid geometry across sampling without scaling rotations or Angstrom deltas."""
import numpy as np


def resample_checkpoint(saved, pixel, size, binning, baseline_xyz, baseline_translations):
    old_pixel=float(saved['pixel_size_A']);old_size=int(saved['fit_size']);old_bin=int(saved['binning'])
    if (old_bin not in (1,2,4,8) or not np.isfinite(old_pixel) or old_pixel<=0
        or not np.isclose(old_pixel/old_bin,pixel/binning,rtol=1e-6,atol=1e-8)
        or not np.isclose(old_size*old_pixel,size*pixel,rtol=1e-6,atol=1e-6)):
        raise ValueError('Checkpoint sampling does not match the native detector field of view.')
    # Absolute detector translations contain half-pixel/flip origin offsets.
    # Transfer the fitted physical correction, not that sampling-specific origin.
    delta_xyz=np.asarray(saved['positions_xyz_A'])-np.asarray(saved['original_positions_xyz_A'])
    delta_t=np.asarray(saved['translations_xy_A'])-np.asarray(saved['original_translations_xy_A'])
    if (delta_xyz.shape!=baseline_xyz.shape or delta_t.shape!=baseline_translations.shape
        or not np.isfinite(delta_xyz).all() or not np.isfinite(delta_t).all()):
        raise ValueError('Invalid checkpoint physical geometry.')
    return baseline_xyz+delta_xyz/pixel,baseline_translations+delta_t/pixel
