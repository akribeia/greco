"""Central Z slabs expressed against the unchanged full reconstruction grid."""
import math
import numpy as np


def validate_percent(value):
    if isinstance(value,bool):raise ValueError('Tomogram thickness must be between 30% and 100%.')
    try:value=float(value)
    except (TypeError,ValueError,OverflowError):raise ValueError('Tomogram thickness must be between 30% and 100%.') from None
    if not math.isfinite(value) or not 30<=value<=100:raise ValueError('Tomogram thickness must be between 30% and 100%.')
    return value


def slab_bounds(depth,percent):
    """Keep the original voxel centre exactly, including odd-size volumes."""
    percent=validate_percent(percent)
    if type(depth) is not int or depth<1:raise ValueError('Invalid tomogram depth.')
    kept=max(1,min(depth,int(math.floor(depth*percent/100+.5))))
    if (depth-kept)%2:kept=min(depth,kept+1)
    start=(depth-kept)//2
    return start,start+kept


def filter_maxima(maxima,score,percent):
    if validate_percent(percent)==100:return maxima
    start,stop=slab_bounds(int(score.shape[0]),percent)
    maxima=maxima[(maxima[:,0]>=start)&(maxima[:,0]<stop)]
    if not len(maxima):return maxima
    peaks=score[tuple(maxima.T)]
    peaks=peaks.get() if hasattr(peaks,'get') else np.asarray(peaks)
    return maxima[np.isfinite(peaks)&(peaks>-1e20)]


def stage_bounds(depth, percent, halo=32):
    """Detection core plus context, always in original full-grid coordinates."""
    start, stop = slab_bounds(depth, min(50., validate_percent(percent)))
    return (start, stop), (max(0, start-halo), min(depth, stop+halo))


def reconstruct_stage(reconstruct, stack, angles, percent, **kwargs):
    """Compute only the useful slab; retain cubic indexing for old consumers.

    The host padding is compatibility storage, not reconstructed density.
    No GPU backprojection or detector filtering is done in that padding.
    """
    depth = int(stack.shape[-1])
    _, bounds = stage_bounds(depth, percent)
    slab = reconstruct(stack, angles, z_bounds=bounds, **kwargs)
    volume = np.zeros((depth, depth, depth), dtype=np.float32)
    volume[bounds[0]:bounds[1]] = slab
    return volume


def score_stage(score_function, volume, selector, percent):
    import cupy as cp
    depth = int(volume.shape[0])
    core, bounds = stage_bounds(depth, percent)
    slab = score_function(volume[bounds[0]:bounds[1]], selector)
    score = cp.full(volume.shape, -1e30, dtype=cp.float32)
    score[core[0]:core[1]] = slab[core[0]-bounds[0]:core[1]-bounds[0]]
    return score
