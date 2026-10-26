"""Geometric pair exclusion, never signal/likelihood-driven pruning."""
import numpy as np


def prune_pairs(support, outside, minimum_views=3, minimum_per_tilt=8,*,active_tilts=None):
    support=np.asarray(support,bool);outside=np.asarray(outside,bool)
    if support.shape!=outside.shape:raise ValueError('Support shape mismatch')
    removed=support & outside
    remaining=support & ~removed
    keep=remaining.sum(1)>=minimum_views
    remaining[~keep]=False
    counts=remaining.sum(0)
    if active_tilts is not None:counts=counts[np.asarray(active_tilts,int)]
    if not keep.any() or np.any(counts<minimum_per_tilt):
        raise ValueError('Geometric exclusion would leave a tilt underconstrained')
    return remaining[keep],keep,removed


def accepted_view_order(mdoc_indices,active):
    """Keep source identities; optimize only accepted views in acquisition order."""
    indices=np.asarray(mdoc_indices,int);active=np.asarray(active,int)
    if active.ndim!=1 or len(active)<3 or np.any(active<0) or np.any(active>=len(indices)) or np.any(np.diff(active)<=0):
        raise ValueError('Invalid accepted geometry views.')
    order=np.argsort(indices)
    fitted=order[np.isin(order,active)]
    inverse=np.full(len(indices),-1,int);inverse[fitted]=np.arange(len(fitted))
    return order,fitted,inverse


def expand_camera_update(cameras,translations,active,new_cameras,new_translations):
    """Excluded cameras stay exactly fixed outside the optimizer's parameters."""
    c=np.array(cameras,copy=True);t=np.array(translations,copy=True)
    c[active]=new_cameras;t[active]=new_translations
    return c,t
