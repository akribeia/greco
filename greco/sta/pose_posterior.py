"""Small, explicit local pose quadrature, not exhaustive Greco EM sampling."""
import numpy as np
from scipy.spatial.transform import Rotation


def compose_pose(current, increment):
    rotation=Rotation.from_euler('ZYX',current[:,:3][:,::-1])*Rotation.from_euler('ZYX',increment[:,:3][:,::-1])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        angles=rotation.as_euler('ZYX')[:,::-1]
    return np.column_stack((angles,current[:,3:]+increment[:,3:]))


def local_posterior(objective, center, surface_radius, limits=None):
    """13 equal-volume nodes; Gaussian local displacement prior, stable softmax.

    Radius corresponds to at most 1/8 of the alignment resolution in pixels.
    All views of one particle share the same six-dimensional hypothesis.
    """
    motion=min(.5,objective.backend.n/max(objective.backend.radius,1)/8)
    scales=np.array([motion/max(surface_radius,1)]*3+[motion]*3)
    offsets=np.vstack((np.zeros(6),np.diag(scales),-np.diag(scales)))
    poses=np.stack([center+offset for offset in offsets],axis=1)
    if limits is not None:
        valid=np.all(abs(poses)<=limits,axis=2)
        # The fitted central pose is always represented.
        valid[:,0]=True
    else:valid=np.ones(poses.shape[:2],bool)
    costs=np.column_stack([objective.per_particle(poses[:,j])[:,0] for j in range(13)])
    costs+=.5*np.sum((offsets/scales)**2,axis=1)[None,:]
    valid&=np.isfinite(costs)
    if np.any(~valid.any(axis=1)):raise ValueError('No finite local pose posterior.')
    logp=np.where(valid,-costs,-np.inf)
    logp-=np.max(logp,axis=1,keepdims=True)
    weights=np.exp(logp);weights/=weights.sum(axis=1,keepdims=True)
    # Negligible mass is omitted from backprojection, explicitly renormalized.
    weights[weights<1e-4]=0;weights/=weights.sum(axis=1,keepdims=True)
    return poses,weights
