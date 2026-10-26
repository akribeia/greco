"""FSC signal prior in Greco's half-map/oversampled accumulator convention.

Estimate reconstruction noise from summed precision, not from an alignment-
filtered reference. Geometry, CTFrames and noise precision are already in the
accumulator. Each half retains its own noise; only the scalar FSC is shared.
"""
import numpy as np


def reconstruction_noise(backend,weight):
    xp=backend.xp;n=backend.n;p=weight.shape[0]
    signed=xp.arange(p,dtype=xp.float64)-p//2;x=xp.arange(p//2+1,dtype=xp.float64)
    total=xp.zeros(n//2+1,xp.float64);counts=xp.zeros_like(total)
    for first in range(0,p,16):
        z=signed[first:first+16]
        r2=z[:,None,None]**2+signed[None,:,None]**2+x[None,None,:]**2
        valid=r2<n*n
        shell=xp.minimum(xp.floor(xp.sqrt(r2[valid])/2+.5).astype(xp.int32),n//2)
        total+=xp.bincount(shell,weights=8*weight[first:first+len(z)][valid],minlength=len(total))
        counts+=xp.bincount(shell,minlength=len(total))
    return backend.host(xp.where(total>1e-20,counts/xp.maximum(total,1e-30),0.))


def fsc_signal_prior(backend,weight,curve):
    """Matches half-map updateSSNRarrays; tau2_fudge is applied by reconstruction."""
    noise=reconstruction_noise(backend,weight)
    fsc=np.array([row['fsc'] for row in curve],float)
    if fsc.shape!=noise.shape or not np.isfinite(fsc).all():raise ValueError('Invalid FSC signal prior.')
    fsc=np.clip(fsc,.001,.999)
    return noise*fsc/(1-fsc),noise
