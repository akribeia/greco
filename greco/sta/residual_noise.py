"""Frozen native half-radial noise in the actual STA observation domain.

Initial own-half maps are deliberately low-pass filtered. No imported spectra,
no map sharing across halves, no division by CTF zeros. Misaligned initial
poses can inflate residual variance; this is not a complete Greco noise EM.
"""
import numpy as np
from .background_noise import regularize_background
from .observation_model import backprojection_statistics


def estimate_half_residual_noise(dataset,backend,reference_resolution_A=12.,batch_observations=2048,per_view=False):
    from greco._numerics.core.projector import initial_lowpass_filter_reference
    xp=backend.xp;n=dataset.n;ns=n//2+1
    halves=np.array([p['grecoRandomSubset']-1 for p in dataset.particles])
    if np.any((halves<0)|(halves>1)) or any(np.sum(halves==h)<2 for h in (0,1)):
        raise ValueError('Residual noise requires two supported particles per independent half.')
    y=xp.fft.fftfreq(n)*n;x=xp.arange(ns)
    radius=xp.sqrt(y[:,None]**2+x[None,:]**2)
    shell=xp.minimum(xp.rint(radius).astype(xp.int32),n//2)
    active=(radius>0)&(radius<=n//2-1)
    counts=xp.bincount(shell[active],minlength=ns)
    premultiplied=bool(getattr(dataset,'ctf_premultiplied',False))
    old_radius=backend.radius;acc=[backend.zeros(),backend.zeros()]
    backend.radius=n//2-1
    try:
        for start,stop,f,c in dataset.loaded_batches(batch_observations):
            a,b=dataset.offsets[[start,stop]]
            owner=np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1]))
            for h in (0,1):
                use=halves[owner]==h
                if use.any():
                    data,weight=backprojection_statistics(f[use],c[use],active,premultiplied)
                    backend.accumulate(acc[h],data,weight,dataset.rotations[a:b][use])
        from .memory_refinement import AccumulatorSpool,release_unused
        spool=None
        if getattr(backend,'staged_accumulators',False):
            spool=AccumulatorSpool(backend.memory_output,backend)
            for h in (0,1):spool.store(h,acc[h])
            acc.clear();release_unused(backend)
        refs=[]
        resolution=max(float(reference_resolution_A),2*dataset.pixel)
        for h in (0,1):
            if spool is None:volume=backend.reconstruct(acc[h],None,4.,device=backend.gpu)
            else:
                with spool.acquire(h) as accumulator:
                    volume=backend.reconstruct(accumulator,None,4.,device=backend.gpu)
            volume=xp.asarray(initial_lowpass_filter_reference(backend.host(volume),n,dataset.pixel,resolution))
            refs.append(backend.prepare_map(volume))
        del acc,volume
        if spool is not None:spool.close()
        ng=len(dataset.group_keys) if per_view else 1
        sums=xp.zeros((2,ng,ns),xp.float64);observations=xp.zeros((2,ng),xp.float64)
        for start,stop,f,c in dataset.loaded_batches(batch_observations):
            a,b=dataset.offsets[[start,stop]]
            owner=np.repeat(np.arange(start,stop),np.diff(dataset.offsets[start:stop+1]))
            for h in (0,1):
                use=halves[owner]==h
                if not use.any():continue
                predicted=backend.project(refs[h],dataset.rotations[a:b][use],dataset.inverse_rotations[a:b][use])
                residual=.5*abs(f[use]-c[use]*predicted)**2
                labels=dataset.groups[a:b][use] if per_view else np.zeros(int(use.sum()),int)
                for group in np.unique(labels):
                    chosen=xp.asarray(labels==group)
                    total=residual[chosen].sum(axis=0)
                    sums[h,group]+=xp.bincount(shell[active],weights=total[active],minlength=ns)/xp.maximum(counts,1)
                    observations[h,group]+=int((labels==group).sum())
        variance=regularize_background(sums,observations,xp)
        if not per_view:variance=variance[:,0]
        return variance,dict(method='frozen_own_half_residual',reference_resolution_A=resolution,
            observations=backend.host(observations).ravel().tolist(),imported_noise=False,
            observation_model='ctf_premultiplied' if premultiplied else 'raw',per_view=per_view)
    finally:
        backend.radius=old_radius
