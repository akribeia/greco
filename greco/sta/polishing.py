"""Projected per-view motion MAP with temporal regularization."""
import numpy as np


def fit_view_shifts(data,prediction,precision,n,maximum_pixels,steps=10,xp=np,prior_sigma=None,previous=None):
    """Batched 2D phase-shift fit; positive shift moves the sampling centre.

    Fixed reference and 2-DOF per observation. Return shifts and full 2x2
    data curvature for subsequent temporal regularization. No CTF division.
    """
    ky=xp.fft.fftfreq(n)*2*xp.pi;kx=xp.arange(n//2+1)*2*xp.pi/n
    kx=xp.broadcast_to(kx[None,:],data.shape[-2:]);ky=xp.broadcast_to(ky[:,None],data.shape[-2:])
    # Drop zero-precision frequencies before repeated phase/line-search work.
    # The retained samples and objective are exactly the same; singleton row
    # keeps the existing batched two-axis reductions unchanged.
    if getattr(precision,'ndim',0)==2:
        use=precision>0
        data=data[:,use][:,None,:];prediction=prediction[:,use][:,None,:]
        kx=kx[use][None,:];ky=ky[use][None,:];precision=precision[use][None,:]
    delta=xp.zeros((len(data),2),xp.float64)
    previous=xp.zeros_like(delta) if previous is None else xp.asarray(previous)
    prior_weight=0. if prior_sigma is None else 1./prior_sigma**2
    # Reference Fisher curvature, not noisy observed power: otherwise noise
    # alone makes an uninformative patch look artificially precise.
    derivatives=[1j*k*prediction for k in (kx,ky)]
    H=xp.stack([xp.stack([2*xp.sum((xp.conj(a)*b).real*precision,axis=(-2,-1)) for b in derivatives],axis=1) for a in derivatives],axis=1)
    scale=xp.maximum(xp.trace(H,axis1=1,axis2=2),1e-20)
    for _ in range(steps):
        aligned=data*xp.exp(1j*(delta[:,0,None,None]*kx+delta[:,1,None,None]*ky))
        residual=aligned-prediction
        derivative=[1j*k*aligned for k in (kx,ky)]
        # |F exp(i phase)|² is constant. Drop that term analytically rather
        # than subtracting two large noisy terms (GPU roundoff at zero signal).
        grad=xp.stack([-2*xp.sum((xp.conj(d)*prediction).real*precision,axis=(-2,-1)) for d in derivative],axis=1)
        grad+=2*prior_weight*(previous+delta)
        step=xp.linalg.solve(H+xp.eye(2)[None]*(scale[:,None,None]*1e-5+2*prior_weight),-grad[...,None])[...,0]
        step=xp.clip(step,-.5,.5)
        old=xp.sum(abs(residual)**2*precision,axis=(-2,-1))+prior_weight*xp.sum((previous+delta)**2,axis=1)
        accepted=xp.zeros(len(data),bool)
        for shrink in (1.,.5,.25,.125):
            trial=xp.clip(delta+shrink*step,-maximum_pixels,maximum_pixels)
            shifted=data*xp.exp(1j*(trial[:,0,None,None]*kx+trial[:,1,None,None]*ky))
            cost=xp.sum(abs(shifted-prediction)**2*precision,axis=(-2,-1))+prior_weight*xp.sum((previous+trial)**2,axis=1)
            use=(cost<old-xp.maximum(old,1e-30)*1e-12)&~accepted
            delta=xp.where(use[:,None],trial,delta);accepted|=use
            if bool(xp.all(accepted)):break
        # Without any update, repeating this deterministic fit cannot improve.
        if not bool(xp.any(accepted)):break
    return delta,H


def smooth_track(measured,curvature,acquisition_indices,sigma_pixels,previous=None):
    """Brownian prior on consecutive acquired views, respecting missing views.

    The prior penalizes cumulative residual motion, not the latest increment.
    This projected 2D model is simpler than Greco's spatial GP3D prior.
    """
    measured=np.asarray(measured,float);count=len(measured)
    previous=np.zeros_like(measured) if previous is None else np.asarray(previous,float)
    order=np.argsort(acquisition_indices);H=np.zeros((2*count,2*count));rhs=np.zeros(2*count)
    for i in range(count):
        block=slice(2*i,2*i+2);H[block,block]=curvature[i];rhs[block]=curvature[i]@(measured[i]+previous[i])
    for i,j in zip(order[:-1],order[1:]):
        gap=max(1,abs(acquisition_indices[j]-acquisition_indices[i]));weight=1/(sigma_pixels**2*gap)
        for d in range(2):
            a,b=2*i+d,2*j+d;H[a,a]+=weight;H[b,b]+=weight;H[a,b]-=weight;H[b,a]-=weight
    H+=np.eye(2*count)*max(float(np.trace(H))/max(1,2*count),1e-20)*1e-8
    return np.linalg.solve(H,rhs).reshape(count,2)-previous
