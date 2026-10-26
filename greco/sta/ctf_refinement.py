"""Per-view mean-defocus sufficient statistics for fixed native projections.

    Preserve astigmatism and beam-depth gradient; optimize only one common
    defocus offset per tilt against own-half reference projections.
    """
import numpy as np


def defocus_evidence(data,prediction,gamma,precision,xp=np):
    p2=abs(prediction)**2*precision
    cross=2*(data*xp.conj(prediction)).real*precision
    return xp.stack((p2,p2*xp.cos(2*gamma),p2*xp.sin(2*gamma),cross*xp.sin(gamma),cross*xp.cos(gamma)),axis=1)


def defocus_cost(evidence,delta_phase,xp=np):
    a,b,d,e,f=[evidence[:,i] for i in range(5)]
    return xp.sum(.5*(a-b*xp.cos(2*delta_phase)-d*xp.sin(2*delta_phase))-e*xp.cos(delta_phase)+f*xp.sin(delta_phase),axis=(-2,-1))


def fit_defocus_offsets(evidence,k1,n,pixel,maximum_A,step_A,xp=np):
    y=xp.fft.fftfreq(n)[:,None]/pixel;x=xp.arange(n//2+1)[None,:]/(n*pixel)
    phase=xp.asarray(k1)[:,None,None]*(x*x+y*y)
    grid=np.linspace(-maximum_A,maximum_A,2*int(np.ceil(maximum_A/step_A))+1)
    curves=xp.stack([defocus_cost(evidence,phase*d,xp) for d in grid],axis=1)
    best=xp.argmin(curves,axis=1);indices=xp.arange(len(evidence));centre=curves[indices,best]
    left=curves[indices,xp.maximum(best-1,0)];right=curves[indices,xp.minimum(best+1,len(grid)-1)]
    spacing=grid[1]-grid[0];curvature=(left-2*centre+right)/spacing**2
    interpolation=.5*(left-right)/xp.maximum(left-2*centre+right,1e-30)
    interpolation=xp.where((best>0)&(best<len(grid)-1),xp.clip(interpolation,-1,1),0.)
    offsets=xp.asarray(grid)[best]+spacing*interpolation
    offsets=xp.clip(offsets,-maximum_A,maximum_A)
    zero=defocus_cost(evidence,xp.zeros_like(phase),xp)
    fitted=defocus_cost(evidence,phase*offsets[:,None,None],xp)
    # Unsupported/no-signal groups must not acquire an arbitrary grid endpoint.
    # A grid-endpoint maximum is not a localized defocus estimate. Retain the
    # previous CTF and report the bound hit rather than imposing ±range blindly.
    valid=(best>0)&(best<len(grid)-1)&(curvature>1e-20)&(fitted<zero)
    return xp.where(valid,offsets,0.),curves,xp.where(valid,curvature,0.)
