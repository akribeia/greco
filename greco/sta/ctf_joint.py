"""Device-resident joint anisotropic CTF likelihood from sufficient statistics.

Native phase convention: increasing defocus subtracts K1*r^2 from gamma.
The three fitted defocus components are mean, cos(2 theta), sin(2 theta).
Reference projections, noise precision and particle beam depth stay fixed.
"""
import numpy as np
from .gpu_lbfgs import minimize


def update_view(view, correction):
    """Absolute native optical state; retain particle beam-depth gradient."""
    p=np.asarray(correction,float)
    du=float(view['grecoCentralDefocusUAngst']);dv=float(view['grecoCentralDefocusVAngst'])
    angle=np.deg2rad(float(view['grecoDefocusAngleDeg']))
    mean=.5*(du+dv)+p[0]
    ca=.5*(du-dv)*np.cos(2*angle)+p[1]
    sa=.5*(du-dv)*np.sin(2*angle)+p[2]
    amplitude=np.hypot(ca,sa)
    result=dict(view)
    result.update(grecoCentralDefocusUAngst=float(mean+amplitude),
                  grecoCentralDefocusVAngst=float(mean-amplitude),
                  grecoDefocusAngleDeg=float(np.rad2deg(.5*np.arctan2(sa,ca))),
                  grecoCtfScaleFactor=float(view.get('grecoCtfScaleFactor',1.)*np.exp(p[3])))
    if not np.isfinite(list(result[k] for k in ('grecoCentralDefocusUAngst','grecoCentralDefocusVAngst','grecoDefocusAngleDeg','grecoCtfScaleFactor'))).all():
        raise ValueError('Non-finite refined optical state')
    return result


def basis(n, pixel, k1, xp=np):
    y=xp.fft.fftfreq(n)[:,None]/pixel
    x=xp.arange(n//2+1)[None,:]/(n*pixel)
    # objective() uses gamma - phase; positive defocus hence positive basis.
    return xp.asarray(k1)[:,None,None,None]*xp.stack(
        (x*x+y*y, x*x-y*y, 2*x*y), axis=0)[None]


def objective(parameters, evidence, phase_basis, regularization=0.,
              fit_scale=True, xp=np):
    """Exact weighted raw-image likelihood and analytic gradient.

    Evidence is defocus_evidence(), not a fitted per-pixel phase surrogate.
    Scale is positive via log scale. Regularization shrinks phase corrections
    towards their shared mean; call separately for each tomogram.
    """
    count=len(evidence)
    p=parameters.reshape(count,4)
    phase=xp.sum(p[:,:3,None,None]*phase_basis,axis=1)
    scale=xp.exp(p[:,3]) if fit_scale else xp.ones(count)
    a,b,d,e,f=[evidence[:,i] for i in range(5)]
    s,c=xp.sin(phase),xp.cos(phase)
    s2,c2=2*s*c,c*c-s*s
    power=.5*(a-b*c2-d*s2)
    cross=-e*c+f*s
    value=xp.sum(scale[:,None,None]**2*power+scale[:,None,None]*cross)
    derivative=scale[:,None,None]**2*(b*s2-d*c2)+scale[:,None,None]*(e*s+f*c)
    g=xp.zeros_like(p)
    g[:,:3]=xp.sum(derivative[:,None]*phase_basis,axis=(-2,-1))
    if fit_scale:
        g[:,3]=xp.sum(2*scale[:,None,None]**2*power+scale[:,None,None]*cross,axis=(-2,-1))
    delta=p[:,:3]-xp.mean(p[:,:3],axis=0)
    value+=regularization*xp.sum(delta*delta)
    g[:,:3]+=2*regularization*delta
    return value,g.ravel()


def fit(evidence,k1,n,pixel,initial_defocus=None,regularization=0.,
        fit_scale=True,max_iterations=300,xp=np,progress=None):
    """Fit one tomogram on GPU; unsupported groups retain zero correction.

    The caller must provide the coarse defocus search initializer and check
    physical bounds before committing STAR corrections.
    """
    evidence=xp.asarray(evidence,dtype=xp.float64)
    supported=xp.sum(evidence[:,0],axis=(-2,-1))>0
    initial=xp.zeros((len(evidence),4),dtype=xp.float64)
    if initial_defocus is not None:initial[:,0]=xp.asarray(initial_defocus)
    phase_basis=basis(n,pixel,k1,xp)
    # Optimize defocus in 100-A units to balance log-scale conditioning.
    units=xp.asarray([100.,100.,100.,1.])
    normalizer=xp.maximum(xp.sum(evidence[:,0]),1e-30)
    def fun(x):
        p=x.reshape(-1,4)*units
        value,g=objective(p.ravel(),evidence,phase_basis,regularization,fit_scale,xp)
        return value/normalizer,(g.reshape(-1,4)*units).ravel()/normalizer
    result=minimize(fun,(initial/units).ravel(),xp=xp,
        max_iterations=max_iterations,progress=progress,gtol=1e-8)
    corrections=result.x.reshape(-1,4)*units
    corrections=xp.where(supported[:,None],corrections,0.)
    return corrections,result
