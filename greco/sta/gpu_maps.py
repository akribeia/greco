"""CuPy implementation of the validated Greco trilinear/skip-gridding path.

Double precision throughout. Does not implement iterative Pipe–Menon gridding.
"""
import numpy as np


def spatial_radius(cp,n):
    k=cp.arange(n,dtype=cp.float64)-n//2
    return cp.sqrt(k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2)


def spatial_filters(backend):
    """Two immutable float64 fields reused by all half-map FFT paths."""
    cached=getattr(backend,'_spatial_filters',None)
    if cached is None:
        cp=backend.xp;n=backend.n;r=spatial_radius(cp,n)
        correction=cp.sinc(r/(2*n))**2
        radius=n/2
        background=cp.where(r>radius+3,1.,cp.where(r>=radius,.5+.5*cp.cos(np.pi*(radius+3-r)/3),0.))
        cached=(correction,background)
        backend._spatial_filters=cached
    return cached


def gridding_correct(cp,volume,backend=None):
    if backend is not None:return volume/spatial_filters(backend)[0]
    n=volume.shape[0];r=spatial_radius(cp,n)/(2*n)
    return volume/cp.sinc(r)**2


def prepare(backend,volume):
    cp=backend.xp;n=backend.n;p=backend.p;npad=2*n
    padded=cp.zeros((npad,)*3,cp.float64);off=n//2
    padded[off:off+n,off:off+n,off:off+n]=gridding_correct(cp,cp.asarray(volume,cp.float64),backend)
    fourier=cp.fft.rfftn(padded)
    fourier/=npad**3
    del padded
    signs=cp.where(cp.arange(npad)%2,-1.,1.)
    fourier*=signs[:,None,None]
    fourier*=signs[None,:,None]
    fourier*=signs[None,None,:n+1]
    signed=cp.arange(npad);signed=cp.where(signed<=n,signed,signed-npad)
    x=cp.arange(n+1);data=cp.zeros(backend.shape,cp.complex128)
    ps=cp.zeros(n//2+1,cp.float64);counts=cp.zeros_like(ps)
    # Slabs avoid multiple full-sized coordinate/mask arrays.
    for first in range(0,npad,16):
        z=signed[first:first+16]
        r2=z[:,None,None]**2+signed[None,:,None]**2+x[None,None,:]**2
        valid=r2<=n*n
        zz,yy,xx=cp.nonzero(valid)
        values=fourier[first+zz,yy,xx]*(8*n)
        data[z[zz]+p//2,signed[yy]+p//2,xx]=values
        shell=cp.floor(cp.sqrt(r2[valid])/2+.5).astype(cp.int32)
        ps+=cp.bincount(shell,weights=cp.abs(values)**2/2,minlength=len(ps))
        counts+=cp.bincount(shell,minlength=len(ps))
    backend.projector.power_spectrum=cp.asnumpy(ps/cp.maximum(counts,1))
    return data


def reconstruct(backend,accumulator,tau2,regularization,device=False):
    cp=backend.xp;n=backend.n;p=backend.p;h=p//2+1;rr=n*n
    signed=cp.arange(p);signed=cp.where(signed<h,signed,signed-p)
    x=cp.arange(h)
    # Decenter input arrays, preserving the reference's inclusive sphere.
    weight=cp.empty(backend.shape,cp.float64);data=cp.empty(backend.shape,cp.complex128)
    radial=cp.zeros(n//2,cp.float64);counts=cp.zeros_like(radial)
    tau=cp.asarray(tau2,cp.float64) if tau2 is not None else None
    for first in range(0,p,16):
        z=signed[first:first+16]
        r2=z[:,None,None]**2+signed[None,:,None]**2+x[None,None,:]**2
        indices=(z[:,None,None]+p//2,signed[None,:,None]+p//2,x[None,None,:])
        w=cp.where(r2<=rr,accumulator[2][indices],0.)
        d=cp.where(r2<=rr,accumulator[0][indices]+1j*accumulator[1][indices],0.)
        if tau is not None:
            shell=cp.minimum(cp.floor(cp.sqrt(r2)/2+.5).astype(cp.int32),len(tau)-1)
            t=tau[shell];positive=t>0
            invtau=cp.where(positive,1/(8*regularization*cp.maximum(t,1e-300)),cp.where(w>1e-20,1/(.001*cp.maximum(w,1e-300)),0.))
            w+=cp.where((r2<rr)&(shell>=5),invtau,0.)
        weight[first:first+len(z)]=w;data[first:first+len(z)]=d
        valid=r2<rr;bins=cp.floor(cp.sqrt(r2[valid])/2).astype(cp.int32)
        radial+=cp.bincount(bins,weights=w[valid],minlength=len(radial))
        counts+=cp.bincount(bins,minlength=len(radial))
    radial/=1000*cp.maximum(counts,1)
    for first in range(0,p,16):
        z=signed[first:first+16]
        r2=z[:,None,None]**2+signed[None,:,None]**2+x[None,None,:]**2
        bins=cp.minimum(cp.floor(cp.sqrt(r2)/2).astype(cp.int32),len(radial)-1)
        w=cp.maximum(weight[first:first+len(z)],radial[bins])
        data[first:first+len(z)]/=cp.where(w!=0,w,1.)
    del weight
    npad=2*n;k=cp.arange(npad);source=cp.where(k<=n,k,k+p-npad)
    fourier=cp.ascontiguousarray(data[source[:,None,None],source[None,:,None],cp.arange(n+1)[None,None,:]])
    del data
    signs=cp.where(k%2,-1.,1.)
    fourier*=signs[:,None,None]
    fourier*=signs[None,:,None]
    fourier*=signs[None,None,:n+1]
    volume=cp.fft.irfftn(fourier,s=(npad,)*3)
    volume*=npad**3
    volume/=(8*n)
    del fourier
    off=n//2;volume=cp.ascontiguousarray(volume[off:off+n,off:off+n,off:off+n])
    background=spatial_filters(backend)[1]
    mean=cp.sum(background*volume)/cp.maximum(cp.sum(background),1e-30)
    volume=(1-background)*volume+background*mean
    result=gridding_correct(cp,volume,backend)
    return result if device else cp.asnumpy(result)
