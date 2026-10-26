"""CuPy Fourier backend for the Greco Python postprocess driver.

The driver retains its I/O, RNG sequence, FSC correction and output conventions.
Only Fourier transforms, phase application, radial reductions and filters move
to GPU, in float64. GPU reductions are numerically equivalent, not bit-exact.
"""
from contextlib import contextmanager
import math
import numpy as np


@contextmanager
def accelerated(port):
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<1:raise RuntimeError('PostProcess GPU backend requires a CUDA GPU.')
    cache={}
    def grids(n):
        if n not in cache:
            wh=n//2+1;k=cp.arange(n);k=cp.where(k<wh,k,k-n);x=cp.arange(wh)
            r2=k[:,None,None]**2+k[None,:,None]**2+x[None,None,:]**2
            ids=cp.rint(cp.sqrt(r2.astype(cp.float64))).astype(cp.int32)
            valid=ids<wh
            cache[n]=(r2,ids,valid,ids[valid])
        return cache[n]
    def fwd(m):
        a=cp.asarray(m,dtype=cp.float64)
        return cp.fft.rfftn(a)/a.size
    def inv(ft,shape):
        return cp.asnumpy(cp.fft.irfftn(cp.asarray(ft),s=tuple(shape))*math.prod(shape))
    def fsc(a,b):
        n=a.shape[-1];wh=n//2+1;_,_,valid,ids=grids(n)
        fa=fwd(a)[valid];fb=fwd(b)[valid]
        cross=cp.bincount(ids,weights=fa.real*fb.real+fa.imag*fb.imag,minlength=wh)
        aa=cp.sqrt(fa.real**2+fa.imag**2);bb=cp.sqrt(fb.real**2+fb.imag**2)
        power_a=cp.bincount(ids,weights=aa*aa,minlength=wh)
        power_b=cp.bincount(ids,weights=bb*bb,minlength=wh)
        return cp.asnumpy(cross/cp.sqrt(power_a*power_b))
    def randomize(m,index,phases):
        n=m.shape[-1];r2,_,_,_=grids(n);selected=(r2>=index*index).reshape(-1)
        ft=fwd(m);flat=ft.reshape(-1);ph=cp.asarray(phases,dtype=cp.float32).astype(cp.float64)
        if ph.size!=int(cp.count_nonzero(selected)):raise ValueError('Phase sequence length mismatch.')
        values=flat[selected];mag=cp.sqrt(values.real**2+values.imag**2)
        flat[selected]=mag*(cp.cos(ph)+1j*cp.sin(ph))
        return inv(ft,m.shape)
    def weighting(ft,curve):
        maximum=0
        for i,v in enumerate(curve):
            if v<.0001:break
            maximum=i
        _,ids,_,_=grids(ft.shape[0]);selected=ids<=maximum
        values=cp.asarray(curve)[cp.minimum(ids,len(curve)-1)]
        safe=cp.maximum(values,0.)
        weights=cp.where(selected & (values>0),cp.sqrt(2*safe/(1+safe)),0.)
        ft*=weights;ft[~selected]=0.
    def bfactor(ft,n,b,pixel):
        r2,_,_,_=grids(n);r=cp.sqrt(r2.astype(cp.float64))/(n*pixel)
        selected=r<=1/(2*pixel)
        ft[selected]*=cp.exp(-(b/4)*r[selected]**2);ft[~selected]=0.
    def lowpass(ft,n,resolution,pixel,filter_edge_width=2,do_highpass_instead=False):
        xf=n*pixel/resolution;cut=int(xf+.5) if xf>=0 else int(xf-.5)
        half=filter_edge_width//2;lo=max(0.,(cut-half)/n);hi=min(ft.shape[2],(cut+half)/n)
        width=hi-lo;r2,_,_,_=grids(n);r=cp.sqrt(r2.astype(cp.float64))/n
        mid=(r>=lo)&(r<=hi)
        if do_highpass_instead:
            ft[r<lo]=0.
            if width>0:ft[mid]*=.5-.5*cp.cos(math.pi*(r[mid]-lo)/width)
        else:
            ft[r>hi]=0.
            if width>0:ft[mid]*=.5+.5*cp.cos(math.pi*(r[mid]-lo)/width)
    def guinier(ft,n,pixel,fit_minres,fit_maxres):
        wh=n//2+1;_,_,valid,ids=grids(n)
        a=ft[valid];mag=cp.sqrt(a.real**2+a.imag**2)
        sums=cp.asnumpy(cp.bincount(ids,weights=mag,minlength=wh))
        counts=cp.asnumpy(cp.bincount(ids,minlength=wh))
        x=[];y=[];w=[]
        for i in range(wh):
            res=n*pixel/i if i else math.inf
            if res>=pixel*2:
                x.append(1/(res*res) if math.isfinite(res) else 0.)
                y.append(math.log(sums[i]/counts[i]) if sums[i]>0 else -99.)
                w.append(1. if sums[i]>0 and res<=fit_minres and res>=fit_maxres else 0.)
        return np.asarray(x),np.asarray(y),np.asarray(w)
    replacements=dict(_fwd=fwd,_inv=inv,calculate_fsc=fsc,randomize_phases_beyond=randomize,
        apply_fsc_weighting=weighting,apply_bfactor=bfactor,low_pass_filter=lowpass,make_guinier_plot=guinier)
    originals={k:getattr(port,k) for k in replacements}
    try:
        for k,v in replacements.items():setattr(port,k,v)
        yield
        cp.cuda.get_current_stream().synchronize()
    finally:
        for k,v in originals.items():setattr(port,k,v)
        cache.clear()
