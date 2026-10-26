"""Mask-corrected half-map FSC with independent high-frequency phases.

Resolution must be measured on unregularized half maps. Applying a tight
mask to strongly filtered MAP maps can create misleading high correlations.
"""
import numpy as np


def masked_fsc(half1,half2,mask,pixel,xp=np,seed=0,randomize_threshold=.8):
    """Return raw, masked, randomized-mask and corrected radial FSC curves.

    High-frequency phases come from independent real white noise FFTs,
    preserving Hermitian symmetry including the rFFT boundary planes.
    """
    n=half1.shape[0]
    if half1.shape!=(n,n,n) or half2.shape!=half1.shape or mask.shape!=half1.shape:
        raise ValueError('Half maps and mask must share a cubic grid.')
    m=xp.asarray(mask,dtype=xp.float64)
    if not bool(xp.isfinite(m).all()) or float(m.min())<0 or float(m.max())>1 or not bool(xp.any(m>0)):
        raise ValueError('Invalid FSC mask.')
    real=[xp.asarray(v,dtype=xp.float64) for v in (half1,half2)]
    freq=xp.fft.fftfreq(n)*n;x=xp.arange(n//2+1)
    shells=xp.rint(xp.sqrt(freq[:,None,None]**2+freq[None,:,None]**2+x[None,None,:]**2)).astype(xp.int32)
    valid=shells<=n//2;ids=shells[valid]
    mult=xp.full(n//2+1,2.,dtype=xp.float64);mult[0]=1
    if n%2==0:mult[-1]=1
    multiplicity=xp.broadcast_to(mult,shells.shape)[valid]
    def host(value):return xp.asnumpy(value) if xp is not np else np.asarray(value)
    def curve(a,b):
        cross=xp.bincount(ids,weights=(a*xp.conj(b)).real[valid]*multiplicity,minlength=n//2+1)
        aa=xp.bincount(ids,weights=abs(a[valid])**2*multiplicity,minlength=n//2+1)
        bb=xp.bincount(ids,weights=abs(b[valid])**2*multiplicity,minlength=n//2+1)
        return host(cross/xp.maximum(xp.sqrt(aa*bb),1e-30))
    fourier=[xp.fft.rfftn(v) for v in real]
    raw=curve(*fourier)
    candidates=np.flatnonzero(raw[1:]<randomize_threshold)+1
    cutoff=int(candidates[0]) if len(candidates) else n//2
    masked=curve(*[xp.fft.rfftn(v*m) for v in real])
    randomized=[]
    for h,F in enumerate(fourier):
        rng=xp.random.RandomState(seed+h+1)
        noise=xp.fft.rfftn(rng.standard_normal((n,n,n)))
        phase=noise/xp.maximum(abs(noise),1e-30)
        shuffled=xp.where(shells>=cutoff,abs(F)*phase,F)
        randomized.append(xp.fft.rfftn(xp.fft.irfftn(shuffled,s=(n,n,n),axes=(0,1,2))*m))
    random_fsc=curve(*randomized)
    corrected=masked.copy()
    region=np.arange(n//2+1)>=cutoff+2
    corrected[region]=(masked[region]-random_fsc[region])/np.maximum(1-random_fsc[region],1e-6)
    corrected=np.clip(corrected,-1,1)
    def records(values):
        return [dict(shell=i,resolution_A=n*pixel/i if i else None,fsc=float(v)) for i,v in enumerate(values)]
    return dict(unmasked=records(raw),masked=records(masked),randomized_mask=records(random_fsc),
                corrected=records(corrected),randomization_shell=cutoff,randomization_resolution_A=n*pixel/max(cutoff,1),
                phase_randomization_seed=seed,method='independent real-noise phase randomization')
