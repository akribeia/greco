"""Peripheral-patch noise spectra, before circle cropping.

The estimator assumes locally stationary noise. Peripheral sample structure
can bias it upward; it is not an exact separation of signal and noise.
Fourier normalization matches the dataset's FFT/n**2 convention.
"""
import numpy as np


def regularize_background(power_sum, counts, xp=np, shrinkage=32.):
    """Pool sparse view groups towards their own half's spectrum.

    Input is (half,view,shell), never pooled across halves. Fixed mild radial
    smoothing reduces periodogram variance; no reference/FSC is consulted.
    """
    power_sum=xp.asarray(power_sum,dtype=xp.float64);counts=xp.asarray(counts,dtype=xp.float64)
    if power_sum.ndim!=3 or power_sum.shape[:2]!=counts.shape or power_sum.shape[0]!=2:
        raise ValueError('Invalid half/view noise accumulator')
    if bool(xp.any(counts.sum(axis=1)<=0)):raise ValueError('Missing half noise observations')
    pooled=power_sum.sum(axis=1)/counts.sum(axis=1)[:,None]
    estimate=(power_sum+shrinkage*pooled[:,None,:])/(counts[:,:,None]+shrinkage)
    padded=xp.pad(estimate,((0,0),(0,0),(2,2)),mode='edge')
    estimate=sum(padded[:,:,i:i+estimate.shape[-1]]*w for i,w in enumerate((1,2,3,2,1)))/9
    floor=xp.maximum(xp.mean(estimate[:,:,2:],axis=(1,2)),1e-30)*1e-6
    estimate=xp.maximum(estimate,floor[:,None,None])
    estimate[:,:,:2]=estimate[:,:,2,None]
    if not bool(xp.isfinite(estimate).all()):raise ValueError('Nonfinite background noise')
    return estimate
