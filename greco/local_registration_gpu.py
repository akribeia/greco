"""CuPy batches for the two recommended local-registration measurements."""

import numpy as np

from .registration import LocalRegistrationResult, _summarize, _summarize_vectorized, cosine_disk_mask


def register_template_zncc_gpu(models, regions, search_radius=32):
    """Batched GPU equivalent of bounded_template_shift_at's ZNCC grid."""
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve
    mm = cp.asarray(models, cp.float64)
    rr = cp.asarray(regions, cp.float64)
    if mm.ndim != 3 or rr.ndim != 3 or len(mm) != len(rr):
        raise ValueError('expected matching model/region batches')
    centred = mm - mm.mean(axis=(-2, -1), keepdims=True)
    numerator = fftconvolve(rr, centred[:, ::-1, ::-1], mode='valid', axes=(-2, -1))
    ones = cp.ones((1, mm.shape[-2], mm.shape[-1]), cp.float64)
    count = mm.shape[-2] * mm.shape[-1]
    patch_sum = fftconvolve(rr, ones, mode='valid', axes=(-2, -1))
    patch_sq = fftconvolve(rr * rr, ones, mode='valid', axes=(-2, -1))
    patch_energy = cp.maximum(patch_sq - patch_sum * patch_sum / count, 0.0)
    model_energy = cp.sum(centred * centred, axis=(-2, -1))[:, None, None]
    denominator = cp.sqrt(model_energy * patch_energy)
    displacement_surface = cp.where(denominator > 1e-12, numerator / denominator, -1.0)
    # Valid-grid index encodes observed displacement; reverse both axes so
    # index-radius is the raster correction, matching bounded_template_shift_at.
    surfaces = cp.asnumpy(displacement_surface[:, ::-1, ::-1])
    results = []
    radius = int(search_radius)
    for surface in surfaces:
        peak = np.unravel_index(np.argmax(surface), surface.shape)
        correction = np.asarray([peak[0] - radius, peak[1] - radius], float)
        for axis in (0, 1):
            index = peak[axis]
            if 0 < index < surface.shape[axis] - 1:
                before = list(peak); after = list(peak)
                before[axis] -= 1; after[axis] += 1
                fm, f0, fp = surface[tuple(before)], surface[peak], surface[tuple(after)]
                curvature = fm - 2 * f0 + fp
                if curvature < -1e-12:
                    correction[axis] += float(np.clip(.5 * (fm - fp) / curvature, -.5, .5))
        boundary = bool(np.any(np.abs(correction) >= radius - .05))
        results.append((correction, surface, boundary))
    return results


def register_local_models_gpu(models, regions, *, method, search_radius=4,
                              alpha=.25, intensity_fraction=.5, mask=None,
                              batch_size=32) -> list[LocalRegistrationResult]:
    import cupy as cp
    from cupyx.scipy.ndimage import correlate

    dtype=cp.float64 if method=='gradient_zncc' else cp.float32
    models=cp.asarray(models,dtype=dtype);regions=cp.asarray(regions,dtype=dtype)
    if models.ndim!=3 or regions.ndim!=3 or len(models)!=len(regions):
        raise ValueError('expected matching model/region batches')
    if method not in ('generalized_cc','gradient_zncc'):
        raise ValueError('GPU batch supports GCC and gradient ZNCC')
    results=[]
    for start in range(0,len(models),batch_size):
        mm=cp.asarray(models[start:start+batch_size]);rr=cp.asarray(regions[start:start+batch_size])
        if method=='generalized_cc':
            support=mm.shape[-1];size=rr.shape[-1];offset=(size-support)//2
            wm=cp.asarray(np.outer(np.hanning(support),np.hanning(support)),cp.float32)
            wr=cp.asarray(np.outer(np.hanning(size),np.hanning(size)),cp.float32)
            padded=cp.zeros((len(mm),size,size),cp.float32)
            padded[:,offset:offset+support,offset:offset+support]=(mm-mm.mean((-2,-1),keepdims=True))*wm
            observed=(rr-rr.mean((-2,-1),keepdims=True))*wr
            cross=cp.conj(cp.fft.fft2(padded,axes=(-2,-1)))*cp.fft.fft2(observed,axes=(-2,-1))
            mag=cp.abs(cross);positive=cp.where(mag>0,mag,cp.nan)
            eps=cp.maximum(cp.nanmedian(positive,axis=(-2,-1)),1.)*1e-12
            corr=cp.fft.ifft2(cross/(mag+eps[:,None,None])**alpha,axes=(-2,-1)).real
            offsets=cp.arange(-search_radius,search_radius+1)
            surfaces=corr[:,(-offsets)%size][:,:,(-offsets)%size]
            scale=cp.max(cp.abs(surfaces),axis=(-2,-1),keepdims=True)
            surfaces=cp.where(scale>1e-12,surfaces/scale,-1)
        else:
            weights=cp.asarray(cosine_disk_mask(mm.shape[-1]) if mask is None else mask,cp.float64)
            patches=cp.lib.stride_tricks.sliding_window_view(rr,mm.shape[-2:],axis=(-2,-1))[:,::-1,::-1]
            total=weights.sum()
            def centre(values):
                return values-cp.sum(values*weights,axis=(-2,-1),keepdims=True)/total
            def corr(model_values,patch_values):
                num=cp.sum(weights*model_values[:,None,None]*patch_values,axis=(-2,-1))
                den=cp.sqrt(cp.sum(weights*model_values**2,axis=(-2,-1))[:,None,None]*cp.sum(weights*patch_values**2,axis=(-2,-1)))
                return cp.where(den>1e-12,num/den,-1)
            intensity=corr(centre(mm),centre(patches))
            ky=cp.asarray([[-1,-2,-1],[0,0,0],[1,2,1]],cp.float64)
            my=centre(correlate(mm,ky[None],mode='nearest')/8);mx=centre(correlate(mm,ky.T[None],mode='nearest')/8)
            py=centre(correlate(patches,ky[None,None,None],mode='nearest')/8);px=centre(correlate(patches,ky.T[None,None,None],mode='nearest')/8)
            surfaces=intensity_fraction*intensity+(1-intensity_fraction)*.5*(corr(my,py)+corr(mx,px))
        host=cp.asnumpy(surfaces)
        if method == 'generalized_cc':
            results.extend(_summarize_vectorized(host, search_radius))
        else:
            results.extend(_summarize(surface,search_radius) for surface in host)
    return results
