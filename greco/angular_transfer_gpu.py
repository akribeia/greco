"""GPU batch application of per-volume angular transfer corrections."""


_taper_cache = {}


def _radial_taper(shape, lowpass_fraction):
    """Return a process-local device taper reused by every GCC target/pass."""
    import cupy as cp
    device = int(cp.cuda.runtime.getDevice())
    key = (device, tuple(shape), float(lowpass_fraction))
    taper = _taper_cache.get(key)
    if taper is None:
        kz=cp.fft.fftfreq(shape[0])[:,None,None];ky=cp.fft.fftfreq(shape[1])[None,:,None];kx=cp.fft.fftfreq(shape[2])[None,None,:]
        radius=cp.sqrt(kz*kz+ky*ky+kx*kx)/.5;start=max(0.,lowpass_fraction-.15)
        taper=cp.ones(shape,cp.float32);transition=(radius>start)&(radius<lowpass_fraction)
        taper=cp.where(radius>=lowpass_fraction,0.,taper);taper=cp.where(transition,.5*(1+cp.cos(cp.pi*(radius-start)/(lowpass_fraction-start))),taper)
        _taper_cache[key] = taper
    return taper


def angular_transfer_operator_gpu(transfers, regularization=.05, max_gain=4.,
                                  lowpass_fraction=.85):
    """Prepare the invariant Fourier multiplier for angular compensation."""
    import cupy as cp
    h = cp.asarray(transfers)
    taper = _radial_taper(h.shape[-3:], lowpass_fraction)
    inverse=cp.conj(h)/(cp.abs(h)**2+regularization);magnitude=cp.abs(inverse)
    inverse*=cp.minimum(1.,max_gain/cp.maximum(magnitude,1e-12));inverse*=taper
    return inverse


def apply_angular_transfer_operator_gpu(volumes, operator):
    """Apply a prepared multiplier, preserving the historical FFT arithmetic."""
    import cupy as cp
    data=cp.asarray(volumes,dtype=cp.float32); inverse=cp.asarray(operator)
    if data.ndim != 4 or inverse.shape not in (data.shape, data.shape[1:]):
        raise ValueError('operator must be one per volume or shared')
    spectrum=cp.fft.fftn(cp.fft.ifftshift(data,axes=(-3,-2,-1)),axes=(-3,-2,-1))
    corrected=cp.fft.fftshift(cp.fft.ifftn(spectrum*inverse,axes=(-3,-2,-1)).real,axes=(-3,-2,-1))
    return corrected.astype(cp.float32,copy=False)

def compensate_angular_transfer_gpu_device(volumes,transfers,regularization=.05,max_gain=4.,lowpass_fraction=.85):
    import cupy as cp
    data=cp.asarray(volumes,dtype=cp.float32);h=cp.asarray(transfers)
    if data.ndim!=4 or h.shape not in (data.shape,data.shape[1:]):
        raise ValueError('transfer must be one per volume or shared')
    operator = angular_transfer_operator_gpu(h, regularization, max_gain, lowpass_fraction)
    return apply_angular_transfer_operator_gpu(data, operator)
