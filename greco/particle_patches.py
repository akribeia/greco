"""Batched raw patch extraction and Greco-compatible preprocessing."""
import numpy as np

CROP=r'''
extern "C" __global__ void crop(const float* src,const int* origins,const unsigned char* visible,
 int views,int height,int width,int n,long count,float* dst){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 long observation=i/((long)n*n);int t=observation%views,x=i%n,y=(i/n)%n;
 if(!visible[observation]){dst[i]=0;return;}
 int xx=origins[2*observation]+x,yy=origins[2*observation+1]+y;
 dst[i]=(xx>=0&&xx<width&&yy>=0&&yy<height)?src[((long)t*height+yy)*width+xx]:0;
}
'''


def integer_origins(centres,n):
    c=np.asarray(centres,float)
    rounded=np.sign(c)*np.floor(abs(c)+.5)
    return np.ascontiguousarray(rounded.astype(np.int32)-n//2),c-rounded


def crop_batch(stack,centres,visible,n,gpu=False,storage_dtype=np.float32):
    storage_dtype=np.dtype(storage_dtype)
    if storage_dtype not in (np.dtype('float16'),np.dtype('float32')):
        raise ValueError('Particle storage must be float16 or float32.')
    origins,_=integer_origins(centres,n)
    shape=(*visible.shape,n,n)
    if gpu:
        import cupy as cp
        src=cp.asarray(stack,dtype=cp.float32)
        dst=cp.empty(shape,cp.float32);kernel=cp.RawKernel(CROP,'crop')
        kernel(((dst.size+255)//256,),(256,),(src,cp.asarray(origins),cp.asarray(visible,np.uint8),
            np.int32(src.shape[0]),np.int32(src.shape[1]),np.int32(src.shape[2]),np.int32(n),np.int64(dst.size),dst))
        if storage_dtype==np.float16:
            lo=float(dst.min());hi=float(dst.max())
            if not np.isfinite([lo,hi]).all() or lo < -65504. or hi > 65504.:
                raise ValueError('Extracted images exceed the float16 storage range or contain nonfinite values.')
            dst=dst.astype(cp.float16)
        return cp.asnumpy(dst)
    result=np.zeros(shape,np.float32)
    for p,t in np.argwhere(visible):
        x,y=origins[p,t];result[p,t]=stack[t,y:y+n,x:x+n]
    if storage_dtype==np.float16:
        lo=float(result.min());hi=float(result.max())
        if not np.isfinite([lo,hi]).all() or lo < -65504. or hi > 65504.:
            raise ValueError('Extracted images exceed the float16 storage range or contain nonfinite values.')
    return result.astype(storage_dtype,copy=False)


def spectra_batch(patches,offsets,gpu=False):
    """Circle taper once, orthonormal FFT, then the stored fractional shift."""
    n=patches.shape[-1]
    if not gpu:
        from greco._numerics.tomo.extraction import crop_circle,fourier_transform_stack_fast_centered,shift_stack
        shape=patches.shape[:-2];raw=np.array(patches,np.float32).reshape(-1,n,n)
        crop_circle(raw,0.,5.)
        fs=fourier_transform_stack_fast_centered(raw)
        return shift_stack(fs,np.asarray(offsets).reshape(-1,2)+n/2,True).reshape(*shape,n//2+1,n)
    import cupy as cp
    raw=cp.asarray(patches).astype(cp.float32,copy=False);shape=raw.shape[:-2]
    y,x=cp.meshgrid(cp.arange(n,dtype=cp.float64)-n/2,cp.arange(n,dtype=cp.float64)-n/2,indexing='ij')
    radius=cp.sqrt(x*x+y*y);outside=radius>n/2;count=int(cp.count_nonzero(outside))
    mean=(raw.astype(cp.float64)*outside).sum(axis=(-2,-1))/count if count else cp.zeros(shape,cp.float64)
    taper=cp.where(radius<n/2-5,1.,cp.where(radius<n/2,.5-.5*cp.cos(np.pi*(n/2-radius)/5),0.))
    raw=(taper*(raw.astype(cp.float64)-mean[...,None,None])).astype(cp.float32)
    fs=cp.fft.rfft2(cp.roll(raw,(-n//2,-n//2),axis=(-2,-1)),axes=(-2,-1),norm='ortho').swapaxes(-1,-2)
    xx=cp.arange(n//2+1,dtype=cp.float64)[:,None];yy=cp.arange(n,dtype=cp.float64)
    yy=cp.where(yy<n/2,yy,yy-n)[None,:]
    offsets=cp.asarray(offsets,cp.float64)
    phase=2*np.pi/n*(offsets[...,0,None,None]*xx+offsets[...,1,None,None]*yy)
    c=cp.cos(phase).astype(cp.float32);s=cp.sin(phase).astype(cp.float32)
    re=(c*fs.real-s*fs.imag).astype(cp.float32);im=(c*fs.imag+s*fs.real).astype(cp.float32)
    return re+1j*im


def batch_size(views,n):
    # Bound both host and device buffers instead of loading the whole assembly.
    return max(1,min(64,128*1024**2//(views*n*n*4)))
