"""Scaled half-complex references shared by all objectives of one half-map."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FourierReference:
    data: object
    scale: float
    shape: tuple

    @property
    def nbytes(self):return self.data.nbytes


PACK=r'''
#include <cuda_fp16.h>
extern "C" __global__ void pack_reference(const double2* input,__half2* output,
 long count,double scale){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 double2 v=input[i];output[i]=__floats2half2_rn((float)(v.x/scale),(float)(v.y/scale));
}
'''


def half_reference(fourier):
    """Own only packed data; do not retain the large original reference/FFT."""
    import cupy as cp
    if fourier.ndim!=3 or fourier.dtype!=cp.complex128:
        raise ValueError('Half reference packing requires a complex128 Fourier map.')
    maximum=cp.ReductionKernel('complex128 v','float64 result',
        'isfinite(v.real()) && isfinite(v.imag()) ? fmax(fabs(v.real()),fabs(v.imag())) : (1.0/0.0)',
        'max(a,b)','result=a','0','greco_reference_max')
    peak=float(maximum(fourier))
    if not np.isfinite(peak):raise ValueError('Nonfinite Fourier reference.')
    scale=max(peak/32752.,np.finfo(np.float64).tiny)
    source=cp.ascontiguousarray(fourier)
    packed=cp.empty((*fourier.shape,2),cp.float16)
    kernel=cp.RawKernel(PACK,'pack_reference',options=('--fmad=false',))
    kernel(((fourier.size+255)//256,),(256,),(source,packed,np.int64(fourier.size),np.float64(scale)))
    return FourierReference(packed,scale,tuple(fourier.shape))
