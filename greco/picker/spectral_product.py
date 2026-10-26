"""Share each tomogram frequency across a batch of template spectra."""
from functools import lru_cache
import numpy as np
import torch

_SOURCE=r'''
// CuPy adds -ftz=true; explicit PTX keeps Torch's IEEE subnormal behavior.
__device__ __forceinline__ float ieee_mul(float a,float b) {
 float result;asm("mul.rn.f32 %0,%1,%2;":"=f"(result):"f"(a),"f"(b));return result;
}
__device__ __forceinline__ float ieee_fma(float a,float b,float c) {
 float result;asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(result):"f"(a),"f"(b),"f"(c));return result;
}
__device__ __forceinline__ float flip_sign(float value) {
 return __int_as_float(__float_as_int(value)^0x80000000u);
}
extern "C" __global__ void shared_spectral_product(
 float2* spectra,const float2* volume,long long count,int rows,int conjugate) {
 long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(q>=count)return;
 float2 factor=volume[q];if(conjugate)factor.y=flip_sign(factor.y);
 for(int row=0;row<rows;++row) {
  long long i=(long long)row*count+q;
  float2 value=spectra[i],result;
  // Match the FP32 operation order of Torch 2.10 complex multiplication.
  result.x=ieee_fma(value.x,factor.x,flip_sign(ieee_mul(value.y,factor.y)));
  result.y=ieee_fma(value.x,factor.y,ieee_mul(value.y,factor.x));
  spectra[i]=result;
 }
}
'''

@lru_cache(maxsize=16)
def _kernel(device):
    import cupy as cp
    with cp.cuda.Device(device):
        kernel=cp.RawKernel(_SOURCE,'shared_spectral_product',options=('--fmad=false',))
        kernel.compile()
    return kernel


def multiply_(spectra,volume):
    """Multiply in place, or return False for layouts requiring Torch.

    Torch owns all buffers. A conjugate view of the volume is read from its
    physical storage with the imaginary sign applied by the kernel.
    """
    if (not spectra.is_cuda or spectra.dtype!=torch.complex64 or spectra.ndim!=4
            or volume.dtype!=torch.complex64 or volume.device!=spectra.device
            or tuple(spectra.shape[-3:])!=tuple(volume.shape)
            or not spectra.is_contiguous() or not volume.is_contiguous()
            or spectra.is_conj() or spectra.is_neg() or volume.is_neg()
            or spectra.requires_grad or volume.requires_grad):
        return False
    if spectra.numel()==0:return True
    import cupy as cp
    conjugate=volume.is_conj();physical=volume.conj() if conjugate else volume
    stream=torch.cuda.current_stream(spectra.device).cuda_stream
    with cp.cuda.Device(spectra.device.index),cp.cuda.ExternalStream(stream,device_id=spectra.device.index):
        arrays=(cp.from_dlpack(spectra),cp.from_dlpack(physical))
        count=volume.numel()
        _kernel(spectra.device.index)(((count+255)//256,),(256,),arrays+(np.int64(count),np.int32(spectra.shape[0]),np.int32(conjugate)))
    return True


def warmup():
    """Compile and require parity with the installed Torch arithmetic."""
    device=torch.device('cuda',torch.cuda.current_device())
    generator=torch.Generator(device=device).manual_seed(941)
    volume=torch.randn((3,5,7),device=device,dtype=torch.complex64,generator=generator)
    values=torch.randn((3,3,5,7),device=device,dtype=torch.complex64,generator=generator)
    # Check subnormal behavior too, without changing global RNG state.
    smallest=float(np.nextafter(np.float32(0),np.float32(1)))
    tiny=float(np.finfo(np.float32).tiny)
    volume.reshape(-1)[:4]=torch.tensor([complex(1e20,-1e20),complex(1,1),
        complex(1e-20,0),complex(0,0)],device=device,dtype=torch.complex64)
    values.reshape(3,-1)[:,:4]=torch.tensor([complex(smallest,smallest),
        complex(tiny,-tiny),complex(1e-20,-1e-20),complex(-0.,0.)],
        device=device,dtype=torch.complex64)
    for factor in [volume,volume.conj()]:
        expected=values*factor;actual=values.clone()
        if not multiply_(actual,factor):raise RuntimeError('Spectral warmup layout unsupported.')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.cuda.synchronize()
