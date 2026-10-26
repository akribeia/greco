"""Exact periodic patch gathering without host-sized index volumes."""
from functools import lru_cache
import logging
import numpy as np
import torch

logger=logging.getLogger(__name__)
_SOURCE=r'''
extern "C" __global__ void periodic_patch(
 const float* volume, const int* base, float* output,
 long long count, int width, int sy, int sx, int sz) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=count)return;
 long long q=i;
 int x=q%width; q/=width;
 int y=q%width; q/=width;
 int z=q%width; q/=width;
 int iz=((long long)base[3*q]+z)%sz;
 int iy=((long long)base[3*q+1]+y)%sy;
 int ix=((long long)base[3*q+2]+x)%sx;
 output[i]=volume[((long long)iz*sy+iy)*sx+ix];
}
'''

@lru_cache(maxsize=16)
def _kernel(device):
    try:
        import cupy as cp
        with cp.cuda.Device(device):
            kernel=cp.RawKernel(_SOURCE,'periodic_patch')
            kernel.compile()
        return kernel
    except Exception as error:
        logger.warning('Periodic patch kernel unavailable; using Torch indexing: %s',error)
        return None


def periodic_patches(pos,volume,shape,nt,padding=0):
    """Gather the same output-frame periodic patches as the CPU index path.

    Only (N,3) origins cross to the GPU. Torch owns all storage and the
    optional NVRTC kernel runs on its current stream; no arithmetic is
    performed on sampled values. Padding adds a symmetric envelope.
    """
    shape=tuple(int(v) for v in shape);nt=int(nt);padding=int(padding)
    positions=np.asarray(pos)
    if len(shape)!=3 or min(shape)<1 or nt<1 or padding<0:
        raise ValueError('Invalid periodic patch dimensions.')
    if positions.ndim!=2 or positions.shape[1]!=3 or not np.issubdtype(positions.dtype,np.integer):
        raise ValueError('Patch positions must be an integer (N,3) array.')
    if volume.ndim!=1 or volume.numel()!=int(np.prod(shape)):
        raise ValueError('Filtered volume shape mismatch.')
    fs=np.asarray(shape,dtype=np.int64)
    shifted=(positions.astype(np.int64,copy=False)-(fs//2+1))%fs
    pre=(fs-1-shifted)%fs
    base=((fs//2-nt//2)[None,:]-pre-padding)%fs
    width=nt+2*padding
    origins=torch.from_numpy(np.ascontiguousarray(base,dtype=np.int32)).to(volume.device)
    output_shape=(len(positions),width,width,width)
    if len(positions)==0:return torch.empty(output_shape,device=volume.device,dtype=volume.dtype)
    kernel=_kernel(volume.device.index) if volume.is_cuda and volume.dtype==torch.float32 and volume.is_contiguous() else None
    if kernel is not None:
        output=torch.empty(output_shape,device=volume.device,dtype=volume.dtype)
        import cupy as cp
        stream=torch.cuda.current_stream(volume.device).cuda_stream
        with cp.cuda.Device(volume.device.index),cp.cuda.ExternalStream(stream,device_id=volume.device.index):
            arrays=tuple(cp.from_dlpack(t) for t in (volume,origins,output))
            kernel(((output.numel()+255)//256,),(256,),arrays+(np.int64(output.numel()),np.int32(width),np.int32(shape[1]),np.int32(shape[2]),np.int32(shape[0])))
    else:
        offsets=torch.arange(width,device=volume.device,dtype=torch.int64)
        z=(origins[:,0,None].long()+offsets)%shape[0]
        y=(origins[:,1,None].long()+offsets)%shape[1]
        x=(origins[:,2,None].long()+offsets)%shape[2]
        indices=(z[:,:,None,None]*shape[1]+y[:,None,:,None])*shape[2]+x[:,None,None,:]
        output=volume[indices]
    return output
