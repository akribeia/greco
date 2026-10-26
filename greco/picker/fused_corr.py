"""Fused FP32 correlation updates, compiled for the active GPU by NVRTC.

The installed CuPy compiler/cache supplies the runtime toolchain. Torch owns
all buffers; the kernel runs on Torch's current stream without copying them.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from .roi import cyclic_intervals

BLOCK=256


@lru_cache(maxsize=16)
def _kernel(device):
    import cupy as cp
    source=Path(__file__).with_name('fused_corr_kernel.cu').read_text()
    with cp.cuda.Device(device):
        kernel=cp.RawKernel(source,'fused_corr_kernel',options=('--fmad=false','--prec-div=true'))
        kernel.compile()
    return kernel


def _view(tensor):
    import cupy as cp
    dtype={torch.float32:np.float32,torch.float64:np.float64,torch.int32:np.int32}[tensor.dtype]
    owner=cp.cuda.UnownedMemory(tensor.data_ptr(),tensor.numel()*tensor.element_size(),tensor,
                              device_id=tensor.device.index)
    return cp.ndarray(tensor.shape,dtype=dtype,memptr=cp.cuda.MemoryPointer(owner,0))


def make_roi_cfg(fast_shape,tomo_shape,device):
    cfg=torch.zeros((3,2,2),dtype=torch.int32)
    for axis,intervals in enumerate(cyclic_intervals(fast_shape,tomo_shape)):
        for index,interval in enumerate(intervals):cfg[axis,index]=torch.tensor(interval,dtype=torch.int32)
    return cfg.to(device)


def update(ccc,std,scores,angles,ang_ids,out_max,sq_part,roi_cfg,out_min,
           b,std_row_stride,has_min=False,fft_scale=1.0):
    import cupy as cp
    if not scores.is_cuda or scores.ndim!=3:raise ValueError('Picker scores must be a CUDA volume.')
    shape=tuple(scores.shape);device=scores.device
    if b<1 or tuple(ccc.shape)!=(b,*shape):raise ValueError('Invalid correlation batch shape.')
    if tuple(std.shape) not in ((1,*shape),(b,*shape)):raise ValueError('Invalid standard-deviation shape.')
    if std_row_stride!=(0 if std.shape[0]==1 else std.stride(0)):raise ValueError('Invalid standard-deviation stride.')
    grid=(scores.numel()+BLOCK-1)//BLOCK
    tensors=[(ccc,torch.float32),(std,torch.float32),(scores,torch.float32),
             (ang_ids,torch.float32),(out_max,torch.float32),(roi_cfg,torch.int32)]
    if angles is not None:tensors.append((angles,torch.float32))
    if sq_part is not None:tensors.append((sq_part,torch.float64))
    if has_min:
        if out_min is None:raise ValueError('Missing minimum output.')
        tensors.append((out_min,torch.float32))
    for tensor,dtype in tensors:
        if tensor.device!=device or tensor.dtype!=dtype or not tensor.is_contiguous():
            raise ValueError('Fused picker buffers must be contiguous, correctly typed and on one GPU.')
    if ang_ids.shape!=(b,) or out_max.shape!=scores.shape or (angles is not None and angles.shape!=scores.shape):
        raise ValueError('Invalid picker output dimensions.')
    if roi_cfg.shape!=(3,2,2) or (sq_part is not None and sq_part.shape!=(grid,)):
        raise ValueError('Invalid picker reduction buffers.')
    if has_min and out_min.shape!=scores.shape:raise ValueError('Invalid picker minimum dimensions.')
    args=(ccc,std,scores,angles if angles is not None else scores,ang_ids,out_max,
          out_min if has_min else out_max,sq_part if sq_part is not None else out_max,roi_cfg)
    stream=torch.cuda.current_stream(device).cuda_stream
    with cp.cuda.Device(device.index),cp.cuda.ExternalStream(stream,device_id=device.index):
        arrays=tuple(_view(t) for t in args)
        _kernel(device.index)((grid,),(BLOCK,),arrays+tuple(np.int32(v) for v in (*shape,b))+
            (np.int64(ccc.stride(0)),np.int64(std_row_stride),np.int32(angles is not None),
             np.int32(has_min),np.int32(sq_part is not None),np.float32(fft_scale)))


def warmup():
    """Require successful compilation and execution before enabling the path."""
    device=torch.device('cuda',torch.cuda.current_device());shape=(3,5,7);b=2
    ccc=torch.zeros((b,*shape),device=device);std=torch.ones((1,*shape),device=device)
    scores=torch.full(shape,float('-inf'),device=device);angles=scores.clone()
    ids=torch.arange(b,device=device,dtype=torch.float32);maximum=torch.empty_like(scores)
    partial=torch.empty((1,),device=device,dtype=torch.float64)
    cfg=make_roi_cfg(shape,shape,device)
    update(ccc,std,scores,angles,ids,maximum,partial,cfg,None,b,0)
    torch.cuda.synchronize(device)
    if not bool((scores==0).all()) or float(partial.sum())!=0:
        raise RuntimeError('Fused picker warmup failed.')
