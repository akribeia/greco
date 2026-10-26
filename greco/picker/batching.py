"""Picker allocator setup and scoped, executable FFT memory probes."""
import logging
import os
import torch

logger=logging.getLogger(__name__)


def configure_allocator():
    if 'PYTORCH_ALLOC_CONF' not in os.environ and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        if torch.cuda.is_initialized():
            logger.debug('CUDA already initialized; keeping the active allocator.')
        else:os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'


def noise_fused(batch,shape,device,noise_correction,noise_every):
    total=torch.cuda.get_device_properties(device).total_memory
    spectrum=shape[0]*shape[1]*(shape[2]//2+1)*8
    return bool(noise_correction and noise_every==1 and not os.environ.get('GRECO_PICKER_NO_FUSE')
                and 2*batch*spectrum<=.45*total)


def probe_batch(batch,shape,device,noise_correction,noise_every,rotating_mask):
    """All allocations die before the caller retries, including on cuFFT OOM."""
    buffers=[]
    try:
        fused=noise_fused(batch,shape,device,noise_correction,noise_every)
        buffers.append(torch.zeros((batch*(2 if fused else 1),*shape),device=device))
        buffers.append(torch.fft.rfftn(buffers[0],dim=(-3,-2,-1)))
        buffers.append(torch.fft.irfftn(buffers[1][:batch],s=shape,dim=(-3,-2,-1)))
        # Normalization/update workspace; cuFFT plans are exercised above.
        for _ in range(2):buffers.append(torch.empty((batch,*shape),device=device))
        if rotating_mask:
            buffers.append(torch.empty((batch,*shape),device=device))
            buffers.append(torch.empty((batch,shape[0],shape[1],shape[2]//2+1),device=device,dtype=torch.complex64))
            for _ in range(3):buffers.append(torch.empty((batch,*shape),device=device))
        if noise_correction and not fused:
            count=min(batch,(batch+noise_every-1)//noise_every)
            buffers.append(torch.empty((count,*shape),device=device))
            buffers.append(torch.empty((count,shape[0],shape[1],shape[2]//2+1),device=device,dtype=torch.complex64))
            buffers.append(torch.empty((count,*shape),device=device))
        torch.cuda.synchronize(device)
    finally:buffers.clear()


def fit_batch(initial,probe,release,minimum=1):
    batch=int(initial)
    if batch<minimum:raise ValueError('Invalid picker batch size.')
    while True:
        failed=False
        try:probe(batch)
        except torch.cuda.OutOfMemoryError:
            failed=True
            if batch<=minimum:raise
        finally:
            # Run after the exception/traceback scope has released probe frames.
            release()
        if not failed:return batch
        next_batch=max(minimum,batch//2)
        logger.warning('FFT batch %s did not fit; retrying with %s.',batch,next_batch)
        batch=next_batch
