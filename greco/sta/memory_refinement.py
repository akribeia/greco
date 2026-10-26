"""Full-precision accumulator staging; no Fourier shells are discarded."""
from greco.project_layout import join as storage_path
from contextlib import contextmanager
from pathlib import Path
import tempfile
import numpy as np


def memory_policy(shape,free_bytes):
    grids=2*int(np.prod(shape))*24
    staged=grids>.35*free_bytes
    if grids>.65*free_bytes:
        raise ValueError(f'Full-precision half accumulators need {grids/2**30:.1f} GiB; '
                         f'GPU has {free_bytes/2**30:.1f} GiB free. Use a larger-memory GPU or larger extraction binning.')
    return dict(staged=staged,two_half_accumulators_bytes=grids,free_gpu_bytes=int(free_bytes),
                accumulator_dtype='float64',full_frequency_validation=True)


def release_unused(backend):
    if backend.gpu:
        backend.xp.fft.config.get_plan_cache().clear()
        backend.xp.get_default_memory_pool().free_all_blocks()
        backend.xp.get_default_pinned_memory_pool().free_all_blocks()


class AccumulatorSpool:
    """Bounded transfers through a temporary disk-backed float64 mapping."""
    def __init__(self,output,backend):
        self.backend=backend;self.shape=tuple(backend.shape)
        self.directory=tempfile.TemporaryDirectory(prefix='.half-grid-',dir=output)
        self.paths={}
        self.slab=max(1,min(16,int(64*2**20/(self.shape[1]*self.shape[2]*8))))

    def store(self,half,arrays):
        path=storage_path(Path(self.directory.name), f'half{half}.dat')
        mapped=np.memmap(path,dtype=np.float64,mode='w+',shape=(3,*self.shape))
        try:
            for channel,array in enumerate(arrays):
                if array.shape!=self.shape or array.dtype!=np.float64:raise ValueError('Accumulator staging requires native float64 grids.')
                for first in range(0,self.shape[0],self.slab):
                    mapped[channel,first:first+self.slab]=self.backend.host(array[first:first+self.slab])
            mapped.flush()
        finally:mapped._mmap.close()
        self.paths[half]=path

    @contextmanager
    def acquire(self,half):
        mapped=np.memmap(self.paths[half],dtype=np.float64,mode='r',shape=(3,*self.shape))
        arrays=[]
        try:
            for channel in range(3):
                array=self.backend.xp.empty(self.shape,self.backend.xp.float64)
                arrays.append(array)
                for first in range(0,self.shape[0],self.slab):
                    array[first:first+self.slab]=self.backend.xp.asarray(mapped[channel,first:first+self.slab])
            del array
            yield arrays
        finally:
            arrays.clear();mapped._mmap.close();release_unused(self.backend)

    def close(self):self.directory.cleanup()
