"""Bounded GPU workspaces and ordered reductions for preprocessing CTF."""
from collections import OrderedDict
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=None)
def _kernel(name):
    import cupy as cp
    source = r'''
    extern "C" __global__ void background_sum(
        const float* image, float* output, int ny, int nx, int radius) {
        int p = blockDim.x * blockIdx.x + threadIdx.x;
        if (p >= ny * nx) return;
        int y = p / nx, x = p % nx;
        float value = 0.f;
        for (int k = -radius; k <= radius; ++k) {
            for (int j = -radius; j <= radius; ++j) {
                int xx = x + j;
                if (xx >= nx - 1) xx -= ny;
                int yy = y + k;
                if (xx < 0) { xx = -xx; yy = -yy; }
                yy = (yy % ny + ny) % ny;
                value += image[yy * nx + xx];
            }
        }
        output[p] = value;
    }
    extern "C" __global__ void spectrum_sum(
        const float* tiles, float* output, int size, int count) {
        int p = blockDim.x * blockIdx.x + threadIdx.x;
        if (p >= size) return;
        float value = tiles[p];
        for (int i = 1; i < count; ++i) value += tiles[(long long)i * size + p];
        output[p] = value;
    }
    extern "C" __global__ void scaled_spectrum_sum(
        const float* tiles, const int* selected, const int* yi,
        const int* xi, float* output, int ny, int nx, int count) {
        int p = blockDim.x * blockIdx.x + threadIdx.x;
        if (p >= ny * nx) return;
        int y = p / nx, x = p % nx;
        float value = 0.f;
        for (int i = 0; i < count; ++i) {
            long long offset = (long long)selected[i] * ny * nx;
            value += tiles[offset + yi[i * ny + y] * nx + xi[i * nx + x]];
        }
        output[p] = value;
    }
    '''
    # Preserve the sequential float32 additions of the reference loops.
    return cp.RawKernel(source, name, options=('--fmad=false',))


def background_sum(image, radius=17):
    import cupy as cp
    image = cp.ascontiguousarray(image, dtype=cp.float32)
    out = cp.empty_like(image)
    ny, nx = image.shape
    _kernel('background_sum')(((out.size + 255)//256,), (256,),
        (image, out, np.int32(ny), np.int32(nx), np.int32(radius)))
    return out


def spectrum_sum(tiles):
    import cupy as cp
    out = cp.empty(tiles.shape[1:], cp.float32)
    _kernel('spectrum_sum')(((out.size + 255)//256,), (256,),
        (tiles, out, np.int32(out.size), np.int32(len(tiles))))
    return out


def half_spectrum(estimator, plane, starts, retain_tiles):
    """Queue tile work without per-tile host synchronisation or FFT replanning.

    Individual 2-D FFTs and ordered float32 accumulation retain the reference
    arithmetic; only normalisation, uploads and downloads are grouped.
    """
    import cupy as cp
    device_input = isinstance(plane, cp.ndarray)
    source = cp.asarray(plane, dtype=cp.float32)
    ts = estimator.tile
    n = source.shape[0]
    y0, x0 = np.asarray(starts, dtype=np.int64).T
    ar = cp.arange(ts)
    free, _ = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    available = free + pool.total_bytes() - pool.used_bytes()
    batch = max(1, min(64, int(min(256*1024**2, available//8)//(ts*ts*20))))
    stds = np.empty(len(starts), np.float64)
    spectra = cp.empty((len(starts), ts, ts//2+1), cp.float32)
    for first in range(0, len(starts), batch):
        stop = min(first+batch, len(starts))
        if device_input:
            # CuPy selects a different reduction for a strided detector view
            # and its contiguous copy. Retain the original per-tile strides.
            tiles = [cp.asarray(plane[y:y+ts, x:x+ts], dtype=cp.float32)
                     for y, x in starts[first:stop]]
        else:
            tiles = source[cp.asarray(y0[first:stop])[:, None, None] + ar[None, :, None],
                           cp.asarray(x0[first:stop])[:, None, None] + ar[None, None, :]]
        means = cp.empty(stop-first, cp.float32)
        squares = cp.empty_like(means)
        for i in range(stop-first):
            means[i] = tiles[i].mean()
            squares[i] = (tiles[i] * tiles[i]).mean()
        m, m2 = means.get(), squares.get()
        std = np.sqrt(np.maximum(m2.astype(np.float64) - m.astype(np.float64)**2, 0.))
        stds[first:stop] = std
        divisors = cp.asarray(np.where(std > 0, std, 1).astype(np.float32))
        for i in range(stop-first):
            # Keep the original scalar-denominator 2-D normalisation path;
            # group its transfers rather than changing its arithmetic layout.
            tile = (tiles[i] - means[i]) / divisors[i]
            tile = tile * estimator.tile_win
            spectra[first+i] = cp.roll(cp.abs(cp.fft.rfft2(tile)), ts//2, axis=0)
        del tiles
    result = spectrum_sum(spectra) / len(starts)
    if retain_tiles:
        centers = np.column_stack((x0+(ts-1)/2-(n-1)/2, y0+(ts-1)/2-(n-1)/2))
        estimator.retained_tiles = (spectra.get(), centers, stds)
    return result


class TileSpectrumCache:
    """Operation-local LRU; never reserves more than 8 GiB or 25% of a GPU."""
    def __init__(self, xp):
        self.xp = xp
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.budget = 0
        if xp is not np:
            free, total = xp.cuda.runtime.memGetInfo()
            pool = xp.get_default_memory_pool()
            available = free + pool.total_bytes() - pool.used_bytes()
            self.budget = int(min(8 * 1024**3, total//4, available//2))
            self.device = xp.cuda.runtime.getDevice()

    def get(self, tiles):
        xp = self.xp
        if xp is not np and xp.cuda.runtime.getDevice() != self.device:
            raise ValueError('A tile spectrum cache cannot change CUDA device.')
        filename = getattr(tiles, 'filename', None)
        if filename is None or tiles.nbytes > self.budget:
            return xp.asarray(np.asarray(tiles), dtype=xp.float32, order='C')
        key = (self.device, str(filename), getattr(tiles, 'offset', 0),
               tiles.shape, tiles.dtype.str)
        if key in self.entries:
            result = self.entries.pop(key)
            self.entries[key] = result
            return result
        while self.entries and self.used_bytes + tiles.nbytes > self.budget:
            _, old = self.entries.popitem(last=False)
            self.used_bytes -= old.nbytes
            del old
        result = xp.asarray(np.asarray(tiles), dtype=xp.float32, order='C')
        self.entries[key] = result
        self.used_bytes += result.nbytes
        return result


def scaled_spectrum_sum(tiles, selected, yi, xi, xp, cache=None):
    out = xp.zeros(tiles.shape[1:], xp.float32)
    if not len(selected):
        return out
    stack = cache.get(tiles) if cache is not None else xp.asarray(np.asarray(tiles), dtype=xp.float32, order='C')
    ny, nx = out.shape
    _kernel('scaled_spectrum_sum')(((out.size + 255)//256,), (256,),
        (stack, xp.asarray(selected, dtype=xp.int32), xp.asarray(yi), xp.asarray(xi),
         out, np.int32(ny), np.int32(nx), np.int32(len(selected))))
    return out / len(tiles)
