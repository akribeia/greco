# Third-party references and adapted components
# Greco PyTorch implementation of block-aligned halo tiling following CSBDeep inference conventions. Greco device handling and prediction pipeline are Greco contributions.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/csbdeep/LICENSE.txt
#
# Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
# Full license conditions and disclaimers are retained at the paths above.
#

"""Block-aligned halo tiling, equivalent to the established depth-two scheme."""
from itertools import product
import math

import numpy as np
import torch


def axis_plan(length, count, block=4, overlap=6):
    if length < 1 or block < 1 or overlap < 0 or length % block or count < 1:
        raise ValueError('Invalid TinyCARE tile grid.')
    n = length // block
    if count == 1:
        return [(slice(0, length), slice(0, length), slice(0, length))]

    def generate(size):
        halo = overlap if n >= size else 0
        size = min(n, size)
        start = written = 0
        tiles = []
        while True:
            stop = start + size
            end = n if stop == n else (size - halo if not tiles else written + size - 2 * halo)
            tiles.append((slice(start * block, stop * block),
                          slice(written * block, end * block),
                          slice((written - start) * block, (end - start) * block)))
            written = end
            if stop == n:
                return tiles
            start = min(stop - 2 * halo, n - size)

    minimum = 2 * overlap + 1
    size = minimum
    while len(generate(size)) > count:
        size += 1
    if size == minimum:
        return generate(size)
    candidates = (generate(size - 1), generate(size))
    return min(candidates, key=lambda plan: abs(len(plan) - count))


def _predict_tile(model, patch, crop, device, mean, std):
    tensor = torch.as_tensor(np.ascontiguousarray(patch), device=device)[None, None]
    tensor = ((tensor - mean) / std).contiguous(memory_format=torch.channels_last_3d)
    return model(tensor)[0, 0][crop].float().cpu().numpy() * std + mean


@torch.inference_mode()
def predict_volume(model, volume, mean, std, *, tile_target=256, precision='fp16'):
    if (isinstance(tile_target, bool) or not isinstance(tile_target, int) or tile_target < 32
            or not np.isfinite(mean) or not np.isfinite(std) or std <= 0):
        raise ValueError('Invalid TinyCARE prediction settings.')
    if precision not in ('fp16', 'fp32'):
        raise ValueError('TinyCARE precision must be fp16 or fp32.')
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or not all(volume.shape):
        raise ValueError('TinyCARE prediction requires a nonempty 3D volume.')
    shape = volume.shape
    padded = np.pad(volume, [(0, (-n) % 4) for n in shape], mode='reflect')
    device = next(model.parameters()).device
    counts = [max(1, math.ceil(n / tile_target)) for n in padded.shape]
    result = np.empty_like(padded, dtype=np.float32)
    while True:
        plans = [axis_plan(n, t) for n, t in zip(padded.shape, counts)]
        try:
            for tile in product(*plans):
                read, write, crop = zip(*tile)
                # Transfer only a tile, not a second complete normalized volume.
                with torch.autocast(device.type, dtype=torch.float16,
                                    enabled=device.type == 'cuda' and precision == 'fp16'):
                    result[write] = _predict_tile(model, padded[read], crop, device, mean, std)
            return result[tuple(slice(0, n) for n in shape)], counts + [1]
        except torch.OutOfMemoryError:
            if device.type != 'cuda':
                raise
            # A smaller core with the same halo preserves the coordinate grid.
            axis = max(range(3), key=lambda i: padded.shape[i] / counts[i])
            if padded.shape[axis] / counts[axis] <= 64:
                raise
            counts[axis] *= 2
        # Leave the exception handler first: its traceback can retain failed
        # convolution activations. Release them before emptying the GPU cache.
        torch.cuda.empty_cache()
