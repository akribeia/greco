# Third-party references and adapted components
# MotionCor3 detector-defect routines adapted for Greco: GDetectHot, CTemplate, GLocalCC, GDetectPatch and GCorrectBad.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/motioncor3/LICENSE.txt
#
# Upstream notice (retained):
# Copyright 2023 Chan Zuckerberg Institute for Advanced Biological Imaging
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""MC3 detector cleanup, before motion estimation and frame summation.

Port of GDetectHot, CTemplate/GLocalCC/GDetectPatch and GCorrectBad.
Detection operates on the gain-corrected unaligned sum. Replacement copies
one unmasked nearby detector pixel in each frame; it never averages frames.
Sparse pixel mappings can be saved and replayed identically in stage B.
"""
from __future__ import annotations

import numpy as np


def _backend(frames):
    if type(frames).__module__.split('.')[0] == 'cupy':
        import cupy as xp
        from cupyx.scipy import ndimage
        return xp, ndimage
    from scipy import ndimage
    return np, ndimage


def detect_defects(frames):
    """Return the MC3 hot/6x6-patch mask in the supplied detector raster."""
    xp, ndi = _backend(frames)
    if frames.ndim != 3 or min(frames.shape) < 1:
        raise ValueError('Expected nonempty frames (N,H,W)')
    image = frames.sum(axis=0, dtype=xp.float32)
    mean, std = image.mean(), image.std()
    hot = image > mean + xp.maximum(6*std, 10)
    kernel = np.full((6,6), -1, np.float32)
    kernel[1:-1,1:-1] = 1
    kernel = (kernel-kernel.mean()) / kernel.std()
    local_mean = ndi.uniform_filter(image, 6, origin=-3, mode='constant')
    variance = ndi.uniform_filter(image*image, 6, origin=-3, mode='constant')-local_mean*local_mean
    numerator = xp.abs(ndi.correlate(image, xp.asarray(kernel), origin=-3, mode='constant') / 36)
    cc = xp.zeros_like(image)
    valid = variance > 0
    cc[valid] = numerator[valid] / xp.sqrt(variance[valid])
    # The CUDA kernel rejects starts where x+6>=width or y+6>=height.
    cc[-6:,:] = 0
    cc[:,-6:] = 0
    seeds = cc > cc.mean()+6*cc.std()
    patches = ndi.binary_dilation(seeds, structure=xp.ones((6,6),bool), origin=(-3,-3))
    return hot | patches, dict(hot_count=int(hot.sum()), patch_seed_count=int(seeds.sum()))


def replacement_map(mask):
    """MC3's deterministic uint32 neighbor search, including padded stride."""
    if type(mask).__module__.split('.')[0] == 'cupy':
        import cupy as cp
        mask = cp.asnumpy(mask)
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    bad = np.column_stack(np.where(mask)).astype(np.int32)
    good = np.empty_like(bad)
    available = None
    for index,(y,x) in enumerate(bad):
        seed = ((int(y)*(w+2)+int(x))*109+619) & 0xffffffff
        found = False
        for j in range(10):
            size = 31+50*j
            state = seed
            for _ in range(500):
                state = (state*997) & 0xffffffff
                p = state % (size*size)
                gy, gx = p//size-size//2+int(y), p%size-size//2+int(x)
                if 0<=gy<h and 0<=gx<w and not mask[gy,gx]:
                    good[index] = gy,gx
                    found = True
                    break
            if found:
                break
            seed = (seed*997) & 0xffffffff
        if not found:
            # Unlike the upstream last-resort arbitrary copy, do not copy a
            # known bad pixel if all local searches failed.
            if available is None:
                available = np.column_stack(np.where(~mask))
            if not len(available):
                raise ValueError('No usable detector pixels remain')
            good[index] = available[seed % len(available)]
    return dict(bad_yx=bad, good_yx=good, shape=np.array(mask.shape,dtype=np.int32))


def apply_correction(frames, correction):
    """Apply saved sparse mapping in place; all sources are unmasked."""
    if tuple(correction['shape']) != tuple(frames.shape[-2:]):
        raise ValueError('Detector correction and movie shapes do not match')
    xp, _ = _backend(frames)
    bad = xp.asarray(correction['bad_yx'],dtype=xp.int32)
    good = xp.asarray(correction['good_yx'],dtype=xp.int32)
    if len(bad):
        frames[:,bad[:,0],bad[:,1]] = frames[:,good[:,0],good[:,1]]
    return frames


def prepare_correction(frames, *, flip_y=False, transpose=False, source_format='TIFF',valid_bounds_yx=None):
    """Detect in MC3's TIFF raster; return mapping in Greco's input raster.

    This changes only which detector pixels are copied, not the output
    image orientation or the user's tilt-axis/handedness configuration.
    """
    if flip_y and transpose:
        raise ValueError('flip_y and transpose are mutually exclusive')
    if valid_bounds_yx is not None:
        top,bottom,left,right=map(int,valid_bounds_yx)
        if (top,bottom,left,right)!=(0,frames.shape[1],0,frames.shape[2]):
            correction=prepare_correction(frames[:,top:bottom,left:right],flip_y=flip_y,
                                          transpose=transpose,source_format=source_format)
            for name in ('bad_yx','good_yx'):correction[name]+=np.array([top,left],np.int32)
            correction['shape']=np.array(frames.shape[-2:],dtype=np.int32)
            return correction
    original = frames.transpose(0,2,1) if transpose else (frames[:,::-1,:] if flip_y else frames)
    tiff_flip = source_format.upper() == 'TIFF'
    canonical = original[:,::-1,:] if tiff_flip else original
    mask, diagnostics = detect_defects(canonical)
    correction = replacement_map(mask)
    for name in ('bad_yx','good_yx'):
        coords = correction[name].copy()
        if tiff_flip:
            coords[:,0] = canonical.shape[1]-1-coords[:,0]
        if flip_y:
            coords[:,0] = original.shape[1]-1-coords[:,0]
        if transpose:
            coords = coords[:,::-1].copy()
        correction[name] = coords
    correction['shape'] = np.array(frames.shape[-2:],dtype=np.int32)
    correction['algorithm'] = np.array('mc3_detector_v1')
    correction.update({key:np.array(value) for key,value in diagnostics.items()})
    return correction
