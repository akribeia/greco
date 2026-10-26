# Greco optimizer — implementation of the GRECO algorithm
# GRECO: Geometric Registration of Electron Cryotomography Observations
# Copyright (C) 2026 Carlos Fernández Rodríguez
#
# This file is part of Greco.
#
# Greco is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3 only.
#
# Greco is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Greco. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-only

"""One-pair geometric column refinement against a fixed central reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


_interior_cylinder_cache = {}

@dataclass(frozen=True)
class ColumnRefinementResult:
    correction_yx: tuple[float, float]
    score: float
    at_boundary: bool
    maxima_zyx: NDArray[np.int32]
    projection_yx: NDArray[np.float32]
    projection_mask_yx: NDArray[np.float32]
    correlation: NDArray[np.float64]
    shifts_y: NDArray[np.int64]
    shifts_x: NDArray[np.int64]


@dataclass(frozen=True)
class MaximaQuality:
    selected_indices: NDArray[np.int32]
    quality: NDArray[np.float32]
    support_fraction: NDArray[np.float32]
    per_view_scores: NDArray[np.float32]




def _correlation_fft_dimension(value: int) -> int:
    """Smallest integer >= value whose prime factors are only 2, 3 and 5."""
    candidate = int(value)
    while True:
        remainder = candidate
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


def masked_normxcorr2(
    fixed_image: ArrayLike,
    moving_image: ArrayLike,
    fixed_mask: ArrayLike,
    moving_mask: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Padfield masked normalized cross-correlation in NumPy."""
    fixed=np.asarray(fixed_image,dtype=np.float64).copy()
    moving=np.asarray(moving_image,dtype=np.float64).copy()
    fm=(np.asarray(fixed_mask)>0).astype(np.float64)
    mm=(np.asarray(moving_mask)>0).astype(np.float64)
    if fixed.ndim!=2 or moving.ndim!=2 or fm.shape!=fixed.shape or mm.shape!=moving.shape:
        raise ValueError('images and masks must be matching 2-D arrays')
    fixed-=min(float(fixed.min()),0.0); moving-=min(float(moving.min()),0.0)
    fixed*=fm; moving*=mm
    moving=np.rot90(moving,2); mm=np.rot90(mm,2)
    combined=(fixed.shape[0]+moving.shape[0]-1,fixed.shape[1]+moving.shape[1]-1)
    shape=tuple(_correlation_fft_dimension(value) for value in combined)
    fft=lambda image: np.fft.fft2(image,s=shape)
    ff,mf,fmask,mmask=fft(fixed),fft(moving),fft(fm),fft(mm)
    overlap=np.rint(np.fft.ifft2(mmask*fmask).real)
    safe=np.maximum(overlap,np.finfo(np.float64).eps)
    sum_fixed=np.fft.ifft2(mmask*ff).real
    sum_moving=np.fft.ifft2(fmask*mf).real
    numerator=np.fft.ifft2(mf*ff).real-sum_fixed*sum_moving/safe
    fixed_denom=np.maximum(np.fft.ifft2(mmask*fft(fixed*fixed)).real-sum_fixed**2/safe,0)
    moving_denom=np.maximum(np.fft.ifft2(fmask*fft(moving*moving)).real-sum_moving**2/safe,0)
    denominator=np.sqrt(fixed_denom*moving_denom)
    correlation=np.zeros(shape,dtype=np.float64)
    tolerance=1000*np.spacing(float(np.max(np.abs(denominator))))
    valid=denominator>tolerance
    correlation[valid]=numerator[valid]/denominator[valid]
    correlation=np.clip(correlation,-1,1)
    crop=np.s_[:combined[0],:combined[1]]
    return correlation[crop],safe[crop]


def masked_normxcorr2_gpu(fixed_image, moving_image, fixed_mask, moving_mask):
    """Device implementation of masked_normxcorr2 (float64 parity)."""
    import cupy as cp
    fixed = cp.asarray(fixed_image, cp.float64).copy()
    moving = cp.asarray(moving_image, cp.float64).copy()
    fm = (cp.asarray(fixed_mask) > 0).astype(cp.float64)
    mm = (cp.asarray(moving_mask) > 0).astype(cp.float64)
    if fixed.ndim != 2 or moving.ndim != 2 or fm.shape != fixed.shape or mm.shape != moving.shape:
        raise ValueError('images and masks must be matching 2-D arrays')
    fixed -= cp.minimum(fixed.min(), 0.0); moving -= cp.minimum(moving.min(), 0.0)
    fixed *= fm; moving *= mm
    moving = cp.rot90(moving, 2); mm = cp.rot90(mm, 2)
    combined = (fixed.shape[0] + moving.shape[0] - 1,
                fixed.shape[1] + moving.shape[1] - 1)
    shape = tuple(_correlation_fft_dimension(value) for value in combined)
    fft = lambda image: cp.fft.fft2(image, s=shape)
    ff, mf, fmask, mmask = fft(fixed), fft(moving), fft(fm), fft(mm)
    overlap = cp.rint(cp.fft.ifft2(mmask * fmask).real)
    safe = cp.maximum(overlap, cp.finfo(cp.float64).eps)
    sum_fixed = cp.fft.ifft2(mmask * ff).real
    sum_moving = cp.fft.ifft2(fmask * mf).real
    numerator = cp.fft.ifft2(mf * ff).real - sum_fixed * sum_moving / safe
    fixed_denom = cp.maximum(cp.fft.ifft2(mmask * fft(fixed * fixed)).real
                             - sum_fixed**2 / safe, 0)
    moving_denom = cp.maximum(cp.fft.ifft2(fmask * fft(moving * moving)).real
                              - sum_moving**2 / safe, 0)
    denominator = cp.sqrt(fixed_denom * moving_denom)
    tolerance = 1000 * np.spacing(float(cp.max(cp.abs(denominator)).get()))
    correlation = cp.where(denominator > tolerance, numerator / denominator, 0.0)
    correlation = cp.clip(correlation, -1, 1)
    crop = np.s_[:combined[0], :combined[1]]
    return cp.asnumpy(correlation[crop]), cp.asnumpy(safe[crop])


def masked_translation(
    fixed_image: ArrayLike,
    moving_image: ArrayLike,
    fixed_mask: ArrayLike,
    moving_mask: ArrayLike,
    *,
    overlap_ratio: float=0.95,
    search_radius: int=20,
    search_sigma: float | None=None,
    search_center_yx: tuple[float, float]=(0.0,0.0),
    use_gpu: bool=False,
) -> tuple[tuple[float,float],float,NDArray[np.float64],bool]:
    """Port of ``MaskedTranslationRegistration.m`` including circular search."""
    correlation,overlap=(masked_normxcorr2_gpu(
        fixed_image,moving_image,fixed_mask,moving_mask) if use_gpu else
        masked_normxcorr2(fixed_image,moving_image,fixed_mask,moving_mask))
    correlation=correlation.copy()
    correlation[overlap < overlap_ratio*float(np.max(overlap))]=0.0
    centre=np.asarray(np.asarray(moving_image).shape,dtype=np.int64)-1
    yy,xx=np.ogrid[:correlation.shape[0],:correlation.shape[1]]
    search=(yy-centre[0])**2+(xx-centre[1])**2 <= search_radius**2
    correlation[~search]=0.0
    if search_sigma is not None:
        if search_sigma<=0: raise ValueError('search_sigma must be positive')
        # Input rasters already contain the frozen robust 2-D alignment.  The
        # centre is the accepted 3-D residual of the nearest interior tilt;
        # this remains a soft prior, not a reduced search radius.
        prior_y=centre[0]+float(search_center_yx[0])
        prior_x=centre[1]+float(search_center_yx[1])
        prior=np.exp(-((yy-prior_y)**2+(xx-prior_x)**2)/(2*search_sigma**2))
        correlation*=prior
    peak_index=np.unravel_index(int(np.argmax(correlation)),correlation.shape)
    # This is the transform convention used by the masked correlation transform.  A
    # raster correction applied with scipy.ndimage.shift has the opposite sign.
    transform=(float(centre[0]-peak_index[0]),float(centre[1]-peak_index[1]))
    boundary=bool(np.hypot(*transform)>=search_radius-0.5)
    return transform,float(correlation[peak_index]),correlation,boundary


def regular_column_centres(size: int, *, cell: int = 32, margin: int = 32) -> NDArray[np.int32]:
    """Centres of 32-pixel cells wholly inside a square interior support."""
    if size % cell or margin < 0 or 2 * margin >= size:
        raise ValueError("invalid regular-grid geometry")
    first = margin + cell // 2
    last = size - margin - cell // 2
    values = np.arange(first, last + 1, cell, dtype=np.int32)
    yy, xx = np.meshgrid(values, values, indexing="ij")
    return np.column_stack((yy.ravel(), xx.ravel()))


def project_points_aligned_axis0(points_zyx: ArrayLike, angle_deg: float, size: int) -> NDArray[np.float64]:
    """Project ``(z,y,x)`` points into an aligned vertical-axis raster."""
    points=np.asarray(points_zyx,dtype=np.float64)
    centre=(size-1)/2.0; radians=np.deg2rad(float(angle_deg))
    u=centre+(points[:,2]-centre)*np.cos(radians)-(points[:,0]-centre)*np.sin(radians)
    return np.column_stack((points[:,1],u))


_median_pair_kernel = None
_absolute_deviation_kernel = None


def _median_last_axis_gpu(values):
    """Match NumPy's float32 middle-pair rounding, including subnormals."""
    import cupy as cp
    global _median_pair_kernel
    ordered = cp.sort(values, axis=-1)
    width = values.shape[-1]
    if width % 2:
        median = ordered[..., width // 2]
    else:
        if _median_pair_kernel is None:
            # PTX without .ftz retains subnormals despite CuPy compiler defaults.
            # The PTX float literal 0f3f000000 is exactly 0.5f.
            _median_pair_kernel = cp.ElementwiseKernel(
                'float32 a, float32 b', 'float32 result',
                r'float sum; asm("add.rn.f32 %0, %1, %2;" : "=f"(sum) : "f"(a), "f"(b)); '
                r'asm("mul.rn.f32 %0, %1, 0f3f000000;" : "=f"(result) : "f"(sum));',
                'greco_median_pair', options=('--fmad=false',))
        median = _median_pair_kernel(ordered[..., width // 2 - 1],
                                     ordered[..., width // 2])
    return cp.where(cp.isnan(ordered[..., -1]), ordered[..., -1], median)


def _background_statistics(background, backend):
    """Return median and MAD; keep score arithmetic on the reference CPU path."""
    if backend == 'gpu' and background.size:
        import cupy as cp
        values = cp.asarray(background)
        median = _median_last_axis_gpu(values)
        global _absolute_deviation_kernel
        if _absolute_deviation_kernel is None:
            _absolute_deviation_kernel = cp.ElementwiseKernel(
                'float32 value, float32 median', 'float32 result',
                r'float delta; asm("sub.rn.f32 %0, %1, %2;" : "=f"(delta) : "f"(value), "f"(median)); '
                r'result = __uint_as_float(__float_as_uint(delta) & 0x7fffffffU);',
                'greco_absolute_deviation', options=('--fmad=false',))
        deviation = _absolute_deviation_kernel(values, median[..., None])
        mad = _median_last_axis_gpu(deviation)
        result = cp.asnumpy(cp.stack((median, mad)))
        return result[0], result[1]
    median = np.median(background, axis=-1)
    return median, np.median(np.abs(background - median[..., None]), axis=-1)


def rank_maxima_multiview(
    maxima_zyx: ArrayLike,
    stack_yx: ArrayLike,
    angles_deg: ArrayLike,
    *,
    excluded_index: int | None,
    patch_radius: int = 4,
    background_radius: int = 8,
    keep: int = 48,
    min_support_fraction: float = 0.8,
    min_view_score: float = 1.0,
    support_search_radius: int = 0,
    backend: str = 'cpu',
) -> MaximaQuality:
    """Rank volume maxima by positive local contrast reproduced across views.

    Candidate positions come exclusively from the leave-one-out volume.  Their
    quality is the median locally standardized contrast above
    ``min_view_score`` over the remaining experimental tilts; support is the
    fraction passing that threshold. The target is never inspected. The GPU
    backend accelerates only search-window background median/MAD statistics;
    CPU patch means, score arithmetic and ranking retain their reference order.
    """
    if backend not in ('cpu', 'gpu'):
        raise ValueError("multiview backend must be 'cpu' or 'gpu'")
    maxima=np.asarray(maxima_zyx,dtype=np.int32)
    stack=np.asarray(stack_yx,dtype=np.float32); angles=np.asarray(angles_deg,dtype=np.float64)
    if stack.ndim!=3 or len(stack)!=len(angles): raise ValueError('stack/angle mismatch')
    n_points=len(maxima); scores=[]
    yy,xx=np.mgrid[-background_radius:background_radius+1,-background_radius:background_radius+1]
    inner=yy**2+xx**2<=patch_radius**2
    annulus=(yy**2+xx**2>patch_radius**2)&(yy**2+xx**2<=background_radius**2)
    inner_y,inner_x=yy[inner],xx[inner]
    background_y,background_x=yy[annulus],xx[annulus]
    for view,(image,angle) in enumerate(zip(stack,angles,strict=True)):
        if excluded_index is not None and view==excluded_index: continue
        projected=project_points_aligned_axis0(maxima,float(angle),image.shape[0])
        view_scores=np.full(n_points,np.nan,dtype=np.float32)
        r=background_radius
        if support_search_radius:
            base=np.rint(projected).astype(np.int64)
            offsets=np.arange(-support_search_radius,support_search_radius+1,dtype=np.int64)
            oy,ox=np.meshgrid(offsets,offsets,indexing='ij')
            cy=base[:,0,None,None]+oy;cx=base[:,1,None,None]+ox
            valid=(cy-r>=0)&(cx-r>=0)&(cy+r<image.shape[0])&(cx+r<image.shape[1])
            safe_y=np.clip(cy,r,image.shape[0]-r-1);safe_x=np.clip(cx,r,image.shape[1]-r-1)
            # Gather only the disk and annulus, with sample offsets first.
            # Moving that axis last preserves the reference's strided mean
            # reduction order, including float32 rounding and signed zero.
            centre_samples=np.moveaxis(image[
                safe_y[None]+inner_y[:,None,None,None],
                safe_x[None]+inner_x[:,None,None,None]],0,-1)
            centre_value=np.mean(centre_samples,axis=-1)
            background=np.moveaxis(image[
                safe_y[None]+background_y[:,None,None,None],
                safe_x[None]+background_x[:,None,None,None]],0,-1)
            background_median, background_mad = _background_statistics(background, backend)
            scale=1.4826*background_mad+1e-6
            candidate_scores=np.where(valid,(centre_value-background_median)/scale,-np.inf)
            best=np.max(candidate_scores,axis=(1,2))
            view_scores[np.isfinite(best)]=best[np.isfinite(best)]
            scores.append(view_scores)
            continue
        for k,(py,px) in enumerate(projected):
            base_y,base_x=int(round(py)),int(round(px))
            best=-np.inf
            for oy in range(-support_search_radius,support_search_radius+1):
                for ox in range(-support_search_radius,support_search_radius+1):
                    cy,cx=base_y+oy,base_x+ox
                    if cy-r<0 or cx-r<0 or cy+r>=image.shape[0] or cx+r>=image.shape[1]: continue
                    patch=image[cy-r:cy+r+1,cx-r:cx+r+1]
                    centre_value=float(np.mean(patch[inner])); background=patch[annulus]
                    scale=1.4826*float(np.median(np.abs(background-np.median(background))))+1e-6
                    best=max(best,(centre_value-float(np.median(background)))/scale)
            if np.isfinite(best): view_scores[k]=best
        scores.append(view_scores)
    per_view=np.asarray(scores,dtype=np.float32).T
    valid=np.isfinite(per_view)
    supported=(per_view>min_view_score)&valid
    support=(np.sum(supported,axis=1)/np.maximum(np.sum(valid,axis=1),1)).astype(np.float32)
    quality=np.zeros(n_points,dtype=np.float32)
    for point_index,row in enumerate(per_view):
        positive=row[np.isfinite(row)&(row>min_view_score)]
        if len(positive): quality[point_index]=np.float32(np.median(positive))
    quality*=support
    eligible=np.flatnonzero(support>=min_support_fraction)
    order=eligible[np.argsort(quality[eligible])[::-1]] if len(eligible) else np.empty(0,dtype=np.int64)
    selected=order[:min(keep,len(order))].astype(np.int32)
    return MaximaQuality(selected,quality,support,per_view)


def _interior_cylinder_gpu(cp: Any, size: int, margin: int) -> Any:
    key = (int(cp.cuda.runtime.getDevice()), int(size), int(margin))
    cached = _interior_cylinder_cache.get(key)
    if cached is not None:
        return cached
    z, x = cp.mgrid[:size, :size].astype(cp.float32)
    centre = cp.float32((size - 1) / 2.0)
    radius = cp.float32(size / 2.0 - margin)
    xz = (x - centre) ** 2 + (z - centre) ** 2 <= radius**2
    mask = cp.zeros((size, size, size), dtype=cp.bool_)
    mask[:, margin : size - margin, :] = xz[:, None, :]
    _interior_cylinder_cache[key] = mask
    return mask


def column_score_volume_gpu(volume_zyx: ArrayLike, selector: str = "point", *,
                            thickness_percent=None):
    """Score a full volume or its configured central detection slab."""
    if thickness_percent is None:
        return _column_score_volume_gpu(volume_zyx, selector)
    from .tomogram_thickness import score_stage
    return score_stage(_column_score_volume_gpu, volume_zyx, selector, thickness_percent)


def _column_score_volume_gpu(volume_zyx: ArrayLike, selector: str = "point"):
    """Build the invariant device-resident score field for column searches."""
    import cupy as cp
    volume = cp.asarray(volume_zyx, dtype=cp.float32)
    if volume.ndim != 3 or volume.shape[1] != volume.shape[2]:
        raise ValueError("score volume must have square XY planes")
    size = int(volume.shape[0])
    if selector == "point":
        score_volume = volume
    elif selector == "spherical_mean":
        from cupyx.scipy.ndimage import convolve
        q=cp.arange(-2,3,dtype=cp.int32); zz0,yy0,xx0=cp.meshgrid(q,q,q,indexing="ij")
        kernel=(zz0**2+yy0**2+xx0**2<=4).astype(cp.float32)
        score_volume=convolve(volume,kernel/kernel.sum(),mode="constant",cval=float(cp.mean(volume).get()))
    elif selector in ("structure_tensor","bright_structure"):
        from cupyx.scipy.ndimage import gaussian_filter
        gradients=cp.gradient(gaussian_filter(volume,1.0,mode="nearest"))
        components=[]
        for left,right in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
            components.append(gaussian_filter(gradients[left]*gradients[right],2.0,mode="nearest"))
        del gradients
        score_volume=cp.empty_like(volume)
        # Closed-form smallest eigenvalue for a real symmetric 3x3 tensor.
        # Avoid cuSOLVER's batch-size limit while retaining the exact metric.
        for z0 in range(0,size,8):
            z1=min(size,z0+8); region=np.s_[z0:z1,:,:]
            a,b,c,d,e,f=(item[region] for item in components)
            q=(a+b+c)/3.0
            p=cp.sqrt(cp.maximum(((a-q)**2+(b-q)**2+(c-q)**2+2*(d*d+e*e+f*f))/6.0,0))
            safe=cp.maximum(p,cp.float32(1e-20))
            aa=(a-q)/safe; bb=(b-q)/safe; cc=(c-q)/safe
            dd=d/safe; ee=e/safe; ff=f/safe
            determinant=aa*bb*cc+2*dd*ee*ff-aa*ff*ff-bb*ee*ee-cc*dd*dd
            phi=cp.arccos(cp.clip(determinant/2.0,-1.0,1.0))/3.0
            smallest=q+2*p*cp.cos(phi+cp.float32(2*np.pi/3))
            score_volume[region]=cp.where(p>1e-20,smallest,q)
        del components
        if selector=="bright_structure":
            structure=cp.maximum(score_volume,0)
            from cupyx.scipy.ndimage import gaussian_laplace
            sigma=3.0
            local=gaussian_filter(volume,sigma,mode="nearest")
            background=gaussian_filter(volume,2.5*sigma,mode="nearest")
            blob=cp.maximum(-(sigma**2)*gaussian_laplace(volume,sigma,mode="nearest"),0.0)
            bright=cp.sqrt(blob*cp.maximum(local-background,0.0))
            # Dimensionless geometric mean: both terms must be present, while
            # their unrelated numerical scales cannot dominate selection.
            structure/=cp.sqrt(cp.mean(structure*structure))+cp.float32(1e-12)
            bright/=cp.sqrt(cp.mean(bright*bright))+cp.float32(1e-12)
            score_volume=cp.sqrt(structure*bright)
    elif selector == "bright_blob":
        from cupyx.scipy.ndimage import gaussian_filter, gaussian_laplace
        sigma=3.0
        local=gaussian_filter(volume,sigma,mode="nearest")
        background=gaussian_filter(volume,2.5*sigma,mode="nearest")
        blob=cp.maximum(-(sigma**2)*gaussian_laplace(volume,sigma,mode="nearest"),0.0)
        contrast=cp.maximum(local-background,0.0)
        score_volume=cp.sqrt(blob*contrast)
    else:
        raise ValueError(f"unknown selector: {selector}")
    return score_volume


def find_column_maxima_gpu(
    volume_zyx: ArrayLike,
    angle_deg: float,
    *,
    cell: int = 32,
    column_width: int = 30,
    margin: int = 32,
    selector: str = "point",
    return_scores: bool = False,
    score_volume=None,
    batch_size: int | None = None,
    thickness_percent=None,
) -> NDArray[np.int32] | tuple[NDArray[np.int32],NDArray[np.float32]]:
    """Find one maximum in every geometric 30x30 detector column.

    ``score_volume`` may be prepared once with :func:`column_score_volume_gpu`
    and shared by searches at multiple angles from the same reconstruction.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates

    volume = cp.asarray(volume_zyx, dtype=cp.float32)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("volume must be cubic")
    size = int(volume.shape[0])
    if score_volume is None:
        score_volume = column_score_volume_gpu(volume, selector, thickness_percent=thickness_percent)
    else:
        score_volume = cp.asarray(score_volume, dtype=cp.float32)
        if score_volume.shape != volume.shape:
            raise ValueError("score volume shape mismatch")
    centres = regular_column_centres(size, cell=cell, margin=margin)
    half = column_width // 2
    centre = (size - 1) / 2.0
    radians = np.deg2rad(float(angle_deg))
    cosine, sine = np.cos(radians), np.sin(radians)
    if abs(cosine) < 1.0e-6:
        raise ValueError("column geometry is singular near 90 degrees")
    cylinder = _interior_cylinder_gpu(cp, size, margin)
    z_values = cp.arange(margin, size - margin, dtype=cp.float32)
    if batch_size is None:
        batch_size = max(1, len(centres))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    maxima_batches = []
    local = cp.arange(column_width, dtype=cp.float32) - half
    for start in range(0, len(centres), batch_size):
        batch = cp.asarray(centres[start:start + batch_size], dtype=cp.float32)
        zz = z_values[None, :, None, None]
        yy = batch[:, 0, None, None, None] + local[None, None, :, None]
        uu = batch[:, 1, None, None, None] + local[None, None, None, :]
        shape = (len(batch), len(z_values), column_width, column_width)
        zz = cp.broadcast_to(zz, shape)
        yy = cp.broadcast_to(yy, shape)
        # X is constant along detector Y; keep that axis broadcast-only.
        compact_x = centre + (uu - centre + (z_values[None, :, None, None] - centre) * sine) / cosine
        xx = cp.broadcast_to(compact_x, shape)
        samples = map_coordinates(score_volume, cp.stack((zz, yy, xx)), order=1,
                                  mode="constant", cval=-cp.inf)
        valid_x = (compact_x >= margin) & (compact_x < size - margin)
        # Z and detector Y do not vary across the full sampled column grid.
        zi = cp.clip(cp.rint(z_values).astype(cp.int32), 0, size - 1)[None, :, None, None]
        compact_y = batch[:, 0, None] + local[None, :]
        yi = cp.clip(cp.rint(compact_y).astype(cp.int32), 0, size - 1)[:, None, :, None]
        xi = cp.clip(cp.rint(compact_x).astype(cp.int32), 0, size - 1)
        samples = cp.where(valid_x & cylinder[zi, yi, xi], samples, -cp.inf)
        peak = cp.argmax(samples.reshape(len(batch), -1), axis=1)
        # argmax guarantees valid flat indices; decode without the generic
        # unravel_index bounds check and its device-to-host synchronization.
        iz = peak // (column_width * column_width)
        iy = (peak // column_width) % column_width
        iu = peak % column_width
        row = cp.arange(len(batch))
        maxima_batches.append(cp.stack((z_values[iz], yy[row, iz, iy, iu],
                                        cp.rint(xx[row, iz, iy, iu])), axis=1))
    maxima_array = cp.asnumpy(cp.concatenate(maxima_batches)).astype(np.int32)
    if thickness_percent is not None:
        from .tomogram_thickness import filter_maxima, validate_percent
        maxima_array = filter_maxima(maxima_array, score_volume,
                                     min(50., validate_percent(thickness_percent)))
    if return_scores:
        scores=cp.asnumpy(score_volume[tuple(maxima_array.T)]).astype(np.float32,copy=False)
        return maxima_array,scores
    return maxima_array


def _landmark_sphere_mask_gpu(volume, landmarks, sphere_diameter):
    """Build an exact union of integer-centred spheres in one GPU launch."""
    import cupy as cp
    size = int(volume.shape[0])
    radius = sphere_diameter / 2.0
    half = int(np.ceil(radius))
    points = np.ascontiguousarray(landmarks, dtype=np.int32)
    # Keep the established slicing behaviour for unusual external inputs.
    if half < 0 or volume.shape != (size, size, size) or points.ndim != 2 or points.shape[1] != 3 or np.any(points < 0) or np.any(points >= size):
        return None
    mask = cp.zeros(volume.shape, dtype=cp.uint32)
    if not len(points):
        return mask.astype(cp.bool_)
    side = 2 * half + 1
    count = len(points) * side**3
    kernel = cp.RawKernel(r"""
    extern "C" __global__ void landmark_spheres(
        const int* points, unsigned int* mask, const long long count,
        const int size, const int half, const int side, const double radius2) {
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < count; i += (long long)blockDim.x * gridDim.x) {
            long long q = i;
            int dx = q % side - half; q /= side;
            int dy = q % side - half; q /= side;
            int dz = q % side - half; q /= side;
            if ((double)(dx*dx + dy*dy + dz*dz) > radius2) continue;
            int z = points[3*q] + dz, y = points[3*q+1] + dy, x = points[3*q+2] + dx;
            if (z >= 0 && z < size && y >= 0 && y < size && x >= 0 && x < size)
                atomicExch(mask + ((long long)z * size + y) * size + x, 1u);
        }
    }
    """, "landmark_spheres")
    kernel((min((count + 255)//256, 65535),), (256,),
           (cp.asarray(points), mask, np.int64(count), np.int32(size),
            np.int32(half), np.int32(side), np.float64(radius**2)))
    return mask.astype(cp.bool_)


def masked_landmark_projection_gpu(
    volume_zyx: ArrayLike,
    landmarks_zyx: ArrayLike,
    angle_deg: float,
    *,
    sphere_diameter: int = 32,
    y_chunk: int = 64,
    return_support: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Reproject landmark density; omit support only for display-only callers."""
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates

    volume = cp.asarray(volume_zyx, dtype=cp.float32)
    size = int(volume.shape[0]); radius = sphere_diameter / 2.0
    mask = _landmark_sphere_mask_gpu(volume, landmarks_zyx, sphere_diameter)
    if mask is None:
        mask = cp.zeros_like(volume, dtype=cp.bool_)
        local = cp.arange(-int(np.ceil(radius)), int(np.ceil(radius)) + 1, dtype=cp.int32)
        dz, dy, dx = cp.meshgrid(local, local, local, indexing="ij")
        ball = dz**2 + dy**2 + dx**2 <= radius**2
        for z, y, x in np.asarray(landmarks_zyx, dtype=np.int32):
            z0=max(0,z-local.size//2); z1=min(size,z+local.size//2+1)
            y0=max(0,y-local.size//2); y1=min(size,y+local.size//2+1)
            x0=max(0,x-local.size//2); x1=min(size,x+local.size//2+1)
            bz0=z0-(z-local.size//2); by0=y0-(y-local.size//2); bx0=x0-(x-local.size//2)
            mask[z0:z1,y0:y1,x0:x1] |= ball[bz0:bz0+z1-z0,by0:by0+y1-y0,bx0:bx0+x1-x0]
    masked = volume * mask
    radians=np.deg2rad(float(angle_deg)); cosine=np.cos(radians); sine=np.sin(radians)
    centre=(size-1)/2.0
    projection=cp.zeros((size,size),dtype=cp.float32)
    support=cp.zeros_like(projection) if return_support else None
    float_mask = mask.astype(cp.float32) if return_support else None
    z=cp.arange(size,dtype=cp.float32)
    u=cp.arange(size,dtype=cp.float32)
    for y0 in range(0,size,y_chunk):
        ys=cp.arange(y0,min(size,y0+y_chunk),dtype=cp.float32)
        yy,uu,zz=cp.meshgrid(ys,u,z,indexing="ij")
        xx=centre+(uu-centre+(zz-centre)*sine)/cosine
        coords=cp.stack((zz,yy,xx))
        projection[y0:y0+len(ys)] = map_coordinates(masked,coords,order=1,mode="constant").sum(axis=2)
        if return_support:
            support[y0:y0+len(ys)] = map_coordinates(float_mask,coords,order=1,mode="constant").sum(axis=2)
    return cp.asnumpy(projection), cp.asnumpy(support) if return_support else None


def refine_from_selected_landmarks(
    volume_zyx: ArrayLike,
    target_yx: ArrayLike,
    landmarks_zyx: ArrayLike,
    angle_deg: float,
    *,
    search_radius: int = 32,
    margin: int = 32,
    search_sigma: float | None = None,
    search_center_yx: tuple[float, float] = (0.0,0.0),
) -> ColumnRefinementResult:
    """Refine a target using an externally quality-selected landmarks subset."""
    landmarks=np.asarray(landmarks_zyx,dtype=np.int32)
    if not len(landmarks): raise ValueError('at least one selected landmark is required')
    projection,support=masked_landmark_projection_gpu(volume_zyx,landmarks,angle_deg)
    square=np.zeros(projection.shape,dtype=np.float32); square[margin:-margin,margin:-margin]=1
    mask=(support>0.25).astype(np.float32)*square
    transform,score,corr,boundary=masked_translation(
        projection,np.asarray(target_yx),mask,square,overlap_ratio=.95,
        search_radius=search_radius,search_sigma=search_sigma,
        search_center_yx=search_center_yx,use_gpu=True)
    centre=np.asarray(target_yx).shape[0]-1; shifts=np.arange(centre,-centre-1,-1,dtype=np.int64)
    return ColumnRefinementResult((-transform[0],-transform[1]),score,boundary,landmarks,
        projection,mask,corr,shifts,shifts)
