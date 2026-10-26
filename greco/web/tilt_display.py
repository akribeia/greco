"""One display transform for detector images and analytic defocus planes.

Array coordinates are x right / y down; flip Y precedes a visual CCW turn.
The inverse maps a displayed pixel to the original aligned detector raster.
"""
import numpy as np

from .projects import InputError


def inverse_matrix(rotation_deg=0,flip_y=False):
    angle=float(rotation_deg)
    if not np.isfinite(angle) or abs(angle)>360 or not isinstance(flip_y,bool):
        raise InputError('Invalid tilt display orientation.')
    c,s=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
    # Exact quarter turns must not put boundary pixels at -1e-15.
    if abs(c)<1e-15:c=0.
    if abs(s)<1e-15:s=0.
    sign=-1. if flip_y else 1.
    return np.array([[c,-s],[sign*s,sign*c]])


def orient_image(image,rotation_deg=0,flip_y=False,background=0.):
    from scipy.ndimage import affine_transform
    matrix=inverse_matrix(rotation_deg,flip_y)[::-1,::-1]
    if not rotation_deg and not flip_y:return image
    if image.ndim==3:
        # Transform colour channels as one spatial raster, never the channel axis.
        backgrounds=np.broadcast_to(background,(image.shape[-1],))
        return np.stack([orient_image(image[...,i],rotation_deg,flip_y,float(backgrounds[i]))
                         for i in range(image.shape[-1])],axis=-1)
    center=(np.asarray(image.shape,dtype=float)-1)/2
    return affine_transform(image,matrix,offset=center-matrix@center,order=1,
                            mode='constant',cval=background,prefilter=False)
