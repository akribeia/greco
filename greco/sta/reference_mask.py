"""Soft reference masks, optionally adapted to extraction in physical units."""
import numpy as np


def load_mask(path, box, pixel, resample_physical=False):
    import mrcfile
    with mrcfile.open(path, permissive=False) as m:
        if m.data.ndim!=3:
            raise ValueError('Reference mask must be a 3D volume.')
        if m.data.shape != (box,)*3 and not resample_physical:
            raise ValueError('Reference mask dimensions must match the extracted box.')
        sampling=np.array([float(m.voxel_size[a]) for a in 'xyz'])
        if not np.isfinite(sampling).all() or np.any(sampling<=0):
            raise ValueError('Reference mask sampling must be positive and finite.')
        if not np.allclose(sampling,pixel,rtol=1e-3) and not resample_physical:
            raise ValueError('Reference mask pixel size must match extraction sampling.')
        mask=np.array(m.data,dtype=np.float64,copy=True)
    if not np.isfinite(mask).all() or mask.min()<0 or mask.max()>1 or not np.any(mask>0):
        raise ValueError('Reference mask must contain finite values in [0, 1] and nonzero support.')
    if resample_physical and (mask.shape!=(box,)*3 or not np.allclose(sampling,pixel,rtol=1e-6)):
        from scipy.ndimage import affine_transform
        ratio=pixel/sampling[::-1]
        offset=np.asarray(mask.shape)//2-ratio*(box//2)
        mask=np.clip(affine_transform(mask,np.diag(ratio),offset=offset,
            output_shape=(box,)*3,order=1,mode='constant',cval=0.,prefilter=False),0.,1.)
        if not np.any(mask>0):raise ValueError('Reference mask has no support on the extraction grid.')
    return mask
