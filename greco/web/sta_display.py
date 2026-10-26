"""Stable display windows for masked STA volumes, independent of slice/axis."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .projects import InputError


@lru_cache(maxsize=32)
def _window(path,size,mtime_ns,zero_centered=False):
    import mrcfile
    with mrcfile.mmap(path,mode='r') as handle:
        # Bounded statistics even for fine-sampled particle maps. Cache only
        # two numbers; a new refinement checkpoint invalidates this key.
        strides=tuple(max(1,(n+127)//128) for n in handle.data.shape)
        sample=handle.data[tuple(slice(None,None,s) for s in strides)]
        finite=sample[np.isfinite(sample)]
        if not finite.size:raise InputError('No finite volume pixels.')
        if zero_centered:
            # Class maps have large exactly-zero masked regions. They must not
            # determine the percentile window. Keep zero at the same mid-grey
            # for every class, without changing the stored map or its sign.
            supported=finite[finite!=0]
            extent=float(np.percentile(np.abs(supported),99)) if supported.size else 0.
            extent=max(extent,np.finfo(np.float32).tiny)
            return -extent,extent
        lo,hi=np.percentile(finite,[1,99])
        if hi<=lo:lo,hi=float(finite.min()),float(finite.max())
    return float(lo),float(hi)


def volume_image(path,index,axis='z',*,zero_centered=False):
    from .reconstruction import volume_plane_image
    path=Path(path);stat=path.stat()
    window=_window(str(path.resolve()),stat.st_size,stat.st_mtime_ns,zero_centered)
    image=volume_plane_image(path,index,axis=axis,intensity_window=window)
    if zero_centered:
        from PIL import Image
        # Preserve both signed halves of the window, but use a common dark
        # background (46/255) instead of mid-grey. Avoid clipping negatives
        # simply to make the zero-valued masked region darker.
        values=np.asarray(image,dtype=np.float32)
        values=np.where(values<=127,values*(46/127),46+(values-127)*(209/128))
        image=Image.fromarray(np.rint(values).astype(np.uint8))
    return image
