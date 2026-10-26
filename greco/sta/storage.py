"""Half-float particle/map storage with full-precision MRC header statistics."""
import numpy as np


def write_mrc(path, value, pixel_size, *, xp=np, image_stack=False, overwrite=False):
    """Convert before device transfer; never scan float16 arrays for statistics."""
    import mrcfile
    value=xp.asarray(value)
    if value.ndim!=3 or not value.size:raise ValueError('Expected a nonempty 3D particle stack or map.')
    # CPU float32 reductions avoid the slow float16 statistics path in set_data.
    work=value.astype(xp.float32,copy=False)
    minimum=float(work.min());maximum=float(work.max())
    if not np.isfinite([minimum,maximum]).all() or minimum < -65504. or maximum > 65504.:
        raise ValueError('STA images contain nonfinite values or exceed the float16 storage range.')
    packed=value.astype(xp.float16,copy=False)
    quantized=packed.astype(xp.float32)
    stats=(float(quantized.min()),float(quantized.max()),
           float(quantized.mean(dtype=xp.float64)),float(quantized.std(dtype=xp.float64)))
    host=np.asarray(packed) if xp is np else xp.asnumpy(packed)
    # new_mmap does not recompute statistics in float16; values describe the
    # quantized image actually written, rather than its pre-conversion input.
    with mrcfile.new_mmap(path,shape=host.shape,mrc_mode=12,overwrite=overwrite) as handle:
        handle.data[:]=host
        if image_stack:handle.set_image_stack()
        handle.voxel_size=pixel_size
        for name,number in zip(('dmin','dmax','dmean','rms'),stats):setattr(handle.header,name,number)
