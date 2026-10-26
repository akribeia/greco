"""Device-batched local projections with CPU geometry parity."""

import numpy as np


def project_volume_patches_gpu_device(volume, points_xyz, angle_deg, *,
                                      observation_radius=16, model_radius=16,
                                      batch_size=32):
    """Batch the validated global-volume local projection on the device."""
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    data = cp.asarray(volume, dtype=cp.float64)
    points = np.asarray(points_xyz, dtype=np.float64)
    if data.ndim != 3 or len(set(data.shape)) != 1 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('expected a cubic volume and (batch,3) xyz points')
    radius = int(observation_radius)
    offsets = cp.arange(-radius, radius + 1, dtype=cp.float64)
    line = cp.linspace(-float(model_radius), float(model_radius),
                       2 * int(model_radius) + 1, dtype=cp.float64)
    dv, du, ss = cp.meshgrid(offsets, offsets, line, indexing='ij')
    angle = cp.deg2rad(cp.float64(angle_deg)); cosine = cp.cos(angle); sine = cp.sin(angle)
    centre = cp.float64((data.shape[0] - 1) / 2)
    output = []
    for start in range(0, len(points), batch_size):
        point = cp.asarray(points[start:start + batch_size], cp.float64)
        x = point[:, 0, None, None, None] + du * cosine + ss * sine
        z = point[:, 2, None, None, None] - du * sine + ss * cosine
        y = point[:, 1, None, None, None] + dv
        samples = map_coordinates(data, cp.stack((z + centre, y + centre, x + centre)),
                                  order=1, mode='constant', cval=0.0)
        output.append(samples.sum(axis=-1).astype(cp.float32))
    return cp.concatenate(output) if output else cp.empty((0, 2*radius+1, 2*radius+1), cp.float32)


def project_local_volumes_gpu_device(volumes, angle_deg, *, observation_size=33, model_radius=16):
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    data=cp.asarray(volumes,dtype=cp.float32)
    if data.ndim!=4 or len(set(data.shape[1:]))!=1:
        raise ValueError('expected (batch,z,y,x) cubic volumes')
    offsets=cp.arange(observation_size,dtype=cp.float32)-cp.float32((observation_size-1)/2)
    line=cp.linspace(-float(model_radius),float(model_radius),2*int(model_radius)+1,dtype=cp.float32)
    dv,du,ss=cp.meshgrid(offsets,offsets,line,indexing='ij')
    angle=cp.deg2rad(cp.float32(angle_deg));c=cp.cos(angle);s=cp.sin(angle)
    centre=cp.float32((data.shape[-1]-1)/2)
    z=-du*s+ss*c+centre;y=dv+centre;x=du*c+ss*s+centre
    batch=cp.arange(len(data),dtype=cp.float32)[:,None,None,None]
    shape=(len(data),)+z.shape
    coords=cp.stack((cp.broadcast_to(batch,shape),cp.broadcast_to(z,shape),
        cp.broadcast_to(y,shape),cp.broadcast_to(x,shape)))
    projected=map_coordinates(data,coords,order=1,mode='constant',cval=0.).sum(axis=-1)
    return projected.astype(cp.float32,copy=False)
