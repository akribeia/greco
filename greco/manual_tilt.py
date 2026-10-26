"""Manual display priors and bounded shared-XY likelihood (no XYZ/angle updates)."""
import numpy as np


def validate_edits(edits,count):
    if not isinstance(edits,list) or not edits or len(edits)>=count:raise ValueError('Select at least one tilt and leave a fixed reference.')
    result=[];seen=set()
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)-{'index','reference','shift_yx'}:raise ValueError('Invalid manual tilt edit.')
        index=edit.get('index');reference=edit.get('reference');shift=edit.get('shift_yx')
        if type(index) is not int or not 0<=index<count or index in seen:raise ValueError('Invalid or duplicate tilt index.')
        if reference is not None and (type(reference) is not int or not 0<=reference<count or reference==index):raise ValueError('Invalid reference tilt.')
        if not isinstance(shift,list) or len(shift)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in shift):raise ValueError('Invalid XY prior.')
        shift=np.asarray(shift,float)
        if not np.isfinite(shift).all() or np.max(abs(shift))>512:raise ValueError('Manual shift must be finite and within 512 bin8 pixels.')
        seen.add(index);result.append(dict(index=index,reference=reference,shift_yx=shift.tolist()))
    return result


def choose_reference(angles,target,excluded,preferred=None):
    available=np.array([t for t in range(len(angles)) if t not in excluded],int)
    if not len(available):raise ValueError('No fixed reference tilt remains.')
    if preferred is not None and preferred in available:return int(preferred)
    return int(available[np.argmin(abs(np.asarray(angles)[available]-angles[target]))])


def display_prior_to_detector(shift_yx,axis,pixel):
    """Moving displayed content +XY means detector model translation -Raxis*XY.

    Raster Flip Y is already applied to both source stack and cameras; it must
    NOT be applied again to this display-space displacement.
    """
    from .landmark_rigid import initial_cameras
    rotation=initial_cameras([0.],axis)[0,:2,:2]
    return -rotation@np.asarray(shift_yx,float)[::-1]*pixel


def shared_xy_objective(surfaces,radius,padding,maximum,xp=np):
    from .sta.gp3d import keys_sample
    def objective(parameter):
        # Bounded displacement about the manual prior/current centre.
        h=xp.tanh(parameter);delta=maximum*h
        xy=xp.broadcast_to(delta*padding+radius,(len(surfaces),2))
        value,gradient=keys_sample(surfaces,xy,xp)
        return value.mean(),gradient.mean(0)*padding*maximum*(1-h*h)
    return objective


def neighbor_column_candidates(score,camera,cell=32,width=30,batch=16,z_bounds=None):
    """One q-min maximum per neighbouring camera column, full 3-D orientation."""
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    n=score.shape[0];center=(n-1)/2
    z0,z1=z_bounds if z_bounds is not None else (n//4,3*n//4)
    grid=np.arange(32,n-32,cell);yy,xx=np.meshgrid(grid,grid,indexing='ij')
    columns=np.stack((xx.ravel(),yy.ravel()),axis=1)
    camera=cp.asarray(camera,cp.float32);local=cp.arange(width)-width//2
    depth=cp.arange(n,dtype=cp.float32)-center;result=[]
    for first in range(0,len(columns),batch):
        chosen=cp.asarray(columns[first:first+batch],cp.float32)
        shape=(len(chosen),n,width,width)
        u=cp.broadcast_to(chosen[:,0,None,None,None]-center+local[None,None,None,:],shape)
        v=cp.broadcast_to(chosen[:,1,None,None,None]-center+local[None,None,:,None],shape)
        w=cp.broadcast_to(depth[None,:,None,None],shape)
        xyz=cp.stack([camera[0,j]*u+camera[1,j]*v+camera[2,j]*w+center for j in range(3)])
        eligible=(xyz[2]>=z0)&(xyz[2]<z1)&(xyz[0]>=n*.1)&(xyz[0]<n*.9)&(xyz[1]>=n*.1)&(xyz[1]<n*.9)
        response=map_coordinates(score,xyz[::-1],order=1,mode='constant',cval=0)
        response=cp.where(eligible & cp.isfinite(response),response,-cp.inf).reshape(len(chosen),-1)
        indices=cp.argmax(response,axis=1);peaks=response[cp.arange(len(chosen)),indices]
        points=xyz.reshape(3,len(chosen),-1)[:,cp.arange(len(chosen)),indices].T
        result.append(cp.asnumpy(points[cp.isfinite(peaks)&(peaks>0)]))
    if not result:return np.empty((0,3))
    points=np.vstack(result)
    # Exact voxel deduplication only, not a competing signal filter.
    return np.unique(np.rint(points).astype(int),axis=0)-center


def image_smoothness_guides(diagnostic,active,targets,translations,axis,pixel):
    """Absolute detector-XY guides from reliable image residuals, not shift curves.

    Average the incident edge corrections: entering edge is +residual, exiting
    edge is -residual. No pair is interpreted as an observation of 3-D geometry.
    """
    votes={int(t):[] for t in targets};quality={int(t):[] for t in targets}
    for pair in diagnostic['pairs']:
        if not pair.get('reliable') or 'smoothness_residual_yx_px' not in pair:continue
        residual=np.asarray(pair['smoothness_residual_yx_px'],float)
        if not np.isfinite(residual).all():continue
        for index,sign in ((pair['left_index'],-1),(pair['right_index'],1)):
            target=int(active[index])
            if target in votes:
                votes[target].append(sign*residual)
                quality[target].append(min(1.,max(0.,float(pair['correlation']))/.4))
    return {t:dict(detector_xy=translations[t]+display_prior_to_detector(np.mean(v,axis=0),axis,pixel)/pixel,
                   correction_yx_px=np.mean(v,axis=0).tolist(),quality=float(np.mean(quality[t])),edges=len(v))
            for t,v in votes.items() if v}


def smooth_xy_penalty(delta,desired,strength,curvature,xp=np):
    """Robust image guidance scaled to the local GCC likelihood curvature."""
    residual=delta-xp.asarray(desired);root=xp.sqrt(1+residual*residual)
    weight=strength*curvature
    return weight*xp.sum(residual*residual/(root+1)),weight*residual/root


def validate_gaussian_search(enabled=False,sigma_px=8.):
    if type(enabled) is not bool:raise ValueError('Gaussian search prior must be a checkbox.')
    if isinstance(sigma_px,bool) or not isinstance(sigma_px,(int,float)) or not np.isfinite(sigma_px) or not .1<=sigma_px<=128:
        raise ValueError('Gaussian sigma must be between 0.1 and 128 bin8 pixels.')
    return dict(gaussian_search_prior=enabled,gaussian_sigma_px=float(sigma_px))


def gaussian_search_penalty(delta_xy,sigma_px,xp=np):
    """Negative twice log N(0,sigma²I), in the mean-column SSE objective units."""
    delta=xp.asarray(delta_xy)
    return xp.sum(delta*delta)/(sigma_px*sigma_px),2*delta/(sigma_px*sigma_px)


def gaussian_search_start(surfaces,radius,padding,maximum,sigma_px,xp=np):
    """Global integer-grid MAP start; the ordinary optimizer refines subpixels."""
    grid=xp.arange(-int(maximum),int(maximum)+1)
    indices=radius+grid*padding
    likelihood=surfaces.mean(0)[indices[:,None],indices[None,:]]
    cost=likelihood+(grid[:,None]**2+grid[None,:]**2)/(sigma_px*sigma_px)
    best=int(xp.argmin(cost));width=len(grid)
    return xp.asarray([grid[best%width],grid[best//width]],dtype=xp.float64)
