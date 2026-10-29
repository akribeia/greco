"""Rigid output-frame leveling from the final landmark plane."""
import numpy as np
from ..project_layout import join


def fit(positions):
    points=np.asarray(positions,float)
    points=points[np.isfinite(points).all(axis=1)]
    if len(points)<3:raise ValueError('Tomogram leveling requires three finite landmarks.')
    centre=points.mean(axis=0)
    _,singular,vh=np.linalg.svd(points-centre,full_matrices=False)
    if singular[1]<=max(singular[0],1.)*1e-8:raise ValueError('Landmarks do not define a reference plane.')
    normal=vh[-1]
    if normal[0]<0:normal=-normal
    weights=np.ones(len(points));scale=0.;iterations=0
    # Huber IRLS retains every landmark, reducing the influence of large
    # perpendicular residuals. MAD sets the scale in the current plane frame.
    for iterations in range(1,31):
        distance=(points-centre)@normal
        location=np.median(distance)
        scale=float(1.4826*np.median(np.abs(distance-location)))
        if scale<1e-8:break
        weights=np.minimum(1.,1.345*scale/np.maximum(np.abs(distance-location),1e-15))
        newcentre=np.average(points,axis=0,weights=weights)
        _,_,vh=np.linalg.svd((points-newcentre)*np.sqrt(weights[:,None]),full_matrices=False)
        newnormal=vh[-1]
        if newnormal@normal<0:newnormal=-newnormal
        change=np.linalg.norm(newnormal-normal)
        centre=newcentre;normal=newnormal
        if change<1e-8:break
    if normal[0]<0:normal=-normal
    target=np.array([1.,0.,0.]);v=np.cross(normal,target);c=float(normal@target)
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    rotation=np.eye(3)+skew+skew@skew/(1+c)
    return rotation,dict(method='huber_irls',iterations=iterations,robust_scale_px=scale,downweighted=int((weights<1).sum()),minimum_weight=float(weights.min()),normal_zyx=normal.tolist(),centroid_zyx=centre.tolist(),rotation_zyx=rotation.tolist(),angle_deg=float(np.rad2deg(np.arccos(np.clip(c,-1,1)))),landmark_count=len(points))


def resolve(data,alignment,size,z_start=0):
    from ..final_reconstruction import export_transform
    transform=np.asarray(export_transform(size,z_start),float)
    if not data.get('advanced_options',{}).get('tomogram_leveling'):return np.eye(3),transform,None
    with np.load(join(alignment,'final_state.npz'),allow_pickle=False) as state:
        positions=np.asarray(state['positions_zyx'],float)
        if 'valid_views' in state:positions=positions[np.asarray(state['valid_views'],bool).sum(axis=1)>=4]
    rotation,report=fit(positions)
    affine=np.eye(4);affine[:3,:3]=rotation;centre=np.full(3,(size-1)/2)
    affine[:3,3]=centre-rotation@centre
    return rotation,transform@affine,report
