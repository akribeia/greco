"""Frozen-image adjacent-tilt smoothness diagnostic; no reconstruction/fit.

Colors are review heuristics, NOT calibrated probabilities or rejection rules.
Nonrigid/parallax/appearance changes can also cause suspicious transitions.
"""
import time
import numpy as np
from scipy.ndimage import gaussian_filter,sobel,shift
from .registration import _summarize

PROTOCOL='adjacent-gradient-smoothness-v1'
THRESHOLDS=dict(green_jump_px=1.,red_jump_px=2.,minimum_correlation=.15,minimum_peak_z=4.)


def prepare(image):
    v=np.asarray(image,np.float32);finite=np.isfinite(v)
    if not finite.any():return None
    v=np.where(finite,v,np.median(v[finite]));lo,hi=np.percentile(v,[1,99])
    if hi-lo<1e-8:return None
    v=np.clip(v,lo,hi);v-=gaussian_filter(v,max(3.,min(v.shape)/32))
    gy=sobel(v,axis=0)/8;gx=sobel(v,axis=1)/8
    gradients=np.stack((gy,gx));scale=float(np.sqrt(np.mean(gradients**2)))
    if scale<1e-8:return None
    gradients/=scale
    window=np.outer(np.hanning(v.shape[0]),np.hanning(v.shape[1]))
    return gradients*window


def measure(reference,moving,radius):
    shape=np.array(reference.shape[-2:])*2
    cross=np.sum(np.fft.fft2(reference,s=shape)*np.conj(np.fft.fft2(moving,s=shape)),axis=0)
    magnitude=abs(cross);eps=max(float(np.median(magnitude[magnitude>0])),1.)*1e-12
    surface=np.fft.ifft2(cross/(magnitude+eps)**.25).real
    offsets=np.arange(-radius,radius+1);surface=surface[np.ix_(offsets%shape[0],offsets%shape[1])]
    scale=max(float(np.max(abs(surface))),1e-12);surface/=scale
    result=_summarize(surface,radius);peak=np.unravel_index(np.argmax(surface),surface.shape)
    background=surface.copy();background[max(0,peak[0]-1):peak[0]+2,max(0,peak[1]-1):peak[1]+2]=np.nan
    median=np.nanmedian(background);sigma=1.4826*np.nanmedian(abs(background-median))
    peak_z=float((surface[peak]-median)/max(float(sigma),1e-8))
    shifted=np.stack([shift(g,result.correction_yx,order=1,mode='constant',cval=0,prefilter=False) for g in moving])
    correlation=float(np.sum(reference*shifted)/max(float(np.sqrt(np.sum(reference**2)*np.sum(shifted**2))),1e-12))
    return dict(correction_yx=result.correction_yx.tolist(),correlation=correlation,peak_z=peak_z,
        boundary=bool(result.boundary),reliable=bool(result.valid and correlation>=THRESHOLDS['minimum_correlation'] and peak_z>=THRESHOLDS['minimum_peak_z']))


def classify_transitions(pairs,angles):
    angles=np.asarray(angles,float);rows=[]
    for i,pair in enumerate(pairs):
        if not pair.get('reliable'):continue
        neighbours=[j for j in range(max(0,i-4),min(len(pairs),i+5)) if j!=i and pairs[j].get('reliable')]
        if len(neighbours)<3:continue
        velocities=[np.asarray(pairs[j]['correction_yx_px'])/(angles[j+1]-angles[j]) for j in neighbours]
        baseline=np.median(velocities,axis=0)*(angles[i+1]-angles[i])
        residual=np.asarray(pair['correction_yx_px'])-baseline
        pair.update(expected_yx_px=baseline.tolist(),smoothness_residual_yx_px=residual.tolist(),jump_px=float(np.linalg.norm(residual)))
    for i,angle in enumerate(angles):
        edges=[pairs[j] for j in (i-1,i) if 0<=j<len(pairs)]
        usable=[p for p in edges if p.get('reliable') and 'jump_px' in p]
        row=dict(index=i,number=i+1,angle_deg=float(angle),color='yellow',reason='Transition evidence is weak or ambiguous.',adjacent_pairs=[p['left_index'] for p in edges])
        if not any(p.get('measured') for p in edges):row.update(color='gray',reason='No usable image texture or transition evidence.')
        elif len(usable)==len(edges) and usable and max(p['jump_px'] for p in usable)<=THRESHOLDS['green_jump_px']:
            row.update(color='green',reason='Neighbour transitions follow the local smooth trend.')
        if len(usable)==2:
            a=np.asarray(usable[0]['smoothness_residual_yx_px']);b=np.asarray(usable[1]['smoothness_residual_yx_px'])
            estimate=.5*(a-b);closure=float(np.linalg.norm(a+b))
            row.update(inferred_offset_yx_px=estimate.tolist(),inferred_offset_px=float(np.linalg.norm(estimate)),opposing_edge_disagreement_px=closure)
            if min(np.linalg.norm(a),np.linalg.norm(b))>=THRESHOLDS['red_jump_px'] and closure<=max(1.,.5*np.linalg.norm(estimate)):
                row.update(color='red',reason='Strong, opposing jumps on both neighbouring transitions: this tilt is a candidate for review.')
        if usable:row['max_transition_jump_px']=float(max(p['jump_px'] for p in usable))
        rows.append(row)
    return rows


def diagnose(stack,angles):
    start=time.monotonic();stack=np.asarray(stack);angles=np.asarray(angles,float)
    if stack.ndim!=3:raise ValueError('2D diagnostic requires a tilt/Y/X image stack.')
    if len(stack)!=len(angles):
        raise ValueError(f'Aligned image/angle count mismatch: {len(stack)} images, {len(angles)} angles.')
    if len(angles)<3 or not np.isfinite(angles).all() or np.any(np.diff(angles)<=0):
        raise ValueError('At least three angle-ordered images required.')
    factor=max(1,int(np.ceil(max(stack.shape[-2:])/256)))
    images=[]
    for plane in stack:
        v=gaussian_filter(np.asarray(plane,np.float32),factor/2)[::factor,::factor] if factor>1 else plane
        images.append(prepare(v))
    radius=max(2,int(np.ceil(16/factor)));pairs=[]
    for i in range(len(images)-1):
        row=dict(left_index=i,right_index=i+1,left_number=i+1,right_number=i+2,measured=False,reliable=False)
        if images[i] is not None and images[i+1] is not None:
            result=measure(images[i],images[i+1],radius);row.update(result,measured=True)
            row['correction_yx_px']=(np.array(result['correction_yx'])*factor).tolist()
        pairs.append(row)
    return dict(protocol=PROTOCOL,rows=classify_transitions(pairs,angles),pairs=pairs,
        thresholds=THRESHOLDS,units='output-bin pixels',search_radius_px=radius*factor,
        calibrated_probability=False,geometry_changed=False,reconstruction_used=False,
        test='Adjacent gradient CC; robust local angular trend; opposing entering/exiting jumps.',
        runtime_s=time.monotonic()-start)
