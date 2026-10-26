"""Budget-aware spatial de-duplication of score-sorted screening candidates."""
import itertools
import numpy as np


def spaced_candidate_indices(positions,limit,radius):
    """Greedy NMS over already ranked candidates, using a small spatial hash.

    Unlike final extraction, no detection threshold is imposed here. A
    smaller-than-particle exclusion radius prevents one textured object
    consuming the expensive full-bank candidate budget.
    """
    positions=np.asarray(positions)
    if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
        raise ValueError('Candidates must be finite (N,3) positions')
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<0:
        raise ValueError('Candidate limit must be a nonnegative integer')
    if not np.isfinite(radius) or radius<0:raise ValueError('Invalid candidate radius')
    if not limit:return np.empty(0,dtype=np.int64)
    if radius==0:return np.arange(min(limit,len(positions)),dtype=np.int64)
    offsets=list(itertools.product((-1,0,1),repeat=3));cells={};selected=[]
    r2=radius*radius
    for i,p in enumerate(positions):
        cell=tuple(np.floor(p/radius).astype(int));blocked=False
        for offset in offsets:
            key=tuple(a+b for a,b in zip(cell,offset))
            if any(float(np.sum((p-q)**2))<=r2 for q in cells.get(key,())):
                blocked=True;break
        if blocked:continue
        cells.setdefault(cell,[]).append(p);selected.append(i)
        if len(selected)==limit:break
    return np.asarray(selected,dtype=np.int64)
