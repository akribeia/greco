"""Remove only geometrically unsupported landmark/view pairs, then refit.

This is a field-of-view check, not a signal/quality filter. Global and local
landmarks obey the same image support. Unsafe image shifts remain a veto.
"""
import numpy as np
from .helpers import projected_centres


def solve_with_view_containment(votes, valid, classes, positions, angles, size,
                               search_radius, robust_scale, max_nfev, tolerance,
                               workspace=None, backend='cpu', containment_margin=20,
                               *, solve):
    work_votes=np.array(votes,float,copy=True);work_valid=np.array(valid,bool,copy=True)
    work_votes[~work_valid]=np.nan
    required_views=np.any(np.isfinite(work_votes).all(axis=2),axis=0)
    removed=[]
    while True:
        active=np.sum(np.isfinite(work_votes).all(axis=2),axis=1)>=4
        if np.sum(active)<2:raise ValueError('insufficient landmarks after view containment exclusion')
        observed=np.isfinite(work_votes[active]).all(axis=2).any(axis=0)
        if np.any(required_views & ~observed):
            raise ValueError('view containment would leave a previously observed tilt unconstrained')
        shifts,active_dr,report=solve(work_votes[active],angles,robust_scale,max_nfev,tolerance,workspace,backend)
        dr=np.zeros_like(positions,dtype=float);dr[active]=active_dr
        centres=projected_centres(positions+dr,angles,size)
        inside=np.all(np.isfinite(centres)&(centres>=containment_margin)&(centres<=size-1-containment_margin),axis=2)
        bad=work_valid & ~inside
        offending=np.flatnonzero(np.any(bad,axis=1))
        shift_safe=bool(np.isfinite(shifts).all() and np.max(np.abs(shifts))<search_radius)
        report=dict(report,containment_removed_pairs=sum(len(item['view_indices']) for item in removed))
        if not shift_safe or not offending.size:
            return shifts,dr,report,work_votes,work_valid,active,removed,shift_safe and not offending.size,offending
        # Monotone support removal terminates after at most L*T pair removals.
        # A landmark remains available on every other in-bounds view.
        for k in offending:
            removed.append(dict(landmark=int(k),class_name=str(classes[k]),
                view_indices=np.flatnonzero(bad[k]).tolist(),supported_views=int(work_valid[k].sum()),
                outside_views=int(bad[k].sum()),policy='geometric_pair_support'))
        work_votes[bad]=np.nan;work_valid[bad]=False
        weak=active & (np.sum(np.isfinite(work_votes).all(axis=2),axis=1)<4)
        # Retire only rows made underconstrained by this exclusion. Previously
        # inactive rows may recover observations in subsequent GCC passes.
        for k in np.flatnonzero(weak & work_valid.any(axis=1)):
            removed.append(dict(landmark=int(k),class_name=str(classes[k]),view_indices=np.flatnonzero(work_valid[k]).tolist(),
                supported_views=int(work_valid[k].sum()),outside_views=0,policy='insufficient_remaining_observations'))
        work_votes[weak]=np.nan;work_valid[weak]=False
