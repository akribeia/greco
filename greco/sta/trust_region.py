"""Independent six-DOF Gauss–Newton solves with actual likelihood acceptance.

GPU objectives accumulate J^T W J; the small 6x6 systems are solved together.
The trust metric measures rotations at the particle surface in image pixels.
No tanh saturation, shared line search, or persistent particle freezing.
"""
import numpy as np


def optimize_trust_region(objective,maxiter=30,detect_stagnation=False):
    count=objective.count
    radius=max(float(getattr(objective,'surface_radius_pixels',objective.backend.n/2)),1.)
    scale=np.array([1/radius]*3+[1.]*3)
    limits=np.asarray(getattr(objective,'step_limits',[np.pi]*3+[objective.backend.n/8]*3),float)
    if limits.shape!=(6,) or not np.isfinite(limits).all() or (limits<=0).any():raise ValueError('Invalid local limits.')
    # An increment is local around the current pose; no hidden pose prior.
    objective.prior[:]=np.inf
    parameters=np.zeros((count,6));active=np.ones(count,bool)
    stats,hessian=objective.normal_equations(parameters)
    before=stats[:,0].copy();initial=before.copy()
    damping=np.full(count,1e-3);trust=np.full(count,min(2.,np.linalg.norm(limits/scale)))
    stationary=np.zeros(count,bool);stalled=np.zeros(count,bool)
    plateau=np.zeros(count,bool);plateau_streak=np.zeros(count,int)
    numerical=np.zeros(count,bool)
    recent_motion=np.full((count,5),np.inf);recent_gain=np.full((count,5),np.inf)
    poll_blocked=np.zeros(count,bool);poll_origin=np.zeros((count,6))
    poll_evaluations=0
    failed=~np.isfinite(stats).all(axis=1)|~np.isfinite(hessian).all(axis=(1,2))
    active[failed]=False
    accepted=np.zeros(count,int);rejected=np.zeros(count,int);iterations=np.zeros(count,int)
    conditioning=np.zeros(count)
    for iteration in range(maxiter):
        ids=np.flatnonzero(active)
        if not len(ids):break
        iterations[ids]+=1
        g=stats[ids,1:]*scale;h=hessian[ids]*scale[None,:,None]*scale[None,None,:]
        diagonal=np.maximum(np.diagonal(h,axis1=1,axis2=2),np.maximum(np.max(np.diagonal(h,axis1=1,axis2=2),axis=1)*1e-8,1e-20)[:,None])
        # Projected gradient accounts for physical bounds, not a saturating map.
        projected=g.copy()
        outward=((parameters[ids]>=limits*(1-1e-10))&(g<0))|((parameters[ids]<=-limits*(1-1e-10))&(g>0))
        projected[outward]=0.
        gradient_norm=np.max(abs(projected)/np.sqrt(diagonal),axis=1)
        quiet=gradient_norm<=1e-6*np.maximum(1.,np.sqrt(2*stats[ids,0]))
        # A tiny damped/clipped trial alone proves nothing. Estimate remaining
        # physical motion with the UNDAMPED free-coordinate Fisher system.
        free=~outward
        probe=h*free[:,:,None]*free[:,None,:]
        probe[:,np.arange(6),np.arange(6)]+=np.where(outward,1.,diagonal*1e-8)
        remaining=np.linalg.solve(probe,-(projected*free)[...,None])[...,0]
        predicted_remaining=np.maximum(0.,-.5*np.sum(projected*remaining,axis=1))
        practical=np.isfinite(remaining).all(axis=1)&(np.linalg.norm(remaining,axis=1)<=.002)&(predicted_remaining<=1e-3)
        plateau_streak[ids]=np.where(practical,plateau_streak[ids]+1,0)
        plateaud=plateau_streak[ids]>=3
        plateau[ids[plateaud & ~quiet]]=True
        stationary[ids[quiet]]=True;active[ids[quiet]]=False
        active[ids[plateaud]]=False
        working=~(quiet|plateaud)
        if not working.any():continue
        ids=ids[working];g=g[working];h=h[working];diagonal=diagonal[working]
        fixed=outward[working]
        # Solve on the free coordinates. Merely clipping an unconstrained
        # Newton step lets an outward bound direction spoil tangential updates.
        free=~fixed
        system=h*free[:,:,None]*free[:,None,:]
        system[:,np.arange(6),np.arange(6)]+=np.where(fixed,1.,damping[ids,None]*diagonal)
        eigen=np.linalg.eigvalsh(h)
        conditioning[ids]=eigen[:,-1]/np.maximum(eigen[:,0],np.maximum(eigen[:,-1]*1e-12,1e-20))
        step=np.linalg.solve(system,-(g*free)[...,None])[...,0]
        step*=np.minimum(1.,trust[ids]/np.maximum(np.linalg.norm(step,axis=1),1e-30))[:,None]
        candidate=parameters.copy()
        candidate[ids]=np.clip(parameters[ids]+step*scale,-limits,limits)
        step=(candidate[ids]-parameters[ids])/scale
        predicted=-np.sum(g*step,axis=1)-.5*np.einsum('bi,bij,bj->b',step,h,step)
        trial_active=np.zeros(count,bool);trial_active[ids]=True
        trial=objective.costs(candidate,trial_active)
        reduction=stats[ids,0]-trial[ids]
        floor=np.maximum(1e-12,64*np.finfo(float).eps*np.maximum(abs(stats[ids,0]),1.))
        ratio=reduction/np.maximum(predicted,1e-30)
        good=np.isfinite(trial[ids])&(predicted>0)&(reduction>floor)&(ratio>.05)
        accepted[ids[good]]+=1;rejected[ids[~good]]+=1
        parameters[ids[good]]=candidate[ids[good]]
        if detect_stagnation:
            # A window of actual accepted displacement/gain, not the size of
            # a damped proposal or the relative change of a huge residual.
            slot=iteration%5
            recent_motion[ids,slot]=np.where(good,np.linalg.norm(step,axis=1),0.)
            recent_gain[ids,slot]=np.where(good,reduction,0.)
            # A probe that finds useful improvement rules out saturation at
            # this location. Recheck only after meaningful physical movement,
            # not every five rejected trials at the identical pose.
            moved=np.linalg.norm((parameters[ids]-poll_origin[ids])/scale,axis=1)>.002
            poll_blocked[ids[moved]]=False
            suspect=(iteration>=9)&~poll_blocked[ids]&(np.sum(recent_motion[ids],axis=1)<=.0002)&(np.sum(recent_gain[ids],axis=1)<=1e-5)
            candidates=ids[suspect]
            if len(candidates):
                # Test undamped directions at fixed physical scales. A shrunken
                # trust radius alone must NEVER establish numerical saturation.
                local=np.flatnonzero(suspect)
                gn=remaining[working][local]
                descent=-g[local]/diagonal[local]
                improved=np.zeros(len(candidates),bool)
                for direction in (gn,descent):
                    direction=direction/np.maximum(np.linalg.norm(direction,axis=1),1e-30)[:,None]
                    for distance in (.0002,.002,.01):
                        for sign in (-1.,1.):
                            proposal=parameters.copy()
                            proposal[candidates]=np.clip(parameters[candidates]+sign*distance*direction*scale,-limits,limits)
                            probe_active=np.zeros(count,bool);probe_active[candidates]=True
                            values=objective.costs(proposal,probe_active);poll_evaluations+=1
                            current_cost=np.where(good[local],trial[candidates],stats[candidates,0])
                            improved|=np.isfinite(values[candidates])&(current_cost-values[candidates]>np.maximum(1e-5,floor[local]))
                stopped=candidates[~improved]
                numerical[stopped]=True;active[stopped]=False
                poll_blocked[candidates[improved]]=True
                poll_origin[candidates[improved]]=parameters[candidates[improved]]
                # Do not poll the same unchanged location every iteration.
                recent_motion[candidates[improved],:]=np.inf
                recent_gain[candidates[improved],:]=np.inf
        damping[ids]=np.where(good,np.maximum(1e-8,damping[ids]*np.where(ratio>.75,.3,1.)),np.minimum(1e12,damping[ids]*5))
        trust[ids]=np.where(good & (ratio>.75),np.minimum(trust[ids]*2,8.),np.where(~good,trust[ids]*.5,trust[ids]))
        # Small unresolved steps are reported, never relabelled successful.
        exhausted=((trust[ids]<1e-5)|(damping[ids]>=1e12))&~numerical[ids]
        stalled[ids[exhausted]]=True;active[ids[exhausted]]=False
        refresh=np.zeros(count,bool);refresh[ids[good]]=True
        if refresh.any():
            new_stats,new_hessian=objective.normal_equations(parameters,refresh)
            bad=refresh & (~np.isfinite(new_stats).all(axis=1)|~np.isfinite(new_hessian).all(axis=(1,2)))
            failed[bad]=True;active[bad]=False
            stats[refresh]=new_stats[refresh];hessian[refresh]=new_hessian[refresh]
    after=objective.costs(parameters)
    bad=(~np.isfinite(after))|(~np.isfinite(parameters).all(axis=1))|(after>before+1e-8)
    parameters[bad]=0.;after[bad]=before[bad];failed|=bad;stationary[bad]=False;plateau[bad]=False;numerical[bad]=False
    at_bounds=np.any(abs(parameters)>=.95*limits,axis=1)
    # All particles may participate again when the reference changes next pass.
    return parameters,dict(initial_cost=float(initial.sum()),final_cost=float(after.sum()),
        evaluations=objective.evaluations,optimizer_iterations=int(iterations.max()),
        optimizer_success=bool((stationary|plateau).all()),rejected_particles=int(bad.sum()),
        optimizer_backend='Independent Gauss–Newton / Levenberg–Marquardt trust region',
        optimizer_converged_particles=int(stationary.sum()),optimizer_limited_particles=int((active & ~failed).sum()),
        optimizer_plateau_particles=int(plateau.sum()),
        optimizer_numerical_plateau_particles=int(numerical.sum()),
        optimizer_failed_particles=int(failed.sum()),optimizer_native_failed_particles=0,
        optimizer_stationary_particles=0,optimizer_stalled_particles=int((stalled & ~failed).sum()),
        optimizer_neighborhood_bound_particles=int(at_bounds.sum()),
        optimizer_statuses=np.where(failed,-1,np.where(stationary,0,np.where(plateau,3,np.where(numerical,4,np.where(stalled,2,1))))).tolist(),
        trust_accepted_steps=int(accepted.sum()),trust_rejected_steps=int(rejected.sum()),
        trust_median_condition=float(np.median(conditioning)),optimizer_poll_evaluations=poll_evaluations)
