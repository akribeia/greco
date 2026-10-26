"""Bounded-work direct polling of a fixed-reference six-DOF likelihood.

No gradient convergence is claimed. Certification means no tested direction
improves cost at the stated specimen/particle-surface displacement scale.
"""
import numpy as np


def refine_poll(objective,initial,active,surface_radius_pixels,minimum_motion=.025,
                initial_motion=.25,max_cycles=16):
    parameters=np.array(initial,dtype=np.float64,copy=True)
    active=np.array(active,dtype=bool,copy=True);requested=active.copy()
    if not 0<minimum_motion<=initial_motion or max_cycles<1:
        raise ValueError('Invalid local polling scale or cycle limit.')
    units=np.array([1/max(float(surface_radius_pixels),1.)]*3+[1.]*3)
    rho=np.full(objective.count,initial_motion);certified=np.zeros(objective.count,bool)
    evaluations=0;cycles=np.zeros(objective.count,dtype=int)
    stats=objective.per_particle(parameters,active);evaluations+=1
    cost=stats[:,0].copy();initial_cost=cost.copy()
    for _ in range(max_cycles):
        if not active.any():break
        cycles[active]+=1;best=parameters.copy();best_cost=cost.copy()
        # A positive spanning coordinate set, plus the scaled gradient direction.
        gradient=stats[:,1:]*units
        norm=np.linalg.norm(gradient,axis=1)
        gradient_direction=-gradient/np.maximum(norm[:,None],1e-300)
        for direction in [*np.eye(6),gradient_direction]:
            for sign in (-1,1):
                trial=parameters.copy();trial[active]+=sign*rho[active,None]*units*(direction if direction.ndim==1 else direction[active])
                limits=getattr(objective,'step_limits',None)
                if limits is not None:trial[active]=np.clip(trial[active],-limits,limits)
                value=objective.per_particle(trial,active)[:,0];evaluations+=1
                # Distinguish an actual decrease from accumulated FP roundoff.
                floor=64*np.finfo(float).eps*np.maximum(abs(best_cost),1.)
                improved=active&np.isfinite(value)&(value<best_cost-floor)
                best[improved]=trial[improved];best_cost[improved]=value[improved]
        moved=active&(best_cost<cost)
        parameters[moved]=best[moved];cost[moved]=best_cost[moved]
        quiet=active&~moved
        done=quiet&(rho<=minimum_motion)
        certified[done]=True;active[done]=False
        rho[quiet&~done]=np.maximum(minimum_motion,.5*rho[quiet&~done])
        if active.any():
            stats=objective.per_particle(parameters,active);evaluations+=1
    return parameters,dict(certified=certified,requested=requested,unresolved=requested&~certified,
                           cycles=cycles,evaluations=evaluations,minimum_motion_pixels=minimum_motion,
                           initial_cost=initial_cost,final_cost=cost)
