# Greco optimizer — implementation of the GRECO algorithm
# GRECO: Geometric Registration of Electron Cryotomography Observations
# Copyright (C) 2026 Carlos Fernández Rodríguez
#
# This file is part of Greco.
#
# Greco is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3 only.
#
# Greco is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Greco. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-only

"""GPU block Schur extension: shifts, free XYZ, common/free roll and free tilt.
Scalar soft_l1 objective, float64 arithmetic, explicit XYZ translation gauge.
Free inclination uses the central view as the global tilt orientation gauge.
No pose prior and no angular smoothing penalty.
"""
import time
import numpy as np
from ..solver import projection_jacobians
from ..helpers import projected_centres
from .advanced_options import NUMERICAL_SOLVER_MAX_ITERATIONS


def fit(votes,positions,angles,size,mode='common',f_scale=.2,maxiter=NUMERICAL_SOLVER_MAX_ITERATIONS,device='gpu',tolerance=1e-9):
    start=time.monotonic()
    if device=='gpu':
        import cupy as xp
        host=xp.asnumpy
    elif device=='cpu':xp=np;host=np.asarray
    else:raise ValueError(device)
    votes=np.asarray(votes,float);positions=np.asarray(positions,float);angles=np.asarray(angles,float)
    n,t,_=votes.shape;valid=np.isfinite(votes).all(2)
    if mode not in ('fixed','common','roll','tilt','tilt_only','tilt_common'):raise ValueError(mode)
    if n<2 or np.any(valid.sum(0)<2) or np.any(valid.sum(1)<3):raise ValueError('Insufficient angular observations')
    P=projection_jacobians(angles);flat=P.reshape(2*t,3)
    basis=np.linalg.qr(flat,mode='complete')[0][:,3:]
    anchor=int(np.argmin(abs(angles)));free=np.delete(np.arange(t),anchor)
    nr=0 if mode in ('fixed','tilt_only') else 1 if mode in ('common','tilt_common') else t;nt=t-1 if mode in ('tilt','tilt_only','tilt_common') else 0
    d=2*t-3+nr+nt
    Q=np.zeros((t,4,d));Q[:,:2,:2*t-3]=basis.reshape(t,2,-1)
    if mode in ('common','tilt_common'):Q[:,2,2*t-3]=1
    elif mode in ('roll','tilt'):Q[np.arange(t),2,2*t-3+np.arange(t)]=1
    if nt:Q[free,3,2*t-3+nr+np.arange(t-1)]=1
    centre=projected_centres(positions,angles,size)-(size-1)/2
    G=np.zeros((n,t,2,4));G[:,:,0,0]=1;G[:,:,1,1]=1
    G[:,:,0,2]=-centre[:,:,1]*np.pi/180;G[:,:,1,2]=centre[:,:,0]*np.pi/180
    rad=np.deg2rad(angles);c=(size-1)/2
    G[:,:,1,3]=(np.sin(rad)[None,:]*(positions[:,2,None]-c)+np.cos(rad)[None,:]*(positions[:,0,None]-c))*np.pi/180
    G=xp.asarray(G);R=xp.asarray(-P);Q=xp.asarray(Q.reshape(4*t,d))
    mask=xp.asarray(valid[:,:,None]);measured=xp.asarray(np.where(valid[:,:,None],votes,0.))
    globals=xp.zeros((t,4),dtype=xp.float64);local=xp.zeros((n,3),dtype=xp.float64)
    def predict(g,r):return xp.einsum('ntca,ta->ntc',G,g)+xp.einsum('tca,na->ntc',R,r)
    def objective(res):
        root=xp.sqrt(1+(res/f_scale)**2)
        return float(xp.sum(res*res/(root+1)))
    residual=xp.where(mask,-measured,0.)
    converged=False;steps=0;line_trials=0;rank_min=None
    for iteration in range(1,maxiter+1):
        root=xp.sqrt(1+(residual/f_scale)**2);grad=xp.where(mask,residual/root,0.)
        cost=objective(residual)
        gg=xp.einsum('ntca,ntc->ta',G,grad).reshape(-1)
        gr=xp.einsum('tca,ntc->na',R,grad)
        optimality=float(xp.maximum(xp.max(xp.abs(Q.T@gg)),xp.max(xp.abs(gr))))
        if optimality<tolerance:converged=True;break
        w=xp.where(mask,1/root if iteration<=3 else 1/root**3,0.)
        Hlocal=xp.einsum('ntc,tca,tcb->nab',w,R,R)
        inv=xp.linalg.inv(Hlocal)
        cross=xp.einsum('ntc,ntca,tcb->ntab',w,G,R).reshape(n,4*t,3)
        blocks=xp.einsum('ntc,ntca,ntcb->tab',w,G,G)
        Hg=xp.zeros((4*t,4*t),dtype=xp.float64)
        for k in range(t):Hg[4*k:4*k+4,4*k:4*k+4]=blocks[k]
        eliminated=xp.matmul(cross,inv)
        flat_cross=cross.transpose(0,2,1).reshape(n*3,4*t)
        subtract=eliminated.transpose(1,0,2).reshape(4*t,n*3)@flat_cross
        Schur=Hg-subtract;small=Q.T@((Schur+Schur.T)*.5)@Q;small=(small+small.T)*.5
        scale=xp.sqrt(xp.maximum(xp.diag(small),np.finfo(float).tiny))
        normalized=small/scale[:,None]/scale[None,:]
        if iteration==1:
            eig=host(xp.linalg.eigvalsh(normalized));rank_min=float(eig[0])
            if not np.isfinite(eig).all() or eig[0]<=np.finfo(float).eps*max(n,d)*eig[-1]:raise RuntimeError(f'Angular Schur rank deficient: {eig[0]:.3g}, mode={mode}')
        rhs=-gg+xp.einsum('nqa,na->q',eliminated,gr)
        scaled_rhs=(Q.T@rhs)/scale
        dgflat=Q@(xp.linalg.solve(normalized,scaled_rhs)/scale)
        dr=-xp.einsum('nab,nb->na',inv,gr+xp.einsum('nqa,q->na',cross,dgflat))
        # Exact translation gauge: prediction is unchanged.
        mean=dr.mean(0);dr-=mean;dg=dgflat.reshape(t,4);dg[:,:2]-=xp.einsum('tca,a->tc',-R,mean)
        delta=xp.where(mask,predict(dg,dr),0.)
        slope=float(xp.sum(grad*delta))
        if not np.isfinite(slope) or slope>=0:
            if optimality<1e-6:converged=True;break
            raise RuntimeError('Angular Schur direction is not descent')
        step=1.
        for trial in range(35):
            proposed=objective(residual+step*delta);line_trials+=1
            if proposed<=cost+1e-4*step*slope:break
            step*=.5
        else:raise RuntimeError('Angular Schur line search failed')
        globals+=step*dg;local+=step*dr;residual=xp.where(mask,predict(globals,local)-measured,0.);steps+=1
        change=float(xp.max(xp.abs(step*delta)))
        if abs(cost-proposed)<tolerance*max(1.,cost) and change<np.sqrt(tolerance):converged=True;break
    if not converged:raise RuntimeError(f'Angular Schur did not converge in {maxiter} iterations')
    g=host(globals);dr=host(local);r=host(residual)[valid]
    report=dict(max_iterations=maxiter,numerical_tolerance=tolerance,observations=int(valid.sum()),iterations=iteration,rms_before=float(np.sqrt(np.mean(votes[valid]**2))),rms_after=float(np.sqrt(np.mean(r*r))),median_after=float(np.median(np.linalg.norm(r,axis=1))),robust_cost=objective(residual),optimality=optimality,backend='gpu_schur_angular' if device=='gpu' else 'cpu_schur_angular',arithmetic='float64',mode=mode,global_system_size=d,rank_min_scaled=rank_min,solve_time_s=time.monotonic()-start,line_search_trials=line_trials,tilt_anchor_index=anchor if mode in ('tilt','tilt_only','tilt_common') else None,rotation_min_deg=float(g[:,2].min()),rotation_max_deg=float(g[:,2].max()),rotation_mean_deg=float(g[:,2].mean()))
    return g[:,:2],dr,g[:,2],g[:,3],report
