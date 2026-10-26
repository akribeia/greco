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

"""Joint landmark votes with image-derived adjacent-tilt constraints.

Landmark XYZ remain free. An exact mean(delta_XYZ)=0 constraint is included
in a (2*T+3) Schur/KKT system; this avoids a dense O(landmarks) pose matrix.
The shift differences are measured image residuals, NOT a smooth-shift curve.
"""
import numpy as np
from ..joint_schur import _projection


def solve_joint_image_constraints(votes,angles,edges,targets,strength=.25,f_scale=.2,max_nfev=300,
          tolerance=1e-10,workspace=None,device='gpu',quality=None):
    from ..joint_schur import solve_joint_votes
    votes=np.asarray(votes,float);angles=np.asarray(angles,float)
    edges=np.asarray(edges,int).reshape(-1,2);targets=np.asarray(targets,float).reshape(-1,2)
    if strength==0 or not len(edges):
        return solve_joint_votes(votes,angles,f_scale,max_nfev,tolerance,workspace,device=device)
    n,t,_=votes.shape
    if targets.shape!=(len(edges),2) or np.any(edges<0) or np.any(edges>=t) or not np.isfinite(targets).all():raise ValueError('Invalid image smoothness constraints')
    if device=='gpu':
        import cupy as xp
        host=xp.asnumpy
    else:xp=np;host=np.asarray
    valid=np.isfinite(votes).all(2)
    if n<2 or not valid.any(0).all() or (valid.sum(1)<2).any():raise ValueError('Insufficient joint observations')
    j=xp.asarray(_projection(angles).reshape(2*t,3));mask=xp.asarray(np.repeat(valid,2,axis=1))
    measured=xp.asarray(np.where(valid[...,None],votes,0).reshape(n,2*t))
    # Build both scalar detector components explicitly.
    d=np.zeros((len(edges),2,2*t))
    for i,(a,b) in enumerate(edges):d[i,:,2*b:2*b+2]=np.eye(2);d[i,:,2*a:2*a+2]=-np.eye(2)
    d=xp.asarray(d.reshape(-1,2*t));target=xp.asarray(targets.ravel())
    counts=valid.sum(0);quality=np.ones(len(edges)) if quality is None else np.asarray(quality,float)
    weights=strength*np.sqrt(counts[edges[:,0]]*counts[edges[:,1]])*quality
    prior_weight=xp.asarray(np.repeat(weights,2));prior_scale=.5
    s=xp.zeros(2*t);r=xp.zeros((n,3))
    def evaluate(s,r):
        e=xp.where(mask,s[None]-r@j.T-measured,0);p=d@s-target
        root=xp.sqrt(1+(e/f_scale)**2);proot=xp.sqrt(1+(p/prior_scale)**2)
        cost=xp.sum(e*e/(root+1))+xp.sum(prior_weight*p*p/(proot+1))
        return e,p,root,proot,float(cost)
    initial_cost=evaluate(s,r)[-1];converged=False
    for iteration in range(1,max_nfev+1):
        e,p,root,proot,cost=evaluate(s,r)
        ge=e/root;gp=prior_weight*p/proot;gs=ge.sum(0)+d.T@gp;gr=-ge@j
        m=xp.arange(1,n)[:,None];gc=(xp.cumsum(gr,axis=0)[:-1]-m*gr[1:])/xp.sqrt(m*(m+1))
        optimality=float(xp.maximum(xp.max(abs(gs)),xp.max(abs(gc))))
        if optimality<=tolerance:converged=True;break
        w=xp.where(mask,1/root if iteration<=3 else 1/root**3,0)
        pw=prior_weight/(proot if iteration<=3 else proot**3)
        cross=-w[:,:,None]*j[None];local=xp.einsum('nq,qd,qe->nde',w,j,j)
        try:inverse=xp.linalg.inv(local)
        except xp.linalg.LinAlgError as error:raise ValueError('Unidentifiable landmark angular support') from error
        eliminated=xp.matmul(cross,inverse);E=eliminated.sum(0)
        schur=xp.diag(w.sum(0))+d.T@(pw[:,None]*d)-xp.einsum('nqd,npd->qp',eliminated,cross)
        br=-gr;bs=-gs
        kkt=xp.zeros((2*t+3,2*t+3));kkt[:2*t,:2*t]=(schur+schur.T)*.5
        kkt[:2*t,2*t:]=-E;kkt[2*t:,:2*t]=-E.T;kkt[2*t:,2*t:]=-inverse.sum(0)
        rhs=xp.concatenate((bs-xp.einsum('nqd,nd->q',eliminated,br),-xp.einsum('nde,ne->d',inverse,br)))
        try:step=xp.linalg.solve(kkt,rhs)
        except xp.linalg.LinAlgError as error:raise RuntimeError('Singular smooth joint KKT system') from error
        ds=step[:2*t];dr=xp.einsum('nde,ne->nd',inverse,br-xp.einsum('nqd,q->nd',cross,ds)-step[2*t:][None])
        slope=float(gs@ds+xp.sum(gr*dr))
        if not np.isfinite(slope) or slope>=0:raise RuntimeError('Smooth joint step is not descent')
        fraction=1.
        for _ in range(40):
            next_cost=evaluate(s+fraction*ds,r+fraction*dr)[-1]
            if next_cost<=cost+1e-4*fraction*slope:break
            fraction*=.5
        else:raise RuntimeError('Smooth joint line search failed')
        s+=fraction*ds;r+=fraction*dr
        change=float(xp.max(abs(fraction*(ds[None]-dr@j.T))))
        if abs(cost-next_cost)<=tolerance*max(1,cost) and change<=np.sqrt(tolerance):converged=True;break
    if not converged:raise RuntimeError('Smooth joint solver exhausted its iteration budget')
    shifts=host(s).reshape(t,2);xyz=host(r)
    if np.max(abs(xyz.mean(0)))>1e-7:raise RuntimeError('Smooth joint XYZ gauge lost numerical precision')
    residual=(shifts[None]-np.einsum('tad,nd->nta',_projection(angles),xyz)-votes)[valid]
    return shifts,xyz,dict(nfev=iteration,median_after=float(np.median(np.linalg.norm(residual,axis=1))),
        rms_after=float(np.sqrt(np.mean(residual**2))),smoothness_edges=len(edges),smoothness_strength=strength,
        smooth_objective_before=initial_cost,smooth_objective_after=evaluate(s,r)[-1],
        gauge='mean(delta_r_zyx)=0',xyz_fixed=False,backend='gpu_schur_kkt_images',
        objective='soft_l1(landmark votes)+weighted soft_l1(image-derived neighbour residuals)')
