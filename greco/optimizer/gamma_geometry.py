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

"""Experimental fixed-landmark camera geometry and exact minimum-change fit."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

def cameras(alpha,beta=0.,gamma=0.):
    alpha,beta,gamma=np.broadcast_arrays(alpha,beta,gamma)
    values=np.stack((alpha,beta,gamma),-1).reshape(-1,3)
    output=[]
    for a,b,g in values:
        output.append((Rotation.from_euler('z',b,degrees=True)*Rotation.from_euler('x',g,degrees=True)*Rotation.from_euler('y',-a,degrees=True)*Rotation.from_euler('x',-g,degrees=True)).as_matrix())
    return np.asarray(output).reshape(alpha.shape+(3,3))



def project(points,rotations):
    return np.einsum('tij,nj->nti',rotations[:,[1,0]],points)

def jacobians(points,rotations):
    v=np.einsum('tij,nj->nti',rotations,points)
    n,t,_=v.shape
    G=np.zeros((n,t,2,5));G[:,:,0,0]=1;G[:,:,1,1]=1
    G[:,:,0,2]=v[:,:,2];G[:,:,0,4]=-v[:,:,0]
    G[:,:,1,3]=-v[:,:,2];G[:,:,1,4]=v[:,:,1]
    G[:,:,:,2:]*=np.pi/180
    return G,-rotations[:,[1,0],:]


def parameter_jacobians(points,rotations,angles):
    G3,R=jacobians(points,rotations)
    a,b,g=np.deg2rad(np.asarray(angles)).T
    da=np.column_stack((np.sin(b)*np.cos(g),-np.cos(b)*np.cos(g),-np.sin(g)))
    db=np.zeros_like(da);db[:,2]=1
    dg=np.column_stack((np.cos(b)*(1-np.cos(a))-np.sin(b)*np.sin(g)*np.sin(a),
                        np.sin(b)*(1-np.cos(a))+np.cos(b)*np.sin(g)*np.sin(a),
                        -np.cos(g)*np.sin(a)))
    directions=np.stack((da,db,dg),axis=-1)
    G=G3.copy();G[:,:,:,2:]=np.einsum('ntca,tak->ntck',G3[:,:,:,2:],directions)
    return G,R


def parameter_scales(points):
    centre=np.mean(points,axis=0);radius=float(np.sqrt(np.mean(np.sum((points-centre)**2,axis=1))))
    if radius<1:raise ValueError('Degenerate landmark radius')
    angle=180/(np.pi*radius)
    return np.array([1.,1.,angle,angle,angle]),radius

def cost_gradient(u,points,reference_projection,alpha_source,observed,weight,scales):
    angles=np.array([[alpha_source+u[2],u[3],u[4]]]);rot=cameras(*angles.T)
    prediction=project(points,rot)[:,0]-u[:2]
    residual=reference_projection-prediction-observed;root=np.sqrt(1+(residual/.2)**2)
    data=float(np.sum(residual**2/(root+1))/len(points))
    G,_=parameter_jacobians(points,rot,angles)
    grad=np.einsum('nca,nc->a',G[:,0],residual/root)/len(points)
    penalty=.5*weight*np.sum((u/scales)**2);grad+=weight*u/scales**2
    return data+penalty,grad

def fit_view(points,alpha_source,current,corrections,weight,scales):
    currot=cameras(alpha_source+current[2],current[3],current[4])
    reference=project(points,currot[None])[:,0]-current[:2]
    bounds=[(None,None),(None,None),((-179.9-alpha_source)/scales[2],(179.9-alpha_source)/scales[2]),(-179.9/scales[3],179.9/scales[3]),(-89./scales[4],89./scales[4])]
    def fun(q):
        value,grad=cost_gradient(q*scales,points,reference,alpha_source,corrections,weight,scales)
        return value,grad*scales
    result=minimize(fun,current/scales,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=2000,ftol=1e-14,gtol=1e-10,maxls=50))
    value,g=fun(result.x);pg=g.copy()
    for j,(lo,hi) in enumerate(bounds):
        if (lo is not None and result.x[j]<=lo+1e-8 and g[j]>0) or (hi is not None and result.x[j]>=hi-1e-8 and g[j]<0):pg[j]=0
    if not result.success and np.max(abs(pg))>1e-6:raise RuntimeError(f'Minimum-change fit failed: {result.message}, gradient={max(abs(pg))}')
    return result.x*scales,dict(iterations=int(result.nit),objective=float(value),normalized_gradient=float(max(abs(pg))),normalized_change=float(np.linalg.norm(result.x)),success=bool(result.success))
