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

"""GPU local WBP and projection with full SO(3) cameras, no single-axis shortcut."""
import numpy as np

_BP=r'''
extern "C" __global__ void bp(const float* patches,const float* C,const unsigned char* valid,float* out,int B,int T,int N){
 int i=blockDim.x*blockIdx.x+threadIdx.x;int V=N*N*N;if(i>=B*V)return;
 int b=i/V,k=i%V;float h=(N-1)*.5f;
 float x=k%N-h,y=(k/N)%N-h,z=k/(N*N)-h,total=0;
 for(int t=0;t<T;t++){
  if(!valid[b*T+t])continue;const float* c=C+9*t;
  float u=c[0]*x+c[1]*y+c[2]*z+h,v=c[3]*x+c[4]*y+c[5]*z+h;
  if(u<0||v<0||u>N-1||v>N-1)continue;
  int ix=(int)floorf(u),iy=(int)floorf(v),jx=min(ix+1,N-1),jy=min(iy+1,N-1);float a=u-ix,d=v-iy;
  const float* p=patches+(b*T+t)*N*N;
  total+=(1-a)*(1-d)*p[iy*N+ix]+a*(1-d)*p[iy*N+jx]+(1-a)*d*p[jy*N+ix]+a*d*p[jy*N+jx];
 }
 out[i]=total;
}'''
_PROJECT=r'''
extern "C" __global__ void project(const float* vol,const float* C,float* out,int B,int N){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=B*N*N)return;int b=i/(N*N),k=i%(N*N);float h=(N-1)*.5f;
 float u=k%N-h,v=k/N-h,total=0;const float* p=vol+b*N*N*N;
 for(int l=0;l<N;l++){
  float w=l-h,x=C[0]*u+C[3]*v+C[6]*w+h,y=C[1]*u+C[4]*v+C[7]*w+h,z=C[2]*u+C[5]*v+C[8]*w+h;
  if(x<0||y<0||z<0||x>N-1||y>N-1||z>N-1)continue;
  int ix=(int)floorf(x),iy=(int)floorf(y),iz=(int)floorf(z),jx=min(ix+1,N-1),jy=min(iy+1,N-1),jz=min(iz+1,N-1);
  float a=x-ix,d=y-iy,e=z-iz;
  total+=(1-e)*((1-d)*((1-a)*p[(iz*N+iy)*N+ix]+a*p[(iz*N+iy)*N+jx])+d*((1-a)*p[(iz*N+jy)*N+ix]+a*p[(iz*N+jy)*N+jx]))
         +e*((1-d)*((1-a)*p[(jz*N+iy)*N+ix]+a*p[(jz*N+iy)*N+jx])+d*((1-a)*p[(jz*N+jy)*N+ix]+a*p[(jz*N+jy)*N+jx]));
 }
 out[i]=total;
}'''

class Engine:
    def __init__(self,n=33):
        import cupy as cp
        self.cp=cp;self.n=n;self.bp_kernel=cp.RawKernel(_BP,'bp');self.project_kernel=cp.RawKernel(_PROJECT,'project')
        self.mask=cp.asarray(__import__('greco.helpers',fromlist=['soft_spherical_mask']).soft_spherical_mask())

    def project(self,volumes,camera):
        cp=self.cp;v=cp.ascontiguousarray(volumes,cp.float32);c=cp.asarray(camera,cp.float32);b=len(v);n=self.n
        out=cp.empty((b,n,n),cp.float32)
        self.project_kernel(((b*n*n+255)//256,),(256,),(v,c,out,np.int32(b),np.int32(n)))
        return out

    def filter(self,patches,beta):
        cp=self.cp;n=self.n;pad=max(64,2**int(np.ceil(np.log2(2*n))))
        fy=cp.fft.fftfreq(pad)[None,:,None];fx=cp.fft.fftfreq(pad)[None,None,:];b=cp.asarray(np.deg2rad(beta))[:,None,None]
        ramp=(2*cp.abs(fx*cp.cos(b)+fy*cp.sin(b))).astype(cp.float32)
        padded=cp.pad(patches,((0,0),(0,0),(0,pad-n),(0,pad-n)))
        return cp.ascontiguousarray(cp.fft.ifft2(cp.fft.fft2(padded,axes=(-2,-1))*ramp[None],axes=(-2,-1)).real[:,:,:n,:n],cp.float32)

    def backproject(self,filtered,cameras,valid):
        cp=self.cp;b,t,n,_=filtered.shape
        out=cp.empty((b,n,n,n),cp.float32)
        self.bp_kernel(((b*n**3+255)//256,),(256,),
            (cp.ascontiguousarray(filtered,cp.float32),cp.asarray(cameras,cp.float32),cp.asarray(valid,cp.uint8),out,np.int32(b),np.int32(t),np.int32(n)))
        return out

    def measure(self,stack,positions,cameras,shifts,frozen,raster_axes,valid,beta,batch=48,radius=4,target_cameras=None,target_shifts=None):
        from .gamma_geometry import project
        from greco.gcc import raw_sample_bases,raw_patch_stacks_gpu,raw_search_regions_gpu,raw_search_support
        from greco.local_registration_gpu import register_local_models_gpu
        from greco.angular_transfer_gpu import angular_transfer_operator_gpu,apply_angular_transfer_operator_gpu
        cp=self.cp;n=self.n;count,t=valid.shape;c=(stack.shape[-1]-1)/2
        centres=project(positions,cameras)+c;by,bx=raw_sample_bases(centres,shifts,frozen,raster_axes,stack.shape[-1])
        target_cameras=cameras if target_cameras is None else target_cameras
        target_shifts=shifts if target_shifts is None else target_shifts
        tc=project(positions,target_cameras)+c
        tby,tbx=raw_sample_bases(tc,target_shifts,frozen,raster_axes,stack.shape[-1])
        votes=np.full((count,t,2),np.nan);boundary=0
        self.last_boundary_mask=np.zeros((count,t),bool)
        # Reconstruct an impulse with the same full camera matrices for each LOO support.
        impulse=cp.zeros((1,n,n,n),cp.float32);impulse[:,n//2,n//2,n//2]=1
        ip=cp.stack([self.project(impulse,cam)[0] for cam in cameras],axis=0)[None]
        ipf=self.filter(ip,beta)
        transfer_cache={}
        for start in range(0,count,batch):
            end=min(start+batch,count)
            patches,inside=raw_patch_stacks_gpu(stack,by[start:end],bx[start:end],valid[start:end],n//2,raster_axes)
            support=valid[start:end]&inside
            f=self.filter(patches,beta);full=self.backproject(f,cameras,support)
            for target,cam in enumerate(target_cameras):
                eligible=support[:,target]&(support.sum(1)>=4)
                eligible&=raw_search_support(tby[start:end,target],tbx[start:end,target],n//2+radius,raster_axes[target],stack.shape[-1])
                ids=np.flatnonzero(eligible)
                if not len(ids):continue
                one=self.backproject(f[ids,target:target+1],cameras[target:target+1],np.ones((len(ids),1),bool))
                loo=(full[ids]-one)*cp.asarray(np.pi/(2*(support[ids].sum(1)-1)),cp.float32)[:,None,None,None]
                operators=[]
                for index in ids:
                    mask=support[index].copy();mask[target]=False;key=mask.tobytes()
                    if key not in transfer_cache:
                        psf=self.backproject(ipf,cameras,mask[None])[0]*np.pi/(2*mask.sum())
                        h=cp.fft.fftn(cp.fft.ifftshift(psf));dc=h[0,0,0]
                        if abs(complex(dc))<1e-12:raise ValueError('Zero impulse-response DC')
                        if len(transfer_cache)>=256:transfer_cache.pop(next(iter(transfer_cache)))
                        transfer_cache[key]=angular_transfer_operator_gpu(h/dc,.05,4.,.8)
                    operators.append(transfer_cache[key])
                models=apply_angular_transfer_operator_gpu(loo,cp.stack(operators))*self.mask[None]
                projected=self.project(models,cam)
                bases=np.column_stack((tby[start+ids,target],tbx[start+ids,target]))
                regions=raw_search_regions_gpu(stack,target,bases,n//2+radius,raster_axes)
                results=register_local_models_gpu(projected,regions,method='generalized_cc',search_radius=radius,alpha=.25)
                for i,result in zip(ids,results):
                    votes[start+i,target]=result.correction_yx
                    boundary+=int(result.boundary)
                    self.last_boundary_mask[start+i,target]=bool(result.boundary)
            if end==count:print(f'Gamma: measured {count} fixed landmarks.',flush=True)
        return votes,dict(observations=int(np.isfinite(votes).all(2).sum()),boundary=boundary,transfer_sets=len(transfer_cache))
