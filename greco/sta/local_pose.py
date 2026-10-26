"""Greco local local six-parameter particle MAP alignment on Fourier views.

Analytic trilinear-map and phase-shift derivatives. Double geometry/reduction;
no Euler grid search or numerical finite-difference gradient on the GPU.
"""
import numpy as np

KERNEL=r'''
#include <cuda_fp16.h>
#ifdef GRECO_HALF_REFERENCE
typedef __half2 ReferenceValue;
__device__ double2 reference_value(const ReferenceValue* map,long index,double scale){
 float2 v=__half22float2(map[index]);return make_double2((double)v.x*scale,(double)v.y*scale);
}
#else
typedef double2 ReferenceValue;
__device__ double2 reference_value(const ReferenceValue* map,long index,double scale){return map[index];}
#endif
template<int CHANNELS> __device__ void evaluate_pose(long i,const ReferenceValue* map,const float2* data,
 const float* ctf,const double* inv,const double* deriv,const double* view,
 const double* shift,const double* precision,const int* frequencies,int nf,int b,int n,int p,double map_scale,double radius,double* out){
 int h=n/2+1;
 long stride=nf;if(i>=b*stride)return;
 int o=i/stride,frequency=frequencies[i%stride],x=frequency%h,row=frequency/h,y=row<=n/2?row:row-n;
 for(int g=0;g<CHANNELS;g++)out[g]=0;
 if(x*x+y*y>radius*radius || (x==0&&y==0))return;
 const double* m=inv+o*9;double a[3]={2*(m[0]*x+m[1]*y),2*(m[3]*x+m[4]*y),2*(m[6]*x+m[7]*y)};
 double sign=a[0]<0?-1:1;for(int j=0;j<3;j++)a[j]*=sign;
 int ix=floor(a[0]),iy=floor(a[1])+p/2,iz=floor(a[2])+p/2,ph=p/2+1;
 if(ix<0||ix+1>=ph||iy<0||iy+1>=p||iz<0||iz+1>=p)return;
 double f[3]={a[0]-floor(a[0]),a[1]-floor(a[1]),a[2]-floor(a[2])};
 double re=0,im=0,gr[3]={0,0,0},gi[3]={0,0,0};
 for(int z=0;z<2;z++)for(int yy=0;yy<2;yy++)for(int xx=0;xx<2;xx++){
  int v[3]={xx,yy,z};double w[3]={xx?f[0]:1-f[0],yy?f[1]:1-f[1],z?f[2]:1-f[2]};
  double2 value=reference_value(map,((long)(iz+z)*p+iy+yy)*ph+ix+xx,map_scale);double wt=w[0]*w[1]*w[2];re+=wt*value.x;im+=wt*value.y;
  if(CHANNELS>1)for(int j=0;j<3;j++){double dw=(v[j]?1:-1)*w[(j+1)%3]*w[(j+2)%3];gr[j]+=dw*value.x;gi[j]+=dw*value.y;}
 }
 im*=sign;for(int j=0;j<3;j++){gr[j]*=sign;}
 const double* v=view+o*9;const double* s=shift+o*3;
 double tx=v[0]*s[0]+v[1]*s[1]+v[2]*s[2],ty=v[3]*s[0]+v[4]*s[1]+v[5]*s[2];
 double phase=-6.2831853071795864769*(x*tx+y*ty)/n,c=cos(phase),ss=sin(phase),ct=ctf[i];
 double pr=ct*(c*re-ss*im),pi=ct*(ss*re+c*im),dr=pr-data[i].x,di=pi-data[i].y,wt=precision[i];
 out[0]=.5*wt*(dr*dr+di*di);
 if(CHANNELS==1)return;
 double jr[6],ji[6];
 for(int j=0;j<3;j++){
  const double* d=deriv+(o*3+j)*9;double dpr=0,dpi=0;
  for(int q=0;q<3;q++){double da=2*(d[q*3]*x+d[q*3+1]*y);dpr+=gr[q]*da;dpi+=gi[q]*da;}
  double rr=ct*(c*dpr-ss*dpi),ii=ct*(ss*dpr+c*dpi);
  jr[j]=rr;ji[j]=ii;
  out[j+1]=wt*(dr*rr+di*ii);
  double dp=-6.2831853071795864769*(x*v[j]+y*v[3+j])/n;
  jr[j+3]=-pi*dp;ji[j+3]=pr*dp;
  out[j+4]=wt*(dr*(-pi*dp)+di*(pr*dp));
 }
 if(CHANNELS==28){
  int k=7;for(int j=0;j<6;j++)for(int l=j;l<6;l++)out[k++]=wt*(jr[j]*jr[l]+ji[j]*ji[l]);
 }
}
extern "C" __global__ void local_cost(const ReferenceValue* map,const float2* data,
 const float* ctf,const double* inv,const double* deriv,const double* view,
 const double* shift,const double* precision,const int* frequencies,int nf,int b,int n,int p,double map_scale,double radius,double* out,const int* active_views){
 int tiles=(nf+255)/256,o=active_views[blockIdx.x/tiles],tile=blockIdx.x%tiles,lane=threadIdx.x;
 int f=tile*256+lane;double value[1]={0};
 if(f<nf)evaluate_pose<1>((long)o*nf+f,map,data,ctf,inv,deriv,view,shift,precision,frequencies,nf,b,n,p,map_scale,radius,value);
 __shared__ double scratch[256];scratch[lane]=value[0];__syncthreads();
 for(int step=128;step;step/=2){if(lane<step)scratch[lane]+=scratch[lane+step];__syncthreads();}
 if(lane==0)out[o*tiles+tile]=scratch[0];
}
extern "C" __global__ void cost_reduce(const double* work,const int* offsets,
 const int* observations,const int* active,int count,int tiles,double* result){
 int particle=blockIdx.x,lane=threadIdx.x;double value=0.;
 int start=offsets[particle],length=(offsets[particle+1]-start)*tiles;
 if(active[particle])for(int i=lane;i<length;i+=256)value+=work[observations[start+i/tiles]*tiles+i%tiles];
 __shared__ double scratch[256];scratch[lane]=value;__syncthreads();
 for(int step=128;step;step/=2){if(lane<step)scratch[lane]+=scratch[lane+step];__syncthreads();}
 if(lane==0)result[particle]=scratch[0];
}
extern "C" __global__ void local_pose(const ReferenceValue* map,const float2* data,
 const float* ctf,const double* inv,const double* deriv,const double* view,
 const double* shift,const double* precision,const int* frequencies,int nf,int b,int n,int p,double map_scale,double radius,double* out,const int* active_views){
 int tiles=(nf+255)/256,o=active_views[blockIdx.x/tiles],tile=blockIdx.x%tiles,lane=threadIdx.x;
 int f=tile*256+lane;double values[7]={0,0,0,0,0,0,0};
 if(f<nf)evaluate_pose<7>((long)o*nf+f,map,data,ctf,inv,deriv,view,shift,precision,frequencies,nf,b,n,p,map_scale,radius,values);
 __shared__ double scratch[7][256];
 for(int g=0;g<7;g++)scratch[g][lane]=values[g];
 __syncthreads();
 for(int step=128;step;step/=2){
  if(lane<step)for(int g=0;g<7;g++)scratch[g][lane]+=scratch[g][lane+step];
  __syncthreads();
 }
 if(lane==0)for(int g=0;g<7;g++)out[(o*7+g)*tiles+tile]=scratch[g][0];
}
extern "C" __global__ void particle_reduce(const double* work,const int* offsets,
 const int* observations,const int* active,int count,int tiles,double* result){
 int particle=blockIdx.x/7,g=blockIdx.x%7,lane=threadIdx.x;
 double value=0.;int start=offsets[particle],length=(offsets[particle+1]-start)*tiles;
 if(active[particle])for(int i=lane;i<length;i+=256){
  int o=observations[start+i/tiles];value+=work[(o*7+g)*tiles+i%tiles];
 }
 __shared__ double scratch[256];scratch[lane]=value;__syncthreads();
 for(int step=128;step;step/=2){if(lane<step)scratch[lane]+=scratch[lane+step];__syncthreads();}
 if(lane==0)result[particle*7+g]=scratch[0];
}
extern "C" __global__ void local_normal(const ReferenceValue* map,const float2* data,
 const float* ctf,const double* inv,const double* deriv,const double* view,
 const double* shift,const double* precision,const int* frequencies,int nf,int b,int n,int p,double map_scale,double radius,double* out,const int* active_views){
 int tiles=(nf+127)/128,o=active_views[blockIdx.x/tiles],tile=blockIdx.x%tiles,lane=threadIdx.x;
 int f=tile*128+lane;double values[28]={0};
 if(f<nf)evaluate_pose<28>((long)o*nf+f,map,data,ctf,inv,deriv,view,shift,precision,frequencies,nf,b,n,p,map_scale,radius,values);
 __shared__ double scratch[28][128];
 for(int g=0;g<28;g++)scratch[g][lane]=values[g];
 __syncthreads();
 for(int step=64;step;step/=2){
  if(lane<step)for(int g=0;g<28;g++)scratch[g][lane]+=scratch[g][lane+step];
  __syncthreads();
 }
 if(lane==0)for(int g=0;g<28;g++)out[(o*28+g)*tiles+tile]=scratch[g][0];
}
extern "C" __global__ void normal_reduce(const double* work,const int* offsets,
 const int* observations,const int* active,int count,int tiles,double* result){
 int particle=blockIdx.x/28,g=blockIdx.x%28,lane=threadIdx.x;
 double value=0.;int start=offsets[particle],length=(offsets[particle+1]-start)*tiles;
 if(active[particle])for(int i=lane;i<length;i+=128){
  int o=observations[start+i/tiles];value+=work[(o*28+g)*tiles+i%tiles];
 }
 __shared__ double scratch[128];scratch[lane]=value;__syncthreads();
 for(int step=64;step;step/=2){if(lane<step)scratch[lane]+=scratch[lane+step];__syncthreads();}
 if(lane==0)result[particle*28+g]=scratch[0];
}
// Batched reduction: four warps reduce independently, then warp zero
// combines their sums. Geometry and likelihood arithmetic are unchanged.
extern "C" __global__ void local_normal_warp(const ReferenceValue* map,const float2* data,
 const float* ctf,const double* inv,const double* deriv,const double* view,
 const double* shift,const double* precision,const int* frequencies,int nf,int b,int n,int p,double map_scale,double radius,double* out,const int* active_views){
 int tiles=(nf+127)/128,o=active_views[blockIdx.x/tiles],tile=blockIdx.x%tiles,lane=threadIdx.x;
 int f=tile*128+lane;double values[28]={0};
 if(f<nf)evaluate_pose<28>((long)o*nf+f,map,data,ctf,inv,deriv,view,shift,precision,frequencies,nf,b,n,p,map_scale,radius,values);
 __shared__ double sums[28][4];
 for(int g=0;g<28;g++){
  double value=values[g];
  for(int step=16;step;step/=2)value+=__shfl_down_sync(0xffffffff,value,step);
  if((lane&31)==0)sums[g][lane/32]=value;
 }
 __syncthreads();
 if(lane<32)for(int g=0;g<28;g++){
  double value=lane<4?sums[g][lane]:0.;
  for(int step=16;step;step/=2)value+=__shfl_down_sync(0xffffffff,value,step);
  if(lane==0)out[(o*28+g)*tiles+tile]=value;
 }
}
'''


def pose_geometry(parameters,base_inverse,views,owner):
    # Same Rz(chi) Ry(theta) Rx(phi) and analytic derivatives as Greco,
    # without Python scalar mm3 loops for every particle/evaluation.
    p=np.asarray(parameters);s=np.sin(p[:,:3]);c=np.cos(p[:,:3]);count=len(p)
    rx=np.zeros((count,3,3));ry=np.zeros_like(rx);rz=np.zeros_like(rx)
    dx=np.zeros_like(rx);dy=np.zeros_like(rx);dz=np.zeros_like(rx)
    rx[:,0,0]=1;rx[:,1,1]=rx[:,2,2]=c[:,0];rx[:,1,2]=-s[:,0];rx[:,2,1]=s[:,0]
    ry[:,1,1]=1;ry[:,0,0]=ry[:,2,2]=c[:,1];ry[:,0,2]=s[:,1];ry[:,2,0]=-s[:,1]
    rz[:,2,2]=1;rz[:,0,0]=rz[:,1,1]=c[:,2];rz[:,0,1]=-s[:,2];rz[:,1,0]=s[:,2]
    dx[:,1,1]=dx[:,2,2]=-s[:,0];dx[:,1,2]=-c[:,0];dx[:,2,1]=c[:,0]
    dy[:,0,0]=dy[:,2,2]=-s[:,1];dy[:,0,2]=c[:,1];dy[:,2,0]=-c[:,1]
    dz[:,0,0]=dz[:,1,1]=-s[:,2];dz[:,0,1]=-c[:,2];dz[:,1,0]=c[:,2]
    z_y=rz@ry;rotation=z_y@rx;d=np.stack((z_y@dx,(rz@dy)@rx,(dz@ry)@rx),axis=1)
    inverse=np.einsum('bji,bjk->bik',rotation[owner],base_inverse)
    derivatives=np.einsum('bcji,bjk->bcik',d[owner],base_inverse)
    return inverse,derivatives,parameters[owner,3:]


def interpolate(map_,inverse,derivatives,n,radius,xp=np):
    """Independent array implementation used for CPU parity and gradient tests."""
    y=xp.arange(n);y=xp.where(y<=n//2,y,y-n);x=xp.arange(n//2+1)
    freq=xp.stack(xp.broadcast_arrays(x[None,:],y[:,None],xp.zeros((n,n//2+1))))
    q=2*xp.einsum('bij,jyx->biyx',xp.asarray(inverse),freq)
    sign=xp.where(q[:,0]<0,-1.,1.);q*=sign[:,None]
    lo=xp.floor(q).astype(xp.int64);f=q-lo;p=map_.shape[0];ph=map_.shape[-1]
    valid=(lo[:,0]>=0)&(lo[:,0]+1<ph)&(lo[:,1]+p//2>=0)&(lo[:,1]+p//2+1<p)&(lo[:,2]+p//2>=0)&(lo[:,2]+p//2+1<p)
    result=xp.zeros((len(inverse),n,n//2+1),xp.complex128);gradient=xp.zeros((len(inverse),3,n,n//2+1),xp.complex128)
    for z in (0,1):
        for yy in (0,1):
            for xx in (0,1):
                bits=(xx,yy,z);w=[f[:,j] if bits[j] else 1-f[:,j] for j in range(3)]
                v=map_[xp.clip(lo[:,2]+z+p//2,0,p-1),xp.clip(lo[:,1]+yy+p//2,0,p-1),xp.clip(lo[:,0]+xx,0,ph-1)]
                result+=w[0]*w[1]*w[2]*v
                for j in range(3):gradient[:,j]+=(1 if bits[j] else -1)*w[(j+1)%3]*w[(j+2)%3]*v
    result=result.real+1j*result.imag*sign
    gradient=gradient.real*sign[:,None]+1j*gradient.imag
    dq=2*xp.einsum('bcij,jyx->bciyx',xp.asarray(derivatives),freq)
    dresult=xp.einsum('biyx,bciyx->bcyx',gradient,dq)
    disk=(freq[0]**2+freq[1]**2<=radius**2)&(freq[0]**2+freq[1]**2>0)
    return result*valid*disk,dresult*valid[:,None]*disk


class LocalPoseObjective:
    def __init__(self,backend,fourier,data,ctf,precision,base_inverse,views,owner,angle_sigma,shift_sigma):
        self.backend=backend;self.xp=backend.xp;self.set_reference(fourier)
        self.data=self.xp.asarray(data,dtype=self.xp.complex64);self.ctf=self.xp.asarray(ctf,dtype=self.xp.float32)
        self.precision=self.xp.asarray(precision,dtype=self.xp.float64)
        self.base=np.asarray(base_inverse);self.views=np.asarray(views);self.owner=np.asarray(owner)
        self.count=int(self.owner.max())+1;self.prior=np.array([angle_sigma]*3+[shift_sigma]*3)**2
        self.shift_origin=None
        self.warp_normal_reduction=True
        if backend.gpu:
            xp=self.xp;n=backend.n
            y=np.arange(n);y=np.where(y<=n//2,y,y-n)
            r2=y[:,None]**2+np.arange(n//2+1)[None,:]**2
            indices=np.flatnonzero((r2<=backend.radius**2)&(r2>0)).astype(np.int32)
            self.frequencies=xp.asarray(indices);self.nf=len(indices)
            self.data=xp.ascontiguousarray(self.data.reshape(len(self.owner),-1)[:,self.frequencies])
            self.ctf=xp.ascontiguousarray(self.ctf.reshape(len(self.owner),-1)[:,self.frequencies])
            self.precision=xp.ascontiguousarray(self.precision.reshape(len(self.owner),-1)[:,self.frequencies])
            self.device_views=xp.asarray(self.views)
            self.tiles=(self.nf+255)//256
            self.work=xp.empty((len(self.owner),7,self.tiles),xp.float64)
            self.device_inverse=xp.empty((len(self.owner),3,3),xp.float64)
            self.device_derivatives=xp.empty((len(self.owner),3,3,3),xp.float64)
            self.device_shift=xp.empty((len(self.owner),3),xp.float64)
            self.host_inverse=np.empty((len(self.owner),3,3))
            self.host_derivatives=np.empty((len(self.owner),3,3,3))
            self.host_shift=np.empty((len(self.owner),3))
            self.kernel=xp.RawKernel(KERNEL,'local_pose',options=self.kernel_options)
            self.reduce_kernel=xp.RawKernel(KERNEL,'particle_reduce',options=self.kernel_options)
            self.device_active_views=xp.empty(len(self.owner),xp.int32)
            self.device_active=xp.empty(self.count,xp.int32)
            self.device_offsets=xp.asarray(np.r_[0,np.cumsum(np.bincount(self.owner,minlength=self.count))],dtype=xp.int32)
            self.device_observations=xp.asarray(np.argsort(self.owner,kind='stable'),dtype=xp.int32)
            self.device_result=xp.empty((self.count,7),xp.float64)
        self.evaluations=0

    def set_reference(self,reference):
        from .reference_storage import FourierReference
        half=isinstance(reference,FourierReference)
        if half and not self.backend.gpu:raise ValueError('Half Fourier references require a GPU.')
        if hasattr(self,'half_reference') and half!=self.half_reference:
            raise ValueError('Cannot change the reference storage type of an existing objective.')
        self.half_reference=half
        self.map=reference.data if half else reference
        self.map_scale=reference.scale if half else 1.
        self.reference_shape=reference.shape
        self.kernel_options=('--fmad=false','-DGRECO_HALF_REFERENCE') if half else ('--fmad=false',)

    def update_precision(self,precision):
        """Noise changes do not invalidate cached particle data or CTF."""
        value=self.xp.asarray(precision,dtype=self.xp.float64)
        if self.backend.gpu:
            value=value.reshape(len(self.owner),-1)[:,self.frequencies]
        if value.shape!=self.precision.shape:raise ValueError('Precision shape changed.')
        self.precision[...] = value

    def update_group_noise(self,noise,groups,shell):
        """Refresh cached likelihood weights directly at compact frequencies."""
        xp=self.xp;g=xp.asarray(groups)
        if self.backend.gpu:
            self.precision[...] = 1/noise[g][:,shell.ravel()[self.frequencies]]
        else:
            self.update_precision((self.precision>0)/noise[g][:,shell])

    def costs(self,parameters,active=None):
        """Trial-step acceptance requires only likelihood, not six derivatives."""
        if not self.backend.gpu:return self.per_particle(parameters,active)[:,0]
        xp=self.xp;active=np.ones(self.count,bool) if active is None else np.asarray(active,bool)
        if active.shape!=(self.count,):raise ValueError('Invalid active particle mask.')
        selected=np.flatnonzero(active[self.owner]).astype(np.int32)
        if not len(selected):return np.zeros(self.count)
        if not hasattr(self,'cost_work'):
            self.cost_work=xp.empty((len(self.owner),self.tiles),xp.float64)
            self.cost_result=xp.empty(self.count,xp.float64)
            self.cost_kernel=xp.RawKernel(KERNEL,'local_cost',options=self.kernel_options)
            self.cost_reduce=xp.RawKernel(KERNEL,'cost_reduce',options=self.kernel_options)
        from .native_lbfgs import pose_geometry_native
        inverse,_,shift=pose_geometry_native(parameters,self.base[selected],self.owner[selected])
        if self.shift_origin is not None:shift=shift+self.shift_origin[selected]
        self.host_inverse[selected]=inverse;self.host_shift[selected]=shift
        self.device_active_views[:len(selected)].set(selected);self.device_active.set(active.astype(np.int32))
        self.device_inverse.set(self.host_inverse);self.device_shift.set(self.host_shift)
        self.cost_kernel((len(selected)*self.tiles,),(256,),
            (self.map,self.data,self.ctf,self.device_inverse,self.device_derivatives,self.device_views,self.device_shift,self.precision,
             self.frequencies,np.int32(self.nf),np.int32(len(self.owner)),np.int32(self.backend.n),np.int32(self.reference_shape[0]),np.float64(self.map_scale),np.float64(self.backend.radius),self.cost_work,self.device_active_views))
        self.cost_reduce((self.count,),(256,),
            (self.cost_work,self.device_offsets,self.device_observations,self.device_active,np.int32(self.count),np.int32(self.tiles),self.cost_result))
        result=self.backend.host(self.cost_result)
        prior=np.where(np.isfinite(self.prior),1/self.prior,0.)
        result[active]+=.5*np.sum(parameters[active]**2*prior,axis=1)
        self.evaluations+=1
        return result

    def normal_equations(self,parameters,active=None):
        """Cost, gradient and real Gauss–Newton curvature, independently per particle."""
        xp=self.xp;backend=self.backend;n=backend.n
        active=np.ones(self.count,dtype=bool) if active is None else np.asarray(active,dtype=bool)
        if active.shape!=(self.count,):raise ValueError('Invalid active particle mask.')
        selected=np.flatnonzero(active[self.owner]).astype(np.int32)
        result=np.zeros((self.count,28))
        if not len(selected):return result[:,:7],np.zeros((self.count,6,6))
        from .native_lbfgs import pose_geometry_native
        inverse,d,shift=pose_geometry_native(parameters,self.base[selected],self.owner[selected])
        if self.shift_origin is not None:shift=shift+self.shift_origin[selected]
        if backend.gpu:
            if not hasattr(self,'normal_work'):
                self.normal_tiles=(self.nf+127)//128
                self.normal_work=xp.empty((len(self.owner),28,self.normal_tiles),xp.float64)
                self.normal_result=xp.empty((self.count,28),xp.float64)
                self.normal_kernel=xp.RawKernel(KERNEL,'local_normal_warp' if getattr(self,'warp_normal_reduction',False) else 'local_normal',options=self.kernel_options)
                self.normal_reduce=xp.RawKernel(KERNEL,'normal_reduce',options=self.kernel_options)
            self.device_active_views[:len(selected)].set(selected)
            self.device_active.set(active.astype(np.int32))
            self.host_inverse[selected]=inverse;self.host_derivatives[selected]=d;self.host_shift[selected]=shift
            self.device_inverse.set(self.host_inverse);self.device_derivatives.set(self.host_derivatives);self.device_shift.set(self.host_shift)
            self.normal_kernel((len(selected)*self.normal_tiles,),(128,),
                (self.map,self.data,self.ctf,self.device_inverse,self.device_derivatives,self.device_views,self.device_shift,self.precision,
                 self.frequencies,np.int32(self.nf),np.int32(len(self.owner)),np.int32(n),np.int32(self.reference_shape[0]),np.float64(self.map_scale),np.float64(backend.radius),self.normal_work,self.device_active_views))
            self.normal_reduce((self.count*28,),(128,),
                (self.normal_work,self.device_offsets,self.device_observations,self.device_active,np.int32(self.count),np.int32(self.normal_tiles),self.normal_result))
            result=backend.host(self.normal_result)
        else:
            pred,dp=interpolate(self.map,inverse,d,n,backend.radius)
            fy=np.fft.fftfreq(n)*n;fx=np.arange(n//2+1)
            freq=np.stack(np.broadcast_arrays(fx[None,:],fy[:,None],np.zeros((n,n//2+1))))
            views=self.views[selected];ctf=self.ctf[selected];precision=self.precision[selected]
            detector=np.einsum('bij,bj->bi',views,shift)
            phase=np.exp(-2j*np.pi/n*np.einsum('bi,iyx->byx',detector,freq))
            value=pred*ctf*phase;delta=value-self.data[selected]
            derivatives=np.concatenate((dp*ctf[:,None]*phase[:,None],-2j*np.pi/n*np.einsum('bij,iyx->bjyx',views,freq)*value[:,None]),axis=1)
            stats=np.empty((len(selected),28));stats[:,0]=.5*np.sum(abs(delta)**2*precision,axis=(1,2))
            stats[:,1:7]=np.sum((np.conj(delta)[:,None]*derivatives).real*precision[:,None],axis=(2,3))
            for i,(j,k) in enumerate(zip(*np.triu_indices(6))):
                stats[:,7+i]=np.sum((np.conj(derivatives[:,j])*derivatives[:,k]).real*precision,axis=(1,2))
            np.add.at(result,self.owner[selected],stats)
        curvature=np.zeros((self.count,6,6));j,k=np.triu_indices(6)
        curvature[:,j,k]=result[:,7:];curvature[:,k,j]=result[:,7:]
        prior=np.where(np.isfinite(self.prior),1/self.prior,0.)
        result[active,0]+=.5*np.sum(parameters[active]**2*prior,axis=1)
        result[active,1:7]+=parameters[active]*prior
        curvature[active]+=np.diag(prior)
        self.evaluations+=1
        return result[:,:7],curvature

    def per_particle(self,parameters,active=None):
        backend=self.backend;xp=self.xp;n=backend.n
        active=np.ones(self.count,dtype=bool) if active is None else np.asarray(active,dtype=bool)
        if active.shape!=(self.count,):raise ValueError('Invalid active particle mask.')
        selected=np.flatnonzero(active[self.owner]).astype(np.int32)
        if not len(selected):return np.zeros((self.count,7))
        from .native_lbfgs import pose_geometry_native
        inverse,d,shift=pose_geometry_native(parameters,self.base[selected],self.owner[selected])
        if self.shift_origin is not None:shift=shift+self.shift_origin[selected]
        if backend.gpu:
            out=self.work
            self.device_active_views[:len(selected)].set(selected)
            self.device_active.set(active.astype(np.int32))
            self.host_inverse[selected]=inverse;self.host_derivatives[selected]=d;self.host_shift[selected]=shift
            self.device_inverse.set(self.host_inverse)
            self.device_derivatives.set(self.host_derivatives)
            self.device_shift.set(self.host_shift)
            self.kernel((len(selected)*self.tiles,),(256,),
                (self.map,self.data,self.ctf,self.device_inverse,self.device_derivatives,self.device_views,self.device_shift,self.precision,
                 self.frequencies,np.int32(self.nf),np.int32(len(self.owner)),np.int32(n),np.int32(self.reference_shape[0]),np.float64(self.map_scale),np.float64(backend.radius),out,self.device_active_views))
            self.reduce_kernel((self.count*7,),(256,),(out,self.device_offsets,self.device_observations,self.device_active,np.int32(self.count),np.int32(self.tiles),self.device_result))
            result=backend.host(self.device_result)
        else:
            pred,dp=interpolate(self.map,inverse,d,n,backend.radius)
            y=np.arange(n);y=np.where(y<=n//2,y,y-n)
            freq=np.stack(np.broadcast_arrays(np.arange(n//2+1)[None,:],y[:,None],np.zeros((n,n//2+1))))
            views=self.views[selected];ctf=self.ctf[selected];precision=self.precision[selected]
            detector=np.einsum('bij,bj->bi',views,shift)
            phase=np.exp(-2j*np.pi/n*np.einsum('bi,iyx->byx',detector,freq))
            value=pred*ctf*phase;delta=value-self.data[selected]
            derivatives=dp*ctf[:,None]*phase[:,None]
            ds=-2j*np.pi/n*np.einsum('bij,iyx->bjyx',views,freq)*value[:,None]
            stats=np.empty((len(inverse),7));stats[:,0]=.5*np.sum(abs(delta)**2*precision,axis=(1,2))
            stats[:,1:]=np.sum((np.conj(delta)[:,None]*np.concatenate((derivatives,ds),axis=1)).real*precision[:,None],axis=(2,3))
            result=np.zeros((self.count,7));np.add.at(result,self.owner[selected],stats)
        finite_prior=np.isfinite(self.prior)
        if finite_prior.any():
            p=parameters[active][:,finite_prior]
            result[active,0]+=.5*np.sum(p**2/self.prior[finite_prior],axis=1)
            regularization=np.zeros((int(active.sum()),6))
            regularization[:,finite_prior]=p/self.prior[finite_prior]
            result[active,1:]+=regularization
        self.evaluations+=1
        return result

    def __call__(self,flat):
        stats=self.per_particle(np.asarray(flat).reshape(self.count,6))
        return float(stats[:,0].sum()),stats[:,1:].ravel()






def optimize_local_search(objective,maxiter=25,stability_rescue=True):
    """Greco's native More–Thuente L-BFGS, independently per particle.

    Fibers batch only GPU evaluations: each particle keeps its own step,
    history, relative gradient stop, and status. No SciPy in this path.
    """
    from .native_lbfgs import minimize_independent
    scales=np.array([.02]*3+[1.]*3);objective.prior[:]=np.inf
    before=objective.per_particle(np.zeros((objective.count,6)))[:,0]
    limits=getattr(objective,'step_limits',None)
    def physical(x):
        return x*scales if limits is None else limits*np.tanh(x*scales/limits)
    def evaluate(x,active):
        stats=objective.per_particle(physical(x),active)
        derivative=scales if limits is None else scales*(1-np.tanh(x*scales/limits)**2)
        return stats[:,0],stats[:,1:]*derivative
    x,diagnostic=minimize_independent(evaluate,objective.count,maxiter,epsilon=1e-5,xtol=1e-4,active_callback=True)
    parameters=physical(x)
    statuses=diagnostic['statuses']
    stationary=np.zeros(objective.count,bool);poll_evaluations=0
    if stability_rescue and (statuses==-1001).any():
        from .local_poll import refine_poll
        parameters,poll=refine_poll(objective,parameters,statuses==-1001,
                                   getattr(objective,'surface_radius_pixels',objective.backend.n/2))
        stationary=poll['certified'];poll_evaluations=poll['evaluations']
    after=objective.per_particle(parameters)[:,0]
    bad=(~np.isfinite(after))|(~np.isfinite(parameters).all(axis=1))|(after>before+1e-8)
    parameters[bad]=0;after[bad]=before[bad]
    stationary[bad]=False
    return parameters,dict(initial_cost=float(before.sum()),final_cost=float(after.sum()),
        evaluations=objective.evaluations,optimizer_iterations=int(diagnostic['iterations'].max()),
        optimizer_success=bool(((statuses>=0)|stationary).all()),rejected_particles=int(bad.sum()),
        optimizer_backend='native independent Greco libLBFGS / More-Thuente',
        optimizer_converged_particles=int((statuses>=0).sum()),
        optimizer_limited_particles=int((statuses==-997).sum()),
        optimizer_native_failed_particles=int(((statuses<0)&(statuses!=-997)).sum()),
        optimizer_failed_particles=int(((statuses<0)&(statuses!=-997)&~stationary).sum()),
        optimizer_stationary_particles=int(stationary.sum()),
        optimizer_neighborhood_bound_particles=0 if limits is None else int(np.any(abs(parameters)>=.95*limits,axis=1).sum()),
        optimizer_poll_evaluations=poll_evaluations,optimizer_poll_motion_pixels=.025,
        optimizer_statuses=statuses.tolist())
