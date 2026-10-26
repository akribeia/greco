"""GPU ports of Greco's trilinear Fourier projector/backprojector.

No orientation or translation search. CPU uses greco._numerics as the reference;
GPU uses the same centered half-complex grid and interpolation stencil.
Atomic float64 accumulation is not claimed bit-identical to serial CPU sums.
"""
import numpy as np

CUDA = r'''
extern "C" __global__ void project(const double2* map,const double* matrices,
 int b,int n,int p,double pf,double radius,double2* dst) {
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
 int h=n/2+1;long stride=(long)n*h;if(i>=b*stride)return;
 int obs=i/stride,x=i%h,y=(i/ h)%n;y=y<=n/2?y:y-n;
 double2 zero=make_double2(0,0);dst[i]=zero;
 if(x*x+y*y>radius*radius)return;
 const double* m=matrices+obs*9;
 double xx=pf*(m[0]*x+m[1]*y),yy=pf*(m[3]*x+m[4]*y),zz=pf*(m[6]*x+m[7]*y);
 bool cj=xx<0;if(cj){xx=-xx;yy=-yy;zz=-zz;}
 int ix=(int)floor(xx),iy=(int)floor(yy)+p/2,iz=(int)floor(zz)+p/2;
 int ph=p/2+1;if(ix<0||ix+1>=ph||iy<0||iy+1>=p||iz<0||iz+1>=p)return;
 double fx=xx-floor(xx),fy=yy-floor(yy),fz=zz-floor(zz);
 double re=0,im=0;
 for(int z=0;z<2;z++)for(int y0=0;y0<2;y0++)for(int x0=0;x0<2;x0++){
  double w=(z?fz:1-fz)*(y0?fy:1-fy)*(x0?fx:1-fx);
  double2 v=map[((long)(iz+z)*p+iy+y0)*ph+ix+x0];re+=w*v.x;im+=w*v.y;
 }
 dst[i]=make_double2(re,cj?-im:im);
}
extern "C" __global__ void backproject(const double2* src,const double* wt,
 const double* matrices,int b,int n,int p,double pf,double radius,
 double* re,double* im,double* weight){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
 int h=n/2+1;long stride=(long)n*h;if(i>=b*stride)return;
 int obs=i/stride,x=i%h,row=(i/h)%n,y=row<=n/2?row:row-n;
 if((row>n/2&&x==0)||x*x+y*y>radius*radius||wt[i]<=0)return;
 const double* m=matrices+obs*9;
 double xx=pf*(m[0]*x+m[1]*y),yy=pf*(m[3]*x+m[4]*y),zz=pf*(m[6]*x+m[7]*y);
 bool cj=xx<0;if(cj){xx=-xx;yy=-yy;zz=-zz;}
 int ix=(int)floor(xx),iy=(int)floor(yy)+p/2,iz=(int)floor(zz)+p/2;
 int ph=p/2+1;if(ix<0||ix+1>=ph||iy<0||iy+1>=p||iz<0||iz+1>=p)return;
 double fx=xx-floor(xx),fy=yy-floor(yy),fz=zz-floor(zz);double2 v=src[i];
 for(int z=0;z<2;z++)for(int y0=0;y0<2;y0++)for(int x0=0;x0<2;x0++){
  double w=(z?fz:1-fz)*(y0?fy:1-fy)*(x0?fx:1-fx);
  long j=((long)(iz+z)*p+iy+y0)*ph+ix+x0;
  atomicAdd(re+j,w*v.x);atomicAdd(im+j,w*(cj?-v.y:v.y));atomicAdd(weight+j,w*wt[i]);
 }
}
'''


class FixedPoseBackend:
    def __init__(self, size, gpu=True, radius=None):
        from greco._numerics.core.projector import Projector
        from greco._numerics.core.backprojector import BackProjector
        self.n=size;self.gpu=gpu;self.radius=min(size//2,float(radius or size//2))
        self.bp_type=BackProjector
        self.projector=Projector(size,padding_factor=2.)
        template=BackProjector(size,padding_factor=2.)
        template.init_zeros()
        self.p=template.pad_size;self.shape=template.data.shape
        if gpu:
            import cupy as cp
            self.xp=cp
            self.project_kernel=cp.RawKernel(CUDA,'project',options=('--fmad=false',))
            self.bp_kernel=cp.RawKernel(CUDA,'backproject',options=('--fmad=false',))
        else:self.xp=np

    def prepare_map(self, volume):
        if self.gpu:
            from .gpu_maps import prepare
            return prepare(self,volume)
        # The validated Greco gridding pre-correction, padding and FFT scaling.
        self.projector.compute_fourier_transform_map(np.array(volume,dtype=np.float64,copy=True))
        return self.xp.asarray(self.projector.data)

    def crop_map(self,fourier_map,radius):
        """Exact centered Fourier subgrid; original box, frequency and phase units.

        Keep one whole original-frequency shell outside the search sphere so
        every trilinear interpolation stencil fits. No map resampling/scaling.
        """
        p=min(fourier_map.shape[0],4*int(np.ceil(radius))+5)
        if p==fourier_map.shape[0]:return fourier_map
        mid=fourier_map.shape[0]//2;half=p//2
        return self.xp.ascontiguousarray(fourier_map[mid-half:mid+half+1,mid-half:mid+half+1,:half+1])

    def project(self, fourier_map, rotations, inverse_rotations=None):
        from greco._numerics.core.projector import _inv3
        if not self.gpu:
            self.projector.data=np.asarray(fourier_map)
            self.projector.pad_size=fourier_map.shape[0]
            self.projector._data_flat=self.projector.data.ravel()
            ph=fourier_map.shape[2];span=fourier_map.shape[1]*ph
            self.projector._off8=np.array([0,1,ph,ph+1,span,span+1,span+ph,span+ph+1],np.int64)
            # Use the core 2D projection interface, rather than the optional
            # allocation wrapper (not present in every greco._numerics revision).
            result=np.zeros((len(rotations),self.n,self.n//2+1),np.complex128)
            for i,r in enumerate(rotations):self.projector.project(result[i],r)
            y=np.where(np.arange(self.n)<=self.n//2,np.arange(self.n),np.arange(self.n)-self.n)
            mask=y[:,None]**2+np.arange(self.n//2+1)[None,:]**2<=self.radius**2
            return result*mask
        cp=self.xp;out=cp.empty((len(rotations),self.n,self.n//2+1),cp.complex128)
        matrices=inverse_rotations if inverse_rotations is not None else np.asarray([_inv3(r) for r in rotations],np.float64)
        self.project_kernel(((out.size+255)//256,),(256,),
            (fourier_map,cp.ascontiguousarray(cp.asarray(matrices,dtype=cp.float64)),np.int32(len(rotations)),np.int32(self.n),
             np.int32(fourier_map.shape[0]),np.float64(2.),np.float64(self.radius),out))
        return out

    def zeros(self):
        return [self.xp.zeros(self.shape,self.xp.float64) for _ in range(3)]

    def accumulate(self, accumulator, data, weight, rotations, inverse_rotations=None):
        if not self.gpu:
            bp=self.bp_type(self.n,padding_factor=2.);bp.init_zeros()
            bp.data=accumulator[0]+1j*accumulator[1];bp.weight=accumulator[2]
            y=np.where(np.arange(self.n)<=self.n//2,np.arange(self.n),np.arange(self.n)-self.n)
            mask=y[:,None]**2+np.arange(self.n//2+1)[None,:]**2<=self.radius**2
            for d,w,r in zip(data,weight,rotations):bp.backproject2Dto3D(d*mask,r,w*mask)
            accumulator[0][...]=bp.data.real;accumulator[1][...]=bp.data.imag
            return
        from greco._numerics.core.projector import _inv3
        cp=self.xp;matrices=cp.ascontiguousarray(cp.asarray(inverse_rotations if inverse_rotations is not None else np.asarray([_inv3(r) for r in rotations],np.float64),dtype=cp.float64))
        src=cp.ascontiguousarray(cp.asarray(data,dtype=cp.complex128));wt=cp.ascontiguousarray(cp.asarray(weight,dtype=cp.float64))
        self.bp_kernel(((src.size+255)//256,),(256,),
            (src,wt,matrices,np.int32(len(rotations)),np.int32(self.n),np.int32(self.p),
             np.float64(2.),np.float64(self.radius),*accumulator))

    def host(self, value):
        return self.xp.asnumpy(value) if self.gpu else np.asarray(value)

    def reconstruct(self, accumulator, tau2=None, regularization=4.,device=False):
        if self.gpu:
            from .gpu_maps import reconstruct
            return reconstruct(self,accumulator,tau2,regularization,device=device)
        bp=self.bp_type(self.n,padding_factor=2.);bp.init_zeros()
        bp.data=self.host(accumulator[0])+1j*self.host(accumulator[1])
        bp.weight=self.host(accumulator[2])
        output=np.zeros((self.n,)*3,np.float64)
        if not np.any(bp.weight>0):return output
        # Retain the tested Greco MAP reconstruction on CPU; class maps are
        # small, whereas particle/view likelihoods and accumulation dominate.
        bp.reconstruct(output,do_map=tau2 is not None,tau2=tau2,
                       tau2_fudge=regularization,minres_map=5)
        return output


def posterior(costs, priors, xp=np):
    """Stable fixed-pose Gaussian mixture E step; do not truncate weak classes."""
    logits=-xp.asarray(costs,dtype=xp.float64)+xp.log(xp.asarray(priors,dtype=xp.float64))
    peak=logits.max(axis=1,keepdims=True)
    exp=xp.exp(logits-peak);den=exp.sum(axis=1,keepdims=True)
    return exp/den,(peak[:,0]+xp.log(den[:,0]))
