"""Greco discrete Ram-Lak WBP extended to full rigid cameras for export only."""
import numpy as np


def center_projections(images,xp):
    """Remove each projection's arbitrary DC before zero-padded WBP.

    Padding an offset intensity image with zero creates a synthetic step at
    its detector boundary. Ram-Lak turns that step into a bright rim. Keep
    the measured spatial variations, without fitting a plane or a background
    surface, and do not change the input stack used for registration.
    """
    images=xp.asarray(images,dtype=xp.float32)
    if images.ndim!=3:raise ValueError('WBP expects a tilt/Y/X projection stack.')
    means=images.mean(axis=(-2,-1),keepdims=True,dtype=xp.float64)
    return images-means.astype(xp.float32)


def ramp_filter(images,cameras,xp):
    images=center_projections(images,xp)
    n=images.shape[-1];p=max(64,2**int(np.ceil(np.log2(2*n))))
    odd=np.r_[np.arange(1,p//2+1,2),np.arange(p//2-1,0,-2)]
    kernel=np.zeros(p);kernel[0]=.25;kernel[1::2]=-1/(np.pi*odd)**2
    ramp=xp.asarray((2*np.fft.fft(kernel).real).astype(np.float32)[:p//2+1])
    grid=xp.arange(p//2+1)/p
    ky=xp.fft.fftfreq(p)[:,None];kx=xp.fft.rfftfreq(p)[None,:]
    axis=xp.asarray(cameras)[:,:2,1].copy();axis/=xp.maximum(xp.linalg.norm(axis,axis=1,keepdims=True),1e-8)
    padded=xp.pad(images,((0,0),(0,p-n),(0,p-n)))
    ft=xp.fft.rfft2(padded,axes=(-2,-1))
    for t in range(len(images)):
        frequency=xp.abs(axis[t,1]*kx-axis[t,0]*ky)
        # At a vertical axis this is exactly Greco's 1D discrete filter.
        ft[t]*=xp.interp(frequency,grid,ramp)+2*xp.maximum(frequency-.5,0)
    return xp.fft.irfft2(ft,s=(p,p),axes=(-2,-1))[:,:n,:n].astype(xp.float32)


SOURCE=r'''
extern "C" __global__ void main_camera_wbp(const float* images,const double* cameras,const double* offsets,int views,int n,int z0,int depth,long count,float* output){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int x=i%n,y=(i/n)%n,z=z0+i/((long)n*n);float center=n/2;
 float xx=x-center,yy=y-center,zz=z-center,value=0;
 for(int t=0;t<views;t++){
  const double* c=cameras+9*t;
  float rx=c[0]*xx+c[1]*yy+c[2]*zz+center+offsets[2*t];
  float ry=c[3]*xx+c[4]*yy+c[5]*zz+center+offsets[2*t+1];
  // R_axis^T R_axis leaves roundoff (~1e-15) at exactly valid borders.
  if(rx<-1e-5f||ry<-1e-5f||rx>n-1+1e-5f||ry>n-1+1e-5f)continue;
  rx=fminf(n-1,fmaxf(0,rx));ry=fminf(n-1,fmaxf(0,ry));
  int ix=(int)floorf(rx),iy=(int)floorf(ry),jx=min(ix+1,n-1),jy=min(iy+1,n-1);
  float fx=rx-ix,fy=ry-iy;const float* a=images+(long)t*n*n;
  value+=(1-fy)*((1-fx)*a[(long)iy*n+ix]+fx*a[(long)iy*n+jx])+fy*((1-fx)*a[(long)jy*n+ix]+fx*a[(long)jy*n+jx]);
 }
 output[i]=value*1.5707963267948966f/views;
}
'''


class RigidCameraWBP:
    def __init__(self,filtered,cameras,reference_cameras,xp):
        self.xp=xp;self.images=xp.ascontiguousarray(filtered);self.cameras=xp.asarray(cameras,xp.float64)
        n=filtered.shape[-1]
        # Preserve fitted camera perturbations about the landmark (n-1)/2
        # centre while reproducing the established Greco n//2 reference grid.
        delta=n//2-(n-1)/2
        offsets=np.einsum('tij,j->ti',np.asarray(cameras)-reference_cameras,np.full(3,delta))[:,:2]
        self.offsets=xp.asarray(offsets,xp.float64)
        self.kernel=xp.RawKernel(SOURCE,'main_camera_wbp')

    def slab(self,z0,depth):
        xp=self.xp;n=self.images.shape[-1];out=xp.empty((depth,n,n),xp.float32)
        self.kernel(((out.size+255)//256,),(256,),
            (self.images,self.cameras,self.offsets,np.int32(len(self.images)),np.int32(n),
             np.int32(z0),np.int32(depth),np.int64(out.size),out))
        return out
