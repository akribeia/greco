"""Arbitrary-camera local WBP, projections and patch sampling on the GPU."""
import numpy as np

SOURCE=r'''
__device__ float bilinear(const float* a,int n,float x,float y){
 int ix=(int)floorf(x),iy=(int)floorf(y);if(ix<0||iy<0||ix+1>=n||iy+1>=n)return 0;
 float fx=x-ix,fy=y-iy;long p=(long)iy*n+ix;
 return (1-fy)*((1-fx)*a[p]+fx*a[p+1])+fy*((1-fx)*a[p+n]+fx*a[p+n+1]);
}
__device__ float tri(const float* a,int n,float x,float y,float z){
 int iz=(int)floorf(z);if(iz<0||iz+1>=n)return 0;
 float f=z-iz;return (1-f)*bilinear(a+(long)iz*n*n,n,x,y)+f*bilinear(a+(long)(iz+1)*n*n,n,x,y);
}
extern "C" __global__ void patches(const float* stack,const double* centers,int views,int image_n,int n,long count,float* output){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 long pair=i/((long)n*n);int t=pair%views,x=i%n,y=(i/n)%n;
 output[i]=bilinear(stack+(long)t*image_n*image_n,image_n,centers[2*pair]+x-(n-1)*.5,centers[2*pair+1]+y-(n-1)*.5);
}
extern "C" __global__ void local_wbp(const float* images,const unsigned char* valid,const double* cameras,const double* weights,int views,int n,int target,long count,float* output){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 long volume_size=(long)n*n*n;long b=i/volume_size;int x=i%n,y=(i/n)%n,z=(i/((long)n*n))%n;
 float center=(n-1)*.5f,xx=x-center,yy=y-center,zz=z-center,value=0,total=0;
 for(int t=0;t<views;t++)if(t!=target&&valid[b*views+t]){
  const double* c=cameras+9*t;float rx=c[0]*xx+c[1]*yy+c[2]*zz+center,ry=c[3]*xx+c[4]*yy+c[5]*zz+center;
  value+=weights[t]*bilinear(images+(b*views+t)*(long)n*n,n,rx,ry);total+=weights[t];
 }
 output[i]=total>0?value/total:0;
}
extern "C" __global__ void project(const float* volumes,const double* camera,int n,long count,float* output){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 long b=i/((long)n*n);float center=(n-1)*.5f,x=i%n-center,y=(i/n)%n-center,value=0;
 for(int z=0;z<n;z++){
  float rz=z-center,xx=camera[0]*x+camera[3]*y+camera[6]*rz+center;
  float yy=camera[1]*x+camera[4]*y+camera[7]*rz+center,zz=camera[2]*x+camera[5]*y+camera[8]*rz+center;
  value+=tri(volumes+b*(long)n*n*n,n,xx,yy,zz);
 }
 output[i]=value;
}
extern "C" __global__ void global_wbp(const float* images,const double* cameras,const double* translations,const double* weights,int views,int n,int z0,int depth,long count,float* output){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int x=i%n,y=(i/n)%n,z=z0+i/((long)n*n);float center=(n-1)*.5f,xx=x-center,yy=y-center,zz=z-center,value=0;
 for(int t=0;t<views;t++){
  const double* c=cameras+9*t;float rx=c[0]*xx+c[1]*yy+c[2]*zz+center+translations[2*t];
  float ry=c[3]*xx+c[4]*yy+c[5]*zz+center+translations[2*t+1];
  value+=weights[t]*bilinear(images+(long)t*n*n,n,rx,ry);
 }
 output[i]=value;
}
'''


class Kernels:
    def __init__(self,xp):
        self.xp=xp;self.module=xp.RawModule(code=SOURCE,name_expressions=('patches','local_wbp','project','global_wbp'))

    def launch(self,name,count,args):self.module.get_function(name)(((count+255)//256,),(256,),args)

    def patches(self,stack,centers,n):
        xp=self.xp;out=xp.empty((*centers.shape[:2],n,n),xp.float32)
        self.launch('patches',out.size,(stack,xp.ascontiguousarray(xp.asarray(centers,xp.float64)),np.int32(stack.shape[0]),np.int32(stack.shape[-1]),np.int32(n),np.int64(out.size),out));return out

    def local_wbp(self,images,valid,cameras,weights,target):
        xp=self.xp;n=images.shape[-1];out=xp.empty((len(images),n,n,n),xp.float32)
        self.launch('local_wbp',out.size,(xp.ascontiguousarray(images),xp.ascontiguousarray(xp.asarray(valid,xp.uint8)),xp.ascontiguousarray(xp.asarray(cameras,xp.float64)),xp.asarray(weights,xp.float64),np.int32(images.shape[1]),np.int32(n),np.int32(target),np.int64(out.size),out));return out

    def project(self,volumes,camera):
        xp=self.xp;n=volumes.shape[-1];out=xp.empty((len(volumes),n,n),xp.float32)
        self.launch('project',out.size,(xp.ascontiguousarray(volumes),xp.asarray(camera,xp.float64),np.int32(n),np.int64(out.size),out));return out

    def global_wbp(self,images,cameras,translations,weights,z0,depth):
        xp=self.xp;n=images.shape[-1];out=xp.empty((depth,n,n),xp.float32)
        self.launch('global_wbp',out.size,(images,xp.asarray(cameras,xp.float64),xp.asarray(translations,xp.float64),xp.asarray(weights,xp.float64),np.int32(len(images)),np.int32(n),np.int32(z0),np.int32(depth),np.int64(out.size),out));return out


def ramp_filter(images,cameras,xp):
    n=images.shape[-1];ky=xp.fft.fftfreq(n)[:,None];kx=xp.fft.rfftfreq(n)[None,:]
    axis=xp.asarray(cameras)[:, :2,1];norm=xp.maximum(xp.linalg.norm(axis,axis=1),1e-8);axis=axis/norm[:,None]
    ramp=xp.abs(axis[:,1,None,None]*kx-axis[:,0,None,None]*ky)
    f=xp.fft.rfft2(images,axes=(-2,-1));f*=ramp
    return xp.fft.irfft2(f,s=(n,n),axes=(-2,-1)).astype(xp.float32)


def angle_weights(angles):
    a=np.deg2rad(angles);w=np.empty(len(a));w[0]=abs(a[1]-a[0])/2;w[-1]=abs(a[-1]-a[-2])/2
    w[1:-1]=abs(a[2:]-a[:-2])/2
    return w/w.sum()
