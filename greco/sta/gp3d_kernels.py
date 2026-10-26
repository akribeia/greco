"""Fused double-precision Keys interpolation: one thread per observation."""
_kernel=None


def keys_sample(surfaces,xy,xp):
    global _kernel
    if _kernel is None:
        _kernel=xp.RawKernel(r'''
        extern "C" __global__ void sample(const double* a,const double* xy,
            double* value,double* grad,const int count,const int h,const int w){
          int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=count)return;
          double x=xy[2*i],y=xy[2*i+1];int bx=(int)floor(x),by=(int)floor(y);
          double t=x-bx,u=y-by,wx[4],wy[4],dx[4],dy[4];
          wx[0]=-.5*t+t*t-.5*t*t*t;wx[1]=1-2.5*t*t+1.5*t*t*t;
          wx[2]=.5*t+2*t*t-1.5*t*t*t;wx[3]=-.5*t*t+.5*t*t*t;
          wy[0]=-.5*u+u*u-.5*u*u*u;wy[1]=1-2.5*u*u+1.5*u*u*u;
          wy[2]=.5*u+2*u*u-1.5*u*u*u;wy[3]=-.5*u*u+.5*u*u*u;
          dx[0]=-.5+2*t-1.5*t*t;dx[1]=-5*t+4.5*t*t;
          dx[2]=.5+4*t-4.5*t*t;dx[3]=-t+1.5*t*t;
          dy[0]=-.5+2*u-1.5*u*u;dy[1]=-5*u+4.5*u*u;
          dy[2]=.5+4*u-4.5*u*u;dy[3]=-u+1.5*u*u;
          double v=0,gx=0,gy=0;
          #pragma unroll
          for(int yy=0;yy<4;yy++){
            int iy=min(max(by+yy-1,0),h-1);
            #pragma unroll
            for(int xx=0;xx<4;xx++){
              int ix=min(max(bx+xx-1,0),w-1);
              double s=a[((long long)i*h+iy)*w+ix];
              v+=s*wx[xx]*wy[yy];gx+=s*dx[xx]*wy[yy];gy+=s*wx[xx]*dy[yy];
            }
          }
          value[i]=v;grad[2*i]=gx;grad[2*i+1]=gy;
        }''','sample')
    surfaces=xp.ascontiguousarray(surfaces,dtype=xp.float64)
    xy=xp.ascontiguousarray(xy,dtype=xp.float64)
    value=xp.empty(len(xy),dtype=xp.float64);gradient=xp.empty((len(xy),2),dtype=xp.float64)
    _kernel(((len(xy)+127)//128,),(128,),(surfaces,xy,value,gradient,
            len(xy),surfaces.shape[1],surfaces.shape[2]))
    return value,gradient
