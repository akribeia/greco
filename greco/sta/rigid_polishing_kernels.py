"""Fused exact frame rotations/derivatives; avoid dozens of tiny CUDA launches."""
_kernel=None


def rotations_and_derivatives(angles,xp):
    global _kernel
    if _kernel is None:
        _kernel=xp.RawKernel(r'''
        __device__ void product(const double* a,const double* b,double* dst){
          for(int i=0;i<3;i++)for(int j=0;j<3;j++){
            double v=0;for(int k=0;k<3;k++)v+=a[3*i+k]*b[3*k+j];dst[3*i+j]=v;
          }
        }
        extern "C" __global__ void frame_rotations(const double* angles,
            double* q,double* dq,int count){
          int frame=blockIdx.x*blockDim.x+threadIdx.x;if(frame>=count)return;
          double rx[9]={1,0,0,0,1,0,0,0,1};
          double ry[9]={1,0,0,0,1,0,0,0,1};
          double rz[9]={1,0,0,0,1,0,0,0,1};
          double derivatives[27]={0};
          double *matrices[3]={rx,ry,rz};
          const int first[3]={1,2,0},second[3]={2,0,1};
          for(int axis=0;axis<3;axis++){
            double s,c;sincos(angles[3*frame+axis],&s,&c);
            int i=first[axis],j=second[axis];double* a=matrices[axis];double* d=derivatives+9*axis;
            a[3*i+i]=a[3*j+j]=c;a[3*i+j]=-s;a[3*j+i]=s;
            d[3*i+i]=d[3*j+j]=-s;d[3*i+j]=-c;d[3*j+i]=c;
          }
          double intermediate[9],value[9];
          product(rz,ry,intermediate);product(intermediate,rx,value);
          for(int i=0;i<9;i++)q[9*frame+i]=value[i];
          product(intermediate,derivatives,value);
          for(int i=0;i<9;i++)dq[27*frame+i]=value[i];
          product(rz,derivatives+9,intermediate);product(intermediate,rx,value);
          for(int i=0;i<9;i++)dq[27*frame+9+i]=value[i];
          product(derivatives+18,ry,intermediate);product(intermediate,rx,value);
          for(int i=0;i<9;i++)dq[27*frame+18+i]=value[i];
        }''','frame_rotations',options=('--fmad=false',))
    angles=xp.ascontiguousarray(xp.asarray(angles,dtype=xp.float64))
    q=xp.empty((len(angles),3,3),dtype=xp.float64)
    dq=xp.empty((len(angles),3,3,3),dtype=xp.float64)
    _kernel(((len(angles)+63)//64,),(64,),(angles,q,dq,len(angles)))
    return q,dq
