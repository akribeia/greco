// FP32 normalization/maxima and FP64 ROI energy, with neutral tail lanes.
extern "C" __global__ void fused_corr_kernel(
    const float* ccc,const float* std,float* scores,float* angles,
    const float* angids,float* out_max,float* out_min,double* sq_part,
    const int* roi,int X,int Y,int Z,int b,long long row_stride,
    long long std_stride,int has_angles,int has_min,int has_sq,float fft_scale) {
    const long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long total=(long long)X*Y*Z;
    double energy=0.;
    if(q<total) {
        int coord[3]={(int)(q/((long long)Y*Z)),(int)((q/Z)%Y),(int)(q%Z)};
        bool inside=true;
        for(int a=0;a<3;a++) {
            const int* c=roi+4*a;int v=coord[a];
            inside &= (v>=c[0] && v<c[0]+c[1]) || (v>=c[2] && v<c[2]+c[3]);
        }
        float maximum=__int_as_float(0xff800000),minimum=__int_as_float(0x7f800000);
        int arg=0;double sum=0.;
        for(int i=0;i<b;i++) {
            float normalized=ccc[(long long)i*row_stride+q];
            if(fft_scale!=1.f) {
                float scaled;
                asm("mul.rn.f32 %0,%1,%2;" : "=f"(scaled) : "f"(normalized),"f"(fft_scale));
                normalized=scaled;
            }
            float v=normalized/std[(long long)i*std_stride+q];
            // Torch reductions propagate the first NaN and first equal max.
            if((v!=v && maximum==maximum) || (maximum==maximum && v>maximum)) {maximum=v;arg=i;}
            if((v!=v && minimum==minimum) || (minimum==minimum && v<minimum)) minimum=v;
            float square=v*v;sum+=(double)square;
        }
        if(maximum>scores[q]) {scores[q]=maximum;if(has_angles)angles[q]=angids[arg];}
        out_max[q]=maximum;if(has_min)out_min[q]=minimum;
        if(inside)energy=sum;
    }
    // Every lane participates, including the final partial voxel block.
    for(int off=16;off>0;off>>=1)energy+=__shfl_down_sync(0xffffffffu,energy,off);
    __shared__ double warps[8];
    if((threadIdx.x&31)==0)warps[threadIdx.x>>5]=energy;
    __syncthreads();
    if(threadIdx.x<32) {
        energy=threadIdx.x<8?warps[threadIdx.x]:0.;
        for(int off=16;off>0;off>>=1)energy+=__shfl_down_sync(0xffffffffu,energy,off);
        if(threadIdx.x==0 && has_sq)sq_part[blockIdx.x]=energy;
    }
}
