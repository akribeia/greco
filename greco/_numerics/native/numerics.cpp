// SPDX-License-Identifier: GPL-2.0-or-later
// RELION-derived numerical routines.
// Upstream source: RELION src/fftw.cpp, RELION src/postprocessing.cpp
//
// Original upstream notices (retained verbatim as comments):
// /***************************************************************************
//  *
//  * Author: "Sjors H.W. Scheres"
//  * MRC Laboratory of Molecular Biology
//  *
//  * This program is free software; you can redistribute it and/or modify
//  * it under the terms of the GNU General Public License as published by
//  * the Free Software Foundation; either version 2 of the License, or
//  * (at your option) any later version.
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU General Public License for more details.
//  *
//  * This complete copyright notice must be included in any revised version of the
//  * source code. Additional authorship citations may be added, but existing
//  * author citations must be preserved.
//  ***************************************************************************/
//
// /***************************************************************************
//  *
//  * Authors:    Roberto Marabini					(roberto@cnb.csic.es)
//  *			   Carlos Oscar S. Sorzano			(coss@cnb.csic.es)
//  *
//  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
//  *
//  * This program is free software; you can redistribute it and/or modify
//  * it under the terms of the GNU General Public License as published by
//  * the Free Software Foundation; either version 2 of the License, or
//  * (at your option) any later version.
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU General Public License for more details.
//  *
//  * You should have received a copy of the GNU General Public License
//  * along with this program; if not, write to the Free Software
//  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
//  * 02111-1307  USA
//  *
//  *	All comments concerning this program package may be sent to the
//  *	e-mail address 'xmipp@cnb.csic.es'
//  ***************************************************************************/
//
// Greco port and adaptations:
// Changes: Greco FFTW/glibc infrastructure; retained RELION-derived FSC and Guinier reductions.
// Modified 2026-10-23: restored source attribution and per-file notices.
// See licenses/relion/NOTICE.txt, licenses/relion/COPYING.txt,
// and licenses/THIRD_PARTY_NOTICES.md.
//

// Greco's minimal CPU numerical kernels. No external application headers or libraries.
// FFTW/glibc implement the transforms and elementary functions. The retained
// normalization and FSC/Guinier conventions derive from the numerical port;
// see licenses/relion/COPYING.txt and licenses/THIRD_PARTY_NOTICES.md.
#include <fftw3.h>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <new>
#include <vector>

#define API extern "C" __attribute__((visibility("default")))
namespace {
std::mutex planner_mutex;
enum class Normalization { raw, forward, both };
struct Grid {
    size_t real, complex;
    int dims[3], rank;
    Grid(int w, int h, int d, int fixed_rank = 0) : rank(0) {
        if (w <= 0 || h <= 0 || d <= 0) throw std::invalid_argument("invalid FFT dimensions");
        size_t limit = std::numeric_limits<size_t>::max() / (2 * sizeof(double));
        real = static_cast<size_t>(w);
        if (real > limit / h) throw std::bad_alloc();
        real *= h;
        if (real > limit / d) throw std::bad_alloc();
        real *= d;
        complex = static_cast<size_t>(w / 2 + 1) * h * d;
        if (fixed_rank == 3 || d > 1) dims[rank++] = d;
        if (fixed_rank >= 2 || h > 1) dims[rank++] = h;
        dims[rank++] = w;
    }
};
template<class T> struct FFT;
template<> struct FFT<double> {
    using Plan = fftw_plan; using Complex = fftw_complex;
    static Plan forward(const Grid& g, double* r, double* c) { return fftw_plan_dft_r2c(g.rank, g.dims, r, reinterpret_cast<Complex*>(c), FFTW_ESTIMATE); }
    static Plan inverse(const Grid& g, double* c, double* r) { return fftw_plan_dft_c2r(g.rank, g.dims, reinterpret_cast<Complex*>(c), r, FFTW_ESTIMATE); }
    static void run_forward(Plan p, double* r, double* c) { fftw_execute_dft_r2c(p, r, reinterpret_cast<Complex*>(c)); }
    static void run_inverse(Plan p, double* c, double* r) { fftw_execute_dft_c2r(p, reinterpret_cast<Complex*>(c), r); }
    static void destroy(Plan p) { fftw_destroy_plan(p); }
};
template<> struct FFT<float> {
    using Plan = fftwf_plan; using Complex = fftwf_complex;
    static Plan forward(const Grid& g, float* r, float* c) { return fftwf_plan_dft_r2c(g.rank, g.dims, r, reinterpret_cast<Complex*>(c), FFTW_ESTIMATE); }
    static Plan inverse(const Grid& g, float* c, float* r) { return fftwf_plan_dft_c2r(g.rank, g.dims, reinterpret_cast<Complex*>(c), r, FFTW_ESTIMATE); }
    static void run_forward(Plan p, float* r, float* c) { fftwf_execute_dft_r2c(p, r, reinterpret_cast<Complex*>(c)); }
    static void run_inverse(Plan p, float* c, float* r) { fftwf_execute_dft_c2r(p, reinterpret_cast<Complex*>(c), r); }
    static void destroy(Plan p) { fftwf_destroy_plan(p); }
};
template<class T> struct Plans {
    using F = FFT<T>;
    typename F::Plan forward = nullptr, inverse = nullptr;
    Plans(const Grid& g, T* real, T* spectrum, bool pair, bool backwards) {
        std::lock_guard<std::mutex> lock(planner_mutex);
        if (pair || !backwards) forward = F::forward(g, real, spectrum);
        if (pair || backwards) inverse = F::inverse(g, spectrum, real);
        if ((pair && (!forward || !inverse)) || (!pair && !(backwards ? inverse : forward))) {
            if (forward) F::destroy(forward);
            if (inverse) F::destroy(inverse);
            throw std::bad_alloc();
        }
    }
    ~Plans() {
        std::lock_guard<std::mutex> lock(planner_mutex);
        if (forward) F::destroy(forward);
        if (inverse) F::destroy(inverse);
    }
    Plans(const Plans&) = delete;
    Plans& operator=(const Plans&) = delete;
};
// Explicit complex-by-real arithmetic preserves the established rounding and
// signed-zero conventions. Compiler contraction/fast-math are disabled.
template<class T> void normalize_complex(T* data, size_t n, T scale) {
    const T zero = T(0), denominator = scale * scale + zero * zero;
    for (size_t i = 0; i < n; ++i) {
        const T re = data[2*i], im = data[2*i+1];
        data[2*i] = (re * scale + im * zero) / denominator;
        data[2*i+1] = (im * scale - re * zero) / denominator;
    }
}
template<class T> int normalized_fft(const T* input, T* output, int w, int h, int d, bool backwards, Normalization norm) noexcept {
    try {
        if (!input || !output) return -1;
        Grid g(w,h,d);
        // Preserve allocation order/alignment of the existing normalized path.
        std::unique_ptr<T[]> real, spectrum;
        if (backwards) { spectrum.reset(new T[2*g.complex]); real.reset(new T[g.real]); }
        else { real.reset(new T[g.real]); spectrum.reset(new T[2*g.complex]); }
        if (backwards) std::memcpy(spectrum.get(), input, 2*g.complex*sizeof(T));
        else std::memcpy(real.get(), input, g.real*sizeof(T));
        Plans<T> plans(g,real.get(),spectrum.get(),true,backwards);
        if (backwards) {
            FFT<T>::run_inverse(plans.inverse,spectrum.get(),real.get());
            if (norm == Normalization::both) {
                const T scale = T(std::sqrt(double(g.real)));
                for (size_t i=0;i<g.real;++i) real[i] /= scale;
            }
            std::memcpy(output,real.get(),g.real*sizeof(T));
        } else {
            FFT<T>::run_forward(plans.forward,real.get(),spectrum.get());
            if (norm != Normalization::raw) normalize_complex(spectrum.get(),g.complex,norm==Normalization::both ? T(std::sqrt(double(g.real))) : T(g.real));
            std::memcpy(output,spectrum.get(),2*g.complex*sizeof(T));
        }
        return 0;
    } catch (const std::invalid_argument&) { return -1; }
      catch (...) { return -2; }
}
struct FFTWBuffer {
    void* value;
    explicit FFTWBuffer(size_t n) : value(fftw_malloc(n)) { if (!value) throw std::bad_alloc(); }
    ~FFTWBuffer() { fftw_free(value); }
    FFTWBuffer(const FFTWBuffer&) = delete;
};
int raw_double_fft(const double* input,double* output,int w,int h,int d,int rank,bool backwards,bool aligned=false) noexcept {
    try {
        if (!input || !output) return -1;
        Grid g(w,h,d,rank);
        if (!backwards) {
            Plans<double> plans(g,const_cast<double*>(input),output,false,false);
            FFT<double>::run_forward(plans.forward,const_cast<double*>(input),output);
        } else {
            FFTWBuffer scratch(2*g.complex*sizeof(double));
            double* spectrum=static_cast<double*>(scratch.value);
            std::memcpy(spectrum,input,2*g.complex*sizeof(double));
            if (aligned) {
                FFTWBuffer result(g.real*sizeof(double));
                double* real=static_cast<double*>(result.value);
                Plans<double> plans(g,real,spectrum,false,true);
                FFT<double>::run_inverse(plans.inverse,spectrum,real);
                std::memcpy(output,real,g.real*sizeof(double));
            } else {
                Plans<double> plans(g,output,spectrum,false,true);
                FFT<double>::run_inverse(plans.inverse,spectrum,output);
            }
        }
        return 0;
    } catch (const std::invalid_argument&) { return -1; }
      catch (...) { return -2; }
}
}
namespace {
// Aligned private buffers preserve the previous worker's FFTW alignment while
// avoiding process creation and protecting caller-owned inverse input.
template<class T> int aligned_raw_fft(const void* input,void* output,int w,int h,int d,int rank,bool backwards) noexcept {
    try {
        if (!input || !output || (rank!=2 && rank!=3) || (rank==2 && d!=1)) return -1;
        Grid g(w,h,d,rank);
        FFTWBuffer r(g.real*sizeof(T)),c(2*g.complex*sizeof(T));
        T* real=static_cast<T*>(r.value); T* spectrum=static_cast<T*>(c.value);
        std::memcpy(backwards?spectrum:real,input,(backwards?2*g.complex:g.real)*sizeof(T));
        Plans<T> plans(g,real,spectrum,false,backwards);
        if (backwards) FFT<T>::run_inverse(plans.inverse,spectrum,real);
        else FFT<T>::run_forward(plans.forward,real,spectrum);
        std::memcpy(output,backwards?real:spectrum,(backwards?g.real:2*g.complex)*sizeof(T));
        return 0;
    } catch (const std::invalid_argument&) { return -1; }
      catch (...) { return -2; }
}
}
API int greco_fft_raw(const void* in,void* out,int w,int h,int d,int rank,int single,int inverse) {
    if ((single!=0 && single!=1) || (inverse!=0 && inverse!=1)) return -1;
    return single ? aligned_raw_fft<float>(in,out,w,h,d,rank,inverse) : aligned_raw_fft<double>(in,out,w,h,d,rank,inverse);
}
API int greco_numerics_version() { return 2; }
API const char* greco_fftw_version() { return fftw_version; }
#define NORMALIZED(SUFFIX,TYPE) \
API int r2c_both_##SUFFIX(const TYPE* in,TYPE* out,int w,int h,int d) { return normalized_fft(in,out,w,h,d,false,Normalization::both); } \
API int c2r_both_##SUFFIX(const TYPE* in,TYPE* out,int w,int h,int d) { return normalized_fft(in,out,w,h,d,true,Normalization::both); } \
API int r2c_fwdonly_##SUFFIX(const TYPE* in,TYPE* out,int w,int h,int d) { return normalized_fft(in,out,w,h,d,false,Normalization::forward); } \
API int c2r_fwdonly_##SUFFIX(const TYPE* in,TYPE* out,int w,int h,int d) { return normalized_fft(in,out,w,h,d,true,Normalization::forward); }
NORMALIZED(f32,float)
NORMALIZED(f64,double)
API int r2c_none_2d_f32(const float* in,float* out,int w,int h) { return normalized_fft(in,out,w,h,1,false,Normalization::raw); }
API int c2r_none_2d_f32(const float* in,float* out,int w,int h) { return normalized_fft(in,out,w,h,1,true,Normalization::raw); }
API int r2c_raw_2d_f64(const double* in,double* out,int w,int h) { return raw_double_fft(in,out,w,h,1,2,false); }
API int c2r_raw_2d_f64(const double* in,double* out,int w,int h) { return raw_double_fft(in,out,w,h,1,2,true); }
API int c2r_raw_2d_f64_aligned(const double* in,double* out,int w,int h) { return raw_double_fft(in,out,w,h,1,2,true,true); }
API int r2c_raw_3d_f64(const double* in,double* out,int w,int h,int d) { return raw_double_fft(in,out,w,h,d,3,false); }
API int c2r_raw_3d_f64(const double* in,double* out,int w,int h,int d) { return raw_double_fft(in,out,w,h,d,3,true); }

// Retained postprocessing reductions: sequential shell order is intentional.
API void fsc_radial_accum(const double* re1,const double* im1,const double* re2,const double* im2,long xs,long ys,long zs,long n,double* num,double* den1,double* den2) {
    for (long t=0;t<n;++t) {
        const long x=t%xs,y=(t/xs)%ys,z=t/(xs*ys);
        const long yy=y<xs?y:y-ys,zz=z<xs?z:z-zs;
        const long shell=std::llround(std::sqrt(double(x*x+yy*yy+zz*zz)));
        if (shell>=xs) continue;
        const std::complex<double> a(re1[t],im1[t]),b(re2[t],im2[t]);
        const double ra=std::sqrt(re1[t]*re1[t]+im1[t]*im1[t]);
        const double rb=std::sqrt(re2[t]*re2[t]+im2[t]*im2[t]);
        num[shell]+=(std::conj(a)*b).real(); den1[shell]+=ra*ra; den2[shell]+=rb*rb;
    }
}
API void guinier_accum(const double* re,const double* im,long xs,long ys,long zs,long n,double* sum,long* count) {
    for (long t=0;t<n;++t) {
        const long x=t%xs,y=(t/xs)%ys,z=t/(xs*ys);
        const long yy=y<xs?y:y-ys,zz=z<xs?z:z-zs;
        const long shell=std::llround(std::sqrt(double(x*x+yy*yy+zz*zz)));
        if (shell>=xs) continue;
        sum[shell]+=std::sqrt(re[t]*re[t]+im[t]*im[t]); ++count[shell];
    }
}
// Use the paired glibc operation, without compiler expansion into two calls.
API void greco_sincos_d(const double* x,double* s,double* c,long n) {
    void (*volatile fn)(double,double*,double*)=::sincos;
    for (long i=0;i<n;++i) fn(x[i],s+i,c+i);
}
API void greco_sin_d(const double* x,double* out,long n) { for (long i=0;i<n;++i) out[i]=std::sin(x[i]); }
API void greco_cos_d(const double* x,double* out,long n) { for (long i=0;i<n;++i) out[i]=std::cos(x[i]); }
API void greco_exp_d(const double* x,double* out,long n) { for (long i=0;i<n;++i) out[i]=std::exp(x[i]); }
API void greco_pow_d(const double* x,double exponent,double* out,long n) { for (long i=0;i<n;++i) out[i]=std::pow(x[i],exponent); }
