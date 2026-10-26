# Third-party references and adapted components
# Selected GPU resampling routines reproduce SciPy ndimage NI_GeometricTransform, NI_ZoomShift and order-1 spline interpolation arithmetic. Original ndimage notice is retained below.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/scipy/LICENSE.txt
# licenses/scipy/NDIMAGE_NOTICE.txt
#
# Upstream notice (retained):
# /* Copyright (C) 2003-2005 Peter J. Verveer
#  *
#  * Redistribution and use in source and binary forms, with or without
#  * modification, are permitted provided that the following conditions
#  * are met:
#  *
#  * 1. Redistributions of source code must retain the above copyright
#  *    notice, this list of conditions and the following disclaimer.
#  *
#  * 2. Redistributions in binary form must reproduce the above
#  *    copyright notice, this list of conditions and the following
#  *    disclaimer in the documentation and/or other materials provided
#  *    with the distribution.
#  *
#  * 3. The name of the author may not be used to endorse or promote
#  *    products derived from this software without specific prior
#  *    written permission.
#  *
#  * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
#  * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
#  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
#  * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#  * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  */
#

"""Bit-exact GPU replicas of the SciPy order-1 resampling used by the
reference driver (``ndi_shift`` / ``map_coordinates``, order=1,
``mode='constant'``).

Why this exists
---------------
The reference 2-D driver resamples on the CPU with SciPy (cumulative
re-materialization, patch extraction, search-region sampling) and runs the
GPU only for LOO / angular transfer / projection / registration.  The plain
CuPy order-1 kernels differ from SciPy's at the ~1e-6 level, and the
iterative GCC loop amplifies that into a ~2 px drift of the final solution.
To keep the whole pipeline on the GPU *and* bit-exact against the reference,
these kernels reproduce the exact arithmetic of SciPy 1.17.1
(`scipy/ndimage/src/ni_interpolation.c`, ``NI_GeometricTransform`` and
``NI_ZoomShift``, plus ``ni_splines.c:get_spline_interpolation_weights``)
for the order-1 / mode='constant' / float32 case:

  * all intermediate arithmetic in IEEE-754 double, same order of
    operations (IEEE determinism => bit-identical results);
  * fraction   ``x  = cc - floor(cc)``;
  * weights    ``w0 = 1.0 - x;  w1 = 1.0 - w0``  (NOT ``w1 = x``);
  * corner order ``(0,0), (0,1), (1,0), (1,1)`` (x offset fastest, the
    SciPy ``fcoordinates`` table order);
  * per corner ``t += ((double)v * w_row) * w_col``;
  * any coordinate outside ``[0, n-1]`` (before flooring) -> ``cval`` for
    the whole point;
  * far-edge footprints (``cc == n-1`` exactly) use mirror extension of the
    out-of-range index (its weight is exactly 0 anyway);
  * a single double -> float32 rounding at store time.

The kernels are compiled with ``--fmad=false`` so no multiply-add
contraction can alter the double arithmetic.

Validated bit-exact against SciPy on real data and through the full
26-stage pipeline (``scripts/bitexact_check.py``).
"""

_SRC = r"""
__device__ __forceinline__ float bilinear_scipy_point(
        const float* __restrict__ im, int Y, int X,
        double cc0, double cc1, double cval)
{
    // SciPy 'constant' mode: any coordinate outside [0, n-1] -> cval point
    if (cc0 < 0.0 || cc0 > (double)(Y - 1) || cc1 < 0.0 || cc1 > (double)(X - 1))
        return (float)cval;
    int s0 = (int)floor(cc0);
    int s1 = (int)floor(cc1);
    double x0 = cc0; x0 -= floor(x0);
    double x1 = cc1; x1 -= floor(x1);
    double w00 = 1.0 - x0; double w01 = 1.0 - w00;
    double w10 = 1.0 - x1; double w11 = 1.0 - w10;
    int i0 = s0, i1 = s0 + 1, j0 = s1, j1 = s1 + 1;
    // far-edge footprint index: SciPy map_coordinate(..., NI_EXTEND_MIRROR)
    if (i1 >= Y) {
        int sz2 = 2 * Y - 2;
        int in_ = (int)((double)i1 / (double)sz2);
        in_ = i1 - sz2 * in_;
        if (in_ >= Y) in_ = sz2 - in_;
        i1 = in_;
    }
    if (j1 >= X) {
        int sz2 = 2 * X - 2;
        int in_ = (int)((double)j1 / (double)sz2);
        in_ = j1 - sz2 * in_;
        if (in_ >= X) in_ = sz2 - in_;
        j1 = in_;
    }
    double t = 0.0;
    // corner order of the SciPy fcoordinates table: (0,0),(0,1),(1,0),(1,1)
    double c;
    c = (double)im[i0 * X + j0]; c *= w00; c *= w10; t += c;
    c = (double)im[i0 * X + j1]; c *= w00; c *= w11; t += c;
    c = (double)im[i1 * X + j0]; c *= w01; c *= w10; t += c;
    c = (double)im[i1 * X + j1]; c *= w01; c *= w11; t += c;
    return (float)t;
}

extern "C" __global__ void greco_scipy_shift(
        const float* __restrict__ data, int T, int Y, int X,
        const double* __restrict__ deltas, const double* __restrict__ cvals,
        float* __restrict__ out)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)T * (long)Y * X;
    if (tid >= n) return;
    long plane = tid / ((long)Y * X);
    int ij = (int)(tid - plane * ((long)Y * (long)X));
    int i = ij / X;
    int j = ij - i * X;
    // NI_ZoomShift: cc = (double)kk; cc += shift(=-delta); cc += nprepad(0)
    double cc0 = (double)i + (-deltas[2 * plane]);
    cc0 += 0.0;
    double cc1 = (double)j + (-deltas[2 * plane + 1]);
    cc1 += 0.0;
    out[tid] = bilinear_scipy_point(data + plane * (long)Y * X, Y, X,
                                    cc0, cc1, cvals[plane]);
}

extern "C" __global__ void greco_scipy_patch(
        const float* __restrict__ data, int plane_stride, int T, int Y, int X,
        int P, int r, int L, const double* __restrict__ centres, double cval,
        float* __restrict__ out)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long pp = (long)P * P;
    long n = (long)L * T * pp;
    if (tid >= n) return;
    int j = (int)(tid % P);
    int i = (int)((tid / P) % P);
    int t = (int)((tid / pp) % T);
    long L_ = tid / (pp * (long)T);
    // map_coordinates: coordinate = offset(int) + centre (float64 add)
    double cy = centres[2 * (L_ * T + t)];
    double cx = centres[2 * (L_ * T + t) + 1];
    double cc0 = (double)(i - r) + cy;
    double cc1 = (double)(j - r) + cx;
    out[tid] = bilinear_scipy_point(data + (long)t * plane_stride, Y, X,
                                    cc0, cc1, cval);
}
"""

_KERNELS = None


def _kernels(cp):
    # compiled with --fmad=false so no multiply-add contraction can alter
    # the double arithmetic (belt and braces: the expressions are written
    # as separate statements, which nvcc does not contract)
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = (
            cp.RawKernel(_SRC, 'greco_scipy_shift', options=('--fmad=false',)),
            cp.RawKernel(_SRC, 'greco_scipy_patch', options=('--fmad=false',)),
        )
    return _KERNELS


def _launch1d(fn, n, args):
    # cupy 14: RawKernel.__call__(grid, block, args, shared_mem=0)
    threads = 256
    blocks = (n + threads - 1) // threads
    fn((blocks, 1, 1), (threads, 1, 1), args)


def shift_stack_scipy_gpu(stack, deltas, cvals):
    """GPU replica of
    ``[ndi_shift(im, d, order=1, mode='constant', cval=c, prefilter=False)
    for im, d, c in zip(stack, deltas, cvals)]`` (bit-exact).

    stack:  (T, Y, X) float32 device array (C-contiguous).
    deltas: (T, 2) float64 (dy, dx) per plane.
    cvals:  (T,) float64 per-plane border value.
    Returns (T, Y, X) float32 device array.
    """
    from cupy import get_array_module
    cp = get_array_module(stack)
    stack = cp.ascontiguousarray(stack, dtype=cp.float32)
    T, Y, X = stack.shape
    deltas = cp.ascontiguousarray(cp.asarray(deltas, dtype=cp.float64).ravel())
    cvals = cp.ascontiguousarray(cp.asarray(cvals, dtype=cp.float64).ravel())
    if deltas.shape != (2 * T,) or cvals.shape != (T,):
        raise ValueError('deltas (T,2) and cvals (T,) must match stack')
    out = cp.empty(stack.shape, cp.float32)
    f_shift, _ = _kernels(cp)
    _launch1d(f_shift, T * Y * X, (stack, T, Y, X, deltas, cvals, out))
    return out


def patch_stack_scipy_gpu(stack, centres, radius, cval=1.0):
    """GPU replica of per-centre
    ``map_coordinates(image, offsets+centre, order=1, mode='constant',
    prefilter=False)`` (bit-exact), with offsets = mgrid(-r..r).

    stack:   (T, Y, X) or (Y, X) float32 device array.
    centres: (L, T, 2) float64 device array (floating extraction centres).
    radius:  int, patch half-size (P = 2*radius+1).
    cval:    float border value (the scipy 1.17.1 map_coordinates default
             is 0.0; the reference driver relies on that default).
    Returns (L, T, P, P) float32 device array.
    """
    from cupy import get_array_module
    cp = get_array_module(stack)
    stack = cp.ascontiguousarray(stack, dtype=cp.float32)
    if stack.ndim == 2:
        stack = stack[None]
    T, Y, X = stack.shape
    centres = cp.ascontiguousarray(cp.asarray(centres, dtype=cp.float64))
    if centres.ndim != 3 or centres.shape[1] != T or centres.shape[2] != 2:
        raise ValueError('centres must have shape (L, T, 2)')
    L = centres.shape[0]
    P = 2 * int(radius) + 1
    out = cp.empty((L, T, P, P), cp.float32)
    _, f_patch = _kernels(cp)
    _launch1d(f_patch, L * T * P * P,
              (stack, Y * X, T, Y, X, P, int(radius), L, centres,
               float(cval), out))
    return out
