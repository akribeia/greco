"""Scalar glibc math and vector loops in Greco's minimal C++ core.

Paired sin/cos and sequential scalar rounding preserve the numerical
conventions of the reference port. The source-built core links glibc directly;
no reference object code is used. Python scalar-loop fallbacks remain available.
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import ctypes
import math
from functools import lru_cache

import numpy as np
from .native.loader import load as load_numerics

__all__ = ['sincos', 'sincos_array', 'sin_array', 'cos_array', 'exp_array']

_libm = ctypes.CDLL("libm.so.6")
_libm.sincos.restype = ctypes.c_int
_libm.sincos.argtypes = [ctypes.c_double,
                         ctypes.POINTER(ctypes.c_double),
                         ctypes.POINTER(ctypes.c_double)]

def sincos(x: float) -> tuple:
    """glibc ``sincos(x)`` -> (sin, cos), bit-exact with the C++ call."""
    s = ctypes.c_double()
    c = ctypes.c_double()
    _libm.sincos(ctypes.c_double(x), ctypes.byref(s), ctypes.byref(c))
    return s.value, c.value


@lru_cache(maxsize=1)
def _native_library():
    lib = load_numerics()
    if lib is not None:
        P = ctypes.POINTER(ctypes.c_double)
        lib.greco_sincos_d.argtypes = [P, P, P, ctypes.c_long]
        lib.greco_sincos_d.restype = None
        for name in ('greco_sin_d', 'greco_cos_d'):
            fn = getattr(lib, name)
            fn.argtypes = [P, P, ctypes.c_long]
            fn.restype = None
    return lib


def _loop_f64(f, x: np.ndarray) -> np.ndarray:
    xf = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty_like(xf)
    outf = out.reshape(-1)
    flat = xf.reshape(-1)
    for i in range(flat.size):
        outf[i] = f(float(flat[i]))
    return out


def _native_loop(fn, x: np.ndarray) -> np.ndarray:
    xf = np.ascontiguousarray(x, dtype=np.float64)
    shape = xf.shape
    n = xf.size
    out = np.empty(n, dtype=np.float64)
    if n:
        P = ctypes.POINTER(ctypes.c_double)
        fn(xf.reshape(-1).ctypes.data_as(P), out.ctypes.data_as(P), n)
    return out.reshape(shape)


def sin_array(x: np.ndarray) -> np.ndarray:
    """Element-wise glibc ``sin`` (the same routine C++ scalar calls use)."""
    lib = _native_library()
    if lib is not None:
        return _native_loop(lib.greco_sin_d, x)
    return _loop_f64(math.sin, x)


def cos_array(x: np.ndarray) -> np.ndarray:
    """Element-wise glibc ``cos`` (the same routine C++ scalar calls use)."""
    lib = _native_library()
    if lib is not None:
        return _native_loop(lib.greco_cos_d, x)
    return _loop_f64(math.cos, x)


def exp_array(x: np.ndarray) -> np.ndarray:
    """Element-wise glibc ``exp`` (f64)."""
    lib = _native_library()
    if lib is not None:
        lib.greco_exp_d.argtypes = [ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.c_long]
        lib.greco_exp_d.restype = None
        return _native_loop(lib.greco_exp_d, x)
    return _loop_f64(math.exp, x)


def sincos_array(x: np.ndarray) -> tuple:
    """Vectorized glibc ``sincos``: ``x`` (any shape, float64) -> (s, c).

    Element-wise identical to the scalar ``sincos``.
    """
    xf = np.ascontiguousarray(x, dtype=np.float64)
    shape = xf.shape
    n = xf.size
    if n == 0:
        return xf.copy(), xf.copy()
    flat = xf.reshape(-1)
    s = np.empty(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    lib = _native_library()
    if lib is not None:
        P = ctypes.POINTER(ctypes.c_double)
        lib.greco_sincos_d(flat.ctypes.data_as(P), s.ctypes.data_as(P),
                         c.ctypes.data_as(P), n)
    else:
        for i in range(n):
            s[i], c[i] = sincos(float(flat[i]))
    return s.reshape(shape), c.reshape(shape)


