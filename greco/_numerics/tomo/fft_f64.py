"""Double-precision FFTW half-spectrum transforms.

Greco's minimal native core provides normalized and raw 2D/3D transforms.
The conventions are retained from the numerical reference; the implementation
links FFTW directly, with no reference binary dependency. GRECO_FFT_ENGINE selects
auto/native, pyfftw or numpy; GRECO_NUMERICS_SO selects a compatible build.
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import ctypes
import os
from typing import Optional

import numpy as np

from . import fft_f32
from ..native.loader import library_path, load as load_numerics

__all__ = ['rfft2_both_f64', 'irfft2_both_f64', 'rfftn3_fwdonly_f64', 'irfftn3_fwdonly_f64', 'rfft2_raw_f64', 'irfft2_raw_f64', 'rfftn3_raw_f64', 'irfftn3_raw_f64']

try:
    from pyfftw.interfaces import numpy_fft as _npf
    _HAVE_PYFFTW = True
except Exception:  # pragma: no cover
    _HAVE_PYFFTW = False

_NATIVE = {"lib": None, "tried": False, "path": None}


def _engine() -> str:
    return os.environ.get("GRECO_FFT_ENGINE", "auto").strip().lower()


def _native_lib() -> Optional[ctypes.CDLL]:
    """Load the checked Greco numerical core with the f64 exports."""
    if _NATIVE["tried"]:
        return _NATIVE["lib"]
    _NATIVE["tried"] = True
    eng = _engine()
    if eng in ('numpy','pyfftw'):return None
    so = fft_f32._find_native_so()
    if so is None:
        if eng == "native":
            raise RuntimeError(
                "Greco numerical core is missing. Build with "
                "python -m greco._numerics.native.build")
        return None
    lib = load_numerics()
    for name in ("r2c_both_f64", "c2r_both_f64"):
        fn = getattr(lib, name)
        fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p]
                       + [ctypes.c_int] * 3)
        fn.restype = ctypes.c_int
    _NATIVE["lib"] = lib
    _NATIVE["path"] = so
    _NATIVE["fwdonly64"] = all(
        hasattr(lib, n) for n in ("r2c_fwdonly_f64", "c2r_fwdonly_f64"))
    if _NATIVE["fwdonly64"]:
        for name in ("r2c_fwdonly_f64", "c2r_fwdonly_f64"):
            fn = getattr(lib, name)
            fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p]
                           + [ctypes.c_int] * 3)
            fn.restype = ctypes.c_int
    _NATIVE["fwdonly32"] = all(
        hasattr(lib, n) for n in ("r2c_fwdonly_f32", "c2r_fwdonly_f32"))
    if _NATIVE["fwdonly32"]:
        for name in ("r2c_fwdonly_f32", "c2r_fwdonly_f32"):
            fn = getattr(lib, name)
            fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p]
                           + [ctypes.c_int] * 3)
            fn.restype = ctypes.c_int
    return lib


def _r2c_native(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Shim r2c: (h, w) f64 -> (h, w//2+1) complex128 (reference order)."""
    lib = _native_lib()
    in_arr = np.ascontiguousarray(arr, dtype=np.float64)
    if in_arr.shape != (h, w):
        raise ValueError(f"_r2c_native: shape {in_arr.shape} != {(h, w)}")
    flat = np.ascontiguousarray(in_arr.reshape(1, h, w))
    out = np.empty((1, h, w // 2 + 1, 2), dtype=np.float64)
    lib.r2c_both_f64(flat.ctypes.data_as(ctypes.c_void_p),
                     out.ctypes.data_as(ctypes.c_void_p), w, h, 1)
    return out.view(np.complex128).reshape((h, w // 2 + 1))


def _c2r_native(fs: np.ndarray, w: int, h: int) -> np.ndarray:
    """Shim c2r: (h, w//2+1) complex128 -> (h, w) f64 (reference order)."""
    lib = _native_lib()
    fs_c = np.ascontiguousarray(fs, dtype=np.complex128)
    if fs_c.shape != (h, w // 2 + 1):
        raise ValueError(f"_c2r_native: shape {fs_c.shape} != "
                         f"{(h, w // 2 + 1)}")
    flat = np.ascontiguousarray(fs_c.reshape(1, h, w // 2 + 1))
    out = np.empty((1, h, w), dtype=np.float64)
    lib.c2r_both_f64(flat.ctypes.data_as(ctypes.c_void_p),
                     out.ctypes.data_as(ctypes.c_void_p), w, h, 1)
    return out[0].copy()


def rfft2_both_f64(img: np.ndarray) -> np.ndarray:
    """2-D forward r2c on ``(h, w)`` f64, reference ``Both`` (raw/sqrt(N))."""
    h, w = img.shape
    if _native_lib() is not None and _engine() != "pyfftw":
        return _r2c_native(img, w, h)
    if _HAVE_PYFFTW:
        fs = _npf.rfft2(img, threads=fft_f32.fftw_threads())
    else:
        fs = np.fft.rfft2(img)
    return (fs / np.sqrt(float(h * w))).astype(np.complex128, copy=False)


def irfft2_both_f64(fs: np.ndarray, s: tuple) -> np.ndarray:
    """2-D inverse from a ``(h, w//2+1)`` half spectrum, reference ``Both``."""
    s = (int(s[0]), int(s[1]))
    h, w = s
    if (w % 2 == 0 and _native_lib() is not None
            and _engine() != "pyfftw"):
        return _c2r_native(fs, w, h)
    if _HAVE_PYFFTW:
        out = _npf.irfft2(fs, s=s, threads=fft_f32.fftw_threads())
    else:
        out = np.fft.irfft2(fs, s=s)
    # pyfftw/numpy irfft2 is 1/N-normalised; reference's c2r (Both) is
    # raw/sqrt(N): net factor * sqrt(N).
    return (out * np.sqrt(float(h * w))).astype(np.float64)


# ------------------------------------------------------------------ 3-D r2c
# FwdOnly normalization (postprocess semantics: fwd raw/N^3, inv raw)

def _r2c_native_fwdonly_f64(arr: np.ndarray, w: int, h: int,
                            d: int) -> np.ndarray:
    """Shim r2c (FwdOnly f64): (d, h, w) -> (d, h, w//2+1) complex128."""
    lib = _native_lib()
    if not _NATIVE["fwdonly64"]:
        raise RuntimeError("Greco numerical core lacks r2c_fwdonly_f64; "
                           "rebuild (see python -m greco._numerics.native.build)")
    in_arr = np.ascontiguousarray(arr, dtype=np.float64)
    if in_arr.shape != (d, h, w):
        raise ValueError(f"_r2c_native_fwdonly_f64: shape {in_arr.shape} "
                         f"!= {(d, h, w)}")
    out = np.empty((d, h, w // 2 + 1, 2), dtype=np.float64)
    lib.r2c_fwdonly_f64(in_arr.ctypes.data_as(ctypes.c_void_p),
                        out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out.view(np.complex128).reshape((d, h, w // 2 + 1))


def _c2r_native_fwdonly_f64(fs: np.ndarray, w: int, h: int,
                            d: int) -> np.ndarray:
    """Shim c2r (FwdOnly f64): (d, h, w//2+1) complex128 -> (d, h, w)."""
    lib = _native_lib()
    if not _NATIVE["fwdonly64"]:
        raise RuntimeError("Greco numerical core lacks c2r_fwdonly_f64; "
                           "rebuild (see python -m greco._numerics.native.build)")
    fs_c = np.ascontiguousarray(fs, dtype=np.complex128)
    if fs_c.shape != (d, h, w // 2 + 1):
        raise ValueError(f"_c2r_native_fwdonly_f64: shape {fs_c.shape} != "
                         f"{(d, h, w // 2 + 1)}")
    out = np.empty((d, h, w), dtype=np.float64)
    lib.c2r_fwdonly_f64(fs_c.ctypes.data_as(ctypes.c_void_p),
                        out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out.copy()


def rfftn3_fwdonly_f64(img: np.ndarray) -> np.ndarray:
    """3-D forward r2c, reference FwdOnly (raw/N**3), f64 engine.

    ``img`` is ``(z, y, x)`` C-order f64 (x fastest); returns the half
    spectrum ``(z, y, x//2+1)`` complex128.  Native path = GT
    ``FFT::FourierTransform`` (``FFT::FwdOnly``); fallback pyfftw/numpy
    (last-ulp differences possible).
    """
    z, y, x = img.shape
    if (x % 2 == 0 and _native_lib() is not None
            and _engine() != "pyfftw"):
        return _r2c_native_fwdonly_f64(
            np.ascontiguousarray(img, dtype=np.float64), x, y, z)
    if _HAVE_PYFFTW:
        fs = _npf.rfftn(img, threads=fft_f32.fftw_threads())
    else:
        fs = np.fft.rfftn(img)
    return (fs / float(x * y * z)).astype(np.complex128, copy=False)


def irfftn3_fwdonly_f64(fs: np.ndarray,
                        s: tuple) -> np.ndarray:
    """3-D inverse r2c, reference FwdOnly (raw c2r, unnormalised), f64.

    ``fs`` is ``(z, y, x//2+1)`` complex128, ``s = (z, y, x)``; returns
    the real ``(z, y, x)`` f64 volume.
    """
    s = (int(s[0]), int(s[1]), int(s[2]))
    z, y, x = s
    if (x % 2 == 0 and _native_lib() is not None
            and _engine() != "pyfftw"):
        return _c2r_native_fwdonly_f64(
            np.ascontiguousarray(fs, dtype=np.complex128), x, y, z)
    if _HAVE_PYFFTW:
        out = _npf.irfftn(fs, s=s, threads=fft_f32.fftw_threads())
    else:
        out = np.fft.irfftn(fs, s=s)
    # numpy irfftn is 1/N^3-normalised; reference's c2r is raw: * N^3.
    return (out * float(x * y * z)).astype(np.float64)


# ------------------------------------------------------------------ 3-D r2c
# FwdOnly f32 (fftwf): GT randomizePhasesBeyond/getFSC (RFLOAT=float)
# call FFT::FourierTransform -> fftwf forward /N^3 and raw c2r inverse.

def _r2c_native_fwdonly_f32(arr: np.ndarray, w: int, h: int,
                            d: int) -> np.ndarray:
    """Shim r2c (FwdOnly f32): (d, h, w) -> (d, h, w//2+1) complex64."""
    lib = _native_lib()
    if not _NATIVE["fwdonly32"]:
        raise RuntimeError("Greco numerical core lacks r2c_fwdonly_f32; rebuild "
                           "(see python -m greco._numerics.native.build)")
    in_arr = np.ascontiguousarray(arr, dtype=np.float32)
    if in_arr.shape != (d, h, w):
        raise ValueError(f"_r2c_native_fwdonly_f32: shape {in_arr.shape} "
                         f"!= {(d, h, w)}")
    out = np.empty((d, h, w // 2 + 1, 2), dtype=np.float32)
    lib.r2c_fwdonly_f32(in_arr.ctypes.data_as(ctypes.c_void_p),
                        out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out.view(np.complex64).reshape((d, h, w // 2 + 1))


def _c2r_native_fwdonly_f32(fs: np.ndarray, w: int, h: int,
                            d: int) -> np.ndarray:
    """Shim c2r (FwdOnly f32): (d, h, w//2+1) complex64 -> (d, h, w)."""
    lib = _native_lib()
    if not _NATIVE["fwdonly32"]:
        raise RuntimeError("Greco numerical core lacks c2r_fwdonly_f32; rebuild "
                           "(see python -m greco._numerics.native.build)")
    fs_c = np.ascontiguousarray(fs, dtype=np.complex64)
    if fs_c.shape != (d, h, w // 2 + 1):
        raise ValueError(f"_c2r_native_fwdonly_f32: shape {fs_c.shape} "
                         f"!= {(d, h, w // 2 + 1)}")
    out = np.empty((d, h, w), dtype=np.float32)
    lib.c2r_fwdonly_f32(fs_c.ctypes.data_as(ctypes.c_void_p),
                        out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out.copy()


# ------------------------------------------------------------------ 2-D raw
# Legacy FourierTransformer (src/fftw.cpp) normalization, f64:
#   forward: fftw_plan_dft_r2c(2, {h, w}, ..., FFTW_ESTIMATE) -- raw r2c
#   inverse: fftw_plan_dft_c2r(2, {h, w}, ..., FFTW_ESTIMATE) -- raw c2r
# (i.e. the round trip multiplies by N^2).  This is what ctf_toolbox uses
# on Image<double> 2-D images in a BASE=double build.  The native path
# calls the identical FFTW entry points from the Greco numerical core, so it is
# bit-exact with the GT binary by construction.

_NATIVE2 = {"lib": None, "tried": False, "path": None}


def _find_native_so2() -> Optional[str]:
    return library_path()


def _native2_lib() -> Optional[ctypes.CDLL]:
    if _NATIVE2["tried"]:
        return _NATIVE2["lib"]
    _NATIVE2["tried"] = True
    if _engine() in ("numpy","pyfftw"):
        return None
    so = _find_native_so2()
    if so is None:
        if _engine() == "native":
            raise RuntimeError(
                "Greco numerical core is missing. Build with "
                "python -m greco._numerics.native.build")
        return None
    lib = load_numerics()
    for name in ("r2c_raw_2d_f64", "c2r_raw_2d_f64"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int
    _NATIVE2["c2r2d_aligned"] = hasattr(lib, "c2r_raw_2d_f64_aligned")
    if _NATIVE2["c2r2d_aligned"]:
        fn = getattr(lib, "c2r_raw_2d_f64_aligned")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int
    _NATIVE2["raw3d"] = all(hasattr(lib, n) for n in
                            ("r2c_raw_3d_f64", "c2r_raw_3d_f64"))
    if _NATIVE2["raw3d"]:
        for name in ("r2c_raw_3d_f64", "c2r_raw_3d_f64"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn.restype = ctypes.c_int
    _NATIVE2["lib"] = lib
    _NATIVE2["path"] = so
    return lib


def rfft2_raw_f64(img: np.ndarray) -> np.ndarray:
    """2-D raw (unnormalised) r2c on ``(h, w)`` f64, reference legacy
    FourierTransformer forward semantics.  Returns ``(h, w//2+1)``
    complex128 in reference (y-first) order."""
    h, w = img.shape
    lib = _native2_lib()
    if lib is not None:
        out = np.empty((h, w // 2 + 1, 2), dtype=np.float64)
        lib.r2c_raw_2d_f64(
            np.ascontiguousarray(img, dtype=np.float64)
            .ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p), w, h)
        return out.view(np.complex128).reshape(h, w // 2 + 1)
    if _HAVE_PYFFTW:
        fs = _npf.rfft2(img, threads=fft_f32.fftw_threads())
    else:
        fs = np.fft.rfft2(img)
    return np.asarray(fs, dtype=np.complex128)


def irfft2_raw_f64(fs: np.ndarray, s: tuple) -> np.ndarray:
    """2-D raw (unnormalised) c2r, reference legacy FourierTransformer
    inverse semantics (output = N^2 x the mathematical inverse)."""
    s = (int(s[0]), int(s[1]))
    h, w = s
    lib = _native2_lib()
    if lib is not None:
        out = np.empty((h, w), dtype=np.float64)
        fn = getattr(lib, "c2r_raw_2d_f64_aligned", None)
        if fn is None:
            fn = lib.c2r_raw_2d_f64
        fn(np.ascontiguousarray(fs, dtype=np.complex128)
           .ctypes.data_as(ctypes.c_void_p),
           out.ctypes.data_as(ctypes.c_void_p), w, h)
        return out
    if _HAVE_PYFFTW:
        out = _npf.irfft2(fs, s=s, threads=fft_f32.fftw_threads())
    else:
        out = np.fft.irfft2(fs, s=s)
    # numpy/pyfftw irfft2 is 1/N-normalised; the legacy raw c2r is not.
    return (out * float(h * w)).astype(np.float64)


# ------------------------------------------------------------------ 3-D raw
# LEGACY FourierTransformer pair (src/fftw.cpp): r2c/c2r plans built with
# FFTW_ESTIMATE (no FFTW_UNALIGNED), no scaling.  Forward raw r2c and
# backward raw c2r (the mathematical inverse times N**3).  Unlike the
# x-first transform helpers above, these work for ODD widths too (the BackProjector
# padding grid P = 2*(round(pf*r_max)+1)+1 is always odd).


def rfftn3_raw_f64(img: np.ndarray) -> np.ndarray:
    """3-D raw (unnormalised) r2c, reference legacy FourierTransformer
    forward semantics.  ``img`` is ``(z, y, x)`` C-order f64 (x fastest);
    returns the half spectrum ``(z, y, x//2+1)`` complex128."""
    z, y, x = img.shape
    lib = _native2_lib()
    if lib is not None and _NATIVE2.get("raw3d"):
        out = np.empty((z, y, x // 2 + 1, 2), dtype=np.float64)
        lib.r2c_raw_3d_f64(
            np.ascontiguousarray(img, dtype=np.float64)
            .ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p), x, y, z)
        return out.view(np.complex128).reshape(z, y, x // 2 + 1)
    if _HAVE_PYFFTW:
        fs = _npf.rfftn(img, axes=(0, 1, 2), threads=fft_f32.fftw_threads())
    else:
        fs = np.fft.rfftn(img, axes=(0, 1, 2))
    return np.asarray(fs, dtype=np.complex128)


def irfftn3_raw_f64(fs: np.ndarray, s: tuple) -> np.ndarray:
    """3-D raw (unnormalised) c2r, reference legacy FourierTransformer
    inverse semantics (output = N**3 x the mathematical inverse).
    ``fs`` is ``(z, y, x//2+1)`` complex128, ``s = (z, y, x)``; returns
    the real ``(z, y, x)`` f64 volume."""
    s = (int(s[0]), int(s[1]), int(s[2]))
    z, y, x = s
    lib = _native2_lib()
    if lib is not None and _NATIVE2.get("raw3d"):
        out = np.empty((z, y, x), dtype=np.float64)
        lib.c2r_raw_3d_f64(
            np.ascontiguousarray(fs, dtype=np.complex128)
            .ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p), x, y, z)
        return out
    if _HAVE_PYFFTW:
        out = _npf.irfftn(fs, s=s, threads=fft_f32.fftw_threads())
    else:
        out = np.fft.irfftn(fs, s=s)
    # numpy/pyfftw irfftn is 1/N^3-normalised; the legacy raw c2r is not.
    return (out * float(z * y * x)).astype(np.float64)
