"""Single-precision FFTW half-spectrum transforms.

The native engine uses Greco's minimal C++ core, built directly against FFTW
without reference headers or object code. Normalization preserves the numerical
conventions of the retained reference port (Both/FwdOnly/raw).
GRECO_FFT_ENGINE selects auto/native, pyfftw or numpy. GRECO_NUMERICS_SO can
select an explicitly built compatible core. Arrays use y/x or z/y/x order.
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import ctypes
import os
from typing import Optional

import numpy as np
from ..native.loader import library_path, load as load_numerics

__all__ = ['rfft2_both', 'irfft2_both', 'fftw_threads']

try:
    import pyfftw
    from pyfftw.interfaces import numpy_fft as _npf
    _HAVE_PYFFTW = True
except Exception:  # pragma: no cover
    _HAVE_PYFFTW = False


def fftw_threads() -> int:
    """Number of FFTW worker threads (GRECO_FFTW_THREADS, default 8)."""
    return max(1, int(os.environ.get("GRECO_FFTW_THREADS", "8")))


def _engine() -> str:
    return os.environ.get("GRECO_FFT_ENGINE", "auto").strip().lower()


def _find_native_so() -> Optional[str]:
    return library_path()


_NATIVE = {"lib": None, "tried": False, "path": None, "fwdonly": False,
           "f64": False}


def _native_lib() -> Optional[ctypes.CDLL]:
    """Load Greco's checked numerical core, or the requested fallback."""
    if _NATIVE["tried"]:
        return _NATIVE["lib"]
    _NATIVE["tried"] = True
    eng = _engine()
    if eng in ('numpy','pyfftw'):return None
    so = _find_native_so()
    if so is None:
        if eng == "native":
            raise RuntimeError(
                "Greco numerical core is missing. Build with "
                "python -m greco._numerics.native.build")
        return None
    lib = load_numerics()
    for name in ("r2c_both_f32", "c2r_both_f32",
                 "r2c_fwdonly_f32", "c2r_fwdonly_f32"):
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p]
                       + [ctypes.c_int] * 3)
        fn.restype = ctypes.c_int
    for name in ("r2c_both_f64", "c2r_both_f64"):
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p]
                           + [ctypes.c_int] * 3)
            fn.restype = ctypes.c_int
    _NATIVE["lib"] = lib
    _NATIVE["path"] = so
    _NATIVE["fwdonly"] = (hasattr(lib, "r2c_fwdonly_f32")
                          and hasattr(lib, "c2r_fwdonly_f32"))
    _NATIVE["f64"] = (hasattr(lib, "r2c_both_f64")
                      and hasattr(lib, "c2r_both_f64"))
    return lib


# ------------------------------------------------------------------ native

def _r2c_native(arr: np.ndarray, w: int, h: int, d: int) -> np.ndarray:
    """Shim r2c: (d, h, w) f32 -> (d, h, w//2+1) complex64 (reference order)."""
    lib = _native_lib()
    in_arr = np.ascontiguousarray(arr, dtype=np.float32)
    if in_arr.shape != (d, h, w):
        raise ValueError(f"_r2c_native: shape {in_arr.shape} != {(d, h, w)}")
    out = np.empty((d, h, w // 2 + 1, 2), dtype=np.float32)
    lib.r2c_both_f32(in_arr.ctypes.data_as(ctypes.c_void_p),
                     out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out.view(np.complex64).reshape((d, h, w // 2 + 1))


def _c2r_native(fs: np.ndarray, w: int, h: int, d: int) -> np.ndarray:
    """Shim c2r: (d, h, w//2+1) complex64 -> (d, h, w) f32 (reference order)."""
    lib = _native_lib()
    fs_c = np.ascontiguousarray(fs, dtype=np.complex64)
    if fs_c.shape != (d, h, w // 2 + 1):
        raise ValueError(f"_c2r_native: shape {fs_c.shape} != "
                         f"{(d, h, w // 2 + 1)}")
    out = np.empty((d, h, w), dtype=np.float32)
    fs_f = fs_c.view(np.float32).reshape((d, h, w // 2 + 1, 2))
    lib.c2r_both_f32(fs_f.ctypes.data_as(ctypes.c_void_p),
                     out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out


def _r2c_native_fwdonly(arr: np.ndarray, w: int, h: int,
                        d: int) -> np.ndarray:
    """Shim r2c (FwdOnly): raw r2c then f32 element-wise /= f32(N)."""
    lib = _native_lib()
    if not _NATIVE["fwdonly"]:
        raise RuntimeError("Greco numerical core lacks r2c_fwdonly_f32; rebuild")
    in_arr = np.ascontiguousarray(arr, dtype=np.float32)
    if in_arr.shape != (d, h, w):
        raise ValueError(f"_r2c_native_fwdonly: shape {in_arr.shape} "
                         f"!= {(d, h, w)}")
    out = np.empty((d, h, w // 2 + 1, 2), dtype=np.float32)
    lib.r2c_fwdonly_f32(in_arr.ctypes.data_as(ctypes.c_void_p),
                        out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out.view(np.complex64).reshape((d, h, w // 2 + 1))


def _c2r_native_fwdonly(fs: np.ndarray, w: int, h: int, d: int) -> np.ndarray:
    """Shim c2r (FwdOnly): raw c2r, unnormalised."""
    lib = _native_lib()
    if not _NATIVE["fwdonly"]:
        raise RuntimeError("Greco numerical core lacks c2r_fwdonly_f32; rebuild")
    fs_c = np.ascontiguousarray(fs, dtype=np.complex64)
    if fs_c.shape != (d, h, w // 2 + 1):
        raise ValueError(f"_c2r_native_fwdonly: shape {fs_c.shape} != "
                         f"{(d, h, w // 2 + 1)}")
    out = np.empty((d, h, w), dtype=np.float32)
    fs_f = fs_c.view(np.float32).reshape((d, h, w // 2 + 1, 2))
    lib.c2r_fwdonly_f32(fs_f.ctypes.data_as(ctypes.c_void_p),
                        out.ctypes.data_as(ctypes.c_void_p), w, h, d)
    return out


# ------------------------------------------------------------------ 2-D r2c

def rfft2_both(img: np.ndarray, planner_effort: int = None,
               threads: int = None) -> np.ndarray:
    """2-D forward r2c, reference ``Both`` (raw/sqrt(N)), f32 engine.

    ``planner_effort``/``threads`` are honoured by the pyfftw engine
    (reference's default ``FFT::FloatPlan`` is FFTW_ESTIMATE, single-threaded:
    ``planner_effort="FFTW_ESTIMATE", threads=1``); the ``reference`` engine
    always uses the GT plan and ignores them.
    """
    img = np.ascontiguousarray(img, dtype=np.float32)
    h, w = img.shape
    if _native_lib() is not None and _engine() != "pyfftw":
        return _r2c_native(img[None], w, h, 1)[0]
    if _HAVE_PYFFTW:
        if planner_effort is None and threads is None:
            fs = _npf.rfft2(img, threads=fftw_threads())
        else:
            fs = _npf.rfft2(img, planner_effort=planner_effort,
                            threads=threads)
    else:
        fs = np.fft.rfft2(img)
    return (fs / np.sqrt(_n2(img.shape))).astype(
        np.complex64 if _HAVE_PYFFTW else np.complex128, copy=False)


def irfft2_both(fs: np.ndarray, s: tuple, planner_effort: int = None,
                threads: int = None) -> np.ndarray:
    """2-D inverse r2c from a reference half spectrum, ``Both``."""
    s = (int(s[0]), int(s[1]))
    h, w = s
    # reference C++ quirk: inverseFourierTransform on an odd real width
    # truncates the output to even (resizeRealToMatch); the port uses
    # the mathematically correct odd-width result instead.
    if (w % 2 == 0 and _native_lib() is not None
            and _engine() != "pyfftw"):
        return _c2r_native(fs[None], w, h, 1)[0]
    if _HAVE_PYFFTW:
        if planner_effort is None and threads is None:
            out = _npf.irfft2(fs, s=s, threads=fftw_threads())
        else:
            out = _npf.irfft2(fs, s=s, planner_effort=planner_effort,
                              threads=threads)
    else:
        out = np.fft.irfft2(fs, s=s)
    # pyfftw/numpy irfft2 is 1/N-normalised; reference's c2r (Both) is
    # raw/sqrt(N): undo the 1/N and apply /sqrt(N)  =>  * sqrt(N).
    return (out * np.sqrt(int(s[0]) * int(s[1]))).astype(np.float32)


def _r2c_native_none2d(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Shim raw r2c 2-D (FFT::None): (h, w) f32 -> (h, w//2+1) c64."""
    lib = _native_lib()
    fn = getattr(lib, "r2c_none_2d_f32", None)
    if fn is None:
        raise RuntimeError("Greco numerical core lacks r2c_none_2d_f32")
    if fn.argtypes is None:
        fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p]
                       + [ctypes.c_int] * 2)
        fn.restype = ctypes.c_int
    in_arr = np.ascontiguousarray(img, dtype=np.float32)
    if in_arr.shape != (h, w):
        raise ValueError(f"_r2c_native_none2d: shape {in_arr.shape} "
                         f"!= {(h, w)}")
    out = np.empty((h, w // 2 + 1, 2), dtype=np.float32)
    fn(in_arr.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p), w, h)
    return out.view(np.complex64).reshape((h, w // 2 + 1))


def _c2r_native_none2d(fs: np.ndarray, w: int, h: int) -> np.ndarray:
    """Shim raw c2r 2-D (FFT::None): (h, w//2+1) c64 -> (h, w) f32."""
    lib = _native_lib()
    fn = getattr(lib, "c2r_none_2d_f32", None)
    if fn is None:
        raise RuntimeError("Greco numerical core lacks c2r_none_2d_f32")
    if fn.argtypes is None:
        fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p]
                       + [ctypes.c_int] * 2)
        fn.restype = ctypes.c_int
    fs_c = np.ascontiguousarray(fs, dtype=np.complex64)
    if fs_c.shape != (h, w // 2 + 1):
        raise ValueError(f"_c2r_native_none2d: shape {fs_c.shape} "
                         f"!= {(h, w // 2 + 1)}")
    out = np.empty((h, w), dtype=np.float32)
    fs_f = fs_c.view(np.float32).reshape((h, w // 2 + 1, 2))
    fn(fs_f.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p), w, h)
    return out


def _n2(shape) -> int:
    return int(shape[0]) * int(shape[1])


# ------------------------------------------------------------------ 3-D


# ------------------------------------------------------------------ 3-D r2c


# ------------------------------------------------------------------ 3-D r2c
# FwdOnly (reference default normalization: forward raw/N, inverse raw)


# ------------------------------------------------------------------ 3-D r2c f64
