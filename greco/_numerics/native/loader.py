"""One checked ABI for Greco's source-built numerical core."""
import ctypes
from functools import lru_cache
import os
from pathlib import Path


def library_path():
    override=os.environ.get('GRECO_NUMERICS_SO')
    path=Path(override).expanduser().resolve() if override else Path(__file__).with_name('greco_numerics.so')
    if path.is_file():return str(path)
    if override:raise RuntimeError('GRECO_NUMERICS_SO does not point to a numerical library: '+str(path))
    return None


def _check_fft(status,function,arguments):
    if status:
        raise RuntimeError(f'Greco numerical kernel {function.__name__} failed ({status}).')
    return status


@lru_cache(maxsize=1)
def load():
    path=library_path()
    if path is None:return None
    lib=ctypes.CDLL(path)
    lib.greco_numerics_version.argtypes=[]
    lib.greco_numerics_version.restype=ctypes.c_int
    if lib.greco_numerics_version()!=2:raise RuntimeError('Unsupported Greco numerical ABI: '+path)
    names={
        **{f'{direction}_{norm}_{precision}':3 for direction in ('r2c','c2r') for norm in ('both','fwdonly') for precision in ('f32','f64')},
        'greco_fft_raw':6,
        'r2c_none_2d_f32':2,'c2r_none_2d_f32':2,
        'r2c_raw_2d_f64':2,'c2r_raw_2d_f64':2,'c2r_raw_2d_f64_aligned':2,
        'r2c_raw_3d_f64':3,'c2r_raw_3d_f64':3,
    }
    for name,dimensions in names.items():
        fn=getattr(lib,name)
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*dimensions
        fn.restype=ctypes.c_int
        fn.errcheck=_check_fft
    return lib
