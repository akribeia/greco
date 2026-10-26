# Third-party references and adapted components
# Greco Python interface and integration for the MotionCor/AreTomo3-derived EER decoder in greco/native/eer_decoder.c.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/motioncor3/LICENSE.txt
# licenses/aretomo3/LICENSE.txt
#
# Copyright 2023 Chan Zuckerberg Institute for Advanced Biological Imaging
# Full license conditions and disclaimers are retained at the paths above.
#

"""EER decoding through the TIFF-backed decoder built in greco/native/."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np

_LIB = None
_LOCK = threading.Lock()


def _lib():
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        source = Path(__file__).with_name('native') / 'eer_decoder.c'
        library = source.with_suffix('.so')
        if not library.exists():
            temporary = None
            try:
                fd, temporary = tempfile.mkstemp(prefix='.eer-decoder-', suffix='.so', dir=library.parent)
                os.close(fd)
                subprocess.run(['gcc','-O3','-shared','-fPIC','-pthread',str(source),
                                '-o',temporary,'-ltiff'], check=True,
                               capture_output=True, text=True, timeout=60)
                os.chmod(temporary, 0o755)
                os.replace(temporary, library)
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError('Cannot build EER decoder: install gcc and libtiff development headers. ' +
                                   (getattr(error,'stderr','') or str(error))) from error
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
        lib = ct.CDLL(str(library))
        lib.eer_info.restype = ct.c_int
        lib.eer_info.argtypes = [ct.c_char_p] + [ct.POINTER(ct.c_int)] * 4
        lib.eer_read_strips.restype = ct.c_int
        lib.eer_read_strips.argtypes = [ct.c_char_p,ct.c_int,ct.c_void_p,ct.c_size_t,
                                       ct.POINTER(ct.c_int),ct.POINTER(ct.c_size_t)]
        lib.eer_decode_groups.restype = ct.c_int
        lib.eer_decode_groups.argtypes = [ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_int)] + [
            ct.c_int] * 5 + [ct.c_void_p,ct.c_void_p,ct.c_int]
        _LIB = lib
        return lib


def eer_info(path):
    """Return raw subframe count, width, height and encoding bits."""
    fields = [ct.c_int() for _ in range(4)]
    rc = _lib().eer_info(os.fsencode(path),*[ct.byref(value) for value in fields])
    if rc:
        raise ValueError(f'Invalid EER file ({rc}): {path}')
    n,w,h,bits = [value.value for value in fields]
    if n < 1 or w != 4096 or h != 4096 or bits not in (7,8):
        raise ValueError('The migrated EER decoder supports 4096×4096, 7/8-bit streams.')
    return n,w,h,bits


def decode_eer(path, group=15, threads=1, upscaling=1):
    """Decode 4K/8K frames; incomplete trailing groups are explicitly reported.

    Raster is unmodified. Returns uint8 frames and acquisition metadata; no gain,
    motion correction, dose weighting or orientation transform is applied here.
    """
    for name,value in [('group',group),('threads',threads),('upscaling',upscaling)]:
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError(name + ' must be a positive integer.')
    if threads > 64:
        raise ValueError('EER decoding supports at most 64 threads.')
    if upscaling not in (1,2):
        raise ValueError('EER upscaling must be 1 or 2.')
    n,w,h,bits = eer_info(path)
    groups = n // group
    if groups < 2:
        raise ValueError('At least two integrated EER frames are needed for even/odd CTF.')
    buf = np.empty(os.path.getsize(path),dtype=np.uint8)
    sizes = (ct.c_int * n)()
    total = ct.c_size_t()
    lib = _lib()
    rc = lib.eer_read_strips(os.fsencode(path),n,ct.c_void_p(buf.ctypes.data),
                            len(buf),sizes,ct.byref(total))
    if rc:
        raise ValueError(f'EER strip read failed ({rc}): {path}')
    sizes_array = np.asarray(list(sizes),dtype=np.int64)
    if np.any(sizes_array <= 0) or sizes_array.sum() != total.value or total.value > len(buf):
        raise ValueError('Invalid EER strip lengths.')
    offsets = np.r_[0,np.cumsum(sizes_array)]
    pointers = (ct.c_void_p * n)(*[buf.ctypes.data+int(offsets[i]) for i in range(n)])
    frames = np.zeros((groups,h*upscaling,w*upscaling),dtype=np.uint8)
    pointer = ct.c_void_p(frames.ctypes.data)
    rc = lib.eer_decode_groups(pointers,sizes,n,group,w,h,bits,
                          pointer if upscaling==1 else None,
                          pointer if upscaling==2 else None,threads)
    if rc:
        raise MemoryError(f'EER decode failed ({rc}); reduce threads or upscaling.')
    # The existing decoder accumulates uint32 then saturates to uint8. Fail
    # rather than silently accept a clipped movie at unusually large grouping.
    if np.any(frames == 255):
        raise ValueError('EER grouping may saturate counts; reduce --eer-group.')
    metadata = dict(format='EER',raw_subframes=n,integrated_frames=groups,
                    group=group,used_subframes=groups*group,
                    dropped_subframes=n-groups*group,encoding_bits=bits,
                    upscaling=upscaling,raster=f'{w*upscaling}×{h*upscaling}, no transform')
    return frames, metadata
