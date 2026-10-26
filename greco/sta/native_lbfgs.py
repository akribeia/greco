"""Native Greco libLBFGS with independent particles and batched evaluations.

The only platform-specific component is a Linux ucontext scheduler. Native
optimizers yield to the caller thread, so CUDA never runs in worker threads.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import numpy as np

_lock=threading.Lock()
_library=None


def library():
    global _library
    if _library is not None:return _library
    with _lock:
        if _library is not None:return _library
        source=Path(__file__).parent/'vendor/liblbfgs'
        digest=hashlib.sha256(b''.join((source/name).read_bytes() for name in
                             ('lbfgs.c','lbfgs.h','arithmetic_ansi.h','batch.c'))).hexdigest()[:16]
        target=Path(__file__).resolve().parents[1]/'native'/('pose_optimizer_'+digest+'.so')
        if not target.exists():
            temporary=None
            try:
                fd,name=tempfile.mkstemp(prefix='.pose-optimizer-',suffix='.so',dir=target.parent)
                os.close(fd)
                temporary=Path(name)
                subprocess.run(['cc','-O3','-fPIC','-shared','-std=gnu11','-fno-fast-math','-ffp-contract=off',
                                '-DLBFGS_FLOAT=64',str(source/'batch.c'),str(source/'lbfgs.c'),'-lm','-o',str(temporary)],
                               check=True,capture_output=True,text=True,timeout=90)
                temporary.chmod(0o755)
                temporary.replace(target)
            except (OSError,subprocess.SubprocessError) as error:
                raise RuntimeError('Cannot build native Greco libLBFGS: '+str(getattr(error,'stderr',error))) from error
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
        lib=ctypes.CDLL(str(target));handle=ctypes.c_void_p
        doubles=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        lib.greco_pose_geometry.argtypes=[doubles,ctypes.c_int,doubles,ints,ctypes.c_int,doubles,doubles,doubles]
        lib.greco_pose_geometry.restype=ctypes.c_int
        lib.greco_lbfgs_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int]
        lib.greco_lbfgs_create.restype=handle
        lib.greco_lbfgs_requests.argtypes=[handle,doubles,ints];lib.greco_lbfgs_requests.restype=ctypes.c_int
        lib.greco_lbfgs_advance.argtypes=[handle,doubles,doubles];lib.greco_lbfgs_advance.restype=None
        lib.greco_lbfgs_diagnostics.argtypes=[handle,ints,ints,ints];lib.greco_lbfgs_diagnostics.restype=None
        lib.greco_lbfgs_destroy.argtypes=[handle];lib.greco_lbfgs_destroy.restype=None
        _library=lib
        return lib


def pose_geometry_native(parameters,base,owner):
    p=np.ascontiguousarray(parameters,dtype=np.float64);b=np.ascontiguousarray(base,dtype=np.float64)
    owners=np.ascontiguousarray(owner,dtype=np.int32);nv=len(owners)
    if p.ndim!=2 or p.shape[1]!=6 or b.shape!=(nv,3,3) or (nv and (owners.min()<0 or owners.max()>=len(p))):
        raise ValueError('Invalid native pose geometry shapes or owners.')
    inverse=np.empty((nv,3,3));derivatives=np.empty((nv,3,3,3));shift=np.empty((nv,3))
    if library().greco_pose_geometry(p,len(p),b,owners,nv,inverse,derivatives,shift):
        raise MemoryError('Cannot allocate native pose geometry.')
    return inverse,derivatives,shift


def minimize_independent(evaluate,count,maxiter=25,epsilon=1e-5,xtol=1e-4,active_callback=False,linesearch=0):
    """evaluate(P×6) -> P costs, P×6 gradients; starts each problem at zero."""
    if linesearch not in (0,1,2,3):raise ValueError('Invalid native line-search method.')
    lib=library();handle=lib.greco_lbfgs_create(count,maxiter,epsilon,xtol,linesearch)
    if not handle:raise MemoryError('Cannot allocate independent libLBFGS contexts.')
    x=np.zeros((count,6),np.float64);active=np.empty(count,np.int32)
    statuses=np.empty(count,np.int32);iterations=np.empty(count,np.int32);evaluations=np.empty(count,np.int32)
    rounds=0
    try:
        while lib.greco_lbfgs_requests(handle,x,active):
            cost,gradient=evaluate(x,active!=0) if active_callback else evaluate(x)
            cost=np.ascontiguousarray(cost,dtype=np.float64);gradient=np.ascontiguousarray(gradient,dtype=np.float64)
            if cost.shape!=(count,) or gradient.shape!=(count,6):raise ValueError('Invalid native optimizer callback shapes.')
            bad=(active!=0)&(~np.isfinite(cost)|~np.isfinite(gradient).all(axis=1))
            if bad.any():
                ids=np.flatnonzero(bad)
                raise ValueError(f'Non-finite native optimizer objective/gradient at evaluation {rounds+1}; '
                                 f'particles {ids.tolist()}, poses {x[ids].tolist()}, costs {cost[ids].tolist()}, '
                                 f'gradients {gradient[ids].tolist()}.')
            lib.greco_lbfgs_advance(handle,cost,gradient);rounds+=1
        lib.greco_lbfgs_diagnostics(handle,statuses,iterations,evaluations)
        return x,dict(statuses=statuses,iterations=iterations,evaluations=evaluations,batch_evaluations=rounds)
    finally:lib.greco_lbfgs_destroy(handle)
