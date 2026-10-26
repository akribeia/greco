"""Minimal MRC I/O (little-endian), numpy-based.

Header layout is the standard 1024-byte MRC header as used by reference
(``src/rwMRC.h``, struct ``MRChead``) and ``mrcfile``:

    off  field
    0    nx (i4)
    4    ny (i4)
    8    nz (i4)
    12   mode (i4)
    16   nxstart (i4)
    20   nystart (i4)
    24   nzstart (i4)
    28   mx (i4)
    32   my (i4)
    36   mz (i4)
    40   cella x,y,z (3 x f4)
    52   cellb alpha,beta,gamma (3 x f4)
    64   mapc (i4)
    68   mapr (i4)
    72   maps (i4)
    76   dmin (f4)
    80   dmax (f4)
    84   dmean (f4)
    88   ispg (i4)
    92   nsymbt (i4)
    96   extra (100 bytes)
    196  origin x,y,z (3 x f4)
    208  map id (4 s)  "MAP "
    212  machst (4 s)
    216  rms (f4)
    220  nlabl (i4)
    224  labels (800 bytes)

Mode numbers follow the reference / old-CCP4 convention (as in reference's
``writeMRC``): 0=8-bit signed, 1=16-bit signed, 2=32-bit float,
4=32-bit complex, 6=16-bit unsigned, 12=16-bit float.

Images are returned as row-major arrays indexed ``[z, y, x]`` where x is
the fastest-changing MRC dimension, matching reference's ``BufferedImage``
(``xdim`` = width, ``ydim`` = height, ``zdim`` = slices).
"""
# Upstream attribution: licenses/THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import os
import struct
import time
from typing import Optional, Tuple

import numpy as np

__all__ = ['MrcImage', 'read_mrc', 'write_mrc', 'MRC_MODES']

MRC_MODES = {
    0: np.dtype("<i1"),
    1: np.dtype("<i2"),
    2: np.dtype("<f4"),
    4: np.dtype("<c8"),
    6: np.dtype("<u2"),
    12: np.dtype("<f2"),
}

_DTYPES_TO_MODE = {
    np.dtype("<i1"): 0,
    np.dtype("<i2"): 1,
    np.dtype("<u2"): 6,
    np.dtype("<f2"): 12,
    np.dtype("<f4"): 2,
    np.dtype("<c8"): 4,
}

MRC_SIZE = 1024


def _decode_float16(f16: np.ndarray) -> np.ndarray:
    """``half2float`` (src/float16.h:145), bit-exact.

    reference does NOT convert f16 subnormals (literal TODO in the code):
    any value with f16 exponent field 0 is flushed to its signed zero
    (even subnormal NaN bit patterns).  For exp==31 the 10-bit payload
    is shifted verbatim to bit 13 (GT keeps NaN payloads; numpy would
    canonicalise them).  Normals convert exactly.
    """
    f16 = np.ascontiguousarray(f16, dtype=np.float16)
    # promote to uint32 up front: numpy keeps uint16 dtype through
    # shifts, which would silently truncate (e.g. 0x03FF << 13)
    u = f16.view(np.uint16).astype(np.uint32)
    bits = np.zeros(f16.shape, dtype=np.uint32)
    bits |= (u & 0x8000) << 16
    exp = (u & 0x7C00) >> 10
    normal = (exp != 0) & (exp != 31)
    bits[normal] |= ((exp[normal] + 112) & 0x1FF) << 23
    bits[normal] |= (u[normal] & 0x03FF) << 13
    naninf = exp == 31
    bits[naninf] |= np.uint32(255) << 23
    bits[naninf] |= (u[naninf] & 0x03FF) << 13
    # exp == 0: stays signed zero (already zero)
    return bits.view(np.float32).copy()


def _encode_float16(f32: np.ndarray) -> np.ndarray:
    """``float2half`` (src/float16.h:97), bit-exact.

    Rounding is round-half-UP (``fractional += 1<<12``), NOT round-to-
    nearest-even; overflow clamps to the f16 max normal with the sign
    preserved (no Inf); f32 subnormals/underflow -> signed zero;
    NaN/Inf keep the truncated payload and sign.
    """
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    u = f32.view(np.uint32)
    sign = (u & 0x80000000) >> 16
    exp = (u & 0x7F800000) >> 23
    frac = u & 0x007FFFFF
    out = sign.astype(np.uint16)  # signed zero (covers exp==0)
    naninf = exp == 255
    if naninf.any():
        out[naninf] |= 0x7C00
        out[naninf] |= (frac[naninf] >> 13).astype(np.uint16)
    normal = (exp != 0) & (exp != 255)
    if normal.any():
        idx = np.nonzero(normal)[0]
        f2 = frac[idx].astype(np.uint32) + 0x1000
        carry = (f2 & 0x800000) != 0
        f2 = np.where(carry, f2 & 0x7FFFFF, f2)
        e2 = exp[idx].astype(np.int64) + carry.astype(np.int64)
        outv = (
            (sign[idx].astype(np.uint16))
            | (((e2 + 15 - 127) & 0x1F).astype(np.uint16) << 10)
            | ((f2 >> 13).astype(np.uint16))
        )
        ovf = e2 > 142
        outv[ovf] = np.where(sign[idx][ovf] != 0, 0xFBFF, 0x7BFF)
        outv[e2 < 113] = sign[idx][e2 < 113].astype(np.uint16)  # underflow
        out[idx] = outv
    return out.view(np.float16).copy()


_C_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def application_label() -> str:
    """Greco application label with a local timestamp and English month name."""
    t = time.localtime()
    return "Greco    %02d-%s-%02d  %02d:%02d:%02d" % (
        t.tm_mday, _C_MONTHS[t.tm_mon - 1], t.tm_year % 100,
        t.tm_hour, t.tm_min, t.tm_sec)


def _seq_stats(flat: np.ndarray) -> Tuple[float, float, float, float]:
    """MRC header statistics using bounded NumPy sequential accumulations.

    Preserve the existing file convention and summation order without a
    compiled project extension. Seeding each block carries rounding across
    block boundaries; temporary storage is bounded for large tomograms.
    """
    values = np.asarray(flat).reshape(-1)
    n = values.size
    if not n:
        return 0.0, 0.0, 0.0, 0.0
    total = squares = 0.0
    minimum = maximum = float(values[0])
    for first in range(0, n, 65536):
        block = values[first:first + 65536].astype(np.float64)
        work = np.empty(block.size + 1, dtype=np.float64)
        work[0] = total
        work[1:] = block
        np.cumsum(work, out=work)
        total = float(work[-1])
        work[0] = squares
        np.multiply(block, block, out=work[1:])
        np.cumsum(work, out=work)
        squares = float(work[-1])
        # Ignore later NaNs for min/max, matching the existing comparisons.
        if not np.isnan(minimum):
            minimum = min(minimum, float(np.fmin.reduce(block)))
            maximum = max(maximum, float(np.fmax.reduce(block)))
    if n == 1:
        return total, 0.0, minimum, maximum
    average = total / n
    variance = (squares / n - average * average) * (n // (n - 1))
    return average, float(np.sqrt(abs(variance))), minimum, maximum


class MrcImage:
    def __init__(self, data: np.ndarray, header: dict, title: str = "",
                 path: Optional[str] = None):
        self.data = data                      # (nz, ny, nx)
        self.header = header
        self.title = title
        self.path = path

    @property
    def xdim(self) -> int:
        return self.data.shape[-1]

    @property
    def ydim(self) -> int:
        return self.data.shape[-2]

    @property
    def zdim(self) -> int:
        return self.data.shape[0]

    @property
    def pixel_size(self) -> float:
        return float(self.header.get("cell_a", 1.0))

    def __repr__(self):  # pragma: no cover
        return f"MrcImage({self.zdim}x{self.ydim}x{self.xdim}, mode {self.header.get('mode')})"


def _pack_header(nx: int, ny: int, nz: int, mode: int, pix: float,
                 dmin: float, dmax: float, dmean: float, arms: float,
                 title: str = "", stack: bool = False,
                 pixel_size_y: Optional[float] = None,
                 frame_nz: Optional[int] = None,
                 pixel_size_z: Optional[float] = None) -> bytes:
    # GT rwMRC.h:454-457: mz and c follow the in-memory frame
    # (Ndim/Zdim), not the stack's nz; WRITE_APPEND bumps only nz
    # to replaceNsize + 1 (rwMRC.h:483), leaving mz/c untouched.
    frame_nz = nz if frame_nz is None else frame_nz
    h = bytearray(MRC_SIZE)
    u = struct.Struct("<i")
    f = struct.Struct("<f")

    def put_u(off: int, val: int) -> None:
        h[off:off + 4] = u.pack(val)

    def put_f(off: int, val: float) -> None:
        h[off:off + 4] = f.pack(val)

    put_u(0, nx); put_u(4, ny); put_u(8, nz); put_u(12, mode)
    put_u(16, 0); put_u(20, 0); put_u(24, 0)
    put_u(28, nx); put_u(32, ny); put_u(36, frame_nz)
    # GT: RawImage::write(fn, double pixelSize) calls
    # Image::setSamplingRateInHeader(RFLOAT rate_x) -- the double is
    # narrowed to float at the call site, and the MDC->MRC writer
    # computes the cell dimension as (float)rate_x * xdim in f32.
    # Reproduce that narrowing (f64 pix * int differs by 1 ulp for
    # pixels like 0.96).
    pix32 = float(np.float32(pix))
    pix32y = float(np.float32(pixel_size_y)) if pixel_size_y is not None else pix32
    put_f(40, pix32 * nx); put_f(44, pix32y * ny)
    if pixel_size_z is not None:
        # GT writeMRC (rwMRC.h:440-442): EMDL_IMAGE_SAMPLINGRATE_Z
        # present -> c = (float)rate_z * header->nz (f32 multiply,
        # header->nz being the pre-APPEND-bump Ndim/Zdim).
        put_f(48, float(np.float32(pixel_size_z)) * frame_nz)
    elif stack:
        # .mrcs: per-frame write (Ndim frames, ZSIZE == 1 for 2D);
        # setSamplingRateInHeader leaves SAMPLINGRATE_Z unset, so
        # rwMRC keeps header->c = frame->nz (1.0 for 2D frames).
        put_f(48, float(frame_nz))
    else:
        # setSamplingRateInHeader(rate_x) only sets SAMPLINGRATE_Z when
        # ZSIZE > 1; otherwise writeMRC keeps the initial header->c = nz.
        put_f(48, pix32 * frame_nz if frame_nz > 1 else float(frame_nz))
    put_f(52, 90.0); put_f(56, 90.0); put_f(60, 90.0)
    put_u(64, 1); put_u(68, 2); put_u(72, 3)
    put_f(76, dmin); put_f(80, dmax); put_f(84, dmean)
    put_u(88, 0); put_u(92, 0)
    # GT writeMRC (rwMRC.h:423-443): origin = (float)(nxStart * aux2)
    # with nxStart = 0 -> f64 product, f32 narrowing: -0.0f when the
    # per-axis sampling rate is negative; axis without a rate (z when
    # pixel_size_z is None) keeps the initial +0.0f.
    def _orig(rate):
        if rate is None:
            return 0.0
        v = 0.0 * float(rate)
        return float(np.float32(v)) if np.isfinite(v) else 0.0

    put_f(196, _orig(pix))
    put_f(200, _orig(pixel_size_y if pixel_size_y is not None else pix))
    put_f(204, _orig(pixel_size_z))
    h[208:212] = b"MAP "
    h[212:216] = b"DA\x00\x00"
    put_f(216, arms)
    # rwMRC.h:447-466: nlabl is ALWAYS 1 with the "reference <ver>   <date>"
    # label (PACKAGE_VERSION undefined in the cb2683 build).
    lab = (title if title else application_label()).encode("ascii", "replace")[:79]
    put_u(220, 1)
    h[224:224 + len(lab)] = lab
    return bytes(h)


_GT_IMAGE_H = ("greco-source/"
               "numerical-reference/src/image.h")


def gt_openfile_precheck(name: str) -> None:
    """GT ``Image::openFile`` (image.h:231) filename semantics for reads.

    Emulates the GT file-path derivation and the open-failure errors so
    the port fails exactly like GT (same message, file and line):

    * no extension (and no ``:fmt``) -> treated as SPIDER (``name + ".spi"``)
    * base file missing              -> image.h:276
      "Cannot read file <file> It does not exist"
    * failed data fopen              -> image.h:319
      "Image::openFile cannot open: <original name>"
    * failed header fopen            -> image.h:342
      "Image::openFile cannot open: <head>"
    """
    # FileName::getFileFormat (filename.cpp:321)
    if "#" in name:
        ext_name = "raw"
    elif ":" in name:
        ext_name = name.rsplit(":", 1)[1].lower()
    elif "." in name:
        ext_name = name.rsplit(".", 1)[1].lower()
    else:
        ext_name = ""

    # FileName::decompose (filename.cpp:95): first '@', all chars
    # before it must be digits ("2@inputs/x.mrcs" -> "inputs/x.mrcs").
    base = name
    at = base.find("@")
    if at > 0 and base[:at].isdigit():
        base = base[at + 1:]
    # FileName::removeFileFormat (filename.cpp:337): cut at last ":"
    if ":" in base:
        base = base[: base.rfind(":")]
    # openFile: cut at the first "%"
    pct = base.find("%")
    if pct != -1:
        base = base[:pct]

    if not os.path.exists(base):
        from ..core.errors import numerical_error
        numerical_error(_GT_IMAGE_H, 276,
                     f"Cannot read file {base} It does not exist")

    head_name = ""
    if "img" in ext_name or "hed" in ext_name:
        dot = base.rfind(".")
        wo = base if dot == -1 else base[:dot]
        base = wo + ".img"
        head_name = wo + ".hed"
    elif ext_name == "":
        # "SPIDER is default format if none is specified" (image.h:301)
        base = base + ".spi"

    from ..core.errors import numerical_error
    if "tif" in ext_name:
        try:
            fh = open(base, "rb")
        except OSError:
            numerical_error(_GT_IMAGE_H, 314,
                         f"Image::openFile cannot open: {name}")
        else:
            fh.close()
    else:
        try:
            fh = open(base, "rb")
        except OSError:
            numerical_error(_GT_IMAGE_H, 319,
                         f"Image::openFile cannot open: {name}")
        else:
            fh.close()
    if head_name != "":
        try:
            fh = open(head_name, "rb")
        except OSError:
            numerical_error(_GT_IMAGE_H, 342,
                         f"Image::openFile cannot open: {head_name}")
        else:
            fh.close()


def read_mrc(path: str) -> MrcImage:
    gt_openfile_precheck(path)
    with open(path, "rb") as fh:
        hdr = fh.read(MRC_SIZE)
        nx, ny, nz, mode, dtype = _unpack_header(path, hdr)
        n = nx * ny * nz
        raw = fh.read(n * dtype.itemsize)
        if len(raw) != n * dtype.itemsize:
            raise ValueError(f"{path}: truncated MRC data block")
    data = np.frombuffer(raw, dtype=dtype)
    # MRC stores x fastest: flat = z*ny*nx + y*nx + x -> (nz, ny, nx) C-order
    data = data.reshape(nz, ny, nx).copy()
    if mode == 12:
        # reference reads f16 into f32 via half2float (subnormals -> 0)
        data = _decode_float16(data)
    return MrcImage(data, _header_dict(hdr), title=_title0(hdr), path=path)


def _unpack_header(path: str, hdr: bytes):
    nx, ny, nz, mode = struct.unpack_from("<4i", hdr, 0)
    if mode not in MRC_MODES:
        raise ValueError(f"{path}: unsupported MRC mode {mode}")
    return nx, ny, nz, mode, MRC_MODES[mode]


def _header_dict(hdr: bytes) -> dict:
    return {
        "nx": struct.unpack_from("<i", hdr, 0)[0],
        "ny": struct.unpack_from("<i", hdr, 4)[0],
        "nz": struct.unpack_from("<i", hdr, 8)[0],
        "mode": struct.unpack_from("<i", hdr, 12)[0],
        "mx": struct.unpack_from("<i", hdr, 28)[0],
        "my": struct.unpack_from("<i", hdr, 32)[0],
        "mz": struct.unpack_from("<i", hdr, 36)[0],
        "cell_a": struct.unpack_from("<f", hdr, 40)[0],
        "cell_b": struct.unpack_from("<f", hdr, 44)[0],
        "cell_c": struct.unpack_from("<f", hdr, 48)[0],
        "origin": struct.unpack_from("<3f", hdr, 196),
        "mmin": struct.unpack_from("<f", hdr, 76)[0],
        "mmax": struct.unpack_from("<f", hdr, 80)[0],
        "mmean": struct.unpack_from("<f", hdr, 84)[0],
        "rms": struct.unpack_from("<f", hdr, 216)[0],
    }


def _title0(hdr: bytes) -> str:
    return hdr[224:304].split(b"\0", 1)[0].decode("latin1").strip()


def write_mrc(path: str, data: np.ndarray, pixel_size: float = 1.0,
              mode: Optional[int] = None, title: str = "",
              stack: Optional[bool] = None,
              mstats: Optional[Tuple[float, float, float, float]] = None,
              pixel_size_y: Optional[float] = None,
              pixel_size_z: Optional[float] = None) -> str:
    """Write an MRC file the way reference's ``writeMRC`` does.

    Floating point data (any precision) is stored as 32-bit float
    (mode 2), matching reference which always writes ``Float`` for
    float/double images.

    ``stack`` marks a .mrcs image stack (reference writes it via the MDC
    writer with the frames in the N dimension and no
    EMDL_IMAGE_SAMPLINGRATE_Z, so header c stays nz, i.e. unit z
    spacing); inferred from the path when omitted.

    ``mstats`` overrides the header statistics (min, max, mean, rms):
    reference copies them from the input header (MDMainHeader) when the
    image was read from a file, instead of recomputing them from the
    (transformed) data.
    """
    data = np.ascontiguousarray(data)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim == 4:
        # GT writeMRC(isStack=true) on a 4-D array (Ndim frames of
        # (zdim, ydim, xdim)): header nz = mz = c = Ndim, a/b = nx/ny
        # (no sampling rates in MDMainHeader).
        nframes, nz, ny, nx = data.shape
    elif data.ndim == 3:
        nz, ny, nx = data.shape
        nframes = 1
    else:
        raise ValueError(f"expected 2D, 3D or 4D data, got {data.ndim}D")
    packed_src = data
    # reference's writeMRC sets amin/amax/amean/arms from data.computeMin/Max/
    # Avg/Stddev on the SOURCE data type (e.g. double) before converting to
    # the file dtype; compute the stats on the original data first.
    if mstats is None:
        dmean, arms, dmin, dmax = _seq_stats(data.reshape(-1))
    else:
        dmin, dmax, dmean, arms = (float(v) for v in mstats)
    if stack is None:
        stack = str(path).lower().endswith('.mrcs')
    if data.ndim == 4 and not stack:
        raise ValueError("4D data requires stack=True")
    if mode is None:
        if data.dtype not in _DTYPES_TO_MODE:
            if np.issubdtype(data.dtype, np.complexfloating):
                data = data.astype("<c8")
            elif np.issubdtype(data.dtype, np.floating):
                data = data.astype("<f4")
            elif np.issubdtype(data.dtype, np.signedinteger):
                data = data.astype("<i2")
            else:
                raise ValueError(f"cannot infer MRC mode from {data.dtype}")
        mode = _DTYPES_TO_MODE[data.dtype]
    if mode == 12:
        # reference writes f16 via float2half (half-up rounding, no Inf)
        packed = _encode_float16(
            data.reshape(-1).astype(np.float32)).view(np.uint16)
    else:
        data = data.astype(MRC_MODES[mode])
        packed = data.reshape(-1)
    hdr = _pack_header(nx, ny, nframes if data.ndim == 4 else nz, mode,
                       pixel_size, dmin, dmax, dmean, arms, title,
                       stack=stack, pixel_size_y=pixel_size_y,
                       pixel_size_z=pixel_size_z)
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(packed.tobytes())
    return path
