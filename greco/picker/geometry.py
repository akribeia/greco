# Third-party references and adapted components
# Selected interpolation and rotation helpers adapt voltools conventions and CUDA Cubic B-Spline Interpolation routines by Danny Ruijters, based on Philippe Thevenaz. Scope: Euler matrix construction, spline_prefilter and cubic interpolation; the Greco picker as a whole is a Greco implementation.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/voltools/LICENSE.txt
# licenses/cubic-interpolation/LICENSE.txt
# licenses/cubic-interpolation/NOTICE.txt
#
# Upstream notice (retained):
# Copyright 2019 Ilja Gubins
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Upstream notice (retained):
# Copyright (c) 2008-2013, Danny Ruijters. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# *  Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# *  Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# *  Neither the name of the copyright holders nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied.
#
# When using this code in a scientific project, please cite one or all of the
# following papers:
# *  Daniel Ruijters and Philippe Thévenaz,
#    GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
#    The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
# *  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
#    Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
#    Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
#
# Upstream notice (retained):
# /*--------------------------------------------------------------------------*\
# Copyright (c) 2008-2012, Danny Ruijters. All rights reserved.
# http://www.dannyruijters.nl/cubicinterpolation/
# This file is part of CUDA Cubic B-Spline Interpolation (CI).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# *  Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# *  Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# *  Neither the name of the copyright holders nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied.
#
# When using this code in a scientific project, please cite one or all of the
# following papers:
# *  Daniel Ruijters and Philippe Thévenaz,
#    GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
#    The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
#    http://dannyruijters.nl/docs/cudaPrefilter3.pdf
# *  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
#    Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
#    Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
# \*--------------------------------------------------------------------------*/
#

"""Geometry helpers: angular search grid and volume rotations.

Rotation conventions replicate voltools.transform exactly (rotation_order
'rzxz'/'ryxz', CCW angle notation, backward mapping around a center), so that
orientations exported to STAR files are directly comparable with Greco picker runs.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import affine_transform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# voltools-compatible Euler matrix construction
# ---------------------------------------------------------------------------

_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0), "sxyx": (0, 0, 1, 0), "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0), "syzx": (1, 0, 0, 0), "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0), "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0), "szxz": (2, 0, 1, 0), "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0), "rzyx": (0, 0, 0, 1), "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1), "rxzx": (0, 1, 1, 1), "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1), "rzxy": (1, 1, 0, 1), "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1), "rzxz": (2, 0, 1, 1), "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}


def _translation_matrix(translation, dtype=np.float32):
    m = np.identity(4, dtype=dtype)
    m[:3, 3] = -np.asarray(translation, dtype=dtype)[:3]
    return m


def rotation_matrix(rotation, rotation_units="rad", rotation_order="rzxz",
                    dtype=np.float32):
    """Build the same 4x4 rotation matrix as voltools.utils.rotation_matrix."""
    if rotation_order not in _AXES2TUPLE:
        raise ValueError(f"Unknown rotation order: {rotation_order}")
    if rotation_units == "deg":
        ai, aj, ak = np.deg2rad(rotation)[:3]
    else:
        ai, aj, ak = rotation[:3]

    # CCW notation
    ai, aj, ak = -ai, -aj, -ak

    firstaxis, parity, repetition, frame = _AXES2TUPLE[rotation_order]
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = np.sin([ai, aj, ak])
    ci, cj, ck = np.cos([ai, aj, ak])
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    m = np.identity(4, dtype=dtype)
    if repetition:
        m[i, i] = cj
        m[i, j] = sj * si
        m[i, k] = sj * ci
        m[j, i] = sj * sk
        m[j, j] = -cj * ss + cc
        m[j, k] = -cj * cs - sc
        m[k, i] = -sj * ck
        m[k, j] = cj * sc + cs
        m[k, k] = cj * cc - ss
    else:
        m[i, i] = cj * ck
        m[i, j] = sj * sc - cs
        m[i, k] = sj * cc + ss
        m[j, i] = cj * sk
        m[j, j] = sj * ss + cc
        m[j, k] = sj * cs - sc
        m[k, i] = -sj
        m[k, j] = cj * si
        m[k, k] = cj * ci
    return m


def transform_matrix(rotation=None, rotation_units="rad", rotation_order="rzxz",
                     center=None, dtype=np.float32):
    """Composed transform T(center) R T(-center) as in voltools.transform_matrix."""
    m = np.identity(4, dtype=dtype)
    if center is not None:
        m = m @ _translation_matrix(tuple(-1 * i for i in center), dtype)
    if rotation is not None:
        m = m @ rotation_matrix(rotation, rotation_units, rotation_order, dtype)
    if center is not None:
        m = m @ _translation_matrix(tuple(center), dtype)
    return m / m[3, 3]


# ---------------------------------------------------------------------------
# CPU rotation (used for wedge plane construction; matches voltools cpu path)
# ---------------------------------------------------------------------------

def rotate_cpu(volume, rotation, rotation_units="rad", rotation_order="rzxz",
               center=None, order=3):
    """Rotate a volume with scipy, replicating voltools.transform on CPU.

    order=3 + prefilter replicates 'filt_bspline'; order=1 gives trilinear.
    """
    if center is None:
        center = np.divide(np.subtract(volume.shape, 1), 2, dtype=np.float32)
    m = transform_matrix(
        rotation=rotation,
        rotation_units=rotation_units,
        rotation_order=rotation_order,
        center=center,
    )
    return affine_transform(
        volume, m, output_shape=volume.shape, order=order, prefilter=(order > 1)
    )


def spline_prefilter(volume):
    """Compute cubic B-spline coefficients exactly like voltools' GPU path.

    Faithful port of Ruijters'/Thévenaz' ConvertToInterpolationCoefficients
    (the SamplesToCoefficients kernels voltools runs on the GPU): clamped
    boundary conditions with Horizon = min(12, n) and pole sqrt(3)-2.
    NOTE: this is NOT identical to scipy.ndimage.spline_filter (mirror
    boundary), which is why it is implemented here explicitly.
    """
    a = np.array(volume, dtype=np.float32, copy=True)
    pole = np.float32(np.sqrt(3.0) - 2.0)
    lam = np.float32((1.0 - pole) * (1.0 - 1.0 / pole))
    neg_pole = np.float32(pole / (pole - 1.0))

    def pass_lines(lines: np.ndarray) -> None:
        """lines: (M, L) float32 C-contiguous; filter along axis 1."""
        m, length = lines.shape
        horizon = min(12, length)
        # causal initialisation (clamped boundary)
        zn = pole ** np.arange(1, horizon + 1, dtype=np.float32)
        s_init = lines[:, 0] * (np.float32(1.0) + pole)
        if horizon > 1:
            s_init += (lines[:, 1:horizon] * zn[1:]).sum(axis=1)
        prev = lam * s_init
        lines[:, 0] = prev
        for k in range(1, length):
            prev = np.float32(lam) * lines[:, k] + pole * prev
            lines[:, k] = prev
        # anticausal initialisation and recursion
        lines[:, length - 1] *= neg_pole
        for k in range(length - 2, -1, -1):
            lines[:, k] = pole * (lines[:, k + 1] - lines[:, k])

    # apply along the fastest axis first, like the CUDA kernels X->Y->Z
    for ax in (a.ndim - 1, a.ndim - 2, a.ndim - 3):
        moved = np.moveaxis(a, ax, -1)
        shp = moved.shape
        lines = np.ascontiguousarray(moved).reshape(-1, shp[-1])
        pass_lines(lines)
        np.copyto(moved, lines.reshape(shp))
    return a


_GRID_CACHE: dict = {}


def rotate_batch_gpu(volumes, angles_rad, rotation_order="rzxz", out=None,
                     interpolation="bspline", matrices=None):
    """Rotate a batch of volumes on the GPU.

    Parameters
    ----------
    volumes : torch.Tensor (B, n, n, n) or (B, C, n, n, n)
        For ``interpolation="bspline"`` these must contain precomputed cubic
        B-spline coefficients (:func:`spline_prefilter`), matching voltools'
        filt_bspline textures.
    angles_rad : torch.Tensor (B, 3) or None
        Rotation tuples per item (radians, rzxz anti-clockwise). May be None
        when ``matrices`` is given.
    out : optional persistent buffer (B, C, n, n, n)
        Voxels whose source coordinate lies outside the volume keep their
        previous value - exactly like voltools reusing its output buffers.
    interpolation : "bspline" (greco_picker parity) or "linear".
    matrices : optional (lin (B, 3, 3) float32, off (B, 3) float32) on device
        Precomputed linear parts and offsets of the transform matrices;
        skips all per-batch CPU work and host<->device roundtrips.

    Returns
    -------
    torch.Tensor with the same spatial shape as ``volumes``.
    """
    import torch

    dev = volumes.device
    if matrices is not None:
        lin_t, off_t = matrices
        squeeze = volumes.dim() == 4
        vols5 = volumes.unsqueeze(1) if squeeze else volumes
        if squeeze and out is not None:
            out = out.unsqueeze(1)
        b, c, n = vols5.shape[:3]
        key = (n, str(dev))
        if key not in _GRID_CACHE:
            g = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                                     indexing="ij"), axis=-1)
            _GRID_CACHE[key] = torch.from_numpy(g.reshape(-1, 3)).to(dev)
        coords = _GRID_CACHE[key].to(torch.float32)          # (P, 3)
        # q = lin @ coords + off  for every batch item, in one bmm
        q_t = torch.bmm(lin_t if lin_t.dim() == 3 else lin_t.unsqueeze(0),
                        coords.T.unsqueeze(0).expand(b, -1, -1))\
            + off_t.unsqueeze(-1)                            # (B, 3, P)
        q_t = q_t.permute(0, 2, 1)                           # (B, P, 3)
        return _bspline_sample(vols5, q_t, n, out=out, squeeze=squeeze)

    if interpolation != "bspline":
        dev = volumes.device
        squeeze = volumes.dim() == 4
        vols4 = volumes.unsqueeze(1) if squeeze else volumes
        b, n = vols4.shape[0], vols4.shape[-1]
        center = np.divide(np.subtract((n, n, n), 1), 2, dtype=np.float64)
        coords = np.stack(
            np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                        indexing="ij"), axis=-1,
        ).astype(np.float64).reshape(-1, 3)
        mats = [
            transform_matrix(rotation=tuple(a), rotation_units="rad",
                             rotation_order=rotation_order, center=center)
            for a in angles_rad.detach().cpu().numpy()
        ]
        lin = np.stack([mm[:3, :3] for mm in mats])
        off = np.stack([mm[:3, 3] for mm in mats])
        q = np.einsum("bij,nj->bni", lin, coords) + off[:, None, :]
        grids = ((q - center[None, None, :]) * 2.0 / (n - 1)).astype(np.float32)
        grid = torch.from_numpy(grids.reshape(b, n, n, n, 3)).to(dev)
        res = torch.nn.functional.grid_sample(
            vols4, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )
        return res.squeeze(1) if squeeze else res

    # ---------------- cubic B-spline (Ruijters fast cubic filtering) -------
    dev = volumes.device
    squeeze = volumes.dim() == 4
    if squeeze:
        volumes = volumes.unsqueeze(1)
        if out is not None:
            out = out.unsqueeze(1)
    b, c, n = volumes.shape[:3]

    key = (n, str(dev))
    if key not in _GRID_CACHE:
        g = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                                 indexing="ij"), axis=-1)
        _GRID_CACHE[key] = torch.from_numpy(g.reshape(-1, 3)).to(dev)
    coords_np = _GRID_CACHE[key].cpu().numpy()  # (P, 3) int
    coords = _GRID_CACHE[key]

    center = np.divide(np.subtract((n, n, n), 1), 2, dtype=np.float64)
    mats = [
        transform_matrix(rotation=tuple(a), rotation_units="rad",
                         rotation_order=rotation_order, center=center)
        for a in angles_rad.detach().cpu().numpy()
    ]
    lin = np.stack([mm[:3, :3].astype(np.float32) for mm in mats])
    off = np.stack([mm[:3, 3].astype(np.float32) for mm in mats])
    q = np.einsum("bij,pj->bpi", lin, coords_np.astype(np.float32))\
        + off[:, None, :]
    q_t = torch.from_numpy(q).to(dev)                       # (B, P, 3)

    return _bspline_sample(volumes, q_t, n, out=out, squeeze=squeeze)


def _bspline_sample(volumes, q_t, n, out=None, squeeze=False):
    """Cubic B-spline sampling of 5D ``volumes`` at positions ``q_t``.

    Ruijters' fast cubic filtering: two trilinear fetches per axis via
    grid_sample, exact for beta3 spline weights (Sigg & Hadwiger).
    """
    import torch

    b, c = volumes.shape[:2]

    # Ruijters' bspline_weights + texel-space fetch positions.
    # The CUDA kernel adds +0.5 when building ndx and cubicTex3D subtracts
    # it again (coord_grid = raw coordinate), so no half-texel shift here.
    cg = q_t
    ind = torch.floor(cg)
    frac = cg - ind
    w0 = (1 - frac) ** 3 / 6.0
    w1 = 2.0 / 3.0 - 0.5 * frac**2 * (2.0 - frac)
    w2 = 2.0 / 3.0 - 0.5 * (1.0 - frac) ** 2 * (1.0 + frac)
    w3 = frac ** 3 / 6.0
    g0w = w0 + w1
    g1w = w2 + w3
    h0 = w1 / g0w - 0.5 + ind                               # (B, P, 3)
    h1 = w3 / g1w + 1.5 + ind

    # Fast cubic filtering (Ruijters/Sigg-Hadwiger): the exact beta3-weighted
    # sum over the four knots of each axis collapses to two linearly
    # interpolated fetches per axis. torch.grid_sample(align_corners=False)
    # reproduces CUDA's hardware texel fetch exactly (texel centers at
    # half-integers, zeros outside), so the full interpolation is eight
    # trilinear samples combined with the group weights g0/g1 per axis.
    def _sample(pos):
        """Trilinear sample of `volumes` at array-space positions (B, P, 3)."""
        x = (pos[..., 2] + 0.5) / n * 2.0 - 1.0
        y = (pos[..., 1] + 0.5) / n * 2.0 - 1.0
        z = (pos[..., 0] + 0.5) / n * 2.0 - 1.0
        grid = torch.stack((x, y, z), dim=-1).reshape(b, -1, 1, 1, 3)
        s = torch.nn.functional.grid_sample(
            volumes, grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        return s.reshape(b, c, -1)

    pairs = (
        (h0[..., 0] - 0.5, h1[..., 0] - 0.5, g0w[..., 0], g1w[..., 0]),
        (h0[..., 1] - 0.5, h1[..., 1] - 0.5, g0w[..., 1], g1w[..., 1]),
        (h0[..., 2] - 0.5, h1[..., 2] - 0.5, g0w[..., 2], g1w[..., 2]),
    )
    acc = None
    for sx in (0, 1):
        px, wx_ = pairs[0][sx], pairs[0][sx + 2]
        for sy in (0, 1):
            py, wy_ = pairs[1][sy], pairs[1][sy + 2]
            for sz in (0, 1):
                pz, wz_ = pairs[2][sz], pairs[2][sz + 2]
                pos = torch.stack((px, py, pz), dim=-1)
                w = (wx_ * wy_ * wz_).reshape(b, 1, -1)
                term = _sample(pos) * w
                acc = term if acc is None else acc + term

    res = acc.reshape(b, c, n, n, n)

    if out is not None:
        # voltools' kernel skips writing voxels whose source coordinate falls
        # outside [-0.5, n-0.5) after the texel half-shift; those keep stale
        # values from previous iterations.
        inb = ((q_t >= -0.5) & (q_t < n - 0.5)).all(dim=-1)      # (B, P)
        res = torch.where(inb.reshape(b, 1, n, n, n), res, out)
        out.copy_(res)
        return out.squeeze(1) if squeeze else out
    return res.squeeze(1) if squeeze else res


# ---------------------------------------------------------------------------
# Deterministic angular search grid; historical provenance is retained in notices.
# ---------------------------------------------------------------------------

def angle_to_angle_list(angle_diff: float, sort_angles: bool = True):
    """Build the deterministic HEALPix × uniform-spin ZXZ orientation bank.

    Historical sampling and indexing are retained; see third-party notices.
    """
    import healpix
    target_pixels = 4 * np.pi / (angle_diff * np.pi / 180) ** 2
    # HEALPix has 12*nside**2 cells. Correct the rounded estimate using the
    # library's integer cell count, including exact boundary requests.
    nside = max(1, int(np.ceil(np.sqrt(target_pixels / 12))))
    while nside > 1 and healpix.nside2npix(nside - 1) >= target_pixels:
        nside -= 1
    while healpix.nside2npix(nside) < target_pixels:
        nside += 1
    used_pixels = healpix.nside2npix(nside)
    logger.info(f"Using an angle difference of {(4 * np.pi / used_pixels) ** .5 * (180 / np.pi):.4f} for Z1 and X")
    logger.info(f"Using an angle difference of {np.rad2deg(2 * np.pi / int(np.ceil(360 / angle_diff))):.4f} for Z2")
    colatitude, longitude = healpix.pix2ang(nside, np.arange(healpix.nside2npix(nside)))
    longitude %= 2 * np.pi
    colatitude %= np.pi
    spin = np.linspace(0, 2 * np.pi, int(np.ceil(360 / angle_diff)), endpoint=False)
    # C order makes the last Euler component vary fastest. Sorting uses the
    # same lexicographic order as existing saved orientation indices.
    bank = np.empty((len(longitude), len(spin), 3), dtype=np.float64)
    bank[:, :, 0] = longitude[:, None]
    bank[:, :, 1] = colatitude[:, None]
    bank[:, :, 2] = spin[None, :]
    result = list(map(tuple, bank.reshape(-1, 3)))
    if sort_angles:
        result.sort()
    return result



def get_angle_list(angle: float, sort_angles: bool = True, symmetry: int = 1):
    """Generate the rotation search list; restricts Z1 range by symmetry."""
    out = angle_to_angle_list(angle, sort_angles=sort_angles)
    max_z1 = 2 * np.pi / symmetry
    return [i for i in out if i[0] < max_z1]


def convert_euler(angles, order_in="ZXZ", order_out="ZXZ", degrees_in=True,
                  degrees_out=True):
    """Convert Euler conventions through SciPy Rotation."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_euler(order_in, angles, degrees=degrees_in)
    return tuple(r.as_euler(order_out, degrees=degrees_out))


# ---------------------------------------------------------------------------
# Greco Picker-2 hybrid: orientation-subset (FPS) selection
# ---------------------------------------------------------------------------


def fps_subset(angle_list, k):
    """Greedy farthest-point (maximin) subset of ``angle_list`` in SO(3).

    Seed is index 0; each next point maximises the minimum geodesic
    distance to the already-selected set (monotone in 1-|q1.q2|, so the
    maximin on the squared chord equals the geodesic maximin).  The
    result is a function of ``angle_list`` ONLY: deterministic,
    byte-identical across runs/machines, and equal to the first ``k``
    ids of the canonical 7776 sequence (SHA-pinned by the research
    audit, Round-6F.0).  Returns the selected indices (int64, in
    selection order).
    """
    from scipy.spatial.transform import Rotation

    angles = np.asarray(angle_list, dtype=np.float64)
    m = len(angles)
    if k <= 0:
        raise ValueError("k must be positive")
    if k > m:
        raise ValueError(f"k={k} exceeds the angle-list size {m}")
    quats = Rotation.from_euler("ZXZ", angles, degrees=False).as_quat()
    sel = [0]
    min_d = np.full(m, np.inf)
    for _ in range(1, k):
        q = quats[sel[-1]][None, :]
        d2 = 1.0 - np.clip(np.abs(quats @ q.T).ravel(), -1.0, 1.0)
        np.minimum(min_d, d2, out=min_d)
        min_d[sel] = -1.0
        sel.append(int(np.argmax(min_d)))
    return np.asarray(sel, dtype=np.int64)
