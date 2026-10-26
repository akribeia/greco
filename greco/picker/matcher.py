"""Batched GPU template matching engine.

Applies the picker weighting and search conventions with batched cuFFT
transforms through PyTorch:

- tomogram Fourier transforms (value and squared value) are computed once;
- every iteration processes a batch of B rotations: templates, randomized
  templates (optional noise correction) and masks are rotated in a single
  grid_sample call and transformed with batched rfft/irfft;
- locally normalised cross-correlation (Roseman) with masked local standard
  deviation, missing-wedge weighting and per-tilt dose/CTF weighting;
- running maximum score/angle maps stay on the GPU between iterations.

Axis convention follows Greco picker: arrays are indexed [x, y, z] where x is the
first axis of the MRC data as stored.
"""

from __future__ import annotations
from greco import __version__
from greco.project_layout import join as storage_path

import json
import logging
import math
import os
import pathlib
import time

import numpy as np
import torch
from scipy.fft import next_fast_len

from .geometry import get_angle_list, rotate_batch_gpu
from .io import read_mrc, read_mrc_meta_data, write_mrc
from .batching import configure_allocator,fit_batch,probe_batch,noise_fused
from .roi import roi_slices as search_roi_slices
from .weights import (
    create_gaussian_band_pass,
    create_wedge,
    phase_randomize_template,
)

logger = logging.getLogger(__name__)

__all__ = ["MatchJob", "detect_mask_spherical"]


def detect_mask_spherical(mask: np.ndarray, tol: float = 1e-4) -> bool:
    """True if the mask only depends on radius (rotation invariant)."""
    n = mask.shape[0]
    c = (n - 1) / 2
    coords = np.arange(n) - c
    r = np.sqrt(
        coords[:, None, None] ** 2
        + coords[None, :, None] ** 2
        + coords[None, None, :] ** 2
    )
    keys = np.round(r, 4)
    _, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.bincount(inverse, weights=mask.ravel())
    counts = np.bincount(inverse)
    means = np.zeros_like(sums)
    ok = counts > 0
    means[ok] = sums[ok] / counts[ok]
    approx = means[inverse].reshape(mask.shape)
    return bool(np.max(np.abs(approx - mask)) < tol)


class MatchJob:
    """One template matching search of a full tomogram (optionally sharded).

    The search covers the complete tomogram; defocus offsets therefore vanish
    (patch centre == tomogram centre), matching Greco picker behaviour for full
    volume searches.
    """

    def __init__(
        self,
        tomogram: pathlib.Path,
        template: pathlib.Path,
        mask: pathlib.Path,
        output_dir: pathlib.Path,
        ts_metadata,  # greco-picker.weights.TiltSeriesMetaData
        voxel_size: float,
        angle_increment: float = 10.0,
        rotational_symmetry: int = 1,
        low_pass: float | None = None,
        high_pass: float | None = None,
        noise_correction: bool = False,
        rng_seed: int = 321,
        mask_is_spherical: bool | str = "auto",
        gpu_id: int = 0,
        shard: tuple[int, int] = (0, 1),
        batch_size: int | None = None,
        rotate_mask: bool = False,
        noise_every: int = 10,
        orientation_ids: list[int] | None = None,
        hybrid_cfg: dict | None = None,
        noise_template_iterations: int = 40,
        symmetric_noise: bool = False,
    ):
        self.tomogram = pathlib.Path(tomogram)
        self.template = pathlib.Path(template)
        self.mask = pathlib.Path(mask)
        self.output_dir = pathlib.Path(output_dir)
        self.ts_metadata = ts_metadata
        self.voxel_size = float(voxel_size)
        self.angle_increment = float(angle_increment)
        self.rotational_symmetry = int(rotational_symmetry)
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.noise_correction = bool(noise_correction)
        self.rng_seed = int(rng_seed)
        self.gpu_id = int(gpu_id)
        self.shard_index, self.shard_total = int(shard[0]), int(shard[1])
        self.batch_size = batch_size
        self.noise_template_iterations = noise_template_iterations
        self.symmetric_noise = bool(symmetric_noise)
        if isinstance(noise_template_iterations,bool) or not isinstance(noise_template_iterations,int) or noise_template_iterations<0:
            raise ValueError('Noise template iterations must be a nonnegative integer')
        # speed shortcuts: the local-normalisation mask is only rotated when
        # explicitly requested (an unrotated normalisation is what Greco picker does
        # for spherical masks anyway); noise-correction rotations are
        # subsampled to every `noise_every`-th angle, which approximates
        # Greco picker's noise term — scores and picks shift slightly for N > 1
        self.rotate_mask = bool(rotate_mask)
        self.noise_every = max(1, int(noise_every))
        stem = self.tomogram.name
        self.tomo_id = stem[: -len(".mrc")] if stem.endswith(".mrc") else stem

        meta = read_mrc_meta_data(self.tomogram)
        self.tomo_shape = meta["shape"]
        tmpl_meta = read_mrc_meta_data(self.template)
        mask_meta = read_mrc_meta_data(self.mask)
        if len(set(tmpl_meta["shape"])) != 1:
            raise ValueError("Template must be cubic")
        if tmpl_meta["shape"] != mask_meta["shape"]:
            raise ValueError("Template and mask shapes differ")

        self.angle_list = get_angle_list(
            self.angle_increment, sort_angles=True,
            symmetry=self.rotational_symmetry,
        )
        self.n_rotations = len(self.angle_list)

        if mask_is_spherical == "auto":
            self.mask_is_spherical = detect_mask_spherical(read_mrc(self.mask))
            logger.info(
                f"Mask auto-detected as "
                f"{'spherical' if self.mask_is_spherical else 'non-spherical'}"
            )
        else:
            self.mask_is_spherical = bool(mask_is_spherical)

        # contiguous shard of the rotation list; a hybrid screen instead
        # runs over an explicit (non-contiguous) orientation subset, which
        # is not shardable - the refinement stage needs the full bank
        self.hybrid_mode = orientation_ids is not None
        self.hybrid_cfg = hybrid_cfg
        if self.hybrid_mode:
            if not self.noise_correction:
                raise ValueError(
                    "Hybrid mode requires noise correction (-r): the "
                    "screen noise twin provides the mu control variate"
                )
            if not isinstance(hybrid_cfg, dict):
                raise ValueError("hybrid_cfg (dict) is required in hybrid mode")
            if self.shard_total != 1:
                raise ValueError(
                    "Hybrid (orientation-subset) runs cannot be sharded: "
                    "the full-bank hill climb needs all orientations"
                )
            ids = [int(i) for i in orientation_ids]
            if not ids or min(ids) < 0 or max(ids) >= self.n_rotations:
                raise ValueError(
                    "orientation_ids out of range for the "
                    f"{self.n_rotations}-rotation angle list"
                )
            if len(set(ids)) != len(ids):
                raise ValueError("orientation_ids contain duplicates")
            self.angle_ids = ids
        else:
            n = self.n_rotations
            base, rem = divmod(n, self.shard_total)
            start = self.shard_index * base + min(self.shard_index, rem)
            size = base + (1 if self.shard_index < rem else 0)
            if size <= 0:
                raise ValueError(
                    f"Shard {self.shard_index} of {self.shard_total} "
                    f"received no rotations (total {self.n_rotations}); "
                    f"reduce the shard count"
                )
            self.angle_ids = list(range(start, start + size))

    # ------------------------------------------------------------------
    def _suffix(self) -> str:
        if self.shard_total > 1:
            return f"_p{self.shard_index}of{self.shard_total}"
        return ""

    # ------------------------------------------------------------------
    def run(self) -> dict:
        t_start = time.time()
        configure_allocator()
        dev = torch.device(f"cuda:{self.gpu_id}")
        torch.cuda.set_device(dev)

        # optional per-stage profiling (GRECO_PICKER_PROFILE=1); CUDA sync only
        # happens when profiling is enabled
        from contextlib import contextmanager

        profile = bool(os.environ.get("GRECO_PICKER_PROFILE"))
        timers: dict = {}

        @contextmanager
        def stage(name):
            if profile:
                torch.cuda.synchronize(dev)
                t0 = time.time()
                yield
                torch.cuda.synchronize(dev)
                timers[name] = timers.get(name, 0.0) + time.time() - t0
            else:
                yield

        # ---------------- load and weight the tomogram ----------------
        tomo = read_mrc(self.tomogram)
        fast_shape = tuple(int(next_fast_len(int(s), real=True))
                           for s in tomo.shape)
        self.fast_shape = fast_shape
        tomo_pad = np.zeros(fast_shape, dtype=np.float32)
        tomo_pad[: tomo.shape[0], : tomo.shape[1], : tomo.shape[2]] = tomo

        # tomogram filter: binary wedge * |CTF(middle tilt)| * band-pass
        tomo_filter = create_wedge(
            fast_shape, self.ts_metadata, self.voxel_size,
            cut_off_radius=1.0, per_tilt_weighting=False,
        ).astype(np.float32)
        if not (self.low_pass is None and self.high_pass is None):
            tomo_filter *= create_gaussian_band_pass(
                fast_shape, self.voxel_size, self.low_pass, self.high_pass
            ).astype(np.float32)

        v = torch.from_numpy(tomo_pad).to(dev)
        fv = torch.fft.rfftn(v, dim=(-3, -2, -1))
        del v
        filt_t = torch.from_numpy(tomo_filter).to(dev)
        v_f = torch.fft.irfftn(fv * filt_t, s=fast_shape, dim=(-3, -2, -1))
        del fv, filt_t
        fv_c = torch.fft.rfftn(v_f, dim=(-3, -2, -1)).conj()
        fv2_c = torch.fft.rfftn(v_f**2, dim=(-3, -2, -1)).conj()
        # Match the template FFT layout once for shared spectral products.
        fv_c = fv_c.contiguous()
        fv2_c = fv2_c.contiguous()
        if self.hybrid_mode:
            # the hybrid refine gathers local patches from the filtered
            # volume, so it stays on the GPU for the whole job
            self._hyb_vf = v_f
        else:
            del v_f

        nx, ny, nz = (int(s) for s in self.tomo_shape)

        # ---------------- template-side constants ----------------
        template_np = read_mrc(self.template)
        mask_np = read_mrc(self.mask)
        tmpl_shape = template_np.shape

        wedge_template = create_wedge(
            tmpl_shape, self.ts_metadata, self.voxel_size, cut_off_radius=1.0,
        ).astype(np.float32)
        if not (self.low_pass is None and self.high_pass is None):
            wedge_template *= create_gaussian_band_pass(
                tmpl_shape, self.voxel_size, self.low_pass, self.high_pass
            ).astype(np.float32)
        wedge_template_t = torch.from_numpy(wedge_template).to(dev)

        rand_np = (
            phase_randomize_template(template_np, mask_np, seed=self.rng_seed,n_iter=self.noise_template_iterations)
            if self.noise_correction else None
        )

        mask_weight = float(mask_np.sum())

        # padding geometry (greco_picker convention)
        cx = [s // 2 for s in fast_shape]
        ct = [s // 2 for s in tmpl_shape]
        mo = [s % 2 for s in tmpl_shape]
        pad_index = tuple(
            slice(cx[i] - ct[i], cx[i] + ct[i] + mo[i]) for i in range(3)
        )

        # B-spline coefficient volumes (voltools filt_bspline parity) and the
        # raw volumes used to initialise the persistent rotation buffers
        from .geometry import spline_prefilter

        raws = {
            "tmpl": torch.from_numpy(template_np).to(dev),
            "mask": torch.from_numpy(mask_np).to(dev),
        }
        coeffs = {
            "tmpl": torch.from_numpy(spline_prefilter(template_np)).to(dev),
            "mask": torch.from_numpy(spline_prefilter(mask_np)).to(dev),
        }
        if self.noise_correction:
            coeffs["rand"] = torch.from_numpy(
                spline_prefilter(rand_np)).to(dev)
            raws["rand"] = torch.from_numpy(rand_np).to(dev)

        # precompute every rotation matrix once: per-batch rotation then
        # needs no host roundtrips or CPU transform_matrix calls
        from .geometry import transform_matrix

        _n = tmpl_shape[0]
        _center = np.divide(np.subtract((_n, _n, _n), 1), 2, dtype=np.float64)
        _mats = [
            transform_matrix(rotation=tuple(a), rotation_units="rad",
                             rotation_order="rzxz", center=_center)
            for a in self.angle_list
        ]
        lin_all = torch.from_numpy(np.stack(
            [m[:3, :3].astype(np.float32) for m in _mats])).to(dev)
        off_all = torch.from_numpy(np.stack(
            [m[:3, 3].astype(np.float32) for m in _mats])).to(dev)
        del _mats

        # result maps (kept in the pre-flip STOPGAP frame until the end)
        scores = torch.full(fast_shape, float("-inf"), device=dev)
        angles = torch.full(fast_shape, float("-inf"), device=dev)
        noise_scores = (
            torch.full(fast_shape, float("-inf"), device=dev)
            if self.noise_correction else None
        )
        noise_min = torch.full(fast_shape, float("inf"), device=dev) if self.noise_correction and self.symmetric_noise else None

        # statistics over the unique search region. Greco picker builds a boolean
        # mask of the real region in OUTPUT frame, flips it and rolls it by
        # -shift into the pre-flip correlation frame; the accumulated voxel
        # set differs from a plain [:nx,:ny,:nz] slice whenever
        # fast_shape != tomo_shape (padded FFT sizes), so replicate it.
        # flip+roll of the contiguous [:nx,:ny,:nz] box is the Cartesian
        # product of the per-axis cyclic intervals of length n_i starting
        # at (s_i - n_i + shift_i) mod s_i - a union of at most 8 contiguous
        # sub-boxes.
        # Slicing those sub-boxes covers exactly the same voxel set as the
        # boolean mask (see tests/check_picker_roi_slices.py) while keeping the
        # per-batch energy accumulation a plain strided reduction instead
        # of a b x ROI boolean scatter-gather.
        shift = tuple(int(math.floor(s / 2)) + 1 for s in fast_shape)
        roi_slices=search_roi_slices(fast_shape,(nx,ny,nz))
        roi_size = int(np.prod((nx, ny, nz)))
        sq_acc = torch.zeros((), device=dev, dtype=torch.float64)

        # ---------------- batch size selection ----------------
        # full-size buffers per rotation: padded real volumes (template,
        # plus mask when it is rotated) and their rfftns, correlation output
        # (plus b/c normalisation maps when the mask rotates), and the
        # noise-correction branch amortised over `noise_every`
        p_bytes = int(np.prod(fast_shape)) * 4
        pr_bytes = (int(np.prod(fast_shape[:-1])) *
                    (fast_shape[-1] // 2 + 1)) * 8
        rot_mask_ch = self.rotate_mask and not self.mask_is_spherical
        n_out = 1 + (2 if rot_mask_ch else 0)
        n_out += (1.0 / self.noise_every if self.noise_correction else 0.0)
        # spectrum product temporaries account for one extra pr_bytes; the
        # persistent rotating-mask pad buffer adds one full real volume
        per_item = ((2 if rot_mask_ch else 1) * p_bytes +
                    (3 if rot_mask_ch else 2) * pr_bytes + n_out * p_bytes +
                    (p_bytes if rot_mask_ch else 0))
        if self.batch_size is not None:
            batch = int(self.batch_size)
            if isinstance(self.batch_size,bool) or batch!=self.batch_size or batch<1:
                raise ValueError('FFT batch size must be a positive integer.')
        else:
            free, _ = torch.cuda.mem_get_info(dev)
            budget = free * 0.6
            # Batch 8 balances FFT throughput and memory; retain OOM backoff.
            batch = int(max(1, min(8,len(self.angle_ids),budget // per_item)))
            batch=fit_batch(batch,lambda value:probe_batch(value,fast_shape,dev,
                self.noise_correction,self.noise_every,rot_mask_ch),torch.cuda.empty_cache)
        # Return probe cache before allocating the persistent working set.
        torch.cuda.empty_cache()
        logger.info(
            f"Rotations in this shard: {len(self.angle_ids)} "
            f"(of {self.n_rotations}); FFT batch size {batch}"
        )

        ids = self.angle_ids
        n_done = 0
        t_loop = time.time()
        t_screen0 = time.time()
        # running max stays on the GPU: the previous float(max_ccc.max())
        # synced every FFT batch and drained the pipeline; the host value is
        # now materialised only at the periodic log points
        running_max_gpu = torch.full((), float("-inf"), device=dev)

        # When the noise twin sees EVERY angle (noise_every == 1, greco_picker's
        # exact mode) it can ride along as an extra rotation channel and share
        # every FFT below instead of re-running the whole pipeline separately.
        # Gate on estimated padded-spectrum memory (~2x rows of complex64).
        fuse_noise=noise_fused(batch,fast_shape,dev,self.noise_correction,self.noise_every)

        # Persistent padded FFT buffers: the paste box (pad_index) is the
        # same for every rotation, so only that box is re-zeroed before
        # each paste; the rest of the buffer stays zero for the whole job
        # instead of a torch.zeros() over rows x fast_shape per batch
        # (ribosome ref: ~19.5 GB of memset per FFT batch at batch 58).
        max_rows = 2 * batch if fuse_noise else batch
        pad_buf = torch.zeros((max_rows, *fast_shape), device=dev)
        pad_buf_m = (
            torch.zeros((batch, *fast_shape), device=dev)
            if self.rotate_mask and not self.mask_is_spherical else None
        )
        pad_buf_r = None
        buf_m=buf_r=None

        # ---------------- optional fused post-FFT update ----------
        # One kernel pass, compiled for the current GPU with bundled NVRTC.
        # replaces /std + max/argmax + in-place score/angle update + ROI
        # FP64 energy. Falls back before the loop if compilation is unavailable,
        # or when
        # GRECO_PICKER_NO_FUSED_CORR=1.
        fused_corr_ok = False
        out_max=sq_part=roi_cfg=out_min=None
        if not os.environ.get("GRECO_PICKER_NO_FUSED_CORR"):
            try:
                from . import fused_corr
                fused_corr.warmup()
                out_max = torch.empty(fast_shape, device=dev)
                _nblocks = (fast_shape[0] * fast_shape[1] * fast_shape[2]
                            + fused_corr.BLOCK - 1) // fused_corr.BLOCK
                sq_part = torch.empty((_nblocks,), dtype=torch.float64,
                                      device=dev)
                roi_cfg = fused_corr.make_roi_cfg(
                    fast_shape, self.tomo_shape, dev)
                out_min = (
                    torch.empty(fast_shape, device=dev)
                    if self.noise_correction and self.symmetric_noise else None
                )
                fused_corr_ok = True
            except Exception as exc:  # pragma: no cover
                out_max=sq_part=roi_cfg=out_min=None
                logger.warning(f"fused corr update unavailable: {exc}")

        shared_product_ok=False
        shared_product_count=0
        try:
            from . import spectral_product
            spectral_product.warmup()
            shared_product_ok=True
        except Exception as exc:
            logger.warning(f"Shared spectral product unavailable: {exc}")

        correlation_scale = 1.0 / math.prod(fast_shape) if fused_corr_ok else 1.0
        correlation_norm = "forward" if fused_corr_ok else "backward"
        fixed_std=None
        std_evaluations=0
        for b0 in range(0, len(ids), batch):
            chunk = ids[b0: b0 + batch]
            b = len(chunk)
            ang_ids = torch.tensor(chunk, dtype=torch.float32, device=dev)

            # ---- rotate template (+ randomized twin + mask) in one call ----
            # persistent buffers replicate voltools' reused output volumes:
            # out-of-bounds voxels keep values from previous rotations and
            # start out holding the unrotated volume
            with stage("prep"):
                if b0 == 0:
                    chan_names = ["tmpl"]
                    if self.rotate_mask and not self.mask_is_spherical:
                        chan_names.append("mask")
                    if fuse_noise:
                        chan_names.append("rand")
                    rot_buf = torch.stack(
                        [raws[k].expand(batch, *tmpl_shape) for k in chan_names],
                        dim=1,
                    ).contiguous()  # (batch, C, n, n, n)

                coeff_stack = torch.stack(
                    [coeffs[k].expand(b, *tmpl_shape) for k in chan_names], dim=1,
                )
                sub = None
                if self.noise_correction and not fuse_noise:
                    sel = np.asarray(chunk) % self.noise_every == 0
                    if sel.any():
                        sub = torch.from_numpy(np.nonzero(sel)[0]).to(dev)
            with stage("rotate"):
                rot = rotate_batch_gpu(
                    coeff_stack, None,
                    out=rot_buf[:b],
                    matrices=(lin_all[ang_ids.long()], off_all[ang_ids.long()]),
                )
            del coeff_stack

            # ---- wedge in Fourier space, back to real space ----
            # the wedge applies to density channels (template / randomized
            # twin) but NOT to the normalisation-mask channel
            with stage("wedge_fft"):
                ft = torch.fft.rfftn(rot, dim=(-3, -2, -1))
                wedged = [i for i, k in enumerate(chan_names) if k != "mask"]
                ft[:, wedged] *= wedge_template_t
                rot = torch.fft.irfftn(ft, s=tmpl_shape,
                                       dim=(-3, -2, -1))
                del ft

            # ---- normalise under (optionally rotated) mask ----
            def norm_and_mask(x, m):
                red = (-3, -2, -1)
                mean = (x * m).sum(dim=red) / mask_weight
                var = ((x**2) * m).sum(dim=red) / mask_weight
                std = torch.sqrt(var - mean**2)
                return ((x - mean[..., None, None, None])
                        / std[..., None, None, None]) * m

            rot_mask_idx = (
                chan_names.index("mask")
                if self.rotate_mask and not self.mask_is_spherical else None
            )
            with stage("norm"):
                if rot_mask_idx is not None:
                    m_b = rot[:, rot_mask_idx]           # (b, n, n, n)
                    m_bc = m_b.unsqueeze(1)              # broadcast over chans
                else:
                    m_b = raws["mask"].unsqueeze(0)      # (1, n, n, n)
                    m_bc = m_b.unsqueeze(0)              # (1, 1, n, n, n)
                tw = norm_and_mask(rot, m_bc)
                tp = tw[:, 0]
                rp = tw[:, -1] if fuse_noise else None
                del tw
            del rot

            # ---- paste into padded buffers and transform ----
            with stage("pad_fwd"):
                rows = 2 * b if fuse_noise else b
                buf_t = pad_buf[:rows]
                buf_t[(slice(None),) + pad_index].zero_()
                buf_t[:b][(slice(None),) + pad_index] = tp
                del tp
                if fuse_noise:
                    buf_t[b:][(slice(None),) + pad_index] = rp
                    del rp
                ftp = torch.fft.rfftn(buf_t, dim=(-3, -2, -1))

            # ---- mask correlation maps ----
            if rot_mask_idx is not None:
                with stage("mask_maps"):
                    buf_m = pad_buf_m[:b]
                    buf_m[(slice(None),) + pad_index].zero_()
                    buf_m[(slice(None),) + pad_index] = m_b
                    fm = torch.fft.rfftn(buf_m, dim=(-3, -2, -1))
                    b_map = torch.fft.irfftn(fv_c.unsqueeze(0) * fm, s=fast_shape,
                                             dim=(-3, -2, -1))
                    c_map = torch.fft.irfftn(fv2_c.unsqueeze(0) * fm, s=fast_shape,
                                             dim=(-3, -2, -1))
                    del fm
            elif b0 == 0:
                with stage("mask_maps"):
                    buf_m = torch.zeros((1, *fast_shape), device=dev)
                    buf_m[(0,) + pad_index] = m_b.squeeze(0)
                    fm = torch.fft.rfftn(buf_m.squeeze(0), dim=(-3, -2, -1))
                    b_map = torch.fft.irfftn(fv_c * fm, s=fast_shape,
                                             dim=(-3, -2, -1)).unsqueeze(0)
                    c_map = torch.fft.irfftn(fv2_c * fm, s=fast_shape,
                                             dim=(-3, -2, -1)).unsqueeze(0)
                    del fm
                    buf_m=None

            # local variance -> standard deviation * mask weight
            with stage("std"):
                if rot_mask_idx is not None or fixed_std is None:
                    mw_t = torch.tensor(mask_weight, device=dev)
                    var_map = c_map / mw_t - (b_map / mw_t) ** 2
                    var_map = torch.where(var_map <= 1e-18,
                                          torch.ones_like(var_map), var_map)
                    std_n = torch.sqrt(var_map) * mw_t
                    del var_map, mw_t
                    std_evaluations+=1
                    if rot_mask_idx is None:
                        fixed_std=std_n
                        del b_map,c_map
                else:
                    std_n=fixed_std

            # ---- correlations with the volume ----
            # ---- within-batch reduction, then global max update ----
            with stage("corr_update"):
                # template half and (fused) twin half are transformed
                # separately so std_n broadcasting stays per-angle; the
                # spectrum product is done in place to skip one b x pr
                # complex temporary per batch
                if shared_product_ok and spectral_product.multiply_(ftp,fv_c):
                    shared_product_count+=1
                else:
                    ftp.mul_(fv_c.unsqueeze(0))
                ccc = torch.fft.irfftn(ftp[:b], s=fast_shape,
                                       dim=(-3, -2, -1), norm=correlation_norm)
                if fused_corr_ok:
                    # one kernel pass: /std + max/argmax over b + in-place
                    # score/angle update + ROI fp64 energy (fused_corr.py)
                    fused_corr.update(
                        ccc, std_n, scores, angles, ang_ids,
                        out_max, sq_part, roi_cfg, None,
                        b, 0 if std_n.shape[0] == 1 else std_n.stride(0),
                        fft_scale=correlation_scale,
                    )
                    del ccc
                    sq_acc += sq_part.sum(dtype=torch.float64)
                    running_max_gpu = torch.maximum(
                        running_max_gpu, out_max.max())
                else:
                    ccc = ccc / std_n
                    max_ccc, idx = ccc.max(dim=0)           # (*P,)
                    upd = max_ccc > scores                  # (*P,)
                    scores = torch.where(upd, max_ccc, scores)
                    best_id = ang_ids[idx.long()]           # (*P,) rotation ids
                    angles = torch.where(upd, best_id, angles)
                    # accumulate sum of squares over the ROI sub-boxes
                    # (exact voxel set); fp32 square per voxel, fp64 sum
                    for _sl in roi_slices:
                        sq_acc += ccc[(slice(None),) + _sl].pow(2).sum(
                            dtype=torch.float64)
                    running_max_gpu = torch.maximum(
                        running_max_gpu, max_ccc.max())
                    del ccc, max_ccc, idx, upd, best_id
                if fuse_noise:
                    # twin shares template angles AND mask -> same std_n map
                    noise_ccc = torch.fft.irfftn(ftp[b:], s=fast_shape,
                                                 dim=(-3, -2, -1), norm=correlation_norm)
                    if fused_corr_ok:
                        fused_corr.update(
                            noise_ccc, std_n, noise_scores, None, ang_ids,
                            out_max, None, roi_cfg,
                            out_min if noise_min is not None else None,
                            b, 0 if std_n.shape[0] == 1 else std_n.stride(0),
                            has_min=noise_min is not None,
                            fft_scale=correlation_scale,
                        )
                        del noise_ccc
                        if noise_min is not None:
                            noise_min = torch.minimum(noise_min, out_min)
                    else:
                        noise_batch = noise_ccc / std_n
                        ns_max = noise_batch.max(dim=0).values
                        if noise_min is not None:
                            noise_min = torch.minimum(noise_min, noise_batch.min(dim=0).values)
                        del noise_batch
                        noise_scores = torch.where(ns_max > noise_scores, ns_max,
                                                   noise_scores)
                        del ns_max

            # ---- noise correction branch (angle-subsampled only; the
            # noise_every == 1 case rode along in the main pipeline) ----
            if sub is not None:
                with stage("noise"):
                    cs_r = coeffs["rand"].unsqueeze(0)\
                        .expand(sub.numel(), *tmpl_shape)
                    rsub = rotate_batch_gpu(
                        cs_r.contiguous(), None,
                        matrices=(lin_all[ang_ids.long()][sub],
                                  off_all[ang_ids.long()][sub]),
                    )
                    ft_r = torch.fft.rfftn(rsub, dim=(-3, -2, -1))
                    rn = torch.fft.irfftn(ft_r * wedge_template_t, s=tmpl_shape,
                                          dim=(-3, -2, -1))
                    del ft_r, rsub
                    m_sub = m_b[sub] if rot_mask_idx is not None else m_bc
                    rn = norm_and_mask(rn, m_sub)
                    if pad_buf_r is None or pad_buf_r.shape[0] < sub.numel():
                        pad_buf_r = torch.zeros(
                            (sub.numel(), *fast_shape), device=dev)
                    buf_r = pad_buf_r[:sub.numel()]
                    buf_r[(slice(None),) + pad_index].zero_()
                    buf_r[(slice(None),) + pad_index] = rn
                    del rn, m_sub
                    frn = torch.fft.rfftn(buf_r, dim=(-3, -2, -1))
                    # partial sub-sets must not broadcast against the full
                    # per-angle std map (rotated-mask case); static masks
                    # yield a single shared map that broadcasts as-is
                    std_sel = std_n[sub] if rot_mask_idx is not None else std_n
                    ccc_n = torch.fft.irfftn(fv_c.unsqueeze(0) * frn,
                                             s=fast_shape,
                                             dim=(-3, -2, -1)) / std_sel
                    ns_max = ccc_n.max(dim=0).values
                    if noise_min is not None:
                        noise_min = torch.minimum(noise_min, ccc_n.min(dim=0).values)
                    noise_scores = torch.where(ns_max > noise_scores, ns_max,
                                               noise_scores)
                    del cs_r,frn,std_sel,ccc_n,ns_max

            had_rot_mask = rot_mask_idx is not None
            del std_n, m_b, m_bc, ftp
            if had_rot_mask:
                del b_map, c_map

            n_done += b
            now = time.time()
            if now - t_loop > 20 or b0 + batch >= len(ids):
                dt = now - t_loop
                rate = n_done / dt
                eta = (len(ids) - n_done) / max(rate, 1e-9)
                running_max = float(running_max_gpu)
                logger.info(
                    f"{self.tomo_id}: {n_done}/{len(ids)} rotations "
                    f"({rate:.1f}/s, ETA {eta/60:.1f} min, "
                    f"max score {running_max:.4f})"
                )
                t_loop = now

        # The hybrid full-bank phase must not retain FFT padding/workspace.
        del pad_buf,pad_buf_m,pad_buf_r,buf_t,buf_m,buf_r
        del out_max,sq_part,roi_cfg,out_min,rot_buf
        del fixed_std
        torch.cuda.empty_cache()
        # ---------------- finalisation ----------------
        # RAW score maps are written here. The global noise correction
        #   scores -= noise_scores; scores += <mean noise over the FFT box>
        # is applied once after shard merging in extract.load_score_maps;
        # single-shard runs therefore receive exactly Greco picker's end-of-job
        # correction (their mean also spans the full padded volume).
        noise_np = None
        noise_mean = None
        if self.noise_correction:
            if noise_min is not None:
                noise_scores = (noise_scores - noise_min) * 0.5
                del noise_min
            noise_mean = float(noise_scores.mean())
            noise_scores = torch.roll(torch.flip(noise_scores, dims=(0, 1, 2)),
                                      shifts=shift, dims=(0, 1, 2))
            noise_np = noise_scores[:nx, :ny, :nz].cpu().numpy()
            if self.hybrid_mode:
                # output-frame twin max map (full fast shape): the mu
                # control variate indexes it at the work-frame Sobol
                # positions - exactly as the frozen research implementation
                # does (flip/roll is a bijection of the periodic domain)
                self._hyb_noise_o = noise_scores
            del noise_scores

        # map the pre-flip STOPGAP frame to the output frame (same
        # permutation used for the noise map above)
        scores = torch.roll(torch.flip(scores, dims=(0, 1, 2)),
                            shifts=shift, dims=(0, 1, 2))
        angles = torch.roll(torch.flip(angles, dims=(0, 1, 2)),
                            shifts=shift, dims=(0, 1, 2))

        corrected_full = None
        if self.hybrid_mode:
            # full fast-shape corrected screen map (output frame): local
            # maxima on it (including the padded work-frame shell) are the
            # refinement candidates.  _hyb_noise_o is kept for the mu
            # control variate (reclaimed in the refine).
            corrected_full = scores - self._hyb_noise_o + noise_mean

        scores_np = scores[:nx, :ny, :nz].cpu().numpy()
        angles_np = angles[:nx, :ny, :nz].cpu().numpy()
        del scores, angles, fv_c, fv2_c
        if not self.hybrid_mode:
            torch.cuda.empty_cache()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = self._suffix()
        score_path = storage_path(self.output_dir, f"{self.tomo_id}_scores{suffix}.mrc")
        angle_path = storage_path(self.output_dir, f"{self.tomo_id}_angles{suffix}.mrc")
        if self.hybrid_mode:
            # hybrid semantics (documented in the job JSON): the score map
            # is the CORRECTED screen statistic S - N + mu_s; the full-bank
            # refined scores live in <tomo>_refined.mrc and the STAR.  No
            # separate noise map is written, so a later `greco-picker extract`
            # on this directory reads the corrected map directly.
            score_out = corrected_full[:nx, :ny, :nz].cpu().numpy(
            ).astype(np.float32)
        else:
            score_out = scores_np.astype(np.float32)
        write_mrc(score_path, score_out, self.voxel_size)
        write_mrc(angle_path, angles_np.astype(np.float32), self.voxel_size)
        outputs = {"scores": str(score_path), "angles": str(angle_path)}
        if noise_np is not None and not self.hybrid_mode:
            noise_path = (
                storage_path(self.output_dir, f"{self.tomo_id}_noise_scores{suffix}.mrc")
            )
            write_mrc(noise_path, noise_np.astype(np.float32),
                      self.voxel_size)
            outputs["noise_scores"] = str(noise_path)

        variance = float((sq_acc.item() / roi_size) / max(len(ids), 1))
        stats = {
            "search_space": int(roi_size * len(ids)),
            "roi": int(roi_size),
            "variance": variance,
            "std": float(np.sqrt(variance)),
        }
        if noise_mean is not None:
            stats["noise_mean"] = noise_mean
        logger.info(
            f"{self.tomo_id}: shard done in {(time.time()-t_start)/60:.1f} min;"
            f" score std {stats['std']:.5f}"
        )

        # ---------------- hybrid refine (screen -> picks) ----------------
        hybrid_info = None
        if self.hybrid_mode:
            from .hybrid import run_hybrid_refine

            wall_screen = time.time() - t_screen0
            t0 = time.time()
            hybrid_info = run_hybrid_refine(
                self, corrected_full, stats, wall_screen=wall_screen,
                orientation_matrices=(lin_all,off_all),
            )
            wall_hybrid = time.time() - t0
            del corrected_full
            outputs.update(hybrid_info["outputs"])
            stats["search_space_full_bank"] = int(
                roi_size * self.n_rotations
            )

        job_json = {
            "program": "greco-picker",
            "version": __version__,
            "mode": "hybrid" if self.hybrid_mode else "exhaustive",
            # canonical extraction rule (Round-6H s21): r <= c <= size-1-r,
            # mask applied before peak search, coordinate == r is valid
            "edge_rule": "closed-mask-first-v1",
            "tomogram": str(self.tomogram.resolve()),
            "tomo_id": self.tomo_id,
            "tomo_shape": list(self.tomo_shape),
            "voxel_size": self.voxel_size,
            "template": str(self.template.resolve()),
            "mask": str(self.mask.resolve()),
            "mask_is_spherical": self.mask_is_spherical,
            "angle_increment": self.angle_increment,
            "rotational_symmetry": self.rotational_symmetry,
            "low_pass": self.low_pass,
            "high_pass": self.high_pass,
            "noise_correction": self.noise_correction,
            "rng_seed": self.rng_seed,
            "noise_template_iterations": self.noise_template_iterations,
            "symmetric_noise": self.symmetric_noise,
            "per_tilt_weighting": self.ts_metadata.per_tilt_weighting,
            "tilt_angles": list(self.ts_metadata.tilt_angles),
            "dose_accumulation": (
                list(self.ts_metadata.dose_accumulation)
                if self.ts_metadata.dose_accumulation is not None else None
            ),
            "shard": {"index": self.shard_index, "total": self.shard_total},
            "angle_ids": [int(i) for i in ids],
            "n_rotations": self.n_rotations,
            "stats": stats,
            "outputs": outputs,
            "runtime_minutes": round((time.time() - t_start) / 60, 2),
            "fft_batch_size": batch,
            "correlation_backend": "nvrtc_fused" if fused_corr_ok else "torch",
            "shared_spectral_products": shared_product_count,
            "correlation_precision": "float32",
            "roi_accumulator_precision": "float64",
            "local_std_evaluations": std_evaluations,
            "local_std_reused": not rot_mask_ch,
        }
        picking_statistics=None
        if hybrid_info is not None:
            picking_statistics=hybrid_info.pop("picking_statistics")
            job_json["hybrid"] = hybrid_info
            job_json["wall_total_s"] = round(
                wall_screen + wall_hybrid, 2
            )
        with open(storage_path(self.output_dir, f"{self.tomo_id}_job{suffix}.json"), "w") as f:
            json.dump(job_json, f, indent=2)

        if profile:
            total = sum(timers.values())
            logger.info(
                f"{self.tomo_id} profile (loop stages, {total:.1f}s): "
                + ", ".join(f"{k}={v:.1f}s ({100.0 * v / max(total, 1e-9):.0f}%)"
                            for k, v in sorted(timers.items(),
                                               key=lambda kv: -kv[1]))
            )

        if picking_statistics is not None:stats["picking_statistics"]=picking_statistics
        return stats
