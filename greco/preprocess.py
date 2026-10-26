"""Greco movie preprocessing and native-spectrum CTF estimation.

Stage A reads movie frames, applies detector and motion correction, estimates
CTF parameters, and saves per-tilt motion trajectories and binned review images.
Stage B replays the saved motion correction to assemble raw, aligned and
CTF-reliability stacks, with matching tilt angles and Greco STAR metadata.
The web workflow uses bin8; native movie data remain available for extraction
and reconstruction at finer sampling.

Dose and acquisition geometry come from explicit options or MDOC metadata.
"""

from __future__ import annotations
from greco.project_layout import join as storage_path, ensure_output, project_root
from greco.preprocessing_star import write_table

import argparse
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.transform import rotate

from .align2d import align_tilt_series, shifts_to_fullres
from .ctf import ctf_parameters_for
from .ctf_estimator import TiltCTF, estimate_plane_ctf, refine_series_bad_tilts
from .ctf_model import (CTFParameters, astigmatism_angle_after_raster_rotation, ctf_transfer,
                        regularized_ctf_inverse)
from .ctf_spatial import correct_tilt_ctf_spatial
from .fourier_binning import fourier_bin_2d
from .mdoc import parse_mdoc
from .movie_io import load_gain as _load_gain, load_movie, frame_doses as movie_frame_doses, prepare_frames as _prepare_frames
from .motion import _Dev, motion_correct
from .detector_defects import prepare_correction, apply_correction
from .reliability import apply_reliability_filter, estimate_reliability


# ----------------------------------------------------------------------------
# frame warping (shared by stage A and the stage B rebuild)
# ----------------------------------------------------------------------------


def _rebuild_aligned_sum(dev: _Dev, frames, full_shifts, parity=None,
                         frame_range=None):
    """Rebuild the plain aligned sum (MC3 scale, i.e. the sum of the full
    intensities, mean ~ N * 0.334 on kai) of the frames warped by their full
    shifts (content displacement in the MC3 full-frame gauge). Every frame,
    including the central one, is Fourier-shifted when its shift is nonzero.
    ``parity`` uses the MC3 convention: 'even' = the
    1-based even frames = i % 2 == 1 here (0-based)."""
    n_frames = frames.shape[0]
    lo, hi = frame_range if frame_range is not None else (0, n_frames)
    acc = None
    h, w = frames.shape[-2:]
    ky = dev.asarray(np.fft.fftfreq(h).astype(np.float32)[:, None])
    kx = dev.asarray(np.fft.rfftfreq(w).astype(np.float32)[None, :])
    for i in range(lo, hi):
        if parity is not None and (i % 2) != parity:
            continue
        dy, dx = float(full_shifts[i][0]), float(full_shifts[i][1])
        spectrum = dev.rfft2(frames[i])
        if np.hypot(dy, dx) >= .1:
            spectrum = spectrum * dev.exp(2j*np.pi*(ky*dy+kx*dx))
        acc = spectrum if acc is None else acc + spectrum
    if acc is None:
        raise ValueError("no frames in subset")
    # CCorrectFullShift accumulates spectra, then performs ONE inverse FFT.
    # MC3 GPositivity2D clamps the FINAL native sum, not individual frames.
    # Fractional Fourier shifts can create negative ringing even from counts.
    return dev.maximum(dev.irfft2(acc, s=(h,w)),0).astype(np.float32)


def _rebuild_aligned_halves_gpu(frames, full_shifts):
    """Accumulate shifted spectra on device; clip each final sum like MC3."""
    import cupy as cp
    dev = _Dev(True)
    h,w=frames.shape[-2:]
    ky=cp.asarray(np.fft.fftfreq(h).astype(np.float32)[:,None])
    kx=cp.asarray(np.fft.rfftfreq(w).astype(np.float32)[None,:])
    halves=[cp.zeros((h,w//2+1),cp.complex64),cp.zeros((h,w//2+1),cp.complex64)]
    for i,(dy,dx) in enumerate(full_shifts):
        spectrum=dev.rfft2(frames[i])
        if np.hypot(dy,dx)>=.1:
            spectrum*=cp.exp(2j*np.pi*(ky*float(dy)+kx*float(dx)))
        halves[i%2]+=spectrum
    # MC3 even = 1-based even frames.
    odd,even=halves
    return (cp.maximum(dev.irfft2(even+odd,s=(h,w)),0).astype(cp.float32),
            cp.maximum(dev.irfft2(even,s=(h,w)),0).astype(cp.float32),
            cp.maximum(dev.irfft2(odd,s=(h,w)),0).astype(cp.float32))


def local_mean_bin(image, factor):
    """Device-preserving reference local mean; used ONLY to measure 2D shifts."""
    if isinstance(factor,bool) or not isinstance(factor,(int,np.integer)) or factor<1:
        raise ValueError('binning must be a positive integer')
    if image.ndim!=2 or any(size%factor for size in image.shape):
        raise ValueError('alignment binning must divide both image dimensions')
    h,w=image.shape
    return image.reshape(h//factor,factor,w//factor,factor).mean(axis=(1,3),dtype=np.float32)


def _alignment_inputs(args, sel, mdoc):
    """Require the new unweighted/local-mean products; never reuse old DW images."""
    paths=[storage_path(storage_path(storage_path(args.output, 'per_tilt'), f'tilt_{idx:04d}'), f'alignment_bin{args.binning}.mrc') for idx in sel]
    missing=[str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError('Missing unweighted local-mean 2D inputs. Rerun preprocessing stage A; '
                         'old dose-weighted/Fourier MC images cannot be used silently. '+missing[0])
    planes=[]
    for path in paths:
        with mrcfile.open(path) as handle:planes.append(np.array(handle.data,np.float32))
    return np.stack(planes),np.asarray([mdoc.tilts[idx].angle_deg for idx in sel],float)


# ----------------------------------------------------------------------------
# stage A: per-tilt motion correction + CTF defocus
# ----------------------------------------------------------------------------

def _movie_mdoc(args):
    mdoc = parse_mdoc(args.mdoc)
    scale = getattr(args,'eer_upscaling',1)
    if scale==2 and any(Path(tilt.movie).suffix.lower()=='.eer' for tilt in mdoc.tilts):
        if not all(Path(tilt.movie).suffix.lower()=='.eer' for tilt in mdoc.tilts):
            raise ValueError('Cannot mix TIFF and upscaled EER in one stack.')
        mdoc.pixel_size_A /= scale
        mdoc.image_size = tuple(size*scale for size in mdoc.image_size)
    return mdoc


def _stage_a_tilt(args, idx, out_root):
    mdoc = _movie_mdoc(args)
    tilt = mdoc.tilts[idx]
    movie_path = storage_path(Path(args.movies_dir), tilt.movie)
    frames_host, movie_metadata = load_movie(movie_path, tilt.n_frames,
        eer_group=getattr(args,'eer_group',15), eer_threads=getattr(args,'eer_threads',1),
        eer_upscaling=getattr(args,'eer_upscaling',1))
    n = len(frames_host)
    from .movie_io import frame_geometry
    movie_metadata['frame_geometry']=frame_geometry(frames_host.shape[-2:],args.transpose)
    frames_host = _prepare_frames(frames_host, args._gain, args.flip_y, args.transpose)

    frame_doses, prior_doses = movie_frame_doses(movie_metadata,tilt,
        args.dose_per_tilt,args.dose_from_mdoc)

    dev = _Dev(True)
    with dev.cuda():
        frames = dev.asarray(frames_host, dtype=np.float32)
        defect_started = time.monotonic()
        correction = prepare_correction(frames, flip_y=args.flip_y, transpose=args.transpose,
                                        source_format=movie_metadata.get('format','TIFF'),
                                        valid_bounds_yx=movie_metadata['frame_geometry']['valid_bounds_yx'])
        apply_correction(frames, correction)
        defect_diagnostics = dict(algorithm=str(correction['algorithm']),
            hot_count=int(correction['hot_count']), patch_seed_count=int(correction['patch_seed_count']),
            corrected_pixels=len(correction['bad_yx']), runtime_s=time.monotonic()-defect_started)
        print(f"tilt {idx}: detector cleanup {defect_diagnostics}", flush=True)
        mc = motion_correct(frames, frame_doses, prior_doses, gpu=True,
                            b_factor=args.mc_b_factor, n_iter=args.mc_iter,
                            tol=args.mc_tol, search_radius=args.mc_search_radius)
        greco_result = None
        greco_spectrum = None
        greco_runtime = 0.0
        greco_result, greco_spectrum, greco_runtime = estimate_plane_ctf(
            mc["aligned"], pixel_A=mdoc.pixel_size_A,
            voltage_kV=mdoc.voltage_kv, cs_mm=args.cs_mm,
            amp_contrast=args.amp_contrast, tilt_deg=tilt.angle_deg,
            tile_size=args.ctf_tile_size, gpu=True,
            tile_output=(storage_path(out_root, f'tilt_{idx:04d}') if getattr(args,'ctf_local_refine',False) else None),
            res_low_A=getattr(args,'ctf_res_low_A',None),res_high_A=getattr(args,'ctf_res_high_A',None),
            valid_bounds_yx=movie_metadata['frame_geometry']['valid_bounds_yx'])
        # Registration must see the SAME plain sum as stage B, not the
        # scalar-dose-weighted image used above for the unchanged CTF fit.
        full,even,odd = _rebuild_aligned_halves_gpu(frames,mc['full_shifts'])
        alignment_bin = dev.asnumpy(local_mean_bin(full,args.binning))
        from .fourier_binning import fourier_bin_2d_gpu
        bin8 = dev.asnumpy(fourier_bin_2d_gpu(full,args.binning))
        # Save the plain native MC sum while it is already resident. This is
        # distinct from --bin1's treated/aligned stacks; no dose or CTF filter.
        sta_bin1 = dev.asnumpy(full) if getattr(args,'output_bin1_sta',False) and args.binning!=1 else None
    df_hint = abs(tilt.defocus_um) * 1000.0 if tilt.defocus_um else None
    df = greco_result.df_mean_A
    score = greco_result.score
    df_source = "greco_ctf"

    tdir = storage_path(out_root, f"tilt_{idx:04d}")
    tdir.mkdir(parents=True, exist_ok=True)
    movie_metadata.update(mdoc_n_frames=tilt.n_frames,retained_dose_e_per_A2=float(frame_doses.sum()),
                          detector_defects=defect_diagnostics)
    (storage_path(tdir, 'movie.json')).write_text(json.dumps(movie_metadata,indent=2)+'\n')
    np.savez_compressed(storage_path(tdir, "shifts.npz"), full_shifts=mc["full_shifts"],
                        weights=mc["weights"], iterations=mc["iterations"])
    np.savez_compressed(storage_path(tdir, 'detector_correction.npz'), **correction)
    if greco_spectrum is not None:
        np.save(storage_path(tdir, "ctf_spectrum.npy"), np.asarray(greco_spectrum, np.float32))
    _write_mrc(storage_path(tdir, f"mc_bin{args.binning}.mrc"), bin8, mdoc.pixel_size_A * args.binning)
    if sta_bin1 is not None:_write_mrc(storage_path(tdir, 'mc_bin1.mrc'),sta_bin1,mdoc.pixel_size_A)
    _write_mrc(storage_path(tdir, f"alignment_bin{args.binning}.mrc"), alignment_bin, mdoc.pixel_size_A * args.binning)
    ctf = dict(z_value=idx, angle_deg=tilt.angle_deg, movie=tilt.movie,
               defocus_A=df, defocus_score=score, df_source=df_source, df_hint_A=df_hint,
               voltage_kv=mdoc.voltage_kv, cs_mm=args.cs_mm,
               amp_contrast=args.amp_contrast, pixel_size_A=mdoc.pixel_size_A)
    ctf['fit_res_low_A']=getattr(args,'ctf_res_low_A',None) or 18*mdoc.pixel_size_A
    ctf['fit_res_high_A']=getattr(args,'ctf_res_high_A',None) or max(2*mdoc.pixel_size_A/.75,3.5)
    if greco_result is not None:
        ctf.update(defocus_min_A=greco_result.df_min_A,
                   defocus_max_A=greco_result.df_max_A,
                   astig_azimuth_raw_deg=greco_result.astig_azimuth_deg,
                   phase_rad=greco_result.phase_rad,
                   ctf_resolution_A=greco_result.resolution_A,
                   ctf_runtime_s=greco_runtime)
    (storage_path(tdir, "ctf.json")).write_text(json.dumps(ctf, indent=2) + "\n")
    return dict(idx=idx, mc_runtime_s=mc["runtime_s"], defocus_A=df)


def _load_detector_correction(tdir):
    path = storage_path(tdir, 'detector_correction.npz')
    if not path.exists():
        raise ValueError('Missing saved detector correction. Rerun preprocessing stage A; '
                         'do not mix old MC shifts with newly corrected frames.')
    with np.load(path, allow_pickle=False) as handle:
        correction = {key:handle[key].copy() for key in handle.files}
    if str(correction.get('algorithm','')) != 'mc3_detector_v1':
        raise ValueError('Unsupported detector correction; rerun preprocessing stage A.')
    return correction


def _write_mrc(path, image, voxel):
    with mrcfile.new(str(path), overwrite=True) as h:
        h.set_data(np.asarray(image, np.float32))
        h.voxel_size = (voxel, voxel, voxel)


# ----------------------------------------------------------------------------
# stage B: astig + 2-D alignment + CTF-reliability assembly
# ----------------------------------------------------------------------------

def _prepare_plane_fourier(image, dy, dx, axis_deg, binning):
    source = np.asarray(image, np.float32)
    background = float(np.mean(source))
    shifted = ndi_shift(source, shift=(dy, dx), order=1, mode="constant",
                        cval=background, prefilter=False)
    rotated = rotate(shifted, angle=axis_deg, resize=False, order=1,
                     mode="constant", cval=background, preserve_range=True)
    return fourier_bin_2d(np.asarray(rotated, np.float32), binning)


def _prepare_plane_fullres(image, dy, dx, axis_deg):
    """The bin8 plane chain without the Fourier binning step (native sampling)."""
    source = np.asarray(image, np.float32)
    background = float(np.mean(source))
    shifted = ndi_shift(source, shift=(dy, dx), order=1, mode="constant",
                        cval=background, prefilter=False)
    rotated = rotate(shifted, angle=axis_deg, resize=False, order=1,
                     mode="constant", cval=background, preserve_range=True)
    return np.asarray(rotated, np.float32)


def _apply_shifted_transfer(image, transfer):
    from scipy.fft import fft2, fftshift, ifft2, ifftshift
    spectrum = fftshift(fft2(np.asarray(image, np.float64)))
    result = ifft2(ifftshift(spectrum * np.asarray(transfer, np.float64))).real
    return np.asarray(result, np.float32)


def _stage_b_tilt(args, idx, mdoc, astig, shift_fullres_yx, out_stack, out_csv_lock, csv_rows,
                  out_stack1=None, out_raw8=None, out_raw1=None, raw_only=False, slot=None,
                  out_ctf_halves=None):
    assembly_started=time.monotonic()
    if raw_only:
        assert out_stack is None and out_stack1 is None,\
            'raw_only must not allocate the treated stacks'
    # slot = position of this tilt inside the selected subset (== idx for a
    # full run, but different when --tilts selects a subset); the output
    # arrays are indexed by slot, the per_tilt artifacts by the mdoc idx.
    if slot is None:
        slot = idx
    tilt = mdoc.tilts[idx]
    movie_path = storage_path(Path(args.movies_dir), tilt.movie)
    frames_host, movie_metadata = load_movie(movie_path, tilt.n_frames,
        eer_group=getattr(args,'eer_group',15), eer_threads=getattr(args,'eer_threads',1),
        eer_upscaling=getattr(args,'eer_upscaling',1))
    frames_host = _prepare_frames(frames_host, args._gain, args.flip_y, args.transpose)
    tdir = storage_path(storage_path(args.output, "per_tilt"), f"tilt_{idx:04d}")
    with np.load(storage_path(tdir, "shifts.npz")) as d:
        full_shifts = d["full_shifts"]
    correction = _load_detector_correction(tdir)
    ctf_json = json.loads((storage_path(tdir, "ctf.json")).read_text())

    dev = _Dev(True)
    with dev.cuda():
        frames = dev.asarray(frames_host, dtype=np.float32)
        apply_correction(frames, correction)
        full,even,odd = _rebuild_aligned_halves_gpu(frames,full_shifts)
        # Fast raw-source path: no native planes need to leave the GPU.
        fast_raw=raw_only and not args.ctf_spatial and out_raw1 is None
        fast_device=raw_only and out_raw1 is None
        raw_binned=None
        if fast_device:
            full_np=even_np=odd_np=None
            native_shape=tuple(full.shape)
        else:
            full_np = dev.asnumpy(full).astype(np.float32)
            even_np = dev.asnumpy(even).astype(np.float32)
            odd_np = dev.asnumpy(odd).astype(np.float32)
            native_shape=full_np.shape

    if fast_raw:
        from .fourier_binning import fourier_bin_2d_gpu
        raw_binned={name:dev.asnumpy(fourier_bin_2d_gpu(image,args.binning))
                    for name,image in (("full",full),("even",even),("odd",odd))}

    dy, dx = map(float, shift_fullres_yx)
    if not raw_only:
        prepared = {name: _prepare_plane_fourier(img, dy, dx, args.axis_deg, args.binning)
                    for name, img in (("full", full_np), ("even", even_np), ("odd", odd_np))}
        expected_shape = tuple(size//args.binning for size in full_np.shape)
        if any(p.shape != expected_shape for p in prepared.values()):
            raise ValueError(f"tilt {idx}: bin8 shape {tuple(p.shape for p in prepared.values())}")
    if all(key in ctf_json for key in ("defocus_min_A", "defocus_max_A",
                                        "astig_azimuth_raw_deg", "phase_rad")):
        astig_azimuth_raw = ctf_json["astig_azimuth_raw_deg"]
        effective_azimuth = astigmatism_angle_after_raster_rotation(
            astig_azimuth_raw, args.axis_deg)
        params = CTFParameters(voltage_kV=mdoc.voltage_kv, cs_mm=args.cs_mm,
                               amplitude_contrast=args.amp_contrast,
                               defocus_u_A=ctf_json["defocus_min_A"],
                               defocus_v_A=ctf_json["defocus_max_A"],
                               astigmatism_angle_deg=effective_azimuth,
                               phase_shift_rad=ctf_json["phase_rad"])
        astig_magnitude = ctf_json["defocus_max_A"] - ctf_json["defocus_min_A"]
    else:
        effective_azimuth = astigmatism_angle_after_raster_rotation(astig["azimuth_deg"], args.axis_deg)
        params = ctf_parameters_for(mdoc.voltage_kv, args.cs_mm, args.amp_contrast,
                                    ctf_json["defocus_A"], effective_azimuth, astig["magnitude_A"])
        astig_magnitude = astig["magnitude_A"]
        astig_azimuth_raw = astig["azimuth_deg"]
    binned_shape = tuple(size//args.binning for size in native_shape)
    transfer = ctf_transfer(binned_shape, mdoc.pixel_size_A * args.binning, params)
    inverse = regularized_ctf_inverse(transfer, args.ctf_regularization)
    # The CTF transfer is FRAME-DEPENDENT through its astigmatism azimuth:
    # the treated raster above is rotated by axis_deg (hence the effective
    # azimuth), while the raw-source raster below is the UNROTATED raw MC
    # raster and must be CTF-weighted with the RAW-raster azimuth.  (No-op
    # when axis_deg == 0 or astigmatism == 0; the treated chain is
    # untouched — validated bit-exact.)
    raw_params = replace(params, astigmatism_angle_deg=float(astig_azimuth_raw))
    raw_inverse = regularized_ctf_inverse(
        ctf_transfer(binned_shape, mdoc.pixel_size_A * args.binning, raw_params),
        args.ctf_regularization)
    spatial_fullres = None
    if args.ctf_spatial:
        geometry_file=storage_path(tdir, 'ctf_geometry.json')
        local_geometry=json.loads(geometry_file.read_text()) if geometry_file.exists() else {}
        spatial_kwargs=dict(pixel_size_A=mdoc.pixel_size_A,
                tilt_deg=tilt.angle_deg,
                tilt_axis_deg=local_geometry.get('tilt_axis_deg',args.ctf_spatial_axis_deg),
                defocus_u_A=params.defocus_u_A,
                defocus_v_A=params.defocus_v_A,
                astigmatism_angle_deg=float(astig_azimuth_raw),
                voltage_kV=mdoc.voltage_kv, cs_mm=args.cs_mm,
                amplitude_contrast=args.amp_contrast,
                phase_shift_rad=params.phase_shift_rad,
                regularization=args.ctf_regularization,
                handedness=local_geometry.get('handedness',args.df_hand),
                alpha_offset_deg=local_geometry.get('alpha_offset_deg',0.),
                beta_offset_deg=local_geometry.get('beta_offset_deg',0.),
                gradient_A_per_pixel=local_geometry.get('gradient_A_per_pixel'),
                coordinate_origin=local_geometry.get('coordinate_origin'))
        if fast_device:
            from .fourier_binning import fourier_bin_2d_gpu
            raw_binned={name:dev.asnumpy(fourier_bin_2d_gpu(
                correct_tilt_ctf_spatial(image,gpu=True,return_device=True,**spatial_kwargs),args.binning))
                for name,image in (("full",full),("even",even),("odd",odd))}
        else:
            spatial_fullres={name:correct_tilt_ctf_spatial(image,gpu=True,**spatial_kwargs)
                for name,image in (("full",full_np),("even",even_np),("odd",odd_np))}
        if not raw_only:
            prepared = {
                name: _prepare_plane_fourier(image, dy, dx, args.axis_deg, args.binning)
                for name, image in spatial_fullres.items()
            }
    if not raw_only:
        reliability = estimate_reliability(prepared["even"], prepared["odd"],
                                           smoothing_sigma=2.0, method="cross_power",
                                           apodization="tukey", tukey_alpha=0.1)
        corrected = (prepared["full"] if args.ctf_spatial else
                     _apply_shifted_transfer(prepared["full"], inverse))
        filtered = apply_reliability_filter(corrected, reliability.reliability_full)
        if not np.isfinite(filtered).all():
            raise ValueError(f"tilt {idx}: NaN/Inf in output plane")

    bin1_row = {}
    if out_stack1 is not None:
        # the same chain at native resolution (no binning): frozen shift ->
        # rotate -> bounded CTF inverse at the native pixel.
        #
        # Reliability: the even/odd estimate at native resolution is ~64x
        # noisier than the binned one (same half-dose frames, unbinned), so
        # R collapses in the 8-31 A band and the filter wipes the entire
        # native-resolution information (measured: 100% of the bin1 power
        # ends up below the bin8 Nyquist, R mean 0.02 vs 0.26 at bin8).
        # Default: no reliability at bin1 ('off'); 'fullres' applies the
        # full-res even/odd filter (physically faithful, visually useless).
        want_half = args.bin1_reliability == 'fullres'
        source1 = spatial_fullres if spatial_fullres is not None else {
            "full": full_np, "even": even_np, "odd": odd_np}
        prepared1 = {"full": _prepare_plane_fullres(source1["full"], dy, dx, args.axis_deg)}
        if want_half:
            prepared1["even"] = _prepare_plane_fullres(source1["even"], dy, dx, args.axis_deg)
            prepared1["odd"] = _prepare_plane_fullres(source1["odd"], dy, dx, args.axis_deg)
        n1 = full_np.shape[0]
        if any(p.shape != (n1, n1) for p in prepared1.values()):
            raise ValueError(f"tilt {idx}: bin1 shape {tuple(p.shape for p in prepared1.values())}")
        transfer1 = ctf_transfer((n1, n1), mdoc.pixel_size_A, params)
        inverse1 = regularized_ctf_inverse(transfer1, args.ctf_regularization)
        corrected1 = (prepared1["full"] if args.ctf_spatial else
                      _apply_shifted_transfer(prepared1["full"], inverse1))
        if args.bin1_reliability == 'fullres':
            rel1 = estimate_reliability(prepared1["even"], prepared1["odd"],
                                        smoothing_sigma=2.0 * args.binning, method="cross_power",
                                        apodization="tukey", tukey_alpha=0.1)
            filtered1 = apply_reliability_filter(corrected1, rel1.reliability_full)
            bin1_row = dict(reliability_full_mean_bin1=float(np.mean(rel1.reliability_full)),
                            ctf_inverse_abs_max_bin1=float(np.max(np.abs(inverse1))))
        else:
            filtered1 = np.asarray(corrected1, np.float32)
            bin1_row = dict(ctf_inverse_abs_max_bin1=float(np.max(np.abs(inverse1))))
        if not np.isfinite(filtered1).all():
            raise ValueError(f"tilt {idx}: NaN/Inf in bin1 output plane")
        out_stack1[slot] = filtered1

    if out_raw8 is not None or out_raw1 is not None or out_ctf_halves is not None:
        # raw-source stacks: CTF + reliability in the RAW MC raster — no
        # frozen shift, no axis rotation.  The 2-D loop samples them on the
        # fly (rotation + frozen + cumulative shift at sample time) so no
        # edge fill is ever baked in: maximum FOV at every pass.  The CTF
        # is shift-independent but FRAME-DEPENDENT in its astigmatism
        # azimuth: the raw raster is unrotated, so it is weighted with the
        # RAW-raster azimuth (raw_inverse / raw_params above), not the
        # treated (rotated-raster) chain.
        raw1_row = {}
        n1 = native_shape[0]
        if out_raw8 is not None:
            rel_even = spatial_fullres["even"] if spatial_fullres is not None else even_np
            rel_odd = spatial_fullres["odd"] if spatial_fullres is not None else odd_np
            rel_raw8 = estimate_reliability(raw_binned['even'] if raw_binned is not None else fourier_bin_2d(rel_even, args.binning),
                                            raw_binned['odd'] if raw_binned is not None else fourier_bin_2d(rel_odd, args.binning),
                                            smoothing_sigma=2.0, method="cross_power",
                                            apodization="tukey", tukey_alpha=0.1)
            binned8 = raw_binned['full'] if raw_binned is not None else fourier_bin_2d(
                spatial_fullres["full"] if spatial_fullres is not None else full_np,
                args.binning)
            raw8 = (binned8 if args.ctf_spatial else
                    _apply_shifted_transfer(binned8, raw_inverse))
            raw8 = apply_reliability_filter(raw8, rel_raw8.reliability_full)
            if not np.isfinite(raw8).all():
                raise ValueError(f"tilt {idx}: NaN/Inf in raw-source bin8 plane")
            out_raw8[slot] = raw8
        if out_ctf_halves is not None:
            for name, source in (("even", even_np), ("odd", odd_np)):
                plane = (spatial_fullres[name] if spatial_fullres is not None
                         else source)
                binned = raw_binned[name] if raw_binned is not None else fourier_bin_2d(plane, args.binning)
                corrected_half = (binned if args.ctf_spatial else
                                  _apply_shifted_transfer(binned, raw_inverse))
                if not np.isfinite(corrected_half).all():
                    raise ValueError(f"tilt {idx}: NaN/Inf in CTF-only {name} half")
                out_ctf_halves[name][slot] = corrected_half
        if out_raw1 is not None:
            if args.ctf_spatial:
                raw1 = spatial_fullres["full"]
            else:
                transfer1r = ctf_transfer((n1, n1), mdoc.pixel_size_A, raw_params)
                inverse1r = regularized_ctf_inverse(transfer1r, args.ctf_regularization)
                raw1 = _apply_shifted_transfer(np.asarray(full_np, np.float32), inverse1r)
            if args.bin1_reliability == 'fullres':
                rel1r = estimate_reliability(
                                             spatial_fullres["even"] if spatial_fullres is not None else even_np,
                                             spatial_fullres["odd"] if spatial_fullres is not None else odd_np,
                                             smoothing_sigma=2.0 * args.binning,
                                             method="cross_power",
                                             apodization="tukey", tukey_alpha=0.1)
                raw1 = apply_reliability_filter(raw1, rel1r.reliability_full)
                raw1_row = dict(reliability_full_mean_raw_bin1=float(np.mean(rel1r.reliability_full)))
            if not np.isfinite(raw1).all():
                raise ValueError(f"tilt {idx}: NaN/Inf in raw-source bin1 plane")
            out_raw1[slot] = raw1
    else:
        raw1_row = {}

    if not raw_only:
        out_stack[slot] = filtered
    row = dict(z_value=idx, tilt_deg=tilt.angle_deg, dy_fullres=dy, dx_fullres=dx,
               defocus_A=ctf_json["defocus_A"], df_hint_A=ctf_json["df_hint_A"],
               astig_azimuth_raw_deg=astig_azimuth_raw, astig_magnitude_A=astig_magnitude,
               astig_azimuth_effective_deg=effective_azimuth,
               ctf_transfer_min=float(np.min(transfer)), ctf_inverse_abs_max=float(np.max(np.abs(inverse))),
               raw_ctf_inverse_abs_max=float(np.max(np.abs(raw_inverse))))
    if not raw_only:
        row.update(reliability_half_mean=float(np.mean(reliability.reliability_half)),
                   reliability_full_mean=float(np.mean(reliability.reliability_full)),
                   phase_agreement_mean=float(np.mean(reliability.phase_agreement)),
                   scale_even=reliability.scale_even, scale_odd=reliability.scale_odd)
    row.update(bin1_row)
    row.update(raw1_row)
    row['assembly_runtime_s']=time.monotonic()-assembly_started
    with out_csv_lock:
        csv_rows.append(row)
    print(f"Assembly tilt {idx}: {row['assembly_runtime_s']:.2f}s ({'GPU binning' if fast_raw else 'native path'})",flush=True)
    return idx


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------

def _map_tilts(n_tilts, gpus):
    shards = [[] for _ in range(gpus)]
    for i in range(n_tilts):
        shards[i % gpus].append(i)
    return shards


def _run_shards(fn, shards, n_devices):
    results = {}
    errors = []
    lock = threading.Lock()

    def worker(device_id, idxs):
        import cupy as cp
        if not idxs:
            return
        with cp.cuda.Device(device_id):
            cp.cuda.Device(device_id).use()
            for idx in idxs:
                try:
                    results[idx] = fn(device_id, idx)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    with lock:
                        errors.append((idx, repr(exc)))

    threads = [threading.Thread(target=worker, args=(d, s)) for d, s in enumerate(shards)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise RuntimeError(f"tilt worker failures: {errors[:3]}")
    return results


def run(args):
    started = time.monotonic()
    mdoc = _movie_mdoc(args)
    n = len(mdoc.tilts)
    if args.tilts:
        sel = [int(v) for v in args.tilts.split(",")]
    else:
        sel = list(range(n))
    ensure_output(args.output)
    if args.axis_deg == 0.0 and mdoc.tilt_axis_angle_deg is not None:
        args.axis_deg = mdoc.tilt_axis_angle_deg + 90.0
        print(f"axis_deg from mdoc TiltAxisAngle: {args.axis_deg}")
    if args.ctf_spatial or getattr(args,'ctf_local_refine',False):
        # Resolve the detector-frame gradient axis, independently of display.
        if args.ctf_spatial_axis_deg is None:
            from .ctf_coordinates import detector_axis
            args.ctf_spatial_axis_deg = detector_axis(mdoc,flip_y=args.flip_y,transpose=args.transpose)
            print("ctf-spatial gradient axis from MDOC in MC image coordinates: "
                  f"{args.ctf_spatial_axis_deg}")
        print(f"ctf-spatial: axis={args.ctf_spatial_axis_deg} deg, "
              f"df_hand={args.df_hand}")
    manifest = dict(mdoc=str(args.mdoc), n_tilts_total=n, tilts=sel,
                    dose_per_tilt_e_per_A2=args.dose_per_tilt,
                    dose_from_mdoc=args.dose_from_mdoc,
                    axis_deg=args.axis_deg, binning=args.binning,
                    cs_mm=args.cs_mm, amp_contrast=args.amp_contrast,
                    ctf_regularization=args.ctf_regularization,
                    ctf_estimator="greco_native",
                    ctf_tile_size=args.ctf_tile_size,
                    voltage_kv=mdoc.voltage_kv, pixel_size_A=mdoc.pixel_size_A,
                    gpus=args.gpus,
                    eer=dict(group=getattr(args,'eer_group',15),threads=getattr(args,'eer_threads',1),
                             upscaling=getattr(args,'eer_upscaling',1),trailing_group='dropped, dose adjusted'),
                    mc=dict(algorithm='mc3_full_frame_corrected_v4',
                            detector_correction='mc3_detector_v1',
                            interpolation='fourier_phase',shift_gauge='full_frame_iterative',
                            correlation_target_size=2048,
                            b_factor=args.mc_b_factor, n_iter=args.mc_iter,
                            tol=args.mc_tol, search_radius=args.mc_search_radius),
                    alignment_input=dict(sum='unweighted FULL MC sum',
                                         binning='local_mean',dose_weighting=False),
                    gain=str(args.gain) if args.gain else None,
                    gain_flip_y=bool(getattr(args,'gain_flip_y',False)),
                    flip_y=args.flip_y, transpose=args.transpose, emit_bin1=args.bin1,
                    bin1_reliability=args.bin1_reliability,
                    ctf_spatial=args.ctf_spatial,
                    ctf_spatial_axis_deg=args.ctf_spatial_axis_deg,
                    df_hand=args.df_hand)
    manifest["ctf_half_stacks"] = bool(args.ctf_half_stacks)
    args._gain = _load_gain(args.gain, flip_y=getattr(args,'gain_flip_y',False))
    if args._gain is not None and getattr(args,'eer_upscaling',1)==2 and all(
            Path(tilt.movie).suffix.lower()=='.eer' for tilt in mdoc.tilts):
        if args._gain.shape==(4096,4096):
            args._gain=np.repeat(np.repeat(args._gain,2,axis=0),2,axis=1)

    import cupy as cp
    n_devices = cp.cuda.runtime.getDeviceCount()
    gpus = min(args.gpus, n_devices)

    # ---------------- stage A ----------------
    if args.stage in ("all", "a"):
        t0 = time.monotonic()
        shards = _map_tilts(len(sel), gpus)
        res = _run_shards(lambda d, idx: _stage_a_tilt(args, sel[idx], storage_path(args.output, "per_tilt")),
                          [s for s in shards], gpus)
        manifest["stage_a_s"] = time.monotonic() - t0
        manifest["stage_a_mc_runtime_total_s"] = float(sum(r["mc_runtime_s"] for r in res.values()))
        print(f"stage A done in {manifest['stage_a_s']:.1f} s (MC total {manifest['stage_a_mc_runtime_total_s']:.1f} s)")
        if len(res) == len(sel):
            spectra = []
            ctf_results = []
            for idx in sel:
                tdir = storage_path(storage_path(args.output, "per_tilt"), f"tilt_{idx:04d}")
                spectra.append(np.load(storage_path(tdir, "ctf_spectrum.npy")))
                row = json.loads((storage_path(tdir, "ctf.json")).read_text())
                ctf_results.append(TiltCTF(
                    tilt_deg=row["angle_deg"],
                    df_min_A=row["defocus_min_A"],
                    df_max_A=row["defocus_max_A"],
                    astig_azimuth_deg=row["astig_azimuth_raw_deg"],
                    phase_deg=float(np.rad2deg(row["phase_rad"])),
                    score=row["defocus_score"],
                    resolution_A=row["ctf_resolution_A"],
                ))
            if len(ctf_results) >= 7:
                refined, changed = refine_series_bad_tilts(
                    spectra, ctf_results, pixel_A=mdoc.pixel_size_A,
                    voltage_kV=mdoc.voltage_kv, cs_mm=args.cs_mm,
                    amp_contrast=args.amp_contrast, tile_size=args.ctf_tile_size,
                    gpu=True, angles_deg=[mdoc.tilts[i].angle_deg for i in sel],
                    res_low_A=args.ctf_res_low_A,res_high_A=args.ctf_res_high_A)
            else:
                refined, changed = ctf_results, []
            if args.ctf_local_refine:
                from .ctf_local_refinement import refine_local_series
                refined,geometry=refine_local_series(
                    [storage_path(storage_path(args.output, 'per_tilt'), f'tilt_{idx:04d}') for idx in sel],refined,
                    pixel_A=mdoc.pixel_size_A,voltage_kV=mdoc.voltage_kv,
                    cs_mm=args.cs_mm,amp_contrast=args.amp_contrast,
                    axis_deg=args.ctf_spatial_axis_deg,tile_size=args.ctf_tile_size,gpu=True,
                    res_low_A=args.ctf_res_low_A,res_high_A=args.ctf_res_high_A,
                    forced_hand=args.df_hand if getattr(args,'ctf_fixed_hand',False) else None)
                manifest['ctf_local_geometry']=geometry
                args.ctf_spatial_axis_deg=geometry['tilt_axis_deg']
                manifest['ctf_spatial_axis_deg']=args.ctf_spatial_axis_deg
            for idx, before, after in zip(sel, ctf_results, refined):
                tdir = storage_path(storage_path(args.output, "per_tilt"), f"tilt_{idx:04d}")
                row = json.loads((storage_path(tdir, "ctf.json")).read_text())
                row.update(initial_defocus_min_A=before.df_min_A,
                           initial_defocus_max_A=before.df_max_A,
                           initial_astig_azimuth_raw_deg=before.astig_azimuth_deg,
                           initial_defocus_score=before.score,
                           defocus_A=after.df_mean_A,
                           defocus_min_A=after.df_min_A,
                           defocus_max_A=after.df_max_A,
                           astig_azimuth_raw_deg=after.astig_azimuth_deg,
                           phase_rad=after.phase_rad,
                           defocus_score=after.score,
                           ctf_resolution_A=after.resolution_A,
                           series_refined=before != after)
                (storage_path(tdir, "ctf.json")).write_text(json.dumps(row, indent=2) + "\n")
            manifest["ctf_estimator"] = "greco_native_series"
            manifest["ctf_series_refined_mdoc_indices"] = [int(sel[i]) for i in changed]
            print(f"CTF series refinement: {len(changed)} weak tilts repaired")

    # ---------------- stage B ----------------
    if args.stage in ("all", "b"):
        t0 = time.monotonic()
        with mrcfile.open(storage_path(storage_path(storage_path(args.output, 'per_tilt'), f'tilt_{sel[0]:04d}'), f'mc_bin{args.binning}.mrc')) as first:
            plane_shape=first.data.shape
        bin8_images = np.empty((len(sel), *plane_shape), np.float32)
        angles = np.empty(len(sel), float)
        for k, idx in enumerate(sel):
            with mrcfile.open(storage_path(storage_path(storage_path(args.output, "per_tilt"), f"tilt_{idx:04d}"), f"mc_bin{args.binning}.mrc"), permissive=True) as h:
                bin8_images[k] = h.data
            angles[k] = mdoc.tilts[idx].angle_deg
        alignment_images,alignment_angles = _alignment_inputs(args,sel,mdoc)
        al = align_tilt_series(alignment_images, alignment_angles, gpu=True)
        shifts_fullres = shifts_to_fullres(al["shifts"], args.binning)
        (storage_path(args.output, "alignment2d.json")).write_text(json.dumps(dict(
            backend=al['backend'],compute_device=al['compute_device'],diagnostics=al['diagnostics'],
            input_sum='unweighted MC sum',input_binning='local_mean',dose_weighting=False,
            output_binning='Fourier; raw 3D products unchanged',
            anchor=al["anchor"], edge_residual_median=float(np.median(al["edge_residuals"])),
            edge_residual_p90=float(np.quantile(al["edge_residuals"], .9)),
            shift_norm_median=float(np.median(np.linalg.norm(shifts_fullres, axis=1))),
            shift_norm_max=float(np.max(np.linalg.norm(shifts_fullres, axis=1))),
            runtime_s=al["runtime_s"]), indent=2) + "\n")
        print(f"2-D alignment: median |shift| {np.median(np.linalg.norm(shifts_fullres, axis=1)):.1f} full-res px, "
              f"edge resid p90 {np.quantile(al['edge_residuals'], .9):.2f} bin8 px")

        # Astigmatism is estimated independently for every tilt in stage A.
        astig = dict(source="greco_native", accepted=True,
                     note="per-tilt astigmatism estimated from the motion-corrected tilt")
        project=project_root(args.output)
        if project is None:
            (storage_path(args.output, "ctf/astigmatism.json")).write_text(json.dumps(astig, indent=2) + "\n")

        raw_only = args.raw_only
        if raw_only and not args.raw_source:
            raise ValueError('--raw-only implies --raw-source')
        # native size = bin8 side x binning (the bin8 planes are the
        # Fourier binned fullres planes); the worker asserts the real
        # fullres shape against this
        if plane_shape[0]!=plane_shape[1]:
            raise ValueError('Preprocessing currently requires square movies.')
        n_pix = plane_shape[0] * int(args.binning)
        out_stack = None if raw_only else np.empty((len(sel), *plane_shape), np.float32)
        out_stack1 = None
        if args.bin1 and not raw_only:
            out_stack1 = np.empty((len(sel), n_pix, n_pix), np.float32)
        out_raw8 = None
        out_raw1 = None
        if args.raw_source:
            out_raw8 = np.empty((len(sel), *plane_shape), np.float32)
            if args.bin1:
                out_raw1 = np.empty((len(sel), n_pix, n_pix), np.float32)
        out_ctf_halves = ({name: np.empty((len(sel), *plane_shape), np.float32)
                           for name in ("even", "odd")}
                          if args.ctf_half_stacks else None)
        csv_rows = []
        lock = threading.Lock()
        shards = _map_tilts(len(sel), gpus)
        t1 = time.monotonic()

        def assemble(device_id, k):
            return _stage_b_tilt(args, sel[k], mdoc, astig, shifts_fullres[k], out_stack, lock,
                                 csv_rows, out_stack1, out_raw8, out_raw1, raw_only=raw_only,
                                 slot=k, out_ctf_halves=out_ctf_halves)

        _run_shards(assemble, shards, gpus)
        manifest["stage_b_s"] = time.monotonic() - t0
        manifest["stage_b_assemble_s"] = time.monotonic() - t1

        # The reference MC3 stacks are assembled in physical-angle order
        # (ascending), not in mdoc acquisition order (kai acquires in a
        # two-step serpentine, so mdoc order is a zigzag).  Write the treated
        # stack in the same angle-sorted order as the reference so plane
        # indices are directly comparable.
        order = np.argsort(angles, kind="stable")
        raw_bin8 = np.ascontiguousarray(bin8_images[order])
        if out_stack is not None:
            out_stack = np.ascontiguousarray(out_stack[order])
        if out_stack1 is not None:
            out_stack1 = np.ascontiguousarray(out_stack1[order])
        if out_raw8 is not None:
            out_raw8 = np.ascontiguousarray(out_raw8[order])
        if out_raw1 is not None:
            out_raw1 = np.ascontiguousarray(out_raw1[order])
        if out_ctf_halves is not None:
            out_ctf_halves = {name: np.ascontiguousarray(stack[order])
                              for name, stack in out_ctf_halves.items()}
        angles = np.asarray(angles)[order]
        if not np.all(np.diff(angles) > 0):
            raise ValueError("output tilt order is not strictly increasing")
        manifest["plane_order"] = "sorted by physical tilt angle (reference convention)"
        manifest["plane_to_mdoc_index"] = [int(sel[k]) for k in order]

        # Primary preprocessing product: the motion-corrected, Fourier-binned
        # tilt series in the raw detector raster.  It has no frozen series
        # shift, axis rotation, CTF inverse or reliability filter baked in.
        # Store it physically in ascending tilt order so it is directly
        # viewable without consulting the acquisition-order MDOC.  The
        # companion CSV uses exactly the same plane order and supplies the
        # frozen shift needed by downstream on-the-fly materialization.
        raw_bin8_path = storage_path(args.output, f"greco_raw_bin{args.binning}.mrcs")
        raw_bin8_tmp = storage_path(args.output, f".{raw_bin8_path.name}.tmp")
        with mrcfile.new(str(raw_bin8_tmp), overwrite=True) as h:
            h.set_data(raw_bin8)
            h.voxel_size = (mdoc.pixel_size_A * args.binning,) * 3
            h.update_header_stats()
        raw_bin8_tmp.replace(raw_bin8_path)
        raw_tlt_path = storage_path(args.output, f"greco_raw_bin{args.binning}.rawtlt")
        raw_tlt_path.write_text("\n".join(f"{a:.6f}" for a in angles) + "\n")
        raw_shifts_path = storage_path(args.output, f"alignment/tilts.star")
        if raw_shifts_path.suffix=='.star':
            write_table(raw_shifts_path,[dict(plane=plane,angle_deg=float(angles[plane]),mdoc_index=int(sel[k]),
                dy_raw=float(shifts_fullres[k][0]),dx_raw=float(shifts_fullres[k][1])) for plane,k in enumerate(order)],'alignment')
        else:
            with raw_shifts_path.open("w") as fh:
                fh.write("plane,angle_deg,mdoc_index,dy_raw,dx_raw\n")
                for plane, k in enumerate(order):
                    mdoc_idx = int(sel[k])
                    dy, dx = shifts_fullres[k]
                    fh.write(f"{plane},{angles[plane]:.6f},{mdoc_idx},"
                             f"{float(dy):.6f},{float(dx):.6f}\n")
        manifest["primary_output_stack"] = str(raw_bin8_path)
        manifest["primary_output_angles"] = str(raw_tlt_path)
        manifest["primary_output_frozen_shifts"] = str(raw_shifts_path)
        manifest["primary_output_processing"] = (
            "motion correction + unweighted frame sum + centered Fourier binning only; "
            "raw detector raster, no series shift/axis rotation/CTF/reliability")
        print(f"raw bin8 tilt series written -> {raw_bin8_path}")

        prefix = None
        if out_stack is not None:
            prefix = storage_path(args.output, f"greco_aligned_ctf_reliability_bin{args.binning}.mrcs")
            tmp = storage_path(args.output, f".{prefix.name}.tmp")
            with mrcfile.new(str(tmp), overwrite=True) as h:
                h.set_data(out_stack)
                h.voxel_size = (mdoc.pixel_size_A * args.binning,) * 3
                h.update_header_stats()
            tmp.replace(prefix)
            (storage_path(args.output, f"greco_aligned_ctf_reliability_bin{args.binning}.rawtlt")).write_text(
                "\n".join(f"{a:.6f}" for a in angles) + "\n")
        if out_stack1 is not None:
            prefix1 = storage_path(args.output, f"greco_aligned_ctf_reliability_bin1.mrcs")
            tmp1 = storage_path(args.output, f".{prefix1.name}.tmp")
            with mrcfile.new(str(tmp1), overwrite=True) as h:
                h.set_data(out_stack1)
                h.voxel_size = (mdoc.pixel_size_A,) * 3
                h.update_header_stats()
            tmp1.replace(prefix1)
            manifest["output_stack_bin1"] = str(prefix1)
            # the bin1 stack shares the bin8 plane order -> same rawtlt
            (storage_path(args.output, "greco_aligned_ctf_reliability_bin1.rawtlt")).write_text(
                "\n".join(f"{a:.6f}" for a in angles) + "\n")
            print(f"bin1 stack written -> {prefix1}")
        if out_raw8 is not None:
            pr8 = storage_path(args.output, f"ctf/reliability_bin{args.binning}.mrcs")
            with mrcfile.new(str(pr8), overwrite=True) as h:
                h.set_data(out_raw8)
                h.voxel_size = (mdoc.pixel_size_A * args.binning,) * 3
                h.update_header_stats()
            (storage_path(args.output, f"ctf/reliability_bin{args.binning}.rawtlt")).write_text(
                "\n".join(f"{a:.6f}" for a in angles) + "\n")
            manifest[f"output_stack_raw_bin{args.binning}"] = str(pr8)
            # frozen 2-D shifts in the RAW MC raster, one row per plane of the
            # raw-source stacks (same angle-sorted order as the mrcs/rawtlt)
            print(f"raw-source bin8 stack written -> {pr8}")
        if out_ctf_halves is not None:
            for name, stack in out_ctf_halves.items():
                path = storage_path(args.output, f"greco_ctf_{name}_bin{args.binning}_rawsource.mrcs")
                with mrcfile.new(str(path), overwrite=True) as h:
                    h.set_data(stack)
                    h.voxel_size = (mdoc.pixel_size_A * args.binning,) * 3
                    h.update_header_stats()
                manifest[f"output_ctf_{name}_raw_bin{args.binning}"] = str(path)
            full_ctf_path = storage_path(args.output, f"greco_ctf_bin{args.binning}_rawsource.mrcs")
            with mrcfile.new(str(full_ctf_path), overwrite=True) as h:
                h.set_data(out_ctf_halves["even"] + out_ctf_halves["odd"])
                h.voxel_size = (mdoc.pixel_size_A * args.binning,) * 3
                h.update_header_stats()
            manifest[f"output_ctf_full_raw_bin{args.binning}"] = str(full_ctf_path)
            manifest["ctf_half_processing"] = (
                "motion-corrected disjoint frame sums, spatial CTF inverse when enabled, "
                "Fourier bin8; raw raster, no reliability, frozen shift or axis rotation")
        if out_raw1 is not None:
            # 'ctf' only in the name: at native resolution the even/odd
            # reliability collapses, so the bin1 raw-source stack is CTF
            # inverse only (the treated bin1's 'reliability' in its name is
            # bin8 uses the aligned grid; bin1 remains on the native detector grid)
            pr1 = storage_path(args.output, f"greco_ctf_bin1_rawsource.mrcs")
            with mrcfile.new(str(pr1), overwrite=True) as h:
                h.set_data(out_raw1)
                h.voxel_size = (mdoc.pixel_size_A,) * 3
                h.update_header_stats()
            (storage_path(args.output, "greco_ctf_bin1_rawsource.rawtlt")).write_text(
                "\n".join(f"{a:.6f}" for a in angles) + "\n")
            manifest["output_stack_raw_bin1"] = str(pr1)
            print(f"raw-source bin1 stack written -> {pr1}")
        write_table(storage_path(args.output,"ctf/ctf.star"),sorted(csv_rows,key=lambda r:r['tilt_deg']),'ctf')
        if prefix is not None:
            manifest["output_stack"] = str(prefix)
        if raw_only:
            manifest["raw_only"] = True
        print(f"stage B done in {manifest['stage_b_s']:.1f} s -> {prefix or '(raw-source only)'}")

    manifest["runtime_s"] = time.monotonic() - started
    (storage_path(args.output, "manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser(description="greco raw movies -> aligned CTF-reliability bin8 stack")
    p.add_argument("--mdoc", type=Path, required=True)
    p.add_argument("--movies-dir", type=Path, required=True)
    p.add_argument("--dose-per-tilt", type=float, required=True,
                   help="total dose per tilt in e-/A^2, divided evenly over the frames (the one user input)")
    p.add_argument("--dose-from-mdoc", action="store_true",
                   help="use the measured per-frame doses from the mdoc instead of --dose-per-tilt")
    p.add_argument("--cs-mm", type=float, default=2.7)
    p.add_argument("--amp-contrast", type=float, default=0.07)
    p.add_argument("--ctf-regularization", type=float, default=0.05)
    p.add_argument("--ctf-tile-size", type=int, default=512,
                   help="tile size for the native CTF estimator")
    p.add_argument("--ctf-spatial", action="store_true",
                   help="replace the single global CTF inverse with the "
                        "spatial correction: per 256-px core the defocus "
                        "is the local value from the tilt-induced Z gradient (512/256 tile/core "
                        "layout, fixed astig ratio). Requires the correct raw-raster tilt axis.")
    p.add_argument('--ctf-local-refine',action='store_true',
                   help='Retain bin1 tile spectra and refine handedness and alpha/beta geometry before spatial correction.')
    p.add_argument('--ctf-fixed-hand',action='store_true',help='Preserve --df-hand during local CTF refinement after a project handedness test.')
    p.add_argument('--ctf-res-low-A',type=float,default=None,help='Low-resolution limit of the CTF fit in Å (larger number); default 18 × bin1 pixel size.')
    p.add_argument('--ctf-res-high-A',type=float,default=None,help='High-resolution limit of the CTF fit in Å (smaller number); default max(2 × bin1 pixel size / .75, 3.5 Å).')
    p.add_argument("--ctf-spatial-axis-deg", type=float, default=None,
                   help="defocus-gradient direction in MC image coordinates. "
                        "Default: MDOC TiltAxisAngle, carried through MC flip/transpose; "
                        "fallback: -90 - RotationAngle. An explicit value is already in MC coordinates.")
    p.add_argument("--df-hand", type=int, choices=(-1, 1), default=1,
                   help="defocus handedness for the spatial CTF gradient; kai: -1")
    p.add_argument("--binning", type=int, default=8, choices=(8,), help="Preprocessing output binning (fixed at 8).")
    p.add_argument('--output-bin1-sta',action='store_true',
                   help='Save each unweighted native MC sum as per_tilt/tilt_NNNN/mc_bin1.mrc for STA extraction, without CTF or alignment transforms.')
    p.add_argument("--bin1", action="store_true",
                   help="also emit the treated native-resolution stack (same "
                        "chain: frozen shift -> rotate -> bounded CTF inverse, "
                        "but without the binning step) for the full-res "
                        "landmark refinement")
    p.add_argument("--bin1-reliability", choices=("off", "fullres"), default="off",
                   help="reliability filter for the bin1 stack: 'off' (default, "
                        "CTF inverse only) or 'fullres' (full-res even/odd "
                        "filter; NOTE: at native resolution the even/odd are "
                        "~64x noisier, so R collapses and the filter wipes the "
                        "8-31 A band)")
    p.add_argument("--axis-deg", type=float, default=0.0,
                   help="raster rotation applied to each plane (0 = TiltAxisAngle+90 from the mdoc)")
    p.add_argument("--raw-source", action="store_true",
                   help="also emit the RAW-SOURCE stacks: CTF + reliability in the raw "
                        "MC raster with NO frozen shift and NO axis rotation baked in "
                        "(bin8 always; bin1 too when --bin1).  The 2-D loop (--raw-source "
                        "in progressive_alignment/refine_landmarks) samples these on the fly with rotation + "
                        "frozen + cumulative shift, so no edge fill is ever baked in "
                        "(maximum FOV at every pass).  Also writes "
                        "alignment/tilts.star (per-plane frozen shift).")
    p.add_argument("--raw-only", action="store_true",
                   help="with --raw-source: emit ONLY the raw-source stacks (the full "
                        "self-contained raw protocol) — skip the treated "
                        "(shift+rotated) bin8/bin1 stacks entirely")
    p.add_argument("--ctf-half-stacks", action="store_true",
                   help="emit raw-raster even/odd CTF-corrected bin8 stacks without "
                        "reliability, for volume denoising experiments")
    p.add_argument("--gain", type=Path, default=None,
                   help="detector gain-reference TIFF or MRC (reciprocal gain: frame *= gain)")
    p.add_argument("--gain-flip-y", action="store_true",
                   help="flip only the gain reference along Y before applying it to movies")
    p.add_argument("--flip-y", action="store_true",
                   help="flip the frames along y before motion correction (the measured "
                        "TIFF->MC3 raster transform for this dataset; needed to match the "
                        "reference stack raster and all downstream calibrations)")
    p.add_argument("--transpose", action="store_true",
                   help="transpose frame y/x axes before motion correction, including "
                        "the gain raster; mutually exclusive with --flip-y")
    p.add_argument("--mc-b-factor", type=float, default=500.0, help="MC3 -Bft[0]")
    p.add_argument("--mc-iter", type=int, default=15, help="MC3 -Iter")
    p.add_argument("--mc-tol", type=float, default=0.10, help="MC3 -Tol (px)")
    p.add_argument("--mc-search-radius", type=int, default=64, help="Half-width in MC correlation-buffer pixels (MC3: 128x128)")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument('--eer-group',type=int,default=15,
                   help='raw EER subframes per integrated motion-correction frame')
    p.add_argument('--eer-upscaling',type=int,choices=(1,2),default=1,
                   help='EER spatial sampling: 1=native 4K, 2=superresolution 8K; mdoc pixel size is native')
    p.add_argument('--eer-threads',type=int,default=1,
                   help='CPU decoder threads per GPU worker (1–64)')
    p.add_argument("--stage", choices=("all", "a", "b"), default="all")
    p.add_argument("--tilts", type=str, default=None, help="comma-separated tilt indices (for development)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.eer_group < 1 or not 1 <= args.eer_threads <= 64:
        p.error('--eer-group must be positive and --eer-threads must be 1–64')
    if args.flip_y and args.transpose:
        p.error("--flip-y and --transpose are mutually exclusive")
    if args.ctf_half_stacks and not args.raw_source:
        p.error("--ctf-half-stacks requires --raw-source")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
