"""Axis-independent global CTF fit for Advanced reconstruction.

Fit raw averaged tile spectra without geometric defocus gradients. Saved
cleaned spectra are used when preprocessing did not retain individual tiles.
"""
import numpy as np
from ..project_layout import join as storage_path


def fit_flat(folder, ctf, options):
    import cupy as cp
    from ..ctf_estimator import _CTFEstimator
    from ..ctf_gpu import spectrum_sum
    estimator=_CTFEstimator(pixel_A=ctf['pixel_size_A'],voltage_kV=ctf['voltage_kv'],
        cs_mm=ctf['cs_mm'],amp_contrast=ctf['amp_contrast'],
        tile_size=options.get('ctf_tile_size',512),gpu=True,
        res_low_A=ctf.get('fit_res_low_A'),res_high_A=ctf.get('fit_res_high_A'))
    tiles_path=storage_path(folder,'ctf_tiles.npy')
    if tiles_path.exists():
        tiles=cp.asarray(np.load(tiles_path),cp.float32)
        spectrum=estimator.clean(spectrum_sum(tiles)/len(tiles))
    else:
        spectrum=cp.asarray(np.load(storage_path(folder,'ctf_spectrum.npy')),cp.float32)
    result=estimator.do_tilt(spectrum,0.,(0.,0.))
    if not np.isfinite(result.df_mean_A) or result.df_mean_A<=0:
        raise ValueError('Invalid Advanced global CTF defocus.')
    fitted=dict(ctf,defocus_min_A=result.df_min_A,defocus_max_A=result.df_max_A,
        astig_azimuth_raw_deg=result.astig_azimuth_deg,phase_rad=result.phase_rad)
    return fitted


def flat_correction(correction, regularization):
    """Remove every spatial/geometric prior, including the tilt lowpass."""
    return dict(correction,tilt_deg=0.,tilt_axis_deg=0.,handedness=1,
        alpha_offset_deg=0.,beta_offset_deg=0.,gradient_A_per_pixel=[0.,0.],
        coordinate_origin='image_pixel_centres',wiener_lowpass=15.,
        regularization=regularization)


def alignment_plane(folder,ctf,options,images,binning=8,*,refined_view=None):
    """Native CTF/even-odd chain; coarse alignment is axis-independent."""
    import cupy as cp
    from ..ctf_spatial import correct_tilt_ctf_spatial
    from ..fourier_binning import fourier_bin_2d_gpu
    from ..reliability import estimate_reliability,apply_reliability_filter
    fitted=fit_flat(folder,ctf,options) if refined_view is None else refined_view['ctf']
    correction=dict(pixel_size_A=fitted['pixel_size_A'],
        defocus_u_A=fitted['defocus_min_A'],defocus_v_A=fitted['defocus_max_A'],
        astigmatism_angle_deg=fitted['astig_azimuth_raw_deg'],voltage_kV=fitted['voltage_kv'],
        cs_mm=fitted['cs_mm'],amplitude_contrast=fitted['amp_contrast'],phase_shift_rad=fitted['phase_rad'],
        gpu=True,filter_mode='tikhonov')
    if refined_view is None:
        correction=flat_correction(correction,options.get('ctf_regularization',.05))
    else:
        geometry=refined_view['geometry']
        correction.update(tilt_deg=geometry['tilt_deg'],tilt_axis_deg=geometry['tilt_axis_deg'],
            handedness=geometry['handedness'],alpha_offset_deg=geometry.get('alpha_offset_deg',0.),
            beta_offset_deg=geometry.get('beta_offset_deg',0.),
            gradient_A_per_pixel=geometry.get('gradient_A_per_pixel'),
            coordinate_origin=geometry.get('coordinate_origin'),
            wiener_lowpass=15.,regularization=options.get('ctf_regularization',.05))
    corrected={name:cp.asnumpy(fourier_bin_2d_gpu(cp.asarray(correct_tilt_ctf_spatial(image,**correction)),binning))
               for name,image in images.items()}
    reliability=estimate_reliability(corrected['even'],corrected['odd'],smoothing_sigma=2.,
        method='cross_power',apodization='tukey',tukey_alpha=.1)
    result=apply_reliability_filter(corrected['full'],reliability.reliability_full)
    if not np.isfinite(result).all():raise ValueError('Nonfinite CTF alignment plane.')
    return result
