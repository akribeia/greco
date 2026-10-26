"""Movie loading shared by both preprocessing passes."""
from pathlib import Path

import numpy as np
import tifffile


def load_gain(path, *, flip_y=False):
    """Load a reciprocal gain and orient it relative to the unchanged movie."""
    if path is None:
        return None
    if Path(path).suffix.lower() in ('.mrc', '.mrcs'):
        import mrcfile
        with mrcfile.open(path) as handle:
            gain = np.array(handle.data, dtype=np.float32)
    else:
        gain = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    if gain.ndim == 3 and gain.shape[0] == 1:
        gain = gain[0]
    if gain.ndim != 2:
        raise ValueError('Gain reference must contain one 2D image.')
    if not np.isfinite(gain).all():
        gain = np.where(np.isfinite(gain), gain, 1.0)
    if flip_y:
        gain = gain[::-1, :]
    return np.ascontiguousarray(gain, dtype=np.float32)


def load_movie(path, expected_frames, *, eer_group=15, eer_threads=1, eer_upscaling=1):
    if Path(path).suffix.lower() == '.eer':
        from .eer import decode_eer
        return decode_eer(path,group=eer_group,threads=eer_threads,upscaling=eer_upscaling)
    frames = tifffile.imread(str(path))
    if frames.ndim != 3 or frames.shape[0] != expected_frames:
        raise ValueError(f'{path}: expected {expected_frames} frames, got {frames.shape}')
    return np.asarray(frames),dict(format='TIFF',integrated_frames=len(frames))


def frame_geometry(shape,transpose=False):
    """Centred working canvas, with original pixels and sampling preserved."""
    height,width=map(int,shape)
    if min(height,width)<1:raise ValueError('Movie dimensions must be positive.')
    if transpose:height,width=width,height
    side=((max(height,width)+255)//256)*256
    top=(side-height)//2;left=(side-width)//2
    return dict(source_shape=list(map(int,shape)),working_shape=[side,side],
                valid_bounds_yx=[top,top+height,left,left+width],
                padding_yx=[[top,side-height-top],[left,side-width-left]],
                fill='gain_corrected_frame_mean')


def prepare_frames(frames,gain=None,flip_y=False,transpose=False):
    """Gain/orient original pixels, then add a neutral frame-specific border.

    The square multiple-of-256 canvas serves MC, Fourier binning and local
    correction cores. The gain is applied before padding; no gain pixels or
    image samples are invented, cropped, stretched or interpolated.
    """
    if frames.ndim!=3:raise ValueError('Expected movie frames (N,H,W).')
    if flip_y and transpose:raise ValueError('flip_y and transpose are mutually exclusive raster transforms')
    if gain is not None and tuple(gain.shape)!=tuple(frames.shape[-2:]):
        raise ValueError(f'Gain shape {gain.shape} does not match movie {frames.shape[-2:]}.')
    geometry=frame_geometry(frames.shape[-2:],transpose)
    if flip_y:
        frames=np.ascontiguousarray(frames[:,::-1,:])
        if gain is not None:gain=np.ascontiguousarray(gain[::-1,:])
    elif transpose:
        frames=np.ascontiguousarray(frames.transpose(0,2,1))
        if gain is not None:gain=np.ascontiguousarray(gain.T)
    if gain is not None:frames=frames*gain[None,:,:]
    frames=frames.astype(np.float32,copy=False)
    side=geometry['working_shape'][0]
    if frames.shape[-2:]==(side,side):return frames
    means=frames.mean(axis=(1,2),dtype=np.float64).astype(np.float32)
    padded=np.empty((len(frames),side,side),np.float32)
    padded[:]=means[:,None,None]
    top,bottom,left,right=geometry['valid_bounds_yx']
    padded[:,top:bottom,left:right]=frames
    return padded


def frame_doses(metadata, tilt, total_dose, from_mdoc=False):
    """Respect the actually retained EER exposure, not the MDOC frame count."""
    n = metadata['integrated_frames']
    if metadata['format'] != 'EER':
        per_frame = tilt.frame_dose_e_per_A2 if from_mdoc else total_dose/n
    else:
        if from_mdoc:
            # NumSubFrames can mean raw events or already integrated frames.
            # Without that convention, a measured MDOC frame dose is ambiguous.
            raise ValueError('--dose-from-mdoc is not supported for EER; supply total --dose-per-tilt.')
        per_frame = total_dose*metadata['group']/metadata['raw_subframes']
    doses = np.full(n,per_frame,float)
    prior = tilt.prior_dose_e_per_A2 + np.cumsum(doses) - doses
    return doses,prior
