"""Views of fitted defocus geometry and the actual cleaned CTF spectra."""
from greco.project_layout import join as storage_path, owned as layout_owned
from greco.preprocessing_star import read_table

import io
import json
import numpy as np
from .projects import InputError


def full_spectrum(half):
    half=np.asarray(half,np.float32)
    n=half.shape[0]
    if half.ndim!=2 or half.shape[1]!=n//2+1 or n%2:
        raise InputError('Invalid saved CTF half spectrum.')
    full=np.empty((n,n),np.float32)
    full[:,n//2:]=half[:,:n//2]
    full[:,:n//2]=half[(-np.arange(n))%n,n//2:0:-1]
    return full


def tilt_records(output, binning):
    return read_table(storage_path(output,f'alignment/tilts.star'))


def defocus_field(ctf, record, options, height, width, binning, corners=False, rotation_deg=0, flip_y=False):
    from ..ctf_estimator import TiltDefocusGeometry
    if corners:
        y,x=np.meshgrid([0,height-1],[0,width-1],indexing='ij')
    else:
        y,x=np.mgrid[:height,:width]
    x=x-(width-1)/2;y=y-(height-1)/2
    x=x*binning-float(record['dx_raw'])
    y=y*binning-float(record['dy_raw'])
    local=ctf.get('local_geometry',{})
    saved_axis=local.get('tilt_axis_deg',options.get('ctf_spatial_axis_deg',0))
    if saved_axis is None and local.get('gradient_A_per_pixel') is None:raise InputError('No resolved spatial CTF axis was saved. Redo CTF before displaying its landscape.')
    fitted_axis=float(saved_axis or 0)
    geometry=TiltDefocusGeometry(float(ctf['angle_deg']),fitted_axis,float(ctf['pixel_size_A']),int(local.get('handedness',options.get('df_hand',1))),float(local.get('alpha_offset_deg',0)),float(local.get('beta_offset_deg',0)),local.get('gradient_A_per_pixel'))
    field=geometry.local_mean_defocus_A(float(ctf['defocus_A']),x,y)
    if rotation_deg or flip_y:
        if corners:
            raise ValueError('Corner samples describe the unrotated detector field.')
        # First construct the field in the same raster as the image. Then
        # apply EXACTLY the image viewer transform, once, to scalar defocus.
        # Do not derive a new gradient direction from a display angle.
        from .tilt_display import orient_image
        field=orient_image(field,rotation_deg,flip_y,float(ctf['defocus_A']))
    return field


def series_defocus_range(output, records, options, height, width, binning):
    lo,hi=float('inf'),float('-inf')
    # A planar field reaches its extrema at the corners. No need to generate
    # every volume-sized field merely to establish a common colour scale.
    for record in records:
        folder=storage_path(storage_path(output, 'per_tilt'), f"tilt_{int(record['mdoc_index']):04d}")
        ctf=json.loads((storage_path(folder, 'ctf.json')).read_text())
        geometry=storage_path(folder, 'ctf_geometry.json')
        if geometry.exists():ctf['local_geometry']=json.loads(geometry.read_text())
        field=defocus_field(ctf,record,options,height,width,binning,corners=True)
        field=field-float(ctf['defocus_A'])
        lo=min(lo,float(field.min()));hi=max(hi,float(field.max()))
    span=max(abs(lo),abs(hi),1.)
    return -span,span


def write_spectrum_stack(output, binning):
    import mrcfile
    records=tilt_records(output,binning)
    stack=np.stack([full_spectrum(np.load(storage_path(storage_path(storage_path(output, 'per_tilt'), f"tilt_{int(row['mdoc_index']):04d}"), 'ctf_spectrum.npy'),allow_pickle=False)) for row in records])
    path=storage_path(output, 'greco_ctf_spectra.mrcs')
    with mrcfile.new(path,overwrite=True) as handle:
        handle.set_data(stack)
    # Fourier pixels are not real-space voxels; do not assign Å/px here.
    return str(path)


def fitted_amplitude_half(ctf, size):
    """Saved fit in the estimator's raw Fourier raster (no refitting)."""
    from ..ctf_estimator import electron_wavelength_A
    pix=float(ctf['pixel_size_A'])
    x=np.arange(size//2+1)/size
    y=(np.arange(size)-size//2)/size
    f2=(x[None,:]**2+y[:,None]**2).astype(np.float32)
    phi=np.arctan2(y[:,None],x[None,:]+1e-30).astype(np.float32)
    mean=float(ctf['defocus_A'])
    low=float(ctf.get('defocus_min_A',mean));high=float(ctf.get('defocus_max_A',mean))
    # The saved azimuth is the minimum-defocus axis, unlike the internal fit.
    az=np.float32(np.deg2rad(float(ctf.get('astig_azimuth_raw_deg',90))-90))
    df=(mean+(high-low)/2*np.cos(2*(phi-az)))/pix
    lam=electron_wavelength_A(float(ctf.get('voltage_kv',300)))/pix
    cs=float(ctf.get('cs_mm',2.7))*1e7/pix
    amp=float(ctf.get('amp_contrast',.07))
    phase=np.float32(np.arcsin(amp)+float(ctf.get('phase_rad',0)))
    chi=phase+(np.pi*lam*f2).astype(np.float32)*df-(.5*np.pi*lam**3*cs*f2*f2).astype(np.float32)
    return (np.abs(np.sin(chi))-.5)*np.exp(-f2)


def comparison_image(half, ctf):
    """Model left, measured right; identical Fourier coordinates and fit band."""
    from PIL import Image
    measured=full_spectrum(half);n=measured.shape[0]
    model=full_spectrum(fitted_amplitude_half(ctf,n))
    pix=float(ctf['pixel_size_A'])
    y,x=np.mgrid[:n,:n]
    frequency=np.hypot(x-n//2,y-n//2)/(n*pix)
    low=float(ctf.get('fit_res_low_A') or 18*pix)
    high=float(ctf.get('fit_res_high_A') or max(2*pix/.75,3.5))
    band=(frequency>=1/low)&(frequency<=1/high)
    samples=measured[band&np.isfinite(measured)]
    if not samples.size:raise InputError('No saved spectrum samples inside the CTF fit interval.')
    lo,hi=np.percentile(samples,[2,99])
    observed=255*np.clip((measured-lo)/max(float(hi-lo),1e-12),0,1)
    predicted=255*np.clip(model+.5,0,1)
    composite=np.where(x<n//2,predicted,observed)
    composite=np.where(band&np.isfinite(composite),composite,128)
    return Image.fromarray(composite.astype('uint8'))


def ctf_png(service, project_id, series_id, index, mode, rotation_deg=0, flip_y=False, df_hand=None, axis_deg=None):
    import mrcfile
    from PIL import Image
    from .preprocessing import status
    rotation_deg=float(rotation_deg)
    if not np.isfinite(rotation_deg) or abs(rotation_deg)>360 or not isinstance(flip_y,bool):
        raise InputError('Invalid CTF preview orientation.')
    if df_hand is not None:
        if isinstance(df_hand,bool) or str(df_hand) not in ('-1','1'):
            raise InputError('Invalid defocus handedness preview.')
        df_hand=int(df_hand)
    if axis_deg is not None:
        axis_deg=float(axis_deg)
        if not np.isfinite(axis_deg) or abs(axis_deg)>360:
            raise InputError('Invalid raw CTF axis preview.')
    job=status(service,project_id,series_id)
    if job.get('state')!='completed':
        raise InputError('Complete preprocessing first.')
    work=service.path(job['work'])
    if not layout_owned(service._root(project_id),work):
        raise InputError('Invalid project output path.')
    output=storage_path(work, 'output')
    binning=job.get('output_binning',job.get('options',{}).get('output_binning',8))
    records=tilt_records(output,binning)
    if not 0<=index<len(records):
        raise InputError('Tilt index out of range.')
    record=records[index]
    folder=storage_path(storage_path(output, 'per_tilt'), f"tilt_{int(record['mdoc_index']):04d}")
    if mode=='rings':
        # Existing jobs retain NPY spectra; new jobs also save the MRCS.
        path=storage_path(output, 'greco_ctf_spectra.mrcs')
        if path.exists():
            with mrcfile.mmap(path,mode='r') as handle:
                spectrum=np.array(handle.data[index])
        else:
            spectrum=full_spectrum(np.load(storage_path(folder, 'ctf_spectrum.npy'),allow_pickle=False))
        lo,hi=np.percentile(spectrum[np.isfinite(spectrum)],[2,99])
        image=Image.fromarray((255*np.clip((spectrum-lo)/max(float(hi-lo),1e-12),0,1)).astype('uint8'))
    elif mode=='comparison':
        ctf=json.loads((storage_path(folder, 'ctf.json')).read_text())
        image=comparison_image(np.load(storage_path(folder, 'ctf_spectrum.npy'),allow_pickle=False),ctf)
    elif mode=='landscape':
        ctf=json.loads((storage_path(folder, 'ctf.json')).read_text())
        geometry=storage_path(folder, 'ctf_geometry.json')
        if geometry.exists():ctf['local_geometry']=json.loads(geometry.read_text())
        with mrcfile.mmap(service.path(job['stack']),mode='r') as handle:
            height,width=handle.data.shape[-2:]
        options=dict(job.get('options',{}))
        # Display the estimated field belonging to THIS image, not a new
        # hypothetical plane assembled from project metadata or viewer controls.
        # Rotation/flip only map display pixels back to the detector raster.
        # A handedness diagnostic does not replace a saved local CTF fit.
        values=defocus_field(ctf,record,options,height,width,binning,rotation_deg=rotation_deg,flip_y=flip_y)
        # Colour scale is fixed across the series and independent of rotation.
        # Outside the rotated image the scalar field is padded with central
        # defocus (neutral), just as the image receives background padding.
        lo,hi=series_defocus_range(output,records,options,height,width,binning)
        fraction=np.clip((values-float(ctf['defocus_A'])-lo)/(hi-lo),0,1)
        pink=np.array([244,92,176]);blue=np.array([57,132,245]);neutral=np.array([230,230,235])
        signed=2*fraction-1;strength=np.abs(signed)[...,None]
        endpoint=np.where((signed<0)[...,None],pink,blue)
        rgb=neutral*(1-strength)+endpoint*strength
        image=Image.fromarray(rgb.astype('uint8'))
    else:
        raise InputError('Unknown CTF view.')
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()
