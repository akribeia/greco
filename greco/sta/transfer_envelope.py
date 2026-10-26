"""Greco local tilt scale and Grant--Grigorieff radiation-damage envelope."""
from greco.project_layout import join as storage_path
import numpy as np


MODEL='grant_grigorieff_2015'


def view_parameters(view):
    model=view.get('grecoDoseWeightingModel','none')
    if model not in ('none',MODEL):raise ValueError('Unsupported dose weighting model.')
    scale=float(view.get('grecoCtfScaleFactor',1.))
    dose=float(view.get('grecoPreExposureElectronPerAngst2',0.))
    if not np.isfinite(scale) or scale<=0 or not np.isfinite(dose) or dose<0:
        raise ValueError('Invalid tilt transfer scale or pre-exposure.')
    if model==MODEL and 'grecoPreExposureElectronPerAngst2' not in view:
        raise ValueError('Dose weighting requires explicit pre-exposure.')
    return scale,dose if model==MODEL else 0.


def envelope(scales,doses,n,pixel,xp=np):
    """Amplitude transfer, not an extra precision weight; DC damage = 1."""
    scales=xp.asarray(scales);doses=xp.asarray(doses)
    y=xp.fft.fftfreq(n)[:,None]/pixel;x=xp.arange(n//2+1)[None,:]/(n*pixel)
    frequency=xp.sqrt(x*x+y*y)
    critical=.245*xp.maximum(frequency,1e-30)**(-1.665)+2.81
    return (scales[:,None,None]*xp.exp(-.5*doses[:,None,None]/critical)).astype(xp.float32)


def extraction_doses(preprocessing,rows):
    """Read complete acquisition, including views not accepted for extraction."""
    import json
    import re
    from datetime import datetime
    from pathlib import Path
    from ..mdoc import parse_mdoc
    prep=Path(preprocessing);config=json.loads((storage_path(prep, 'config.json')).read_text())
    options=config['options'];mdoc=parse_mdoc(storage_path(prep, 'input.mdoc'));text=(storage_path(prep, 'input.mdoc')).read_text()
    dose=float(options.get('dose_per_tilt',config.get('series',{}).get('metadata',{}).get('dose_per_tilt',float('nan'))))
    if not np.isfinite(dose) or dose<=0:raise ValueError('Extraction requires a positive exposure dose.')
    exposure=np.array([t.n_frames*t.frame_dose_e_per_A2 if options.get('dose_from_mdoc',False) else dose for t in mdoc.tilts])
    prior=np.array([t.prior_dose_e_per_A2 for t in mdoc.tilts])
    # Older MDOCs omit prior dose entirely. Derive from the complete acquisition,
    # never from the angularly sorted/accepted subset. Explicit records win.
    if 'PriorRecordDose' not in text:
        dates=re.findall(r'^DateTime\s*=\s*(.+)$',text,re.MULTILINE)
        if dates:
            if len(dates)!=len(mdoc.tilts):raise ValueError('Incomplete acquisition timestamps for dose ordering.')
            times=[datetime.strptime(' '.join(s.split()),'%d-%b-%Y %H:%M:%S') for s in dates]
            if len(set(times))!=len(times):raise ValueError('Ambiguous acquisition timestamps for dose ordering.')
            order=np.argsort(times);source='acquisition_timestamps_configured_exposures'
        else:order=np.arange(len(exposure));source='acquisition_order_configured_exposures'
        prior[order]=np.cumsum(exposure[order])-exposure[order]
    else:
        if len(re.findall(r'^PriorRecordDose\s*=',text,re.MULTILINE))!=len(mdoc.tilts):
            raise ValueError('Incomplete MDOC pre-exposure records.')
        source='mdoc_prior_record_dose'
    if not np.isfinite(prior).all() or np.any(prior<0) or not np.isfinite(exposure).all() or np.any(exposure<=0):
        raise ValueError('Invalid acquisition doses.')
    result=[]
    for row in rows:
        i=int(row['mdoc_index']);retained=exposure[i]
        movie=storage_path(storage_path(storage_path(prep, 'output/per_tilt'), f'tilt_{i:04d}'), 'movie.json')
        if movie.exists():retained=float(json.loads(movie.read_text()).get('retained_dose_e_per_A2',retained))
        if not np.isfinite(retained) or retained<=0:raise ValueError('Invalid retained exposure dose.')
        result.append(dict(prior=float(prior[i]),exposure=float(retained),source=source))
    return result
