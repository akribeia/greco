"""Gradient-aware refinement of tiled CTF spectra.

Uses the Greco spectrum layout, tile-quality gating, handedness comparison
and alpha/beta search. Tile spectra use motion-corrected native sampling.
Upstream implementation provenance is recorded in licenses/THIRD_PARTY_NOTICES.md.
"""
from greco.project_layout import join as storage_path
import json
from pathlib import Path
import numpy as np
from .ctf_estimator import _CTFEstimator, TiltCTF, TiltDefocusGeometry

DEFOCUS_CONVENTION='physical_local_defocus_v2'


def current_hand_test(orientation):
    """Accept handedness tests only for the current physical-gradient convention."""
    return bool(orientation.get('df_hand_tested',False)) and orientation.get('df_hand_convention')==DEFOCUS_CONVENTION


def fixed_handedness(config):
    """Resolve the already validated job choice, without reselecting its sign."""
    if config.get('ctf_fixed_hand',False) or current_hand_test(config.get('orientation') or {}):
        return int(config['options']['df_hand'])
    return None


def scaled_average(tiles, centers, stds, central, geometry, xp=np, cache=None):
    """Pull local spectra onto the central-defocus frequency grid.

    Greco layout is (tile, tile//2+1), transposed relative to the other port.
    Divide by all image tiles, not only the accepted subset.
    """
    n=tiles.shape[1]
    mean=float(np.mean(stds));std=float(np.std(stds))
    good=(stds>0)&(stds>=max(mean-2*std,.2*mean))
    defocus=geometry.local_mean_defocus_A(central,centers[:,0],centers[:,1])
    if xp is not np:
        from .ctf_gpu import scaled_spectrum_sum
        selected=np.flatnonzero(good & (defocus>0))
        scales=np.sqrt(central/defocus[selected])
        xi=np.clip(np.trunc(np.arange(n//2+1,dtype=np.float64)[None,:]*scales[:,None]),0,n//2).astype(np.int32)
        yi=(np.clip(np.trunc((np.arange(n,dtype=np.float64)-n//2)[None,:]*scales[:,None]),-n//2,n//2-1)+n//2).astype(np.int32)
        return scaled_spectrum_sum(tiles,selected,yi,xi,xp,cache)
    out=xp.zeros(tiles.shape[1:],dtype=xp.float32)
    for i in np.flatnonzero(good & (defocus>0)):
        # chi ~ defocus * frequency**2: local_df * source_k**2
        # must equal central_df * output_k**2. This is an inverse sampling
        # map, not a forward image magnification. Dividing by this factor
        # reverses the inferred physical defocus gradient/hand.
        source_scale=float(np.sqrt(central/defocus[i]))
        xi=np.clip(np.trunc(np.arange(n//2+1)*source_scale).astype(np.int64),0,n//2)
        yi=np.clip(np.trunc((np.arange(n)-n//2)*source_scale).astype(np.int64),-n//2,n//2-1)+n//2
        tile=xp.asarray(tiles[i],dtype=xp.float32)
        out+=tile[xp.asarray(yi)[:,None],xp.asarray(xi)[None,:]]
    return out/len(tiles)


def test_defocus_hand(folders, initial, *, pixel_A, voltage_kV, cs_mm,
                      amp_contrast, axis_deg, tile_size=512, gpu=True,
                      res_low_A=None, res_high_A=None):
    """Read-only native-style +/- trials with identical defocus profiling.

    The reported uncertainty is a paired tilt diagnostic, not a calibrated
    posterior probability. Zero tilt carries no handedness information.
    """
    if len(folders)!=len(initial):raise ValueError('CTF folders/results mismatch.')
    est=_CTFEstimator(pixel_A=pixel_A,voltage_kV=voltage_kV,cs_mm=cs_mm,
        amp_contrast=amp_contrast,tile_size=tile_size,gpu=gpu,
        res_low_A=res_low_A,res_high_A=res_high_A)
    if gpu:
        import cupy as xp
    else:xp=np
    from .ctf_gpu import TileSpectrumCache
    cache=TileSpectrumCache(xp)
    rows=[]
    for folder,ref in zip(folders,initial):
        if not 0.1<abs(ref.tilt_deg)<=35.9:continue
        folder=Path(folder)
        with np.load(storage_path(folder, 'ctf_tiles_metadata.npz'),allow_pickle=False) as data:
            centers,stds=data['centers'],data['stds']
        if not np.any(stds>0):continue
        tiles=np.load(storage_path(folder, 'ctf_tiles.npy'),mmap_mode='r',allow_pickle=False)
        row=dict(tilt_deg=ref.tilt_deg)
        for hand in (1,-1):
            geometry=TiltDefocusGeometry(ref.tilt_deg,axis_deg,pixel_A,hand,0,0)
            clean=est.clean(scaled_average(tiles,centers,stds,ref.df_mean_A,geometry,xp,cache))
            sb=clean.reshape(-1)[est.idx2]
            p=np.array([ref.df_mean_A,ref.astig_ratio,ref.astig_azimuth_deg-90,ref.phase_deg,-1e20,1e20])
            est._refine_param(p,sb,max(p[0]-2500,2000),p[0]+2500,50.,0)
            row[str(hand)]=float(p[4])
        if np.isfinite([row['1'],row['-1']]).all():rows.append(row)
    if len(rows)<3:raise ValueError('Handedness test needs at least three usable nonzero tilts within ±35.9 degrees.')
    delta=np.array([r['1']-r['-1'] for r in rows])
    mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    decisive=abs(mean)>max(2*se,1e-6)
    return dict(handedness=(1 if mean>0 else -1) if decisive else None,
        preferred_hand=1 if mean>0 else -1,conclusive=decisive,
        hand_scores={'+1':float(np.mean([r['1'] for r in rows])),'-1':float(np.mean([r['-1'] for r in rows]))},
        paired_difference=mean,paired_standard_error=se,n_views=len(rows),per_tilt=rows,
        axis_deg=float(axis_deg),defocus_convention=DEFOCUS_CONVENTION,
        method='physical frequency pullback; matched defocus profiling; alpha=beta=0')


def resolve_raster_axis(folders, initial, *, axis_deg, **params):
    """Resolve the acquisition/movie quarter-turn using measured spectra.

    MDOC describes acquisition coordinates, not necessarily the stored movie
    raster. Compare both mappings with BOTH signs so changing the requested
    hand cannot change the selected raster mapping. Keep one physical tilt
    model for the series; this is not a free per-image defocus plane.
    """
    usable=sorted((i for i,r in enumerate(initial) if 10<=abs(r.tilt_deg)<=35.9),
                  key=lambda i:initial[i].tilt_deg)
    if len(usable)<3:
        return float(axis_deg),dict(source='insufficient_tilt_information',offset_deg=0.)
    indices=[usable[i] for i in np.unique(np.linspace(0,len(usable)-1,min(6,len(usable))).round().astype(int))]
    trials=[test_defocus_hand([folders[i] for i in indices],[initial[i] for i in indices],
                             axis_deg=float(axis_deg)+offset,**params) for offset in (0.,90.)]
    signs=[str(t['preferred_hand']) for t in trials]
    scores=[np.array([r[sign] for r in t['per_tilt']]) for t,sign in zip(trials,signs)]
    difference=scores[1]-scores[0]
    gain=float(difference.mean())
    error=float(difference.std(ddof=1)/np.sqrt(len(difference)))
    selected=bool(gain>max(2*error,1e-6))
    offset=90. if selected else 0.
    return float(axis_deg)+offset,dict(source='local_tile_spectra_raster_mapping',
        input_axis_deg=float(axis_deg),offset_deg=offset,
        candidate_scores=[float(x.mean()) for x in scores],
        paired_difference=gain,paired_standard_error=error,n_views=len(indices),
        conclusive=abs(gain)>max(2*error,1e-6))


def refine_local_series(folders, initial, *, pixel_A, voltage_kV, cs_mm,
                        amp_contrast, axis_deg, tile_size=512, gpu=True,
                        res_low_A=None,res_high_A=None,forced_hand=None,write_outputs=True,axes_deg=None):
    if not folders or len(folders)!=len(initial):
        raise ValueError('Local CTF inputs/results must match.')
    if axes_deg is None:
        axis_deg,raster_mapping=resolve_raster_axis(folders,initial,axis_deg=axis_deg,
            pixel_A=pixel_A,voltage_kV=voltage_kV,cs_mm=cs_mm,amp_contrast=amp_contrast,
            tile_size=tile_size,gpu=gpu,res_low_A=res_low_A,res_high_A=res_high_A)
        axes=np.full(len(initial),axis_deg)
    else:
        axes=np.asarray(axes_deg,float)
        if axes.shape!=(len(initial),) or not np.isfinite(axes).all():
            raise ValueError('Final CTF axes must match the supplied views.')
        raster_mapping=dict(source='final_alignment_geometry',offset_deg=0.)
    print(f'Local CTF raster axis: {axis_deg:.3f} deg ({raster_mapping})',flush=True)
    est=_CTFEstimator(pixel_A=pixel_A,voltage_kV=voltage_kV,cs_mm=cs_mm,
                      amp_contrast=amp_contrast,tile_size=tile_size,gpu=gpu,
                      res_low_A=res_low_A,res_high_A=res_high_A)
    if gpu:
        import cupy as xp
    else:
        xp=np
    from .ctf_gpu import TileSpectrumCache
    cache=TileSpectrumCache(xp)
    low=[i for i,r in enumerate(initial) if abs(r.tilt_deg)<=35.9]
    if not low:
        raise ValueError('Local geometry refinement requires low-tilt views.')
    folders=[Path(folder) for folder in folders]
    metadata=[]
    for folder in folders:
        with np.load(storage_path(folder, 'ctf_tiles_metadata.npz'),allow_pickle=False) as item:
            metadata.append((item['centers'],item['stds']))

    def evaluate(reference, hand, alpha, beta, indices):
        result=list(reference);spectra={}
        for i in indices:
            ref=reference[i]
            geometry=TiltDefocusGeometry(ref.tilt_deg,float(axes[i]),pixel_A,hand,alpha,beta)
            tiles=np.load(storage_path(folders[i], 'ctf_tiles.npy'),mmap_mode='r',allow_pickle=False)
            centers,stds=metadata[i]
            raw=scaled_average(tiles,centers,stds,ref.df_mean_A,geometry,xp,cache)
            clean=est.clean(raw)
            sb=clean.reshape(-1)[est.idx2]
            p=np.array([ref.df_mean_A,ref.astig_ratio,ref.astig_azimuth_deg-90,ref.phase_deg,-1e20,1e20])
            est._refine_param(p,sb,max(p[0]-2500,2000),p[0]+2500,50.,0)
            p[5]=est._ctf_res(p,clean)
            result[i]=TiltCTF(ref.tilt_deg,p[0]*(1-p[1]),p[0]*(1+p[1]),p[2]+90,p[3],p[4],p[5])
            spectra[i]=est.dev.asnumpy(clean)
        score=float(np.mean([result[i].score for i in low]))
        return score,result,spectra

    trials=[evaluate(initial,hand,0.,0.,range(len(initial))) for hand in (1,-1)]
    if forced_hand not in (None,1,-1):raise ValueError('Invalid fixed defocus hand.')
    winner=int(trials[1][0]>trials[0][0]) if forced_hand is None else int(forced_hand==-1)
    hand=(1,-1)[winner]
    best,results,_=trials[winner];alpha=beta=0.
    diagnostic=dict(hand_scores={'+1':trials[0][0],'-1':trials[1][0]},search=[],raster_mapping=raster_mapping)
    for which,step,count in [('alpha',3.1,22),('alpha',1.1,9),('beta',3.,7),('beta',1.1,7)]:
        reference=list(results)
        baseline,baseline_results,_=evaluate(reference,hand,alpha,beta,low)
        best=baseline;results=baseline_results
        origin=alpha if which=='alpha' else beta
        best_offset=origin
        for j in range(count):
            offset=origin+(j-count//2)*step
            if abs(offset)>30:continue
            a,b=(offset,beta) if which=='alpha' else (alpha,offset)
            score,candidate,_=evaluate(reference,hand,a,b,low)
            if score>best:
                best=score;results=candidate;best_offset=offset
        if which=='alpha':alpha=best_offset
        else:beta=best_offset
        diagnostic['search'].append(dict(parameter=which,step=step,value=best_offset,score=best))
        print(f'Local CTF {which}: {best_offset:+.2f}°, score {best:.5f}',flush=True)
    _,results,spectra=evaluate(results,hand,alpha,beta,range(len(initial)))
    model=dict(model='greco_tilt_induced_z_v1',tilt_axis_deg=float(axis_deg),handedness=hand,
               coordinate_origin='image_pixel_centres',particle_gradient_model='saved_defocus_plane',
               coordinate_frame='mc_detector_xy',gradient_axis_convention='degrees_from_x_y_down',
               alpha_offset_deg=float(alpha),beta_offset_deg=float(beta),
               defocus_convention=DEFOCUS_CONVENTION,
               source='tile_spectrum_refinement',diagnostics=diagnostic)
    if axes_deg is not None:model['tilt_axes_deg']=axes.tolist()
    if write_outputs:
        for i,folder in enumerate(folders):
            np.save(storage_path(folder, 'ctf_spectrum.npy'),spectra[i])
            (storage_path(folder, 'ctf_geometry.json')).write_text(json.dumps(dict(model,tilt_axis_deg=float(axes[i]),tilt_deg=results[i].tilt_deg,pixel_size_A=pixel_A),indent=2))
    return results,model
