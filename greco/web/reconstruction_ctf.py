"""Final CTF diagnostics in the saved reconstruction's display geometry."""
import io
import json
from pathlib import Path
import numpy as np
from greco.project_layout import join,preprocessing_work
from greco.preprocessing_star import read_table
from .projects import InputError


def render(work,source_index,mode,base_png):
    from PIL import Image
    from .ctf_view import comparison_image
    from greco.ctf_estimator import TiltDefocusGeometry
    from greco.alignment_geometry import resolve
    from greco.gcc import raw_affine
    work=Path(work);config=json.loads((work/'config.json').read_text());binning=int(config['inputs'].get('binning',8))
    report=join(work/'alignment','ctf_refinement.json')
    if not report.is_file():raise InputError('Final CTF re-estimation is not available.')
    fitted=json.loads(report.read_text())
    prep=preprocessing_work(Path(config['inputs']['stack']))
    rows=read_table(join(prep,'ctf/ctf.star'));z=int(rows[source_index]['z_value'])
    view=fitted['views'].get(str(z))
    if view is None:raise InputError('No final CTF for this tilt.')
    ctf=dict(view['ctf'])
    ctf['defocus_A']=.5*(float(ctf['defocus_min_A'])+float(ctf['defocus_max_A']))
    if mode=='comparison':
        folder=join(join(prep,'per_tilt'),f'tilt_{z:04d}')
        image=comparison_image(np.load(join(folder,'ctf_spectrum.npy'),allow_pickle=False),ctf)
    elif mode=='landscape':
        base=np.asarray(Image.open(io.BytesIO(base_png)).convert('RGB'),dtype=float)
        height,width=base.shape[:2]
        if height!=width:raise InputError('Expected square reconstruction raster.')
        flip=bool(config['orientation'].get('flip_y',False))
        with np.load(join(work,'alignment/final_state.npz'),allow_pickle=False) as state:
            expected=np.loadtxt(config['inputs']['rawtlt'],ndmin=1)
            axis=resolve(state,expected,config['orientation']['rotation_deg'])[1][source_index]
            shift=state['cumulative_shifts_yx'][source_index]
        frozen_rows=read_table(config['inputs']['frozen_shifts']);row=frozen_rows[source_index]
        frozen=np.array([float(row['dy_raw']),float(row['dx_raw'])])/binning
        if flip:frozen[0]*=-1
        a,b,c,d,center=raw_affine(float(axis),width)
        yy,xx=np.mgrid[:height,:width].astype(float)
        py=yy-center-shift[0];px=xx-center-shift[1]
        ry=a*py+b*px+center-frozen[0];rx=c*py+d*px+center-frozen[1]
        if flip:ry=height-1-ry
        def field(v,x,y):
            geom=v['geometry'];ct=v['ctf']
            geometry=TiltDefocusGeometry(float(geom['tilt_deg']),float(geom['tilt_axis_deg']),float(ct['pixel_size_A']),int(geom['handedness']),float(geom.get('alpha_offset_deg',0)),float(geom.get('beta_offset_deg',0)),geom.get('gradient_A_per_pixel'))
            return geometry.local_mean_defocus_A(0.,x,y)
        values=field(view,(rx-center)*binning,(ry-center)*binning)
        valid=(rx>=0)&(rx<=width-1)&(ry>=0)&(ry<=height-1)
        extent=center*binning
        cx,cy=np.meshgrid([-extent,extent],[-extent,extent])
        span=max(1.,*(float(np.max(np.abs(field(v,cx,cy)))) for v in fitted['views'].values()))
        signed=np.clip(values/span,-1,1);signed=np.where(valid,signed,0.)
        strength=np.abs(signed)[...,None]
        endpoint=np.where((signed<0)[...,None],np.array([244,92,176]),np.array([57,132,245]))
        colour=np.array([230,230,235])*(1-strength)+endpoint*strength
        image=Image.fromarray(np.clip(.5*base+.5*colour,0,255).astype('uint8'))
    else:raise InputError('Unknown final CTF diagnostic.')
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()
