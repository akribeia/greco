"""Greco postprocessing tables and vector PDF report, without plotting dependencies."""
from pathlib import Path
import math
import numpy as np
from ..particle_star import write_loop
from .._numerics.io.mrc import write_mrc, application_label


def finite(value):
    value=float(value)
    return value if math.isfinite(value) else None


def publish(pp):
    """Publish evaluated arrays; this function performs no map processing."""
    root=Path(pp.args.o)
    maps=root/'volumes';tables=root
    maps.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    write_mrc(str(maps/'map.mrc'),pp.I1,pixel_size=pp.angpix,title=application_label())
    if pp.do_mask:
        write_mrc(str(maps/'map_masked.mrc'),pp.I1*pp.Im,pixel_size=pp.angpix,title=application_label())
    n=pp.I1.shape[-1];a=pp.args
    general=dict(grecoFormatName='greco-postprocess-v1',grecoFormatVersion=1,
        grecoBoxSize=n,grecoPixelSizeAngst=pp.angpix,grecoResolutionAngst=finite(pp.global_resol),
        grecoBfactorAngstSquared=finite(pp.global_bfactor),grecoMaskCorrected=bool(pp.do_mask),
        grecoHalfMap1=str(a.i),grecoHalfMap2=str(a.i2 or a.i.replace('half1','half2')),
        grecoMap=str(maps/'map.mrc'),grecoLowPassAngst=finite(pp.applied_filter))
    if a.interpolate:general['grecoInterpolatedResolutionAngst']=finite(pp.fract_resol)
    if pp.do_mask:
        general.update(grecoMask=str(a.mask),grecoMaskedMap=str(maps/'map_masked.mrc'),
            grecoMaskFraction=finite(pp.frac_solvent_mask),grecoRandomizeFromAngst=n*pp.angpix/pp.randomize_at)
    if pp.molweight>0:general.update(grecoMolecularWeightKDa=pp.molweight,grecoMolecularWeightFraction=finite(pp.frac_molweight))
    if a.auto_bfac:
        general.update(grecoGuinierFitSlope=finite(pp.global_slope),grecoGuinierFitIntercept=finite(pp.global_intercept),grecoGuinierFitCorrelation=finite(pp.global_corr_coeff))
    curves={'corrected':pp.fsc_true,'unmasked':pp.fsc_unmasked}
    if pp.do_mask:curves.update(masked=pp.fsc_masked,randomized_mask=pp.fsc_random_masked,mask_fraction=pp.fsc_part_fracmask)
    if pp.molweight>0:curves['molecular_weight']=pp.fsc_part_molweight
    labels={'corrected':'grecoFscCorrected','unmasked':'grecoFscUnmasked','masked':'grecoFscMasked',
        'randomized_mask':'grecoFscRandomizedMask','mask_fraction':'grecoFscMaskFraction','molecular_weight':'grecoFscMolecularWeight'}
    with (tables/'analysis.star').open('w') as stream:
        write_loop(stream,'general',list(general),[list(general.values())])
        write_loop(stream,'fsc',['grecoShell','grecoSpatialFrequencyInvAngst','grecoResolutionAngst',*[labels[k] for k in curves]],
            [(i,i/(n*pp.angpix),n*pp.angpix/i if i else None,*[finite(v[i]) for v in curves.values()]) for i in range(len(pp.fsc_true))])
    x,y,_=pp.guinierin;guinier={'Original':y}
    if a.mtf:guinier['MtfCorrected']=pp.guinierinvmtf[1]
    if pp.do_fsc_weighting:guinier['Weighted']=pp.guinierweighted[1]
    if a.auto_bfac or abs(a.adhoc_bfac)>0:guinier['Sharpened']=pp.guiniersharpen[1]
    if a.auto_bfac:guinier['Fit']=pp.global_slope*np.asarray(x)+pp.global_intercept
    with (tables/'analysis.star').open('a') as stream:
        write_loop(stream,'guinier',['grecoSpatialFrequencySquaredInvAngstSquared',*['grecoLogAmplitude'+k for k in guinier]],
            [(finite(v),*[finite(values[i]) for values in guinier.values()]) for i,v in enumerate(x)])
    write_pdf(tables/'report.pdf',general,curves,np.arange(len(pp.fsc_true))/(n*pp.angpix),guinier,x)
    return general


def update_summary(path,report):
    """Add worker execution metadata to the single scientific table."""
    from ..particle_star import rows
    blocks={}
    for block,row in rows(path):blocks.setdefault(block,[]).append(row)
    general=blocks['general'][0]
    general.update(grecoBackend=report.get('backend','cpu'),grecoElapsedSeconds=report.get('elapsed_seconds'),
        grecoSourceMask=report.get('mask_path'),grecoEvaluator=report.get('evaluator','Greco PostProcess'))
    temporary=Path(path).with_suffix('.star.tmp')
    with temporary.open('w') as stream:
        for block,records in blocks.items():
            columns=list(records[0])
            write_loop(stream,block,columns,([r.get(k) for k in columns] for r in records))
    temporary.replace(path)


def read_analysis(path):
    """Read the Greco postprocessing summary and FSC for the web viewer."""
    from ..particle_star import rows
    general=None;curves={}
    labels={'grecoFscCorrected':'corrected','grecoFscUnmasked':'unmasked','grecoFscMasked':'masked',
        'grecoFscRandomizedMask':'randomized_mask','grecoFscMaskFraction':'mask_fraction','grecoFscMolecularWeight':'molecular_weight'}
    for block,row in rows(path):
        if block=='general':general=row
        elif block=='fsc':
            for label,name in labels.items():
                if label in row:curves.setdefault(name,[]).append(dict(shell=row['grecoShell'],resolution_A=row['grecoResolutionAngst'],fsc=row[label]))
    if general is None or general.get('grecoFormatName')!='greco-postprocess-v1':raise ValueError('Invalid Greco postprocessing analysis.')
    return dict(resolution_A=general['grecoResolutionAngst'],interpolated_resolution_A=general.get('grecoInterpolatedResolutionAngst'),
        bfactor_A2=general['grecoBfactorAngstSquared'],mask_path=general.get('grecoSourceMask',general.get('grecoMask')),
        mask_corrected=bool(general['grecoMaskCorrected']),backend=general.get('grecoBackend','cpu'),
        elapsed_seconds=general.get('grecoElapsedSeconds'),evaluator=general.get('grecoEvaluator','Greco PostProcess'),curves=curves)


class _Page:
    def __init__(self):self.ops=[];self.images=[]
    def rect(self,x,y,w,h,color):self.ops.append(f'{color} rg {x} {y} {w} {h} re f')
    def text(self,x,y,text,size=10,color='.12 .18 .26',bold=False):
        text=str(text).replace('\\','\\\\').replace('(','\\(').replace(')','\\)').replace('\n',' ')
        self.ops.append(f'BT /{"F2" if bold else "F1"} {size} Tf {color} rg {x} {y} Td ({text}) Tj ET')
    def line(self,points,color,width=1):
        if len(points)<2:return
        path=f'{points[0][0]:.3f} {points[0][1]:.3f} m '+' '.join(f'{x:.3f} {y:.3f} l' for x,y in points[1:])
        self.ops.append(f'{color} RG {width} w {path} S')
    def frame(self,subtitle,page,total=2,section="POSTPROCESS"):
        self.rect(0,0,595,842,'.975 .981 .987');self.rect(0,738,595,104,'.07 .12 .18')
        self.rect(36,774,4,32,'.26 .80 .66');self.text(52,790,'GRECO',23,'1 1 1',True)
        self.text(52,768,section+' / '+subtitle,10,'.68 .78 .85')
        self.line([(36,48),(559,48)],'.83 .87 .90');self.text(36,30,'Greco  /  Scientific report',9,'.42 .48 .55')
        self.text(527,30,f'{page} / {total}',9,'.42 .48 .55')


def _chart(page,x,y,w,h,series,xlabel,ylabel,ybounds=None,threshold=False,invalid_floor=None,legend=True):
    colors=['.06 .56 .43','.24 .45 .77','.73 .40 .17','.61 .37 .70','.38 .46 .51']
    valid=[(name,np.asarray(xs),np.asarray(ys)) for name,xs,ys in series]
    points=[(xx,yy) for _,xs,ys in valid for xx,yy in zip(xs,ys) if np.isfinite(xx) and np.isfinite(yy) and (invalid_floor is None or yy>invalid_floor)]
    if not points:
        page.text(x+15,y+h/2,'No finite curve samples available');return
    xmin=min(v[0] for v in points);xmax=max(v[0] for v in points)
    ymin,ymax=ybounds or (min(v[1] for v in points),max(v[1] for v in points))
    if xmax<=xmin:xmax=xmin+1
    if ymax<=ymin:ymax=ymin+1
    if not ybounds:
        pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad
    X=lambda v:x+(v-xmin)/(xmax-xmin)*w
    Y=lambda v:y+(v-ymin)/(ymax-ymin)*h
    page.rect(x,y,w,h,'1 1 1')
    for i in range(6):
        yy=y+i*h/5;xx=x+i*w/5
        page.line([(x,yy),(x+w,yy)],'.88 .91 .93',.5)
        page.text(x-31,yy-3,f'{ymin+i*(ymax-ymin)/5:.2g}',8,'.42 .48 .55')
        page.text(xx-9,y-16,f'{xmin+i*(xmax-xmin)/5:.3g}',8,'.42 .48 .55')
    if threshold:
        page.line([(x,Y(.143)),(x+w,Y(.143))],'.60 .66 .70',.7)
        page.text(x+w-61,Y(.143)+5,'FSC 0.143',8,'.42 .48 .55')
    page.ops.append(f'q {x} {y} {w} {h} re W n')
    for j,(_,xs,ys) in enumerate(valid):
        segment=[]
        for xx,yy in zip(xs,ys):
            if np.isfinite(xx) and np.isfinite(yy) and (invalid_floor is None or yy>invalid_floor):segment.append((X(xx),Y(yy)))
            else:page.line(segment,colors[j%len(colors)],1.6);segment=[]
        page.line(segment,colors[j%len(colors)],1.6)
    page.ops.append('Q');page.text(x,y+h+17,ylabel,10,bold=True);page.text(x,y-36,xlabel,9,'.42 .48 .55')
    for j,(name,_,_) in enumerate(valid if legend else []):
        col=j%2;row=j//2;lx=x+col*w/2;ly=y-64-row*18
        page.line([(lx,ly+3),(lx+15,ly+3)],colors[j%len(colors)],2)
        page.text(lx+22,ly,name,9)


def write_pdf(path,general,curves,frequencies,guinier,squared_frequencies):
    pages=[_Page(),_Page()]
    p=pages[0];p.frame('RESOLUTION & FSC',1)
    values=[('RESOLUTION',general['grecoResolutionAngst'],'A'),('SHARPENING B',general['grecoBfactorAngstSquared'],'A^2'),('SAMPLING',general['grecoPixelSizeAngst'],'A / px')]
    for i,(label,value,unit) in enumerate(values):
        px=36+i*178;p.rect(px,646,166,68,'1 1 1');p.text(px+12,693,label,8,'.42 .48 .55',True)
        p.text(px+12,664,('N/A' if value is None else f'{value:.2f}')+' '+unit,17,bold=True)
    p.text(36,616,'Fourier shell correlation',16,bold=True)
    names={'corrected':'Corrected' if general['grecoMaskCorrected'] else 'Unmasked','unmasked':'Unmasked','masked':'Masked','randomized_mask':'Phase-randomized · masked'}
    series=[(names[k],frequencies,v) for k,v in curves.items() if k in names and (k!='unmasked' or general['grecoMaskCorrected'])]
    _chart(p,72,245,465,320,series,'Spatial frequency (1/A)','Correlation',(-.1,1.05),True)
    p.text(36,118,'ANALYSIS',9,'.06 .56 .43',True)
    p.text(36,98,'Phase-randomization mask correction' if general['grecoMaskCorrected'] else 'Unmasked half-map FSC',10)
    p.text(36,80,'Threshold: 0.143  |  Box: '+str(general['grecoBoxSize'])+' px',9,'.42 .48 .55')
    p=pages[1];p.frame('SHARPENING & OUTPUTS',2)
    p.text(36,696,'Guinier analysis',19,bold=True)
    _chart(p,72,330,465,295,[(k,squared_frequencies,v) for k,v in guinier.items()],
        'Squared spatial frequency (1/A^2)','Log amplitude',invalid_floor=-99)
    p.text(36,195,'OUTPUTS',9,'.06 .56 .43',True)
    for i,label in enumerate(['volumes/map.mrc - processed map',*(['volumes/map_masked.mrc - masked map'] if general['grecoMaskCorrected'] else []),'analysis.star - summary, FSC and Guinier curves']):
        p.text(36,174-i*17,label,9)
    p.text(36,71,'Tables retain full precision; the report rounds values for display.',9,'.42 .48 .55')
    write_document(path,pages)


def write_document(path,pages):
    """Write vector text/charts and optional RGB images as a standard PDF."""
    import zlib
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'',b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>',b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>']
    kids=[]
    for page in pages:
        resources=[]
        for name,array in page.images:
            array=np.ascontiguousarray(array,dtype=np.uint8);height,width=array.shape[:2]
            payload=zlib.compress(array.tobytes())
            index=len(objects)+1
            objects.append(f'<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(payload)} >>\nstream\n'.encode()+payload+b'\nendstream')
            resources.append(f'/{name} {index} 0 R')
        number=len(objects)+1;kids.append(f'{number} 0 R')
        objects.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> /XObject << {" ".join(resources)} >> >> /Contents {number+1} 0 R >>'.encode())
        stream='\n'.join(page.ops).encode('latin-1','replace');objects.append(f'<< /Length {len(stream)} >>\nstream\n'.encode()+stream+b'\nendstream')
    objects[1]=f'<< /Type /Pages /Count {len(pages)} /Kids [{" ".join(kids)}] >>'.encode()
    data=bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n');offsets=[0]
    for i,obj in enumerate(objects,1):offsets.append(len(data));data.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref=len(data);data.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:data.extend(f'{offset:010d} 00000 n \n'.encode())
    data.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    import os,tempfile
    fd,temporary=tempfile.mkstemp(prefix='.'+Path(path).stem+'-',suffix='.tmp',dir=Path(path).parent)
    try:
        os.fchmod(fd,Path(path).parent.stat().st_mode & 0o666)
        with os.fdopen(fd,'wb') as stream:stream.write(data)
        os.replace(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)
