"""CPU-only reports from immutable CTF and motion correction STARs."""
from pathlib import Path
import copy, json
import numpy as np
from ..particle_star import rows
from .corrections import source_chain
from .postprocess_report import _Page,_chart,write_document


def collect(manifest):
    """Resolve absolute states once and retain the immediate-input comparison."""
    manifest=Path(manifest).resolve();base,chain=source_chain(manifest)
    particles={};views={};observations={};tomograms={}
    for block,row in rows(base):
        if block=='tomograms':tomograms[row['grecoTomogramId']]=row
        elif block=='views':views[(row['grecoTomogramId'],row['grecoViewId'])]=row
        elif block=='particles':particles[row['grecoParticleId']]=row
        elif block=='observations':observations[(row['grecoParticleId'],row['grecoViewId'])]={}
    before=None
    for source in reversed(chain):
        if source==manifest:before=(copy.deepcopy(particles),copy.deepcopy(views),copy.deepcopy(observations))
        for block,row in rows(source):
            if block=='particles':particles.setdefault(row['grecoParticleId'],{}).update(row)
            elif block=='view_corrections':views[(row['grecoTomogramId'],row['grecoViewId'])].update({k:v for k,v in row.items() if v is not None})
            elif block=='observations':observations.setdefault((row['grecoParticleId'],row['grecoViewId']),{}).update(row)
    if before is None:raise ValueError('Report requires a saved correction STAR, not a raw extraction.')
    return dict(base=str(base),tomograms=tomograms,before=before,after=(particles,views,observations))


def _motion(row):return np.array([float(row.get('grecoResidualShift'+d+'Angst',0.)) for d in 'XY'])
def _xyz_motion(row):return np.array([float(row.get('grecoGpMotion'+d+'Angst',0.)) for d in 'XYZ'])
def _xyz(p):return np.array([float(p['grecoCoordinate'+d+'Angst']) for d in 'XYZ'])
def _central(v):return .5*(float(v['grecoCentralDefocusUAngst'])+float(v['grecoCentralDefocusVAngst']))
def _frame(v,key):return int(v['grecoMdocIndex'])+1 if 'grecoMdocIndex' in v else int(v['grecoTiltIndex'])+1 if 'grecoTiltIndex' in v else int(key[1])
def _camera(v):return np.array([[float(v[f'grecoProjection{i}{j}']) for j in range(3)] for i in range(3)])
def _native(v):return all(f'grecoProjection{i}{j}' in v for i in range(3) for j in range(3)) and v.get('grecoDefocusHand') in (-1,1) and v.get('grecoRasterFlipY') in (0,1)
def _depth(v,xyz):return xyz@_camera(v)[2]*v['grecoDefocusHand']*(-1 if v['grecoRasterFlipY'] else 1)


def series_data(data,tid,kind):
    bp,bv,bo=data['before'];ap,av,ao=data['after']
    pids=sorted(pid for pid,p in ap.items() if p['grecoTomogramId']==tid)
    keys=sorted((k for k in av if k[0]==tid),key=lambda k:(float(av[k]['grecoTiltAngleDeg']),_frame(av[k],k)))
    shape=(len(pids),len(keys));matrices={k:np.full(shape,np.nan) for k in ['x','y','norm','total','xyz','defocus','before_offset','after_offset']}
    pi={pid:i for i,pid in enumerate(pids)};vi={key[1]:i for i,key in enumerate(keys)}
    has_xyz=False
    for (pid,vid),new in ao.items():
        if pid not in pi or vid not in vi:continue
        i,j=pi[pid],vi[vid];old=bo.get((pid,vid),{})
        delta=_motion(new)-_motion(old)
        matrices['x'][i,j],matrices['y'][i,j]=delta
        matrices['norm'][i,j]=np.linalg.norm(delta);matrices['total'][i,j]=np.linalg.norm(_motion(new))
        has_xyz=has_xyz or any('grecoGpMotion'+d+'Angst' in new for d in 'XYZ')
        matrices['xyz'][i,j]=np.linalg.norm(_xyz_motion(new)-_xyz_motion(old))
        if kind=='ctf_refinement':
            matrices['before_offset'][i,j]=float(old.get('grecoDefocusOffsetAngst',0.))
            matrices['after_offset'][i,j]=float(new.get('grecoDefocusOffsetAngst',0.))
            oldv,newv=bv[(tid,vid)],av[(tid,vid)]
            offset=float(new.get('grecoDefocusOffsetAngst',0.))-float(old.get('grecoDefocusOffsetAngst',0.))
            depth=_depth(newv,_xyz(ap[pid]))-_depth(oldv,_xyz(bp[pid])) if _native(oldv) and _native(newv) else 0.
            matrices['defocus'][i,j]=_central(newv)-_central(oldv)+offset+depth
    finite=matrices['norm'][np.isfinite(matrices['norm'])]
    return dict(pids=pids,keys=keys,x=np.array([float(av[k]['grecoTiltAngleDeg']) for k in keys]),
        angles=np.array([float(av[k].get('grecoTiltAngleDeg',0.)) for k in keys]),matrices=matrices,has_xyz=has_xyz,
        observations=len(finite),static_z=np.array([_xyz(ap[pid])[2]-_xyz(bp[pid])[2] for pid in pids]))


def _finite(v):return np.asarray(v)[np.isfinite(v)]
def _quantile(v,q):
    v=_finite(v);return float(np.quantile(v,q)) if len(v) else 0.

def _image(page,array,x,y,w,h,lo=None,hi=None,signed=False):
    a=np.asarray(array,float);finite=_finite(a)
    if lo is None:lo=float(finite.min()) if len(finite) else 0.
    if hi is None:hi=float(finite.max()) if len(finite) else 1.
    if signed:
        limit=max(abs(lo),abs(hi),1e-12);lo=-limit;hi=limit
    if hi<=lo:hi=lo+1e-12
    f=np.nan_to_num(np.clip((a-lo)/(hi-lo),0,1),nan=.5)
    if signed:
        blue=np.array([.15,.36,.69]);white=np.array([.97,.98,.99]);red=np.array([.80,.26,.21])
        rgb=np.where((f<=.5)[...,None],blue+(white-blue)*(2*f)[...,None],white+(red-white)*(2*f-1)[...,None])
    else:
        dark=np.array([.08,.15,.23]);green=np.array([.08,.65,.51]);light=np.array([.96,.85,.42])
        rgb=np.where((f<=.5)[...,None],dark+(green-dark)*(2*f)[...,None],green+(light-green)*(2*f-1)[...,None])
    rgb[~np.isfinite(a)]=[.88,.90,.92]
    name='Im'+str(len(page.images));page.images.append((name,np.rint(rgb*255).astype(np.uint8)))
    page.ops.append(f'q {w} 0 0 {h} {x} {y} cm /{name} Do Q')
    return lo,hi


def _cards(page,values):
    for i,(title,value) in enumerate(values):
        x=36+i*178;page.rect(x,640,166,68,'1 1 1');page.text(x+12,686,title,8,'.42 .48 .55',True);page.text(x+12,659,value,17,bold=True)


def publish(work,config=None,result=None):
    """Build an immutable-results report; no fitting, filtering or GPU work."""
    work=Path(work);config=config or json.loads((work/'config.json').read_text())
    result=result or json.loads((work/'result.json').read_text())
    kind=config.get('action');kind=kind if kind in ('polishing','ctf_refinement') else ('polishing' if work.parent.name=='polishing' else 'ctf_refinement')
    data=collect(work/'particles.star');pages=[]
    def new(title):
        page=_Page();pages.append((page,title));return page
    page=new('OVERVIEW');page.text(36,713,'Polishing' if kind=='polishing' else 'CTF refinement',22,bold=True)
    particles=data['after'][0];views=data['after'][1]
    _cards(page,[('PARTICLES',str(len(particles))),('TILT SERIES',str(len(data['tomograms']))),('VIEWS',str(len(views)))])
    page.text(36,604,'Job '+work.name+'  /  '+work.parent.parent.name,13,bold=True)
    notes=['Comparison: this job versus its immediate input STAR.',
        'Cumulative residuals are measured relative to the extraction baseline.',
        'Missing observations are grey; zero corrections remain visible.',
        'Plots and tables follow tilt angle, from negative to positive.']
    if kind=='ctf_refinement':notes+=['Table Delta Z means the change in mean optical defocus.',
        'It is not a fitted physical displacement of particle coordinates.',
        'Central fields include the median saved observation-defocus offset.']
    else:notes+=['All particles and measured views are included in the heatmap atlas.',
        'Individual line plots use deterministic motion-percentile examples.',
        'Detector residual X/Y, GP3D motion and rigid geometry are distinct.',
        'Heatmaps do not add camera rotations or static particle shifts to residuals.']
    for i,line in enumerate(notes):page.text(36,574-i*21,line,10)
    diagnostics=result.get('diagnostics',{});fits=diagnostics.get('fits',[])
    page.text(36,330,'FIT SUMMARY',10,'.06 .56 .43',True)
    for i,text in enumerate([f'Converged fits: {sum(bool(f.get("converged")) for f in fits)} / {len(fits)}',
        'Model: '+str(diagnostics.get('motion_model',diagnostics.get('ctf_model','saved observation corrections'))),
        'Reference: '+Path(str(result.get('reference_job',config.get('reference_job','not recorded')))).name,
        'Own-half reference maps are retained by the scientific processing workflow.']):page.text(36,306-i*20,text[:96],10)
    for tid,tomo in data['tomograms'].items():
        sd=series_data(data,tid,kind)
        if not sd['pids'] or not sd['keys']:continue
        title=str(tomo['grecoTomogramName']);m=sd['matrices'];x=sd['x'];keys=sd['keys'];pids=sd['pids']
        page=new('SERIES SUMMARY');page.text(36,713,title[:74],17,bold=True)
        measure=m['norm'] if kind=='polishing' else abs(m['defocus'])
        _cards(page,[('OBSERVATIONS',str(sd['observations'])),('MEDIAN CHANGE',f'{_quantile(measure,.5):.2f} A'),('P95 CHANGE',f'{_quantile(measure,.95):.2f} A')])
        if kind=='polishing':
            lines=[('Median residual',x,np.nanmedian(m['norm'],axis=0)),('P95 residual',x,np.nanquantile(m['norm'],.95,axis=0)),('Median cumulative',x,np.nanmedian(m['total'],axis=0))]
            if sd['has_xyz']:lines.append(('Median GP3D change',x,np.nanmedian(m['xyz'],axis=0)))
            _chart(page,72,325,465,245,lines,'Tilt angle (deg)','Motion (A)')
            bp,bv,_=data['before'];ap,av,_=data['after']
            rigid=np.array([[float(av[k].get('grecoGeometryShift'+d+'Angst',0.))-float(bv[k].get('grecoGeometryShift'+d+'Angst',0.)) for d in 'XY'] for k in keys])
            page.text(36,185,f'Maximum shared rigid translation: {np.linalg.norm(rigid,axis=1).max():.3f} A',10)
            page.text(36,164,f'P95 static particle Z change: {_quantile(abs(sd["static_z"]),.95):.3f} A',10)
            page.text(36,143,'Views are sorted by tilt angle, from negative to positive.',9)
        else:
            bv=data['before'][1];av=data['after'][1]
            before=np.array([_central(bv[k]) for k in keys])+np.nan_to_num(np.nanmedian(m['before_offset'],axis=0))
            after=np.array([_central(av[k]) for k in keys])+np.nan_to_num(np.nanmedian(m['after_offset'],axis=0))
            _chart(page,72,270,465,300,[('Input central defocus',x,before),('Refined central defocus',x,after)],'Tilt angle (deg)','Defocus (A)')
        # Atlas contains every particle, with bounded raster blocks instead of vector overload.
        fields=[('x','Incremental detector X',True),('y','Incremental detector Y',True),('xyz' if sd['has_xyz'] else 'norm','Incremental GP3D magnitude' if sd['has_xyz'] else 'Incremental detector magnitude',False),('total','Cumulative detector magnitude',False)] if kind=='polishing' else []
        for start in (range(0,len(pids),256) if fields else []):
            stop=min(start+256,len(pids));page=new('PARTICLE / TILT ATLAS');page.text(36,706,title[:74],15,bold=True)
            page.text(36,682,f'Rows {start+1}-{stop} of {len(pids)}; particle IDs {pids[start]} to {pids[stop-1]}',10)
            page.text(36,663,'Each row is one particle; columns follow increasing tilt angle.',9,'.42 .48 .55')
            height=110 if len(fields)>1 else 420;top=620
            for f,(key,label,signed) in enumerate(fields):
                y=top-f*140-height;page.text(72,y+height+8,label,10,bold=True)
                page.text(398,y+height+8,f'{x[0]:.1f} to {x[-1]:.1f} deg',8)
                values=m[key];available=_finite(values);limit=float(np.max(abs(available))) if len(available) else 1.
                lo,hi=_image(page,values[start:stop],72,y,465,height,-limit if signed else 0.,limit,signed)
                page.text(36,y+height-8,str(pids[start]),7);page.text(36,y+3,str(pids[stop-1]),7)
                _image(page,np.linspace(hi,lo,180)[:,None],548,y,7,height,lo,hi,signed)
                page.text(541,y+height+2,f'{hi:.3g}',7);page.text(541,y-10,f'{lo:.3g}',7)
            page.text(36,64,'Grey = missing observation.',8,'.42 .48 .55')
        if kind=='polishing':
            # Representative line plots complement, rather than replace, the complete atlas.
            rank=np.argsort(np.nan_to_num(np.nanmax(m['norm'],axis=1),nan=-1.),kind='stable')
            examples=np.unique(rank[np.rint(np.linspace(0,len(rank)-1,min(6,len(rank)))).astype(int)])
            for start in range(0,len(examples),2):
                page=new('PARTICLE TRAJECTORIES');page.text(36,708,title[:74],15,bold=True)
                for f,i in enumerate(examples[start:start+2]):
                    _chart(page,72,450-f*300,465,175,[('X',x,m['x'][i]),('Y',x,m['y'][i])],
                        'Tilt angle (deg)',f'Particle {pids[i]} / incremental detector motion (A)')
        # A numerical per-view appendix preserves readable values beyond the plots.
        for start in range(0,len(keys),24):
            page=new('PER-TILT VALUES');page.text(36,706,title[:74],15,bold=True)
            headers=['Frame','Tilt (deg)','Nobs','Median dX','Median dY','P95 motion'] if kind=='polishing' else ['Frame','Tilt (deg)','Input df','Refined df','Delta Z','U - V','Scale']
            positions=[36,83,143,204,307,425] if kind=='polishing' else [36,82,151,245,344,429,517]
            for position,label in zip(positions,headers):page.text(position,672,label,8,bold=True)
            for f,key in enumerate(keys[start:start+24]):
                j=start+f;view=data['after'][1][key];angle=float(view.get('grecoTiltAngleDeg',0.));frame=_frame(view,key)
                if kind=='polishing':
                    values=[str(frame),f'{angle:.2f}',str(np.isfinite(m['norm'][:,j]).sum()),f'{_quantile(m["x"][:,j],.5):+.3f}',f'{_quantile(m["y"][:,j],.5):+.3f}',f'{_quantile(m["norm"][:,j],.95):.3f}'];positions=[36,83,143,204,307,425]
                else:
                    old=data['before'][1][key];astig=float(view['grecoCentralDefocusUAngst'])-float(view['grecoCentralDefocusVAngst'])
                    values=[str(frame),f'{angle:.2f}',f'{_central(old)+_quantile(m["before_offset"][:,j],.5):.2f}',f'{_central(view)+_quantile(m["after_offset"][:,j],.5):.2f}',f'{_quantile(m["defocus"][:,j],.5):+.2f}',f'{astig:.2f}',f'{float(view.get("grecoCtfScaleFactor",1.)):.3f}'];positions=[36,82,151,245,344,429,517]
                y=645-f*22
                if f%2==0:page.rect(32,y-6,527,21,'1 1 1')
                for position,value in zip(positions,values):page.text(position,y,value,9)
            page.text(36,78,'Distances and defocus: Angstrom. Frame identities follow the saved MDOC.',9)
    total=len(pages)
    for i,(page,title) in enumerate(pages,1):
        frame=_Page();frame.frame(title,i,total,'POLISHING' if kind=='polishing' else 'CTF REFINE');page.ops=frame.ops+page.ops
    path=work/'report.pdf';write_document(path,[p for p,_ in pages])
    return dict(path=str(path),pages=total,particles=len(particles),tomograms=len(data['tomograms']))
