"""Picking STAR schema using the shared Greco tokenizer and full precision writer."""
import os,stat,tempfile
from pathlib import Path
import numpy as np
from .particle_star import rows,write_loop,tokens

COLUMNS=(['grecoCoordinate'+a+'Angst' for a in 'XYZ']+
    ['grecoAngle'+a for a in ('Rot','Tilt','Psi')]+
    ['grecoScore','grecoPickingCutoff','grecoPickingSearchStd','grecoPixelSizeAngst','grecoTomogramName'])


def _columns(path,block):
    """Read loop headers with the common tokenizer, including empty particle tables."""
    current=None;header=False;columns=[]
    with Path(path).open() as stream:
        for value,quoted in tokens(stream):
            if not quoted and value.startswith('data_'):
                if current==block and columns:return columns
                current=value[5:];header=False;columns=[]
            elif not quoted and value=='loop_':header=True;columns=[]
            elif header and not quoted and value.startswith('_'):columns.append(value[1:])
            elif header:
                if current==block:return columns
                header=False
    return columns if current==block else []


def read_particles(path):
    columns=_columns(path,'particles')
    records=[];general=None
    for block,record in rows(path):
        if block=='general':
            if general is not None:raise ValueError('Multiple picking general records.')
            general=record
        elif block=='particles':records.append(record)
    if any(not label.startswith('greco') for label in columns):raise ValueError('Unsupported picking STAR labels.')
    if general is not None:
        if (general.get('grecoFormatName')!='greco-picking-v1' or general.get('grecoFormatVersion')!=1
            or general.get('grecoCoordinateConvention')!='centered_xyz_angstrom'
            or general.get('grecoCoordinateOrigin')!='shape_xyz_divided_by_2'
            or general.get('grecoAngleConvention')!='ZYZ_degrees'):
            raise ValueError('Unsupported picking STAR convention.')
    return columns,records


def write_particles(path,columns,records,*,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(columns)
    if any(not label.startswith('greco') for label in columns):raise ValueError('Unsupported picking STAR labels.')
    if len(set(columns))!=len(columns):raise ValueError('Duplicate picking STAR labels.')
    general=dict(grecoFormatName='greco-picking-v1',grecoFormatVersion=1,
        grecoCoordinateConvention='centered_xyz_angstrom',
        grecoCoordinateOrigin='shape_xyz_divided_by_2',
        grecoAngleConvention='ZYZ_degrees',**(metadata or {}))
    if 'grecoParticleId' not in columns:
        columns=['grecoParticleId',*columns];records=((i,*record) for i,record in enumerate(records,1))
    mode=stat.S_IMODE(path.stat().st_mode) if path.exists() else stat.S_IMODE(path.parent.stat().st_mode)&0o666
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    os.chmod(tmp,mode)
    try:
        with os.fdopen(fd,'w') as stream:
            write_loop(stream,'general',list(general),[list(general.values())])
            write_loop(stream,'particles',columns,records)
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()


def coordinates_xyz(path,volume):
    """Convert explicit centered Angstrom coordinates to source MRC raster pixels."""
    import mrcfile
    _,particles=read_particles(path)
    xyz=np.asarray([[p['grecoCoordinate'+a+'Angst'] for a in 'XYZ'] for p in particles],float).reshape(-1,3)
    with mrcfile.open(volume,header_only=True) as m:
        shape=np.array([m.header.nx,m.header.ny,m.header.nz],float);pixel=float(m.voxel_size.x)
    if not np.isfinite(pixel) or pixel<=0 or not np.isfinite(xyz).all():raise ValueError('Invalid picking coordinates or sampling.')
    for block,record in rows(path):
        if block=='general':
            declared=[record.get('grecoTomogramSize'+a) for a in 'XYZ']
            if any(v is not None for v in declared) and declared!=shape.tolist():raise ValueError('Picking STAR and tomogram dimensions differ.')
            break
        if block=='particles':break
    if any(not np.isclose(p['grecoPixelSizeAngst'],pixel,rtol=1e-6,atol=0) for p in particles):
        raise ValueError('Picking STAR and tomogram sampling differ.')
    return xyz/pixel+shape/2


def metadata_for(config,shape_xyz):
    return dict(grecoTomogramName=config['series_name'],grecoTomogramUid=config.get('series_id'),grecoTomogramPath=config['volume'],
        grecoSourceReconstruction=config.get('picking_source_work'),grecoPixelSizeAngst=float(config['voxel_size_A']),
        **{'grecoTomogramSize'+a:int(n) for a,n in zip('XYZ',shape_xyz)})


def input_manifest(config,work):
    """Input manifest for a Greco picking workspace."""
    work=Path(work)
    return dict(series_id=config.get('series_id'),series_name=config['series_name'],
        volume=config['volume'],template=str(work/'inputs/template.mrc'),mask=str(work/'inputs/mask.mrc'),
        source_template=config['template']['path'],source_mask=config['mask']['path'],
        source_reconstruction=config.get('picking_source_work'))
