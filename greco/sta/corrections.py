"""Immutable observation corrections and particle poses over raw extraction STARs."""
from greco.project_layout import join as storage_path
from pathlib import Path
from ..particle_star import rows,FORMAT

FORMATS={'greco-refinement-v1','greco-observation-refinement-v1','greco-classification-v1'}


def source_chain(path):
    current=Path(path).resolve();chain=[];seen=set()
    for _ in range(32):
        if current in seen:raise ValueError('Circular particle STAR source chain.')
        seen.add(current)
        general=next((r for b,r in rows(current) if b=='general'),{})
        if general.get('grecoFormatName')==FORMAT:return current,chain
        if general.get('grecoFormatName') not in FORMATS:raise ValueError('Unsupported corrected particle STAR.')
        chain.append(current)
        current=Path(general['grecoSourceParticleStar'])
        if not current.is_absolute():current=storage_path(chain[-1].parent, current)
        current=current.resolve()
    raise ValueError('Particle STAR source chain is too deep.')


def correction_tables(path):
    base,chain=source_chain(path);particles={};observations={}
    for source in reversed(chain):
        seen_particles=set();seen_observations=set()
        for block,row in rows(source):
            if block=='particles':
                pid=row['grecoParticleId']
                if pid in seen_particles:raise ValueError('Duplicate corrected particle identity.')
                seen_particles.add(pid);particles.setdefault(pid,{}).update(row)
            elif block=='observations':
                key=(row['grecoParticleId'],row['grecoViewId'])
                if key in seen_observations:raise ValueError('Duplicate corrected observation identity.')
                seen_observations.add(key)
                observations.setdefault(key,{}).update(row)
    return base,chain,particles,observations


def corrected_views(path):
    """Absolute rigid view states; newest state wins, never add rotations."""
    _,chain=source_chain(path);result={}
    for source in reversed(chain):
        for block,row in rows(source):
            if block!='view_corrections':continue
            key=(row['grecoTomogramId'],row['grecoViewId'])
            result.setdefault(key,{}).update({k:v for k,v in row.items() if v is not None})
    return result


def write_corrections(path,source,particles,observations,kind,views=None):
    from ..particle_star import write_loop
    seen_observations=set()
    for row in observations:
        if any(row.get(c) is None for c in ('grecoParticleId','grecoViewId')):
            raise ValueError('Observation correction requires explicit particle/view identity.')
        key=(row['grecoParticleId'],row['grecoViewId'])
        if key in seen_observations:raise ValueError('Duplicate corrected observation identity.')
        seen_observations.add(key)
    temporary=Path(path).with_suffix('.star.tmp')
    records=[]
    for p in particles:
        row=dict(p)
        if 'grecoImageStack' in row:row['grecoSourceImageStack']=row.pop('grecoImageStack')
        records.append(row)
    with temporary.open('w') as stream:
        write_loop(stream,'general',['grecoFormatName','grecoFormatVersion','grecoSourceParticleStar','grecoPosesRefined','grecoCorrectionKind'],
            [('greco-observation-refinement-v1',1,str(Path(source).resolve()),1,kind)])
        columns=list(records[0]);write_loop(stream,'particles',columns,[tuple(r[c] for c in columns) for r in records])
        columns=['grecoParticleId','grecoViewId','grecoResidualShiftXAngst','grecoResidualShiftYAngst','grecoDefocusOffsetAngst']
        columns += [c for c in ('grecoGpMotionXAngst','grecoGpMotionYAngst','grecoGpMotionZAngst') if any(c in row for row in observations)]
        write_loop(stream,'observations',columns,[tuple(r.get(c,0.) for c in columns) for r in observations])
        geometry=corrected_views(source)
        if views is not None:
            for row in views:
                key=(row['grecoTomogramId'],row['grecoViewId'])
                geometry.setdefault(key,{}).update(row)
        if geometry:
            columns=['grecoTomogramId','grecoViewId']+[f'grecoProjection{i}{j}' for i in range(3) for j in range(3)]+['grecoGeometryShiftXAngst','grecoGeometryShiftYAngst']
            optical=('grecoCentralDefocusUAngst','grecoCentralDefocusVAngst','grecoDefocusAngleDeg','grecoCtfScaleFactor')
            columns += [c for c in optical if any(c in r for r in geometry.values())]
            write_loop(stream,'view_corrections',columns,[tuple(r.get(c) for c in columns) for r in geometry.values()])
    temporary.replace(path)


def reextraction_tables(path):
    """Recover total crop corrections, including those already baked into MRCS.

    Applied* labels are provenance only during ordinary dataset loading.
    They are needed when replaying the original MC images for a NEW extraction.
    """
    base,chain,particles,observations=correction_tables(path)
    for block,row in rows(base):
        if block!='observations':continue
        applied=[row.get('grecoAppliedResidualShift'+d+'Angst',0.) for d in 'XY']+[row.get('grecoAppliedDefocusOffsetAngst',0.)]
        if not any(applied):continue
        key=(row['grecoParticleId'],row['grecoViewId'])
        value=observations.setdefault(key,dict(grecoParticleId=key[0],grecoViewId=key[1]))
        for name,delta in zip(('grecoResidualShiftXAngst','grecoResidualShiftYAngst','grecoDefocusOffsetAngst'),applied):value[name]=value.get(name,0.)+delta
    return base,chain,particles,observations
