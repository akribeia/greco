"""Greco STAR 1.0: explicit scientific metadata alongside real MRCS patches.

See docs/greco-particle-star-v1.md. Labels include their STAR leading underscore
on disk; Python dictionaries omit it. No implicit JSON or NumPy image archive.
"""
from greco.project_layout import join as storage_path
import math
from pathlib import Path
import re

FORMAT='greco-star-mrcs-v1'


def encode(value):
    if value is None:return '?'
    if isinstance(value,bool):return str(int(value))
    if isinstance(value,int):return str(value)
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('STAR numeric fields must be finite.')
        return format(value,'.17g')
    value=str(value)
    if '\x00' in value:raise ValueError('NUL is not allowed in STAR text.')
    try:float(value);numeric=True
    except ValueError:numeric=False
    if value and not numeric and not any(c.isspace() for c in value) and not any(c in value for c in "'\"#") and not value.startswith(('_',';')) and value not in ('?','.') and not value.lower().startswith(('data_','loop_','stop_','save_','global_')):return value
    for quote in ("'",'"'):
        if '\n' not in value and '\r' not in value and not re.search(re.escape(quote)+r'(?=\s|$)',value):return quote+value+quote
    if any(line.startswith(';') for line in value.splitlines()):raise ValueError('STAR multiline text cannot contain a semicolon at column 1.')
    return ';\n'+value+'\n;'


def _write_loop(stream,name,columns,rows,*,label_prefix='greco'):
    stream.write('\ndata_'+name+'\n\nloop_\n')
    for i,column in enumerate(columns,1):
        if not re.fullmatch(re.escape(label_prefix)+r'[A-Za-z0-9]+',column):raise ValueError('Invalid Greco STAR label.')
        stream.write('_'+column+' #'+str(i)+'\n')
    for row in rows:
        if len(row)!=len(columns):raise ValueError('STAR column count mismatch.')
        line=[]
        for item in row:
            token=encode(item)
            if token.startswith(';\n'):
                if line:stream.write(' '.join(line)+'\n');line=[]
                stream.write(token+'\n')
            else:line.append(token)
        if line:stream.write(' '.join(line)+'\n')


def tokens(stream):
    """STAR single/double quoted and semicolon-delimited text; no escapes."""
    multiline=None
    for line in stream:
        if multiline is not None:
            if line.startswith(';'):
                yield '\n'.join(multiline),True;multiline=None
            else:multiline.append(line.rstrip('\n'))
            continue
        if line.startswith(';'):
            multiline=[]
            if line[1:].rstrip('\n'):multiline.append(line[1:].rstrip('\n'))
            continue
        if "'" not in line and '"' not in line:
            for token in line.split('#',1)[0].split():yield token,False
            continue
        i=0
        while i<len(line):
            if line[i].isspace():i+=1;continue
            if line[i]=='#':break
            if line[i] in "'\"":
                quote=line[i];start=i+1;i=start
                while i<len(line) and not (line[i]==quote and (i+1==len(line) or line[i+1].isspace())):i+=1
                if i==len(line):raise ValueError('Unterminated STAR quote.')
                yield line[start:i],True;i+=1
            else:
                start=i
                while i<len(line) and not line[i].isspace():i+=1
                yield line[start:i],False
    if multiline is not None:raise ValueError('Unterminated STAR multiline text.')


def _rows(path):
    """Stream (block, row dict), without holding observation tables in RAM."""
    block=None;columns=[];values=[];header=False
    with Path(path).open(encoding='utf-8') as stream:
        for value,quoted in tokens(stream):
            control=not quoted and (value.startswith('data_') or value=='loop_' or value=='stop_')
            if control:
                if values:raise ValueError('Incomplete STAR row.')
                if value.startswith('data_'):block=value[5:];columns=[];header=False
                elif value=='loop_':columns=[];header=True
                else:header=False;columns=[]
                continue
            if not quoted and value.startswith('_') and header:
                if value[1:] in columns:raise ValueError('Duplicate STAR label.')
                columns.append(value[1:]);continue
            if block is None or not columns:raise ValueError('Expected a STAR data block and loop.')
            header=False
            if not quoted:
                if value in ('?','.'):value=None
                else:
                    try:value=int(value)
                    except ValueError:
                        try:value=float(value)
                        except ValueError:pass
            values.append(value)
            if len(values)==len(columns):yield block,dict(zip(columns,values));values=[]
        if values:raise ValueError('Incomplete STAR row at EOF.')


def header(path):
    general=None;tomograms=[]
    for block,row in rows(path):
        if block=='general':
            if general is not None:raise ValueError('Multiple STAR general rows.')
            general=row
        elif block=='tomograms':tomograms.append(row)
        else:break
    if general and general.get('grecoFormatName') in ('greco-refinement-v1','greco-observation-refinement-v1'):
        from .sta.corrections import source_chain
        base,_=source_chain(path)
        return header(base)
    if not general or general.get('grecoFormatVersion')!=1 or general.get('grecoFormatName')!=FORMAT:
        raise ValueError('Unsupported Greco particle STAR format.')
    if not tomograms:raise ValueError('The STAR has no tomograms.')
    return general,tomograms


def image_path(star,relative):
    root=Path(star).resolve().parent
    relative=Path(str(relative))
    if relative.is_absolute():raise ValueError('STAR image paths must be relative.')
    target=(storage_path(root, relative)).resolve()
    if not target.is_relative_to(root):raise ValueError('STAR image path escapes its extraction directory.')
    return target


def image_sources(star):
    """Enumerate the per-particle stacks of the source extraction."""
    from .sta.corrections import source_chain
    star,_=source_chain(star)
    header(star)
    sources=[]
    for block,row in rows(star):
        if block=='observations':break
        if block=='particles' and row.get('grecoImageStack') is not None:
            sources.append(image_path(star,row['grecoImageStack']))
    return sources


TILT_INDEX_LABELS={'grecoTiltIndex','grecoMdocIndex'}


def write_loop(stream,name,columns,records):
    """Write human-facing tilt identities starting at one; callers use zero."""
    columns=list(columns)
    indexed=TILT_INDEX_LABELS.intersection(columns)
    if not indexed:
        return _write_loop(stream,name,columns,records)
    source_columns=columns
    columns=[c for c in columns if c!='grecoTiltIndexBase']+['grecoTiltIndexBase']
    def converted():
        for record in records:
            if len(record)!=len(source_columns):raise ValueError('STAR column count mismatch.')
            row=dict(zip(source_columns,record))
            for label in indexed:
                if row[label] is not None:row[label]=int(row[label])+1
            row['grecoTiltIndexBase']=1
            yield [row[c] for c in columns]
    return _write_loop(stream,name,columns,converted())


def rows(path,*,raw=False):
    """Read zero-based internal tilt identities; raw=True exposes disk values."""
    for block,row in _rows(path):
        if not raw:
            base=row.pop('grecoTiltIndexBase',None)
            if TILT_INDEX_LABELS.intersection(row) and base!=1:
                raise ValueError('STAR tilt indices must use the current one-based format.')
            for label in TILT_INDEX_LABELS.intersection(row):
                if row[label] is not None:row[label]=int(row[label])-base
        yield block,row
