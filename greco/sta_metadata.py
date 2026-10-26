"""Lossless scientific STAR tables for STA histories, fits and FSC curves.

Values live in ordinary named columns. A schema records nested field paths and
container types, preserving optional fields without JSON encoded STAR cells.
"""
import copy
import re
from pathlib import Path
from .particle_star import rows,write_loop


def _path(parts):return '/'.join(str(p).replace('~','~0').replace('/','~1') for p in parts)
def _parts(path):return [] if path=='' else [p.replace('~1','/').replace('~0','~') for p in path.split('/')]
def _label(path):
    units={'A':'Angst','A2':'AngstSquared','deg':'Deg','seconds':'Seconds'}
    words=re.findall(r'[A-Za-z0-9]+',path)
    return 'greco'+(''.join(units.get(w,w[:1].upper()+w[1:]) for w in words) or 'Value')


def write(path,value,kind=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value=copy.deepcopy(value);sections=[]
    # FSC curves are separate readable loops, rather than thousands of columns.
    if isinstance(value,dict):
        def curves(node,parts=()):
            if isinstance(node,dict):
                for key,item in list(node.items()):
                    if isinstance(item,list) and item and all(isinstance(r,dict) and 'shell' in r for r in item):
                        sections.append(('curve'+str(len(sections)+1),_path((*parts,key)),'list',item));node[key]=[]
                    else:curves(item,(*parts,key))
        curves(value)
    root_kind='list' if isinstance(value,list) else 'dict' if isinstance(value,dict) else 'scalar'
    records=value if root_kind=='list' else [value]
    sections.insert(0,('values','',root_kind,records))
    schemas=[];structures=[];tables=[]
    for section,target,root,records in sections:
        fields={};used={'grecoRecordIndex'};values=[]
        for index,record in enumerate(records):
            scalars={}
            def visit(item,parts=()):
                field=_path(parts)
                typ='dict' if isinstance(item,dict) else 'list' if isinstance(item,list) else 'null' if item is None else 'bool' if isinstance(item,bool) else 'int' if isinstance(item,int) else 'float' if isinstance(item,float) else 'str'
                structures.append((section,index,field,typ))
                if typ in ('dict','list'):
                    for key,child in (item.items() if typ=='dict' else enumerate(item)):visit(child,(*parts,key))
                else:
                    if field not in fields:
                        label=_label(field);base=label;suffix=2
                        while label in used:label=base+str(suffix);suffix+=1
                        fields[field]=label;used.add(label)
                    scalars[field]=item
            visit(record);values.append(scalars)
        schemas.extend((section,label,field) for field,label in fields.items())
        tables.append((section,['grecoRecordIndex',*fields.values()],[(i,*[record.get(field) for field in fields]) for i,record in enumerate(values)]))
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        write_loop(stream,'general',['grecoFormatName','grecoFormatVersion','grecoTableKind'],[('greco-sta-metadata-v1',1,kind or path.stem)])
        write_loop(stream,'sections',['grecoSection','grecoFieldPath','grecoContainerType'],[(s,p,k) for s,p,k,_ in sections])
        write_loop(stream,'fields',['grecoSection','grecoFieldLabel','grecoFieldPath'],schemas)
        write_loop(stream,'structure',['grecoSection','grecoRecordIndex','grecoFieldPath','grecoValueType'],structures)
        for name,columns,records in tables:write_loop(stream,name,columns,records)
    temporary.replace(path)


def read(path):
    sections=[];fields={};structure=[];tables={};valid=False
    for block,row in rows(path):
        if block=='general':
            valid=row.get('grecoFormatName')=='greco-sta-metadata-v1' and row.get('grecoFormatVersion')==1
            if not valid:raise ValueError('Unknown STA scientific table format.')
        elif block=='sections':sections.append((row['grecoSection'],row['grecoFieldPath'],row['grecoContainerType']))
        elif block=='fields':fields[(row['grecoSection'],row['grecoFieldPath'])]=row['grecoFieldLabel']
        elif block=='structure':structure.append(row)
        else:tables[(block,row['grecoRecordIndex'])]=row
    if not valid:raise ValueError('Missing STA scientific table header.')
    objects={};records={}
    for row in structure:
        section,index,path,typ=(row[k] for k in ('grecoSection','grecoRecordIndex','grecoFieldPath','grecoValueType'))
        key=(section,index)
        if typ in ('dict','list'):value={} if typ=='dict' else []
        else:
            raw=tables[key][fields[(section,path)]]
            value=None if typ=='null' else bool(raw) if typ=='bool' else int(raw) if typ=='int' else float(raw) if typ=='float' else str(raw)
        objects[(section,index,path)]=value
        parts=_parts(path)
        if not parts:records[key]=value
        else:
            parent=objects[(section,index,_path(parts[:-1]))]
            if isinstance(parent,list):
                if int(parts[-1])!=len(parent):raise ValueError('Invalid STA list ordering.')
                parent.append(value)
            else:parent[parts[-1]]=value
    result=None
    for section,target,kind in sections:
        values=[records[k] for k in sorted(records) if k[0]==section]
        value=values if kind=='list' else values[0]
        if section=='values':result=value;continue
        parts=_parts(target);parent=result
        for part in parts[:-1]:parent=parent[int(part)] if isinstance(parent,list) else parent[part]
        parent[parts[-1]]=value
    return result
