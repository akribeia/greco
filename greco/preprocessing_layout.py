"""Names of assembled image stacks, separate from CTF fit diagnostics."""
import re


def stack_name(name):
    match=re.fullmatch(r'greco_raw_bin(\d+)\.(mrcs|rawtlt)',name)
    if match:return f'raw_bin{match[1]}.{match[2]}'
    match=re.fullmatch(r'greco_2d_aligned_bin(\d+)\.(mrcs|rawtlt)',name)
    if match:return f'aligned_bin{match[1]}.{match[2]}'
    match=re.fullmatch(r'greco_aligned_ctf_reliability_bin(\d+)\.(mrcs|rawtlt)',name)
    if match:return f'aligned_spatial_ctf_reliability_bin{match[1]}.{match[2]}'
    match=re.fullmatch(r'reliability_bin(\d+)\.(mrcs|rawtlt)',name)
    if match:return f'spatial_ctf_reliability_bin{match[1]}.{match[2]}'
    match=re.fullmatch(r'flat_reliability_bin(\d+)\.(mrcs|json|rawtlt)',name)
    if match:return f'flat_ctf_reliability_bin{match[1]}.{match[2]}'
    match=re.fullmatch(r'greco_ctf_(?:(even|odd)_)?bin(\d+)_rawsource\.(mrcs|rawtlt)',name)
    if match:return f'spatial_ctf_{match[1]+"_" if match[1] else ""}bin{match[2]}.{match[3]}'
    return None
