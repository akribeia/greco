"""Optional antithetic noise correction for particle picking."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SymmetricNoiseBank:
    rows: object


def half_range_max(P,bank,m_flat,mw,chunk_p=2048,chunk_a=2048):
    """Mean of maxima for R and -R: (max C_R - min C_R)/2.

    Both extrema share the same matrix product and positive local SPR.
    A rotation-independent offset of the null template's correlations
    cancels, without averaging templates or selecting a favorable seed.
    """
    import torch
    from .hybrid import local_spr
    rows=bank.rows;n=len(P);dev=P.device
    upper=torch.full((n,),float('-inf'),device=dev)
    lower=torch.full((n,),float('inf'),device=dev)
    spr=local_spr(P,m_flat,mw)
    for a0 in range(0,len(rows),chunk_a):
        for p0 in range(0,n,chunk_p):
            p1=min(p0+chunk_p,n)
            values=(rows[a0:a0+chunk_a] @ P[p0:p1].T)/spr[p0:p1][None,:]
            upper[p0:p1]=torch.maximum(upper[p0:p1],values.max(dim=0).values)
            lower[p0:p1]=torch.minimum(lower[p0:p1],values.min(dim=0).values)
    return ((upper-lower)*.5).cpu().numpy(),np.zeros(n,dtype=np.int64)
