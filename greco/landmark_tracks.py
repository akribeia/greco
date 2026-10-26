"""Contiguous angular support for reconstruction landmarks."""
import numpy as np


def longest_run(row):
    best=(0,0);start=length=0
    for i,value in enumerate(np.asarray(row,bool)):
        if value:
            if not length:start=i
            length+=1
            if length>best[1]:best=(start,length)
        else:length=0
    return best
