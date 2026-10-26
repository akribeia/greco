"""Exact cyclic search-region geometry in the pre-flip correlation frame."""
from itertools import product


def cyclic_intervals(fast_shape,tomo_shape):
    if len(fast_shape)!=3 or len(tomo_shape)!=3:raise ValueError('Picker ROI must be three-dimensional.')
    axes=[]
    for size,length in zip(fast_shape,tomo_shape):
        if not 0<length<=size:raise ValueError('Invalid picker ROI dimensions.')
        start=(size-length+size//2+1)%size
        if size==length:axes.append(((0,size),))
        else:
            first=min(length,size-start)
            axes.append(((start,first),(0,length-first)) if first<length else ((start,first),))
    return tuple(axes)


def roi_slices(fast_shape,tomo_shape):
    return [tuple(slice(start,start+length) for start,length in box)
            for box in product(*cyclic_intervals(fast_shape,tomo_shape))]
