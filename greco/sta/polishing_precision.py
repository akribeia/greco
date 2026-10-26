"""Select frozen half-radial or half/view-radial polishing precision."""


def precision(variance, half, shell, active, groups, xp):
    if variance.ndim == 2:
        return active / variance[half][shell]
    if variance.ndim != 3:
        raise ValueError('Expected half/shell or half/view/shell variance')
    selected = variance[half, xp.asarray(groups)]
    return active[None] / selected[:, shell]
