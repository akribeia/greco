"""Backprojection statistics for raw and CTF-premultiplied observations.

Greco premultiplied input stores Y and an effective forward transfer T².
Its reconstruction accumulates Y/variance and T²/variance, NOT Y*T² and
T⁴. Keeping these sufficient statistics explicit avoids division at CTF zeros.
"""

def backprojection_statistics(data,transfer,precision,premultiplied=False):
    if premultiplied:
        return data*precision,transfer*precision
    return data*transfer*precision,transfer**2*precision
