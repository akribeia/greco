"""Artifact locations for a single picking job in a category and series."""

def artifact_relative(parts):
    p=list(parts)
    if not p:return p
    # Matching writes through the same resolver as extraction and publication.
    if p[0]=='matching' and len(p)==2 and p[1].endswith(('_scores.mrc','_angles.mrc','_refined.mrc','_job.json')):p=p[1:]
    name=p[0]
    names={'template.mrc':['inputs','template.mrc'],'mask.mrc':['inputs','mask.mrc'],
           'examples.png':['diagnostics','examples.png']}
    if name in names:return [*names[name],*p[1:]]
    for suffix,target in (('_noise_scores.mrc','noise_scores.mrc'),('_scores.mrc','scores.mrc'),('_angles.mrc','angles.mrc'),('_refined.mrc','refined.mrc'),('_job.json','config.json')):
        if name.endswith(suffix):return ['matching',target,*p[1:]]
    return p
