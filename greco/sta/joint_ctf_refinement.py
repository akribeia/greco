"""Joint native per-tilt optics using the validated masked raw likelihood."""
from greco.project_layout import join as storage_path
import numpy as np
from .ctf_joint import fit,objective,basis,update_view
from .ctf_refinement import fit_defocus_offsets


def fit_views(dataset,backend,evidence,k1,options,output,progress):
    from ..web.particle_worker import save_json
    xp=backend.xp;n=dataset.n;pixel=dataset.pixel
    for key in dataset.group_keys:
        if any(field not in dataset.views[key] for field in ('grecoCentralDefocusUAngst','grecoCentralDefocusVAngst')):
            raise ValueError('Joint CTF requires native central-defocus metadata; re-extract these particles.')
    maximum=options.get('maximum_defocus_A',3000.)
    initial,_,_=fit_defocus_offsets(evidence,k1,n,pixel,maximum,options.get('defocus_step_A',25.),xp)
    corrections=xp.zeros((len(k1),4),xp.float64);diagnostics=[]
    for tid in dict.fromkeys(key[0] for key in dataset.group_keys):
        ids=np.array([g for g,key in enumerate(dataset.group_keys) if key[0]==tid])
        ev=evidence[ids]
        regularization=.1*(n//2+1)*n/1e6*float(np.mean(k1[ids])**2)
        name=dataset.tomos[tid]['grecoTomogramName']
        candidate,optimizer=fit(ev,k1[ids],n,pixel,initial[ids],regularization=regularization,
            fit_scale=True,max_iterations=1000,xp=xp,
            progress=lambda iteration:progress('Joint CTF: '+name,optimizer_iteration=iteration))
        pb=basis(n,pixel,k1[ids],xp);fractions=[0.,.125,.25,.5,1.];costs=[]
        for fraction in fractions:
            state=candidate*fraction
            valid=bool(xp.all(xp.abs(state[:,0])<=maximum)) and bool(xp.all(xp.linalg.norm(state[:,1:3],axis=1)<=2000.)) and bool(xp.all(xp.abs(state[:,3])<=np.log(4.)))
            costs.append(float(objective(state.ravel(),ev,pb,regularization,True,xp)[0]) if valid else None)
        chosen=int(np.argmin([float('inf') if value is None else value for value in costs]))
        corrections[ids]=candidate*fractions[chosen]
        diagnostics.append(dict(tomogram=name,fraction=fractions[chosen],costs=costs,
            regularization=regularization,iterations=optimizer.nit,optimizer_message=optimizer.message,
            converged=bool(optimizer.success)))
        save_json(storage_path(output, 'ctf_fits.json'),diagnostics)
    updates=backend.host(corrections);np.save(storage_path(output, 'optical_corrections.npy'),updates)
    views=[]
    for g,key in enumerate(dataset.group_keys):
        if 'grecoCentralDefocusUAngst' not in dataset.views[key] or 'grecoCentralDefocusVAngst' not in dataset.views[key]:
            raise ValueError('Joint CTF requires native central-defocus metadata; re-extract these particles.')
        view=update_view(dataset.views[key],updates[g])
        for d in 'XY':view['grecoGeometryShift'+d+'Angst']=view.get('grecoGeometryShift'+d+'Angst',0.)
        views.append(view)
    return views,dict(ctf_model='joint_mean_astigmatism_scale',astigmatism_refined=True,
        contrast_scale_refined=True,beam_gradient_preserved=True,gpu_optimizer=backend.gpu,
        defocus_offsets_A=updates[:,0].tolist(),astig_cos_offsets_A=updates[:,1].tolist(),
        astig_sin_offsets_A=updates[:,2].tolist(),relative_contrast_scales=np.exp(updates[:,3]).tolist(),
        groups=dataset.group_keys,fits=diagnostics,
        median_abs_defocus_offset_A=float(np.median(abs(updates[:,0]))))
