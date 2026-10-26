"""Standalone rigid-tilt image objective; not connected to web submission.

Each view has ONE shift pair and ONE inclination. The axis is common.
Frozen leave-one-out models are compared as images, never localized by a
centroid or independently shifted to their best per-landmark CC peak.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def safe_pairs(positions,angles,axis,shifts,frozen,valid,size,*,radius=16,
               axis_bound=1.,tilt_bound=.5,shift_bound=2.):
    from .helpers import projected_centres
    from .gcc import raw_sample_bases
    safe=np.asarray(valid,bool).copy()
    # Check extrema and the current pose. Small bounded angular intervals
    # receive an additional 1-pixel safety allowance for interior extrema.
    for da in (-axis_bound,0,axis_bound):
        a=np.deg2rad(axis+da);extent=radius*(abs(np.cos(a))+abs(np.sin(a)))+1
        for dt in (-tilt_bound,0,tilt_bound):
            centres=projected_centres(positions,angles+dt,size)
            for dy in (-shift_bound,shift_bound):
                for dx in (-shift_bound,shift_bound):
                    by,bx=raw_sample_bases(centres,shifts+[dy,dx],frozen,axis+da,size)
                    safe &= (by>=extent)&(bx>=extent)&(by<=size-1-extent)&(bx<=size-1-extent)
    return safe


def freeze_references(stack,positions,angles,axis,shifts,frozen,valid,*,radius=16,batch=8,fixed_safe=False):
    """Build one immutable LOO reference bank; keep 3-D models for verification."""
    import cupy as cp
    from .helpers import projected_centres,soft_spherical_mask
    from .gcc import raw_sample_bases,raw_patch_stacks_gpu
    from .local_loo_gpu import BatchedLocalLOOWBPGPU
    from .local_projection_gpu import project_local_volumes_gpu_device,project_volume_patches_gpu_device
    from .angular_transfer_gpu import compensate_angular_transfer_gpu_device
    from .tomography import gpu_reconstruct_wbp_batch_device
    n=stack.shape[-1];s=2*radius+1
    by,bx=raw_sample_bases(projected_centres(positions,angles,n),shifts,frozen,axis,n)
    patches,inside=raw_patch_stacks_gpu(stack,by,bx,valid,radius,axis)
    support=valid&inside;ids=np.flatnonzero(support.sum(axis=1)>=2)
    loo=BatchedLocalLOOWBPGPU(patches[ids],angles,support[ids],batch_size=batch)
    safe=support.copy() if fixed_safe else safe_pairs(positions,angles,axis,shifts,frozen,support,n,radius=radius)
    impulse=cp.zeros((s,s,s),cp.float32);impulse[radius,radius,radius]=1
    iv=cp.stack([project_volume_patches_gpu_device(impulse,[[0,0,0]],float(a),
        observation_radius=radius,model_radius=radius)[0] for a in angles])
    mask=cp.asarray(soft_spherical_mask(s,radius=radius-2,edge=3));cache={}
    kk=[];tt=[];volumes=[];templates=[];derivatives=[]
    for target,angle in enumerate(angles):
        for start in range(0,len(ids),batch):
            local,models=loo.target_models(target,np.arange(start,min(start+batch,len(ids))))
            landmarks=ids[local];keep=safe[landmarks,target]
            landmarks=landmarks[keep];models=models[keep]
            if not len(landmarks):continue
            transfer=[]
            for k in landmarks:
                vv=support[k].copy();vv[target]=False;key=vv.tobytes()
                if key not in cache:
                    if len(cache)>=2*len(angles):cache.pop(next(iter(cache)))
                    psf=gpu_reconstruct_wbp_batch_device(iv[None].copy(),angles,valid_bt=vv[None])[0]
                    h=cp.fft.fftn(cp.fft.ifftshift(psf));cache[key]=(h/cp.maximum(cp.max(cp.abs(h)),1e-12)).astype(cp.complex64)
                transfer.append(cache[key])
            models=compensate_angular_transfer_gpu_device(models,cp.stack(transfer),.05,4.,.8)*mask[None]
            project=lambda a:project_local_volumes_gpu_device(models,float(a),observation_size=s,model_radius=radius)
            base=project(angle);derivative=(project(angle+.1)-project(angle-.1))/.2
            kk.extend(landmarks.tolist());tt.extend([target]*len(landmarks))
            volumes.append(models);templates.append(base);derivatives.append(derivative)
        print(f'Frozen LOO references {target+1}/{len(angles)}',flush=True)
    counts=np.bincount(tt,minlength=len(angles))
    if np.any(counts<8):raise ValueError('Rigid image fit needs at least eight safe landmarks per tilt.')
    return dict(k=np.array(kk),t=np.array(tt),volumes=cp.concatenate(volumes),
        templates=cp.concatenate(templates),derivatives=cp.concatenate(derivatives),counts=counts)


class RigidImageObjective:
    def __init__(self,stack,positions,angles,axis,shifts,frozen,k,t,templates,derivatives,*,device='cuda',refine_xyz=False,translation_scale=2.):
        import torch
        from .registration import cosine_disk_mask
        self.torch=torch;self.device=device
        tensor=lambda x,dtype:torch.as_tensor(x,dtype=dtype,device=device)
        self.stack=tensor(stack,torch.float32)[None,None]
        self.positions=tensor(positions,torch.float64);self.angles=tensor(angles,torch.float64)
        self.refine_xyz=refine_xyz;self.translation_scale=float(translation_scale)
        # Gauge only: exclude a rigid translation/rotation of the entire cloud.
        # Individual landmark coordinates still have 3L-6 free directions.
        centred=np.asarray(positions)-np.mean(positions,axis=0)
        modes=[]
        for e in np.eye(3):
            modes.append(np.broadcast_to(e,centred.shape).ravel())
            modes.append(np.cross(np.broadcast_to(e,centred.shape),centred).ravel())
        u,s,_=np.linalg.svd(np.stack(modes,axis=1),full_matrices=False)
        self.xyz_gauge=tensor(u[:,s>max(s[0],1)*1e-10],torch.float64)
        self.axis=float(axis);self.shifts=tensor(shifts,torch.float64);self.frozen=tensor(frozen,torch.float64)
        self.k=tensor(k,torch.long);self.t=tensor(t,torch.long)
        self.templates=tensor(templates,torch.float32);self.derivatives=tensor(derivatives,torch.float32)
        self.n=stack.shape[-1];self.size=templates.shape[-1];self.views=len(angles)
        self.anchor=int(np.argmin(np.abs(angles)));self.tilt_ids=np.delete(np.arange(self.views),self.anchor)
        yy,xx=np.mgrid[:self.size,:self.size]-(self.size-1)/2
        self.yy=tensor(yy,torch.float64);self.xx=tensor(xx,torch.float64)
        self.weights=tensor(cosine_disk_mask(self.size,radius=self.size//2-2,edge=3),torch.float32)
        self.counts=torch.bincount(self.t,minlength=self.views).to(torch.float32)
        # Fixed opposite checkerboard of pixel blocks is diagnostic holdout,
        # never a per-landmark displacement, visibility gate or pruning rule.
        y,x=np.mgrid[:self.size,:self.size]
        self.fit_weights=self.weights*tensor(((y//4+x//4)%2)==0,torch.float32)
        self.eval_weights=self.weights*tensor(((y//4+x//4)%2)==1,torch.float32)

    def decode(self,p,axis_free,tilts_free):
        torch=self.torch;ds=p[:2*self.views].reshape(self.views,2)*self.translation_scale
        index=2*self.views;axis=p.new_tensor(self.axis)
        if axis_free:axis=axis+p[index];index+=1
        dt=torch.zeros(self.views,dtype=p.dtype,device=p.device)
        if tilts_free:dt=dt.scatter(0,torch.as_tensor(self.tilt_ids,device=p.device),p[index:index+self.views-1]*.5)
        return ds,axis,dt

    def xyz_update(self,p,axis_free,tilts_free):
        if not self.refine_xyz:return self.torch.zeros_like(self.positions)
        offset=2*self.views+int(axis_free)+(self.views-1 if tilts_free else 0)
        raw=p[offset:]*self.translation_scale
        return (raw-self.xyz_gauge@(self.xyz_gauge.T@raw)).reshape(self.positions.shape)

    def observations(self,p,axis_free,tilts_free):
        torch=self.torch
        import torch.nn.functional as F
        ds,axis,dt=self.decode(p,axis_free,tilts_free)
        pos=(self.positions+self.xyz_update(p,axis_free,tilts_free))[self.k];theta=(self.angles[self.t]+dt[self.t])*np.pi/180
        centre=(self.n-1)/2
        py=pos[:,1]-self.shifts[self.t,0]-ds[self.t,0]-centre
        px=(pos[:,2]-centre)*torch.cos(theta)-(pos[:,0]-centre)*torch.sin(theta)-self.shifts[self.t,1]-ds[self.t,1]
        a=axis*np.pi/180;ca,sa=torch.cos(a),torch.sin(a)
        ry=ca*(py[:,None,None]+self.yy)+sa*(px[:,None,None]+self.xx)+centre-self.frozen[self.t,0,None,None]
        rx=-sa*(py[:,None,None]+self.yy)+ca*(px[:,None,None]+self.xx)+centre-self.frozen[self.t,1,None,None]
        tz=(2*self.t.to(torch.float64)/(self.views-1)-1)[:,None,None].expand_as(rx)
        grid=torch.stack((2*rx/(self.n-1)-1,2*ry/(self.n-1)-1,tz),dim=-1).to(torch.float32)[None]
        return F.grid_sample(self.stack,grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0]

    def correlations(self,p,axis_free,tilts_free,*,heldout=False,exact_templates=None):
        torch=self.torch
        obs=self.observations(p,axis_free,tilts_free)
        _,_,dt=self.decode(p,axis_free,tilts_free)
        model=self.templates+self.derivatives*dt[self.t,None,None].to(torch.float32) if exact_templates is None else exact_templates
        weights=self.eval_weights if heldout else self.fit_weights
        def ncc(a,b):
            total=weights.sum();a=a-(a*weights).sum((-2,-1),keepdim=True)/total
            b=b-(b*weights).sum((-2,-1),keepdim=True)/total
            numerator=(weights*a*b).sum((-2,-1))
            denominator=torch.sqrt((weights*a*a).sum((-2,-1))*(weights*b*b).sum((-2,-1))).clamp_min(1e-12)
            return (numerator/denominator).clamp(-1,1)
        # Identical central derivative operator and weights on both patches.
        def gradients(image):
            import torch.nn.functional as F
            padded=F.pad(image[:,None],(1,1,1,1),mode='replicate')[:,0]
            return .5*(padded[:,2:,1:-1]-padded[:,:-2,1:-1]),.5*(padded[:,1:-1,2:]-padded[:,1:-1,:-2])
        my,mx=gradients(model);oy,ox=gradients(obs)
        return .5*ncc(model,obs)+.25*ncc(my,oy)+.25*ncc(mx,ox)

    def evaluate(self,p,axis_free,tilts_free,*,gradient=True):
        torch=self.torch
        q=torch.tensor(p,dtype=torch.float64,device=self.device,requires_grad=gradient)
        score=self.correlations(q,axis_free,tilts_free)
        error=1-score
        robust=torch.sqrt(error*error+.15**2)-.15
        data=(robust/self.counts[self.t]).sum()/self.views
        end=2*self.views+int(axis_free)+(self.views-1 if tilts_free else 0)
        angular=q[2*self.views:end]
        # Weak bounded angular prior; no centroid or per-landmark variables.
        prior=.001*(angular*angular).sum()/max(1,len(angular))+.0001*(q[:2*self.views]**2).mean()
        if self.refine_xyz:prior=prior+.0001*(q[end:]**2).mean()
        loss=data.to(torch.float64)+prior
        if not gradient:return float(loss.detach().cpu())
        loss.backward()
        return float(loss.detach().cpu()),q.grad.detach().cpu().numpy()

    def fit(self,axis_free=False,tilts_free=False,*,start=None,maxiter=100,bounds=None):
        import time
        count=2*self.views+int(axis_free)+(self.views-1 if tilts_free else 0)
        if self.refine_xyz:count+=self.positions.numel()
        p=np.zeros(count)
        if start is not None:p[:len(start)]=start
        started=time.monotonic();history=[]
        def callback(x):
            if len(history)%10==0:print(f'Rigid image iteration {len(history)+1}: objective {self.evaluate(x,axis_free,tilts_free,gradient=False):.6f}',flush=True)
            history.append(x.copy())
        initial=self.evaluate(p,axis_free,tilts_free,gradient=False)
        result=minimize(lambda x:self.evaluate(x,axis_free,tilts_free),p,jac=True,method='L-BFGS-B',bounds=bounds if bounds is not None else [(-1,1)]*count,
            callback=callback,options=dict(maxiter=maxiter,ftol=1e-9,gtol=1e-5,maxls=20))
        q=self.torch.tensor(result.x,dtype=self.torch.float64,device=self.device)
        ds,axis,dt=self.decode(q,axis_free,tilts_free)
        with self.torch.no_grad():
            fit_score=float(self.correlations(q,axis_free,tilts_free).mean().cpu())
            heldout=float(self.correlations(q,axis_free,tilts_free,heldout=True).mean().cpu())
        return result.x,dict(axis_rotation_deg=float(axis.cpu()),axis_change_deg=float(axis.cpu())-self.axis,
            tilt_changes_deg=dt.cpu().numpy(),shift_changes_yx=ds.cpu().numpy(),
            fit_score=fit_score,heldout_score=heldout,objective_before=initial,objective_after=float(result.fun),
            optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),
            runtime_s=time.monotonic()-started,at_bound_indices=np.flatnonzero(np.abs(result.x)>.99).tolist(),
            xyz_fixed=not self.refine_xyz,xyz_changes_zyx=self.xyz_update(q,axis_free,tilts_free).cpu().numpy(),per_landmark_shift_parameters=0)

    def exact_templates(self,bank,refined_angles):
        import cupy as cp
        import torch
        from .local_projection_gpu import project_local_volumes_gpu_device
        output=cp.empty_like(bank['templates']);radius=self.size//2
        for view,angle in enumerate(refined_angles):
            ids=np.flatnonzero(bank['t']==view)
            for start in range(0,len(ids),8):
                chosen=ids[start:start+8]
                output[chosen]=project_local_volumes_gpu_device(bank['volumes'][chosen],float(angle),observation_size=self.size,model_radius=radius)
        return torch.utils.dlpack.from_dlpack(output)

