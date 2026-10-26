"""Rigid frame geometry + static XYZ + independent-half GP motion on device.

Matches Greco's Rz(chi) Ry(theta) Rx(phi) frame parameterization, without
its optional non-rigid image-deformation model. One rigid geometry is shared
by both half sets; each particle only sees its own reference half map.
"""
import numpy as np
from .gpu_lbfgs import minimize


def rotations_and_derivatives(angles,xp=np):
    """Batched exact Greco Tait--Bryan rotations and analytic derivatives."""
    if xp is not np:
        from .rigid_polishing_kernels import rotations_and_derivatives as fused
        return fused(angles,xp)
    angles=xp.asarray(angles);count=len(angles)
    matrices=[];derivatives=[]
    for axis in range(3):
        c=xp.cos(angles[:,axis]);s=xp.sin(angles[:,axis])
        a=xp.broadcast_to(xp.eye(3),(count,3,3)).copy();d=xp.zeros_like(a)
        i,j=((1,2),(2,0),(0,1))[axis]
        a[:,i,i]=a[:,j,j]=c;a[:,i,j]=-s;a[:,j,i]=s
        d[:,i,i]=d[:,j,j]=-s;d[:,i,j]=-c;d[:,j,i]=c
        matrices.append(a);derivatives.append(d)
    rx,ry,rz=matrices;dx,dy,dz=derivatives
    return rz@ry@rx,xp.stack((rz@ry@dx,rz@dy@rx,dz@ry@rx),axis=1)


class RigidMotionObjective:
    """One frame block plus one static/GP block per independent half set.

    The first acquisition frame is anchored to remove the arbitrary common
    rigid coordinate-frame gauge. No biological/non-rigid warp is introduced.
    """
    def __init__(self,blocks,xyz,cameras,xp=np):
        self.xp=xp;self.blocks=blocks;self.xyz=[xp.asarray(a) for a in xyz]
        self.cameras=[xp.asarray(a) for a in cameras];self.fc=blocks[0].fc
        if any(b.fc!=self.fc for b in blocks):raise ValueError('Frame count mismatch')
        self.frame_size=5*(self.fc-1)
        self.offsets=np.r_[self.frame_size,self.frame_size+np.cumsum([b.size for b in blocks])]
        self.size=int(self.offsets[-1])

    def decode(self,x):
        xp=self.xp;frame=xp.concatenate((xp.zeros((1,5)),x[:self.frame_size].reshape(-1,5)))
        rotation,derivative=rotations_and_derivatives(frame[:,:3],xp)
        return frame,rotation,derivative

    def projected(self,x,index):
        xp=self.xp;frame,q,_=self.decode(x);b=self.blocks[index]
        track,_=b.trajectory(x[self.offsets[index]:self.offsets[index+1]])
        camera=self.cameras[index]
        position=self.xyz[index][b.owners]+track[b.owners,b.frames]
        original=xp.einsum('mij,mj->mi',camera,self.xyz[index][b.owners])
        moved=xp.einsum('mij,mj->mi',camera,position)
        refined=xp.einsum('mij,mj->mi',q[b.frames],moved)
        return refined[:,:2]-original[:,:2]+frame[b.frames,3:],track

    def value_gradient(self,x):
        xp=self.xp;frame,q,dq=self.decode(x);frame_grad=xp.zeros((self.fc,5))
        total=xp.asarray(0.);grad=xp.zeros(self.size)
        for i,b in enumerate(self.blocks):
            part=x[self.offsets[i]:self.offsets[i+1]];track,coeff=b.trajectory(part)
            camera=self.cameras[i];position=self.xyz[i][b.owners]+track[b.owners,b.frames]
            original=xp.einsum('mij,mj->mi',camera,self.xyz[i][b.owners])
            moved=xp.einsum('mij,mj->mi',camera,position)
            refined=xp.einsum('mij,mj->mi',q[b.frames],moved)
            projected=refined[:,:2]-original[:,:2]+frame[b.frames,3:]
            value,detector_grad=b.detector_value_gradient(projected)
            total+=value+xp.sum((coeff+b.previous)**2)
            updated=(q[b.frames]@camera)[:,:2]
            xyz_grad=xp.einsum('mij,mi->mj',updated,detector_grad)
            grad[self.offsets[i]:self.offsets[i+1]]=b.track_gradient(xyz_grad,coeff)
            angle_change=xp.einsum('maij,mj->mai',dq[b.frames,:,:2],moved)
            angle_grad=xp.einsum('mai,mi->ma',angle_change,detector_grad)
            xp.add.at(frame_grad,b.frames,xp.concatenate((angle_grad,detector_grad),axis=1))
        grad[:self.frame_size]=frame_grad[1:].ravel()
        return total,grad

    def parameter_scale(self):
        xp=self.xp;info=xp.zeros((self.fc,5))
        for i,b in enumerate(self.blocks):
            centre=xp.full((len(b.owners),2),float(b.radius));detector=[];step=.05
            from .gp3d import keys_sample
            for axis in range(2):
                delta=xp.zeros_like(centre);delta[:,axis]=step*b.padding
                curvature=(keys_sample(b.surfaces,centre+delta,xp)[1][:,axis]-keys_sample(b.surfaces,centre-delta,xp)[1][:,axis])*b.padding/(2*step)
                detector.append(xp.abs(curvature))
            w=xp.maximum(xp.stack(detector,axis=1),1e-8)
            position=xp.einsum('mij,mj->mi',self.cameras[i],self.xyz[i][b.owners])
            _,dq=rotations_and_derivatives(xp.zeros((self.fc,3)),xp)
            jac=xp.einsum('maij,mj->mai',dq[b.frames,:,:2],position)
            angle_info=xp.sum(jac*jac*w[:,None,:],axis=-1)
            xp.add.at(info,b.frames,xp.concatenate((angle_info,w),axis=1))
        frame=1/xp.sqrt(xp.maximum(info[1:].ravel(),1e-8))
        return xp.concatenate([frame,*[b.parameter_scale() for b in self.blocks]])


def fit_rigid_motion(objective,max_iterations=10000,progress=None):
    xp=objective.xp;scale=objective.parameter_scale()
    def fun(x):
        value,g=objective.value_gradient(x*scale)
        return value,g*scale
    result=minimize(fun,xp.zeros(objective.size),xp=xp,max_iterations=max_iterations,progress=progress)
    if not np.isfinite(result.fun) or not bool(xp.isfinite(result.x).all()):raise ValueError('Non-finite rigid polishing fit')
    # Safeguarded acceptance for the entire tomogram, not individual tilts.
    fractions=(0.,.125,.25,.5,1.)
    fraction=min(fractions,key=lambda f:float(fun(f*result.x)[0]))
    parameters=fraction*result.x*scale
    return parameters,dict(iterations=int(result.nit),converged=bool(result.success),
        optimizer_message=result.message,optimizer='device_lbfgs' if xp is not np else 'numpy_lbfgs',
        optimizer_evaluations=int(result.nfev),acceptance_scale=fraction,
        initial_objective=float(fun(xp.zeros(objective.size))[0]),
        final_objective=float(objective.value_gradient(parameters)[0]))
