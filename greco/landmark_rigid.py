"""Landmark adapter for the exact rigid-frame objective used by polishing."""
import numpy as np
from .sta.gp3d import MotionObjective
from .sta.rigid_polishing import RigidMotionObjective,fit_rigid_motion,rotations_and_derivatives


def initial_cameras(angles,axis):
    values=np.zeros((len(angles),3));values[:,1]=-np.deg2rad(angles)
    ry=rotations_and_derivatives(values)[0]
    rz=rotations_and_derivatives(np.array([[0.,0.,np.deg2rad(axis)]]))[0][0]
    return rz[None]@ry


def cloud_gauge(xyz,xp=np):
    xyz=np.asarray(xyz,float);centered=xyz-xyz.mean(axis=0);modes=[]
    for axis in np.eye(3):
        modes.extend([np.broadcast_to(axis,xyz.shape).ravel(),np.cross(np.broadcast_to(axis,xyz.shape),centered).ravel()])
    u,s,_=np.linalg.svd(np.stack(modes,axis=1),full_matrices=False)
    return xp.asarray(u[:,s>max(s[0],1.)*1e-10])


class StaticLandmarks(MotionObjective):
    """Static XYZ landmarks; no independent motion or GP deformation per view."""
    def __init__(self,xyz,cameras,owners,frames,surfaces,radius,padding,maximum,frame_count,xp=np):
        super().__init__(xp.empty((len(xyz),0)),cameras,owners,frames,surfaces,radius,padding,np.ones(frame_count-1),maximum,xp)
        self.gauge=cloud_gauge(xyz,xp)

    def trajectory(self,x):
        xp=self.xp;static=x[:3*self.pc];static=static-self.gauge@(self.gauge.T@static)
        track=xp.broadcast_to(static.reshape(self.pc,1,3),(self.pc,self.fc,3))
        return track,xp.empty((self.fc-1,0,3))

    def track_gradient(self,xyz_gradient,coeff):
        xp=self.xp;g=xp.zeros((self.pc,3));xp.add.at(g,self.owners,xyz_gradient);g=g.ravel()
        return g-self.gauge@(self.gauge.T@g)


def make_objective(xyz,cameras,owners,frames,surfaces,radius,padding,maximum,xp=np):
    block=StaticLandmarks(xyz,np.asarray(cameras)[frames],owners,frames,surfaces,radius,padding,maximum,len(cameras),xp)
    return RigidMotionObjective([block],[xyz],[np.asarray(cameras)[frames]],xp)


def updated_geometry(objective,parameters,cameras,xyz,translations):
    xp=objective.xp;frame,q,_=objective.decode(parameters)
    block=objective.blocks[0];track,_=block.trajectory(parameters[objective.offsets[0]:objective.offsets[1]])
    return q@xp.asarray(cameras),xp.asarray(xyz)+track[:,0],xp.asarray(translations)+frame[:,3:]
