"""Greco GP motion parameterization with batched GPU correlation likelihood.

Camera geometry is fixed. Static XYZ nuisance offsets and whitened spatial
GP velocity modes are jointly optimized; velocities integrate in acquisition
order. Half-sets are fitted separately by the caller. This is not the full
Greco frame-angle/CTF optimizer, and is not a bit-exact C++ implementation.
"""
import numpy as np


def spatial_basis(xyz, sigma_velocity, sigma_divergence, max_modes=-1, xp=np):
    """Greco exponential spatial covariance and sqrt-eigenvalue basis."""
    xyz=xp.asarray(xyz,dtype=xp.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)==0 or not bool(xp.isfinite(xyz).all()):
        raise ValueError('GP requires finite XYZ positions')
    if sigma_velocity<=0 or sigma_divergence<=0:
        raise ValueError('GP sigmas must be positive')
    distance=xp.sqrt(xp.sum((xyz[:,None]-xyz[None,:])**2,axis=2))
    covariance=sigma_velocity**2*xp.exp(-distance/sigma_divergence)
    values,vectors=xp.linalg.eigh(covariance)
    order=xp.argsort(values)[::-1]
    values=values[order];vectors=vectors[:,order]
    # Eigenvector signs are arbitrary (and differ across LAPACK/cuSOLVER).
    # Canonical signs make diagnostics and optimizer starts reproducible.
    pivot=xp.argmax(xp.abs(vectors),axis=0)
    vectors*=xp.where(vectors[pivot,xp.arange(len(values))]>=0,1.,-1.)[None,:]
    keep=values>1e-10
    if max_modes>0:keep&=xp.arange(len(values))<max_modes
    return vectors[:,keep]*xp.sqrt(values[keep])[None,:],values[keep]


def correlation_surfaces(data,prediction,precision,n,maximum_pixels,padding=2,xp=np):
    """Compact oversampled -SSE surfaces, retaining exact half-plane weights.

At integer/padding shifts this is -2 Re sum(data exp(+i phase) conj(pred)*w)
up to an additive constant. Fourier padding then Keys interpolation follows
the Greco local-CC strategy. No peak picking, division by CTF or whitening
by the observed noisy power is performed.
    """
    if padding not in (1,2,4):raise ValueError('Unsupported correlation padding')
    size=n*padding
    half=data*xp.conj(prediction)*precision
    half=half.copy()
    # irfft includes both Hermitian partners; the existing likelihood counts
    # each stored half-plane sample once. Nyquist is excluded by the fit band.
    half[:,:,1:n//2]*=.5
    padded=xp.zeros((len(data),size,size//2+1),dtype=half.dtype)
    padded[:,:n//2,:n//2+1]=half[:,:n//2]
    padded[:,-n//2:,:n//2+1]=half[:,n//2:]
    surface=-2*xp.fft.irfft2(padded,s=(size,size),axes=(-2,-1))*size**2
    radius=int(np.ceil(maximum_pixels*padding))+3
    if 2*radius+1>=size:raise ValueError('Correlation search exceeds extraction support')
    axis=xp.arange(-radius,radius+1)%size
    surface=surface[:,axis[:,None],axis[None,:]]
    # Remove an arbitrary, huge noise-independent constant before optimizing.
    return (surface-surface[:,radius:radius+1,radius:radius+1]).astype(xp.float64),radius


def keys_sample(surfaces,xy,xp=np):
    """Keys a=-1/2 bicubic value and analytic XY gradient (Greco spline)."""
    if xp is not np:
        from .gp3d_kernels import keys_sample as fused
        return fused(surfaces,xy,xp)
    return keys_sample_reference(surfaces,xy,xp)


def keys_sample_reference(surfaces,xy,xp=np):
    """Unfused mathematical reference, also used for before/after benchmarks."""
    xy=xp.asarray(xy,dtype=xp.float64)
    base=xp.floor(xy).astype(xp.int64);frac=xy-base
    def weights(t):
        return (xp.stack((-.5*t+t*t-.5*t**3,1-2.5*t*t+1.5*t**3,
                          .5*t+2*t*t-1.5*t**3,-.5*t*t+.5*t**3),axis=1),
                xp.stack((-.5+2*t-1.5*t*t,-5*t+4.5*t*t,
                          .5+4*t-4.5*t*t,-t+1.5*t*t),axis=1))
    wx,dx=weights(frac[:,0]);wy,dy=weights(frac[:,1])
    value=xp.zeros(len(xy));gradient=xp.zeros((len(xy),2));ids=xp.arange(len(xy))
    for y in range(4):
        for x in range(4):
            samples=surfaces[ids,xp.clip(base[:,1]+y-1,0,surfaces.shape[1]-1),
                            xp.clip(base[:,0]+x-1,0,surfaces.shape[2]-1)]
            value+=samples*wx[:,x]*wy[:,y]
            gradient[:,0]+=samples*dx[:,x]*wy[:,y]
            gradient[:,1]+=samples*wx[:,x]*dy[:,y]
    return value,gradient


class MotionObjective:
    """Fixed-camera GP3D profile likelihood, including static particle XYZ."""
    def __init__(self,basis,cameras,owners,frames,surfaces,radius,padding,
                 frame_gaps,maximum_pixels,xp=np,previous_coefficients=None):
        self.xp=xp;self.basis=xp.asarray(basis);self.pc,self.bc=self.basis.shape
        self.cameras=xp.asarray(cameras)[:,:2];self.owners=xp.asarray(owners,dtype=xp.int64)
        self.frames=xp.asarray(frames,dtype=xp.int64);self.surfaces=xp.asarray(surfaces)
        self.fc=len(frame_gaps)+1;self.gaps=xp.sqrt(xp.asarray(frame_gaps))
        if not bool(xp.all(self.gaps>0)):raise ValueError('Acquisition gaps must be positive')
        self.radius=radius;self.padding=padding;self.maximum=maximum_pixels
        self.previous=xp.zeros((self.fc-1,self.bc,3)) if previous_coefficients is None else xp.asarray(previous_coefficients)
        self.size=3*self.pc+3*self.bc*(self.fc-1)
        # Smooth support barrier, not independent observation clipping.
        self.barrier=max(1.,float(xp.max(xp.abs(self.surfaces))))*100

    def trajectory(self,x):
        xp=self.xp;x=xp.asarray(x)
        static=x[:3*self.pc].reshape(self.pc,3)
        coeff=x[3*self.pc:].reshape(self.fc-1,self.bc,3)
        increments=(self.basis@coeff.transpose(1,0,2).reshape(self.bc,-1)).reshape(self.pc,self.fc-1,3)*self.gaps[None,:,None]
        track=xp.concatenate((xp.zeros((self.pc,1,3)),xp.cumsum(increments,axis=1)),axis=1)
        return static[:,None,:]+track,coeff

    def value_gradient(self,x):
        xp=self.xp;track,coeff=self.trajectory(x)
        projected=xp.einsum('mij,mj->mi',self.cameras,track[self.owners,self.frames])
        total,grad=self.detector_value_gradient(projected)
        xyz_grad=xp.einsum('mij,mi->mj',self.cameras,grad)
        return total+xp.sum((coeff+self.previous)**2),self.track_gradient(xyz_grad,coeff)

    def detector_value_gradient(self,projected):
        """Shared compact likelihood and differentiable extraction-support guard."""
        xp=self.xp
        # The nominal range defines the CC patch, not a hard trajectory bound.
        # Penalize only excursions beyond the valid interpolation margin, per
        # observation. Never shrink other particles to satisfy one outlier.
        guard=(self.radius-2)/self.padding
        # Smooth saturation avoids the derivative jump of a clipped CC
        # coordinate at the outer interpolation margin. The nominal search
        # region itself is unchanged; only the extra numerical margin bends.
        transition=guard-.25
        excess=xp.maximum(xp.abs(projected)-transition,0)
        tangent=xp.tanh(excess/.25)
        clipped=xp.where(excess>0,xp.sign(projected)*(transition+.25*tangent),projected)
        coordinate_derivative=xp.where(excess>0,1-tangent*tangent,1.)
        value,grad=keys_sample(self.surfaces,clipped*self.padding+self.radius,xp)
        outside=xp.sign(projected)*xp.maximum(xp.abs(projected)-guard,0)
        # A quartic barrier has continuous first/second derivatives at the
        # interpolation margin. There is no nominal-range hard clipping.
        grad=grad*self.padding*coordinate_derivative+4*self.barrier*outside**3
        total=xp.sum(value)+self.barrier*xp.sum(outside**4)
        return total,grad

    def track_gradient(self,xyz_grad,coeff):
        xp=self.xp
        gathered=xp.zeros((self.pc,self.fc,3))
        xp.add.at(gathered,(self.owners,self.frames),xyz_grad)
        static=xp.sum(gathered,axis=1)
        suffix=xp.cumsum(gathered[:,::-1],axis=1)[:,::-1][:,1:]
        velocity=(self.basis.T@suffix.reshape(self.pc,-1)).reshape(self.bc,self.fc-1,3).transpose(1,0,2)*self.gaps[:,None,None]
        velocity+=2*(coeff+self.previous)
        return xp.concatenate((static.ravel(),velocity.ravel()))

    def parameter_scale(self):
        """Jacobi scaling for static XYZ versus integrated GP velocity modes.

This changes optimizer coordinates only, never the likelihood/prior. Without
it, a large spatial mode integrated over many frames dwarfs static parameters.
        """
        xp=self.xp;centre=xp.full((len(self.owners),2),float(self.radius))
        detector=[];step=.05
        for axis in range(2):
            offset=xp.zeros_like(centre);offset[:,axis]=self.padding*step
            plus=keys_sample(self.surfaces,centre+offset,xp)[1][:,axis]
            minus=keys_sample(self.surfaces,centre-offset,xp)[1][:,axis]
            detector.append(xp.abs((plus-minus)*self.padding/(2*step)))
        detector=xp.stack(detector,axis=1)
        detector=xp.maximum(detector,xp.maximum(xp.mean(detector)*1e-3,1e-12))
        information=xp.einsum('mij,mi->mj',self.cameras**2,detector)
        gathered=xp.zeros((self.pc,self.fc,3));xp.add.at(gathered,(self.owners,self.frames),information)
        static=xp.sum(gathered,axis=1)
        suffix=xp.cumsum(gathered[:,::-1],axis=1)[:,::-1][:,1:]
        velocity=((self.basis**2).T@suffix.reshape(self.pc,-1)).reshape(self.bc,self.fc-1,3).transpose(1,0,2)*self.gaps[:,None,None]**2+2
        return 1/xp.sqrt(xp.maximum(xp.concatenate((static.ravel(),velocity.ravel())),1e-8))


def fit_motion(objective,max_iterations=200,progress=None,optimizer=None):
    """Device-resident optimizer on GPU; retained SciPy CPU reference."""
    xp=objective.xp
    host=lambda a:np.asarray(a) if xp is np else xp.asnumpy(a)
    parameter_scale=objective.parameter_scale()
    use_device=xp is not np and optimizer!='scipy_reference'
    if use_device:
        from .gpu_lbfgs import minimize
        def device_fun(x):
            value,gradient=objective.value_gradient(x*parameter_scale)
            return value,gradient*parameter_scale
        result=minimize(device_fun,xp.zeros(objective.size),xp,max_iterations,progress)
        fun=device_fun
        initial=xp.zeros(objective.size);baseline=float(fun(initial)[0])
    else:
        from scipy.optimize import minimize
        def fun(x):
            value,gradient=objective.value_gradient(xp.asarray(x)*parameter_scale)
            return float(value),host(gradient*parameter_scale)
        initial=np.zeros(objective.size);baseline=fun(initial)[0]
        evaluations=[0]
        def callback(x):
            evaluations[0]+=1
            if progress:progress(evaluations[0])
        result=minimize(fun,initial,jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=max_iterations,maxls=30,ftol=1e-10,gtol=1e-5,maxcor=10))
    if not np.isfinite(result.fun) or not bool(xp.isfinite(xp.asarray(result.x)).all()):
        raise ValueError('Non-finite GP3D optimizer result')
    # Accept one scale for the entire coupled GP, including its prior.
    candidates=[(float(fun(scale*result.x)[0]),scale) for scale in (0.,.125,.25,.5,1.)]
    value,scale=min(candidates)
    track,_=objective.trajectory(xp.asarray(scale*result.x)*parameter_scale)
    shifts=xp.einsum('mij,mj->mi',objective.cameras,track[objective.owners,objective.frames])
    if value>baseline:
        track,_=objective.trajectory(xp.asarray(initial));shifts*=0;scale=0.;value=baseline
    return host(shifts),host(track),dict(iterations=int(result.nit),converged=bool(result.success),
        optimizer_message=str(result.message),initial_objective=baseline,final_objective=value,
        acceptance_scale=float(scale),modes=objective.bc,particles=objective.pc,
        observations=len(shifts),maximum_abs_shift_pixels=float(xp.max(xp.abs(shifts))),
        nominal_range_exceeded=int(xp.sum(xp.any(xp.abs(shifts)>objective.maximum,axis=1))),
        interpolation_margin_pixels=(objective.radius-2)/objective.padding,
        optimizer='device_lbfgs' if use_device else 'scipy_reference',
        optimizer_evaluations=int(result.nfev))
