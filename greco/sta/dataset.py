"""Bounded MRCS reader and optional resident-GPU cache of fixed-pose views."""
from pathlib import Path
import numpy as np
from ..particle_star import header,rows,image_path
from ..particle_patches import spectra_batch


def ctf_phase(parameters, n, pixel, xp=np):
    """Phase of -Greco CTF on the native half-complex detector grid."""
    p=xp.asarray(parameters,dtype=xp.float64)
    x=xp.arange(n//2+1,dtype=xp.float64)[None,None,:]/(n*pixel)
    y=xp.arange(n,dtype=xp.float64)
    y=xp.where(y<n//2,y,y-n)[None,:,None]/(n*pixel)
    du,dv,angle=[p[:,i,None,None] for i in range(3)]
    a=xp.deg2rad(angle);s=xp.sin(a);c=xp.cos(a)
    axx=-du*c*c-dv*s*s;axy=(dv-du)*s*c;ayy=-du*s*s-dv*c*c
    k1,k2,k3,k5=[p[:,i,None,None] for i in range(3,7)]
    gamma=k1*(axx*x*x+2*axy*x*y+ayy*y*y)+k2*(x*x+y*y)**2-k5-k3
    return gamma


def ctf_batch(parameters, n, pixel, xp=np):
    """Vectorized -Greco CTF (raw detector contrast → positive density)."""
    return xp.sin(ctf_phase(parameters,n,pixel,xp)).astype(xp.float32)


class ParticleDataset:
    def __init__(self, path, backend):
        from scipy.spatial.transform import Rotation
        from greco._numerics.core.ctf import CTF
        self.path=Path(path);self.backend=backend;self.xp=backend.xp
        from .corrections import correction_tables
        self.raw_path,self.correction_sources,pose_updates,observation_updates=correction_tables(path)
        general,tomos=header(path);self.n=int(general['grecoBoxSize']);self.pixel=float(general['grecoPixelSizeAngst'])
        if self.n!=backend.n:raise ValueError('Inconsistent classification grid.')
        if any(general[k]!=0 for k in ('grecoCtfApplied','grecoContrastInverted','grecoCircleCropApplied')):
            raise ValueError('Classification requires raw, uncorrected extracted patches.')
        if general['grecoCircleFalloffPixel']!=5 or general['grecoCoordinateConvention']!='exported_tomogram_center_nminus1_over2_xyz_angstrom' or general['grecoEulerConvention']!='passive_zyz_degrees':
            raise ValueError('Unsupported extraction convention.')
        self.general=general;self.tomos={t['grecoTomogramId']:t for t in tomos}
        self.views={};self.particles=[];self.observations={};seen=set();self.applied_observation_corrections={}
        for block,row in rows(self.raw_path):
            if block=='views':self.views[(row['grecoTomogramId'],row['grecoViewId'])]=row
            elif block=='particles':
                pid=row['grecoParticleId']
                if pid in seen:raise ValueError('Duplicate particle ID.')
                seen.add(pid);self.particles.append(row);self.observations[pid]=[]
            elif block=='observations':
                pid=row['grecoParticleId']
                if pid not in seen:raise ValueError('Observation has unknown particle.')
                self.observations[pid].append(row)
                applied={name:row.get(source,0.) for name,source in zip(
                    ('grecoResidualShiftXAngst','grecoResidualShiftYAngst','grecoDefocusOffsetAngst'),
                    ('grecoAppliedResidualShiftXAngst','grecoAppliedResidualShiftYAngst','grecoAppliedDefocusOffsetAngst'))}
                if any(applied.values()):self.applied_observation_corrections[(pid,row['grecoViewId'])]=applied
        if len(self.particles)!=general['grecoParticleCount']:raise ValueError('Particle count mismatch.')
        self.raw_views={key:dict(value) for key,value in self.views.items()}
        from .corrections import corrected_views
        for key,update in corrected_views(path).items():
            if key not in self.views:raise ValueError('Corrected geometry has unknown tilt identity.')
            self.views[key].update(update)
        original_xyz={p['grecoParticleId']:np.array([p['grecoCoordinate'+d+'Angst'] for d in 'XYZ']) for p in self.particles}
        if pose_updates and set(pose_updates)!=seen:raise ValueError('Corrected particle identities differ from extraction.')
        for p in self.particles:
            update=pose_updates.get(p['grecoParticleId'],{})
            if update.get('grecoRandomSubset',p['grecoRandomSubset'])!=p['grecoRandomSubset']:raise ValueError('Corrections changed half labels.')
            for key in ('grecoTomogramId','grecoUsable','grecoImageCount','grecoImageStartIndex'):
                if key in p and update.get(key,p[key])!=p[key]:raise ValueError('Corrections changed immutable particle membership or image indexing.')
            p.update({k:v for k,v in update.items() if k not in ('grecoImageStack','grecoSourceImageStack')})
        valid_keys={(pid,o['grecoViewId']) for pid,observations in self.observations.items() for o in observations}
        if set(observation_updates)-valid_keys:raise ValueError('Correction has unknown particle/view identity.')
        self.all_particles=self.particles
        self.particles=[p for p in self.all_particles if p['grecoUsable']==1]
        self.offsets=[0];matrices=[];parameters=[];groups=[];view_matrices=[];shifts=[];self.observation_keys=[];self.group_keys=list(self.views)
        self.observation_corrections=observation_updates
        group_index={key:i for i,key in enumerate(self.group_keys)}
        coefficients={}
        from ..particle_ctf import star_particle_defocus
        self.particle_gradient_model='native_beam'
        self.ctf_premultiplied=False
        for key,v in self.views.items():
            c=CTF(kv=v['grecoVoltageKV'],cs=v['grecoCsMM'],q0=v['grecoAmplitudeContrast'],phase_shift=v['grecoPhaseShiftDeg']).initialise()
            coefficients[key]=(c._k1,c._k2,c._k3,c._k5)
        for p in self.particles:
            pose=Rotation.from_euler('ZYZ',[p['grecoAngle'+a] for a in ('Rot','Tilt','Psi')],degrees=True).as_matrix().T
            obs=self.observations[p['grecoParticleId']]
            if len(obs)!=self.tomos[p['grecoTomogramId']]['grecoViewCount']:raise ValueError('Missing particle views.')
            valid=[o for o in obs if o['grecoVisible']==1]
            if len(valid)<2:raise ValueError('Usable particle has insufficient observations.')
            for o in valid:
                key=(p['grecoTomogramId'],o['grecoViewId']);v=self.views[key]
                view_matrix=np.array([[v[f'grecoProjection{i}{j}'] for j in range(3)] for i in range(3)])
                r=view_matrix@pose;view_matrices.append(view_matrix)
                if not np.allclose(r@r.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1.,atol=1e-8):raise ValueError('Invalid particle/view rotation.')
                du,dv=star_particle_defocus(p,v,o)
                correction=observation_updates.get((p['grecoParticleId'],o['grecoViewId']),{})
                df=correction.get('grecoDefocusOffsetAngst',0.)
                du+=df;dv+=df
                xyz=np.array([p['grecoCoordinate'+d+'Angst'] for d in 'XYZ'])
                raw_view=self.raw_views[key]
                raw_matrix=np.array([[raw_view[f'grecoProjection{i}{j}'] for j in range(3)] for i in range(3)])
                geometry_shift=np.array([v.get('grecoGeometryShift'+d+'Angst',0.)-raw_view.get('grecoGeometryShift'+d+'Angst',0.) for d in 'XY'])
                shift=(view_matrix[:2]@xyz-raw_matrix[:2]@original_xyz[p['grecoParticleId']]+geometry_shift+np.array([correction.get('grecoResidualShift'+d+'Angst',0.) for d in 'XY']))/self.pixel
                shifts.append(shift);self.observation_keys.append((p['grecoParticleId'],o['grecoViewId']))
                matrices.append(r);parameters.append([du,dv,v['grecoDefocusAngleDeg'],*coefficients[key]])
                groups.append(group_index[key])
            self.offsets.append(len(matrices))
        if len(matrices)!=general['grecoObservationCount'] or len(self.particles)!=general['grecoSupportedParticleCount']:
            raise ValueError('Extraction support totals differ from STAR.')
        self.rotations=np.asarray(matrices);self.parameters=np.asarray(parameters,float)
        self.view_rotations=np.asarray(view_matrices)
        self.correction_shifts=np.asarray(shifts)
        if not np.isfinite(self.correction_shifts).all():raise ValueError('Invalid corrected image shifts.')
        self.groups=np.asarray(groups,np.int32);self.offsets=np.asarray(self.offsets,np.int64)
        from .transfer_envelope import view_parameters
        self.transfer_scales,self.transfer_doses=np.array([view_parameters(self.views[self.group_keys[g]]) for g in self.groups]).T
        self.dose_weighting=any(v.get('grecoDoseWeightingModel','none')!='none' for v in self.views.values())
        self.tilt_scale_weighting=any(v.get('grecoCtfScaleFactor',1.)!=1. for v in self.views.values())
        from greco._numerics.core.projector import _inv3
        self.inverse_rotations=self.xp.asarray(np.asarray([_inv3(r) for r in self.rotations],np.float64))
        if not np.isfinite(self.parameters).all():raise ValueError('Invalid CTF parameters.')
        self.cached=None

    def observation_batch_limit(self, requested):
        """Bound image workspaces as well as the persistent Fourier grids.

        Raw tapering, phase/CTF evaluation and MAP backprojection retain several
        float64/complex128 planes at once. Budget 128 bytes per detector pixel
        per observation, leaving room for maps, accumulators and FFT plans.
        A particle remains indivisible; batches() may exceed this target for
        one particle, but never groups multiple particles above the target.
        """
        requested=int(requested)
        if requested<1:raise ValueError('Observation batch size must be positive.')
        if not self.backend.gpu:return requested
        cp=self.xp
        available=int(cp.cuda.runtime.memGetInfo()[0])+int(cp.get_default_memory_pool().free_bytes())
        # An explicit pool limit can be lower than the physical device capacity.
        pool=cp.get_default_memory_pool();limit=pool.get_limit()
        if limit:available=min(available,max(0,int(limit)-int(pool.used_bytes())))
        workspace=min(1024**3,max(0,available//8))
        return max(1,min(requested,workspace//(self.n*self.n*128)))

    def batches(self, max_observations=256):
        max_observations=self.observation_batch_limit(max_observations)
        start=0
        while start<len(self.particles):
            stop=start+1
            while stop<len(self.particles) and self.offsets[stop+1]-self.offsets[start]<=max_observations:stop+=1
            yield start,stop
            start=stop

    def load(self, start, stop):
        a,b=self.offsets[start],self.offsets[stop]
        if self.cached is not None:return self.cached[0][a:b],self.cached[1][a:b]
        return self.transform(start,stop,self.read_raw(start,stop))

    def read_raw(self,start,stop):
        """CPU-only bounded read; safe to overlap with main-thread CUDA work."""
        import mrcfile
        patches=[];offsets=[]
        for p in self.particles[start:stop]:
            obs=self.observations[p['grecoParticleId']];valid=[o for o in obs if o['grecoVisible']==1]
            file=image_path(self.raw_path,p['grecoImageStack'])
            with mrcfile.mmap(file,mode='r') as m:
                if m.data.dtype not in (np.dtype('float16'),np.dtype('float32')) or not np.isclose(float(m.voxel_size.x),self.pixel):raise ValueError('Invalid particle sampling/type.')
                if m.data.shape[-2:]!=(self.n,self.n):raise ValueError('Invalid particle box.')
                data=m.data.reshape(-1,self.n,self.n)
                expected=p['grecoImageCount']
                if len(data)!=expected:raise ValueError('Invalid particle stack length.')
                indices=np.array([int(o['grecoImageIndex'])-1 for o in valid])
                if np.any(indices<0) or np.any(indices>=len(data)):raise ValueError('Invalid MRCS image indices.')
                patches.append(np.array(data[indices],copy=True))
            offsets.extend([[o['grecoOffsetXPixel'],o['grecoOffsetYPixel']] for o in valid])
        offsets=np.asarray(offsets,float)
        if not np.isfinite(offsets).all() or np.any(abs(offsets)>.5000001):raise ValueError('Invalid fractional centering offsets.')
        raw=np.concatenate(patches)
        return raw,offsets

    def transform(self,start,stop,raw_offsets):
        raw,offsets=raw_offsets;xp=self.xp;a,b=self.offsets[start],self.offsets[stop]
        # Orthonormal spectra → Greco normalized 2-D FFT (divide by n).
        f=spectra_batch(raw,offsets,self.backend.gpu).swapaxes(-1,-2)/self.n
        shifts=xp.asarray(self.correction_shifts[a:b])
        if np.any(self.correction_shifts[a:b]):
            ky=xp.fft.fftfreq(self.n)*self.n;kx=xp.arange(self.n//2+1)
            f*=xp.exp(2j*xp.pi/self.n*(shifts[:,0,None,None]*kx[None,None,:]+shifts[:,1,None,None]*ky[None,:,None]))
        c=ctf_batch(self.parameters[a:b],self.n,self.pixel,xp)
        c*=self.transfer_envelope(a,b)
        if not bool(xp.isfinite(f).all()):raise ValueError('Non-finite particle data.')
        return (f*c,c*c) if self.ctf_premultiplied else (f,c)

    def transfer_envelope(self,a,b):
        from .transfer_envelope import envelope
        return envelope(self.transfer_scales[a:b],self.transfer_doses[a:b],self.n,self.pixel,self.xp)

    def configure_observation_model(self,model):
        if model not in ('raw','ctf_premultiplied'):raise ValueError('Unknown STA observation model.')
        requested=model=='ctf_premultiplied'
        if self.cached is not None and requested!=self.ctf_premultiplied:
            raise ValueError('Configure the observation model before caching.')
        self.ctf_premultiplied=requested

    def loaded_batches(self,max_observations=256):
        """One look-ahead raw batch; CUDA remains exclusively on the caller thread."""
        if not self.backend.gpu or self.cached is not None:
            for start,stop in self.batches(max_observations):
                yield start,stop,*self.load(start,stop)
            return
        from concurrent.futures import ThreadPoolExecutor
        batches=iter(self.batches(max_observations));current=next(batches,None)
        if current is None:return
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='sta-read') as pool:
            pending=pool.submit(self.read_raw,*current)
            while current is not None:
                raw=pending.result();following=next(batches,None)
                if following is not None:pending=pool.submit(self.read_raw,*following)
                f,c=self.transform(*current,raw)
                yield *current,f,c
                del f,c,raw
                current=following

    def cache(self, progress=None):
        if not self.backend.gpu:return False
        cp=self.xp;count=int(self.offsets[-1]);shape=(count,self.n,self.n//2+1)
        # Keep room for all class grids, batched workspaces and CUDA allocations.
        required=int(np.prod(shape))*12
        if required>cp.cuda.runtime.memGetInfo()[0]*.4:return False
        f=cp.empty(shape,cp.complex64);c=cp.empty(shape,cp.float32)
        for start,stop in self.batches():
            a,b=self.offsets[start],self.offsets[stop];values,ctfs=self.load(start,stop)
            f[a:b]=values;c[a:b]=ctfs
            if progress:progress(stop,len(self.particles))
        self.cached=(f,c)
        return True
