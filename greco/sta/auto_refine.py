"""Conservative local auto-refinement controller (not Greco's EM sampler)."""
import numpy as np


def alignment_reference(volume, radius, xp=np):
    """Filter after masking; zero unsupported frequencies before projection."""
    n=volume.shape[0]
    f=xp.fft.fftfreq(n)*n
    r=xp.sqrt(f[:,None,None]**2+f[None,:,None]**2+xp.arange(n//2+1)[None,None,:]**2)
    # One-shell inward cosine taper, with no signal above the cutoff.
    w=xp.where(r<=radius-1,1.,xp.where(r<radius,.5+.5*xp.cos(xp.pi*(r-radius+1)),0.))
    return xp.fft.irfftn(xp.fft.rfftn(volume)*w,s=volume.shape,axes=(0,1,2))


def supported_shell(curve, threshold=.143):
    """First downward crossing, never the last isolated high-frequency spike."""
    for row in curve[1:]:
        if not np.isfinite(row['fsc']) or row['fsc'] < threshold:
            return max(1, row['shell']-1)
    return curve[-1]['shell']


class AutoRefine:
    def __init__(self, radius, maximum, diameter_pixels, pixel):
        self.radius=float(radius);self.maximum=float(maximum)
        self.diameter=diameter_pixels;self.pixel=pixel
        self.best=0;self.stable=0;self.last_shell=None
        self.last_fsc=None
        self.fsc_window=[]
        self.search_factor=1.;self.last_search_factor=1.

    def update(self, curve, angle_rms_deg, shift_rms_A, bound_fraction):
        shell=supported_shell(curve)
        # Follow demonstrated signal while limiting each bandwidth increase to
        # 25%. A two-shell cap makes reaching fine sampling need dozens of passes.
        target=min(self.maximum,max(self.radius+2,self.radius*1.25),shell+1)
        advanced=target>self.radius+.01
        values=np.asarray([row['fsc'] for row in curve[1:shell+2]])
        fsc_change=float('inf') if self.last_fsc is None else float(np.max(np.abs(values[:min(len(values),len(self.last_fsc))]-self.last_fsc[:min(len(values),len(self.last_fsc))])))
        self.fsc_window=(self.fsc_window+[values.copy()])[-4:]
        common=min(map(len,self.fsc_window))
        window_change=float(np.max(np.ptp(np.stack([v[:common] for v in self.fsc_window]),axis=0))) if common else float('inf')
        improved=shell>self.best
        self.best=max(self.best,shell)
        # Angular displacement at the particle surface, not Euler-vector size.
        resolution_pixels=2*self.maximum/max(self.radius,1)
        tolerance_pixels=max(.05, .05*resolution_pixels)
        movement=max(np.deg2rad(angle_rms_deg)*self.diameter/2,shift_rms_A/self.pixel)
        quiet=movement<tolerance_pixels and bound_fraction<.05
        settled=quiet and not advanced and not improved and fsc_change<=.005 and window_change<=.005
        used_search_factor=self.search_factor
        if advanced:self.radius=target
        if settled:
            self.search_factor=max(.125,self.search_factor*.5)
        elif advanced or not quiet:
            self.search_factor=min(1.,self.search_factor*2)
        # Physical motion, supported bandwidth and a four-curve plateau are
        # the evidence; no compulsory replay at an arbitrary minimum range.
        self.stable=self.stable+1 if settled else 0
        self.last_search_factor=used_search_factor
        self.last_fsc=values.copy()
        self.last_shell=shell
        return dict(next_radius=self.radius,supported_shell=shell,
                    stable_iterations=self.stable,converged=self.stable>=3,
                    movement_pixels=movement,tolerance_pixels=tolerance_pixels,
                    fsc_change=fsc_change if np.isfinite(fsc_change) else None,
                    fsc_window_change=window_change if np.isfinite(window_change) else None,
                    search_factor=self.search_factor,used_search_factor=used_search_factor,
                    fsc_improved=improved)
