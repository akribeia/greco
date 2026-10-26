"""Device-resident L-BFGS with safeguarded strong-Wolfe line search.

Only scalar stopping/line-search decisions cross to Python. Parameters,
gradients, directions and curvature history remain on the selected device.
NumPy support is retained for deterministic numerical tests.
"""
from types import SimpleNamespace
import numpy as np


def minimize(fun, x0, xp=np, max_iterations=1000, progress=None,
             memory=10, ftol=1e-10, gtol=1e-5, max_linesearch=30):
    x=xp.array(x0,dtype=xp.float64,copy=True)
    value,g=fun(x);value=float(value);history=[];evaluations=1;restarts=0
    if not np.isfinite(value) or not bool(xp.isfinite(g).all()):
        raise ValueError('Non-finite initial GPU optimizer objective/gradient')
    success=False;message='Iteration limit';iteration=0;stable=0

    for iteration in range(1,max_iterations+1):
        if float(xp.max(xp.abs(g)))<=gtol:
            success=True;message='Gradient tolerance';iteration-=1;break
        q=g.copy();alphas=[]
        for s,y,rho in reversed(history):
            alpha=rho*xp.dot(s,q);alphas.append(alpha);q-=alpha*y
        if history:
            s,y,_=history[-1];q*=xp.dot(s,y)/xp.maximum(xp.dot(y,y),1e-30)
        for (s,y,rho),alpha in zip(history,reversed(alphas)):
            q+=s*(alpha-rho*xp.dot(y,q))
        direction=-q;derivative=float(xp.dot(g,direction))
        if not np.isfinite(derivative) or derivative>=0:
            history.clear();direction=-g;derivative=-float(xp.dot(g,g));restarts+=1
        old_value=value;best=None;lo=0.;hi=None;lo_value=value
        step=1.;accepted=None
        # Strong Wolfe with bisection zoom. An Armijo fallback may be used,
        # but cannot itself declare convergence or supply negative curvature.
        for trial in range(max_linesearch):
            candidate=x+step*direction
            fv,gv=fun(candidate);fv=float(fv);evaluations+=1
            finite=np.isfinite(fv) and bool(xp.isfinite(gv).all())
            armijo=finite and fv<=old_value+1e-4*step*derivative
            if armijo and (best is None or fv<best[0]):
                best=(fv,candidate,gv,step)
            if not armijo or (lo>0 and fv>=lo_value):
                hi=step
            else:
                slope=float(xp.dot(gv,direction))
                if abs(slope)<=-.9*derivative:
                    accepted=(fv,candidate,gv,step);break
                if slope>=0:hi=step
                else:lo=step;lo_value=fv
            step=(lo+hi)/2 if hi is not None else min(step*2,64.)
            if step<1e-14 or (hi is not None and hi-lo<1e-14):break
        wolfe=accepted is not None
        if accepted is None:accepted=best
        if accepted is None:
            if history:
                history.clear();restarts+=1;stable=0;continue
            message='Line search stalled';break
        value,new_x,new_g,_=accepted
        s=new_x-x;y=new_g-g;sy=float(xp.dot(s,y))
        if sy>1e-12*float(xp.linalg.norm(s))*float(xp.linalg.norm(y)):
            history.append((s,y,1/sy))
            if len(history)>memory:history.pop(0)
        else:history.clear()
        x,g=new_x,new_g
        relative=(old_value-value)/max(1.,abs(old_value),abs(value))
        stable=stable+1 if wolfe and 0<=relative<=ftol else 0
        if progress:progress(iteration)
        if stable>=3:
            success=True;message='Relative objective plateau (3 Wolfe steps)';break
    return SimpleNamespace(x=x,fun=value,jac=g,nit=iteration,success=success,
                           message=message,nfev=evaluations,restarts=restarts)
