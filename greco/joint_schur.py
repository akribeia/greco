"""Small-system robust joint fit, with the historical XYZ translation gauge.

Eliminate independent 3x3 landmark blocks instead of materializing the dense
Helmert design. The objective is the same scalar soft_l1 loss as Greco. XYZ is
free at every solve; no spatial or angular prior is introduced.
"""
import time
import numpy as np


def _projection(angles):
    radians = np.deg2rad(angles)
    j = np.zeros((len(angles), 2, 3), dtype=np.float64)
    j[:, 0, 1] = 1.
    j[:, 1, 0] = -np.sin(radians)
    j[:, 1, 2] = np.cos(radians)
    return j


def solve_joint_votes(vote_matrix, angles, f_scale=.2, max_nfev=300,
                      tolerance=1e-10, workspace=None, *, device='gpu'):
    started = time.monotonic()
    votes = np.asarray(vote_matrix, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if votes.ndim != 3 or votes.shape[1:] != (len(angles), 2):
        raise ValueError('expected (landmark, tilt, 2) votes')
    n, t, _ = votes.shape
    if n < 2 or t < 3 or not np.isfinite(angles).all() or f_scale <= 0 or max_nfev < 1 or tolerance <= 0:
        raise ValueError('invalid landmarks, angles or robust scale')
    valid = np.isfinite(votes).all(axis=2)
    if not valid.any(axis=0).all() or (valid.sum(axis=1) < 2).any():
        raise ValueError('unidentifiable joint fit: insufficient angular/observation support')
    if device == 'gpu':
        import cupy as xp
    elif device == 'cpu':
        xp = np
    else:
        raise ValueError('unknown Schur device')
    host = (lambda a: np.asarray(a)) if device == 'cpu' else xp.asnumpy
    j_host = _projection(angles)
    flat_j = j_host.reshape(2*t, 3)
    # Remove the three known translation gauges from the shift system. After
    # solving, recenter XYZ and shifts together to impose mean(delta_r)=0.
    if np.linalg.matrix_rank(flat_j) < 3:
        raise ValueError('unidentifiable joint fit: insufficient angular support')
    basis_host = np.linalg.qr(flat_j, mode='complete')[0][:, 3:]
    j = xp.asarray(flat_j)
    basis = xp.asarray(basis_host)
    mask = xp.asarray(np.repeat(valid, 2, axis=1))
    measured = xp.asarray(np.where(valid[..., None], votes, 0.).reshape(n, 2*t))
    shifts = xp.zeros(2*t, dtype=xp.float64)
    dr = xp.zeros((n, 3), dtype=xp.float64)

    def blocks(weights):
        cross = -weights[:, :, None] * j[None]
        local = -xp.einsum('nqd,qe->nde', cross, j)
        inverse = xp.linalg.inv(local)
        eliminated = xp.matmul(cross, inverse)
        schur = xp.diag(weights.sum(axis=0)) - xp.einsum('nqd,npd->qp', eliminated, cross)
        small = basis.T @ ((schur + schur.T)*.5) @ basis
        return cross, inverse, eliminated, (small + small.T)*.5

    # Identifiability depends on the observation mask, not on robust weights.
    # Check a <= (2*T-3) square matrix once, rather than an O(N) wide SVD.
    rank_key = (n, angles.tobytes(), valid.tobytes())
    if workspace is None or workspace.get('_schur_rank_key') != rank_key:
        initial_weights = mask.astype(xp.float64)
        local = xp.einsum('nq,qd,qe->nde', initial_weights, j, j)
        eigen = host(xp.linalg.eigvalsh(local))
        if np.any(eigen[:, 0] <= np.finfo(float).eps * max(n, t) * eigen[:, -1]):
            raise ValueError('unidentifiable joint fit: insufficient landmark angular support')
        _, _, _, fisher = blocks(initial_weights)
        scale = xp.sqrt(xp.maximum(xp.diag(fisher), np.finfo(float).tiny))
        normalized = fisher / scale[:, None] / scale[None, :]
        eigen = host(xp.linalg.eigvalsh(normalized))
        if not len(eigen) or eigen[0] <= np.finfo(float).eps * max(n, 2*t) * eigen[-1]:
            raise ValueError('unidentifiable joint fit: disconnected or rank-deficient observations')
        if workspace is not None:
            # Only retain the latest small validation; changing masks must
            # never accumulate gigabytes of obsolete dense designs.
            workspace['_schur_rank_key'] = rank_key

    def objective(residual):
        root = xp.sqrt(1. + (residual/f_scale)**2)
        # Equivalent to f_scale**2*(root-1), without small-value cancellation.
        return float(xp.sum(residual*residual/(root+1.)))

    converged = False
    line_search_evaluations = 0
    for iteration in range(1, max_nfev+1):
        residual = xp.where(mask, shifts[None] - dr @ j.T - measured, 0.)
        root = xp.sqrt(1. + (residual/f_scale)**2)
        gradient_rows = residual/root
        cost = objective(residual)
        gradient_s = gradient_rows.sum(axis=0)
        gradient_r = -gradient_rows @ j
        # O(N) Helmert-transformed gradient, exactly the historical coordinate
        # optimality measure, without constructing its N x (N-1) basis.
        m = xp.arange(1, n, dtype=xp.float64)[:, None]
        contrast_gradient = (xp.cumsum(gradient_r, axis=0)[:-1] - m*gradient_r[1:]) / xp.sqrt(m*(m+1.))
        optimality = float(xp.maximum(xp.max(xp.abs(gradient_s)), xp.max(xp.abs(contrast_gradient))))
        if optimality <= tolerance:
            converged = True
            break
        # IRLS supplies a safe initial approach; exact Newton curvature then
        # removes the very slow tail of soft_l1 IRLS on large noisy pools.
        weights = xp.where(mask, 1./root if iteration <= 3 else 1./root**3, 0.)
        cross, inverse, eliminated, small = blocks(weights)
        rhs_r = gradient_rows @ j
        rhs_s = -gradient_s - xp.einsum('nqd,nd->q', eliminated, rhs_r)
        try:
            delta_s = basis @ xp.linalg.solve(small, basis.T @ rhs_s)
        except xp.linalg.LinAlgError as error:
            raise RuntimeError('joint Schur step is numerically singular') from error
        delta_r = xp.einsum('nde,ne->nd', inverse,
                            rhs_r-xp.einsum('nqd,q->nd', cross, delta_s))
        translation = delta_r.mean(axis=0)
        delta_r -= translation
        delta_s -= j @ translation
        delta_residual = xp.where(mask, delta_s[None]-delta_r @ j.T, 0.)
        slope = float(xp.sum(gradient_rows*delta_residual))
        if not np.isfinite(slope) or slope >= 0:
            raise RuntimeError('joint Schur step is not a finite descent direction')
        step = 1.
        for _ in range(40):
            proposed_cost = objective(residual+step*delta_residual)
            line_search_evaluations += 1
            if proposed_cost <= cost + 1e-4*step*slope:
                break
            step *= .5
        else:
            raise RuntimeError('joint Schur line search did not converge')
        shifts += step*delta_s
        dr += step*delta_r
        change = float(xp.max(xp.abs(step*delta_residual)))
        # Require both cost and the measurable update to settle. This does not
        # alter GCC's shift-based, three-consecutive-pass convergence criterion.
        if abs(cost-proposed_cost) <= tolerance*max(1., cost) and change <= tolerance**.5:
            converged = True
            break
    if not converged:
        raise RuntimeError('joint fit did not converge within the Schur iteration budget')
    shifts_host = host(shifts).reshape(t, 2)
    dr_host = host(dr)
    residual_host = (shifts_host[None]-np.einsum('tad,nd->nta', j_host, dr_host)-votes)[valid]
    before = votes[valid]
    final_root = np.sqrt(1.+(residual_host/f_scale)**2)
    final_gradient = residual_host/final_root
    gradient_matrix = np.zeros_like(votes)
    gradient_matrix[valid] = final_gradient
    gs = gradient_matrix.sum(axis=0)
    gr = -np.einsum('nta,tad->nd', gradient_matrix, j_host)
    m_host = np.arange(1, n)[:, None]
    gc = (np.cumsum(gr, axis=0)[:-1]-m_host*gr[1:])/np.sqrt(m_host*(m_host+1))
    diagnostics = dict(observations=int(valid.sum()), parameters=2*t+3*(n-1),
        rms_before=float(np.sqrt(np.mean(before**2))), rms_after=float(np.sqrt(np.mean(residual_host**2))),
        median_before=float(np.median(np.linalg.norm(before, axis=1))),
        median_after=float(np.median(np.linalg.norm(residual_host, axis=1))),
        nfev=iteration, optimality=float(max(np.max(np.abs(gs)), np.max(np.abs(gc)))),
        robust_cost=float(np.sum(residual_host**2/(final_root+1.))),
        gauge='mean(delta_r_zyx)=0', equation='vote_yx = delta_shift_yx - J_yx_zyx @ delta_r_zyx',
        loss='soft_l1', f_scale=float(f_scale), backend='gpu_schur' if device=='gpu' else 'cpu_schur',
        arithmetic='float64', shift_system_size=2*t-3,
        line_search_evaluations=line_search_evaluations, solve_time_s=time.monotonic()-started)
    return shifts_host, dr_host, diagnostics
