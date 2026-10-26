"""Fixed-geometry joint fitting of raster corrections and landmark positions.

PCC(model, observed) returns a raster correction. If observed content is at
J dr - s relative to the model, PCC returns s - J dr. Updating the raster
by s AND the extraction centre by J dr removes that displacement.
The gauge mean(dr)=0 is imposed exactly through an orthonormal contrast basis.
No extra tilt-centering gauge is applied: that would constrain identifiable
shift components after the landmark translation gauge has been fixed.
"""
import numpy as np
from scipy.linalg import helmert
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix


def projection_jacobians(angles):
    angles = np.deg2rad(np.asarray(angles, dtype=float))
    j = np.zeros((len(angles), 2, 3))
    j[:, 0, 1] = 1
    j[:, 1, 0] = -np.sin(angles)
    j[:, 1, 2] = np.cos(angles)
    return j


def solve_joint_votes(vote_matrix, angles, f_scale=0.2, max_nfev=300, tolerance=1e-10,
                      workspace=None, split_components=False, backend='cpu'):
    """Fit (landmark, tilt, yx) votes; missing pairs use NaN.

    soft_l1 operates on scalar detector residuals, in bin8 pixels. Reject
    rank-deficient observation graphs instead of inventing unconstrained
    landmark depths or shifts. Covariances are deliberately not reported.
    """
    if backend == 'gpu_schur':
        from .joint_schur import solve_joint_votes as schur_fit
        return schur_fit(vote_matrix, angles, f_scale, max_nfev, tolerance,
                         workspace=workspace)
    votes = np.asarray(vote_matrix, float)
    angles = np.asarray(angles, float)
    if votes.ndim != 3 or votes.shape[1:] != (len(angles), 2):
        raise ValueError('expected (landmark, tilt, 2) votes')
    n, t, _ = votes.shape
    if n < 2 or not np.isfinite(angles).all() or f_scale <= 0 or max_nfev < 1 or tolerance <= 0:
        raise ValueError('invalid landmarks, angles or robust scale')
    landmark, tilt = np.where(np.isfinite(votes).all(axis=-1))
    cache_key = None
    prepared = None
    if workspace is not None:
        cache_key = (n, t, np.asarray(angles, float).tobytes(),
                     np.isfinite(votes).all(axis=-1).tobytes())
        prepared = workspace.get(cache_key)
    if prepared is None:
        basis = helmert(n).T
        j = projection_jacobians(angles)
        design = np.zeros((len(landmark), 2, 2*t + 3*(n-1)))
        for row, (k, i) in enumerate(zip(landmark, tilt)):
            design[row, :, 2*i:2*i+2] = np.eye(2)
            design[row, :, 2*t:] = (-j[i, :, None, :] * basis[k, None, :, None]).reshape(2, -1)
        design = design.reshape(2*len(landmark), -1)
        if backend in ('gpu_lsmr','gpu_lsmr32') and len(design):
            import cupy as cp
            # The identifiability check was an expensive CPU SVD on every
            # native pass. Keep its float64 rank contract on the GPU too.
            rank=int(cp.linalg.matrix_rank(cp.asarray(design,cp.float64)).get())
        else:
            rank=np.linalg.matrix_rank(design) if len(design) else 0
        if len(design) == 0 or rank < design.shape[1]:
            raise ValueError('unidentifiable joint fit: insufficient angular/observation support')
        sparse = csr_matrix(design)
        prepared = basis, design, sparse, landmark, tilt
        if workspace is not None:
            workspace[cache_key] = prepared
    else:
        basis, design, sparse, landmark, tilt = prepared
    measured = votes[landmark, tilt].ravel()
    def fit(block, values):
        return least_squares(lambda x: block @ x-values, np.zeros(block.shape[1]),
                             jac=lambda x: block.copy(), loss='soft_l1', f_scale=f_scale,
                             tr_solver='lsmr', max_nfev=max_nfev,
                             ftol=tolerance, xtol=tolerance, gtol=tolerance)
    if backend in ('gpu_lsmr', 'gpu_lsmr32'):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
        from cupyx.scipy.sparse.linalg import lsmr as gpu_lsmr
        gpu_dtype = cp.float32 if backend.endswith('32') else cp.float64
        b = cp.asarray(measured, gpu_dtype)
        x = cp.zeros(design.shape[1], gpu_dtype)
        previous_cost = np.inf
        a = gpu_csr_matrix(cp.asarray(design, dtype=gpu_dtype))
        data_rows = cp.searchsorted(a.indptr[1:], cp.arange(a.data.size), side='right')
        for gpu_iteration in range(1, max_nfev + 1):
            residual_gpu = a @ x - b
            root = cp.sqrt(1.0 + (residual_gpu / f_scale) ** 2)
            weights = 1.0 / cp.sqrt(root)
            aw = a.copy()
            aw.data *= weights[data_rows]
            proposed = gpu_lsmr(
                aw, b * weights, x0=x, atol=tolerance, btol=tolerance,
                maxiter=min(max_nfev, design.shape[1]))[0]
            cost = float((f_scale * f_scale * cp.sum(root - 1.0)).get())
            relative = abs(previous_cost - cost) / max(1.0, abs(previous_cost))
            x = proposed
            if relative < tolerance:
                break
            previous_cost = cost
        solution = cp.asnumpy(x)
        gpu_residual = design @ solution - measured
        gpu_root = np.sqrt(1.0 + (gpu_residual / f_scale) ** 2)
        gpu_gradient = design.T @ (gpu_residual / gpu_root)
        result_nfev = gpu_iteration
        result_optimality = float(np.max(np.abs(gpu_gradient)))
        result_cost = float(f_scale * f_scale * np.sum(gpu_root - 1.0))
    elif backend != 'cpu':
        raise ValueError('unknown joint-solver backend')
    elif split_components:
        from concurrent.futures import ThreadPoolExecutor
        blocks = []
        for component in (0, 1):
            rows = np.arange(component, len(measured), 2)
            columns = np.flatnonzero(np.asarray(np.abs(design[rows]).sum(axis=0)).ravel())
            blocks.append((rows, columns, sparse[rows][:, columns], measured[rows]))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: fit(item[2], item[3]), blocks))
        solution = np.zeros(design.shape[1])
        for (_, columns, _, _), result in zip(blocks, results, strict=True):
            solution[columns] = result.x
        result_nfev = max(result.nfev for result in results)
        result_optimality = float(max(result.optimality for result in results))
        result_cost = float(sum(result.cost for result in results))
    else:
        results = [fit(sparse, measured)]
        solution = results[0].x
        result_nfev = results[0].nfev
        result_optimality = float(results[0].optimality)
        result_cost = float(results[0].cost)
    if backend == 'cpu' and any(not result.success or not np.isfinite(result.x).all() for result in results):
        message = '; '.join(result.message for result in results if not result.success)
        raise RuntimeError('joint fit did not converge: ' + message)
    shifts = solution[:2*t].reshape(t, 2)
    dr = basis @ solution[2*t:].reshape(n-1, 3)
    residual = (design @ solution-measured).reshape(-1, 2)
    before = measured.reshape(-1, 2)
    diagnostics = dict(observations=len(landmark), parameters=design.shape[1],
                       rms_before=float(np.sqrt(np.mean(before**2))),
                       rms_after=float(np.sqrt(np.mean(residual**2))),
                       median_before=float(np.median(np.linalg.norm(before, axis=1))),
                       median_after=float(np.median(np.linalg.norm(residual, axis=1))),
                       nfev=result_nfev, optimality=result_optimality,
                       robust_cost=result_cost,
                       gauge='mean(delta_r_zyx)=0',
                       equation='vote_yx = delta_shift_yx - J_yx_zyx @ delta_r_zyx',
                       loss='soft_l1', f_scale=float(f_scale))
    return shifts, dr, diagnostics


def subpixel_patch_stacks(stack, centres, radius=16):
    """Floating extraction centres; never mutate or round the input raster."""
    offsets = np.mgrid[-radius:radius+1, -radius:radius+1].astype(float)
    output = []
    for by_view in centres:
        patches = []
        for image, centre in zip(stack, by_view, strict=True):
            if np.any(centre-radius < 0) or np.any(centre+radius > np.array(image.shape)-1):
                raise ValueError('landmark patch outside detector support')
            patch = map_coordinates(image, offsets+np.asarray(centre)[:, None, None],
                                    order=1, mode='constant', prefilter=False).astype(np.float32)
            patch -= patch.mean()
            scale = float(patch.std())
            if scale < 1e-6:
                raise ValueError('constant landmark patch')
            patches.append(patch/scale)
        output.append(patches)
    return np.asarray(output, np.float32)
