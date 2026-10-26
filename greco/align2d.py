"""Robust multiscale 2D tilt-series alignment in original input order."""
import time
import numpy as np


def align_tilt_series(images, angles, *, anchor_index=None, lags=(1, 2, 3, 4),
                      max_shift=128.0,
                      pyramid_factors=(4, 2, 1), refinement_rounds=None,
                      consensus_radius=2, consensus_passes=2,
                      initial_max_shift=None, initial_peak_method='hann', gpu=False):
    """Complete multiscale quality-weighted graph, consensus and component rescue.

    Public shifts retain INPUT plane order, in input-image pixels, TO APPLY.
    The reference implementation expects increasing-angle order internally.
    No external package/repository or saved shift table is imported at runtime.
    ``initial_max_shift`` bounds acquisition motion (default: 75% of the
    image width, subject to overlap scoring). Subsequent graph remeasurement
    is residual-only, capped at 8 image pixels; max_shift is not an absolute
    cap on the initial acquisition displacement.
    Initial peaks use Hann by default; ``initial_peak_method='dual'`` retains
    Hann or untapered weighting for explicit controls.
    """
    from .align2d_robust import align_stack_2d_robust
    from .align2d_backend import registration_backend
    started=time.monotonic()
    images=np.asarray(images,np.float32);angles=np.asarray(angles,float)
    if images.ndim!=3 or angles.shape!=(len(images),) or not np.isfinite(angles).all():
        raise ValueError('image/angle mismatch or nonfinite angles')
    if len(images)<4:
        raise ValueError('robust 2D alignment requires at least four tilts')
    order=np.argsort(angles,kind='stable');inverse=np.argsort(order)
    expected=int(order[np.argmin(np.abs(angles[order]))])
    if anchor_index is not None and anchor_index!=expected:
        raise ValueError('robust 2D anchor must be the tilt nearest zero')
    rounds=2 if refinement_rounds is None else refinement_rounds
    with registration_backend(gpu):
        result=align_stack_2d_robust(images[order],angles[order],max_shift=max_shift,
            initial_max_shift=initial_max_shift,
            initial_peak_method=initial_peak_method,
            graph_lags=lags,pyramid_factors=pyramid_factors,graph_refinement_rounds=rounds,
            upsample_factor=10,consensus_radius=consensus_radius,consensus_passes=consensus_passes)
    residuals=np.asarray([np.linalg.norm(edge.residual_yx) for edge in result.final_edge_diagnostics])
    diagnostics=dict(algorithm='robust_graph_multiscale_consensus_component_rescue',
        graph_refinement_rounds=int(rounds),consensus_radius=int(consensus_radius),
        consensus_passes=int(consensus_passes),upsample_factor=10,
        initial_search_limit_px=float(max(max_shift,.75*min(images.shape[1:]))
                                      if initial_max_shift is None else initial_max_shift),
        graph_refinement_residual_limit_px=float(min(max_shift,8.0)),
        initial_search_method=('hann_tapered_linear_cc' if initial_peak_method=='hann'
                               else 'overlap_ranked_tapered_and_untapered_linear_cc'),
        internal_plane_order='increasing tilt angle',public_plane_order='input',
        node_confidence=result.node_confidence[inverse].tolist(),
        candidate_bad_indices=order[np.flatnonzero(result.candidate_bad_tilts)].tolist(),
        terminal_bad_indices=order[np.flatnonzero(result.terminal_bad_tilts)].tolist(),
        consensus_corrections_yx=result.consensus_corrections_yx[inverse].tolist(),
        terminal_rescue_corrections_yx=result.terminal_rescue_corrections_yx[inverse].tolist(),
        interleaved_branch_mode=bool(result.interleaved_branch_mode),
        terminal_rescues=[dict(accepted=bool(item.accepted),side=item.terminal_side,
                             indices=order[list(item.terminal_indices)].tolist(),
                             offset_yx=list(item.component_offset_yx),shadow_state=item.shadow_state)
                         for item in result.terminal_rescue_diagnostics])
    return dict(shifts=result.shifts_yx[inverse],anchor=int(order[result.reference_index]),
                edge_residuals=residuals,runtime_s=time.monotonic()-started,
                backend='robust',compute_device='gpu' if gpu else 'cpu',diagnostics=diagnostics)


def shifts_to_fullres(shifts_binned, binning=8):
    return np.asarray(shifts_binned, float) * float(binning)
