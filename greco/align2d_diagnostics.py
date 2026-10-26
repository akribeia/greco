from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class GraphObservabilityDiagnostics:
    connected: bool
    incidence_rank: int
    base_laplacian_eigenvalues: NDArray[np.float64]
    final_laplacian_eigenvalues: NDArray[np.float64]
    base_normalized_eigenvalues: NDArray[np.float64]
    final_normalized_eigenvalues: NDArray[np.float64]
    base_fiedler_vector: NDArray[np.float64]
    final_fiedler_vector: NDArray[np.float64]
    base_edge_leverage: NDArray[np.float64]
    final_edge_leverage: NDArray[np.float64]
    base_edge_effective_resistance: NDArray[np.float64]
    final_edge_effective_resistance: NDArray[np.float64]
    base_node_reference_variance: NDArray[np.float64]
    final_node_reference_variance: NDArray[np.float64]
    base_cut_conductance: NDArray[np.float64]
    final_cut_conductance: NDArray[np.float64]
    base_cut_relative_variance: NDArray[np.float64]
    final_cut_relative_variance: NDArray[np.float64]
    base_hodge_residual_yx: NDArray[np.float64]
    final_hodge_residual_yx: NDArray[np.float64]
    base_hodge_energy: float
    final_hodge_energy: float
    final_state_residual_yx: NDArray[np.float64]
    final_state_weighted_energy: float


@dataclass(frozen=True)
class ZNCCPeak:
    rank: int
    shift_yx: tuple[float, float]
    score: float
    overlap_fraction: float
    peak_to_sidelobe_ratio: float
    at_search_boundary: bool
    curvature_eigenvalues: tuple[float, float]


@dataclass(frozen=True)
class EdgeZNCCDiagnostics:
    reference: int
    moving: int
    lag: int
    analysis_factor: int
    search_center_yx: tuple[float, float]
    search_radius: float
    peaks: tuple[ZNCCPeak, ...]
    max_t_pvalue: float
    max_t_replicates: int


@dataclass(frozen=True)
class _WeightedGraphState:
    laplacian_eigenvalues: NDArray[np.float64]
    normalized_eigenvalues: NDArray[np.float64]
    fiedler_vector: NDArray[np.float64]
    edge_leverage: NDArray[np.float64]
    edge_effective_resistance: NDArray[np.float64]
    node_reference_variance: NDArray[np.float64]
    cut_conductance: NDArray[np.float64]
    cut_relative_variance: NDArray[np.float64]
    hodge_residual_yx: NDArray[np.float64]
    hodge_energy: float


def _canonical_vector_sign(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    if not vector.size:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0 else vector


def _incidence_matrix(
    count: int,
    references: NDArray[np.int64],
    movings: NDArray[np.int64],
) -> NDArray[np.float64]:
    incidence = np.zeros((references.size, count), dtype=np.float64)
    rows = np.arange(references.size)
    incidence[rows, references] = -1.0
    incidence[rows, movings] = 1.0
    return incidence


def _normalized_laplacian(
    laplacian: NDArray[np.float64],
) -> NDArray[np.float64]:
    degree = np.diag(laplacian)
    inverse_root = np.zeros_like(degree)
    positive = degree > np.finfo(np.float64).eps
    inverse_root[positive] = 1.0 / np.sqrt(degree[positive])
    return inverse_root[:, None] * laplacian * inverse_root[None, :]


def _weighted_graph_state(
    incidence: NDArray[np.float64],
    measurements: NDArray[np.float64],
    weights: NDArray[np.float64],
    reference_index: int,
) -> _WeightedGraphState:
    count = incidence.shape[1]
    safe_weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    laplacian = incidence.T @ (safe_weights[:, None] * incidence)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normalized_eigenvalues = np.maximum(
        np.linalg.eigvalsh(_normalized_laplacian(laplacian)),
        0.0,
    )
    fiedler = (
        _canonical_vector_sign(eigenvectors[:, 1].copy())
        if count > 1
        else np.zeros(count, dtype=np.float64)
    )

    tolerance = max(float(eigenvalues[-1]) * count * np.finfo(np.float64).eps, 1e-12)
    inverse_eigenvalues = np.zeros_like(eigenvalues)
    positive = eigenvalues > tolerance
    inverse_eigenvalues[positive] = 1.0 / eigenvalues[positive]
    laplacian_pinv = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    edge_effective_resistance = np.einsum(
        "ei,ij,ej->e",
        incidence,
        laplacian_pinv,
        incidence,
        optimize=True,
    )
    edge_leverage = np.clip(safe_weights * edge_effective_resistance, 0.0, 1.0)

    reference_contrasts = np.eye(count, dtype=np.float64)
    reference_contrasts[:, reference_index] -= 1.0
    node_reference_variance = np.einsum(
        "ni,ij,nj->n",
        reference_contrasts,
        laplacian_pinv,
        reference_contrasts,
        optimize=True,
    )
    node_reference_variance = np.maximum(node_reference_variance, 0.0)

    degrees = np.diag(laplacian)
    cut_conductance = np.empty(max(count - 1, 0), dtype=np.float64)
    cut_relative_variance = np.empty(max(count - 1, 0), dtype=np.float64)
    for boundary in range(count - 1):
        left = np.arange(count) <= boundary
        crossing = left[incidence.argmin(axis=1)] != left[incidence.argmax(axis=1)]
        cut_weight = float(np.sum(safe_weights[crossing]))
        left_volume = float(np.sum(degrees[left]))
        right_volume = float(np.sum(degrees[~left]))
        cut_conductance[boundary] = cut_weight / max(
            min(left_volume, right_volume),
            np.finfo(np.float64).eps,
        )
        contrast = np.empty(count, dtype=np.float64)
        contrast[left] = 1.0 / np.count_nonzero(left)
        contrast[~left] = -1.0 / np.count_nonzero(~left)
        cut_relative_variance[boundary] = float(contrast @ laplacian_pinv @ contrast)

    columns = [index for index in range(count) if index != reference_index]
    design = incidence[:, columns]
    root_weights = np.sqrt(safe_weights)
    weighted_design = design * root_weights[:, None]
    weighted_measurements = measurements * root_weights[:, None]
    solution = np.linalg.lstsq(
        weighted_design,
        weighted_measurements,
        rcond=None,
    )[0]
    residual = weighted_design @ solution - weighted_measurements
    energy = float(np.sum(residual**2))
    return _WeightedGraphState(
        laplacian_eigenvalues=eigenvalues,
        normalized_eigenvalues=normalized_eigenvalues,
        fiedler_vector=fiedler,
        edge_leverage=edge_leverage,
        edge_effective_resistance=np.maximum(edge_effective_resistance, 0.0),
        node_reference_variance=node_reference_variance,
        cut_conductance=cut_conductance,
        cut_relative_variance=np.maximum(cut_relative_variance, 0.0),
        hodge_residual_yx=residual,
        hodge_energy=energy,
    )


def graph_observability_diagnostics(
    count: int,
    references: ArrayLike,
    movings: ArrayLike,
    measurements_yx: ArrayLike,
    base_weights: ArrayLike,
    final_weights: ArrayLike,
    reference_index: int,
    final_shifts_yx: ArrayLike,
) -> GraphObservabilityDiagnostics:
    references_array = np.asarray(references, dtype=np.int64)
    movings_array = np.asarray(movings, dtype=np.int64)
    measurements = np.asarray(measurements_yx, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    final = np.asarray(final_weights, dtype=np.float64)
    shifts = np.asarray(final_shifts_yx, dtype=np.float64)
    edge_count = references_array.size
    if (
        movings_array.shape != (edge_count,)
        or measurements.shape != (edge_count, 2)
        or base.shape != (edge_count,)
        or final.shape != (edge_count,)
        or shifts.shape != (count, 2)
    ):
        raise ValueError("graph diagnostic arrays have incompatible shapes")
    if not 0 <= reference_index < count:
        raise ValueError("reference_index is outside the graph")
    if (
        np.any(references_array < 0)
        or np.any(references_array >= count)
        or np.any(movings_array < 0)
        or np.any(movings_array >= count)
    ):
        raise ValueError("edge endpoints are outside the graph")
    incidence = _incidence_matrix(count, references_array, movings_array)
    gauge_fixed = np.delete(incidence, reference_index, axis=1)
    rank = int(np.linalg.matrix_rank(gauge_fixed))
    base_state = _weighted_graph_state(
        incidence,
        measurements,
        base,
        reference_index,
    )
    final_state = _weighted_graph_state(
        incidence,
        measurements,
        final,
        reference_index,
    )
    final_residual = incidence @ shifts - measurements
    final_energy = float(np.sum(np.maximum(final, 1e-12)[:, None] * final_residual**2))
    return GraphObservabilityDiagnostics(
        connected=rank == count - 1,
        incidence_rank=rank,
        base_laplacian_eigenvalues=base_state.laplacian_eigenvalues,
        final_laplacian_eigenvalues=final_state.laplacian_eigenvalues,
        base_normalized_eigenvalues=base_state.normalized_eigenvalues,
        final_normalized_eigenvalues=final_state.normalized_eigenvalues,
        base_fiedler_vector=base_state.fiedler_vector,
        final_fiedler_vector=final_state.fiedler_vector,
        base_edge_leverage=base_state.edge_leverage,
        final_edge_leverage=final_state.edge_leverage,
        base_edge_effective_resistance=base_state.edge_effective_resistance,
        final_edge_effective_resistance=final_state.edge_effective_resistance,
        base_node_reference_variance=base_state.node_reference_variance,
        final_node_reference_variance=final_state.node_reference_variance,
        base_cut_conductance=base_state.cut_conductance,
        final_cut_conductance=final_state.cut_conductance,
        base_cut_relative_variance=base_state.cut_relative_variance,
        final_cut_relative_variance=final_state.cut_relative_variance,
        base_hodge_residual_yx=base_state.hodge_residual_yx,
        final_hodge_residual_yx=final_state.hodge_residual_yx,
        base_hodge_energy=base_state.hodge_energy,
        final_hodge_energy=final_state.hodge_energy,
        final_state_residual_yx=final_residual,
        final_state_weighted_energy=final_energy,
    )


def _integral_image(image: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.pad(image.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _rectangle_sum(
    integral: NDArray[np.float64],
    y0: NDArray[np.int64],
    y1: NDArray[np.int64],
    x0: NDArray[np.int64],
    x1: NDArray[np.int64],
) -> NDArray[np.float64]:
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _search_shifts(
    shape: tuple[int, int],
    search_center_yx: NDArray[np.float64],
    max_shift: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    limits = np.asarray(shape, dtype=np.int64) - 1
    lower = np.maximum(
        np.floor(search_center_yx - max_shift).astype(np.int64),
        -limits,
    )
    upper = np.minimum(
        np.ceil(search_center_yx + max_shift).astype(np.int64),
        limits,
    )
    if np.any(upper < lower):
        raise ValueError("ZNCC search window has no valid shifts")
    return (
        np.arange(lower[0], upper[0] + 1, dtype=np.int64),
        np.arange(lower[1], upper[1] + 1, dtype=np.int64),
    )


def _rectangular_zncc_window(
    reference: NDArray[np.float64],
    moving: NDArray[np.float64],
    search_center_yx: NDArray[np.float64],
    max_shift: float,
    min_overlap: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    height, width = reference.shape
    shifts_y, shifts_x = _search_shifts(reference.shape, search_center_yx, max_shift)
    dy, dx = np.meshgrid(shifts_y, shifts_x, indexing="ij")
    ref_y0 = np.maximum(0, dy)
    ref_y1 = np.minimum(height, height + dy)
    ref_x0 = np.maximum(0, dx)
    ref_x1 = np.minimum(width, width + dx)
    mov_y0 = ref_y0 - dy
    mov_y1 = ref_y1 - dy
    mov_x0 = ref_x0 - dx
    mov_x1 = ref_x1 - dx
    overlap = (ref_y1 - ref_y0) * (ref_x1 - ref_x0)

    reference_integral = _integral_image(reference)
    reference_square_integral = _integral_image(reference**2)
    moving_integral = _integral_image(moving)
    moving_square_integral = _integral_image(moving**2)
    sum_reference = _rectangle_sum(
        reference_integral,
        ref_y0,
        ref_y1,
        ref_x0,
        ref_x1,
    )
    sum_reference_square = _rectangle_sum(
        reference_square_integral,
        ref_y0,
        ref_y1,
        ref_x0,
        ref_x1,
    )
    sum_moving = _rectangle_sum(
        moving_integral,
        mov_y0,
        mov_y1,
        mov_x0,
        mov_x1,
    )
    sum_moving_square = _rectangle_sum(
        moving_square_integral,
        mov_y0,
        mov_y1,
        mov_x0,
        mov_x1,
    )
    correlation = fftconvolve(reference, moving[::-1, ::-1], mode="full")
    center = np.asarray(reference.shape, dtype=np.int64) - 1
    cross = correlation[center[0] + dy, center[1] + dx]
    safe_overlap = np.maximum(overlap, 1)
    covariance = cross - sum_reference * sum_moving / safe_overlap
    variance_reference = sum_reference_square - sum_reference**2 / safe_overlap
    variance_moving = sum_moving_square - sum_moving**2 / safe_overlap
    denominator = np.sqrt(np.maximum(variance_reference * variance_moving, 0.0))
    valid = (overlap >= min_overlap * height * width) & (denominator > np.finfo(np.float64).eps)
    scores = np.full(overlap.shape, -np.inf, dtype=np.float64)
    scores[valid] = covariance[valid] / denominator[valid]
    overlap_fraction = overlap.astype(np.float64) / (height * width)
    return scores, overlap_fraction, shifts_y, shifts_x


def _generic_masked_zncc_window(
    reference: NDArray[np.float64],
    moving: NDArray[np.float64],
    reference_mask: NDArray[np.float64],
    moving_mask: NDArray[np.float64],
    search_center_yx: NDArray[np.float64],
    max_shift: float,
    min_overlap: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    def correlate(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
        return fftconvolve(left, right[::-1, ::-1], mode="full")

    overlap_full = correlate(reference_mask, moving_mask)
    sum_reference_full = correlate(reference_mask * reference, moving_mask)
    sum_moving_full = correlate(reference_mask, moving_mask * moving)
    sum_reference_square_full = correlate(reference_mask * reference**2, moving_mask)
    sum_moving_square_full = correlate(reference_mask, moving_mask * moving**2)
    cross_full = correlate(reference_mask * reference, moving_mask * moving)
    shifts_y, shifts_x = _search_shifts(reference.shape, search_center_yx, max_shift)
    center = np.asarray(reference.shape, dtype=np.int64) - 1
    indices = np.ix_(center[0] + shifts_y, center[1] + shifts_x)
    overlap = np.maximum(overlap_full[indices], 0.0)
    safe_overlap = np.maximum(overlap, np.finfo(np.float64).eps)
    sum_reference = sum_reference_full[indices]
    sum_moving = sum_moving_full[indices]
    covariance = cross_full[indices] - sum_reference * sum_moving / safe_overlap
    variance_reference = sum_reference_square_full[indices] - sum_reference**2 / safe_overlap
    variance_moving = sum_moving_square_full[indices] - sum_moving**2 / safe_overlap
    denominator = np.sqrt(np.maximum(variance_reference * variance_moving, 0.0))
    maximum_overlap = min(float(reference_mask.sum()), float(moving_mask.sum()))
    valid = (overlap >= min_overlap * maximum_overlap) & (denominator > np.finfo(np.float64).eps)
    scores = np.full(overlap.shape, -np.inf, dtype=np.float64)
    scores[valid] = covariance[valid] / denominator[valid]
    return scores, overlap / max(maximum_overlap, 1.0), shifts_y, shifts_x


def _zncc_window(
    reference: ArrayLike,
    moving: ArrayLike,
    *,
    search_center_yx: ArrayLike,
    max_shift: float,
    min_overlap: float,
    reference_mask: ArrayLike | None = None,
    moving_mask: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    reference_array = np.asarray(reference, dtype=np.float64)
    moving_array = np.asarray(moving, dtype=np.float64)
    center = np.asarray(search_center_yx, dtype=np.float64)
    if reference_array.ndim != 2 or moving_array.shape != reference_array.shape:
        raise ValueError("reference and moving must be equal-size 2D images")
    if center.shape != (2,) or max_shift <= 0 or not 0 < min_overlap <= 1:
        raise ValueError("invalid ZNCC search parameters")
    if reference_mask is None and moving_mask is None:
        return _rectangular_zncc_window(
            reference_array,
            moving_array,
            center,
            max_shift,
            min_overlap,
        )
    reference_mask_array = np.ones_like(reference_array)
    moving_mask_array = np.ones_like(moving_array)
    if reference_mask is not None:
        reference_mask_array = np.clip(
            np.asarray(reference_mask, dtype=np.float64),
            0.0,
            1.0,
        )
    if moving_mask is not None:
        moving_mask_array = np.clip(
            np.asarray(moving_mask, dtype=np.float64),
            0.0,
            1.0,
        )
    if (
        reference_mask_array.shape != reference_array.shape
        or moving_mask_array.shape != moving_array.shape
    ):
        raise ValueError("ZNCC masks must match the image shape")
    return _generic_masked_zncc_window(
        reference_array,
        moving_array,
        reference_mask_array,
        moving_mask_array,
        center,
        max_shift,
        min_overlap,
    )


def _parabolic_offset(before: float, center: float, after: float) -> float:
    denominator = before - 2.0 * center + after
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))


def masked_zncc_top_k(
    reference: ArrayLike,
    moving: ArrayLike,
    *,
    search_center_yx: ArrayLike = (0.0, 0.0),
    max_shift: float,
    top_k: int = 3,
    min_overlap: float = 0.25,
    suppression_radius: int = 5,
    reference_mask: ArrayLike | None = None,
    moving_mask: ArrayLike | None = None,
) -> tuple[ZNCCPeak, ...]:
    if top_k < 1 or suppression_radius < 1:
        raise ValueError("top_k and suppression_radius must be positive")
    scores, overlap, shifts_y, shifts_x = _zncc_window(
        reference,
        moving,
        search_center_yx=search_center_yx,
        max_shift=max_shift,
        min_overlap=min_overlap,
        reference_mask=reference_mask,
        moving_mask=moving_mask,
    )
    working = scores.copy()
    raw_peaks: list[tuple[int, int, float, float, bool, tuple[float, float]]] = []
    for _ in range(top_k):
        if not np.isfinite(working).any():
            break
        y_index, x_index = np.unravel_index(np.nanargmax(working), working.shape)
        score = float(working[y_index, x_index])
        if not np.isfinite(score):
            break
        y_offset = 0.0
        x_offset = 0.0
        if 0 < y_index < scores.shape[0] - 1:
            y_offset = _parabolic_offset(
                float(scores[y_index - 1, x_index]),
                score,
                float(scores[y_index + 1, x_index]),
            )
        if 0 < x_index < scores.shape[1] - 1:
            x_offset = _parabolic_offset(
                float(scores[y_index, x_index - 1]),
                score,
                float(scores[y_index, x_index + 1]),
            )
        array_interior = bool(
            0 < y_index < scores.shape[0] - 1 and 0 < x_index < scores.shape[1] - 1
        )
        local_support_interior = bool(
            array_interior
            and np.all(np.isfinite(scores[y_index - 1 : y_index + 2, x_index - 1 : x_index + 2]))
        )
        if local_support_interior:
            hessian = np.zeros((2, 2), dtype=np.float64)
            hessian[0, 0] = (
                2.0 * score - scores[y_index - 1, x_index] - scores[y_index + 1, x_index]
            )
            hessian[1, 1] = (
                2.0 * score - scores[y_index, x_index - 1] - scores[y_index, x_index + 1]
            )
            hessian[0, 1] = hessian[1, 0] = -0.25 * (
                scores[y_index + 1, x_index + 1]
                - scores[y_index + 1, x_index - 1]
                - scores[y_index - 1, x_index + 1]
                + scores[y_index - 1, x_index - 1]
            )
            curvature = np.sort(np.linalg.eigvalsh(hessian))
        else:
            curvature = np.full(2, np.nan, dtype=np.float64)
        at_boundary = bool(not local_support_interior)
        raw_peaks.append(
            (
                y_index,
                x_index,
                score,
                float(overlap[y_index, x_index]),
                at_boundary,
                (float(curvature[0]), float(curvature[1])),
            )
        )
        y0 = max(0, y_index - suppression_radius)
        y1 = min(working.shape[0], y_index + suppression_radius + 1)
        x0 = max(0, x_index - suppression_radius)
        x1 = min(working.shape[1], x_index + suppression_radius + 1)
        working[y0:y1, x0:x1] = -np.inf

    sidelobes = scores.copy()
    for y_index, x_index, *_ in raw_peaks:
        y0 = max(0, y_index - suppression_radius)
        y1 = min(sidelobes.shape[0], y_index + suppression_radius + 1)
        x0 = max(0, x_index - suppression_radius)
        x1 = min(sidelobes.shape[1], x_index + suppression_radius + 1)
        sidelobes[y0:y1, x0:x1] = np.nan
    finite_sidelobes = sidelobes[np.isfinite(sidelobes)]
    baseline = float(np.median(finite_sidelobes)) if finite_sidelobes.size else 0.0
    mad = (
        1.4826 * float(np.median(np.abs(finite_sidelobes - baseline)))
        if finite_sidelobes.size
        else 0.0
    )
    denominator = max(mad, np.finfo(np.float64).eps)

    peaks: list[ZNCCPeak] = []
    for rank, raw in enumerate(raw_peaks, start=1):
        y_index, x_index, score, overlap_fraction, at_boundary, curvature = raw
        y_offset = 0.0
        x_offset = 0.0
        if 0 < y_index < scores.shape[0] - 1:
            y_offset = _parabolic_offset(
                float(scores[y_index - 1, x_index]),
                score,
                float(scores[y_index + 1, x_index]),
            )
        if 0 < x_index < scores.shape[1] - 1:
            x_offset = _parabolic_offset(
                float(scores[y_index, x_index - 1]),
                score,
                float(scores[y_index, x_index + 1]),
            )
        peaks.append(
            ZNCCPeak(
                rank=rank,
                shift_yx=(
                    float(shifts_y[y_index] + y_offset),
                    float(shifts_x[x_index] + x_offset),
                ),
                score=score,
                overlap_fraction=overlap_fraction,
                peak_to_sidelobe_ratio=(score - baseline) / denominator,
                at_search_boundary=at_boundary,
                curvature_eigenvalues=curvature,
            )
        )
    return tuple(peaks)


def masked_zncc_max_t_pvalue(
    reference: ArrayLike,
    moving: ArrayLike,
    observed_score: float,
    *,
    search_center_yx: ArrayLike,
    max_shift: float,
    min_overlap: float = 0.25,
    replicates: int = 99,
    seed: int = 0,
) -> float:
    if replicates < 1:
        return float("nan")
    reference_array = np.asarray(reference, dtype=np.float64)
    moving_array = np.asarray(moving, dtype=np.float64)
    if reference_array.shape != moving_array.shape or reference_array.ndim != 2:
        raise ValueError("reference and moving must be equal-size 2D images")
    amplitude = np.abs(np.fft.fft2(moving_array))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(replicates):
        noise_spectrum = np.fft.fft2(rng.normal(size=moving_array.shape))
        unit_phase = noise_spectrum / np.maximum(
            np.abs(noise_spectrum),
            np.finfo(np.float64).eps,
        )
        randomized_spectrum = amplitude * unit_phase
        randomized_spectrum[0, 0] = 0.0
        randomized = np.fft.ifft2(randomized_spectrum).real
        scores, _, _, _ = _zncc_window(
            reference_array,
            randomized,
            search_center_yx=search_center_yx,
            max_shift=max_shift,
            min_overlap=min_overlap,
        )
        null_maximum = float(np.max(scores[np.isfinite(scores)]))
        exceedances += int(null_maximum >= observed_score)
    return float((1 + exceedances) / (replicates + 1))
