from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from .align2d_backend import kernels

from .align2d_diagnostics import (
    EdgeZNCCDiagnostics,
    GraphObservabilityDiagnostics,
    ZNCCPeak,
    graph_observability_diagnostics,
    masked_zncc_max_t_pvalue,
    masked_zncc_top_k,
)

FloatImage = NDArray[np.float32]
FloatStack = NDArray[np.float32]


@dataclass(frozen=True)
class RegistrationEdge:
    reference: int
    moving: int
    lag: int
    shift_yx: NDArray[np.float64]
    score: float
    peak_prominence: float
    confidence: float
    robust_weight: float = 1.0


@dataclass(frozen=True)
class FinalEdgeDiagnostics:
    reference: int
    moving: int
    lag: int
    measured_shift_yx: tuple[float, float]
    predicted_shift_yx: tuple[float, float]
    residual_yx: tuple[float, float]
    initial_score: float
    final_score: float
    initial_confidence: float
    final_confidence: float
    solver_robust_weight: float
    final_robust_weight: float


@dataclass(frozen=True)
class TerminalRescueDiagnostics:
    boundary_index: int
    terminal_side: Literal["negative", "positive"]
    terminal_indices: tuple[int, ...]
    accepted: bool
    cut_capacity: float
    cut_baseline_capacity: float
    cut_median_score: float
    cut_low_score_fraction: float
    cut_median_before: float
    cut_median_after: float
    cut_p10_before: float
    cut_p10_after: float
    connected_p10_before: float
    connected_p10_after: float
    internal_median_before: float
    internal_median_after: float
    median_graph_residual: float
    p90_graph_residual: float
    consensus_inlier_count: int
    consensus_inlier_fraction: float
    component_offset_yx: tuple[float, float]
    max_correction_norm: float
    maximum_inlier_search_fraction: float
    shadow_state: Literal["accepted", "not_needed", "ambiguous", "rejected"]
    shadow_terminal_endpoint_count: int
    shadow_anchor_endpoint_count: int
    shadow_lag_count: int
    shadow_leave_one_node_out_count: int
    shadow_leave_one_node_out_max_deviation: float
    shadow_crossfit_positive_fraction: float
    shadow_crossfit_median_gain: float
    shadow_bimodal: bool


@dataclass(frozen=True)
class _WeakTerminalCut:
    boundary_index: int
    terminal_side: Literal["negative", "positive"]
    terminal_indices: tuple[int, ...]
    capacity: float
    baseline_capacity: float
    median_score: float
    low_score_fraction: float


@dataclass(frozen=True)
class _RescueShadowAssessment:
    state: Literal["accepted", "not_needed", "ambiguous", "rejected"]
    terminal_endpoint_count: int
    anchor_endpoint_count: int
    lag_count: int
    leave_one_node_out_count: int
    leave_one_node_out_max_deviation: float
    crossfit_positive_fraction: float
    crossfit_median_gain: float
    bimodal: bool


@dataclass(frozen=True)
class Alignment2DResult:
    aligned_stack: FloatStack
    shifts_yx: NDArray[np.float64]
    reference_index: int
    edges: tuple[RegistrationEdge, ...]
    node_confidence: NDArray[np.float64]
    initial_node_confidence: NDArray[np.float64]
    final_edge_diagnostics: tuple[FinalEdgeDiagnostics, ...]
    candidate_bad_tilts: NDArray[np.bool_]
    isolated_bad_tilts: NDArray[np.bool_]
    terminal_bad_tilts: NDArray[np.bool_]
    consensus_corrections_yx: NDArray[np.float64]
    interleaved_branch_mode: bool
    interleaved_branch_offset_yx: NDArray[np.float64]
    interleaved_branch_signal: float
    interleaved_branch_spread: float
    terminal_rescue_corrections_yx: NDArray[np.float64]
    terminal_rescue_diagnostics: tuple[TerminalRescueDiagnostics, ...]
    graph_observability: GraphObservabilityDiagnostics | None
    shadow_zncc_diagnostics: tuple[EdgeZNCCDiagnostics, ...]


def _neutral_level(image: NDArray[np.float32]) -> float:
    finite = image[np.isfinite(image)]
    return float(np.median(finite)) if finite.size else 0.0


def apply_translation(
    image: ArrayLike,
    shift_yx: ArrayLike,
    *,
    fill_value: float | None = None,
) -> FloatImage:
    """Translate an image and fill unsupported pixels with a neutral gray level."""

    data = np.asarray(image, dtype=np.float32)
    neutral = _neutral_level(data) if fill_value is None else float(fill_value)
    return kernels().shift(
        data,
        shift=np.asarray(shift_yx, dtype=np.float64),
        order=1,
        mode="constant",
        cval=neutral,
        prefilter=False,
    ).astype(np.float32, copy=False)


def _translated_support(shape: tuple[int, int], shift_yx: ArrayLike) -> NDArray[np.float32]:
    return kernels().shift(
        np.ones(shape, dtype=np.float32),
        shift=np.asarray(shift_yx, dtype=np.float64),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)


def apply_stack_translations(stack: ArrayLike, shifts_yx: ArrayLike) -> FloatStack:
    data = np.asarray(stack, dtype=np.float32)
    shifts = np.asarray(shifts_yx, dtype=np.float64)
    if shifts.shape != (data.shape[0], 2):
        raise ValueError(f"Expected shifts shape {(data.shape[0], 2)}, got {shifts.shape}")
    return np.stack([apply_translation(image, delta) for image, delta in zip(data, shifts)])


def _metric_image(image: ArrayLike, *, window: bool = True) -> NDArray[np.float32]:
    data = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.all():
        data = np.where(finite, data, _neutral_level(data)).astype(np.float32)
    low, high = np.percentile(data, (1.0, 99.0))
    if high > low:
        data = np.clip(data, low, high)
    # Registration must follow specimen structure rather than the broad illumination bands
    # present in these tilt series. This filtering is metric-only: output intensities are never
    # normalized or high-pass filtered.
    highpass_sigma = max(3.0, min(data.shape) / 32.0)
    data = data - kernels().gaussian_filter(data, highpass_sigma, mode="reflect")
    data = data - float(data.mean())
    scale = float(data.std())
    if scale > np.finfo(np.float32).eps:
        data = data / scale
    if window:
        taper = np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))
        data = data * taper.astype(np.float32)
    return data.astype(np.float32, copy=False)


def _multiband_metric_image(image: ArrayLike) -> NDArray[np.float32]:
    """Return a contrast-invariant metric which preserves several structural scales."""

    data = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.all():
        data = np.where(finite, data, _neutral_level(data)).astype(np.float32)
    low, high = np.percentile(data, (1.0, 99.0))
    if high > low:
        data = np.clip((data - low) / (high - low), 0.0, 1.0)
    else:
        data = np.zeros_like(data)

    # These sigmas are (1, 4, 16, 48) at the normal 1024-pixel, bin-4 working size.
    # Scaling them with the field of view keeps the represented frequency bands comparable
    # for smaller unit tests and for future bin-2/bin-1 refinements.
    scale = max(min(data.shape) / 1024.0, 0.25)
    sigmas = tuple(value * scale for value in (1.0, 4.0, 16.0, 48.0))
    blurred = [kernels().gaussian_filter(data, sigma, mode="reflect") for sigma in sigmas]
    bands = [blurred[index] - blurred[index + 1] for index in range(3)]
    normalized: list[NDArray[np.float32]] = []
    epsilon = np.finfo(np.float32).eps
    for band in bands:
        deviation = float(band.std())
        normalized.append((band / max(deviation, epsilon)).astype(np.float32, copy=False))
    metric = 0.25 * normalized[0] + 0.45 * normalized[1] + 0.30 * normalized[2]
    metric -= float(metric.mean())
    metric /= max(float(metric.std()), epsilon)
    return metric.astype(np.float32, copy=False)


def _overlap_score(
    reference: ArrayLike,
    moving: ArrayLike,
    shift_yx: ArrayLike,
    *,
    prepared: bool = False,
) -> float:
    reference_data = np.asarray(reference, dtype=np.float32)
    moving_data = np.asarray(moving, dtype=np.float32)
    if not prepared:
        reference_data = _metric_image(reference_data, window=False)
        moving_data = _metric_image(moving_data, window=False)
    backend = kernels()
    if backend.overlap_score is not None:
        return backend.overlap_score(reference_data, moving_data, shift_yx, prepared=True)
    moved = apply_translation(moving_data, shift_yx, fill_value=0.0)
    support = _translated_support(moving_data.shape, shift_yx) > 0.999
    border = max(2, min(reference_data.shape) // 100)
    support[:border] = False
    support[-border:] = False
    support[:, :border] = False
    support[:, -border:] = False
    if support.mean() < 0.25:
        return -1.0
    left = reference_data[support].astype(np.float64)
    right = moved[support].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else -1.0


def _parabolic_offset(before: float, center: float, after: float) -> float:
    denominator = before - 2.0 * center + after
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))


def _bounded_xcorr(
    reference: ArrayLike,
    moving: ArrayLike,
    *,
    max_shift: float,
    prepared: bool = False,
    search_center_yx: ArrayLike | None = None,
    taper: bool = True,
) -> tuple[NDArray[np.float64], float]:
    """Linear cross-correlation whose peak search is genuinely shift-bounded."""

    reference_metric = (
        np.asarray(reference, dtype=np.float32) if prepared else _metric_image(reference)
    )
    moving_metric = np.asarray(moving, dtype=np.float32) if prepared else _metric_image(moving)
    if prepared and taper:
        window = np.outer(
            np.hanning(reference_metric.shape[0]), np.hanning(reference_metric.shape[1])
        ).astype(np.float32)
        reference_metric = reference_metric * window
        moving_metric = moving_metric * window
    correlation = kernels().fftconvolve(reference_metric, moving_metric[::-1, ::-1], mode="full")
    center = np.asarray(reference_metric.shape, dtype=np.int64) - 1
    radius = max(1, int(np.ceil(max_shift)))
    search_center = center.astype(np.float64)
    if search_center_yx is not None:
        search_center += np.asarray(search_center_yx, dtype=np.float64)
    lower = np.maximum(np.floor(search_center - radius).astype(np.int64), 0)
    upper = np.minimum(
        np.ceil(search_center + radius).astype(np.int64) + 1,
        correlation.shape,
    )
    if np.any(upper <= lower):
        raise ValueError("bounded cross-correlation search does not overlap valid shifts")
    y0, x0 = lower
    y1, x1 = upper
    search = correlation[y0:y1, x0:x1]
    peak_local = np.asarray(np.unravel_index(np.argmax(search), search.shape), dtype=np.int64)
    peak = peak_local + np.asarray([y0, x0])
    shift = (peak - center).astype(np.float64)
    for axis in range(2):
        if peak[axis] <= 0 or peak[axis] >= correlation.shape[axis] - 1:
            continue
        lower = peak.copy()
        upper = peak.copy()
        lower[axis] -= 1
        upper[axis] += 1
        shift[axis] += _parabolic_offset(
            float(correlation[tuple(lower)]),
            float(correlation[tuple(peak)]),
            float(correlation[tuple(upper)]),
        )
    suppressed = search.copy()
    py, px = peak_local
    suppressed[max(0, py - 3) : py + 4, max(0, px - 3) : px + 4] = -np.inf
    second = (
        float(np.max(suppressed)) if np.isfinite(suppressed).any() else float(np.median(search))
    )
    baseline = float(np.median(search))
    numerator = max(float(search[tuple(peak_local)]) - baseline, 0.0)
    denominator = max(second - baseline, np.finfo(np.float32).eps)
    prominence = float(np.clip(numerator / denominator, 1.0, 20.0))
    return shift, prominence


def _block_mean(image: NDArray[np.float32], factor: int) -> NDArray[np.float32]:
    if factor == 1:
        return image
    height = image.shape[0] // factor
    width = image.shape[1] // factor
    trimmed = image[: height * factor, : width * factor]
    return trimmed.reshape(height, factor, width, factor).mean(axis=(1, 3), dtype=np.float32)


def _build_pyramids(stack: NDArray[np.float32], factors: tuple[int, ...]) -> dict[int, FloatStack]:
    return {
        factor: (
            stack
            if factor == 1
            else np.stack([_block_mean(image, factor) for image in stack]).astype(np.float32)
        )
        for factor in factors
    }


def _build_metric_pyramids(pyramids: dict[int, FloatStack]) -> dict[int, FloatStack]:
    return {
        factor: np.stack([_metric_image(image, window=False) for image in stack]).astype(np.float32)
        for factor, stack in pyramids.items()
    }


def _phase_candidates(
    reference: NDArray[np.float32],
    moving: NDArray[np.float32],
    predicted_shift_yx: NDArray[np.float64],
    *,
    max_residual: float,
    upsample_factor: int,
    prepared: bool = False,
) -> list[tuple[NDArray[np.float64], Literal["xcorr", "phase"]]]:
    reference_metric = (
        np.asarray(reference, dtype=np.float32)
        if prepared
        else _metric_image(reference, window=False)
    )
    moving_metric = (
        np.asarray(moving, dtype=np.float32) if prepared else _metric_image(moving, window=False)
    )
    prealigned = apply_translation(moving_metric, predicted_shift_yx, fill_value=0.0)
    taper = np.outer(
        np.hanning(reference_metric.shape[0]), np.hanning(reference_metric.shape[1])
    ).astype(np.float32)
    reference_metric = reference_metric * taper
    moving_metric = prealigned * taper
    candidates: list[tuple[NDArray[np.float64], Literal["xcorr", "phase"]]] = []
    for normalization, label in ((None, "xcorr"), ("phase", "phase")):
        residual, _, _ = kernels().phase_cross_correlation(
            reference_metric,
            moving_metric,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
        residual = np.asarray(residual, dtype=np.float64)
        if np.isfinite(residual).all() and np.all(np.abs(residual) <= max_residual):
            candidates.append((predicted_shift_yx + residual, label))
    return candidates


def _multiscale_edge(
    pyramids: dict[int, FloatStack],
    metric_pyramids: dict[int, FloatStack],
    reference_index: int,
    moving_index: int,
    *,
    factors: tuple[int, ...],
    max_shift: float,
    upsample_factor: int,
    predicted_shift_yx: NDArray[np.float64] | None = None,
    refinement_max_residual: float | None = None,
    initial_max_shift: float | None = None,
    initial_peak_method: str = 'hann',
) -> tuple[NDArray[np.float64], float, float]:
    coarsest = factors[0]
    reference_coarse = metric_pyramids[coarsest][reference_index]
    moving_coarse = metric_pyramids[coarsest][moving_index]
    if predicted_shift_yx is None:
        # Acquisition motion is not a residual around an existing alignment.
        # In particular, the true first-neighbour lag can exceed max_shift.
        initial_limit = max_shift if initial_max_shift is None else initial_max_shift
        # Hann-only acquisition is the selected default after same-image
        # controls. Coarse overlap ranking against an untapered hypothesis
        # could discard the useful Hann basin before fine registration.
        coarse_shift, prominence = _bounded_xcorr(
            reference_coarse, moving_coarse,
            max_shift=initial_limit / coarsest, prepared=True, taper=True,
        )
        if initial_peak_method == 'dual':
            alternate, alternate_prominence = _bounded_xcorr(reference_coarse, moving_coarse,
                max_shift=initial_limit / coarsest, prepared=True, taper=False)
            if _overlap_score(reference_coarse,moving_coarse,alternate,prepared=True) > _overlap_score(reference_coarse,moving_coarse,coarse_shift,prepared=True):
                coarse_shift,prominence=alternate,alternate_prominence
        shift = coarse_shift * coarsest
        remaining_factors = factors[1:]
    else:
        shift = np.asarray(predicted_shift_yx, dtype=np.float64).copy()
        prominence = 1.0
        remaining_factors = factors

    best_score = -1.0
    for factor in remaining_factors:
        reference = metric_pyramids[factor][reference_index]
        moving = metric_pyramids[factor][moving_index]
        scaled = shift / factor
        if refinement_max_residual is not None:
            residual_limit = max(2.0, refinement_max_residual / factor)
        else:
            residual_limit = max(2.0, min(max_shift / factor, 12.0 if factor > 1 else 8.0))
        candidates = [(scaled, "xcorr")]
        candidates.extend(
            _phase_candidates(
                reference,
                moving,
                scaled,
                max_residual=residual_limit,
                upsample_factor=upsample_factor,
                prepared=True,
            )
        )
        scored = [
            (_overlap_score(reference, moving, candidate, prepared=True), candidate)
            for candidate, _ in candidates
        ]
        best_score, best = max(scored, key=lambda item: item[0])
        shift = best * factor

    if not remaining_factors:
        best_score = _overlap_score(
            reference_coarse, moving_coarse, shift / coarsest, prepared=True
        )
    return shift, float(best_score), prominence


def _edge_confidence(score: float, prominence: float, lag: int) -> float:
    correlation_term = np.clip((score - 0.02) / 0.65, 0.02, 1.0)
    peak_term = np.clip((prominence - 1.0) / 1.5, 0.15, 1.0)
    return float(np.clip(correlation_term * peak_term / np.sqrt(lag), 0.005, 1.0))


def _final_score_confidence(score: float) -> float:
    """Similarity confidence at a fixed final state, without stale peak metadata.

    Peak prominence belongs to the search that produced the original edge and is no longer a
    valid property after consensus or component rescue.  Ambiguity at the final state is exported
    independently by the shadow ZNCC hypotheses.  Keeping this term lag-free also ensures the
    graph diagnostic applies its explicit lag penalty exactly once.
    """

    return float(np.clip((score - 0.02) / 0.65, 0.02, 1.0))


def _estimate_edges(
    pyramids: dict[int, FloatStack],
    metric_pyramids: dict[int, FloatStack],
    *,
    factors: tuple[int, ...],
    lags: tuple[int, ...],
    max_shift: float,
    upsample_factor: int,
    predicted_shifts: NDArray[np.float64] | None = None,
    refinement_max_residual: float | None = None,
    initial_max_shift: float | None = None,
    initial_peak_method: str = 'hann',
) -> list[RegistrationEdge]:
    count = pyramids[1].shape[0]
    edges: list[RegistrationEdge] = []
    for lag in lags:
        for reference in range(count - lag):
            moving = reference + lag
            predicted = None
            if predicted_shifts is not None:
                predicted = predicted_shifts[moving] - predicted_shifts[reference]
            shift, score, prominence = _multiscale_edge(
                pyramids,
                metric_pyramids,
                reference,
                moving,
                factors=factors,
                max_shift=max_shift,
                upsample_factor=upsample_factor,
                predicted_shift_yx=predicted,
                refinement_max_residual=refinement_max_residual,
                initial_max_shift=initial_max_shift,
                initial_peak_method=initial_peak_method,
            )
            edges.append(
                RegistrationEdge(
                    reference=reference,
                    moving=moving,
                    lag=lag,
                    shift_yx=shift,
                    score=score,
                    peak_prominence=prominence,
                    confidence=_edge_confidence(score, prominence, lag),
                )
            )
    return edges


def _cut_statistics(
    edges: list[RegistrationEdge],
    boundary_index: int,
    excluded: NDArray[np.bool_] | None = None,
) -> tuple[float, float, float]:
    spanning = [
        edge
        for edge in edges
        if edge.reference <= boundary_index < edge.moving
        and (excluded is None or (not excluded[edge.reference] and not excluded[edge.moving]))
    ]
    if not spanning:
        return 0.0, -1.0, 1.0
    scores = np.asarray([edge.score for edge in spanning], dtype=np.float64)
    lag_weights = np.asarray([1.0 / edge.lag**1.5 for edge in spanning], dtype=np.float64)
    evidence = np.maximum(scores - 0.005, 0.0)
    capacity = float(np.dot(evidence, lag_weights) / lag_weights.sum())
    return capacity, float(np.median(scores)), float(np.mean(scores < 0.01))


def _detect_weak_terminal_cuts(
    count: int,
    edges: list[RegistrationEdge],
    reference_index: int,
    *,
    maximum_lag: int,
    baseline_window: int = 6,
    excluded: ArrayLike | None = None,
) -> tuple[_WeakTerminalCut, ...]:
    """Find a low-conductance cut which detaches an otherwise coherent terminal block."""

    if count <= 2 * maximum_lag or not 0 <= reference_index < count:
        return ()
    if excluded is None:
        excluded_mask = np.zeros(count, dtype=np.bool_)
    else:
        excluded_mask = np.asarray(excluded, dtype=np.bool_)
        if excluded_mask.shape != (count,):
            raise ValueError("excluded must have one entry per graph node")
    statistics = {
        boundary: _cut_statistics(edges, boundary, excluded_mask) for boundary in range(count - 1)
    }
    candidates: dict[Literal["negative", "positive"], list[_WeakTerminalCut]] = {
        "negative": [],
        "positive": [],
    }
    for boundary in range(count - 1):
        if boundary < reference_index:
            side: Literal["negative", "positive"] = "negative"
            terminal_indices = tuple(range(boundary + 1))
            inward_boundaries = range(
                boundary + 1,
                min(reference_index, boundary + baseline_window + 1),
            )
            outward_neighbor = boundary - 1
            inward_neighbor = boundary + 1
        else:
            side = "positive"
            terminal_indices = tuple(range(boundary + 1, count))
            inward_boundaries = range(max(reference_index, boundary - baseline_window), boundary)
            outward_neighbor = boundary + 1
            inward_neighbor = boundary - 1
        if len(terminal_indices) < maximum_lag:
            continue
        inward_capacities = [statistics[index][0] for index in inward_boundaries]
        if len(inward_capacities) < 3:
            continue
        baseline = float(np.median(inward_capacities))
        capacity, median_score, low_fraction = statistics[boundary]
        if baseline <= np.finfo(np.float64).eps:
            continue
        neighbor_indices = [
            index for index in (outward_neighbor, inward_neighbor) if 0 <= index < count - 1
        ]
        if len(neighbor_indices) < 2:
            continue
        neighbor_capacity = min(statistics[index][0] for index in neighbor_indices)
        if not (
            median_score < 0.01
            and low_fraction >= 0.70
            and capacity < 0.20 * baseline
            and capacity < 0.35 * neighbor_capacity
        ):
            continue
        candidates[side].append(
            _WeakTerminalCut(
                boundary_index=boundary,
                terminal_side=side,
                terminal_indices=terminal_indices,
                capacity=capacity,
                baseline_capacity=baseline,
                median_score=median_score,
                low_score_fraction=low_fraction,
            )
        )

    selected: list[_WeakTerminalCut] = []
    for side in ("negative", "positive"):
        if candidates[side]:
            selected.append(
                min(
                    candidates[side],
                    key=lambda cut: cut.capacity / cut.baseline_capacity,
                )
            )
    return tuple(selected)


def _solve_translation_graph(
    count: int,
    edges: list[RegistrationEdge],
    reference_index: int,
    *,
    iterations: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if count < 1 or not 0 <= reference_index < count:
        raise ValueError("invalid graph size or reference index")
    columns = [index for index in range(count) if index != reference_index]
    column_lookup = {node: column for column, node in enumerate(columns)}
    design = np.zeros((len(edges), count - 1), dtype=np.float64)
    observations = np.empty((len(edges), 2), dtype=np.float64)
    base_weights = np.empty(len(edges), dtype=np.float64)
    for row, edge in enumerate(edges):
        if edge.reference != reference_index:
            design[row, column_lookup[edge.reference]] = -1.0
        if edge.moving != reference_index:
            design[row, column_lookup[edge.moving]] = 1.0
        observations[row] = edge.shift_yx
        # Consecutive tilts define the local registration. Longer edges close loops and
        # bridge bad projections, but must not outvote all first-neighbor measurements
        # merely because there are two additional lag families in the graph.
        base_weights[row] = edge.confidence / (edge.lag**1.5)

    rank = int(np.linalg.matrix_rank(design))
    if rank != count - 1:
        raise ValueError(
            "translation graph is disconnected or rank deficient: "
            f"rank {rank}, expected {count - 1}"
        )

    robust = np.ones(len(edges), dtype=np.float64)
    solution = np.zeros((count - 1, 2), dtype=np.float64)
    for _ in range(iterations):
        weights = np.sqrt(np.maximum(base_weights * robust, 1e-8))
        solution = np.linalg.lstsq(
            design * weights[:, None], observations * weights[:, None], rcond=None
        )[0]
        residual_vectors = design @ solution - observations
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        center = float(np.median(residual_norms))
        scale = max(1.4826 * float(np.median(np.abs(residual_norms - center))), 0.25)
        cutoff = center + 1.5 * scale
        updated = np.minimum(1.0, cutoff / np.maximum(residual_norms, 1e-8))
        if np.max(np.abs(updated - robust)) < 1e-3:
            robust = updated
            break
        robust = updated

    shifts = np.zeros((count, 2), dtype=np.float64)
    shifts[columns] = solution
    residual_norms = np.linalg.norm(design @ solution - observations, axis=1)
    return shifts, robust, residual_norms


def _estimate_multiband_residual_edge(
    metrics: NDArray[np.float32],
    reference_index: int,
    moving_index: int,
    predicted_shift_yx: NDArray[np.float64],
    *,
    max_residual: float,
    upsample_factor: int,
) -> RegistrationEdge:
    reference = metrics[reference_index]
    moving = metrics[moving_index]
    predicted = np.asarray(predicted_shift_yx, dtype=np.float64)
    bounded_shift, prominence = _bounded_xcorr(
        reference,
        moving,
        max_shift=max_residual,
        prepared=True,
        search_center_yx=predicted,
    )
    candidates: list[tuple[NDArray[np.float64], float]] = [
        (predicted, 1.0),
        (bounded_shift, prominence),
    ]
    candidates.extend(
        (candidate, 1.0)
        for candidate, _ in _phase_candidates(
            reference,
            moving,
            predicted,
            max_residual=max_residual,
            upsample_factor=upsample_factor,
            prepared=True,
        )
    )
    scored = [
        (_overlap_score(reference, moving, shift, prepared=True), shift, peak)
        for shift, peak in candidates
    ]
    score, best_shift, best_prominence = max(scored, key=lambda item: item[0])
    lag = moving_index - reference_index
    return RegistrationEdge(
        reference=reference_index,
        moving=moving_index,
        lag=lag,
        shift_yx=best_shift - predicted,
        score=float(score),
        peak_prominence=float(best_prominence),
        confidence=_edge_confidence(float(score), float(best_prominence), lag),
    )


def _build_terminal_rescue_edges(
    metrics: NDArray[np.float32],
    shifts: NDArray[np.float64],
    cut: _WeakTerminalCut,
    excluded: NDArray[np.bool_],
    *,
    maximum_lag: int,
    max_residual: float,
    upsample_factor: int,
) -> list[RegistrationEdge]:
    count = metrics.shape[0]
    terminal = set(cut.terminal_indices)
    if cut.terminal_side == "negative":
        anchors = set(
            range(
                cut.boundary_index + 1,
                min(count, cut.boundary_index + maximum_lag + 1),
            )
        )
    else:
        anchors = set(
            range(
                max(0, cut.boundary_index - maximum_lag + 1),
                cut.boundary_index + 1,
            )
        )
    eligible = terminal | anchors
    edges: list[RegistrationEdge] = []
    for lag in range(1, maximum_lag + 1):
        for reference in range(count - lag):
            moving = reference + lag
            if reference not in eligible or moving not in eligible:
                continue
            if excluded[reference] or excluded[moving]:
                continue
            if not (reference <= cut.boundary_index < moving):
                continue
            predicted = shifts[moving] - shifts[reference]
            edges.append(
                _estimate_multiband_residual_edge(
                    metrics,
                    reference,
                    moving,
                    predicted,
                    max_residual=max_residual,
                    upsample_factor=upsample_factor,
                )
            )
    return edges


def _solve_rigid_terminal_offset(
    count: int,
    edges: list[RegistrationEdge],
    cut: _WeakTerminalCut,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    corrections = np.zeros((count, 2), dtype=np.float64)
    if not edges:
        return corrections, np.empty(0, dtype=np.float64)
    # For a negative terminal component the measured edge is terminal -> anchor and
    # d = c_anchor - c_terminal, hence the minus sign.  The orientation is reversed
    # for a positive terminal component.  Constraining every terminal node to one
    # vector fixes the weak component's gauge without disturbing its internal alignment.
    candidates = _terminal_offset_candidates(edges, cut)
    # Once an edge has passed the multiband candidate search, its score is a validation
    # statistic rather than a geometric precision estimate.  Score-squared weighting can
    # let one attractive false peak pull a small cut away from the vector median, exactly
    # the failure mode this consensus is meant to prevent.
    offset, _ = _robust_vector_location(candidates, np.ones(len(edges), dtype=np.float64))
    corrections[list(cut.terminal_indices)] = offset
    residuals = np.linalg.norm(candidates - offset, axis=1)
    return corrections, residuals


def _terminal_offset_candidates(
    edges: list[RegistrationEdge],
    cut: _WeakTerminalCut,
) -> NDArray[np.float64]:
    sign = -1.0 if cut.terminal_side == "negative" else 1.0
    return sign * np.asarray([edge.shift_yx for edge in edges], dtype=np.float64)


def _has_bimodal_vector_consensus(candidates: NDArray[np.float64]) -> bool:
    if candidates.shape[0] < 6:
        return False
    distances = np.linalg.norm(candidates[:, None, :] - candidates[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centers = np.stack((candidates[first], candidates[second])).astype(np.float64)
    labels = np.zeros(candidates.shape[0], dtype=np.int8)
    for _ in range(12):
        updated_labels = np.argmin(
            np.linalg.norm(candidates[:, None, :] - centers[None, :, :], axis=2),
            axis=1,
        ).astype(np.int8)
        if np.array_equal(updated_labels, labels) and _ > 0:
            break
        labels = updated_labels
        for label in (0, 1):
            members = candidates[labels == label]
            if members.size:
                centers[label] = np.median(members, axis=0)
    counts = np.bincount(labels, minlength=2)
    minimum_cluster = max(2, int(np.ceil(0.25 * candidates.shape[0])))
    if np.min(counts) < minimum_cluster:
        return False
    within = np.asarray(
        [
            np.median(np.linalg.norm(candidates[labels == label] - centers[label], axis=1))
            for label in (0, 1)
        ],
        dtype=np.float64,
    )
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    return separation > max(8.0, 3.0 * (float(np.max(within)) + 0.25))


def _assess_terminal_rescue_shadow(
    metrics: NDArray[np.float32],
    shifts: NDArray[np.float64],
    cut: _WeakTerminalCut,
    edges: list[RegistrationEdge],
    corrections: NDArray[np.float64],
    cut_before: NDArray[np.float64],
    cut_after: NDArray[np.float64],
    *,
    baseline_accepted: bool,
    max_residual: float,
) -> _RescueShadowAssessment:
    terminal = set(cut.terminal_indices)
    terminal_endpoints = {
        endpoint
        for edge in edges
        for endpoint in (edge.reference, edge.moving)
        if endpoint in terminal
    }
    anchor_endpoints = {
        endpoint
        for edge in edges
        for endpoint in (edge.reference, edge.moving)
        if endpoint not in terminal
    }
    lag_count = len({edge.lag for edge in edges})
    finite_before = cut_before[np.isfinite(cut_before)]
    before_median = float(np.median(finite_before)) if finite_before.size else -1.0
    before_p10 = float(np.percentile(finite_before, 10)) if finite_before.size else -1.0
    not_needed = before_median >= 0.03 and before_p10 >= 0.02

    candidates = _terminal_offset_candidates(edges, cut)
    bimodal = _has_bimodal_vector_consensus(candidates)
    full_offset = (
        corrections[list(cut.terminal_indices)][0]
        if cut.terminal_indices
        else np.zeros(2, dtype=np.float64)
    )
    fold_gains: list[float] = []
    fold_deviations: list[float] = []
    endpoints = sorted(terminal_endpoints | anchor_endpoints)
    zero = np.zeros_like(corrections)
    for held_node in endpoints:
        train_indices = [
            index
            for index, edge in enumerate(edges)
            if held_node not in (edge.reference, edge.moving)
        ]
        test_indices = [
            index for index, edge in enumerate(edges) if held_node in (edge.reference, edge.moving)
        ]
        if len(train_indices) < 3 or not test_indices:
            continue
        train_values = candidates[train_indices]
        fold_offset, _ = _robust_vector_location(
            train_values,
            np.ones(len(train_indices), dtype=np.float64),
        )
        fold_corrections = np.zeros_like(corrections)
        fold_corrections[list(cut.terminal_indices)] = fold_offset
        before_scores = np.asarray(
            [_score_rescue_edge(metrics, shifts, zero, edges[index]) for index in test_indices],
            dtype=np.float64,
        )
        after_scores = np.asarray(
            [
                _score_rescue_edge(
                    metrics,
                    shifts,
                    fold_corrections,
                    edges[index],
                )
                for index in test_indices
            ],
            dtype=np.float64,
        )
        fold_gains.append(float(np.median(after_scores - before_scores)))
        fold_deviations.append(float(np.linalg.norm(fold_offset - full_offset)))

    fold_count = len(fold_gains)
    positive_fraction = float(np.mean(np.asarray(fold_gains) > 0.0)) if fold_gains else 0.0
    median_gain = float(np.median(fold_gains)) if fold_gains else -np.inf
    maximum_deviation = float(np.max(fold_deviations)) if fold_deviations else np.inf
    coverage_ok = len(terminal_endpoints) >= 2 and len(anchor_endpoints) >= 2 and lag_count >= 2
    crossfit_ok = (
        fold_count >= 2
        and positive_fraction >= 0.75
        and median_gain >= 0.01
        and maximum_deviation <= max(6.0, 0.05 * max_residual)
    )
    finite_after = cut_after[np.isfinite(cut_after)]
    after_median = float(np.median(finite_after)) if finite_after.size else -1.0
    if not_needed:
        state: Literal["accepted", "not_needed", "ambiguous", "rejected"] = "not_needed"
    elif baseline_accepted and coverage_ok and crossfit_ok and not bimodal:
        state = "accepted"
    elif baseline_accepted or after_median >= before_median + 0.02:
        state = "ambiguous"
    else:
        state = "rejected"
    return _RescueShadowAssessment(
        state=state,
        terminal_endpoint_count=len(terminal_endpoints),
        anchor_endpoint_count=len(anchor_endpoints),
        lag_count=lag_count,
        leave_one_node_out_count=fold_count,
        leave_one_node_out_max_deviation=maximum_deviation,
        crossfit_positive_fraction=positive_fraction,
        crossfit_median_gain=median_gain,
        bimodal=bimodal,
    )


def _score_rescue_edge(
    metrics: NDArray[np.float32],
    shifts: NDArray[np.float64],
    corrections: NDArray[np.float64],
    edge: RegistrationEdge,
) -> float:
    relative = (
        shifts[edge.moving]
        + corrections[edge.moving]
        - shifts[edge.reference]
        - corrections[edge.reference]
    )
    return _overlap_score(
        metrics[edge.reference],
        metrics[edge.moving],
        relative,
        prepared=True,
    )


def _attempt_terminal_rescue(
    metrics: NDArray[np.float32],
    shifts: NDArray[np.float64],
    cut: _WeakTerminalCut,
    excluded: NDArray[np.bool_],
    *,
    maximum_lag: int,
    max_residual: float,
    upsample_factor: int,
) -> tuple[NDArray[np.float64], TerminalRescueDiagnostics]:
    edges = _build_terminal_rescue_edges(
        metrics,
        shifts,
        cut,
        excluded,
        maximum_lag=maximum_lag,
        max_residual=max_residual,
        upsample_factor=upsample_factor,
    )
    corrections, graph_residuals = _solve_rigid_terminal_offset(metrics.shape[0], edges, cut)
    zero = np.zeros_like(corrections)
    cut_before_all = np.asarray(
        [_score_rescue_edge(metrics, shifts, zero, edge) for edge in edges],
        dtype=np.float64,
    )
    cut_after_all = np.asarray(
        [_score_rescue_edge(metrics, shifts, corrections, edge) for edge in edges],
        dtype=np.float64,
    )
    consensus_limit = max(8.0, 0.03 * max_residual)
    inlier_mask = graph_residuals <= consensus_limit
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_fraction = float(inlier_count / len(edges)) if edges else 0.0
    cut_before = cut_before_all[inlier_mask]
    cut_after = cut_after_all[inlier_mask]
    inlier_residuals = graph_residuals[inlier_mask]
    inlier_search_fraction = (
        float(
            np.max(
                np.abs(np.asarray([edge.shift_yx for edge in edges], dtype=np.float64)[inlier_mask])
            )
            / max_residual
        )
        if inlier_count
        else np.inf
    )
    terminal = set(cut.terminal_indices)
    internal_scores: list[float] = []
    for lag in range(1, maximum_lag + 1):
        for reference in range(metrics.shape[0] - lag):
            moving = reference + lag
            if reference not in terminal or moving not in terminal:
                continue
            if excluded[reference] or excluded[moving]:
                continue
            relative = shifts[moving] - shifts[reference]
            internal_scores.append(
                _overlap_score(
                    metrics[reference],
                    metrics[moving],
                    relative,
                    prepared=True,
                )
            )
    internal_before = np.asarray(internal_scores, dtype=np.float64)
    # A rigid component translation cancels from every internal relative shift.
    internal_after = internal_before.copy()
    connected_before = np.concatenate((cut_before, internal_before))
    connected_after = np.concatenate((cut_after, internal_after))

    def percentile(values: NDArray[np.float64], value: float, fallback: float) -> float:
        return float(np.percentile(values, value)) if values.size else fallback

    cut_median_before = percentile(cut_before, 50, -1.0)
    cut_median_after = percentile(cut_after, 50, -1.0)
    cut_p10_before = percentile(cut_before, 10, -1.0)
    cut_p10_after = percentile(cut_after, 10, -1.0)
    connected_p10_before = percentile(connected_before, 10, -1.0)
    connected_p10_after = percentile(connected_after, 10, -1.0)
    internal_median_before = percentile(internal_before, 50, 0.0)
    internal_median_after = percentile(internal_after, 50, 0.0)
    median_residual = percentile(inlier_residuals, 50, np.inf)
    p90_residual = percentile(inlier_residuals, 90, np.inf)
    terminal_corrections = corrections[list(cut.terminal_indices)]
    max_correction = (
        float(np.max(np.linalg.norm(terminal_corrections, axis=1)))
        if terminal_corrections.size
        else np.inf
    )
    component_offset = (
        (
            float(terminal_corrections[0, 0]),
            float(terminal_corrections[0, 1]),
        )
        if terminal_corrections.size
        else (0.0, 0.0)
    )
    internal_allowance = max(0.02, 0.10 * abs(internal_median_before))
    accepted = bool(
        cut_after.size >= maximum_lag
        and inlier_count >= 3
        and inlier_fraction >= 0.80
        and np.isfinite(corrections).all()
        and cut_median_after >= max(0.03, cut_median_before + 0.03, 2.0 * cut_median_before)
        and cut_p10_after >= 0.02
        and connected_p10_after >= max(0.02, connected_p10_before)
        and internal_median_after >= internal_median_before - internal_allowance
        and max_correction <= max_residual
        and inlier_search_fraction <= 0.95
        and median_residual <= max(10.0, 0.08 * max_residual)
        and p90_residual <= max(25.0, 0.15 * max_residual)
    )
    shadow = _assess_terminal_rescue_shadow(
        metrics,
        shifts,
        cut,
        edges,
        corrections,
        cut_before_all,
        cut_after_all,
        baseline_accepted=accepted,
        max_residual=max_residual,
    )
    diagnostics = TerminalRescueDiagnostics(
        boundary_index=cut.boundary_index,
        terminal_side=cut.terminal_side,
        terminal_indices=cut.terminal_indices,
        accepted=accepted,
        cut_capacity=cut.capacity,
        cut_baseline_capacity=cut.baseline_capacity,
        cut_median_score=cut.median_score,
        cut_low_score_fraction=cut.low_score_fraction,
        cut_median_before=cut_median_before,
        cut_median_after=cut_median_after,
        cut_p10_before=cut_p10_before,
        cut_p10_after=cut_p10_after,
        connected_p10_before=connected_p10_before,
        connected_p10_after=connected_p10_after,
        internal_median_before=internal_median_before,
        internal_median_after=internal_median_after,
        median_graph_residual=median_residual,
        p90_graph_residual=p90_residual,
        consensus_inlier_count=inlier_count,
        consensus_inlier_fraction=inlier_fraction,
        component_offset_yx=component_offset,
        max_correction_norm=max_correction,
        maximum_inlier_search_fraction=inlier_search_fraction,
        shadow_state=shadow.state,
        shadow_terminal_endpoint_count=shadow.terminal_endpoint_count,
        shadow_anchor_endpoint_count=shadow.anchor_endpoint_count,
        shadow_lag_count=shadow.lag_count,
        shadow_leave_one_node_out_count=shadow.leave_one_node_out_count,
        shadow_leave_one_node_out_max_deviation=(shadow.leave_one_node_out_max_deviation),
        shadow_crossfit_positive_fraction=shadow.crossfit_positive_fraction,
        shadow_crossfit_median_gain=shadow.crossfit_median_gain,
        shadow_bimodal=shadow.bimodal,
    )
    return corrections if accepted else zero, diagnostics


def _rescue_weak_terminal_components(
    stack: NDArray[np.float32],
    shifts: NDArray[np.float64],
    cuts: tuple[_WeakTerminalCut, ...],
    excluded: NDArray[np.bool_],
    *,
    maximum_lag: int,
    max_residual: float,
    upsample_factor: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[TerminalRescueDiagnostics, ...],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    current = shifts.copy()
    total_corrections = np.zeros_like(current)
    rescued = np.zeros(stack.shape[0], dtype=np.bool_)
    rejected = np.zeros(stack.shape[0], dtype=np.bool_)
    diagnostics: list[TerminalRescueDiagnostics] = []
    if not cuts:
        return current, total_corrections, (), rescued, rejected
    required_indices: set[int] = set()
    for cut in cuts:
        required_indices.update(cut.terminal_indices)
        if cut.terminal_side == "negative":
            required_indices.update(
                range(
                    cut.boundary_index + 1,
                    min(stack.shape[0], cut.boundary_index + maximum_lag + 1),
                )
            )
        else:
            required_indices.update(
                range(
                    max(0, cut.boundary_index - maximum_lag + 1),
                    cut.boundary_index + 1,
                )
            )
    metrics = np.empty_like(stack, dtype=np.float32)
    for index in sorted(required_indices):
        metrics[index] = _multiband_metric_image(stack[index])
    for cut in cuts:
        corrections, diagnostic = _attempt_terminal_rescue(
            metrics,
            current,
            cut,
            excluded,
            maximum_lag=maximum_lag,
            max_residual=max_residual,
            upsample_factor=upsample_factor,
        )
        diagnostics.append(diagnostic)
        terminal_indices = list(cut.terminal_indices)
        if diagnostic.accepted:
            current += corrections
            total_corrections += corrections
            rescued[terminal_indices] = True
        else:
            rejected[terminal_indices] = True
    return current, total_corrections, tuple(diagnostics), rescued, rejected


def _terminal_low_quality_nodes(
    count: int,
    edges: list[RegistrationEdge],
    reference_index: int,
    *,
    threshold_ratio: float = 0.6,
    minimum_run: int = 3,
) -> NDArray[np.bool_]:
    """Find terminal blocks disconnected from the well-registered central component."""

    adjacent = np.full(count - 1, np.nan, dtype=np.float64)
    for edge in edges:
        if edge.lag == 1:
            adjacent[edge.reference] = edge.score
    discarded = np.zeros(count, dtype=np.bool_)

    def terminal_boundary(edge_indices: list[int]) -> int | None:
        for start in range(3, len(edge_indices) - minimum_run + 1):
            inner = adjacent[edge_indices[max(0, start - 5) : start]]
            inner = inner[np.isfinite(inner)]
            if inner.size < 3:
                continue
            inner_level = float(np.median(inner))
            threshold = max(0.005, threshold_ratio * inner_level)
            run = edge_indices[start : start + minimum_run]
            if not np.all(np.isfinite(adjacent[run]) & (adjacent[run] < threshold)):
                continue
            outer = adjacent[edge_indices[start:]]
            outer = outer[np.isfinite(outer)]
            if outer.size and float(np.mean(outer < 0.8 * inner_level)) >= 0.6:
                return edge_indices[start]
        return None

    negative_boundary = terminal_boundary(list(range(reference_index - 1, -1, -1)))
    if negative_boundary is not None:
        discarded[: negative_boundary + 1] = True
    positive_boundary = terminal_boundary(list(range(reference_index, count - 1)))
    if positive_boundary is not None:
        discarded[positive_boundary + 1 :] = True
    return discarded


def _node_quality(
    stack: NDArray[np.float32],
    edges: list[RegistrationEdge],
    robust_weights: NDArray[np.float64],
    reference_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    count = stack.shape[0]
    confidence = np.zeros(count, dtype=np.float64)
    incident_scores: list[list[float]] = [[] for _ in range(count)]
    for edge, robust in zip(edges, robust_weights):
        value = edge.confidence * robust
        incident_scores[edge.reference].append(value)
        incident_scores[edge.moving].append(value)
    for index, values in enumerate(incident_scores):
        confidence[index] = float(np.median(values)) if values else 0.0

    texture = np.asarray(
        [
            np.median(np.abs(np.diff(image, axis=0))) + np.median(np.abs(np.diff(image, axis=1)))
            for image in stack
        ],
        dtype=np.float64,
    )
    texture_median = max(float(np.median(texture)), np.finfo(np.float64).eps)
    texture_ratio = texture / texture_median
    typical_confidence = float(np.median(confidence))
    low_confidence = confidence < max(0.015, typical_confidence * 0.22)

    direct_scores: list[list[float]] = [[] for _ in range(count)]
    bypass_score = np.zeros(count, dtype=np.float64)
    for edge in edges:
        if edge.lag == 1:
            direct_scores[edge.reference].append(edge.score)
            direct_scores[edge.moving].append(edge.score)
        if edge.lag == 2:
            middle = edge.reference + 1
            bypass_score[middle] = max(bypass_score[middle], edge.score)
    direct_score = np.asarray(
        [float(np.median(values)) if values else 0.0 for values in direct_scores]
    )
    typical_direct = max(float(np.median(direct_score)), 0.02)
    isolated_bad = low_confidence & (
        (texture_ratio < 0.25)
        | (
            (direct_score < max(0.015, typical_direct * 0.35))
            & (bypass_score > np.maximum(direct_score * 1.8, 0.02))
        )
    )
    isolated_bad[[0, count - 1]] = False
    terminal_bad = _terminal_low_quality_nodes(count, edges, reference_index)
    candidate_bad = isolated_bad | terminal_bad
    return confidence, candidate_bad, isolated_bad, terminal_bad


def _robust_vector_location(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    location = np.median(values, axis=0)
    for _ in range(25):
        distances = np.linalg.norm(values - location, axis=1)
        scale = max(1.4826 * float(np.median(distances)), 0.25)
        robust = np.minimum(1.0, 1.5 * scale / np.maximum(distances, 1e-8))
        updated = np.average(values, axis=0, weights=weights * robust)
        if np.linalg.norm(updated - location) < 1e-4:
            location = updated
            break
        location = updated
    spread = float(np.median(np.linalg.norm(values - location, axis=1)))
    return location, spread


def _solve_interleaved_branches(
    count: int,
    edges: list[RegistrationEdge],
    robust_weights: NDArray[np.float64],
    reference_index: int,
    angles_deg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Align even/odd angular subsequences and merge their coordinate frames robustly."""

    branch_shifts = np.zeros((count, 2), dtype=np.float64)
    for parity in (0, 1):
        nodes = list(range(parity, count, 2))
        lookup = {node: index for index, node in enumerate(nodes)}
        component_edges: list[RegistrationEdge] = []
        for edge in edges:
            if edge.reference % 2 != parity or edge.moving % 2 != parity:
                continue
            component_edges.append(
                RegistrationEdge(
                    reference=lookup[edge.reference],
                    moving=lookup[edge.moving],
                    lag=max(1, edge.lag // 2),
                    shift_yx=edge.shift_yx,
                    score=edge.score,
                    peak_prominence=edge.peak_prominence,
                    confidence=edge.confidence,
                    robust_weight=edge.robust_weight,
                )
            )
        branch_reference = min(nodes, key=lambda node: abs(node - reference_index))
        component_shifts, _, _ = _solve_translation_graph(
            len(nodes), component_edges, lookup[branch_reference]
        )
        branch_shifts[nodes] = component_shifts

    offsets: list[NDArray[np.float64]] = []
    offset_weights: list[float] = []
    odd_edge_scores: list[float] = []
    even_edge_scores: list[float] = []
    maximum_angle = max(float(np.max(np.abs(angles_deg))), 1.0)
    high_angle_scores: dict[str, list[float]] = {"odd": [], "even": []}
    for edge, robust in zip(edges, robust_weights):
        midpoint_angle = 0.5 * (angles_deg[edge.reference] + angles_deg[edge.moving])
        same_branch = edge.reference % 2 == edge.moving % 2
        (even_edge_scores if same_branch else odd_edge_scores).append(edge.score)
        if abs(midpoint_angle) >= 0.65 * maximum_angle:
            high_angle_scores["even" if same_branch else "odd"].append(edge.score)
        if same_branch:
            continue
        coefficient = (1 if edge.moving % 2 == 0 else 0) - (1 if edge.reference % 2 == 0 else 0)
        predicted = branch_shifts[edge.moving] - branch_shifts[edge.reference]
        offsets.append((edge.shift_yx - predicted) / coefficient)
        centrality = 1.0 / (1.0 + abs(midpoint_angle) / 30.0)
        offset_weights.append(max(edge.confidence * robust * centrality, 1e-6))

    if not offsets:
        raise ValueError("interleaved branch graph has no cross-branch edges")
    offset, spread = _robust_vector_location(
        np.asarray(offsets, dtype=np.float64), np.asarray(offset_weights, dtype=np.float64)
    )
    branch_shifts[::2] += offset
    branch_shifts -= branch_shifts[reference_index]
    global_signal = float(np.median(even_edge_scores) - np.median(odd_edge_scores))
    high_signal = 0.0
    if high_angle_scores["even"] and high_angle_scores["odd"]:
        high_signal = float(
            np.median(high_angle_scores["even"]) - np.median(high_angle_scores["odd"])
        )
    return branch_shifts, offset, max(global_signal, high_signal), spread


def _consensus_refinement(
    stack: NDArray[np.float32],
    shifts: NDArray[np.float64],
    reference_index: int,
    excluded: NDArray[np.bool_],
    *,
    radius: int,
    passes: int,
    upsample_factor: int,
    stride: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    current = shifts.copy()
    total_corrections = np.zeros_like(current)
    count = stack.shape[0]
    for pass_index in range(passes):
        aligned = apply_stack_translations(stack, current)
        aligned_metrics = np.stack(
            [_metric_image(image, window=False) for image in aligned]
        ).astype(np.float32)
        supports = np.stack([_translated_support(stack.shape[1:], shift) for shift in current])
        corrections = np.zeros_like(current)
        limit = 6.0 if pass_index == 0 else 3.0
        for index in range(count):
            if index == reference_index or excluded[index]:
                continue
            neighbors = [
                neighbor
                for neighbor in range(
                    max(0, index - radius * stride),
                    min(count, index + radius * stride + 1),
                )
                if neighbor != index and (neighbor - index) % stride == 0 and not excluded[neighbor]
            ]
            if len(neighbors) < 2:
                continue
            numerator = np.zeros(stack.shape[1:], dtype=np.float64)
            denominator = np.zeros(stack.shape[1:], dtype=np.float64)
            for neighbor in neighbors:
                weight = stride / abs(neighbor - index)
                numerator += aligned_metrics[neighbor] * supports[neighbor] * weight
                denominator += supports[neighbor] * weight
            template = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0.5,
            ).astype(np.float32)
            candidates = [(np.zeros(2, dtype=np.float64), "xcorr")]
            candidates.extend(
                _phase_candidates(
                    template,
                    aligned_metrics[index],
                    np.zeros(2, dtype=np.float64),
                    max_residual=limit,
                    upsample_factor=upsample_factor,
                    prepared=True,
                )
            )
            scored = [
                (
                    _overlap_score(template, aligned_metrics[index], shift, prepared=True),
                    shift,
                )
                for shift, _ in candidates
            ]
            zero_score = scored[0][0]
            best_score, best = max(scored, key=lambda item: item[0])
            if best_score >= zero_score + 0.01:
                corrections[index] = best
        current += corrections
        current -= current[reference_index]
        total_corrections += corrections
        if np.max(np.abs(corrections)) < 0.05:
            break
    return current, total_corrections


def _final_edge_diagnostics(
    metrics: NDArray[np.float32],
    edges: list[RegistrationEdge],
    robust_weights: NDArray[np.float64],
    shifts: NDArray[np.float64],
) -> tuple[
    tuple[FinalEdgeDiagnostics, ...],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    diagnostics: list[FinalEdgeDiagnostics] = []
    incident_confidences: list[list[float]] = [[] for _ in range(metrics.shape[0])]
    residual_vectors = np.asarray(
        [shifts[edge.moving] - shifts[edge.reference] - edge.shift_yx for edge in edges],
        dtype=np.float64,
    )
    residual_norms = np.linalg.norm(residual_vectors, axis=1)
    center = float(np.median(residual_norms))
    scale = max(1.4826 * float(np.median(np.abs(residual_norms - center))), 0.25)
    cutoff = center + 1.5 * scale
    final_robust_weights = np.minimum(
        1.0,
        cutoff / np.maximum(residual_norms, 1e-8),
    )
    for edge_index, (edge, solver_robust_weight) in enumerate(zip(edges, robust_weights)):
        predicted = shifts[edge.moving] - shifts[edge.reference]
        final_score = _overlap_score(
            metrics[edge.reference],
            metrics[edge.moving],
            predicted,
            prepared=True,
        )
        final_confidence = _final_score_confidence(final_score)
        final_robust_weight = float(final_robust_weights[edge_index])
        weighted_confidence = final_confidence * final_robust_weight
        incident_confidences[edge.reference].append(weighted_confidence)
        incident_confidences[edge.moving].append(weighted_confidence)
        residual = residual_vectors[edge_index]
        diagnostics.append(
            FinalEdgeDiagnostics(
                reference=edge.reference,
                moving=edge.moving,
                lag=edge.lag,
                measured_shift_yx=(float(edge.shift_yx[0]), float(edge.shift_yx[1])),
                predicted_shift_yx=(float(predicted[0]), float(predicted[1])),
                residual_yx=(float(residual[0]), float(residual[1])),
                initial_score=edge.score,
                final_score=final_score,
                initial_confidence=edge.confidence,
                final_confidence=final_confidence,
                solver_robust_weight=float(solver_robust_weight),
                final_robust_weight=final_robust_weight,
            )
        )
    node_confidence = np.asarray(
        [float(np.median(values)) if values else 0.0 for values in incident_confidences],
        dtype=np.float64,
    )
    return tuple(diagnostics), node_confidence, final_robust_weights


def _graph_diagnostics(
    count: int,
    edges: list[RegistrationEdge],
    final_edges: tuple[FinalEdgeDiagnostics, ...],
    reference_index: int,
    final_shifts: NDArray[np.float64],
) -> GraphObservabilityDiagnostics:
    references = np.asarray([edge.reference for edge in edges], dtype=np.int64)
    movings = np.asarray([edge.moving for edge in edges], dtype=np.int64)
    measurements = np.asarray([edge.shift_yx for edge in edges], dtype=np.float64)
    base_weights = np.asarray(
        [edge.confidence / edge.lag**1.5 for edge in edges],
        dtype=np.float64,
    )
    final_weights = np.asarray(
        [edge.final_confidence * edge.final_robust_weight / edge.lag**1.5 for edge in final_edges],
        dtype=np.float64,
    )
    return graph_observability_diagnostics(
        count,
        references,
        movings,
        measurements,
        base_weights,
        final_weights,
        reference_index,
        final_shifts,
    )


def _shadow_zncc_diagnostics(
    stack: NDArray[np.float32],
    edges: list[RegistrationEdge],
    final_edges: tuple[FinalEdgeDiagnostics, ...],
    shifts: NDArray[np.float64],
    graph: GraphObservabilityDiagnostics,
    *,
    factor: int,
    top_k: int,
    max_shift: float,
    max_t_edge_count: int,
    max_t_replicates: int,
    seed: int,
) -> tuple[EdgeZNCCDiagnostics, ...]:
    if factor < 1 or top_k < 1 or max_t_edge_count < 0 or max_t_replicates < 0:
        raise ValueError("invalid shadow ZNCC configuration")
    if min(stack.shape[1:]) // factor < 32:
        raise ValueError("shadow ZNCC images must remain at least 32 pixels wide")
    diagnostic_stack = (
        stack
        if factor == 1
        else np.stack([_block_mean(image, factor) for image in stack]).astype(np.float32)
    )
    metrics = np.stack([_multiband_metric_image(image) for image in diagnostic_stack]).astype(
        np.float32
    )
    edge_scores = np.asarray([item.final_score for item in final_edges], dtype=np.float64)
    priority = graph.final_edge_leverage + np.clip(0.10 - edge_scores, 0.0, 1.0)
    calibration_indices = set(
        np.argsort(priority)[-min(max_t_edge_count, len(edges)) :].astype(int).tolist()
        if max_t_edge_count
        else []
    )
    diagnostics: list[EdgeZNCCDiagnostics] = []
    radius_at_factor = max_shift / factor
    for edge_index, edge in enumerate(edges):
        center = (shifts[edge.moving] - shifts[edge.reference]) / factor
        peaks_at_factor = masked_zncc_top_k(
            metrics[edge.reference],
            metrics[edge.moving],
            search_center_yx=center,
            max_shift=radius_at_factor,
            top_k=top_k,
            min_overlap=0.25,
            suppression_radius=max(2, int(np.ceil(6.0 / factor))),
        )
        peaks = tuple(
            ZNCCPeak(
                rank=peak.rank,
                shift_yx=(peak.shift_yx[0] * factor, peak.shift_yx[1] * factor),
                score=peak.score,
                overlap_fraction=peak.overlap_fraction,
                peak_to_sidelobe_ratio=peak.peak_to_sidelobe_ratio,
                at_search_boundary=peak.at_search_boundary,
                curvature_eigenvalues=(
                    peak.curvature_eigenvalues[0] / factor**2,
                    peak.curvature_eigenvalues[1] / factor**2,
                ),
            )
            for peak in peaks_at_factor
        )
        pvalue = float("nan")
        replicates = 0
        if edge_index in calibration_indices and peaks_at_factor and max_t_replicates:
            replicates = max_t_replicates
            pvalue = masked_zncc_max_t_pvalue(
                metrics[edge.reference],
                metrics[edge.moving],
                peaks_at_factor[0].score,
                search_center_yx=center,
                max_shift=radius_at_factor,
                min_overlap=0.25,
                replicates=max_t_replicates,
                seed=seed + 1009 * edge.reference + 9176 * edge.moving,
            )
        diagnostics.append(
            EdgeZNCCDiagnostics(
                reference=edge.reference,
                moving=edge.moving,
                lag=edge.lag,
                analysis_factor=factor,
                search_center_yx=(
                    float(center[0] * factor),
                    float(center[1] * factor),
                ),
                search_radius=max_shift,
                peaks=peaks,
                max_t_pvalue=pvalue,
                max_t_replicates=replicates,
            )
        )
    return tuple(diagnostics)


def align_stack_2d_robust(
    stack: ArrayLike,
    angles_deg: ArrayLike,
    *,
    upsample_factor: int = 10,
    max_shift: float = 100.0,
    initial_max_shift: float | None = None,
    initial_peak_method: str = 'hann',
    graph_lags: tuple[int, ...] = (1, 2, 3, 4),
    pyramid_factors: tuple[int, ...] = (4, 2, 1),
    graph_refinement_rounds: int = 2,
    consensus_radius: int = 2,
    consensus_passes: int = 2,
    shadow_diagnostics: bool = False,
    shadow_zncc_factor: int = 2,
    shadow_zncc_top_k: int = 3,
    shadow_max_t_edge_count: int = 12,
    shadow_max_t_replicates: int = 99,
    shadow_seed: int = 0,
) -> Alignment2DResult:
    """Robust translation-only alignment using a multiscale registration graph."""

    data = np.asarray(stack, dtype=np.float32)
    if initial_peak_method not in ('hann','dual'):
        raise ValueError('initial_peak_method must be hann or dual')
    angles = np.asarray(angles_deg, dtype=np.float64)
    if data.ndim != 3 or angles.shape != (data.shape[0],):
        raise ValueError("stack and angles have incompatible shapes")
    if max_shift <= 0 or upsample_factor < 1 or graph_refinement_rounds < 1:
        raise ValueError("max_shift, upsample_factor and graph_refinement_rounds must be positive")
    initial_limit = (max(max_shift, 0.75 * min(data.shape[1:]))
                     if initial_max_shift is None else float(initial_max_shift))
    if not np.isfinite(initial_limit) or initial_limit <= 0:
        raise ValueError("initial_max_shift must be finite and positive")
    if (
        shadow_zncc_factor < 1
        or shadow_zncc_top_k < 1
        or shadow_max_t_edge_count < 0
        or shadow_max_t_replicates < 0
    ):
        raise ValueError("invalid shadow diagnostic configuration")
    if not graph_lags or any(lag < 1 for lag in graph_lags):
        raise ValueError("graph_lags must contain positive integers")
    valid_lags = tuple(sorted({lag for lag in graph_lags if lag < data.shape[0]}))
    if not valid_lags:
        raise ValueError("graph_lags must connect valid distinct projections")
    factors = tuple(sorted(set(pyramid_factors), reverse=True))
    if not factors or factors[-1] != 1 or any(factor < 1 for factor in factors):
        raise ValueError("pyramid_factors must be positive and include 1")
    if min(data.shape[1:]) // factors[0] < 16:
        raise ValueError("coarsest pyramid images must be at least 16 pixels wide")

    reference_index = int(np.argmin(np.abs(angles)))
    pyramids = _build_pyramids(data, factors)
    metric_pyramids = _build_metric_pyramids(pyramids)
    edges = _estimate_edges(
        pyramids,
        metric_pyramids,
        factors=factors,
        lags=valid_lags,
        max_shift=max_shift,
        upsample_factor=upsample_factor,
        initial_max_shift=initial_limit,
        initial_peak_method=initial_peak_method,
    )
    shifts, robust, _ = _solve_translation_graph(data.shape[0], edges, reference_index)

    # Refine around graph-predicted positions without reopening a full-range
    # search: broad remeasurement can reinforce spurious, mutually consistent
    # peaks. Acquisition-motion discovery belongs to the coarse pass above.
    refined_edges = edges
    for _ in range(graph_refinement_rounds):
        refined_edges = _estimate_edges(
            pyramids,
            metric_pyramids,
            factors=(1,),
            lags=valid_lags,
            max_shift=max_shift,
            upsample_factor=upsample_factor,
            predicted_shifts=shifts,
            refinement_max_residual=min(max_shift, 8.0),
        )
        shifts, robust, _ = _solve_translation_graph(data.shape[0], refined_edges, reference_index)
    branch_shifts = shifts
    branch_offset = np.zeros(2, dtype=np.float64)
    branch_signal = 0.0
    branch_spread = np.inf
    try:
        branch_shifts, branch_offset, branch_signal, branch_spread = _solve_interleaved_branches(
            data.shape[0],
            refined_edges,
            robust,
            reference_index,
            angles,
        )
    except ValueError as error:
        if not str(error).startswith(
            (
                "translation graph is disconnected or rank deficient",
                "interleaved branch graph has no cross-branch edges",
            )
        ):
            raise
    branch_mode = branch_signal >= 0.08 and branch_spread <= 20.0
    if branch_mode:
        shifts = branch_shifts
    initial_node_confidence, candidate_bad, isolated_bad, terminal_bad = _node_quality(
        data, refined_edges, robust, reference_index
    )
    shifts, consensus_corrections = _consensus_refinement(
        data,
        shifts,
        reference_index,
        candidate_bad,
        radius=consensus_radius,
        passes=consensus_passes,
        upsample_factor=upsample_factor,
        stride=2 if branch_mode else 1,
    )
    maximum_lag = max(valid_lags)
    weak_terminal_cuts = _detect_weak_terminal_cuts(
        data.shape[0],
        refined_edges,
        reference_index,
        maximum_lag=maximum_lag,
        excluded=isolated_bad,
    )
    minimum_dimension = float(min(data.shape[1:]))
    rescue_max_residual = min(
        0.70 * minimum_dimension,
        max(2.0 * max_shift, 0.40 * minimum_dimension),
    )
    (
        shifts,
        terminal_rescue_corrections,
        terminal_rescue_diagnostics,
        rescued_terminal,
        rejected_terminal,
    ) = _rescue_weak_terminal_components(
        data,
        shifts,
        weak_terminal_cuts,
        isolated_bad,
        maximum_lag=maximum_lag,
        max_residual=rescue_max_residual,
        upsample_factor=upsample_factor,
    )
    terminal_bad = (terminal_bad & ~rescued_terminal) | rejected_terminal
    candidate_bad = isolated_bad | terminal_bad
    final_edge_diagnostics, node_confidence, _ = _final_edge_diagnostics(
        metric_pyramids[1],
        refined_edges,
        robust,
        shifts,
    )
    graph_observability = (
        _graph_diagnostics(
            data.shape[0],
            refined_edges,
            final_edge_diagnostics,
            reference_index,
            shifts,
        )
        if shadow_diagnostics
        else None
    )
    shadow_zncc = (
        _shadow_zncc_diagnostics(
            data,
            refined_edges,
            final_edge_diagnostics,
            shifts,
            graph_observability,
            factor=shadow_zncc_factor,
            top_k=shadow_zncc_top_k,
            max_shift=rescue_max_residual,
            max_t_edge_count=shadow_max_t_edge_count,
            max_t_replicates=shadow_max_t_replicates,
            seed=shadow_seed,
        )
        if shadow_diagnostics and graph_observability is not None
        else ()
    )
    aligned = apply_stack_translations(data, shifts)
    final_edges = tuple(
        RegistrationEdge(
            reference=edge.reference,
            moving=edge.moving,
            lag=edge.lag,
            shift_yx=edge.shift_yx,
            score=edge.score,
            peak_prominence=edge.peak_prominence,
            confidence=edge.confidence,
            robust_weight=float(weight),
        )
        for edge, weight in zip(refined_edges, robust)
    )
    return Alignment2DResult(
        aligned_stack=aligned,
        shifts_yx=shifts,
        reference_index=reference_index,
        edges=final_edges,
        node_confidence=node_confidence,
        initial_node_confidence=initial_node_confidence,
        final_edge_diagnostics=final_edge_diagnostics,
        candidate_bad_tilts=candidate_bad,
        isolated_bad_tilts=isolated_bad,
        terminal_bad_tilts=terminal_bad,
        consensus_corrections_yx=consensus_corrections,
        interleaved_branch_mode=branch_mode,
        interleaved_branch_offset_yx=branch_offset,
        interleaved_branch_signal=branch_signal,
        interleaved_branch_spread=branch_spread,
        terminal_rescue_corrections_yx=terminal_rescue_corrections,
        terminal_rescue_diagnostics=terminal_rescue_diagnostics,
        graph_observability=graph_observability,
        shadow_zncc_diagnostics=shadow_zncc,
    )
