"""Peak extraction from score maps and full precision Greco STAR export.

Replicates greco_picker_extract_candidates (--reference5-compat): Rickgauer et al.
(2017) error-function cut-off, greedy max selection with spherical box-out,
centered Angstrom coordinates and ZXZ->ZYZ Euler conversion.
"""

from __future__ import annotations
from greco.project_layout import join as storage_path

import json
import logging
import math
import pathlib

import numpy as np
from scipy.special import erfcinv

from .geometry import convert_euler, get_angle_list
from .io import PICK_COLUMNS, read_mrc, write_star
from .weights import spherical_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical edge rule (shared by exhaustive and hybrid extraction)
# ---------------------------------------------------------------------------

#: Provenance tag for the canonical edge rule (Round-6H s21): closed
#: interval r <= c <= size-1-r per axis, mask applied BEFORE peak
#: search, coordinate == r valid.  Recorded in job JSONs of both modes.
EDGE_RULE = "closed-mask-first-v1"


def edge_radius_vox(particle_diameter_A: float, voxel_A: float) -> int:
    """r = floor(D/2 / voxel): the production edge radius in voxels."""
    return int((particle_diameter_A / 2.0) / voxel_A)


def edge_keep(pos, shape, r: int) -> np.ndarray:
    """Per-position form of :func:`edge_mask` for a scored position list."""
    pos = np.asarray(pos, dtype=int)
    keep = np.ones(len(pos), dtype=bool)
    for a in range(3):
        keep &= (pos[:, a] >= r) & (pos[:, a] <= shape[a] - 1 - r)
    return keep


def greedy_nms_sparse(pos, sc, cut, n_max: int, r: int) -> np.ndarray:
    """Greedy argmax NMS over a SCORED POSITION LIST (hybrid extraction).

    Sparse form of the volume-based picker in :func:`extract_particles`
    with the identical rules: candidates are visited in descending score
    order (stable for ties); a candidate above ``cut`` is picked and
    suppresses every position within Euclidean distance <= r; the loop
    stops when the best surviving score <= cut or ``n_max`` picks were
    made.  ``pos``: (n,3) int; ``sc``: (n,) float.  Returns pick
    indices into ``pos``.
    """
    pos = np.ascontiguousarray(pos, dtype=np.int64)
    sc = np.asarray(sc)   # dtype kept as-is: research-parity comparisons
    r2 = r * r
    order = np.argsort(-sc, kind="stable")
    alive = np.ones(len(pos), dtype=bool)
    picks = []
    for i in order:
        if not alive[i]:
            continue
        if sc[i] <= cut:
            break
        picks.append(int(i))
        alive &= ~(((pos - pos[i]) ** 2).sum(1) <= r2)
        if len(picks) >= n_max:
            break
    return np.array(picks, dtype=int)


def load_score_maps(output_dir: pathlib.Path, tomo_id: str):
    """Load score/angle maps, merging shard parts if present.

    Applies the global noise correction in a single place when raw maps plus
    per-shard ``_noise_scores`` maps are found::

        scores -= merged_noise; scores += mean(shard noise means)

    The per-shard ``noise_mean`` values were computed by the matcher over the
    full padded FFT volume (Greco picker convention), so a single-shard run receives
    exactly Greco picker's end-of-job correction. Legacy outputs that only contain
    already-corrected ``_scores.mrc`` files are returned unchanged.
    """
    output_dir = pathlib.Path(output_dir)
    single = storage_path(output_dir, f"{tomo_id}_scores.mrc")
    if single.exists():
        scores = read_mrc(single)
        angles = read_mrc(storage_path(output_dir, f"{tomo_id}_angles.mrc"))
        job_files = [storage_path(output_dir, f"{tomo_id}_job.json")]
        noise_files = []
        if (storage_path(output_dir, f"{tomo_id}_noise_scores.mrc")).exists():
            noise_files = [storage_path(output_dir, f"{tomo_id}_noise_scores.mrc")]
    else:
        part_files = sorted(output_dir.glob(f"{tomo_id}_scores_p*of*.mrc"))
        if not part_files:
            raise FileNotFoundError(f"No score maps for {tomo_id} in {output_dir}")
        scores = None
        angles = None
        job_files = []
        for pf in part_files:
            s = read_mrc(pf)
            a = read_mrc(
                pf.with_name(pf.name.replace("_scores_", "_angles_"))
            )
            jf = pathlib.Path(
                str(pf).replace("_scores_", "_job_").replace(".mrc", ".json")
            )
            job_files.append(jf)
            if scores is None:
                scores, angles = s, a
            else:
                better = s > scores
                angles = np.where(better, a, angles)
                angles = np.where(s == scores, np.minimum(a, angles), angles)
                scores = np.where(better, s, scores)
        noise_files = sorted(output_dir.glob(f"{tomo_id}_noise_scores_p*of*.mrc"))
    jobs = []
    for jf in job_files:
        with open(jf) as f:
            jobs.append(json.load(f))

    if noise_files:
        means = [
            j["stats"]["noise_mean"]
            for j in jobs
            if "noise_mean" in j.get("stats", {})
        ]
        if len(means) == len(jobs) and len(noise_files) == len(job_files):
            noise = read_mrc(noise_files[0])
            for nf in noise_files[1:]:
                np.maximum(noise, read_mrc(nf), out=noise)
            scores = scores - noise + float(np.mean(means))
        else:
            logger.warning(
                f"{tomo_id}: incomplete noise-correction data "
                f"({len(means)}/{len(jobs)} job means, "
                f"{len(noise_files)}/{len(job_files)} noise maps); "
                f"leaving score maps as they are"
            )
    return scores.astype(np.float32), angles.astype(np.float32), jobs


def merged_stats(jobs: list[dict]) -> dict:
    """Pooled statistics across shards.

    The variance is combined weighted by each shard's search_space (rotation
    count x ROI size), i.e. the maximum-likelihood pooled estimate. Greco picker
    averages sub-job variances unweighted; with equal-size shards both agree.
    """
    search_space = sum(j["stats"]["search_space"] for j in jobs)
    variance = (
        sum(j["stats"]["variance"] * j["stats"]["search_space"] for j in jobs)
        / search_space
    )
    return {
        "search_space": search_space,
        "variance": variance,
        "std": float(np.sqrt(variance)),
    }


def find_tomo_id(output_dir: pathlib.Path) -> str:
    """Discover the tomogram id from the job json in an output directory."""
    output_dir = pathlib.Path(output_dir)
    current=storage_path(output_dir,'matching/config.json')
    if current.is_file():return json.loads(current.read_text())['tomo_id']
    for jf in sorted(output_dir.glob("*_job.json")):
        with open(jf) as f:
            return json.load(f)["tomo_id"]
    raise FileNotFoundError(f"No *_job.json found in {output_dir}")


def extract_particles(
    output_dir: pathlib.Path,
    n_particles: int = 1000,
    particle_diameter: float | None = None,
    cut_off: float | None = None,
    n_false_positives: float = 1.0,
    tomo_id: str | None = None,
    threshold_factor: float = 1.0,
) -> tuple[list[tuple], list[float], dict]:
    """Greedy peak picking; returns centered coordinates, ZYZ poses, scores and statistics."""
    output_dir = pathlib.Path(output_dir)
    tomo_id = tomo_id or find_tomo_id(output_dir)
    score_volume, angle_volume, jobs = load_score_maps(output_dir, tomo_id)
    stats = merged_stats(jobs)

    sigma = stats["std"]
    search_space = stats["search_space"]
    voxel_size = float(jobs[0]["voxel_size"])
    angle_list = get_angle_list(
        float(jobs[0]["angle_increment"]), sort_angles=True,
        symmetry=int(jobs[0].get("rotational_symmetry", 1)),
    )

    if math.isinf(sigma) or math.isnan(sigma):
        raise ValueError("Score statistics are NaN/inf; check the search")

    if particle_diameter is None:
        raise ValueError("--particle-diameter is required")
    particle_radius_px = int((particle_diameter / 2) / voxel_size)

    # mask volume edges
    r = particle_radius_px
    score_volume[:r, :, :] = 0
    score_volume[:, :r, :] = 0
    score_volume[:, :, :r] = 0
    score_volume[-r:, :, :] = 0
    score_volume[:, -r:, :] = 0
    score_volume[:, :, -r:] = 0

    if isinstance(threshold_factor,bool) or not math.isfinite(threshold_factor) or threshold_factor<=0:
        raise ValueError('Threshold factor must be finite and positive.')
    automatic_cut_off=None
    if cut_off is None:
        # Rickgauer et al. (2017): N^-1 = erfc(theta/(sigma sqrt(2)))/2
        cut_off = float(
            erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * sigma
        )
        automatic_cut_off=cut_off
        cut_off*=threshold_factor
    elif cut_off < 0:
        cut_off = 0.0
    logger.info(f"cut off for particle extraction: {cut_off}")

    cut_box = int(particle_radius_px) * 2 + 1
    cut_mask = (spherical_mask(cut_box, particle_radius_px, cut_box // 2) == 0) * 1

    dims = np.array(jobs[0]["tomo_shape"], dtype=float)

    data = []
    scores_out = []
    for _ in range(n_particles):
        ind = np.unravel_index(int(np.argmax(score_volume)), score_volume.shape)
        lcc_max = float(score_volume[ind])
        if lcc_max <= cut_off or np.isnan(lcc_max):
            break
        scores_out.append(lcc_max)

        rotation = convert_euler(
            [-1 * a for a in angle_list[int(angle_volume[ind])]],
            order_in="ZXZ",
            order_out="ZYZ",
            degrees_in=False,
            degrees_out=True,
        )

        centered = [(i - d / 2) * voxel_size for i, d in zip(ind, dims)]

        data.append(
            (
                centered[0],
                centered[1],
                centered[2],
                rotation[0],
                rotation[1],
                rotation[2],
                lcc_max,
                cut_off,
                sigma,
                voxel_size,
                tomo_id,
            )
        )

        start = [i - particle_radius_px for i in ind]
        region = (
            slice(start[0], start[0] + cut_box),
            slice(start[1], start[1] + cut_box),
            slice(start[2], start[2] + cut_box),
        )
        score_volume[region] *= cut_mask

    info = {"cut_off": cut_off, "sigma": sigma, "search_space": search_space,
            "automatic_cut_off": automatic_cut_off,"threshold_factor":threshold_factor if automatic_cut_off is not None else 1.0}
    return data, scores_out, info


def extract_and_write(
    output_dir: pathlib.Path,
    n_particles: int = 1000,
    particle_diameter: float | None = None,
    cut_off: float | None = None,
    n_false_positives: float = 1.0,
    star_path: pathlib.Path | None = None,
    threshold_factor: float = 1.0,
):
    """Extract peaks and write the canonical Greco picking STAR table."""
    output_dir = pathlib.Path(output_dir)
    data, scores_out, info = extract_particles(
        output_dir, n_particles=n_particles,
        particle_diameter=particle_diameter, cut_off=cut_off,
        n_false_positives=n_false_positives,
        threshold_factor=threshold_factor,
    )
    star_path = star_path or storage_path(output_dir, "particles.star")
    write_star(star_path, PICK_COLUMNS, data)
    logger.info(f"Wrote {len(data)} particles to {star_path}")

    statistics={"n_particles":len(data),"edge_rule":EDGE_RULE,**info}
    return star_path,len(data),statistics
