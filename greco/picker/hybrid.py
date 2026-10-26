"""Greco Picker-2 hybrid pipeline (frozen research implementation, port).

The validated pipeline from the greco-picker-r3-math research rounds
(6B/6D/6F/6H), ported 1:1 into production so that
``greco-picker match --hybrid fps`` reproduces the research output:

  1. SCREEN   - the production rotation loop over an FPS orientation
                subset (``MatchJob`` with ``orientation_ids``);
  2. CANDIDATES - local maxima of the corrected screen map (full
                fast-shape work frame), top C;
  3. HILL CLIMB - full-bank spatial refinement: step 0 exact full-bank
                score at every candidate, then up to 3 steps of 3x3x3
                (27 incl. self) full-bank ``max_bank`` evaluations,
                re-centring only candidates that moved;
  4. MU       - control variate mu = mu_s + E[N_full - N_screen] with
                4 independently scrambled Sobol sets (default K=4096
                per set, SE = between-seed std / 2); optional adaptive
                escalation (4096 -> 8192 -> 16384) when a final score
                sits within c*SE of the cut;
  5. CUTOFF   - autonomous: erfcinv(2*n_fp/(ROI*M_all))*sqrt(2)*sigma_own,
                M_all = FULL bank size (15552), sigma_own = the screen's
                own twin-RMS statistic.  No exhaustive quantity enters;
  6. EXTRACT  - ONE canonical edge rule (``edge_keep``, applied before
                peak search) + the production greedy NMS at the cut.

Every numeric primitive (patch gather, local SPR, bank GEMM, chunked
GPU ``max_bank``, (nt+2)-cube envelope, Sobol estimator, hill-climb
trajectory) is a line-for-line port of the research code.  The
exhaustive path is untouched.
"""

from __future__ import annotations
from greco.project_layout import join as storage_path

import time

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.special import erfcinv
from scipy.stats import qmc

from .extract import (
    EDGE_RULE,
    edge_keep,
    edge_radius_vox,
    greedy_nms_sparse,
)
from .geometry import rotate_batch_gpu, spline_prefilter, transform_matrix
from .noise_model import SymmetricNoiseBank,half_range_max
from .io import PICK_COLUMNS, read_mrc, write_mrc, write_star
from .weights import (
    create_gaussian_band_pass,
    create_wedge,
    phase_randomize_template,
)

__all__ = [
    "run_hybrid_refine",
    "autonomous_cut",
    "climb_wall_total",
    "hill_climb",
    "sobol_mu",
    "adaptive_mu",
    "local_maxima_pos",
    "pre_from_out",
]

# 3x3x3 offsets (27 incl. self), row-major candidate-major ordering;
# index 13 is the self offset
OFF27 = np.stack(
    np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1
).reshape(-1, 3)

# scrambled-Sobol control-variate seeds (frozen, research s22/s23)
SOBOL_SEEDS = (69, 70, 71, 72)
CV_MU_KMAX = 16384
DEFAULT_MU_ESC_CONS = 3.0


# ---------------------------------------------------------------------------
# Frame conventions and local score primitives (port of round6b/oracle)
# ---------------------------------------------------------------------------

def pre_from_out(out_idx, shape):
    """pre[i] = flip(roll(out, -shift))[i]: inverse of the production
    finalize (output frame -> pre-flip STOPGAP work frame)."""
    pre = []
    for ax, (i, n) in enumerate(zip(out_idx, shape)):
        sh = int(np.floor(n / 2)) + 1
        pre.append((n - 1 - ((i - sh) % n)) % n)
    return tuple(pre)


def patch_gather(pos, v_f_flat, fs_c, nt):
    """(n,3) work-frame positions -> (n, nt^3) GPU patches."""
    from .patch_gather import periodic_patches
    return periodic_patches(pos,v_f_flat,fs_c,nt).reshape(len(pos),int(nt)**3)


def local_spr(P, m_flat, mw):
    """(n, nt^3) patches -> (n,) masked SPR (double-precision variance),
    the exact per-voxel normalisation of the production score maps."""
    b = P @ m_flat
    c2 = (P.double() ** 2) @ m_flat.double()
    var = torch.clamp(c2 / mw - (b.double() / mw) ** 2, min=1e-18)
    return torch.sqrt(var).float() * mw


def bank_gemm(bank, pos, v_f_flat, fs_c, nt, m_flat, mw, chunk=4096):
    """Full-bank spec (n, M) at work-frame positions, chunked."""
    if isinstance(bank,(tuple,SymmetricNoiseBank)):
        # Each independent twin is maximized over orientations BEFORE
        # averaging. A max of averaged templates would be a different null.
        P=patch_gather(pos,v_f_flat,fs_c,nt)
        values,_=max_bank(P,bank,m_flat,mw)
        del P
        return values[:,None]
    spec = np.empty((len(pos), bank.shape[0]), dtype=np.float32)
    for i0 in range(0, len(pos), chunk):
        sl = slice(i0, min(i0 + chunk, len(pos)))
        P = patch_gather(pos[sl], v_f_flat, fs_c, nt)
        spr = local_spr(P, m_flat, mw)
        spec[sl] = ((bank @ P.T) / spr[None, :]).T.cpu().numpy()
    torch.cuda.synchronize(v_f_flat.device)
    return spec


def max_bank(P, bank, m_flat, mw, chunk_p=2048, chunk_a=2048):
    """GPU-only (max score, argmax id) per patch row over the whole bank,
    with the running max/argmax kept on the GPU (port of round6d)."""
    if isinstance(bank,SymmetricNoiseBank):
        return half_range_max(P,bank,m_flat,mw,chunk_p,chunk_a)
    if isinstance(bank,tuple):
        values=[max_bank(P,b,m_flat,mw,chunk_p,chunk_a)[0] for b in bank]
        return np.mean(values,axis=0),np.zeros(len(P),dtype=np.int64)
    n = P.shape[0]
    dev = P.device
    best = torch.full((n,), float("-inf"), device=dev)
    bid = torch.zeros(n, dtype=torch.int64, device=dev)
    spr = local_spr(P, m_flat, mw)
    M = bank.shape[0]
    for a0 in range(0, M, chunk_a):
        B = bank[a0:a0 + chunk_a]
        for p0 in range(0, n, chunk_p):
            p1 = min(p0 + chunk_p, n)
            sc = (B @ P[p0:p1].T) / spr[p0:p1][None, :]
            mb, mi = sc.max(dim=0)
            better = mb > best[p0:p1]
            best[p0:p1] = torch.where(better, mb, best[p0:p1])
            bid[p0:p1] = torch.where(better, (a0 + mi).to(torch.int64),
                                     bid[p0:p1])
    return best.cpu().numpy(), bid.cpu().numpy()


def envelope_gather(p, v_f_flat, fs_c, nt):
    """(nt+2)^3 periodic envelopes (1-voxel padding) for the 27-patch
    3x3x3 shell around each position (port of round6d.envelope_gather,
    where the research constant 38 = N+2 with N = 36)."""
    from .patch_gather import periodic_patches
    return periodic_patches(p,v_f_flat,fs_c,nt,padding=1)


def build_patches(env, nt):
    """(nt+2)^3 envelopes -> (n*27, nt^3) GPU patch matrix (row-major)."""
    P = torch.cat([
        env[:, o[0] + 1:o[0] + 1 + int(nt), o[1] + 1:o[1] + 1 + int(nt),
            o[2] + 1:o[2] + 1 + int(nt)].reshape(env.shape[0], -1)
        for o in OFF27], 1).reshape(-1, int(nt) ** 3)
    return P


def step_neighbours(pos_sel, bank, tbank, v_f_flat, fs_c, nt, m_flat, mw,
                    Bsel=400):
    """Full-bank 3x3x3 (27 incl. self) score for each position, GPU-only
    max_bank.  Returns (S, N, id) candidate-major (n*27 rows) + wall."""
    n = len(pos_sel)
    S = np.empty(n * 27)
    N = np.empty(n * 27)
    id_ = np.empty(n * 27, dtype=np.int64)
    dev = v_f_flat.device
    t0 = time.time()
    for s0 in range(0, n, Bsel):
        s1 = min(s0 + Bsel, n)
        env = envelope_gather(pos_sel[s0:s1], v_f_flat, fs_c, nt)
        P = build_patches(env, nt)
        del env
        Sx, ix = max_bank(P, bank, m_flat, mw)
        Nx, _ = max_bank(P, tbank, m_flat, mw)
        del P
        S[s0 * 27:s1 * 27] = Sx
        N[s0 * 27:s1 * 27] = Nx
        id_[s0 * 27:s1 * 27] = ix
    torch.cuda.synchronize(dev)
    wall = time.time() - t0
    peak = float(torch.cuda.max_memory_allocated(dev)) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(dev)
    return S, N, id_, wall, peak


# ---------------------------------------------------------------------------
# Full-bank orientation bank (port of final_bench._rotate_bank)
# ---------------------------------------------------------------------------

def build_orientation_bank(template_np, mask_np, wedge_t, angle_list, dev,
                           batch=63, *, matrices=None):
    """Rotate + wedge-filter + mask-normalise a template over ALL
    orientations of ``angle_list`` (batched production rotation path).
    Returns (M, nt^3) GPU fp32 rows.  Identical physics to the screen
    loop's per-angle normalised rotated template (``tp``)."""
    nt = template_np.shape[0]
    if matrices is None:
        center = np.divide(np.subtract((nt,) * 3, 1), 2, dtype=np.float64)
        mats = [
            transform_matrix(rotation=tuple(float(x) for x in a),
                             rotation_units="rad", rotation_order="rzxz",
                             center=center)
            for a in angle_list
        ]
        lin_all = torch.from_numpy(np.stack(
            [m[:3, :3].astype(np.float32) for m in mats])).to(dev)
        off_all = torch.from_numpy(np.stack(
            [m[:3, 3].astype(np.float32) for m in mats])).to(dev)
        del mats
    else:
        lin_all,off_all=matrices
        expected=len(angle_list)
        if (lin_all.shape!=(expected,3,3) or off_all.shape!=(expected,3)
                or lin_all.dtype!=torch.float32 or off_all.dtype!=torch.float32
                or lin_all.device!=torch.device(dev) or off_all.device!=torch.device(dev)):
            raise ValueError('Invalid orientation matrices for template bank.')
    m_t = torch.from_numpy(np.ascontiguousarray(mask_np)).to(dev)
    mw = float(m_t.sum())
    coeffs = torch.from_numpy(spline_prefilter(template_np)).to(dev)
    M = len(angle_list)
    rows = torch.empty((M, nt ** 3), dtype=torch.float32, device=dev)
    for b0 in range(0, M, batch):
        b = min(batch, M - b0)
        cs = coeffs.unsqueeze(0).expand(b, nt, nt, nt).contiguous()
        rot = rotate_batch_gpu(
            cs.unsqueeze(1), None,
            matrices=(lin_all[b0:b0 + b], off_all[b0:b0 + b]),
        )[:, 0]
        ft = torch.fft.rfftn(rot, dim=(-3, -2, -1)) * wedge_t
        rot2 = torch.fft.irfftn(ft, s=(nt,) * 3, dim=(-3, -2, -1))
        mean = (rot2 * m_t).sum(dim=(-3, -2, -1)) / mw
        var = ((rot2 ** 2) * m_t).sum(dim=(-3, -2, -1)) / mw
        stdr = torch.sqrt(var - mean ** 2).clamp_min(1e-12)
        rows[b0:b0 + b] = (
            ((rot2 - mean[..., None, None, None])
             / stdr[..., None, None, None]) * m_t
        ).reshape(b, -1)
        del cs, rot, ft, rot2
    torch.cuda.synchronize(dev)
    return rows


# ---------------------------------------------------------------------------
# Hill-climb candidate refinement
# ---------------------------------------------------------------------------

def climb_wall_total(step_walls):
    """Total elapsed time across all hill-climb steps."""
    return sum(step_walls)


def hill_climb(pos, C, bank, tbank, v_f_flat, fs_c, nt, m_flat, mw,
               max_steps=3):
    """Full-bank spatial hill climb over ALL C candidates.

    Step 0: exact full-bank score at every candidate centre.
    Step t (1..max): full-bank 3x3x3 for every ACTIVE candidate; a
    candidate moves to the best of the 26 non-self neighbours iff it
    strictly improves (the additive mu cancels in spatial comparisons);
    only movers stay active.  Every scored position (centres + every
    step's 27 rows of the active set) is carried to extraction.
    """
    fs = np.asarray(fs_c)
    t0 = time.time()
    spec_Sc = bank_gemm(bank, pos[:C], v_f_flat, fs_c, nt, m_flat, mw)
    spec_Nc = bank_gemm(tbank, pos[:C], v_f_flat, fs_c, nt, m_flat, mw)
    S_c = spec_Sc[np.arange(C), spec_Sc.argmax(1)]
    N_c = spec_Nc.max(axis=1)
    id_c = spec_Sc.argmax(1)
    del spec_Sc, spec_Nc
    wall0 = time.time() - t0

    cur = pos[:C].copy()
    cur_sc = S_c - N_c
    cur_id = id_c.copy()
    all_pos = [pos[:C].copy()]
    all_S = [S_c.copy()]
    all_N = [N_c.copy()]
    all_id = [id_c.copy()]
    active = np.ones(C, dtype=bool)
    stats = [{"step": 0, "n_eval": int(C), "n_moved": 0,
              "wall_s": round(wall0, 2)}]
    for step in range(1, max_steps + 1):
        n_act = int(active.sum())
        if n_act == 0:
            break
        t0 = time.time()
        cur_before = cur[active].copy()
        S, N, id_, wall, peak = step_neighbours(
            cur_before, bank, tbank, v_f_flat, fs_c, nt, m_flat, mw
        )
        sc = (S - N).reshape(n_act, 27)
        # compare against the 26 NON-self neighbours only: the self row
        # is the current centre re-evaluated in a different GEMM layout,
        # so its fp value differs by ~1e-6 and would create phantom moves
        sc26 = sc.copy()
        sc26[:, 13] = float("-inf")
        boff = sc26.argmax(1)
        bsc = sc26[np.arange(n_act), boff]
        bid = id_.reshape(n_act, 27)[np.arange(n_act), boff]
        base = cur_sc[active]
        moved = bsc > base
        idx = np.where(active)[0]
        m = idx[moved]
        cur[m] = (cur_before[moved] + OFF27[boff[moved]]) % fs
        cur_sc[m] = bsc[moved]
        cur_id[m] = bid[moved]
        active[:] = False
        active[m] = True
        # scored rows are ALL 27 of the ACTIVE set (not just movers)
        nb_pos_all = ((cur_before[:, None, :] + OFF27[None, :, :]) % fs
                      ).reshape(-1, 3)
        all_pos.append(nb_pos_all)
        all_S.append(S.copy())
        all_N.append(N.copy())
        all_id.append(id_.copy())
        wall = time.time() - t0
        stats.append({"step": step, "n_eval": n_act,
                      "n_moved": int(moved.sum()),
                      "wall_s": round(wall, 2),
                      "peak_mem_gb": round(peak, 1)})
    out = {
        "stats": stats,
        # the total is the SUM of all step walls (t0 is reset at the top
        # of each loop iteration; subtracting a single t0 from the end
        # would report only the last step - the old research bug)
        "wall_steps_total_s": round(
            climb_wall_total([s["wall_s"] for s in stats]), 2),
        "all_pos": np.concatenate(all_pos),
        "all_S": np.concatenate(all_S),
        "all_N": np.concatenate(all_N),
        "all_id": np.concatenate(all_id),
        "final_pos": cur, "final_sc": cur_sc, "final_id": cur_id,
        "unresolved": active,
    }
    return out


# ---------------------------------------------------------------------------
# Scrambled-Sobol control-variate mu (port of adaptive_mu.py, est "D")
# ---------------------------------------------------------------------------

def _noise_delta(pk, noise2, tbank, v_f_flat, fs_c, nt, m_flat, mw):
    Pk = patch_gather(pk, v_f_flat, fs_c, nt)
    Nf, _ = max_bank(Pk, tbank, m_flat, mw)
    Ns = noise2[pk[:, 0], pk[:, 1], pk[:, 2]].cpu().numpy()
    del Pk
    return (np.asarray(Nf) - Ns).astype(np.float64)


def sobol_mu(noise2, mu_s, tbank, v_f_flat, fs_c, nt, m_flat, mw, K=4096):
    """mu = mu_s + mean over 4 independently scrambled Sobol sets of
    (1/K) sum_x [N_full(x) - N_screen(x)].  SE = between-seed std / 2.
    Deterministic for the fixed seeds (frozen, research s22).

    NOTE (frozen-implementation fidelity): the research CV indexes the
    SCREEN NOISE MAP in the OUTPUT frame with the work-frame Sobol
    coordinates.  The flip/roll is a bijection of the full periodic
    fast-shape voxel set, so this is statistically the same uniform
    sampling of E[N_screen] - but the specific samples (and thus the
    deterministic mu value) differ from a work-frame indexing, and the
    production port must reproduce the frozen research mu bit-for-bit.
    ``noise2`` must therefore be the output-frame screen noise map."""
    hi = np.asarray(fs_c, dtype=int)
    ests = []
    for s in SOBOL_SEEDS:
        u = qmc.Sobol(d=3, scramble=True, seed=s).random(K)
        pk = np.clip((u * np.asarray(fs_c, dtype=float)).astype(int),
                     0, hi - 1)
        ests.append(float(mu_s + _noise_delta(pk, noise2, tbank, v_f_flat,
                                              fs_c, nt, m_flat, mw).mean()))
    mu = float(np.mean(ests))
    se = float(np.std(ests, ddof=1) / 2.0)
    return mu, se, {str(s): round(v, 9) for s, v in zip(SOBOL_SEEDS, ests)}


def _d_min_of(P, zt, cut, picks, se, c):
    """d_min = min |z_i - cut| over NMS-relevant candidates: the picks,
    plus scored positions within 10*SE below the cut (the only ones
    that could flip classification)."""
    band = 10 * se
    near = (zt < cut) & (zt > cut - band)
    pick_mask = np.zeros(len(zt), dtype=bool)
    pick_mask[np.asarray(picks, dtype=int)] = True
    rel = np.where(near | pick_mask)[0]
    if not len(rel):
        return float("inf")
    return float(np.min(np.abs(zt[rel] - cut)))


def adaptive_mu(noise2, mu_s, tbank, v_f_flat, fs_c, nt, m_flat, mw,
                cut, P, Z, base_K=4096, Kmax=CV_MU_KMAX,
                c=DEFAULT_MU_ESC_CONS, r=9, n_max=1000):
    """Stage 1 + stage 2 escalation loop (frozen research s23): estimate
    mu at K, classify the scored set; if the nearest relevant score sits
    within c*SE of the cut, double K (4096 -> 8192 -> 16384) until the
    pick set is stable or Kmax is reached.  The chosen K and the full
    per-level history are recorded for audit."""
    history = []
    prev_picks = None
    n_esc = 0
    K = base_K
    while True:
        mu, se, per_seed = sobol_mu(noise2, mu_s, tbank, v_f_flat, fs_c,
                                    nt, m_flat, mw, K=K)
        zt = Z + mu
        ph = greedy_nms_sparse(P, zt, cut, n_max, r)
        d_min = _d_min_of(P, zt, cut, ph, se, c)
        picks_sig = tuple(sorted(P[ph].tolist()))
        changed = None if prev_picks is None else picks_sig != prev_picks
        history.append({"K": int(K), "mu": round(mu, 9),
                        "SE": round(se, 9), "d_min": round(d_min, 9),
                        "n_picks": int(len(ph)),
                        "picks_changed": changed,
                        "per_seed": per_seed})
        accepted = (d_min > c * se) or (K >= Kmax) or\
            (changed is False and d_min > c * se / 2)
        if accepted or K >= Kmax:
            return {"mu": mu, "SE": se, "K": int(K), "d_min": d_min,
                    "c": c, "n_escalations": n_esc,
                    "stable": bool(d_min > c * se),
                    "accepted": True,
                    "boundary_at_kmax": bool(K >= Kmax
                                             and d_min <= c * se),
                    "history": history, "picks": ph.tolist()}
        prev_picks = picks_sig
        n_esc += 1
        K *= 2


# ---------------------------------------------------------------------------
# Candidates + cutoff (port of pyramid_eval.local_maxima / s20 cutoff)
# ---------------------------------------------------------------------------

def local_maxima_pos(S, cap=8000):
    """Local maxima of a score volume (3x3x3, boundary layer excluded),
    sorted by score (stable), capped.  Port of the research screen's
    candidate generator."""
    mx = maximum_filter(S, size=3, mode="nearest")
    pk = (S == mx)
    inner = np.zeros_like(pk)
    inner[1:-1, 1:-1, 1:-1] = pk[1:-1, 1:-1, 1:-1]
    idx = np.argwhere(inner)
    vals = S[idx[:, 0], idx[:, 1], idx[:, 2]]
    order = np.argsort(-vals, kind="stable")
    k = min(len(idx), cap)
    return idx[order[:k]], vals[order[:k]], int(len(idx))


def autonomous_cut(sigma_own, roi, m_all, n_fp=1.0):
    """Self-calibrated cutoff (Rickgauer 2017 rule) with the run's OWN
    sigma and the FULL-bank search space ROI x M_all.  Hybrid autonomy
    (research s20): M_all is the full orientation bank (15552 at 10 deg,
    symmetry 1) even though the screen only ran a subset; no
    exhaustive-run quantity may enter this expression."""
    if m_all <= 0 or roi <= 0 or sigma_own <= 0:
        raise ValueError("autonomous_cut needs positive roi/m_all/sigma")
    return float(erfcinv(2.0 * n_fp / (roi * m_all)) * np.sqrt(2)
                 * sigma_own)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_hybrid_refine(job, corrected_full, stats, wall_screen, *, orientation_matrices=None):
    """Screen -> candidates -> full-bank hill climb -> Sobol mu ->
    autonomous cut -> canonical edge + greedy NMS -> STAR + job section.

    ``job`` is the MatchJob that just finished its screen run
    (``job.hybrid_mode`` is True); ``corrected_full`` is the full
    fast-shape corrected screen map (output frame, GPU); ``stats`` is
    the screen's stats dict (sigma_own = stats['std'], roi, noise_mean).
    Returns the ``hybrid`` section of the job JSON; ``["outputs"]``
    holds the new output paths (the caller merges them)."""
    cfg = job.hybrid_cfg
    dev = torch.device(f"cuda:{job.gpu_id}")
    torch.cuda.set_device(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t_refine0 = time.time()

    fs_c = tuple(job.fast_shape)
    tomo_shape = tuple(job.tomo_shape)
    template_np = read_mrc(job.template).astype(np.float32)
    mask_np = read_mrc(job.mask).astype(np.float32)
    nt = template_np.shape[0]

    v_f_flat = job._hyb_vf.reshape(-1)
    m_flat = torch.from_numpy(np.ascontiguousarray(
        mask_np.ravel())).to(dev)
    mw = float(m_flat.sum())

    n_candidates = int(cfg.get("n_candidates", 2000))
    max_climb_steps = int(cfg.get("max_climb_steps", 3))
    mu_samples = int(cfg.get("mu_samples", 4096))
    mu_adaptive = bool(cfg.get("mu_adaptive", False))
    mu_esc_cons = float(cfg.get("mu_esc_cons", DEFAULT_MU_ESC_CONS))
    n_fp = float(cfg.get("n_false_positives", 1.0))
    threshold_factor=float(cfg.get('threshold_factor',1.0))
    if not np.isfinite(threshold_factor) or threshold_factor<=0:
        raise ValueError('Threshold factor must be finite and positive.')
    n_max = int(cfg.get("n_particles", 1000))
    diameter = float(cfg.get("particle_diameter", 300.0))

    # ---------------- full-bank orientation banks (one-time) -------------
    t0 = time.time()
    # byte-identical to the screen loop's wedge_template_t: float32
    # fields multiplied in float32 (research W36*BP36 convention)
    wedge = create_wedge(
        template_np.shape, job.ts_metadata, job.voxel_size,
        cut_off_radius=1.0,
    ).astype(np.float32)
    if not (job.low_pass is None and job.high_pass is None):
        wedge *= create_gaussian_band_pass(
            template_np.shape, job.voxel_size, job.low_pass, job.high_pass
        ).astype(np.float32)
    wedge_t = torch.from_numpy(wedge).to(dev)
    bank = build_orientation_bank(
        template_np, mask_np, wedge_t, job.angle_list, dev,matrices=orientation_matrices
    )
    twin_np = phase_randomize_template(
        template_np, mask_np, seed=job.rng_seed,n_iter=job.noise_template_iterations
    )
    tbank = build_orientation_bank(
        twin_np, mask_np, wedge_t, job.angle_list, dev,matrices=orientation_matrices
    )
    noise_seeds=cfg.get('noise_ensemble_seeds')
    if noise_seeds is not None:
        if not isinstance(noise_seeds,list) or not 2<=len(noise_seeds)<=8 or len(set(noise_seeds))!=len(noise_seeds) or any(isinstance(s,bool) or not isinstance(s,int) or s<0 for s in noise_seeds):
            raise ValueError('Noise ensemble needs 2–8 distinct nonnegative integer seeds')
        banks=[]
        for seed in noise_seeds:
            if seed==job.rng_seed:banks.append(tbank)
            else:
                null=phase_randomize_template(template_np,mask_np,seed=seed,n_iter=job.noise_template_iterations)
                banks.append(build_orientation_bank(null,mask_np,wedge_t,job.angle_list,dev,matrices=orientation_matrices))
        tbank=tuple(banks)
    if cfg.get('symmetric_noise',False):
        tbank=tuple(SymmetricNoiseBank(b) for b in tbank) if isinstance(tbank,tuple) else SymmetricNoiseBank(tbank)
    del wedge, wedge_t, twin_np, template_np, mask_np
    wall_bank = time.time() - t0
    bank_peak = float(torch.cuda.max_memory_allocated(dev)) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(dev)

    # ---------------- candidates: local maxima of the corrected screen ----
    t0 = time.time()
    candidate_source=cfg.get('candidate_source','corrected')
    if candidate_source not in ('corrected','signal'):
        raise ValueError('Unknown hybrid candidate source')
    if candidate_source=='signal':
        corrected_full=corrected_full+job._hyb_noise_o-stats['noise_mean']
    corrected_np = corrected_full.cpu().numpy()
    del corrected_full
    kp, vals, n_local_max = local_maxima_pos(corrected_np)
    del corrected_np, vals
    pos = np.ascontiguousarray(kp, dtype=np.int64)   # work frame (f=1)
    if cfg.get('candidate_geometric_support',False):
        from .candidate_pool import spaced_candidate_indices
        # Reject ineligible centres BEFORE spending the full-bank budget.
        valid=edge_keep(pos,tomo_shape,edge_radius_vox(diameter,job.voxel_size))
        pos=pos[valid]
        radius=float(cfg.get('candidate_nms_radius',0.))
        pos=pos[spaced_candidate_indices(pos,n_candidates,radius)]
    C = min(n_candidates, len(pos))
    wall_cand = time.time() - t0

    # ---------------- full-bank spatial hill climb ------------------------
    hc = hill_climb(pos, C, bank, tbank, v_f_flat, fs_c, nt, m_flat, mw,
                    max_steps=max_climb_steps)
    del pos, kp
    all_pos = hc["all_pos"]
    all_id = hc["all_id"]
    z = hc["all_S"] - hc["all_N"]
    n_unresolved = int(hc["unresolved"].sum())
    wall_climb = hc["wall_steps_total_s"]
    climb_stats = hc["stats"]
    del hc

    # ---------------- autonomous cutoff (own sigma, FULL bank) ------------
    roi = int(stats["roi"])
    m_all = int(job.n_rotations)
    sigma_own = float(stats["std"])
    cut = autonomous_cut(sigma_own, roi, m_all, n_fp)
    automatic_cut_off=cut
    cut*=threshold_factor
    r_edge = edge_radius_vox(diameter, job.voxel_size)
    # canonical edge rule, applied BEFORE peak search, in the work frame
    # (the padded fast shape - the frame the scored positions live in)
    keep = edge_keep(all_pos, fs_c, r_edge)
    P, Z = all_pos[keep], z[keep]

    # ---------------- control-variate mu (scrambled Sobol) ----------------
    # research fidelity: the OUTPUT-frame screen noise map feeds the
    # control variate (see sobol_mu docstring)
    noise2 = job._hyb_noise_o
    t0 = time.time()
    if mu_adaptive:
        am = adaptive_mu(
            noise2, stats["noise_mean"], tbank, v_f_flat,
            fs_c, nt, m_flat, mw, cut, P, Z,
            base_K=mu_samples, c=mu_esc_cons, r=r_edge, n_max=n_max,
        )
        mu, se = am["mu"], am["SE"]
        mu_info = {k: v for k, v in am.items() if k != "picks"}
        mu_info["boundary_sensitive"] =\
            bool(am["d_min"] <= am["c"] * am["SE"])
    else:
        mu, se, per_seed = sobol_mu(
            noise2, stats["noise_mean"], tbank, v_f_flat,
            fs_c, nt, m_flat, mw, K=mu_samples
        )
        # d_min / boundary_sensitive are completed below with the final
        # NMS picks (always reported, per the production contract)
        mu_info = {
            "mu": round(mu, 9), "SE": round(se, 9), "K": int(mu_samples),
            "n_escalations": 0, "c": mu_esc_cons,
            "per_seed": per_seed,
        }
    wall_mu = time.time() - t0
    del job._hyb_noise_o, noise2

    # ---------------- refined scores + greedy NMS at the cut --------------
    t0 = time.time()
    sc = Z + mu                                  # keep-filtered scores
    ph = greedy_nms_sparse(P, sc, cut, n_max, r_edge)
    wall_nms = time.time() - t0
    if not mu_adaptive:
        d_min = _d_min_of(P, sc, cut, ph, se, mu_esc_cons)
        mu_info["d_min"] = round(d_min, 9)
        mu_info["boundary_sensitive"] =\
            bool(d_min <= mu_esc_cons * se)
        mu_info["stable"] = bool(d_min > mu_esc_cons * se)
    pk = P[ph]
    psc = sc[ph]
    pid = all_id[keep][ph]

    # ---------------- outputs: refined map, STAR, summary -----------------
    t0 = time.time()
    job.output_dir.mkdir(parents=True, exist_ok=True)
    # visualization map: refined scores at every scored position (best
    # per voxel), -inf elsewhere
    sc_all = (z + mu).astype(np.float32)
    refined = np.full(fs_c, float("-inf"), dtype=np.float32)
    np.maximum.at(refined, (all_pos[:, 0], all_pos[:, 1], all_pos[:, 2]),
                  sc_all)
    del all_pos, all_id, z, P, Z, sc, sc_all
    refined_path = storage_path(job.output_dir, f"{job.tomo_id}_refined.mrc")
    write_mrc(refined_path, refined, job.voxel_size)
    del refined

    dims = np.array(tomo_shape, dtype=float)
    centered = (pk.astype(float) - dims / 2.0) * job.voxel_size
    rows = []
    if len(pk):
        # the SAME conversion the production extractor uses, so the
        # STAR angles are computed identically in both modes
        from .geometry import convert_euler

        zyz = np.array([
            convert_euler([-1.0 * float(a) for a in
                           job.angle_list[int(j)]],
                          order_in="ZXZ", order_out="ZYZ",
                          degrees_in=False, degrees_out=True)
            for j in pid
        ])
    for i in range(len(pk)):
        rows.append((
            float(centered[i, 0]), float(centered[i, 1]),
            float(centered[i, 2]),
            float(zyz[i, 0]), float(zyz[i, 1]), float(zyz[i, 2]),
            float(psc[i]), float(cut), float(sigma_own),
            float(job.voxel_size), job.tomo_id,
        ))
    star_path = storage_path(job.output_dir, "particles.star")
    write_star(star_path, PICK_COLUMNS, rows)

    picking_statistics={
        "mode":"hybrid","n_particles":int(len(pk)),"cut_off":float(cut),
        "automatic_cut_off":float(automatic_cut_off),"threshold_factor":threshold_factor,
        "sigma":float(sigma_own),"search_space":int(roi*m_all),"edge_rule":EDGE_RULE,
    }
    wall_star = time.time() - t0

    refine_total = time.time() - t_refine0
    peak_mem_gb = max(
        [bank_peak] + [s.get("peak_mem_gb", 0.0) for s in climb_stats]
    )
    hybrid_info = {
        "orientation_subset": "fps",
        "fps": dict(cfg.get("fps", {})),
        "n_screen_rotations": int(len(job.angle_ids)),
        "n_full_bank": int(m_all),
        "sigma_own": float(sigma_own),
        "mu_s": round(float(stats["noise_mean"]), 9),
        "mu": round(float(mu), 9),
        "mu_info": mu_info,
        "mu_adaptive": mu_adaptive,
        "noise_ensemble_seeds": noise_seeds,
        "symmetric_noise": bool(cfg.get('symmetric_noise',False)),
        "candidate_source": candidate_source,
        "candidate_geometric_support": bool(cfg.get('candidate_geometric_support',False)),
        "candidate_nms_radius": float(cfg.get('candidate_nms_radius',0.)),
        "cut": round(float(cut), 9),
        "n_false_positives": n_fp,
        "roi": int(roi),
        "search_space_full_bank": int(roi * m_all),
        "n_local_max": int(n_local_max),
        "n_candidates": int(C),
        "climb": {
            "steps": climb_stats,
            "wall_total_s": wall_climb,
            "n_unresolved_after_climb": n_unresolved,
        },
        "edge_rule": EDGE_RULE,
        "edge_r": int(r_edge),
        "particle_diameter": diameter,
        "wall": {
            "screen_s": round(wall_screen, 2),
            "bank_s": round(wall_bank, 2),
            "candidates_s": round(wall_cand, 2),
            "climb_s": round(wall_climb, 2),
            "mu_s": round(wall_mu, 2),
            "nms_s": round(wall_nms, 3),
            "star_s": round(wall_star, 3),
            # research-comparable: screen + climb + mu + NMS (the bank is
            # one-time per template/angle list and is excluded)
            "cached_bank_pipeline_s": round(
                wall_screen + wall_climb + wall_mu + wall_nms, 2),
            "refine_total_s": round(refine_total, 2),
        },
        "n_picks": int(len(pk)),
        "peak_mem_gb": round(peak_mem_gb, 1),
        "outputs": {
            "refined": str(refined_path),
            "star": str(star_path),
        },
    }
    hybrid_info["picking_statistics"]=picking_statistics
    return hybrid_info
