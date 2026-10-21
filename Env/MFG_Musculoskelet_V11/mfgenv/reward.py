"""
Reward functions for musculoskeletal simulation environment.
Provides imitation reward components and (currently stubbed) goal reward.
@author: YAKE
"""
import math
import logging
from functools import lru_cache
from typing import Any, Tuple, Dict, Optional, List

import numpy as np
from numba import njit

from .common_utils import (
    quat_to_mat,
    orient6_to_mat,
    rotation_geodesic,
    inverse_convert_ref_traj_qpos,
    inverse_convert_ref_traj_qvel
    )

# Configure module-level logger
logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

_EPSILON = 1e-8
_JERK_COST_SCALE = 5e-2

def _np_idx(values: List[int]) -> np.ndarray:
    return np.array(values, dtype=np.int64)

_SYNERGY_GROUPS = {
    "hip_flex_r": _np_idx([29]),
    "hip_ext_r": _np_idx([6, 31, 32]),
    "knee_flex_r": _np_idx([6, 12, 13, 31, 32]),
    "knee_ext_r": _np_idx([29, 37, 38, 39]),
    "ankle_df_r": _np_idx([35]),
    "ankle_pf_r": _np_idx([12, 13, 33]),
    "hip_flex_l": _np_idx([69]),
    "hip_ext_l": _np_idx([46, 71, 72]),
    "knee_flex_l": _np_idx([46,52, 53,71, 72]),
    "knee_ext_l": _np_idx([69, 77, 78, 79]),
    "ankle_df_l": _np_idx([75]),
    "ankle_pf_l": _np_idx([52, 53, 73]),
}
# _SYNERGY_GROUPS = {
#     "hip_flex_r": _np_idx([0, 1, 20, 23, 24, 28, 29, 30, 34]),
#     "hip_ext_r": _np_idx([1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 22, 31, 32]),
#     "knee_flex_r": _np_idx([6, 7, 12, 13, 23, 30, 31, 32]),
#     "knee_ext_r": _np_idx([29, 37, 38, 39]),
#     "ankle_df_r": _np_idx([8, 9, 35]),
#     "ankle_pf_r": _np_idx([10, 11, 12, 13, 25, 26, 33, 36]),
#     "hip_flex_l": _np_idx([40, 41, 60, 63, 64, 68, 69, 70, 74]),
#     "hip_ext_l": _np_idx([41, 42, 43, 44, 45, 46, 54, 55, 56, 57, 58, 59, 62, 71, 72]),
#     "knee_flex_l": _np_idx([46, 47, 52, 53, 63, 70, 71, 72]),
#     "knee_ext_l": _np_idx([69, 77, 78, 79]),
#     "ankle_df_l": _np_idx([48, 49, 75]),
#     "ankle_pf_l": _np_idx([50, 51, 52, 53, 65, 66, 73, 76]),
# }
_ANTAGONIST_PAIRS = [
    ("hip_flex_r", "hip_ext_r"),
    ("knee_flex_r", "knee_ext_r"),
    ("ankle_df_r", "ankle_pf_r"),
    ("hip_flex_l", "hip_ext_l"),
    ("knee_flex_l", "knee_ext_l"),
    ("ankle_df_l", "ankle_pf_l"),
]

@njit
def _weighted_mse(delta: np.ndarray, w: np.ndarray, w_sum: float) -> float:
    """
    Compute a weighted mean-squared error for a 1D residual vector.

    Parameters
    ----------
    delta : np.ndarray, shape (N,)
        Residuals (e.g., y_pred - y_true).
    w : np.ndarray, shape (N,)
        Nonnegative weights corresponding to each element of `delta`.
    w_sum : float
        Precomputed sum of weights, i.e., `w.sum()`.

    Returns
    -------
    float
        Weighted MSE = (Σ_i w[i] * delta[i]^2) / w_sum.
        Returns 0.0 if `w_sum <= 0`, `w_sum` is non-finite, or `N == 0`.

    Notes
    -----
    - Designed for `numba.njit` hot path; uses a simple for-loop accumulation.
    - Caller is responsible for shape consistency between `delta` and `w`.
    """
    n = delta.shape[0]
    tot = 0.0
    if not np.isfinite(w_sum) or w_sum <= 0.0 or n == 0:
        return 0.0

    for i in range(n):
        di = delta[i]
        tot += w[i] * (di * di)

    return tot / w_sum

@njit
def _mse3d(a: np.ndarray, b: np.ndarray, w: np.ndarray, w_sum: float) -> float:
    """
    Compute a weighted mean-squared error between two (N,3) point clouds.

    Parameters
    ----------
    a : np.ndarray, shape (N,3)
        First set of 3D points.
    b : np.ndarray, shape (N,3)
        Second set of 3D points.
    w : np.ndarray, shape (N,)
        Nonnegative weights per point.
    w_sum : float
        Precomputed sum of weights, i.e., `w.sum()`.

    Returns
    -------
    float
        Weighted MSE over 3D coordinates:
          (Σ_i w[i] * ||a[i] - b[i]||^2) / w_sum.
        Returns 0.0 if shapes are incompatible, `N == 0`, or `w_sum` invalid.

    Notes
    -----
    - Flattens (N,3) arrays to stride through memory efficiently inside Numba.
    - Avoids sqrt by using squared Euclidean distance.
    """
    n = a.shape[0]
    if (b.shape[0] != n) or (a.shape[1] != 3) or (b.shape[1] != 3) or n == 0:
        return 0.0
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return 0.0
    
    af = a.ravel()
    bf = b.ravel()
    
    tot = 0.0
    base = 0
    for i in range(n):
        j0 = base
        j1 = base + 1
        j2 = base + 2
        dx = af[j0] - bf[j0]
        dy = af[j1] - bf[j1]
        dz = af[j2] - bf[j2]
        tot += w[i] * (dx * dx + dy * dy + dz * dz)
        # Test
        # tot += w[i] * (dx * dx + dy * dy + dz * dz)**0.5
        base += 3

    return tot / w_sum

@lru_cache(maxsize=None)
def _build_joint_weights(
        num_j: int, 
        decay: float | None, 
        frac: float) -> Tuple[np.ndarray, float]:
    """
    Construct per-joint weights with an optional tail decay, and return (weights, sum).

    Parameters
    ----------
    num_j : int
        Number of joints (N). If N <= 0, returns an empty array and 0.0.
    decay : float or None
        If in (0,1), the last `k = max(1, int(N * frac))` joints are assigned this
        decay weight; otherwise all weights are 1.0.
    frac : float
        Fraction of joints at the tail to apply the decay to (used only if `decay`
        is valid as above).

    Returns
    -------
    w : np.ndarray, shape (N,), float64
        Per-joint weights, ones except for the last k entries possibly set to `decay`.
    w_sum : float
        Sum of weights, i.e., `w.sum()`.

    Notes
    -----
    - Results are cached (unbounded) by arguments via `functools.lru_cache`.
    - Ensures at least one joint is decayed when `decay` is active and N > 0.
    """
    if num_j <= 0:
        return np.empty(0, dtype=np.float64), 0.0
    
    w = np.ones(num_j, dtype=np.float64)

    if (decay is not None) and (decay > 0.0) and (decay < 1.0) and (frac > 0.0):
        k = int(num_j * frac)
        if k < 1:
            k = 1
        elif k > num_j:
            k = num_j
        w[num_j - k:] = float(decay)

    return w, float(w.sum())

_EXTRA_WEIGHTS = {
    "pelvis":           5.0,
    "head_sensor":      2.0,
    "toes_l_sensor":    2.0,
    "toes_r_sensor":    2.0,
    "hand_l_sensor":    2.0,
    "hand_r_sensor":    2.0,
    }

def _build_site_weight_cache(
        env: Any,
        sim_sites_pos: Dict[str, np.ndarray],
        ref_sites_pos: Dict[str, np.ndarray],
        sim_pel: Dict[str, np.ndarray],
        ref_pel: Dict[str, np.ndarray],
        *,
        force_rebuild: bool = False,
        ) -> Tuple[List[str], np.ndarray, float]:
    """
    Build (or reuse) cached weighting and buffer structures for site-based
    imitation metrics, returning the site order, per-site weights, and weight sum.

    Parameters
    ----------
    env : Any
        Environment on which caches are stored/updated:
          - env.site_weight_overrides : dict[str, float], optional per-site multipliers.
          - Internal caches populated/used:
              _imm_site_order : list[str]
              _imm_wvec       : np.ndarray, shape (N,), float32
              _imm_wsum       : float
              _imm_bufs       : dict of float32 arrays:
                                { "pos_sim": (N,3), "pos_ref": (N,3),
                                  "vel_sim": (N,3), "vel_ref": (N,3) }
              _imm_wvec_eff   : np.ndarray, shape (N,), float32
    sim_sites_pos : Dict[str, np.ndarray]
        Simulated site positions keyed by site name. Defines which reference
        sites are eligible (intersection by name).
    ref_sites_pos : Dict[str, np.ndarray]
        Reference site positions keyed by site name. Ordering is derived from
        this dict, filtered by `sim_sites_pos`.
    sim_pel : Dict[str, np.ndarray]
        Simulated pelvis components (not directly used here; included for API symmetry).
    ref_pel : Dict[str, np.ndarray]
        Reference pelvis components (not directly used here; included for API symmetry).
    force_rebuild : bool, default=False
        If True, rebuild caches even if a compatible cached order exists.

    Returns
    -------
    order : list[str]
        Final site order, always starting with "pelvis" followed by names that
        appear in both `ref_sites_pos` and `sim_sites_pos`.
    w_vec : np.ndarray, shape (N,), float32
        Per-site weights (after applying `_EXTRA_WEIGHTS` defaults and
        `env.site_weight_overrides`, with invalid values sanitized to 1.0).
    w_sum : float
        Sum of `w_vec`. If invalid or nonpositive, falls back to `N`.

    Notes
    -----
    - Fast path: if an existing `_imm_site_order` matches the newly computed
      `order` (same list and length) and `force_rebuild` is False, the function
      returns cached `(order, _imm_wvec, _imm_wsum)` without reallocations.
    - Buffer cache `_imm_bufs` is ensured to contain four (N,3) float32 arrays
      (C-contiguous) for sim/ref positions and velocities; reallocated only when
      shape/dtype/contiguity mismatch.
    - `_imm_wvec_eff` is prepared as a float32 vector of shape (N,) for later
      per-step effective weights (not modified here).
    """
    edict = env.__dict__
    sim_keyset = set(sim_sites_pos.keys())
    order = ["pelvis"]
    order.extend(k for k in ref_sites_pos if k in sim_keyset)
    N = len(order)

    if (not force_rebuild):
        cached_order = edict.get("_imm_site_order")
        if (isinstance(cached_order, list) and len(cached_order) == N and cached_order == order):
            return edict["_imm_site_order"], edict["_imm_wvec"], edict["_imm_wsum"]

    # ---- update weighted vector ----
    overrides = getattr(env, "site_weight_overrides", None) or {}
    w_vec = edict.get("_imm_wvec")
    if (w_vec is None) or (w_vec.dtype != np.float32) or (w_vec.shape != (N,)):
        w_vec = np.empty(N, dtype=np.float32)

    wsum = 0.0
    has_bad = False
    _extra = _EXTRA_WEIGHTS
    for i, name in enumerate(order):
        base = _extra.get(name, 1.0)
        mul = overrides.get(name, 1.0)
        try:
            v = float(base) * float(mul)
        except Exception:
            v = 1.0
            has_bad = True
        if not np.isfinite(v):
            v = 1.0
            has_bad = True
        w_vec[i] = v
        wsum += v

    if has_bad or (not np.isfinite(wsum)) or (wsum <= 0.0):
        w_vec.fill(1.0)
        wsum = float(N)

    # ---- prepare cache ----
    bufs = edict.get("_imm_bufs")
    need_new_buf = True
    if isinstance(bufs, dict):
        ps = bufs.get("pos_sim"); pr = bufs.get("pos_ref")
        vs = bufs.get("vel_sim"); vr = bufs.get("vel_ref")
        if (isinstance(ps, np.ndarray) and ps.shape == (N, 3) and ps.dtype == np.float32 and ps.flags.c_contiguous and
            isinstance(pr, np.ndarray) and pr.shape == (N, 3) and pr.dtype == np.float32 and pr.flags.c_contiguous and
            isinstance(vs, np.ndarray) and vs.shape == (N, 3) and vs.dtype == np.float32 and vs.flags.c_contiguous and
            isinstance(vr, np.ndarray) and vr.shape == (N, 3) and vr.dtype == np.float32 and vr.flags.c_contiguous):
            need_new_buf = False

    if need_new_buf:
        bufs = {
            "pos_sim": np.empty((N, 3), dtype=np.float32, order="C"),
            "pos_ref": np.empty((N, 3), dtype=np.float32, order="C"),
            "vel_sim": np.empty((N, 3), dtype=np.float32, order="C"),
            "vel_ref": np.empty((N, 3), dtype=np.float32, order="C"),
        }

    w_eff = edict.get("_imm_wvec_eff")
    if (w_eff is None) or (w_eff.dtype != np.float32) or (w_eff.shape != (N,)):
        w_eff = np.empty_like(w_vec, dtype=np.float32)

    edict.update({
        "_imm_site_order": order,
        "_imm_wvec": w_vec,
        "_imm_wsum": float(wsum),
        "_imm_bufs": bufs,
        "_imm_wvec_eff": w_eff,
    })

    return order, w_vec, float(wsum)

def _fill_site_buffers(
        buf: Dict[str, np.ndarray],
        order: List[str],
        c_sites: Dict[str, Dict[str, np.ndarray]],
        r_sites: Dict[str, Dict[str, np.ndarray]],
        c_pel: Dict[str, np.ndarray],
        r_pel: Dict[str, np.ndarray],
        ) -> None:
    """
    Populate per-site position/velocity buffers for simulated (current) and
    reference data in the prescribed `order`.

    Parameters
    ----------
    buf : dict
        Preallocated float32 buffers with keys and shapes:
          - "pos_sim": (N, 3)
          - "pos_ref": (N, 3)
          - "vel_sim": (N, 3)
          - "vel_ref": (N, 3)
        where N = len(order).
    order : list[str]
        Site packing order. Must start with "pelvis"; remaining names are site keys.
    c_sites : dict
        Current (sim) site components:
          { "pos": {name: (3,)}, "linvel": {name: (3,)} }.
    r_sites : dict
        Reference site components in the same structure as `c_sites`.
    c_pel : dict
        Current pelvis components:
          { "pos": (3,), "linvel": (3,) }.
    r_pel : dict
        Reference pelvis components:
          { "pos": (3,), "linvel": (3,) }.

    Returns
    -------
    None
        Fills `buf` in-place:
          - Row 0 from pelvis (current/ref).
          - Rows 1..N-1 from sites in `order[1:]`.

    Notes
    -----
    - Assumes all entries exist; missing site names will raise `KeyError`.
    - No allocations besides transient Python lists of views; array writes are in-place.
    """
    pos_s = buf["pos_sim"]; pos_r = buf["pos_ref"]
    vel_s = buf["vel_sim"]; vel_r = buf["vel_ref"]
    
    cpos = c_sites["pos"];  rpos = r_sites["pos"]
    cvel = c_sites["linvel"]; rvel = r_sites["linvel"]
    
    n = len(order)
    if n == 0:
        return
    
    pos_s[0, :] = c_pel["pos"]
    pos_r[0, :] = r_pel["pos"]
    vel_s[0, :] = c_pel["linvel"]
    vel_r[0, :] = r_pel["linvel"]
    if n == 1:
        return
    
    names = order[1:]
    cps = [cpos[name] for name in names]
    rps = [rpos[name] for name in names]
    cvs = [cvel[name] for name in names]
    rvs = [rvel[name] for name in names]

    for i in range(1, n):
        j = i - 1
        pos_s[i, :] = cps[j]
        pos_r[i, :] = rps[j]
        vel_s[i, :] = cvs[j]
        vel_r[i, :] = rvs[j]

def compute_reward(
        env: Any, 
        obs_components: Dict[str, Any],
        ) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the total reward and per-term diagnostics based on the current phase.

    Parameters
    ----------
    env : Any
        Environment providing:
          - phase : object or str
              Phase name; supported values: "IMITATION", "REACH_GOAL".
          - reward_weights : dict
              Weights for each term, keys may include:
                * "imitation" : float
                * "goal"      : float
                * "smooth"    : float
          - get_imitation_reward(env, obs_components) -> (float, dict)
          - get_goal_reward(env) -> (float, dict)
          - get_smooth_reward(env) -> (float, dict)
          - logger : optional, used for error reporting.
    obs_components : Dict[str, Any]
        Auxiliary structures produced by the observation builder, passed through
        to `get_imitation_reward`.

    Returns
    -------
    total : float
        Weighted sum of active reward terms for the current phase.
    info : dict
        Per-component diagnostics, e.g.:
          {
            "imitation": { ... } or {"skipped": True, "weight": 0.0},
            "goal":      { ... } or {"skipped": True, "weight": 0.0},
            "smooth":    { ... } or {"skipped": True, "weight": 0.0}
          }
        If the phase is unknown, returns:
          (0.0, {"error": "Unknown phase '...'"})

    Notes
    -----
    - IMITATION phase: combines "imitation" and "smooth" terms (if weights ≠ 0).
    - REACH_GOAL phase: combines "goal" and "smooth" terms (if weights ≠ 0).
    - Terms with zero weight are skipped and annotated as such in `info`.
    """
    # --- Local cache ---
    phase_obj = getattr(env, "phase", None)
    phase = getattr(phase_obj, "name", str(phase_obj))
    rw = env.reward_weights
    
    _imm   = get_imitation_reward
    _goal  = get_goal_reward
    _terr  = get_terrain_reward
    _sm    = get_smooth_reward
    
    w_smooth = float(rw.get("smooth", 0.0))
    w_imm    = float(rw.get("imitation", 0.0))
    w_goal   = float(rw.get("goal", 0.0))
    w_terrain = float(rw.get("terrain", 0.0))
    w_alive = float(rw.get("alive", 0.0))
    w_grf = float(rw.get("grf", 0.0))
    w_com = float(rw.get("com", 0.0))
    w_upward = float(rw.get("upward", 0.0))
    
    total: float = 0.0
    info: Dict[str, Any] = {}
    
    if phase == "IMITATION":
        if w_alive != 0.0:
            r_alive, alive_info = get_alive_reward(env, obs_components)
            total += w_alive * float(r_alive)
            alive_info["weight"] = w_alive
        else:
            alive_info = {"skipped": True, "weight": 0.0}
        info["alive"] = alive_info

        if w_imm != 0.0:
            r_imm, imm_info = _imm(env, obs_components)
            total += w_imm * float(r_imm)
        else:
            imm_info = {"skipped": True, "weight": 0.0}
        info["imitation"] = imm_info

        if w_terrain != 0.0:
            r_terrain, terrain_info = _terr(env, obs_components)
            total += w_terrain * float(r_terrain)
            terrain_info["weight"] = w_terrain
        else:
            terrain_info = {"skipped": True, "weight": 0.0}
        info["terrain"] = terrain_info

        if w_com != 0.0:
            r_com, com_info = get_com_balance_reward(env, obs_components)
            total += w_com * float(r_com)
            com_info["weight"] = w_com
        else:
            com_info = {"skipped": True, "weight": 0.0}
        info["com"] = com_info

        if w_upward != 0.0:
            r_upward, upward_info = get_upward_reward(env, obs_components)
            total += w_upward * float(r_upward)
            upward_info["weight"] = w_upward
        else:
            upward_info = {"skipped": True, "weight": 0.0}
        info["upward"] = upward_info
        
        if w_smooth != 0.0:
            r_smooth, smooth_info = _sm(env)
            total += w_smooth * float(r_smooth)
        else:
            smooth_info = {"skipped": True, "weight": 0.0}
        info["smooth"] = smooth_info

        if w_grf != 0.0:
            grf_cost, grf_info = get_grf_reward(env, obs_components)
            total += w_grf * float(grf_cost)
            grf_info["weight"] = w_grf
        else:
            grf_info = {"skipped": True, "weight": 0.0}
        info["grf"] = grf_info
        return float(total), info

    elif phase == "REACH_GOAL":
        if w_alive != 0.0:
            r_alive, alive_info = get_alive_reward(env, obs_components)
            total += w_alive * float(r_alive)
            alive_info["weight"] = w_alive
        else:
            alive_info = {"skipped": True, "weight": 0.0}
        info["alive"] = alive_info

        if w_goal != 0.0:
            r_goal, goal_info = _goal(env)
            total += w_goal * float(r_goal)
        else:
            goal_info = {"skipped": True, "weight": 0.0}
        info["goal"] = goal_info

        if w_terrain != 0.0:
            r_terrain, terrain_info = _terr(env, obs_components)
            total += w_terrain * float(r_terrain)
            terrain_info["weight"] = w_terrain
        else:
            terrain_info = {"skipped": True, "weight": 0.0}
        info["terrain"] = terrain_info

        if w_com != 0.0:
            r_com, com_info = get_com_balance_reward(env, obs_components)
            total += w_com * float(r_com)
            com_info["weight"] = w_com
        else:
            com_info = {"skipped": True, "weight": 0.0}
        info["com"] = com_info

        if w_upward != 0.0:
            r_upward, upward_info = get_upward_reward(env, obs_components)
            total += w_upward * float(r_upward)
            upward_info["weight"] = w_upward
        else:
            upward_info = {"skipped": True, "weight": 0.0}
        info["upward"] = upward_info
        
        if w_smooth != 0.0:
            r_smooth, smooth_info = _sm(env)
            total += w_smooth * float(r_smooth)
        else:
            smooth_info = {"skipped": True, "weight": 0.0}
        info["smooth"] = smooth_info

        if w_grf != 0.0:
            grf_cost, grf_info = get_grf_reward(env, obs_components)
            total += w_grf * float(grf_cost)
            grf_info["weight"] = w_grf
        else:
            grf_info = {"skipped": True, "weight": 0.0}
        info["grf"] = grf_info
        return float(total), info

    logger = getattr(env, "logger", None)
    if logger is not None:
        logger.error("compute_reward failed: %r", phase)
    return 0.0, {"error": f"Unknown phase {phase!r}"}


def get_alive_reward(env: Any, comps: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Return a small dense reward while the pelvis remains in a valid height range."""
    params = getattr(env, "alive_reward", {}) or {}
    reward_value = float(params.get("reward", 1.0))
    fallen_value = float(params.get("fallen_reward", 0.0))
    height_range = params.get("pelvis_height_range", (0.6, 1.3))
    if not isinstance(height_range, (list, tuple)) or len(height_range) != 2:
        height_range = (0.6, 1.3)
    min_height, max_height = (float(height_range[0]), float(height_range[1]))
    use_relative = bool(params.get("use_terrain_relative_height", False))

    pelvis = comps.get("curr", {}).get("pelvis", {})
    pelvis_pos = pelvis.get("pos", None)
    if pelvis_pos is None:
        try:
            pelvis_pos = env.data.xpos[env.model.body("pelvis").id]
        except Exception as exc:
            return fallen_value, {"alive": False, "error": str(exc)}

    height = float(pelvis_pos[2])
    terrain_height = None
    if use_relative:
        terrain_height = _sample_terrain_height(env, float(pelvis_pos[0]), float(pelvis_pos[1]))
        if terrain_height is not None:
            height -= float(terrain_height)

    alive = (height >= min_height) and (height <= max_height)
    return (reward_value if alive else fallen_value), {
        "alive": bool(alive),
        "height": float(height),
        "terrain_height": None if terrain_height is None else float(terrain_height),
        "height_range": [float(min_height), float(max_height)],
        "reward": float(reward_value if alive else fallen_value),
    }


def get_grf_reward(env: Any, comps: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Return a positive GRF cost; use a negative reward weight to penalize it."""
    params = getattr(env, "grf_reward", {}) or {}
    max_total_bw = float(params.get("max_total_force_bw", 1.6))
    max_vertical_bw = float(params.get("max_vertical_force_bw", 1.3))
    total_weight = float(params.get("total_force_weight", 0.5))
    vertical_weight = float(params.get("vertical_force_weight", 1.0))

    body_weight = float(getattr(env, "total_mass", 0.0)) * 9.81
    if not np.isfinite(body_weight) or body_weight <= _EPSILON:
        return 0.0, {"skipped": True, "reason": "invalid_body_weight", "body_weight": body_weight}

    try:
        from .state import get_GRF_info

        _, grf_comp = get_GRF_info(env)
    except Exception as exc:
        return 0.0, {"skipped": True, "reason": "grf_unavailable", "error": str(exc)}

    per_foot = []
    costs = []
    for side in ("right", "left"):
        force = np.asarray(grf_comp[side]["GRF_force"], dtype=np.float64)
        total_bw = float(np.linalg.norm(force) / body_weight)
        vertical_bw = float(abs(force[2]) / body_weight)
        total_excess = max(0.0, total_bw - max_total_bw)
        vertical_excess = max(0.0, vertical_bw - max_vertical_bw)
        cost = total_weight * total_excess * total_excess + vertical_weight * vertical_excess * vertical_excess
        costs.append(cost)
        per_foot.append(
            {
                "side": side,
                "total_force_bw": total_bw,
                "vertical_force_bw": vertical_bw,
                "total_excess": float(total_excess),
                "vertical_excess": float(vertical_excess),
                "cost": float(cost),
            }
        )

    mean_cost = float(np.mean(costs)) if costs else 0.0
    return mean_cost, {
        "cost": mean_cost,
        "max_total_force_bw": float(max_total_bw),
        "max_vertical_force_bw": float(max_vertical_bw),
        "total_force_weight": float(total_weight),
        "vertical_force_weight": float(vertical_weight),
        "feet": per_foot,
    }


def _world_com_kinematics(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(env.model.body_mass[1:], dtype=np.float64)
    total_mass = float(masses.sum())
    if not np.isfinite(total_mass) or total_mass <= _EPSILON:
        raise ValueError(f"invalid total mass: {total_mass}")

    cvel = np.asarray(env.data.cvel[1:, 3:6], dtype=np.float64)
    if cvel.shape[0] != masses.shape[0]:
        raise ValueError(f"COM velocity shape mismatch: masses={masses.shape}, cvel={cvel.shape}")

    com_pos = np.asarray(env.data.subtree_com[0], dtype=np.float64)
    com_vel = (masses @ cvel) / total_mass
    return com_pos, com_vel


def _body_axis_world(env: Any, body_name: str, body_axis: Tuple[float, float, float]) -> Optional[np.ndarray]:
    try:
        bid = int(env.model.body(str(body_name)).id)
    except Exception:
        return None

    axis = np.asarray(body_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= _EPSILON:
        return None
    axis = axis / norm

    rot = np.asarray(env.data.xmat[bid], dtype=np.float64).reshape(3, 3)
    vec = rot @ axis
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm <= _EPSILON:
        return None
    return vec / vec_norm


def _upper_body_vector(env: Any, params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    min_length = max(float(params.get("min_vector_length", 0.05)), 1e-6)

    if bool(params.get("use_sites", True)):
        start_site = str(params.get("start_site", "pelvis_sensor"))
        end_site = str(params.get("end_site", "thorax_sensor"))
        start = _site_world_pos(env, start_site)
        end = _site_world_pos(env, end_site)
        if start is not None and end is not None:
            vec = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
            length = float(np.linalg.norm(vec))
            if length >= min_length:
                return vec / length, {
                    "source": "sites",
                    "start_site": start_site,
                    "end_site": end_site,
                    "length": length,
                }

    body_name = str(params.get("body_name", "thorax"))
    body_axis_raw = params.get("body_axis", (0.0, 0.0, 1.0))
    if not isinstance(body_axis_raw, (list, tuple)) or len(body_axis_raw) != 3:
        body_axis_raw = (0.0, 0.0, 1.0)
    body_axis = tuple(float(v) for v in body_axis_raw)
    vec = _body_axis_world(env, body_name, body_axis)
    if vec is None:
        return None, {"source": "none", "reason": "upper_body_vector_unavailable"}
    return vec, {"source": "body_axis", "body_name": body_name, "body_axis": list(body_axis)}


def get_upward_reward(env: Any, comps: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Reward the upper-body vector for staying broadly upward without requiring
    exact alignment with world z.
    """
    params = getattr(env, "upward_reward", {}) or {}
    vec, source_info = _upper_body_vector(env, params)
    if vec is None:
        return 0.0, {"skipped": True, **source_info}

    target_dot = float(params.get("target_dot", 0.85))
    target_dot = float(np.clip(target_dot, -1.0, 1.0))
    sigma_dot = max(float(params.get("sigma_dot", 0.12)), 1e-6)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot_up = float(np.clip(np.dot(vec, world_up), -1.0, 1.0))

    # Saturate at full reward once the vector is upright enough. This avoids
    # forcing a humanoid segment to be perfectly parallel to world z.
    deficit = max(0.0, target_dot - dot_up)
    reward = math.exp(-((deficit / sigma_dot) ** 2))

    tilt_angle_deg = math.degrees(math.acos(dot_up))
    allowed_tilt_deg = math.degrees(math.acos(target_dot))
    return float(reward), {
        "reward": float(reward),
        "dot_up": float(dot_up),
        "target_dot": float(target_dot),
        "deficit": float(deficit),
        "sigma_dot": float(sigma_dot),
        "tilt_angle_deg": float(tilt_angle_deg),
        "allowed_tilt_deg": float(allowed_tilt_deg),
        "vector": vec.astype(float).tolist(),
        **source_info,
    }


def _mean_site_world_pos(env: Any, site_names: Tuple[str, ...]) -> Optional[np.ndarray]:
    positions = []
    for site_name in site_names:
        pos = _site_world_pos(env, str(site_name))
        if pos is not None:
            positions.append(np.asarray(pos, dtype=np.float64))
    if not positions:
        return None
    return np.mean(np.stack(positions, axis=0), axis=0)


def _walking_axes(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    heading = np.asarray(getattr(env, "pelvis_heading", np.array([1.0, 0.0, 0.0])), dtype=np.float64)[:2]
    norm = float(np.linalg.norm(heading))
    if norm <= _EPSILON:
        heading = np.array([1.0, 0.0], dtype=np.float64)
    else:
        heading = heading / norm
    lateral = np.array([-heading[1], heading[0]], dtype=np.float64)
    return heading, lateral


def _project_xy(vec_xy: np.ndarray, forward: np.ndarray, lateral: np.ndarray) -> Tuple[float, float]:
    return float(np.dot(vec_xy, forward)), float(np.dot(vec_xy, lateral))


def get_com_balance_reward(env: Any, comps: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Reward COM balance by keeping COM/capture-point projection near the support feet.

    The capture point term gives the policy an anticipatory signal: high COM velocity
    away from the stance foot is penalized before the pelvis actually falls.
    """
    params = getattr(env, "com_reward", {}) or {}

    try:
        com_pos, com_vel = _world_com_kinematics(env)
    except Exception as exc:
        return 0.0, {"skipped": True, "reason": "com_unavailable", "error": str(exc)}

    right_sites = tuple(params.get("right_support_sites", ("toes_r_sensor", "calcn_r_sensor")))
    left_sites = tuple(params.get("left_support_sites", ("toes_l_sensor", "calcn_l_sensor")))
    right_center = _mean_site_world_pos(env, right_sites)
    left_center = _mean_site_world_pos(env, left_sites)

    right_contact, left_contact = _foot_contact_flags(env, comps)
    support_points = []
    contact_mode = "contact"
    if right_contact and right_center is not None:
        support_points.append(right_center)
    if left_contact and left_center is not None:
        support_points.append(left_center)

    if not support_points:
        contact_mode = "fallback_feet"
        if right_center is not None:
            support_points.append(right_center)
        if left_center is not None:
            support_points.append(left_center)
    if not support_points:
        return 0.0, {"skipped": True, "reason": "support_sites_unavailable"}

    support_center = np.mean(np.stack(support_points, axis=0), axis=0)
    support_height = float(np.mean([p[2] for p in support_points]))
    com_height = max(float(com_pos[2]) - support_height, 0.10)
    omega = math.sqrt(9.81 / com_height)
    capture_xy = com_pos[:2] + com_vel[:2] / max(omega, _EPSILON)

    forward_axis, lateral_axis = _walking_axes(env)
    target_xy = support_center[:2].copy()
    target_xy += forward_axis * float(params.get("target_forward_offset", 0.02))
    target_xy += lateral_axis * float(params.get("target_lateral_offset", 0.0))

    pos_f, pos_l = _project_xy(com_pos[:2] - target_xy, forward_axis, lateral_axis)
    cap_f, cap_l = _project_xy(capture_xy - target_xy, forward_axis, lateral_axis)
    vel_f, vel_l = _project_xy(com_vel[:2], forward_axis, lateral_axis)

    pos_f_sigma = max(float(params.get("position_forward_sigma", 0.25)), 1e-6)
    pos_l_sigma = max(float(params.get("position_lateral_sigma", 0.10)), 1e-6)
    cap_f_sigma = max(float(params.get("capture_forward_sigma", 0.35)), 1e-6)
    cap_l_sigma = max(float(params.get("capture_lateral_sigma", 0.14)), 1e-6)
    lat_vel_sigma = max(float(params.get("lateral_velocity_sigma", 0.35)), 1e-6)
    height_sigma = max(float(params.get("height_sigma", 0.15)), 1e-6)
    height_min = float(params.get("height_min", 0.65))

    position_reward = math.exp(-((pos_f / pos_f_sigma) ** 2 + (pos_l / pos_l_sigma) ** 2))
    capture_reward = math.exp(-((cap_f / cap_f_sigma) ** 2 + (cap_l / cap_l_sigma) ** 2))
    lateral_velocity_reward = math.exp(-((vel_l / lat_vel_sigma) ** 2))
    height_reward = math.exp(-((max(0.0, height_min - com_height) / height_sigma) ** 2))

    component_weights = {
        "position": float(params.get("position_weight", 0.5)),
        "capture": float(params.get("capture_weight", 1.0)),
        "lateral_velocity": float(params.get("lateral_velocity_weight", 0.25)),
        "height": float(params.get("height_weight", 0.0)),
    }
    component_rewards = {
        "position": position_reward,
        "capture": capture_reward,
        "lateral_velocity": lateral_velocity_reward,
        "height": height_reward,
    }

    active_weight = sum(max(0.0, w) for w in component_weights.values())
    if active_weight <= _EPSILON:
        return 0.0, {"skipped": True, "reason": "zero_component_weight"}

    reward = sum(max(0.0, component_weights[k]) * component_rewards[k] for k in component_rewards)
    reward /= active_weight

    return float(reward), {
        "reward": float(reward),
        "contact_mode": contact_mode,
        "right_contact": bool(right_contact),
        "left_contact": bool(left_contact),
        "com_pos": com_pos.astype(float).tolist(),
        "com_vel": com_vel.astype(float).tolist(),
        "support_center": support_center.astype(float).tolist(),
        "capture_xy": capture_xy.astype(float).tolist(),
        "com_height": float(com_height),
        "omega": float(omega),
        "errors": {
            "position_forward": float(pos_f),
            "position_lateral": float(pos_l),
            "capture_forward": float(cap_f),
            "capture_lateral": float(cap_l),
            "velocity_forward": float(vel_f),
            "velocity_lateral": float(vel_l),
        },
        "components": {
            "position": {"weight": component_weights["position"], "reward": float(position_reward)},
            "capture": {"weight": component_weights["capture"], "reward": float(capture_reward)},
            "lateral_velocity": {
                "weight": component_weights["lateral_velocity"],
                "reward": float(lateral_velocity_reward),
            },
            "height": {"weight": component_weights["height"], "reward": float(height_reward)},
        },
    }


def get_imitation_reward(
        env: Any,
        comps: Dict[str, Any],
        include_orient_angvel: bool = False,
        site_weights: Dict[str, float] | None = None,
        ) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the imitation reward from site- and joint-level tracking terms.

    Parameters
    ----------
    env : Any
        Environment providing:
          - imitation_weights : dict with keys
              * "site_pos", "site_vel", "site_orn", "site_angvel",
              * "joint_angle", "joint_angvel"
            (floats; zero disables the term).
          - compute_site_imitation(env, comps, include_orient_angvel, site_weights)
            → (r_pos, r_vel, r_orn or None, r_angvel or None, info)
          - compute_joint_imitation(env)
            → (r_angle, r_angvel, info)
    comps : Dict[str, Any]
        Observation components bundle from `get_state`, passed to
        `compute_site_imitation`.
    include_orient_angvel : bool, default=False
        If True, request orientation (6D) and angular-velocity site terms from
        `compute_site_imitation` even if the corresponding weights are zero
        (useful for logging/diagnostics).
    site_weights : dict[str, float] or None, optional
        Optional per-site weighting overrides forwarded to `compute_site_imitation`.

    Returns
    -------
    total : float
        Weighted sum of active imitation terms:
          w_sp*r_sp + w_sv*r_sv + w_so*r_so + w_sa*r_sa + w_ja*r_ja + w_jv*r_jv.
    breakdown : dict
        Per-term values and sub-infos, e.g.:
          {
            "site_pos": float,
            "site_vel": float,
            "site_orn": float (if available),
            "site_angvel": float (if available),
            "site_info": {...},
            "joint_angle": float,
            "joint_angvel": float,
            "joint_info": {...}
          }
        When a group is disabled, a "skipped" entry with the corresponding
        weights is provided instead.

    Notes
    -----
    - Site terms are computed only if any of {site_pos, site_vel, site_orn,
      site_angvel} has nonzero weight, or `include_orient_angvel=True`.
    - Joint terms are computed only if any of {joint_angle, joint_angvel}
      has nonzero weight.
    """
    iw = env.imitation_weights
    coeffs = env.reward_coefficients
    _site_imm = compute_site_imitation
    _joint_imm = compute_joint_imitation
    
    w_sp = float(iw.get("site_pos", 1.0))
    w_sv = float(iw.get("site_vel", 1.0))
    enable_site_orn = bool(
        getattr(env, "enable_site_orn", float(iw.get("site_orn", coeffs.get("site_orn", 0.0))) != 0.0)
    )
    enable_site_angvel = bool(
        getattr(env, "enable_site_angvel", float(iw.get("site_angvel", coeffs.get("site_angvel", 0.0))) != 0.0)
    )
    w_so = float(iw.get("site_orn", 1.0 if enable_site_orn else 0.0))
    w_sa = float(iw.get("site_angvel", 1.0 if enable_site_angvel else 0.0))
    w_ja = float(iw.get("joint_angle", 1.0))
    w_jv = float(iw.get("joint_angvel", 1.0))
    
    need_site_posvel = (w_sp != 0.0) or (w_sv != 0.0)
    need_site_orient = enable_site_orn or enable_site_angvel
    need_site_any    = need_site_posvel or need_site_orient
    need_joint_any   = (w_ja != 0.0) or (w_jv != 0.0)
    
    total = 0.0
    breakdown: Dict[str, Any] = {}
    
    # Site tracking
    if need_site_any or include_orient_angvel:
        inc_ori_av = include_orient_angvel or need_site_orient
        r_sp, r_sv, r_so, r_sa, site_info = _site_imm(
            env,
            comps,
            include_orient_angvel=inc_ori_av,
            site_weights=site_weights
        )
        if w_sp != 0.0: total += w_sp * float(r_sp); breakdown["site_pos"] = float(r_sp)
        if w_sv != 0.0: total += w_sv * float(r_sv); breakdown["site_vel"] = float(r_sv)
        if (r_so is not None) and enable_site_orn:
            v = float(r_so); total += w_so * v; breakdown["site_orn"] = v
        if (r_sa is not None) and enable_site_angvel:
            v = float(r_sa); total += w_sa * v; breakdown["site_angvel"] = v
        breakdown["site_info"] = site_info
    else:
        breakdown["site_info"] = {"skipped": True, "weights": {"pos": w_sp, "vel": w_sv, "orn": w_so, "angvel": w_sa}}
        
    # Joint tracking
    if need_joint_any:
        r_ja, r_jv, joint_info = _joint_imm(env)
        if w_ja != 0.0: total += w_ja * float(r_ja); breakdown["joint_angle"]  = float(r_ja)
        if w_jv != 0.0: total += w_jv * float(r_jv); breakdown["joint_angvel"] = float(r_jv)
        breakdown["joint_info"] = joint_info
    else:
        breakdown["joint_info"] = {"skipped": True, "weights": {"angle": w_ja, "angvel": w_jv}}

    return float(total), breakdown
            
def compute_site_imitation(
        env: Any,
        comps: Dict[str, Any],
        include_orient_angvel: bool = False,
        site_weights: Optional[Dict[str, float]] = None,
        ) -> Tuple[float, float, Optional[float], Optional[float], Dict[str, Any]]:
    """
    Compute site-level imitation rewards (positions, linear velocities, and
    optionally orientations and angular velocities) with per-site weighting.

    Parameters
    ----------
    env : Any
        Environment providing:
          - relative_pelvis : bool
              If True, pelvis terms are treated in a pelvis-local way:
              position/linvel deltas for pelvis are zeroed on the sim side and
              the ref pelvis values are used directly.
          - reward_coefficients : dict
              Exponent coefficients for each term, e.g.:
                "site_pos", "site_vel", "site_orn", "site_angvel".
          - EPSILON : float (optional) small threshold for snapping exp ≈ 1.0.
          - Internal caches populated by `_build_site_weight_cache`:
              * _imm_bufs : dict with (N,3) buffers for sim/ref pos/vel.
              * _imm_wvec_eff : (N,) float32 effective weights (may be reused).
    comps : Dict[str, Any]
        Output from `get_state`:
          {
            "curr": {"sites": {...}, "pelvis": {...}},
            "ref":  {"sites": {...}, "pelvis": {...}}
          }
        Where each sites dict includes:
          - "pos": {name: (3,)},
          - "linvel": {name: (3,)},
          - ["orient": {name: (6,)}],
          - ["angvel": {name: (3,)}].
        Pelvis dicts include: "pos", "linvel", and optionally "orient", "angvel".
    include_orient_angvel : bool, default=False
        If True, also compute orientation (6D) and angular-velocity rewards
        when the corresponding components exist in both current and reference.
    site_weights : dict[str, float] or None, optional
        Optional multiplicative overrides per site name (including "pelvis").
        Applied on top of base weights from `_build_site_weight_cache`.

    Returns
    -------
    r_sp : float
        Position imitation reward in [0, 1], computed as exp(-c_pos * MSE_pos).
    r_sv : float
        Linear-velocity imitation reward in [0, 1], exp(-c_vel * MSE_vel).
    r_so : float or None
        Orientation imitation reward in [0, 1] if computed, else None.
    r_sa : float or None
        Angular-velocity imitation reward in [0, 1] if computed, else None.
    info : dict
        Diagnostics:
          {
            "pos_err": float,             # weighted MSE of positions
            "vel_err": float,             # weighted MSE of linear velocities
            "orn_err": float or None,     # mean squared geodesic angle (weighted)
            "angvel_err": float or None,  # weighted MSE of angular velocities
            "weights_sum": float          # Σ weights actually used
          }

    Notes
    -----
    - Site ordering and base weights are produced by `_build_site_weight_cache`,
      which also ensures reusable (N,3) float32 buffers. `_fill_site_buffers`
      populates these buffers each call.
    - Position/velocity errors use `_mse3d` with weights; orientation error uses
      squared geodesic angle between rotation matrices; angular-velocity error
      is squared L2.
    - Rewards are `exp(-coef * error)` per term, with coefficients read from
      `env.reward_coefficients`. Very small (1 - reward) values are snapped to
      0 using `EPSILON` for numerical cleanliness.
    - If any effective weight sum is invalid (NaN/≤0), uniform weights are used.
    """
    try:
        curr = comps["curr"]; ref = comps["ref"]
        c_sites, r_sites = curr["sites"], ref["sites"]
        c_pel,   r_pel   = curr["pelvis"], ref["pelvis"]
        rel = bool(getattr(env, "relative_pelvis", False))
    
        order, w_base, w_sum_base = _build_site_weight_cache(
            env, c_sites["pos"], r_sites["pos"], c_pel, r_pel
        )
        bufs = env._imm_bufs
        _fill_site_buffers(bufs, order, c_sites, r_sites, c_pel, r_pel)
        
        if rel:
            i0 = 0
            ps0 = bufs["pos_sim"]; pr0 = bufs["pos_ref"]
            vs0 = bufs["vel_sim"]; vr0 = bufs["vel_ref"]
            ps0[i0, :].fill(0.0)
            pr0[i0, :] = np.asarray(r_pel["pos"], dtype=np.float32, order="C")
            pr0[i0, 2] = 0.0
            vs0[i0, :].fill(0.0)
            vr0[i0, :] = np.asarray(r_pel["linvel"], dtype=np.float32, order="C")
            vr0[i0, 2] = 0.0
            
        if site_weights is None:
            w_eff = w_base
            w_sum = w_sum_base
        else:
            eff = getattr(env, "_imm_wvec_eff", None)
            if (eff is None) or (eff.shape != w_base.shape) or (eff.dtype != w_base.dtype):
                eff = w_base.copy()
                env._imm_wvec_eff = eff
            else:
                np.copyto(eff, w_base, casting="no")
                
            for i, name in enumerate(order):
                eff[i] *= float(site_weights.get(name, 1.0))
                
            w_eff = eff
            # Test
            # w_eff=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            #     1., 1., 1., 1.])
            w_sum = float(w_eff.sum())
            if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
                w_eff.fill(1.0)
                w_sum = float(w_eff.size)
    
        pos_err = _mse3d(bufs["pos_sim"], bufs["pos_ref"], w_eff, w_sum)
        vel_err = _mse3d(bufs["vel_sim"], bufs["vel_ref"], w_eff, w_sum)
    
        # orient/angvel (optional)
        need_oa = bool(include_orient_angvel)
        has_orn = ("orient" in c_sites) and ("orient" in r_sites)
        has_ang = ("angvel" in c_sites) and ("angvel" in r_sites)
        use_oa  = need_oa and has_orn and has_ang
    
        orn_err = ang_err = None
        if use_oa:
            c_orn = c_sites["orient"]; r_orn = r_sites["orient"]
            c_ang = c_sites["angvel"]; r_ang = r_sites["angvel"]
            if rel:
                Rs_pel = np.eye(3, dtype=np.float64)
                Rr_pel = quat_to_mat(r_pel["orient"])
            else:
                Rs_pel = quat_to_mat(c_pel["orient"])
                Rr_pel = quat_to_mat(r_pel["orient"])

            tot_orn = 0.0
            tot_ang = 0.0
            
            for i, name in enumerate(order):
                wi = float(w_eff[i])
                if name == "pelvis":
                    Rs = Rs_pel; Rr = Rr_pel
                else:
                    Rs = orient6_to_mat(c_orn[name])
                    Rr = orient6_to_mat(r_orn[name])
                th = rotation_geodesic(Rr, Rs)
                tot_orn += wi * (th * th)
                
                if name == "pelvis":
                    if rel:
                        dvx, dvy, dvz = float(r_pel["angvel"][0]), float(r_pel["angvel"][1]), float(r_pel["angvel"][2])
                        dv2 = dvx*dvx + dvy*dvy + dvz*dvz
                    else:
                        dv = c_pel["angvel"] - r_pel["angvel"]
                        dv2 = float(dv[0])*float(dv[0]) + float(dv[1])*float(dv[1]) + float(dv[2])*float(dv[2])
                else:
                    dv = c_ang[name] - r_ang[name]
                    dv2 = float(dv[0])*float(dv[0]) + float(dv[1])*float(dv[1]) + float(dv[2])*float(dv[2])

                tot_ang += wi * dv2

            orn_err = tot_orn / w_sum
            ang_err = tot_ang / w_sum
            
        coeffs = env.reward_coefficients
        EPS    = float(getattr(env, "EPSILON", 1e-8))
    
        r_sp = math.exp(-float(coeffs.get("site_pos", 20.0)) * float(pos_err))
        r_sv = math.exp(-float(coeffs.get("site_vel", 0.05)) * float(vel_err))
        if abs(r_sp - 1.0) < EPS: r_sp = 1.0
        if abs(r_sv - 1.0) < EPS: r_sv = 1.0
    
        r_so = r_sa = None
        if use_oa:
            r_so = math.exp(-float(coeffs.get("site_orn", 1.0)) * float(orn_err))
            r_sa = math.exp(-float(coeffs.get("site_angvel", 1.0)) * float(ang_err))
            if abs(r_so - 1.0) < EPS: r_so = 1.0
            if abs(r_sa - 1.0) < EPS: r_sa = 1.0
    
        info = {
            "pos_err": float(pos_err),
            "vel_err": float(vel_err),
            "orn_err": (None if orn_err is None else float(orn_err)),
            "angvel_err": (None if ang_err is None else float(ang_err)),
            "weights_sum": float(w_sum),
        }
        return float(r_sp), float(r_sv), (None if r_so is None else float(r_so)), (None if r_sa is None else float(r_sa)), info

    except Exception as e:
        logger = getattr(env, "logger", None)
        if logger is not None:
            logger.error("compute_site_imitation failed: %s", e)
        return 0.0, 0.0, None, None, {"error": str(e)}
    
def compute_joint_imitation(env: Any) -> Tuple[float, float, Dict[str, Any]]:
    """
    Compute joint-angle and joint-velocity imitation rewards w/ cached weights.

    Parameters
    ----------
    env : Any
        Environment providing:
          - data.qpos, data.qvel : simulated generalized coordinates/velocities.
          - ref_traj.get_reference_trajectories() -> (qpos_ref, qvel_ref)
          - reward_coefficients : dict/obj with fields or keys:
              * joint_weight_decay : float in (0,1) or None
              * joint_decay_fraction: float in (0,1], default 0.2
              * joint_angle : float, default 20.0
              * joint_angvel: float, default 0.05
          - EPSILON : float (optional), small threshold for snapping exp≈1.
        Internal caches used/updated:
          - _imm_joint_wvec : (M,) float64 weights
          - _imm_joint_wsum : float
          - _imm_joint_buf_dth, _imm_joint_buf_dw : (M,) float64 scratch

    Returns
    -------
    r_ja : float
        Joint-angle imitation reward ∈ [0,1], exp(-k_ja * MSE_angle).
    r_jv : float
        Joint-velocity imitation reward ∈ [0,1], exp(-k_jv * MSE_angvel).
    info : dict
        {
          "ang_err": float,     # weighted MSE of wrapped angle deltas
          "angvel_err": float   # weighted MSE of velocity deltas
        }
        On error: {"error": str} and rewards = 0.0.

    Notes
    -----
    - Pelvis translation DOFs are excluded.
    - Angle deltas are wrapped to (-π, π] before squaring.
    - Per-joint weights are built once via `_build_joint_weights`
      and cached; if invalid, fall back to uniform weights.
    - Rewards are snapped to 1.0 when |r - 1| < EPSILON for numerical cleanliness.
    """
    try:
        # Simulation
        state_map = env.human_state_map
        qpos = env.data.qpos[state_map.qpos_indices]
        qvel = env.data.qvel[state_map.qvel_indices]
        jpos_c = inverse_convert_ref_traj_qpos(qpos)
        jvel_c = inverse_convert_ref_traj_qvel(qvel, jpos_c)
           
        jpos_c = jpos_c[3:]
        jvel_c = jvel_c[3:] 
        
        # Reference
        ref_qpos, ref_qvel = env.ref_traj.get_reference_trajectories()
        jpos_r = ref_qpos[3:]
        jvel_r = ref_qvel[3:]
    
        if jpos_c.size != jpos_r.size or jvel_c.size != jvel_r.size:
            raise ValueError(
                f"Joint length mismatch: sim {jpos_c.size}/{jvel_c.size}, "
                f"ref {jpos_r.size}/{jvel_r.size}"
            )
    
        M = jpos_c.size
        wv = getattr(env, "_imm_joint_wvec", None)
        if (wv is None) or (wv.size != M):
            coeffs = env.reward_coefficients
            decay = getattr(coeffs, "joint_weight_decay", coeffs.get("joint_weight_decay", None))
            frac  = float(getattr(coeffs, "joint_decay_fraction",
                                  coeffs.get("joint_decay_fraction", 0.2)))
            wv, wsum = _build_joint_weights(M, decay, frac)   # 你已有
            env._imm_joint_wvec = wv
            env._imm_joint_wsum = wsum
        else:
            wsum = env._imm_joint_wsum
            
        if not np.isfinite(wsum) or wsum <= 0.0:
            wv = np.ones(M, dtype=np.float64)
            wsum = float(M)
            env._imm_joint_wvec = wv
            env._imm_joint_wsum = wsum
        
        dth = getattr(env, "_imm_joint_buf_dth", None)
        if (dth is None) or (dth.shape[0] != M):
            dth = np.empty(M, dtype=np.float64)
            env._imm_joint_buf_dth = dth
            
        dw = getattr(env, "_imm_joint_buf_dw", None)
        if (dw is None) or (dw.shape[0] != M):
            dw = np.empty(M, dtype=np.float64)
            env._imm_joint_buf_dw = dw
        
        np.subtract(jpos_c, jpos_r, out=dth)
        dth += np.pi
        np.remainder(dth, 2.0 * np.pi, out=dth)
        dth -= np.pi
        
        np.subtract(jvel_c, jvel_r, out=dw)
        # Test
        # dth=abs(dth)
        np.multiply(dth, dth, out=dth)
        ang_err = float(np.dot(wv, dth)) / wsum
        
        np.multiply(dw, dw, out=dw)
        angvel_err = float(np.dot(wv, dw)) / wsum
        
        coeffs = env.reward_coefficients
        k_ja = float(getattr(coeffs, "joint_angle", 20.0) if hasattr(coeffs, "joint_angle") else coeffs.get("joint_angle", 20.0))
        k_jv = float(getattr(coeffs, "joint_angvel", 0.05) if hasattr(coeffs, "joint_angvel") else coeffs.get("joint_angvel", 0.05))
        EPS  = float(getattr(env, "EPSILON", 1e-8))

        r_ja = math.exp(-k_ja * ang_err)
        r_jv = math.exp(-k_jv * angvel_err)
        if abs(r_ja - 1.0) < EPS: r_ja = 1.0
        if abs(r_jv - 1.0) < EPS: r_jv = 1.0

        return float(r_ja), float(r_jv), {"ang_err": float(ang_err), "angvel_err": float(angvel_err)}

    except Exception as e:
        logger = getattr(env, "logger", None)
        if logger is not None:
            logger.error("compute_joint_imitation failed: %s", e)
        return 0.0, 0.0, {"error": str(e)}


def get_goal_reward(env: Any) -> Tuple[float, Dict[str, Any]]:
    weights = getattr(env, "goal_weights", {}) or {}
    params = getattr(env, "goal_reward", {}) or {}
    w_position = float(weights.get("position", 1.0))
    w_speed = float(weights.get("speed", 0.0))

    goal_pos = getattr(env, "goal_pos", None)
    try:
        pelvis_id = int(env.model.body("pelvis").id)
        pelvis_pos = np.asarray(env.data.xpos[pelvis_id], dtype=np.float64)
        pelvis_vel = np.asarray(env.data.qvel[:3], dtype=np.float64)
    except Exception as exc:
        return 0.0, {"error": str(exc)}

    has_goal_pos = goal_pos is not None
    if has_goal_pos:
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
    total = 0.0
    info: Dict[str, Any] = {}

    if w_position != 0.0 and has_goal_pos:
        sigma = max(float(params.get("position_sigma", 1.0)), 1e-6)
        delta_xy = goal_pos[:2] - pelvis_pos[:2]
        distance_xy = float(np.linalg.norm(delta_xy))
        position_reward = math.exp(-((distance_xy / sigma) ** 2))
        total += w_position * position_reward
        info["position"] = {
            "weight": w_position,
            "reward": float(position_reward),
            "distance_xy": distance_xy,
            "sigma": sigma,
            "goal_pos": goal_pos.astype(float).tolist(),
            "pelvis_pos": pelvis_pos.astype(float).tolist(),
        }
    else:
        reason = "zero_weight" if w_position == 0.0 else "goal_pos_unavailable"
        info["position"] = {"skipped": True, "weight": w_position, "reason": reason}

    if w_speed != 0.0:
        target_speed = params.get("target_speed", None)
        if target_speed is None:
            target_speed = getattr(getattr(env, "ref_traj", None), "speed", 0.0)
        target_speed = float(target_speed)
        sigma = max(float(params.get("speed_sigma", 0.3)), 1e-6)
        direction_mode = str(params.get("speed_direction", "goal"))
        if direction_mode == "pelvis" or not has_goal_pos:
            direction = np.asarray(getattr(env, "pelvis_heading", np.array([1.0, 0.0, 0.0])), dtype=np.float64)
            direction = direction[:2]
        else:
            direction = goal_pos[:2] - pelvis_pos[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = np.asarray(getattr(env, "pelvis_heading", np.array([1.0, 0.0, 0.0])), dtype=np.float64)[:2]
            norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            forward_speed = 0.0
            direction_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction_xy = direction / norm
            forward_speed = float(np.dot(pelvis_vel[:2], direction_xy))
        speed_err = forward_speed - target_speed
        speed_reward = math.exp(-((speed_err / sigma) ** 2))
        total += w_speed * speed_reward
        info["speed"] = {
            "weight": w_speed,
            "reward": float(speed_reward),
            "forward_speed": float(forward_speed),
            "target_speed": target_speed,
            "speed_err": float(speed_err),
            "sigma": sigma,
            "direction": direction_xy.astype(float).tolist(),
            "direction_mode": direction_mode,
        }
    else:
        info["speed"] = {"skipped": True, "weight": w_speed}

    return float(total), info


def _sample_terrain_height(env: Any, x: float, y: float) -> Optional[float]:
    if int(getattr(env.model, "nhfield", 0)) <= 0:
        return None
    try:
        from .terrain import sample_hfield_height_m

        return sample_hfield_height_m(
            env.model,
            env.data,
            float(x),
            float(y),
            hfield_name=str(getattr(env, "terrain_hfield_name", "random_terrain")),
            geom_name=str(getattr(env, "terrain_floor_geom_name", "floor")),
        )
    except Exception as exc:
        logger_obj = getattr(env, "logger", None)
        if logger_obj is not None:
            logger_obj.debug("terrain height sample failed: %s", exc)
        return None


def _site_world_pos(env: Any, site_name: str) -> Optional[np.ndarray]:
    cache = getattr(env, "_terrain_reward_site_ids", None)
    if cache is None:
        cache = {}
        env._terrain_reward_site_ids = cache

    sid = cache.get(site_name)
    if sid is None:
        try:
            sid = int(env.model.site(site_name).id)
        except Exception:
            cache[site_name] = -1
            return None
        cache[site_name] = sid
    elif sid < 0:
        return None

    return env.data.site_xpos[int(sid)]


def _foot_contact_flags(env: Any, comps: Dict[str, Any]) -> Tuple[bool, bool]:
    contacts = comps.get("foot_contacts", None)
    if contacts is not None and len(contacts) >= 2:
        return bool(float(contacts[0]) > 0.5), bool(float(contacts[1]) > 0.5)

    grf = comps.get("grf", None)
    if isinstance(grf, dict) and "right" in grf and "left" in grf:
        threshold = float(getattr(env, "contact_threshold", 5e-2)) * float(env.total_mass) * 9.81
        rf = grf["right"].get("GRF_force", None)
        lf = grf["left"].get("GRF_force", None)
        right = bool(rf is not None and abs(float(rf[2])) > threshold)
        left = bool(lf is not None and abs(float(lf[2])) > threshold)
        return right, left

    try:
        from .state import compute_foot_contacts, get_GRF_info

        _, grf_comp = get_GRF_info(env)
        threshold = float(getattr(env, "contact_threshold", 5e-2)) * float(env.total_mass) * 9.81
        flags = compute_foot_contacts(grf_comp, threshold=threshold)
        return bool(float(flags[0]) > 0.5), bool(float(flags[1]) > 0.5)
    except Exception as exc:
        logger_obj = getattr(env, "logger", None)
        if logger_obj is not None:
            logger_obj.debug("foot contact computation failed: %s", exc)

    return False, False


def _terrain_pelvis_height_reward(env: Any, comps: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    params = getattr(env, "terrain_reward", {}) or {}
    pelvis = comps.get("curr", {}).get("pelvis", {})
    pelvis_pos = pelvis.get("pos", None)
    if pelvis_pos is None:
        try:
            pelvis_pos = env.data.xpos[env.model.body("pelvis").id]
        except Exception as exc:
            return 0.0, {"error": str(exc)}

    terrain_h = _sample_terrain_height(env, float(pelvis_pos[0]), float(pelvis_pos[1]))
    if terrain_h is None:
        return 0.0, {"skipped": True, "reason": "terrain_height_unavailable"}

    nominal = float(params.get("nominal_pelvis_height", getattr(env, "pelvis_z_offset", 0.95)))
    sigma = max(float(params.get("pelvis_height_sigma", 0.12)), 1e-6)
    rel_height = float(pelvis_pos[2]) - float(terrain_h)
    err = rel_height - nominal
    reward = math.exp(-((err / sigma) ** 2))
    return float(reward), {
        "terrain_height": float(terrain_h),
        "relative_height": float(rel_height),
        "target_height": float(nominal),
        "err": float(err),
        "sigma": float(sigma),
    }


def _terrain_foot_clearance_cost(env: Any, comps: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    params = getattr(env, "terrain_reward", {}) or {}
    min_clearance = float(params.get("foot_clearance_min", 0.10))
    sigma = max(float(params.get("foot_clearance_sigma", 0.05)), 1e-6)
    swing_only = bool(params.get("foot_clearance_swing_only", True))

    right_contact, left_contact = _foot_contact_flags(env, comps)
    side_specs = (
        ("right", tuple(params.get("right_foot_sites", ("toes_r_sensor", "calcn_r_sensor"))), right_contact),
        ("left", tuple(params.get("left_foot_sites", ("toes_l_sensor", "calcn_l_sensor"))), left_contact),
    )

    costs = []
    per_site = []
    for side, site_names, in_contact in side_specs:
        if swing_only and in_contact:
            continue
        for name in site_names:
            pos = _site_world_pos(env, str(name))
            if pos is None:
                continue
            terrain_h = _sample_terrain_height(env, float(pos[0]), float(pos[1]))
            if terrain_h is None:
                continue
            clearance = float(pos[2]) - float(terrain_h)
            deficit = max(0.0, min_clearance - clearance)
            cost = (deficit / sigma) ** 2
            costs.append(cost)
            per_site.append({
                "side": side,
                "site": str(name),
                "clearance": float(clearance),
                "deficit": float(deficit),
                "cost": float(cost),
            })

    mean_cost = float(np.mean(costs)) if costs else 0.0
    return mean_cost, {
        "cost": mean_cost,
        "min_clearance": float(min_clearance),
        "sigma": float(sigma),
        "swing_only": swing_only,
        "right_contact": right_contact,
        "left_contact": left_contact,
        "sites": per_site,
    }


def get_terrain_reward(env: Any, comps: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    weights = getattr(env, "terrain_weights", {}) or {}
    w_pelvis = float(weights.get("pelvis_height", 0.0))
    w_clearance = float(weights.get("foot_clearance", 0.0))

    total = 0.0
    info: Dict[str, Any] = {}

    if w_pelvis != 0.0:
        r_pelvis, pelvis_info = _terrain_pelvis_height_reward(env, comps)
        total += w_pelvis * float(r_pelvis)
        info["pelvis_height"] = {"weight": w_pelvis, "reward": float(r_pelvis), **pelvis_info}
    else:
        info["pelvis_height"] = {"skipped": True, "weight": w_pelvis}

    if w_clearance != 0.0:
        clearance_cost, clearance_info = _terrain_foot_clearance_cost(env, comps)
        total -= w_clearance * float(clearance_cost)
        info["foot_clearance"] = {
            "weight": w_clearance,
            "cost": float(clearance_cost),
            **clearance_info,
        }
    else:
        info["foot_clearance"] = {"skipped": True, "weight": w_clearance}

    return float(total), info

def _compute_energy_cost(env: Any) -> Tuple[float, Dict[str, float]]:
    """
    Compute the muscle-energy penalty when muscle actuators are enabled.

    Uses the Mujoco activation buffer (`env.data.act`) and evaluates:
        raw = 0.5 * ||a||₁ + ||a||₂²
        scaled = 1e-3 * raw
    """
    if not bool(getattr(env, "_has_muscle_actuators", False)):
        return 0.0, {}

    act = env.data.act
    if act.size == 0:
        return 0.0, {}

    l2_sq = float(np.dot(act, act))
    l1 = float(np.sum(np.abs(act)))

    raw = 0.5 * l1 + l2_sq
    scaled = 1e-3 * raw
    
    vols = getattr(env, "muscle_volumes", None)
    vols = (vols[:act.size] if (isinstance(vols, np.ndarray) and vols.size >= act.size) else None)
    if isinstance(vols, np.ndarray) and vols.shape == act.shape:
        metab_cost = float(np.einsum('i,i,i->', vols, act, act))
    else:
        metab_cost = 0.0

    return scaled, {"l1": l1, "l2_sq": l2_sq, "metab_cost": metab_cost}


def _compute_synergy_penalty(env: Any) -> Tuple[float, Dict[str, float]]:
    """
    Compute synergy/co-contraction penalties for major lower-limb muscle groups.
    """
    if not bool(getattr(env, "_has_muscle_actuators", False)):
        return 0.0, {}

    act = env.data.act[:80]
    volumes = getattr(env, "muscle_volumes", None)
    if volumes is None or volumes.size < 80:
        return 0.0, {}
    vol = volumes[:80]

    means: Dict[str, float] = {}
    variance_total = 0.0

    for name, idx in _SYNERGY_GROUPS.items():
        w = vol[idx]
        a = act[idx]
        w_sum = float(w.sum())
        if w_sum <= _EPSILON:
            means[name] = 0.0
            continue
        mean = float(np.dot(w, a) / w_sum)
        var = float(np.dot(w, a * a) / w_sum - mean * mean)
        if var < 0.0:
            var = 0.0
        variance_total += var
        means[name] = mean

    cocon_total = 0.0
    for g_a, g_b in _ANTAGONIST_PAIRS:
        mean_a = means[g_a]
        mean_b = means[g_b]
        denom = mean_a + mean_b
        if denom <= _EPSILON:
            continue
        cocon_total += 2.0 * min(mean_a, mean_b) / denom

    total = variance_total + cocon_total
    return total, {"variance": variance_total, "co_contraction": cocon_total, "mean": means}


def _compute_jerk_cost(env: Any) -> Tuple[float, Dict[str, float]]:
    """
    Return jerk penalty using the precomputed second difference buffer.
    """
    buf = getattr(env, "_jerk_buf", None)
    if buf is None or buf.size == 0 or int(getattr(env, "_jerk_have_prev", 0)) < 2:
        return 0.0, {}

    cost = float(np.dot(buf, buf)) * _JERK_COST_SCALE
    return cost, {"l2_sq": cost}


def get_smooth_reward(env: Any) -> Tuple[float, Dict[str, Any]]:
    """
    计算平滑性惩罚项，当前拆分为 energy / synergy / jerk 三部分。
    """
    weights = getattr(env, "smooth_weights", {}) or {}

    w_energy = float(weights.get("energy", 0.0))
    w_synergy = float(weights.get("synergy", 0.0))
    w_jerk = float(weights.get("jerk", 0.0))

    need_energy = (w_energy != 0.0)
    need_synergy = (w_synergy != 0.0)
    need_jerk = (w_jerk != 0.0)

    total = 0.0
    breakdown: Dict[str, Any] = {}

    # Energy term
    if need_energy:
        e_cost, e_info = _compute_energy_cost(env)
        total += w_energy * float(e_cost)
        breakdown["energy"] = {"weight": w_energy, "cost": float(e_cost), "metrics": e_info}
    else:
        breakdown["energy"] = {"skipped": True, "weight": w_energy}

    # Synergy term
    if need_synergy:
        s_cost, s_info = _compute_synergy_penalty(env)
        total += w_synergy * float(s_cost)
        breakdown["synergy"] = {"weight": w_synergy, "cost": float(s_cost), "metrics": s_info}
    else:
        breakdown["synergy"] = {"skipped": True, "weight": w_synergy}

    # Jerk term
    if need_jerk:
        j_cost, j_info = _compute_jerk_cost(env)
        total += w_jerk * float(j_cost)
        breakdown["jerk"] = {"weight": w_jerk, "cost": float(j_cost), "metrics": j_info}
    else:
        breakdown["jerk"] = {"skipped": True, "weight": w_jerk}

    return float(total), breakdown
