"""
Termination conditions for the musculoskeletal simulation environment.
Determines if the episode should end based on conditions such as falling,
COM Y deviation, excessive pelvis/torso angle deviations and high contact forces.
@author: YAKE
"""

import math
import logging
from typing import Any, Dict, Tuple, List, Optional

# Configure module-level logger
logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def check_termination(
        env: Any, 
        comps: Dict[str, Any],
        conditions: List[str] = None
        ) -> Tuple[bool, Dict[str, bool]]:
    """
    Evaluate episode termination conditions and return a boolean flag plus details.

    Parameters
    ----------
    env : Any
        Environment object providing:
          - site_threshold : float (optional), used by the site-deviation check.
          - logger (optional) with .error(msg, *args) for error reporting.
        Also expected to be compatible with helper functions:
          - _has_fallen(env) -> bool
          - _site_deviation_exceeded(env, comps, threshold: float) -> (bool, dict)
    comps : Dict[str, Any]
        Observation components bundle (typically from `get_state`), forwarded
        to the site-deviation checker.
    conditions : list[str], optional
        Names of termination checks to run. Available keys:
          - "has_fallen"
          - "site_deviation_exceeded"
        If None, all available checks are evaluated. Passing an unknown name
        raises ValueError.

    Returns
    -------
    terminated : bool
        True if any requested condition evaluates to True (i.e., termination
        should occur), False otherwise.
    details : dict
        Per-condition results:
          - For "has_fallen": bool
          - For "site_deviation_exceeded": dict with diagnostic info if exceeded,
            otherwise False
        If a condition raises an exception, it is logged (when logger present)
        and its entry is set to True (fail-safe terminate).

    Notes
    -----
    - The threshold for "site_deviation_exceeded" defaults to
      `float(getattr(env, "site_threshold", 0.15))`.
    - Any non-boolean detail value is cast to bool when computing `terminated`.
    """
    _fall = _has_fallen
    _site = _site_deviation_exceeded
    # 范围
    available = {
        "has_fallen": lambda: _fall(env),
        "site_deviation_exceeded": lambda: _site(
            env, comps, threshold=float(getattr(env, "site_threshold", 10)) #0.3
        ),
    }
    if conditions is None:
        conditions = list(available.keys())

    invalid = set(conditions) - set(available.keys())
    if invalid:
        raise ValueError(f"Invalid termination conditions: {invalid}")

    details: Dict[str, Any] = {}
    for cond in conditions:
        try:
            result = available[cond]()
            if cond == "site_deviation_exceeded":
                exceeded, info = result
                details[cond] = info if exceeded else False
            else:
                details[cond] = bool(result)
        except Exception as e:
            logger = getattr(env, "logger", None)
            if logger is not None:
                logger.error("Error evaluating '%s': %s", cond, e)
            details[cond] = True
        
    terminated = any((v if isinstance(v, bool) else bool(v)) for v in details.values())
    return terminated, details

def _has_fallen(
        env: Any,
        pelvis_height_range: Optional[Tuple[float, float]] = None,
        ) -> bool:
    """
    Check whether the simulated agent has fallen, based on pelvis height.

    The height limits come from ``env.alive_reward`` unless an explicit range
    is supplied.  When ``use_terrain_relative_height`` is enabled, the terrain
    surface below the pelvis is subtracted from its world-frame height.  This
    prevents a standing agent from being classified as fallen simply because
    it has walked uphill above the old absolute world-z limit.

    Parameters
    ----------
    env : Any
        Environment object providing:
          - model.body("pelvis").id : int
              Numeric ID of the pelvis body.
          - data.xpos : np.ndarray, shape (nbody, 3)
              World-frame positions of all bodies; pelvis height is read
              from xpos[pelvis_id, 2].
          - logger (optional) with `.error(msg, *args)` for error reporting.
    pelvis_height_range : tuple of float, optional
        Explicit acceptable pelvis height range. If omitted, read
        ``env.alive_reward['pelvis_height_range']`` and fall back to
        ``(0.6, 1.3)``.

    Returns
    -------
    fallen : bool
        True if pelvis height is below the minimum or above the maximum of
        `pelvis_height_range`, or if an error occurs. False otherwise.

    Notes
    -----
    - The pelvis body id is cached on the first call as `env._pelvis_body_id`
      to avoid repeated model lookups.
    - Terrain-relative height is used only when requested and an hfield exists.
    - In case of exceptions (e.g., missing pelvis body), the function logs
      an error (if a logger is available) and returns True as a fail-safe.
    """
    try:
        params = getattr(env, "alive_reward", {}) or {}
        if pelvis_height_range is None:
            pelvis_height_range = params.get("pelvis_height_range", (0.6, 1.3))
        if not isinstance(pelvis_height_range, (list, tuple)) or len(pelvis_height_range) != 2:
            pelvis_height_range = (0.6, 1.3)

        bid = getattr(env, "_pelvis_body_id", None)
        if bid is None:
            bid = env.model.body("pelvis").id
            env._pelvis_body_id = bid

        pelvis_pos = env.data.xpos[bid]
        z = float(pelvis_pos[2])
        use_relative = bool(params.get("use_terrain_relative_height", False))
        if use_relative and int(getattr(env.model, "nhfield", 0)) > 0:
            from .terrain import sample_hfield_height_m

            terrain_height = sample_hfield_height_m(
                env.model,
                env.data,
                float(pelvis_pos[0]),
                float(pelvis_pos[1]),
                hfield_name=str(getattr(env, "terrain_hfield_name", "random_terrain")),
                geom_name=str(getattr(env, "terrain_floor_geom_name", "floor")),
            )
            z -= float(terrain_height)

        min_z, max_z = (float(pelvis_height_range[0]), float(pelvis_height_range[1]))
        if (z < min_z) or (z > max_z):
            return True
        return False
    except Exception as e:
        logger = getattr(env, "logger", None)
        if logger is not None:
            logger.error("has_fallen error: %s", e)
        return True

# _SCALE2_BY_NAME: Dict[str, float] = {
#     "femur_r_sensor": 0.70**2,
#     "tibia_r_sensor": 0.75**2,
#     "talus_r_sensor": 0.80**2,
#     "calcn_r_sensor": 0.85**2,
#     "toes_r_sensor":  0.90**2,

#     "femur_l_sensor": 0.70**2,
#     "tibia_l_sensor": 0.75**2,
#     "talus_l_sensor": 0.80**2,
#     "calcn_l_sensor": 0.85**2,
#     "toes_l_sensor":  0.90**2,
    
#     "lumbar_sensor":  0.70**2,
#     "thorax_sensor":  0.75**2,
#     "neck_sensor":    0.80**2,
#     "head_sensor":    0.85**2,
    
#     "humerus_r_sensor": 0.80**2,
#     "ulna_r_sensor":    1.00**2,
#     "hand_r_sensor":    1.20**2,
    
#     "humerus_l_sensor": 0.80**2,
#     "ulna_l_sensor":    1.00**2,
#     "hand_l_sensor":    1.20**2,
# }

# _PELVIS_SCALE2 = 0.60**2

def _site_deviation_exceeded(
        env: Any,
        comps: Dict[str, Any],
        threshold: float = 0.15
        ) -> Tuple[bool, Dict[str, float]]:
    """
    Check whether any tracked site (including pelvis) deviates from its reference
    position beyond a given L2 threshold.

    Parameters
    ----------
    env : Any
        Environment object. Uses:
          - relative_pelvis : bool
              If True, pelvis deviation is measured as the reference pelvis
              position magnitude in the local frame (sim pelvis treated as origin).
          - logger (optional) with `.error(msg, *args)` for error reporting.
    comps : Dict[str, Any]
        Components dictionary from `get_state`, with structure:
          {
            "curr": {"sites": {"pos": {name: (3,)}}, "pelvis": {"pos": (3,)}},
            "ref":  {"sites": {"pos": {name: (3,)}}, "pelvis": {"pos": (3,)}}
          }
        Only the "pos" maps are used here.
    threshold : float, default=0.15
        Maximum allowed Euclidean distance (meters) between current and reference
        positions for each site. Values strictly greater than this trigger
        an exceedance.

    Returns
    -------
    exceeded_any : bool
        True if any site (or the pelvis) exceeds `threshold`; False otherwise.
    exceeded : Dict[str, float]
        Mapping of site name → Euclidean deviation (meters) for those exceeding
        the threshold. Empty dict if nothing exceeded.

    Notes
    -----
    - Pelvis handling:
        * If `relative_pelvis=True`, the simulated pelvis is treated as the
          local origin; the check uses ‖ref_pelvis_pos‖.
        * Otherwise, the pelvis check uses ‖curr_pelvis_pos - ref_pelvis_pos‖.
    - Missing reference entries are skipped for non-pelvis sites.
    - On exception, logs the error (if a logger is available) and returns
      `(True, {})` to fail-safe terminate.
    """
    try:
        curr = comps["curr"]; ref = comps["ref"]
        c_sites, r_sites = curr["sites"], ref["sites"]
        c_pel,   r_pel   = curr["pelvis"], ref["pelvis"]

        c_pos = c_sites["pos"]
        r_pos = r_sites["pos"]

        rel = bool(getattr(env, "relative_pelvis", False))
        thr2_base = float(threshold) * float(threshold)
        pel_thr2 = thr2_base
        # pel_thr2 = thr2_base * _PELVIS_SCALE2

        exceeded: Dict[str, float] = {}
        if rel:
            rx, ry, rz = float(r_pel["pos"][0]), float(r_pel["pos"][1]), float(r_pel["pos"][2])
            dsq = rx*rx + ry*ry + rz*rz
        else:
            dx = float(c_pel["pos"][0]) - float(r_pel["pos"][0])
            dy = float(c_pel["pos"][1]) - float(r_pel["pos"][1])
            dz = float(c_pel["pos"][2]) - float(r_pel["pos"][2])
            dsq = dx*dx + dy*dy + dz*dz
        if dsq > pel_thr2:
            exceeded["pelvis"] = float(math.sqrt(dsq))
            
        for name, v_sim in c_pos.items():
            if name == "pelvis":
                continue
            v_ref = r_pos.get(name, None)
            if v_ref is None:
                continue

            dx = float(v_sim[0]) - float(v_ref[0])
            dy = float(v_sim[1]) - float(v_ref[1])
            dz = float(v_sim[2]) - float(v_ref[2])
            dsq = dx*dx + dy*dy + dz*dz
            
            thr2 = thr2_base
            # s2 = _SCALE2_BY_NAME.get(name, 1.0)
            # thr2 = thr2_base * s2
            if dsq > thr2:
                exceeded[name] = float(math.sqrt(dsq))

        return (len(exceeded) > 0), exceeded

    except Exception as e:
        logger = getattr(env, "logger", None)
        if logger is not None:
            logger.error("site_deviation_exceeded_fast_direct error: %s", e)
        return True, {}
    
