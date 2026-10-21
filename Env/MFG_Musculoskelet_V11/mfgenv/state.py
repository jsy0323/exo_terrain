"""
State-related utilities for musculoskeletal simulation environment.
Handles state extraction, normalization, and statistics collection.
@author: YAKE
"""

import numpy as np
import mujoco

from .common_utils import (
    convert_ref_traj_qpos,
    convert_ref_traj_qvel,
    )
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

# Configure module-level logger
logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

_ZERO3 = np.array((0.0, 0.0, 0.0), dtype=np.float64)
_ZERO18 = np.zeros(18, dtype=np.float64)
_I3    = np.eye(3, dtype=np.float64)
_OFFS3 = np.array([0, 1, 2], dtype=np.int64)
_FORCE_SLICES  = (slice(3, 6),  slice(12, 15))
_TORQUE_SLICES = (slice(6, 9),  slice(15, 18))

def _clip32(a: np.ndarray | None, eps: float = 1e-8) -> np.ndarray | None:
    """Inplace zeroing of near-zero values, returning float32 view."""
    if a is None: 
        return None
    out = a.astype(np.float32, copy=False)
    small = np.abs(out) < eps
    out[small] = 0.0
    return out

def _next_pow2(x: int) -> int:
    """
    Compute the next power of two greater than or equal to x.
    
    Parameters
    ----------
    x : int
        Input value. If x <= 0, it is clamped to 1.
    
    Returns
    -------
    int
        The smallest power of two ≥ x.
   """
    x = int(max(1, x))
    return 1 << (x - 1).bit_length()

# quaternion multip
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion multiplication: c = a * b (w,x,y,z)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    c = np.empty(4, dtype=np.float64)
    c[0] = aw*bw - ax*bx - ay*by - az*bz
    c[1] = aw*bx + ax*bw + ay*bz - az*by
    c[2] = aw*by - ax*bz + ay*bw + az*bx
    c[3] = aw*bz + ax*by - ay*bx + az*bw
    return c

# quaternion inver
def _qinv(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (conjugate), assuming unit quaternion."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    out[0] = q[0]
    out[1] = -q[1]
    out[2] = -q[2]
    out[3] = -q[3]
    return out

def _Rw2l_from_heading(x) -> np.ndarray:
    """
    Construct a 3×3 world→local rotation matrix from a horizontal heading vector.

    The resulting local frame is:
      - +X : forward, aligned with the given heading projected in the world XY plane
      - +Y : left, a +90° rotation of +X in the XY plane
      - +Z : up, identical to world +Z

    The matrix is arranged with **rows equal to local axes expressed in world
    coordinates**, so multiplying a world-space vector on the left,
    `Rw2l @ v_world`, yields its coordinates in the local (pelvis-yaw) frame.

    Parameters
    ----------
    x : array-like of length ≥ 2
        Horizontal heading. Only `x[0]` (world X) and `x[1]` (world Y) are used.
        The vector need not be normalized.

    Returns
    -------
    Rw2l : np.ndarray, shape (3, 3), dtype=float64
        World→local rotation matrix with rows:
          [x̂_x, x̂_y, 0],
          [ŷ_x, ŷ_y, 0],
          [  0 ,  0 , 1]
        where x̂ is the normalized heading in XY and ŷ is its left-normal.
    """
    x0 = float(x[0]); x1 = float(x[1])
    y0 = -x1; y1 = x0
    ny2 = y0*y0 + y1*y1

    R = np.empty((3, 3), dtype=np.float64)
    if ny2 < 1e-6:
        # fallback to identity frame
        R[0,0] = 1.0; R[0,1] = 0.0; R[0,2] = 0.0
        R[1,0] = 0.0; R[1,1] = 1.0; R[1,2] = 0.0
        R[2,0] = 0.0; R[2,1] = 0.0; R[2,2] = 1.0
        return R

    inv = 1.0 / np.sqrt(ny2)
    y0 *= inv; y1 *= inv
    R[0,0] =  y1; R[0,1] = -y0; R[0,2] = 0.0
    R[1,0] =  y0; R[1,1] =  y1; R[1,2] = 0.0
    R[2,0] =  0.0; R[2,1] =  0.0; R[2,2] = 1.0
    return R

def _prepare_grf_ctx(env: Any) -> Dict[str, Any]:
    """
    Prepare zero-copy views and scratch buffers for GRF extraction.

    This helper collects active contact views from MuJoCo (`data.contact`),
    sets up an optional pelvis-aligned world→local transform, and ensures
    reusable scratch buffers sized to current/maximum contacts.

    Performance notes
    -----------------
    - Uses direct views into MuJoCo arrays (no dtype casts / copies).
    - Caches pelvis id and R_world_to_local_pelvis on `env` when needed.
    - Scratch buffers grow by powers of two and are reused across calls.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.model, env.data (MuJoCo objects)
          - env.relative_pelvis : bool (optional)
          - env.pelvis_heading  : (3,) ndarray (optional; used when relative)
          - env._pelvis_bid     : int (optional cache)
          - env.R_world_to_local_pelvis : (3,3) ndarray (optional cache)
          - env._grf_scratch    : dict (optional cache)

        Module-level constants expected (if world frame is used):
          - _ZERO3 : np.ndarray shape (3,), zeros (float64)
          - _I3    : np.ndarray shape (3,3), identity (float64)

    Returns
    -------
    ctx : dict
        {
          "ncon"    : int,
          "frames"  : np.ndarray (ncon,3,3),   # view
          "g1"      : np.ndarray (ncon,),      # view
          "g2"      : np.ndarray (ncon,),      # view
          "pos_all" : np.ndarray (ncon,3),     # view
          "origin"  : np.ndarray (3,),         # view or _ZERO3
          "Rw2l"    : np.ndarray (3,3),        # cached or _I3
          "scratch" : dict { "tmp6":(6,), "forces":(cap,3), "positions":(cap,3), "cap":int }
        }

    Raises
    ------
    ValueError
        If contact frame array has an unexpected shape.
    """
    data  = env.data
    model = env.model
    c     = data.contact

    # --- active contacts ---
    ncon = int(data.ncon)

    # --- views of contact arrays (no copies) ---
    frames_raw = c.frame[:ncon]
    if frames_raw.ndim == 2 and frames_raw.shape[1] == 9:
        frames = frames_raw.reshape(ncon, 3, 3)            # view
    elif frames_raw.ndim == 3 and frames_raw.shape[1:] == (3, 3):
        frames = frames_raw                                 # view
    else:
        raise ValueError(f"Unexpected contact.frame shape: {frames_raw.shape}")

    g1      = c.geom1[:ncon]                                # view
    g2      = c.geom2[:ncon]                                # view
    pos_all = c.pos[:ncon]                                  # view

    # --- world → local transform (pelvis-aligned if requested) ---
    if getattr(env, "relative_pelvis", False):
        pid = getattr(env, "_pelvis_bid", None)
        if pid is None:
            pid = model.body("pelvis").id
            env._pelvis_bid = pid
        origin = data.xpos[pid]                             # view (3,)

        Rw2l = getattr(env, "R_world_to_local_pelvis", None)
        if not (isinstance(Rw2l, np.ndarray) and Rw2l.shape == (3, 3)):
            x = np.asarray(env.pelvis_heading, dtype=np.float64)
            Rw2l = _Rw2l_from_heading(x)
            env.R_world_to_local_pelvis = Rw2l
    else:
        origin = _ZERO3
        Rw2l   = _I3

    # --- scratch buffers (reuse; grow only when needed) ---
    # Capacity target: max(current contacts, model limit, small floor)
    nconmax   = int(getattr(model, "nconmax", 0)) if hasattr(model, "nconmax") else 0
    need_cap  = max(ncon, nconmax, 64)
    scr       = getattr(env, "_grf_scratch", None)
    ok = isinstance(scr, dict) and \
         isinstance(scr.get("tmp6"), np.ndarray) and scr["tmp6"].shape == (6,)
    if not ok:
        scr = {"tmp6": np.zeros(6, dtype=np.float64)}
    cap = int(scr.get("cap", 0))
    if cap < need_cap or ("forces" not in scr) or ("positions" not in scr):
        cap = _next_pow2(need_cap)
        scr["forces"]    = np.zeros((cap, 3), dtype=np.float64)
        scr["positions"] = np.zeros((cap, 3), dtype=np.float64)
        scr["cap"]       = cap
    env._grf_scratch = scr

    return {
        "ncon":    ncon,
        "frames":  frames,
        "g1":      g1,
        "g2":      g2,
        "pos_all": pos_all,
        "origin":  origin,
        "Rw2l":    Rw2l,
        "scratch": scr,
    }

def _compute_grf(
        body_id: int,
        geom_ids: List[int],
        env: Any,
        *,
        ctx: Optional[Dict[str, Any]] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate ground-reaction force (GRF) for a given body over selected contact geoms.

    Hot-path design choices
    -----------------------
    - Single pass over all active contacts (O(ncon)).
    - Set membership for `geom_ids` (amortized O(1) checks).
    - Scalar accumulators; no per-contact heap arrays.
    - Manual mat-vecs instead of `@` to avoid temporaries.
    - Reuses `ctx` from `_prepare_grf_ctx(env)` (views + tiny scratch).

    Parameters
    ----------
    body_id : int
        MuJoCo body id. Its world position `data.xpos[body_id]` is the torque origin.
    geom_ids : list[int]
        Contact geom ids to include (e.g., all sole/boot geoms of one foot).
    env : Any
        Must provide `env.model` and `env.data` (MuJoCo objects).
    ctx : dict, optional
        Context produced by `_prepare_grf_ctx(env)`. If None, it will be created.

    Returns
    -------
    pos_local : np.ndarray, shape (3,)
        Force-weighted average contact position, expressed in the local frame
        defined by `ctx["Rw2l"]` and `ctx["origin"]`.
    force_local : np.ndarray, shape (3,)
        Net GRF in the local frame.
    torque_local : np.ndarray, shape (3,)
        Net torque about `data.xpos[body_id]`, expressed in the local frame.

    Notes
    -----
    - On no contacts or near-zero net force, returns three zero vectors.
    - Depends on tiny module-level constants `_ZERO3` (shape (3,)) if present.
    """
    if ctx is None:
        ctx = _prepare_grf_ctx(env)

    data, model = env.data, env.model
    ncon    = ctx["ncon"]
    if ncon == 0:
        return _ZERO3, _ZERO3, _ZERO3

    frames  = ctx["frames"]    # (ncon,3,3)
    g1      = ctx["g1"]        # (ncon,)
    g2      = ctx["g2"]        # (ncon,)
    pos_all = ctx["pos_all"]   # (ncon,3)
    origin  = ctx["origin"]    # (3,)
    Rw2l    = ctx["Rw2l"]      # (3,3)
    tmp6    = ctx["scratch"]["tmp6"]  # (6,)
    ref_pos = data.xpos[body_id]

    # Precompute membership set once
    gid_set = set(int(x) for x in geom_ids)

    # Accumulators in world frame
    sf0 = sf1 = sf2 = 0.0                    # sum(F)
    sFP0 = sFP1 = sFP2 = 0.0                 # sum(F cross P)
    tau0 = tau1 = tau2 = 0.0                 # sum( (P-ref) × F )
    hit  = 0

    mj_contactForce = mujoco.mj_contactForce

    for i in range(ncon):
        gi1 = int(g1[i])
        gi2 = int(g2[i])
        if (gi1 not in gid_set) and (gi2 not in gid_set):
            continue

        # Contact force (local) → tmp6[0:3]
        mj_contactForce(model, data, i, tmp6)

        # f_world = frame^T * f_local (manual mat-vec)
        R = frames[i]
        fx_loc, fy_loc, fz_loc = tmp6[0], tmp6[1], tmp6[2]
        fx = R[0,0]*fx_loc + R[1,0]*fy_loc + R[2,0]*fz_loc
        fy = R[0,1]*fx_loc + R[1,1]*fy_loc + R[2,1]*fz_loc
        fz = R[0,2]*fx_loc + R[1,2]*fy_loc + R[2,2]*fz_loc

        px, py, pz = pos_all[i, 0], pos_all[i, 1], pos_all[i, 2]
        rx, ry, rz = (px - ref_pos[0]), (py - ref_pos[1]), (pz - ref_pos[2])

        # sum F
        sf0 += fx; sf1 += fy; sf2 += fz
        # sum(F x P)
        sFP0 += fx * px; sFP1 += fy * py; sFP2 += fz * pz
        # sum[(P - ref) × F]
        tau0 += ry * fz - rz * fy
        tau1 += rz * fx - rx * fz
        tau2 += rx * fy - ry * fx

        hit += 1

    if hit == 0:
        return _ZERO3, _ZERO3, _ZERO3

    # tiny net force, treat as no contact
    if (sf0*sf0 + sf1*sf1 + sf2*sf2) <= 1e-16:
        return _ZERO3, _ZERO3, _ZERO3

    # Force-weighted average contact position
    ax = sFP0 / sf0 if sf0 != 0.0 else 0.0
    ay = sFP1 / sf1 if sf1 != 0.0 else 0.0
    az = sFP2 / sf2 if sf2 != 0.0 else 0.0

    # World → local
    cx, cy, cz = (ax - origin[0]), (ay - origin[1]), (az - origin[2])

    # pos_local = Rw2l @ [cx,cy,cz]
    pl0 = Rw2l[0,0]*cx + Rw2l[0,1]*cy + Rw2l[0,2]*cz
    pl1 = Rw2l[1,0]*cx + Rw2l[1,1]*cy + Rw2l[1,2]*cz
    pl2 = Rw2l[2,0]*cx + Rw2l[2,1]*cy + Rw2l[2,2]*cz

    # force_local = Rw2l @ [sf0,sf1,sf2]
    fl0 = Rw2l[0,0]*sf0 + Rw2l[0,1]*sf1 + Rw2l[0,2]*sf2
    fl1 = Rw2l[1,0]*sf0 + Rw2l[1,1]*sf1 + Rw2l[1,2]*sf2
    fl2 = Rw2l[2,0]*sf0 + Rw2l[2,1]*sf1 + Rw2l[2,2]*sf2

    # torque_local = Rw2l @ [tau0,tau1,tau2]
    tl0 = Rw2l[0,0]*tau0 + Rw2l[0,1]*tau1 + Rw2l[0,2]*tau2
    tl1 = Rw2l[1,0]*tau0 + Rw2l[1,1]*tau1 + Rw2l[1,2]*tau2
    tl2 = Rw2l[2,0]*tau0 + Rw2l[2,1]*tau1 + Rw2l[2,2]*tau2

    # Pack tiny outputs 
    pos_local    = np.array((pl0, pl1, pl2), dtype=np.float64)
    force_local  = np.array((fl0, fl1, fl2), dtype=np.float64)
    torque_local = np.array((tl0, tl1, tl2), dtype=np.float64)
    return pos_local, force_local, torque_local

def get_GRF_info(env: Any) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """
    Aggregate ground-reaction results for both feet with a single flat buffer.

    Hot-path design
    ---------------
    - Early-out returns shared zero buffers (no per-call alloc) when ncon == 0.
    - Caches (once) foot body ids and geom id tuples on `env._grf_foot_cfg`.
    - Prepares a single GRF context via `_prepare_grf_ctx(env)` (views + scratch).
    - Writes per-side outputs directly into one preallocated `grf_vec`:
        layout = [R.pos(3), R.force(3), R.torque(3), L.pos(3), L.force(3), L.torque(3)]
      The nested dict returns **views** into `grf_vec` to avoid tiny-array allocs.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.model, env.data : MuJoCo objects
          - Optional cache slot `env._grf_foot_cfg` populated on first call.

    Returns
    -------
    grf_vec : np.ndarray, shape (18,), float64
        Flat vector concatenating per-foot [pos(3), force(3), torque(3)] in the
        local frame chosen by `_prepare_grf_ctx` (world or pelvis-relative).
    info : dict
        {
          "right": {"GRF_pos": view(3,), "GRF_force": view(3,), "GRF_torque": view(3,)},
          "left" : {"GRF_pos": view(3,), "GRF_force": view(3,), "GRF_torque": view(3,)}
        }
        Each value is a view into `grf_vec` (do not resize or change its dtype).

    Notes
    -----
    - Requires module-level constants `_ZERO3` (shape (3,)) and `_ZERO18` (shape (18,))
      if you rely on the zero early-out path being allocation-free.
    """
    data, model = env.data, env.model

    # Early-out: zero contacts
    if int(data.ncon) == 0:
        grf_vec = _ZERO18
        info = {
            "right": {"GRF_pos": _ZERO3, "GRF_force": _ZERO3, "GRF_torque": _ZERO3},
            "left":  {"GRF_pos": _ZERO3, "GRF_force": _ZERO3, "GRF_torque": _ZERO3},
        }
        return grf_vec, info

    # Cache body ids and geom id tuples once
    cfg = getattr(env, "_grf_foot_cfg", None)
    if cfg is None:
        r_body = model.body("calcn_r").id
        l_body = model.body("calcn_l").id
        # TODO: If any of these names may not exist, adjust here as needed.
        right_geoms = [model.geom(n).id for n in (
            "C_r_foot","C_r_bofoot1","C_r_bofoot2"
        )]
        left_geoms  = [model.geom(n).id for n in (
            "C_l_foot","C_l_bofoot1","C_l_bofoot2"
        )]
        cfg = {
            "right": (r_body, tuple(int(g) for g in right_geoms)),
            "left":  (l_body,  tuple(int(g) for g in left_geoms)),
        }
        setattr(env, "_grf_foot_cfg", cfg)

    # Shared context
    ctx = _prepare_grf_ctx(env)

    # Allocate the single flat output
    # Layout per foot: [pos(3), force(3), torque(3)] → 9; right then left → 18
    grf_vec = np.empty(18, dtype=np.float64)

    # Build dict
    out: Dict[str, Dict[str, np.ndarray]] = {
        "right": {
            "GRF_pos":    grf_vec[ 0: 3],
            "GRF_force":  grf_vec[ 3: 6],
            "GRF_torque": grf_vec[ 6: 9],
        },
        "left": {
            "GRF_pos":    grf_vec[ 9:12],
            "GRF_force":  grf_vec[12:15],
            "GRF_torque": grf_vec[15:18],
        },
    }

    # Fill per side (one pass each).
    (r_body, r_geoms) = cfg["right"]
    pos, force, torque = _compute_grf(r_body, r_geoms, env, ctx=ctx)
    grf_vec[0:3]  = pos
    grf_vec[3:6]  = force
    grf_vec[6:9]  = torque

    (l_body, l_geoms) = cfg["left"]
    pos, force, torque = _compute_grf(l_body, l_geoms, env, ctx=ctx)
    grf_vec[9:12]  = pos
    grf_vec[12:15] = force
    grf_vec[15:18] = torque

    return grf_vec, out

def _normalize_grf(
        grf_info: np.ndarray,
        total_mass: float = 76.0,
        torque_scale: float = 100.0,
        clip_force: bool = False,
        clip_torque: bool = True,
        *,
        inplace: bool = False,
        dtype: Union[np.dtype, str, None] = None
    ) -> np.ndarray:
    """
    Normalize ground reaction force (GRF) vectors for both feet.

    This function rescales force and torque components of the 18-element GRF
    vector while leaving contact positions unchanged. It supports batched inputs,
    clipping, and in-place operation where possible.

    Parameters
    ----------
    grf_info : np.ndarray, shape (18,) or (B,18)
        Raw GRF vector(s) arranged as:
        [pos_r(3), force_r(3), torque_r(3),
         pos_l(3), force_l(3), torque_l(3)].
    total_mass : float, default=76.0
        Body mass (kg). Forces are divided by (mass * 9.81).
    torque_scale : float, default=100.0
        Normalization constant for torques.
    clip_force : bool, default=False
        If True, force components are clipped to [-3, 3] after scaling.
    clip_torque : bool, default=True
        If True, torque components are clipped to [-2, 2] after scaling.
    inplace : bool, default=False
        If True and `dtype` matches input, modifies the array in place.
    dtype : np.dtype or str, optional
        Target dtype for the output (default is input dtype).

    Returns
    -------
    np.ndarray
        Normalized GRF vector(s), same shape as input. If input was 1D,
        returns shape (18,); if input was 2D, returns shape (B,18).

    Raises
    ------
    ValueError
        If `grf_info` has an invalid shape, or if `total_mass` or
        `torque_scale` are non-positive.

    Notes
    -----
    - Position channels are returned unchanged.
    - Scaling is performed slice-by-slice to minimize allocations.
    - NaN/inf values are replaced by 0.0.
    """
    arr = np.asarray(grf_info)

    # --- Normalize shape to (B, 18) ---
    squeeze = False
    if arr.ndim == 1:
        if arr.shape[0] != 18:
            raise ValueError(f"Expected 18 elements, got {arr.shape}")
        view = arr[np.newaxis, :]
        squeeze = True
    elif arr.ndim == 2 and arr.shape[1] == 18:
        view = arr
    else:
        raise ValueError(f"Expected shape (18,) or (B,18), got {arr.shape}")

    if total_mass <= 0.0:
        raise ValueError(f"total_mass must be positive, got {total_mass}")
    if torque_scale <= 0.0:
        raise ValueError(f"torque_scale must be positive, got {torque_scale}")

    # --- Choose output buffer ---
    target_dtype = np.dtype(dtype) if dtype is not None else view.dtype
    if inplace and view.dtype == target_dtype and view.flags.writeable:
        out = view  # true in-place
    else:
        out = view.astype(target_dtype, copy=True)  # single allocation

    # --- Precompute scales ---
    one  = target_dtype.type(1.0)
    g    = target_dtype.type(9.81)
    inv_weight = one / (target_dtype.type(total_mass) * g)
    inv_torque = one / target_dtype.type(torque_scale)

    # --- Scale forces in-place ---
    for s in _FORCE_SLICES:
        out[:, s] *= inv_weight
        if clip_force:
            block = out[:, s]
            np.clip(block, target_dtype.type(-3.0), target_dtype.type(3.0), out=block)

    # --- Scale torques in-place ---
    for s in _TORQUE_SLICES:
        out[:, s] *= inv_torque
        if clip_torque:
            block = out[:, s]
            np.clip(block, target_dtype.type(-2.0), target_dtype.type(2.0), out=block)

    # --- Sanitize numerics in-place ---
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return out[0] if squeeze else out

def compute_foot_contacts(
        grf_comp: Dict[str, Dict[str, np.ndarray]],
        threshold: float,
        *,
        use_vertical: bool = True
        ) -> np.ndarray:
    """
    Compute binary foot-contact flags (right, left) from GRF data.

    Parameters
    ----------
    grf_comp : dict
        Nested GRF dictionary, typically from `get_GRF_info` or `get_GRF_info1`.
        Must contain:
          - grf_comp["right"]["GRF_force"] : np.ndarray (3,)
          - grf_comp["left"]["GRF_force"]  : np.ndarray (3,)
        Force vectors are assumed to be in the chosen local frame.
    threshold : float
        Contact detection threshold. Units depend on normalization:
          - If `use_vertical=True`: compared against absolute vertical force (z).
          - If `use_vertical=False`: compared against squared force magnitude.
    use_vertical : bool, default=True
        If True, detect contact using only the vertical force component.
        If False, detect contact using the full 3D force norm.

    Returns
    -------
    contacts : np.ndarray, shape=(2,), dtype=float32
        Binary flags [right_contact, left_contact], with values 0.0 or 1.0.

    Notes
    -----
    - Designed for efficiency: avoids allocations beyond a tiny (2,) array.
    - When `use_vertical=False`, compares squared norms to avoid a costly sqrt.
    - Suitable as an input feature or for gait phase detection.
    """
    rf = grf_comp["right"]["GRF_force"]
    lf = grf_comp["left"]["GRF_force"]

    out = np.empty(2, dtype=np.float32)

    if use_vertical:
        # Only check z-force
        out[0] = 1.0 if abs(rf[2]) > threshold else 0.0
        out[1] = 1.0 if abs(lf[2]) > threshold else 0.0
    else:
        thr2 = threshold * threshold
        out[0] = 1.0 if (rf[0]*rf[0] + rf[1]*rf[1] + rf[2]*rf[2]) > thr2 else 0.0
        out[1] = 1.0 if (lf[0]*lf[0] + lf[1]*lf[1] + lf[2]*lf[2]) > thr2 else 0.0

    return out

def get_pelvis_kinematics(
        env: Any,
        *,
        add_yaw_features: bool = True,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Retrieve pelvis pose and velocities with an optional pelvis-aligned local frame,
    packaged into a compact state vector suitable for policy inputs.

    Hot-path design
    ---------------
    - Uses direct MuJoCo views for `xpos`, `qpos`, `qvel` (no dtype casts/copies).
    - Caches pelvis body id on `env._pelvis_bid`.
    - If `env.relative_pelvis` is True, uses (or builds once and caches) a yaw-aligned
      world→local rotation `env.R_world_to_local_pelvis`, then applies manual 3×3
      mat–vecs to avoid temporaries.

    State layout
    ------------
    If `add_yaw_features` is True (recommended):
        state = [
          tz,                      # pelvis height (world z)
          qw, qx, qy, qz,          # pelvis orientation (quaternion from qpos[3:7])
          vx, vy, vz,              # linear velocity (world or local, see below)
          wx, wy, wz,              # angular velocity (world or local)
          sin(yaw), cos(yaw),      # from `env.pelvis_heading` (x-axis in world XY)
          yaw_rate                 # z-angular velocity (local if relative_pelvis else world)
        ]  # shape (14,)
    Otherwise (no yaw features): same as above without the last three entries (shape (11,)).

    Frame convention
    ----------------
    - If `env.relative_pelvis` is False: velocities are returned in the world frame.
    - If `env.relative_pelvis` is True:
        * The local frame has x = pelvis forward (from `env.pelvis_heading`), 
          y = +90° left in the world XY plane, z = world up.
        * Linear/angular velocities are rotated into this local frame.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.model, env.data (MuJoCo objects)
          - env.data.qpos (len ≥ 7), env.data.qvel (len ≥ 6)
          - env.pelvis_heading : np.ndarray(3,) (used when relative_pelvis=True)
          - env.relative_pelvis : bool (optional)
          - env._pelvis_bid : int (optional cache)
          - env.R_world_to_local_pelvis : (3,3) ndarray (optional cache)
    add_yaw_features : bool, default=True
        Whether to append (sin_yaw, cos_yaw, yaw_rate) to the state.

    Returns
    -------
    state : np.ndarray, shape (11,) or (14,), dtype=float32
        Packed pelvis features as listed in the layout section.
    components : Dict[str, np.ndarray]
        Views into `state` (where possible) and MuJoCo arrays:
        {
          "pos"      : (3,) world pelvis position (view into MuJoCo),
          "pos_z"    : (1,) view of tz in `state`,
          "orient"   : (4,) view of quaternion in `state`,
          "linvel"   : (3,) view (world or local),
          "angvel"   : (3,) view (world or local),
          ["yaw_sin"]: (1,), ["yaw_cos"]: (1,), ["yaw_rate"]: (1,)  # if enabled
        }

    Raises
    ------
    ValueError
        If `qpos` or `qvel` are shorter than required (len(qpos) < 7 or len(qvel) < 6).

    Notes
    -----
    - Quaternion orientation is taken directly from `qpos[3:7]` (no normalization).
    - Local-frame rotation is cached on first construction to avoid per-step work.
    - Manual 3×3 mat–vecs are used to minimize intermediate allocations.
    """
    data, model = env.data, env.model

    qpos = data.qpos
    qvel = data.qvel
    if qpos.shape[0] < 7 or qvel.shape[0] < 6:
        raise ValueError(f"Expected len(qpos)>=7 and len(qvel)>=6, got {qpos.shape[0]} and {qvel.shape[0]}")

    # Cache pelvis body id.
    pid = getattr(env, "_pelvis_bid", None)
    if pid is None:
        pid = model.body("pelvis").id
        env._pelvis_bid = pid
        
    # World-frame pelvis position and velocities 
    pos_w = data.xpos[pid]         # (3,)
    tz    = pos_w[2]
    v_w   = qvel[0:3]              # (3,)
    w_w   = qvel[3:6]              # (3,)

    # Optional local frame (pelvis-aligned yaw frame).
    use_local = bool(getattr(env, "relative_pelvis", False))
    if use_local:
        Rw2l = getattr(env, "R_world_to_local_pelvis", None)
        if not (isinstance(Rw2l, np.ndarray) and Rw2l.shape == (3, 3)):
            x = np.asarray(env.pelvis_heading, dtype=np.float64)
            Rw2l = _Rw2l_from_heading(x)
            env.R_world_to_local_pelvis = Rw2l
            
        lin0 = Rw2l[0,0]*v_w[0] + Rw2l[0,1]*v_w[1] + Rw2l[0,2]*v_w[2]
        lin1 = Rw2l[1,0]*v_w[0] + Rw2l[1,1]*v_w[1] + Rw2l[1,2]*v_w[2]
        lin2 = Rw2l[2,0]*v_w[0] + Rw2l[2,1]*v_w[1] + Rw2l[2,2]*v_w[2]
        ang0 = Rw2l[0,0]*w_w[0] + Rw2l[0,1]*w_w[1] + Rw2l[0,2]*w_w[2]
        ang1 = Rw2l[1,0]*w_w[0] + Rw2l[1,1]*w_w[1] + Rw2l[1,2]*w_w[2]
        ang2 = Rw2l[2,0]*w_w[0] + Rw2l[2,1]*w_w[1] + Rw2l[2,2]*w_w[2]
    else:
        # World frame
        lin0, lin1, lin2 = v_w[0], v_w[1], v_w[2]
        ang0, ang1, ang2 = w_w[0], w_w[1], w_w[2]

    # orientation
    orient = qpos[3:7]                    # (4,) quaternion
    
    # Preallocate state buffer (+3 if yaw features requested).
    base_dim = 11
    extra_dim = 3 if add_yaw_features else 0
    state = np.empty(base_dim + extra_dim, dtype=np.float32)

    # pack: tz, quat(4), linvel(3), angvel(3)
    state[0]      = tz
    state[1:5]    = orient.astype(np.float32, copy=False)
    state[5:8]    = (np.float32(lin0), np.float32(lin1), np.float32(lin2))
    state[8:11]   = (np.float32(ang0), np.float32(ang1), np.float32(ang2))

    # Yaw features from pelvis heading (cheap & robust).
    if add_yaw_features:
        xh = env.pelvis_heading  # (3,)
        # cos_yaw = xh[0], sin_yaw = xh[1]
        state[11] = np.float32(xh[1])                         # sin_yaw
        state[12] = np.float32(xh[0])                         # cos_yaw
        state[13] = np.float32(ang2 if use_local else w_w[2]) # yaw_rate

    # component dict: return views into `state` when possible (zero extra allocs)
    comp: Dict[str, np.ndarray] = {
        "pos":     pos_w,            # world position (view into MuJoCo)
        "pos_z":   state[0:1],       # tz view
        "orient":  state[1:5],       # quat view
        "linvel":  state[5:8],       # view (possibly local)
        "angvel":  state[8:11],      # view (possibly local)
    }
    if add_yaw_features:
        comp["yaw_sin"]  = state[11:12]
        comp["yaw_cos"]  = state[12:13]
        comp["yaw_rate"] = state[13:14]

    return state, comp

def _ensure_site_cache(env: Any) -> Dict[str, Any]:
    """
    Build and cache fast-access metadata for joint sites.

    This routine materializes `env._joint_sites` into compact NumPy arrays and
    lookup maps to minimize Python overhead in hot paths (e.g., site kinematics
    extraction). It is idempotent: on subsequent calls it returns the same
    cached dictionary without recomputation.

    Expected Input State
    --------------------
    env._joint_sites : list[tuple]
        A list of tuples with the structure:
          (sid, name, (vadr, vdim), (gadr, gdim))
        where:
          - sid  : int, MuJoCo site id
          - name : str, site name
          - vadr : int or None, start index into `sensordata` for a 3D linear-vel block
          - vdim : int or None, length of that block (≥3 when present)
          - gadr : int or None, start index into `sensordata` for a 3D angular-vel block
          - gdim : int or None, length of that block (≥3 when present)

    Parameters
    ----------
    env : Any
        Container with:
          - env.model : mujoco.MjModel
          - env._ensure_joint_site_metadata() : optional callable to populate `_joint_sites`
          - (optional caches) env._pelvis_bid, env._site_cache

    Returns
    -------
    cache : Dict[str, Any]
        A dictionary stored on `env._site_cache` containing:
          - "sids"  : np.ndarray[int32], shape (N,)
          - "names" : tuple[str],        length N (immutable for safety)
          - "vadr"  : np.ndarray[int32], shape (N,), -1 if absent
          - "vdim"  : np.ndarray[int32], shape (N,),  0 if absent
          - "gadr"  : np.ndarray[int32], shape (N,), -1 if absent
          - "gdim"  : np.ndarray[int32], shape (N,),  0 if absent
          - "has_v" : np.ndarray[bool],  shape (N,), (vadr>=0 & vdim>=3)
          - "has_g" : np.ndarray[bool],  shape (N,), (gadr>=0 & gdim>=3)
          - "idx_v" : np.ndarray[int64], indices where `has_v` is True
          - "idx_g" : np.ndarray[int64], indices where `has_g` is True
          - "pelvis_bid" : int or None, pelvis body id if resolvable
          - "pelvis_sid" : int or None, site id for "pelvis_sensor" if present
          - "name_to_index" : dict[str, int], name → row index
          - "name_to_sid"   : dict[str, int], name → site id

    Behavior & Performance
    ----------------------
    - Calls `env._ensure_joint_site_metadata()` once if available to populate
      `env._joint_sites`.
    - Allocates fixed-width arrays (int32 for indices/dims) and fills them in a
      single tight loop.
    - Precomputes boolean availability masks (`has_v`, `has_g`) and their index
      lists (`idx_v`, `idx_g`) to avoid repeated `np.nonzero` calls downstream.
    - Attempts to cache `env._pelvis_bid` by resolving the "pelvis" body id and
      looks up an optional "pelvis_sensor" site; failures are tolerated and stored
      as `None`.

    Raises
    ------
    ValueError
        - If `env._joint_sites` is missing after attempting initialization.
        - If `env._joint_sites` is empty.
    """
    # Fast return if already cached
    cache = getattr(env, "_site_cache", None)
    if cache is not None:
        return cache
    
    # Ensure joint-site metadata exists once
    ensure_meta = getattr(env, "_ensure_joint_site_metadata", None)
    if callable(ensure_meta):
        ensure_meta()
    if not hasattr(env, "_joint_sites"):
        raise ValueError("env._joint_sites not found; call env._ensure_joint_site_metadata() first.")

    js = env._joint_sites  # List[(sid, name, (vadr, vdim), (gadr, gdim))]
    N = len(js)
    if N == 0:
        raise ValueError("env._joint_sites is empty.")
    
    # Preallocate compact arrays
    sids  = np.empty(N, dtype=np.int32)
    vadr  = np.full(N, -1, dtype=np.int32)
    vdim  = np.zeros(N, dtype=np.int32)
    gadr  = np.full(N, -1, dtype=np.int32)
    gdim  = np.zeros(N, dtype=np.int32)
    names = [None] * N
    
    # Tight loop
    for i in range(N):
        sid, name, vel_meta, gyr_meta = js[i]
        sids[i]  = int(sid)
        names[i] = name
        va, vd   = vel_meta
        ga, gd   = gyr_meta
        if va is not None: vadr[i] = int(va)
        if vd is not None: vdim[i] = int(vd)
        if ga is not None: gadr[i] = int(ga)
        if gd is not None: gdim[i] = int(gd)
        
    # Availability masks (vectorized)
    has_v = (vadr >= 0) & (vdim >= 3)   # velocimeter present
    has_g = (gadr >= 0) & (gdim >= 3)   # gyro present
    
    # Precompute nonzero indices to avoid repeated np.nonzero later
    idx_v = np.nonzero(has_v)[0]
    idx_g = np.nonzero(has_g)[0]
    
    model = env.model
    pelvis_bid = getattr(env, "_pelvis_bid", None)
    if pelvis_bid is None:
        try:
            pelvis_bid = model.body("pelvis").id
            env._pelvis_bid = pelvis_bid
        except Exception:
            pelvis_bid = None
    try:
        pelvis_sid = model.site("pelvis_sensor").id
    except Exception:
        pelvis_sid = None

    name_to_index = {names[i]: i for i in range(N)}
    name_to_sid   = {names[i]: int(sids[i]) for i in range(N)}

    cache = {
        "sids": sids,                      # (N,) int32
        "names": tuple(names),             # immutable for safety
        "vadr": vadr,  "vdim": vdim,       # (N,) int32
        "gadr": gadr,  "gdim": gdim,       # (N,) int32
        "has_v": has_v, "has_g": has_g,    # (N,) bool
        "idx_v": idx_v, "idx_g": idx_g,    # precomputed indices
        "pelvis_bid": pelvis_bid,
        "pelvis_sid": pelvis_sid,
        "name_to_index": name_to_index,
        "name_to_sid":   name_to_sid,
    }
    env._site_cache = cache
    return cache

def _ensure_site_components_cache(
        env: Any,
        names: Iterable[str],
        include_orient_angvel: bool,
        ) -> Dict[str, Any]:
    """
    Ensure (create or reuse) compact per-site buffers for kinematic components.

    This helper builds a single cached object on `env._site_comp_cache` that
    contains tightly-sized float32 arrays for positions/velocities (and
    optionally orientation-6D & angular velocity), plus dictionaries that map
    each site name to a zero-copy row view into those arrays. If a cache already
    exists with the same site-name tuple and the same orientation/angvel flag,
    it is returned as-is without reallocations.

    Parameters
    ----------
    env : Any
        Container on which the cache is stored as `env._site_comp_cache`.
    names : Iterable[str]
        Ordered collection of site names. Converted to a tuple and used as the
        cache key to guarantee stable row ordering and reuse.
    include_orient_angvel : bool
        If True, also allocate:
          - orientation in 6D representation per site (shape (N, 6))
          - angular velocity per site (shape (N, 3))

    Returns
    -------
    cache : Dict[str, Any]
        Dictionary with arrays and name→row-view mappings:
        {
          "names"          : tuple[str],            # immutable key for reuse
          "orient_enabled" : bool,                  # equals include_orient_angvel
          "pos"            : np.ndarray, (N, 3)     # float32
          "linvel"         : np.ndarray, (N, 3)     # float32
          "orient6"        : np.ndarray|(None),     # (N, 6) float32 if enabled
          "angvel"         : np.ndarray|(None),     # (N, 3) float32 if enabled
          "dict"           : {
                               "pos"    : {name: pos[i]    (view)},
                               "linvel" : {name: linvel[i] (view)},
                               # present only if enabled:
                               "orient" : {name: orient6[i](view)},
                               "angvel" : {name: angvel[i] (view)},
                             },
        }

    Raises
    ------
    ValueError
        If `names` is empty.

    Notes
    -----
    - Arrays are allocated exactly once per (names tuple, include_orient_angvel)
      configuration and are intended to be updated in-place each step.
    - The dictionaries store row views, not copies, so writing through either the
      arrays or the dict entries updates the same underlying buffers.
    - Using float32 reduces bandwidth while keeping sufficient precision for
      typical kinematics features.
    """
    # Convert names to a tuple.
    names_t = tuple(names)
    orient_flag = bool(include_orient_angvel)
    
    cache = getattr(env, "_site_comp_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("orient_enabled", None) is orient_flag
        and cache.get("names", None) == names_t
        ):
        return cache

    N = len(names_t)
    if N == 0:
        raise ValueError("Empty site name list passed to _ensure_site_components_cache.")
        
    # Allocate tightly-sized float32 buffers
    pos_arr    = np.empty((N, 3), dtype=np.float32)
    linvel_arr = np.empty((N, 3), dtype=np.float32)
    
    # Build dicts as name → row view (zero extra copies)
    pos_dict    = dict(zip(names_t, (pos_arr[i]    for i in range(N))))
    linvel_dict = dict(zip(names_t, (linvel_arr[i] for i in range(N))))
    
    orient_arr = angvel_arr = None
    comp_dict: Dict[str, Dict[str, np.ndarray]] = {
        "pos":    pos_dict,
        "linvel": linvel_dict,
    }
    
    if orient_flag:
        orient_arr = np.empty((N, 6), dtype=np.float32)
        angvel_arr = np.empty((N, 3), dtype=np.float32)
        comp_dict["orient"] = dict(zip(names_t, (orient_arr[i] for i in range(N))))
        comp_dict["angvel"] = dict(zip(names_t, (angvel_arr[i] for i in range(N))))

    cache = {
        "names": names_t,
        "orient_enabled": orient_flag,
        "pos": pos_arr,
        "linvel": linvel_arr,
        "orient6": orient_arr,
        "angvel": angvel_arr,
        "dict": comp_dict,
    }
    env._site_comp_cache = cache
    return cache

def _update_site_components_cache_values(
        cache: Dict[str, Any],
        pos_loc: np.ndarray,
        linvel_loc: np.ndarray,
        orient6: np.ndarray | None,
        angvel_loc: np.ndarray | None,
        ) -> None:
    """
    Update the site-component cache arrays in-place with the latest kinematics.

    Parameters
    ----------
    cache : dict
        Cache object previously built by `_ensure_site_components_cache`. Must
        contain keys:
          - "pos"    : np.ndarray, shape (N,3), float32
          - "linvel" : np.ndarray, shape (N,3), float32
          - "orient6": np.ndarray, shape (N,6), float32 or None
          - "angvel" : np.ndarray, shape (N,3), float32 or None
    pos_loc : np.ndarray, shape (N,3)
        Updated local-frame positions.
    linvel_loc : np.ndarray, shape (N,3)
        Updated local-frame linear velocities.
    orient6 : np.ndarray or None, shape (N,6)
        Updated 6D orientations (row-major [x_axis, y_axis]) if enabled.
        If None, orientation updates are skipped.
    angvel_loc : np.ndarray or None, shape (N,3)
        Updated local-frame angular velocities if enabled.
        If None, angular-velocity updates are skipped.

    Returns
    -------
    None
        Updates are performed in-place. No value is returned.

    Notes
    -----
    - Uses `np.copyto` with casting="no" for fast element-wise assignment
      without implicit dtype conversions.
    - Orientation and angular-velocity arrays are only updated if both the
      cache entry and the provided argument are non-None.
    - This function is designed for hot-path use during each simulation step
      and avoids unnecessary allocations.
    """
    # Always update positions and linear velocities
    np.copyto(cache["pos"], pos_loc, casting="no")
    np.copyto(cache["linvel"], linvel_loc, casting="no")

    # Update orientation and angular velocity only if enabled
    o_arr = cache.get("orient6")
    if o_arr is not None and orient6 is not None:
        np.copyto(o_arr, orient6, casting="no")
    a_arr = cache.get("angvel")
    if a_arr is not None and angvel_loc is not None:
        np.copyto(a_arr, angvel_loc, casting="no")

def get_site_kinematics(
        env: Any,
        include_orient_angvel: bool = False,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute per-site kinematics in either world frame or a pelvis-aligned local frame.

    This function batches MuJoCo site queries to produce a flat feature vector
    (positions, linear velocities, and optionally 6D orientations + angular
    velocities) along with a dictionary of **zero-copy row views** for fast
    downstream access. It relies on `_ensure_site_cache` for site/sensor metadata
    and `_ensure_site_components_cache` for reusable float32 buffers.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.model, env.data : MuJoCo objects
          - env.relative_pelvis : bool (optional)
          - env.use_rigid_velocity_error : bool (optional)
          - env.pelvis_heading : (3,) ndarray (required when relative_pelvis=True)
          - env._pelvis_gyro : tuple[int|None, int] (optional gyroscope spec)
          - `_ensure_site_cache(env)` that exposes:
              * "sids","names","vadr","gadr","has_v","has_g","idx_v","idx_g",
                "pelvis_bid","pelvis_sid"
          - `_ensure_site_components_cache(env, names, include_orient_angvel)`
            to allocate/reuse float32 output buffers.
    include_orient_angvel : bool, default=False
        Whether to compute and return per-site 6D orientation and angular velocity.

    Returns
    -------
    site_state : np.ndarray, dtype=float32
        Flat feature vector with layout:
          if include_orient_angvel == True (per site → 15 dims):
            [pos(3), orient6(6), linvel(3), angvel(3)] × N   → shape (15*N,)
          else (per site → 6 dims):
            [pos(3), linvel(3)] × N                           → shape (6*N,)
    components : Dict[str, Any]
        Dictionaries of name→row-view into cached arrays (float32), enabling
        zero-copy access:
          {
            "pos"    : {name: (3,)},
            "linvel" : {name: (3,)},
            # present only when include_orient_angvel:
            "orient" : {name: (6,)},
            "angvel" : {name: (3,)}
          }

    Raises
    ------
    ValueError
        If required caches are missing or malformed (handled upstream by
        `_ensure_site_cache`), or if internal assumptions on shapes fail.

    Notes
    -----
    - Uses vectorized `einsum` for batched frame transforms and avoids temporary
      allocations via manual packing and `astype(..., copy=False)`.
    - Small numerical noise is suppressed by `_clip32` (values |x| < 1e−8 → 0).
    - The per-step dicts in `components` are stable views into reusable buffers;
      do not change their dtype/shape in-place.
    """
    data = env.data
    
    # ---- Cached site metadata (ids, sensor addresses, etc.) ------------------------
    scache = _ensure_site_cache(env)
    sids   = scache["sids"];  names = scache["names"]
    vadr   = scache["vadr"];  has_v = scache["has_v"]; idx_v = scache.get("idx_v")
    gadr   = scache["gadr"];  has_g = scache["has_g"]; idx_g = scache.get("idx_g")
    pelvis_bid = scache["pelvis_bid"]; pelvis_sid = scache["pelvis_sid"]
    if idx_v is None: idx_v = np.nonzero(has_v)[0]
    if idx_g is None: idx_g = np.nonzero(has_g)[0]
    N = int(sids.size)
    
    # Direct MuJoCo views for all sites
    site_xpos = data.site_xpos[sids]                      # (N,3)
    site_xmat = data.site_xmat.reshape(-1, 3, 3)[sids]    # (N,3,3)
    sens      = data.sensordata                           # (S,)
    
    # ---- Choose frame: world or pelvis-aligned local (yaw frame) ------------------
    use_local = bool(getattr(env, "relative_pelvis", False))
    rigid_vel_err = bool(getattr(env, "use_rigid_velocity_error", False))
    
    if use_local and pelvis_bid is not None:
        origin = data.xpos[pelvis_bid]  # (3,)
        Rw2l = getattr(env, "R_world_to_local_pelvis", None)
        if not (isinstance(Rw2l, np.ndarray) and Rw2l.shape == (3, 3)):
            x = np.asarray(env.pelvis_heading, dtype=np.float64)
            Rw2l = _Rw2l_from_heading(x)
            env.R_world_to_local_pelvis = Rw2l

        vpl_w = data.qvel[0:3]  # (3,)  # pelvis linear vel (world)
        # pelvis angular base vel (world)
        gadr_pel, gdim_pel = getattr(env, "_pelvis_gyro", (None, 0))
        if (pelvis_sid is not None) and (gadr_pel is not None) and (gdim_pel >= 3):
            mat_p = data.site_xmat[pelvis_sid].reshape(3, 3)
            wloc  = sens[gadr_pel:gadr_pel+3]
            vpa_w = mat_p @ wloc
        else:
            vpa_w = data.qvel[3:6]
    else:
        origin = _ZERO3
        Rw2l   = _I3
        vpl_w  = _ZERO3
        vpa_w  = _ZERO3
        
    # ---- World-frame linear / angular velocities per site (from sensors) ----------
    # Linear velocity: site_xmat * local-sensor(3,)
    vw_world = np.zeros((N, 3), dtype=np.float64)
    if idx_v.size > 0:
        v_local = sens[vadr[idx_v, None] + _OFFS3]
        vw_world[idx_v] = np.einsum('nij,nj->ni', site_xmat[idx_v], v_local, optimize=True)

    ww_world = None
    if include_orient_angvel:
        ww_world = np.zeros((N, 3), dtype=np.float64)
        if idx_g.size > 0:
            w_local = sens[gadr[idx_g, None] + _OFFS3]
            ww_world[idx_g] = np.einsum('nij,nj->ni', site_xmat[idx_g], w_local, optimize=True)
    
    # ---- Transform to chosen frame -------------------------------------------------
    # Position
    p_delta    = site_xpos - origin[None, :]                        # (N,3)
    pos_loc    = (Rw2l @ p_delta.T).T                               # (N,3)
    
    # linvel
    if use_local and rigid_vel_err:
        # dv = v_site - v_pelvis - ω_pelvis × (p_site - p_pelvis)
        cross_term = np.cross(vpa_w[None, :], p_delta)                     # (N,3)
        dv = vw_world - vpl_w[None, :] - cross_term
    else:
        dv = vw_world - vpl_w[None, :]
    linvel_loc = (Rw2l @ dv.T).T                       

    orient6 = angvel_loc = None
    if include_orient_angvel:
        R_loc = np.einsum('ij,njk->nik', Rw2l, site_xmat, optimize=True)
        orient6 = R_loc[:, 0:2, :].reshape(N, 6)                             # (N,6)
        dw      = ww_world - (vpa_w[None, :] if use_local else 0.0)
        angvel_loc = (Rw2l @ dw.T).T                                      

    pos_loc     = _clip32(pos_loc)
    linvel_loc  = _clip32(linvel_loc)
    if include_orient_angvel:
        orient6    = _clip32(orient6)
        angvel_loc = _clip32(angvel_loc)

    # ---- Pack flat state (single buffer; no Python list concat) -----------------
    if include_orient_angvel:
        total = (3 + 6 + 3 + 3) * N
        site_state = np.empty(total, dtype=np.float32)
        off = 0
        block = pos_loc.astype(np.float32, copy=False).ravel(order='C')
        site_state[off:off+block.size] = block; off += block.size
        block = orient6.astype(np.float32, copy=False).ravel(order='C')
        site_state[off:off+block.size] = block; off += block.size
        block = linvel_loc.astype(np.float32, copy=False).ravel(order='C')
        site_state[off:off+block.size] = block; off += block.size
        block = angvel_loc.astype(np.float32, copy=False).ravel(order='C')
        site_state[off:off+block.size] = block
    else:
        total = (3 + 3) * N
        site_state = np.empty(total, dtype=np.float32)
        off = 0
        block = pos_loc.astype(np.float32, copy=False).ravel(order='C')
        site_state[off:off+block.size] = block; off += block.size
        block = linvel_loc.astype(np.float32, copy=False).ravel(order='C')
        site_state[off:off+block.size] = block

    # ---- Populate component dict via cached row views (zero-copy) -----------------
    ccache = _ensure_site_components_cache(env, names, include_orient_angvel)
    _update_site_components_cache_values(ccache, pos_loc.astype(np.float32, copy=False),
                                         linvel_loc.astype(np.float32, copy=False),
                                         None if orient6 is None else orient6.astype(np.float32, copy=False),
                                         None if angvel_loc is None else angvel_loc.astype(np.float32, copy=False))

    comp: Dict[str, Any] = {
        'pos':        ccache["dict"]["pos"],
        'linvel':     ccache["dict"]["linvel"],
    }
    if include_orient_angvel:
        comp['orient'] = ccache["dict"]["orient"]
        comp['angvel'] = ccache["dict"]["angvel"]

    return site_state, comp

def _ensure_com_cache(env: Any) -> Dict[str, Any]:
    """
    Ensure that the center-of-mass (COM) metadata is precomputed and cached on the environment.

    This helper computes and stores:
      - body masses (excluding world body 0)
      - total system mass
      - pelvis body ID (if available)

    The result is cached in `env._com_cache` and reused on subsequent calls.

    Parameters
    ----------
    env : Any
        Environment instance providing:
          - env.model.body_mass : ndarray of per-body masses (length ≥ 2)
          - env.model.body(name).id : to resolve pelvis body ID
          - env._pelvis_bid : optional pre-cached pelvis body ID

    Returns
    -------
    cache : dict
        {
          "masses"     : np.ndarray, shape=(nbody-1,), float64
                         Mass of all bodies except world body (index 0).
          "total_mass" : float
                         Total system mass (sum of `masses`).
          "pelvis_bid" : int or None
                         MuJoCo body ID of "pelvis", if resolvable.
        }

    Raises
    ------
    ValueError
        If model.body_mass is malformed or total_mass is non-positive/invalid.

    Notes
    -----
    - This cache is intended to support COM kinematics queries (position/velocity).
    - The `pelvis_bid` is resolved once and stored on both `env._com_cache`
      and `env._pelvis_bid` for reuse.
    - Uses float64 for stability in accumulation, then returns float scalars/arrays.
    """
    cc = getattr(env, "_com_cache", None)
    if cc is not None:
        return cc

    model = env.model
    body_mass = model.body_mass
    if body_mass.ndim != 1 or body_mass.size < 2:
        raise ValueError(f"Unexpected model.body_mass shape: {getattr(model.body_mass, 'shape', None)}")
    
    masses = body_mass[1:]
    total_mass = float(masses.sum())
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError(f"Nonpositive or invalid total mass: {total_mass}")

    pid = getattr(env, "_pelvis_bid", None)
    if pid is None:
        try:
            pid = model.body("pelvis").id
            env._pelvis_bid = pid
        except Exception:
            pid = None

    cc = {
        "masses": masses, 
        "total_mass": total_mass, 
        "pelvis_bid": pid,
        }
    env._com_cache = cc
    return cc

def get_COM_kinematics(env: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute the system Center-of-Mass (COM) position and linear velocity with an
    optional pelvis-aligned local-frame transform, using only zero-copy MuJoCo views
    and tiny on-stack temporaries.

    Pipeline
    --------
    1) World-frame COM:
       - Position from `data.subtree_com[0]` (view, shape (3,)).
       - Linear velocity as mass-weighted average of per-body linear spatial
         velocities: `(masses @ data.cvel[1:, 3:6]) / total_mass`.
       - Masses/total mass/pelvis id are provided by `_ensure_com_cache(env)`.

    2) Optional world→local (pelvis yaw frame):
       - Enabled if `env.relative_pelvis` is True and pelvis id is available.
       - Origin: `data.xpos[pelvis_bid]`.
       - Rotation `Rw2l`:
           * Uses `env.R_world_to_local_pelvis` if present and (3,3).
           * Otherwise, built once from `env.pelvis_heading` (x: forward in world XY,
             y: +90° left in XY, z: world up) and cached back to `env`.
       - Apply to `(COM_pos - origin)` and `(COM_vel - pelvis_base_lin_vel)` where
         pelvis base linear velocity is `data.qvel[0:3]`.

    Parameters
    ----------
    env : Any
        Must provide MuJoCo `env.model`/`env.data`, plus caches set up by
        `_ensure_com_cache(env)`. The following optional attributes are used:
          - env.relative_pelvis : bool
          - env.R_world_to_local_pelvis : np.ndarray (3,3)
          - env.pelvis_heading : np.ndarray (3,)
          - env._pelvis_bid : int (from cache)

    Returns
    -------
    state : np.ndarray, shape (6,), dtype=float32
        [COM_pos_x, COM_pos_y, COM_pos_z, COM_vel_x, COM_vel_y, COM_vel_z]
        in world frame or pelvis-local frame (if enabled). Small values are
        zeroed by `_clip32`.
    components : Dict[str, np.ndarray]
        Views into `state`:
          - "com_pos" : state[0:3]
          - "com_vel" : state[3:6]

    Notes
    -----
    - Uses MuJoCo views wherever possible; no dtype casts on hot path.
    - Only a single small output buffer (`state`) is allocated per call.
    - The pelvis-local frame keeps z aligned with world up (no roll/pitch).
    """
    data = env.data
    cc = _ensure_com_cache(env)
    masses = cc["masses"]             # (nbody-1,)
    total_mass = cc["total_mass"]     # scalar
    pelvis_bid = cc["pelvis_bid"]
    
    # --- COM position/vel in world frame -----------------------------------------------
    com_pos_world = data.subtree_com[0]
    cvel_lin = data.cvel[1:, 3:6]
    com_vel_world = (masses @ cvel_lin) / total_mass

    # --- Optional transform: world -> pelvis-local ---------------------------------
    use_local = bool(getattr(env, "relative_pelvis", False))
    if use_local and pelvis_bid is not None:
        origin = data.xpos[pelvis_bid]          # (3,)
        v_pl_w = data.qvel[0:3]                 # (3,)
        Rw2l = getattr(env, "R_world_to_local_pelvis", None)
        if not (isinstance(Rw2l, np.ndarray) and Rw2l.shape == (3, 3)):
            x = np.asarray(env.pelvis_heading, dtype=np.float64)
            Rw2l = _Rw2l_from_heading(x)
            env.R_world_to_local_pelvis = Rw2l

        # pos_local = Rw2l @ (com_pos_world - origin)
        rx = com_pos_world[0] - origin[0]
        ry = com_pos_world[1] - origin[1]
        rz = com_pos_world[2] - origin[2]
        px = Rw2l[0,0]*rx + Rw2l[0,1]*ry + Rw2l[0,2]*rz
        py = Rw2l[1,0]*rx + Rw2l[1,1]*ry + Rw2l[1,2]*rz
        pz = Rw2l[2,0]*rx + Rw2l[2,1]*ry + Rw2l[2,2]*rz

        # vel_local = Rw2l @ (com_vel_world - v_pl_w)
        vx = com_vel_world[0] - v_pl_w[0]
        vy = com_vel_world[1] - v_pl_w[1]
        vz = com_vel_world[2] - v_pl_w[2]
        lx = Rw2l[0,0]*vx + Rw2l[0,1]*vy + Rw2l[0,2]*vz
        ly = Rw2l[1,0]*vx + Rw2l[1,1]*vy + Rw2l[1,2]*vz
        lz = Rw2l[2,0]*vx + Rw2l[2,1]*vy + Rw2l[2,2]*vz
    else:
        # World frame output
        px, py, pz = com_pos_world[0], com_pos_world[1], com_pos_world[2]
        lx, ly, lz = com_vel_world[0], com_vel_world[1], com_vel_world[2]

    # --- Pack directly to float32 + tiny cleanup in-place ------------------------
    state = np.empty(6, dtype=np.float32)
    state[0] = px; state[1] = py; state[2] = pz
    state[3] = lx; state[4] = ly; state[5] = lz
    state = _clip32(state)

    components: Dict[str, np.ndarray] = {
        "com_pos": state[0:3],  # views into state
        "com_vel": state[3:6],
    }
    return state, components

def get_joint_states(
        env: Any,
        exclude_pelvis: bool = True,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract human joint coordinates and velocities in the original actor order.

    When human_state_map is present, passive interface joints are omitted.
    They are supplied separately as critic-only state by the environment.

    Parameters
    ----------
    env : Any
        Environment instance. Must provide:
          - env.data.qpos : np.ndarray, shape (nq,)
          - env.data.qvel : np.ndarray, shape (nv,)
    exclude_pelvis : bool, default=True
        If True, exclude the first 7 qpos and 6 qvel entries corresponding
        to the pelvis free joint. If False, include them.

    Returns
    -------
    joint_state : np.ndarray, shape=(nq_eff + nv_eff,)
        Concatenated float32 vector of joint positions followed by velocities.
        - nq_eff = nq-7 if exclude_pelvis else nq
        - nv_eff = nv-6 if exclude_pelvis else nv
    comp : dict
        {
          "joint_qpos": np.ndarray, view into first nq_eff entries of joint_state,
          "joint_qvel": np.ndarray, view into last nv_eff entries of joint_state
        }

    Raises
    ------
    ValueError
        If qpos/qvel are shorter than expected given `exclude_pelvis`.

    Notes
    -----
    - Uses np.asarray with dtype float32 and C-contiguous order to ensure
      efficient downstream use.
    - Returns a single flat buffer, with dictionary entries as views into it.
    """
    data = env.data
    qpos = data.qpos
    qvel = data.qvel
    # Keep the original human ordering and observation size for both actors.
    mapping = getattr(env, 'human_state_map', None)
    if mapping is not None:
        qpos = qpos[mapping.qpos_indices]
        qvel = qvel[mapping.qvel_indices]

    qpos0 = 7 if exclude_pelvis else 0
    qvel0 = 6 if exclude_pelvis else 0
    if qpos.size <= qpos0 or qvel.size <= qvel0:
        raise ValueError("qpos/qvel too short to exclude pelvis DOFs.")
    
    nqp = qpos.size - qpos0
    nqv = qvel.size - qvel0
    total = nqp + nqv
    
    joint_state = np.empty(total, dtype=np.float32)
    
    joint_state[:nqp] = np.asarray(qpos[qpos0:], dtype=np.float32, order="C")
    joint_state[nqp:] = np.asarray(qvel[qvel0:], dtype=np.float32, order="C")

    comp = {
        "joint_qpos": joint_state[:nqp],
        "joint_qvel": joint_state[nqp:],
    }
    return joint_state, comp

def compute_ref_pelvis_kinematics(
        env: Any,
        num_future: int = 1,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute pelvis kinematics (position, orientation, velocity) from the
    reference trajectory, optionally relative to the simulated pelvis,
    and return current + future states.

    Parameters
    ----------
    env : Any
        Environment with attributes:
          - ref_traj : ReferenceTrajectories
                Must provide .get_reference_window(num_steps, step_major=True)
          - relative_pelvis : bool
                If True, express reference in pelvis-local yaw frame.
          - use_rigid_velocity_error : bool
                If True, subtract ω × r correction when computing relative velocities.
          - data, model : MuJoCo state/model, to fetch simulated pelvis pose/vel.
          - pelvis_heading : np.ndarray(3,), x-axis direction of pelvis in world.
    num_future : int, default=1
        Number of future steps to include in the output window. Must be >=0.

    Returns
    -------
    state : np.ndarray, shape=(K,14), float32
        K = 1 + num_future reference pelvis states. Each row layout:
          [ tz_abs,
            pos_rel(3) or pos_abs(3),
            quat_rel(4) or quat_abs(4),
            linvel_rel(3) or linvel_abs(3),
            angvel_rel(3) or angvel_abs(3) ]
    components : dict
        {
          "current": {
              "tz_abs" : (1,),
              "pos"    : (3,),
              "orient" : (4,),
              "linvel" : (3,),
              "angvel" : (3,)
          },
          "future": {
              "tz_abs" : (S,),
              "pos"    : (S,3),
              "orient" : (S,4),
              "linvel" : (S,3),
              "angvel" : (S,3)
          }
        }
        If num_future==0, the "future" arrays are empty views.

    Notes
    -----
    - In relative mode, subtracts simulated pelvis pose/vel and rotates into
      pelvis-local yaw frame via `Rw2l`.
    - Quaternion difference is computed as q_loc = q_sim^{-1} ⊗ q_ref.
    - Outputs are clipped to float32 with small values zeroed by `_clip32`.
    - Designed for imitation learning where both current and short-horizon
      future pelvis states are required.
    """
    relative = bool(getattr(env, "relative_pelvis", False))
    rigid_vel_err = bool(getattr(env, "use_rigid_velocity_error", False))
    data, model = env.data, env.model
    
    if relative:
        pid = getattr(env, "_pelvis_bid", None)
        if pid is None:
            pid = model.body("pelvis").id
            env._pelvis_bid = pid
            
        p_sim_w = data.xpos[pid]        # (3,) f64
        q_sim_w = data.xquat[pid]       # (4,) f64
        v_sim_w = data.qvel[0:3]        # (3,) f64
        w_sim_w = data.qvel[3:6]        # (3,) f64
        
        Rw2l = getattr(env, "R_world_to_local_pelvis", None)
        Rw2l = getattr(env, "R_world_to_local_pelvis", None)
        if not (isinstance(Rw2l, np.ndarray) and Rw2l.shape == (3, 3)):
            x = np.asarray(env.pelvis_heading, dtype=np.float64)
            Rw2l = _Rw2l_from_heading(x)
            env.R_world_to_local_pelvis = Rw2l
    else:
        p_sim_w = _ZERO3
        q_sim_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v_sim_w = _ZERO3
        w_sim_w = _ZERO3
        Rw2l    = _I3

    K = 1 + max(0, int(num_future))
    Q, V = env.ref_traj.get_reference_window(num_steps=K, step_major=True)

    state = np.empty((K, 14), dtype=np.float32)
    
    for i in range(K):
        qpos_ref = Q[i]; qvel_ref = V[i]
        d_qpos = convert_ref_traj_qpos(qpos_ref)           # [..., x,y,z, qw,qx,qy,qz, ...]
        d_qvel = convert_ref_traj_qvel(qvel_ref, qpos_ref) # [..., vx,vy,vz, wx,wy,wz, ...]

        p_ref_w = d_qpos[0:3]
        q_ref_w = d_qpos[3:7]
        v_ref_w = d_qvel[0:3]
        w_ref_w = d_qvel[3:6]
        
        row = state[i]
        # tz_abs
        row[0] = np.float32(p_ref_w[2])

        if relative:
            r_w = p_ref_w - p_sim_w
            dv_w = v_ref_w - v_sim_w
            if rigid_vel_err:
                dv_w = dv_w - np.cross(w_sim_w, r_w)
            dw_w = w_ref_w - w_sim_w
            
            p_loc = Rw2l @ r_w;  p_loc = _clip32(p_loc)
            v_loc = Rw2l @ dv_w; v_loc = _clip32(v_loc)
            w_loc = Rw2l @ dw_w; w_loc = _clip32(w_loc)
            q_loc = _qmul(_qinv(q_sim_w), q_ref_w); q_loc = _clip32(q_loc)
            
            row[1:4]   = p_loc.astype(np.float32, copy=False)
            row[4:8]   = q_loc.astype(np.float32, copy=False)
            row[8:11]  = v_loc.astype(np.float32, copy=False)
            row[11:14] = w_loc.astype(np.float32, copy=False)
        else:
            row[1:4]   = p_ref_w.astype(np.float32, copy=False)
            row[4:8]   = q_ref_w.astype(np.float32, copy=False)
            row[8:11]  = v_ref_w.astype(np.float32, copy=False)
            row[11:14] = w_ref_w.astype(np.float32, copy=False)

    cur = {
        "tz_abs": state[0, 0:1],
        "pos":    state[0, 1:4],
        "orient": state[0, 4:8],
        "linvel": state[0, 8:11],
        "angvel": state[0,11:14],
        }
    if K > 1:
        fut = {
            "tz_abs":  state[1:, 0],        # (S,)
            "pos":     state[1:, 1:4],      # (S,3)
            "orient":  state[1:, 4:8],      # (S,4)
            "linvel":  state[1:, 8:11],     # (S,3)
            "angvel":  state[1:,11:14],     # (S,3)
        }
    else:
        fut = {
            "tz_abs":  state[:0, 0],
            "pos":     state[:0, 1:4],
            "orient":  state[:0, 4:8],
            "linvel":  state[:0, 8:11],
            "angvel":  state[:0,11:14],
        }

    components: Dict[str, Dict[str, np.ndarray]] = {"current": cur, "future": fut}
    return state, components

def compute_ref_site_kinematics(
        env: Any,
        num_future: int = 1,
        include_orient_angvel: bool = False,
        ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute per-site kinematics from the reference trajectory over a window
    (current + future), optionally expressed in a pelvis-local yaw frame (ref-traj-based).

    Parameters
    ----------
    env : Any
        Environment providing:
          - model, data : MuJoCo model/state for the simulated pelvis frame.
          - ref_traj.get_reference_window(num_steps, step_major=True)
              → (Q, V) reference qpos/qvel window.
          - relative_pelvis : bool
              If True, express reference site states relative to simulated pelvis.
          - use_rigid_velocity_error : bool
              If True, subtract ω × r when forming relative linear velocity.
          - pelvis_heading : (3,) ndarray (used when relative_pelvis=True).
          - _pelvis_bid, _pelvis_gyro : optional caches for pelvis id/gyro.
    num_future : int, default=1
        Number of future steps; window size K = 1 + max(0, num_future).
    include_orient_angvel : bool, default=False
        If True, also compute 6D orientation (first two local axes) and
        angular velocity for each site.

    Returns
    -------
    out : dict
        {
          "current": {
            "pos"    : {name: (3,) float32},
            "linvel" : {name: (3,) float32},
            ["orient": {name: (6,) float32}],
            ["angvel": {name: (3,) float32}],
          },
          "future": {
            "pos"    : {name: (S,3) float32},
            "linvel" : {name: (S,3) float32},
            ["orient": {name: (S,6) float32}],
            ["angvel": {name: (S,3) float32}],
          }
        }
        where S = max(0, num_future). Values are views into internally
        allocated float32 buffers.

    Notes
    -----
    - Builds/uses a persistent `env._ref_data` (MjData) for reference forward pass.
    - World→local rotation:
        * relative_pelvis=False → identity; results are world-frame.
        * relative_pelvis=True  → yaw-only pelvis.
    - Linear velocity:
        v_rel = (v_site_ref − v_pelvis_sim) [− ω_pelvis_sim × (p_site_ref − p_pelvis_sim) if enabled],
        then rotated by `Rw2l`. Angular velocity is treated analogously if requested.
    - Site linear/angular velocities come from site sensors (vadr/gadr) when available,
      transformed by the site frames; missing sensors yield zeros for those sites.
    """
    model = env.model
    
    # single persistent ref_data
    ref_data = getattr(env, "_ref_data", None)
    if ref_data is None:
        ref_data = mujoco.MjData(model)
        env._ref_data = ref_data
    
    # site meta (precomputed, compact, with indices)
    scache = _ensure_site_cache(env)
    sids   = scache["sids"];  names  = scache["names"]
    vadr   = scache["vadr"];  has_v  = scache["has_v"];  idx_v = scache.get("idx_v")
    gadr   = scache["gadr"];  has_g  = scache["has_g"];  idx_g = scache.get("idx_g")
    pelvis_sid = scache["pelvis_sid"]
    if idx_v is None: idx_v = np.nonzero(has_v)[0]
    if idx_g is None: idx_g = np.nonzero(has_g)[0]
    N = int(sids.size)
    
    # mode switch
    use_local     = bool(getattr(env, "relative_pelvis", False))
    rigid_vel_err = bool(getattr(env, "use_rigid_velocity_error", False))
    do_orient     = bool(include_orient_angvel)
    has_v_any     = (idx_v.size > 0)
    has_g_any     = (idx_g.size > 0) and do_orient
    
    # time window: current(i) + future(S) with S=num_future
    K = 1 + max(0, int(num_future))
    Q, V = env.ref_traj.get_reference_window(num_steps=K, step_major=True)
    
    # Output buffering
    pos_buf = np.empty((K, N, 3), dtype=np.float32)
    lin_buf = np.empty((K, N, 3), dtype=np.float32)
    if do_orient:
        orient_buf = np.empty((K, N, 6), dtype=np.float32)
        ang_buf    = np.empty((K, N, 3), dtype=np.float32)
    else:
        orient_buf = ang_buf = None
        
    # pre-allocate scratch cache
    vw_world = np.empty((N, 3), dtype=np.float64)
    if do_orient:
        ww_world = np.empty((N, 3), dtype=np.float64)  
        
    # main loop for each frame
    for t in range(K):
        # forward on ref_data
        qpos_ref = Q[t]; qvel_ref = V[t]
        d_qpos = convert_ref_traj_qpos(qpos_ref)
        d_qvel = convert_ref_traj_qvel(qvel_ref, qpos_ref)
        d_qpos, d_qvel = env.human_state_map.expand(d_qpos, d_qvel)
        ref_data.qpos[:] = d_qpos
        ref_data.qvel[:] = d_qvel
        mujoco.mj_forward(model, ref_data)
        
        if use_local:
            pid = getattr(env, "_pelvis_bid", None)
            if pid is None:
                pid = model.body("pelvis").id
                env._pelvis_bid = pid

            p_ref_w = ref_data.xpos[pid]                 # (3,)
            v_ref_w = ref_data.qvel[0:3]                 # (3,)
            
            gadr_pel, gdim_pel = getattr(env, "_pelvis_gyro", (None, 0))
            if (pelvis_sid is not None) and (gadr_pel is not None) and (gdim_pel >= 3):
                mat_pel_site = ref_data.site_xmat[pelvis_sid].reshape(3,3)  # local→world
                w_ref_w = mat_pel_site @ ref_data.sensordata[gadr_pel:gadr_pel+3]
            else:
                w_ref_w = ref_data.qvel[3:6]

            xaxis_world = ref_data.xmat[pid].reshape(3,3)[:, 0]
            Rw2l_t = _Rw2l_from_heading(xaxis_world)
        else:
            p_ref_w = v_ref_w = w_ref_w = _ZERO3
            Rw2l_t  = _I3
        
        # Site world quantity (position/linvel/angvel)
        site_xpos = ref_data.site_xpos[sids]                         # (N,3)
        site_xmat = ref_data.site_xmat.reshape(-1,3,3)[sids]         # (N,3,3)
        sens      = ref_data.sensordata
        
        # world linear velocity per site
        if has_v_any:
            v_local = sens[vadr[idx_v, None] + _OFFS3]
            vw_world.fill(0.0)
            vw_world[idx_v] = (site_xmat[idx_v] @ v_local[..., None])[:, :, 0]
        else:
            vw_world.fill(0.0)
            
        # (optional) world angular velocity per site
        if do_orient:
            if has_g_any:
                w_local = sens[gadr[idx_g, None] + _OFFS3]           # (Ng,3)
                ww_world.fill(0.0)
                ww_world[idx_g] = (site_xmat[idx_g] @ w_local[..., None])[:, :, 0]
            else:
                ww_world.fill(0.0)
        
        # to chosen frame
        if use_local:
            p_delta = site_xpos - p_ref_w[None, :]                   # (N,3)
            if rigid_vel_err:
                dv_w = vw_world - v_ref_w[None, :] - np.cross(w_ref_w[None, :], p_delta)
            else:
                dv_w = vw_world - v_ref_w[None, :]
            
            p_loc = (Rw2l_t @ p_delta.T).T
            v_loc = (Rw2l_t @ dv_w.T).T
            if do_orient:
                dw_w  = ww_world - w_ref_w[None, :]
                w_loc = (Rw2l_t @ dw_w.T).T
        else:
            p_loc = site_xpos
            v_loc = vw_world
            if do_orient:
                w_loc = ww_world

        pos_buf[t] = _clip32(p_loc)
        lin_buf[t] = _clip32(v_loc)

        if do_orient:
            r0 = Rw2l_t[0, :]
            r1 = Rw2l_t[1, :]
            row0 = np.tensordot(site_xmat, r0, axes=([1], [0]))
            row1 = np.tensordot(site_xmat, r1, axes=([1], [0]))
        
            orient6 = np.empty((N, 6), dtype=np.float32)
            orient6[:, 0:3] = _clip32(row0)
            orient6[:, 3:6] = _clip32(row1)
            orient_buf[t] = orient6
            ang_buf[t]    = _clip32(w_loc)
    
    # build zero-copy component dicts
    S = K - 1
    # current maps
    current: Dict[str, Dict[str, np.ndarray]] = {
        "pos":    {names[i]: pos_buf[0, i] for i in range(N)},
        "linvel": {names[i]: lin_buf[0, i] for i in range(N)},
    }
    if do_orient:
        current["orient"] = {names[i]: orient_buf[0, i] for i in range(N)}
        current["angvel"] = {names[i]: ang_buf[0, i]    for i in range(N)}
    
    # future maps
    if S > 0:
        future: Dict[str, Dict[str, np.ndarray]] = {
            "pos":    {names[i]: pos_buf[1:, i] for i in range(N)},
            "linvel": {names[i]: lin_buf[1:, i] for i in range(N)},
        }
        if do_orient:
            future["orient"] = {names[i]: orient_buf[1:, i] for i in range(N)}
            future["angvel"] = {names[i]: ang_buf[1:, i]    for i in range(N)}
    else:
        future = {
            "pos":    {names[i]: pos_buf[:0, i] for i in range(N)},
            "linvel": {names[i]: lin_buf[:0, i] for i in range(N)},
        }
        if do_orient:
            future["orient"] = {names[i]: orient_buf[:0, i] for i in range(N)}
            future["angvel"] = {names[i]: ang_buf[:0, i]    for i in range(N)}

    return {"current": current, "future": future}

def build_pos_ref(
        env: Any,
        site_keys: Optional[List[str]] = None,
        num_future: int = 1,
        include_orient_angvel: bool = False,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a flattened reference vector for future pelvis/site kinematics and
    return current reference components for convenience.

    Parameters
    ----------
    env : Any
        Environment providing reference trajectories and site metadata.
        Uses:
          - compute_ref_pelvis_kinematics(env, num_future)
          - compute_ref_site_kinematics(env, num_future, include_orient_angvel)
          - _ensure_site_cache(env) when `site_keys` is None.
    site_keys : list[str] or None, optional
        Site names to include. If None, all sites from `_ensure_site_cache(env)["names"]`
        are used, preserving that order.
    num_future : int, default=1
        Number of future steps S. If S == 0, the returned flat vector is empty.
    include_orient_angvel : bool, default=False
        If True, append per-site 6D orientation and angular velocity blocks.

    Returns
    -------
    flat_ref : np.ndarray, shape=(L,), float32
        Flattened future reference features with layout:
          - Pelvis block (always first): pos(S,3), quat(S,4), linvel(S,3), angvel(S,3)
              → 13 * S values.
          - Site blocks (in `site_keys` order), concatenated by quantity:
              1) all sites' pos(S,3),
              2) all sites' linvel(S,3),
              3) [optional] all sites' orient6(S,6),
              4) [optional] all sites' angvel(S,3).
            Per-site contribution: 6*S (pos+linvel) or 15*S (if `include_orient_angvel`).
            Total length: L = 13*S + n_site * per_site * S.
        If S == 0, returns an empty (0,) array.
    ref_current : dict
        {
          "pelvis": pelv_current_dict,     # from compute_ref_pelvis_kinematics(...)"current"
          "sites" : sites_current_dict     # from compute_ref_site_kinematics(...)"current"
        }
        Values are views into internal float32 buffers suitable for read-only use.

    Notes
    -----
    - This function only flattens the **future** window (S steps). Current-step
      components are returned separately in `ref_current`.
    - `site_keys` order defines the packing order of per-site blocks.
    - May raise a KeyError at runtime if a name in `site_keys` is not present
      in the reference site dictionaries.
    """
    _, pelv = compute_ref_pelvis_kinematics(env, num_future=num_future)
    sites = compute_ref_site_kinematics(env, num_future=num_future, include_orient_angvel=include_orient_angvel)
    
    S = int(max(0, num_future))
    if S == 0:
        return np.zeros((0,), dtype=np.float32), {"pelvis": pelv["current"], "sites": sites["current"]}
    
    if site_keys is None:
        scache = _ensure_site_cache(env)
        site_keys = list(scache["names"])
    n_site = len(site_keys)
    
    p_pos  = pelv["future"]["pos"][:S]      # (S,3)
    p_ori  = pelv["future"]["orient"][:S]   # (S,4)
    p_lin  = pelv["future"]["linvel"][:S]   # (S,3)
    p_ang  = pelv["future"]["angvel"][:S]   # (S,3)
    
    f_pos = sites["future"]["pos"]
    f_lin = sites["future"]["linvel"] 
    if include_orient_angvel:
        f_ori = sites["future"]["orient"]
        f_ang = sites["future"]["angvel"]

    L = 13 * S
    per_site = 6 if not include_orient_angvel else 15
    L += per_site * n_site * S
    out = np.empty((L,), dtype=np.float32)
    
    idx = 0
    idx = 0
    a = p_pos.ravel(order='C');  out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size
    a = p_ori.ravel(order='C');  out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size
    a = p_lin.ravel(order='C');  out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size
    a = p_ang.ravel(order='C');  out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size

    for name in site_keys:
        a = f_pos[name].ravel(order='C')
        out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size

    for name in site_keys:
        a = f_lin[name].ravel(order='C')
        out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size

    if include_orient_angvel:
        for name in site_keys:
            a = f_ori[name].ravel(order='C')
            out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size
        for name in site_keys:
            a = f_ang[name].ravel(order='C')
            out[idx:idx+a.size] = a.astype(np.float32, copy=False); idx += a.size

    ref_current = {"pelvis": pelv["current"], "sites": sites["current"]}
    return out, ref_current

def build_next_joint_ref(env: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build the next-step reference joint state (qpos/qvel tails) from the
    reference trajectory.

    Parameters
    ----------
    env : Any
        Environment instance with:
          - env.ref_traj.get_reference_window(start_steps=1, step_major=True)
            returning (Q, V), arrays of reference qpos/qvel.

    Returns
    -------
    out : np.ndarray, shape=(N,), float32
        Concatenated vector of next-step joint positions and velocities,
        excluding the pelvis DOFs:
          out = [qpos_tail, qvel_tail].
    comp : dict
        {
          "joint_qpos": np.ndarray, view of out[:len(qpos_tail)],
          "joint_qvel": np.ndarray, view of out[len(qpos_tail):],
        }

    Notes
    -----
    - The first 6 DOFs (pelvis free-joint) are skipped from both qpos and qvel.
    - Output is float32, suitable for direct use in RL policies or reward
      functions.
    """
    Q, V = env.ref_traj.get_reference_window(start_steps=1, step_major=True)

    qpos_ref = Q[0]
    qvel_ref = V[0]

    qpos_tail = qpos_ref[6:]
    qvel_tail = qvel_ref[6:]

    total = qpos_tail.size + qvel_tail.size
    out = np.empty(total, dtype=np.float32)
    out[:qpos_tail.size] = qpos_tail.astype(np.float32, copy=False)
    out[qpos_tail.size:] = qvel_tail.astype(np.float32, copy=False)

    comp = {
        "joint_qpos": out[:qpos_tail.size],
        "joint_qvel": out[qpos_tail.size:],
    }
    return out, comp

_DEFAULT_SITE_KEYS = ('toes_r_sensor', 'toes_l_sensor', 'hand_r_sensor', 'hand_l_sensor', 'head_sensor')
def get_state(
        env: Any,
        include_orient_angvel: Optional[bool] = None
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the full observation state vector, concatenating simulated
    kinematics/dynamics and future reference features.

    Parameters
    ----------
    env : Any
        Environment instance providing:
          - get_pelvis_kinematics(env)
          - get_site_kinematics(env, include_orient_angvel)
          - get_GRF_info(env)
          - compute_foot_contacts(grf_comp, threshold)
          - get_COM_kinematics(env)
          - get_joint_states(env)
          - build_pos_ref(env, site_keys, include_orient_angvel)
          - env.contact_threshold : float (optional)
          - env.total_mass : float
    include_orient_angvel : bool or None, default=None
        If None, uses ``env.include_orient_angvel`` when available, otherwise
        falls back to False. If True, include site orientation (6D) and
        angular velocities in site-related features.

    Returns
    -------
    obs : np.ndarray, shape=(M,), float32
        Flattened observation vector concatenating:
          [pelvis_state,
           site_state,
           grf_vec,
           foot_contacts,
           com_state,
           joint_state,
           site_future_ref]
    comp : dict
        {
          "curr": {
              "pelvis": dict from get_pelvis_kinematics,
              "sites" : dict from get_site_kinematics
          },
          "ref": dict from build_pos_ref (current/future reference components)
        }

    Notes
    -----
    - Foot contact threshold = env.contact_threshold × (mass × g).
    - All outputs are float32, except MuJoCo-derived views converted inline.
    - Designed as the main observation builder for RL policies.
    """
    if include_orient_angvel is None:
        include_orient_angvel = bool(getattr(env, "include_orient_angvel", False))

    # --- sim-side ---
    pelv_state, pelv_sim = get_pelvis_kinematics(env)
    site_state, site_sim = get_site_kinematics(env, include_orient_angvel)
    grf_vec, grf_comp    = get_GRF_info(env)
    thr = getattr(env, "contact_threshold", 5e-2) * (env.total_mass * 9.81)
    fc  = compute_foot_contacts(grf_comp, threshold=thr)
    com_state, _         = get_COM_kinematics(env)
    joint_state, _       = get_joint_states(env)

    curr = {'pelvis': pelv_sim, 'sites': site_sim}
    
    # --- ref-side --
    site_future, site_ref = build_pos_ref(
        env, 
        # site_keys=list(_DEFAULT_SITE_KEYS), 
        include_orient_angvel=include_orient_angvel
        )
    
    s0 = int(pelv_state.size)
    s1 = int(site_state.size)
    s2 = int(grf_vec.size)
    s3 = int(fc.size)
    s4 = int(com_state.size)
    s5 = int(joint_state.size)
    s6 = int(site_future.size)
    total = s0 + s1 + s2 + s3 + s4 + s5 + s6
    obs = np.empty(total, dtype=np.float32)

    i = 0
    obs[i:i+s0] = pelv_state; i += s0
    obs[i:i+s1] = site_state; i += s1
    obs[i:i+s2] = grf_vec.astype(np.float32, copy=False); i += s2
    obs[i:i+s3] = fc; i += s3
    obs[i:i+s4] = com_state; i += s4
    obs[i:i+s5] = joint_state; i += s5
    obs[i:i+s6] = site_future; i += s6

    return obs, {'curr': curr, 'ref': site_ref}

def get_state_size(env: Any, include_orient_angvel: Optional[bool] = None) -> int:
    """
    Compute the dimensionality of the full state vector.

    Parameters
    ----------
    env : Any
        MuJoCo environment instance.

    Returns
    -------
    size : int
        Number of elements in the state returned by get_state().
    """
    if include_orient_angvel is None:
        include_orient_angvel = bool(getattr(env, "include_orient_angvel", False))

    cached = getattr(env, "_state_size", None)
    cached_flag = getattr(env, "_state_size_include_orient_angvel", None)
    if cached is not None and cached_flag == bool(include_orient_angvel):
        return int(cached)
    state, _ = get_state(env, include_orient_angvel=include_orient_angvel)
    env._state_size = int(state.size)
    env._state_size_include_orient_angvel = bool(include_orient_angvel)
    return env._state_size

def get_actuator_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_muscle_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_energy_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_terrain_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_state_extend(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_state_extend_size(env: Any) -> int:
    pass
    
# foot slip velocity
