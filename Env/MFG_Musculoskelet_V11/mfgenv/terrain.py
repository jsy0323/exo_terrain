from __future__ import annotations

from typing import Iterable

import numpy as np

STRUCTURED_TERRAIN_KINDS = ("flat", "slope", "stairs", "rough")
_STRUCTURED_TERRAIN_KIND_ALIASES = {
    "level": "flat",
    "plane": "flat",
    "stair": "stairs",
    "step": "stairs",
    "steps": "stairs",
    "roughness": "rough",
}


def normalize_structured_kinds(kinds: Iterable[str] | str | None) -> tuple[str, ...]:
    if kinds is None:
        return STRUCTURED_TERRAIN_KINDS

    if isinstance(kinds, str):
        raw = kinds.strip().lower()
        if raw in {"", "all", "any", "mixed", "none", "null"}:
            return STRUCTURED_TERRAIN_KINDS
        raw_kinds = [part for part in raw.replace(",", " ").split() if part]
    else:
        raw_kinds = [str(kind).strip().lower() for kind in kinds if str(kind).strip()]

    normalized = []
    for kind in raw_kinds:
        kind = _STRUCTURED_TERRAIN_KIND_ALIASES.get(kind, kind)
        if kind not in STRUCTURED_TERRAIN_KINDS:
            allowed = ", ".join(STRUCTURED_TERRAIN_KINDS)
            raise ValueError(f"Unsupported terrain_structured_kinds entry: {kind!r}; allowed: {allowed}")
        if kind not in normalized:
            normalized.append(kind)

    if not normalized:
        raise ValueError("terrain_structured_kinds must not be empty")
    return tuple(normalized)


def apply_terrain_config_overrides(base_cfg: dict, terrain_cfg: dict | None) -> None:
    if not terrain_cfg:
        return
    if not isinstance(terrain_cfg, dict):
        raise TypeError(f"terrain config overrides must be a dict, got {type(terrain_cfg).__name__}")

    for key, value in terrain_cfg.items():
        config_key = str(key)
        if not config_key.startswith("terrain_"):
            config_key = f"terrain_{config_key}"
        base_cfg[config_key] = value


def _hfield_id(model, hfield_name: str):
    import mujoco

    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
    if hid < 0:
        raise ValueError(f"MuJoCo hfield not found: {hfield_name}")
    return hid


def _geom_id(model, geom_name: str):
    import mujoco

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"MuJoCo geom not found: {geom_name}")
    return gid


def get_hfield_heightmap_m(model, hfield_name: str = "random_terrain") -> np.ndarray:
    """Return the full hfield heightmap in meters as (nrow, ncol)."""
    hid = _hfield_id(model, hfield_name)
    adr = int(model.hfield_adr[hid])
    nrow = int(model.hfield_nrow[hid])
    ncol = int(model.hfield_ncol[hid])
    z_scale = float(model.hfield_size[hid][2])

    h_norm = model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol)
    return np.asarray(h_norm * z_scale, dtype=np.float32)


def smooth_heightmap(height: np.ndarray, passes: int) -> np.ndarray:
    out = height.astype(np.float32, copy=True)
    for _ in range(max(0, int(passes))):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 5.0
    return out


def apply_flat_start(
    height_norm: np.ndarray,
    x_half: float,
    flat_start_length: float,
    transition_length: float,
    flat_height_norm: float = 0.0,
) -> np.ndarray:
    if flat_start_length <= -x_half:
        return height_norm

    nrow, ncol = height_norm.shape
    x = np.linspace(-x_half, x_half, ncol, dtype=np.float32)
    transition = max(float(transition_length), 1e-6)
    alpha = np.clip((x - float(flat_start_length)) / transition, 0.0, 1.0).reshape(1, ncol)
    return float(flat_height_norm) * (1.0 - alpha) + height_norm * alpha


def _sample_range(
    rng: np.random.Generator,
    value_range: tuple[float, float],
) -> float:
    low, high = sorted(float(v) for v in value_range)
    return float(rng.uniform(low, high))


def _vertical_direction(
    rng: np.random.Generator,
    current_height: float,
    max_height: float,
) -> float:
    if current_height <= max_height * 0.2:
        return 1.0
    if current_height >= max_height * 0.8:
        return -1.0
    return float(rng.choice((-1.0, 1.0)))


def build_structured_heightmap_m(
    nrow: int,
    ncol: int,
    x_half: float,
    height_scale: float,
    rng: np.random.Generator | None = None,
    flat_start_length: float = 2.0,
    flat_start_height: float = 0.0,
    segment_length_range: tuple[float, float] = (1.0, 3.0),
    slope_range_deg: tuple[float, float] = (3.0, 8.0),
    step_height_range: tuple[float, float] = (0.03, 0.08),
    step_run_range: tuple[float, float] = (0.3, 0.6),
    roughness_range: tuple[float, float] = (0.01, 0.04),
    structured_kinds: Iterable[str] | str | None = None,
) -> np.ndarray:
    """Build a forward curriculum terrain from selected structured terrain segments."""
    rng = rng if rng is not None else np.random.default_rng()
    allowed_kinds = normalize_structured_kinds(structured_kinds)
    max_height = max(float(height_scale), 1e-6)
    current_height = float(np.clip(flat_start_height, 0.0, max_height))
    x = np.linspace(-float(x_half), float(x_half), int(ncol), dtype=np.float32)
    profile = np.full(int(ncol), current_height, dtype=np.float32)
    cursor = float(flat_start_length)
    segment_kinds = []

    while cursor < float(x_half):
        if not segment_kinds:
            if len(allowed_kinds) == 1:
                segment_kinds = [allowed_kinds[0]]
            else:
                segment_kinds = list(rng.permutation(allowed_kinds))
        kind = segment_kinds.pop()
        length = _sample_range(rng, segment_length_range)
        end = min(cursor + length, float(x_half))
        mask = (x >= cursor) & (x < end if end < float(x_half) else x <= end)
        local_x = x[mask] - cursor
        if not np.any(mask):
            cursor = end
            continue

        if kind == "flat":
            values = np.full(local_x.shape, current_height, dtype=np.float32)
        elif kind == "slope":
            direction = _vertical_direction(rng, current_height, max_height)
            slope = np.tan(np.deg2rad(_sample_range(rng, slope_range_deg)))
            values = current_height + direction * slope * local_x
        elif kind == "stairs":
            direction = _vertical_direction(rng, current_height, max_height)
            step_height = _sample_range(rng, step_height_range)
            step_run = max(_sample_range(rng, step_run_range), 1e-6)
            values = current_height + direction * step_height * np.floor(local_x / step_run)
        else:
            amplitude = _sample_range(rng, roughness_range)
            noise = rng.uniform(-amplitude, amplitude, size=local_x.shape).astype(np.float32)
            noise = smooth_heightmap(noise.reshape(1, -1), passes=2).reshape(-1)
            taper = np.sin(np.linspace(0.0, np.pi, local_x.size, dtype=np.float32))
            values = current_height + noise * taper

        values = np.clip(values, 0.0, max_height).astype(np.float32, copy=False)
        profile[mask] = values
        current_height = float(values[-1])
        cursor = end

    return np.repeat(profile.reshape(1, -1), int(nrow), axis=0)


def randomize_hfield(
    model,
    hfield_name: str = "random_terrain",
    rng: np.random.Generator | None = None,
    height_scale: float | None = None,
    smooth_passes: int = 8,
    flat_start_length: float = 2.0,
    flat_transition_length: float = 1.0,
    flat_start_height: float = 0.0,
    mode: str = "random",
    structured_segment_length_range: tuple[float, float] = (1.0, 3.0),
    structured_slope_range_deg: tuple[float, float] = (3.0, 8.0),
    structured_step_height_range: tuple[float, float] = (0.03, 0.08),
    structured_step_run_range: tuple[float, float] = (0.3, 0.6),
    structured_roughness_range: tuple[float, float] = (0.01, 0.04),
    structured_kinds: Iterable[str] | str | None = None,
) -> np.ndarray:
    """Randomize MuJoCo hfield data in-place and return the normalized heightmap."""
    rng = rng if rng is not None else np.random.default_rng()
    hid = _hfield_id(model, hfield_name)
    adr = int(model.hfield_adr[hid])
    nrow = int(model.hfield_nrow[hid])
    ncol = int(model.hfield_ncol[hid])
    x_half = float(model.hfield_size[hid][0])
    z_scale = float(model.hfield_size[hid][2])

    if mode == "structured":
        structured_height_scale = float(height_scale) if height_scale is not None else z_scale
        height_m = build_structured_heightmap_m(
            nrow=nrow,
            ncol=ncol,
            x_half=x_half,
            height_scale=min(structured_height_scale, z_scale),
            rng=rng,
            flat_start_length=flat_start_length,
            flat_start_height=flat_start_height,
            segment_length_range=structured_segment_length_range,
            slope_range_deg=structured_slope_range_deg,
            step_height_range=structured_step_height_range,
            step_run_range=structured_step_run_range,
            roughness_range=structured_roughness_range,
            structured_kinds=structured_kinds,
        )
        height = height_m / max(z_scale, 1e-8)
    elif mode == "random":
        height = rng.uniform(0.0, 1.0, size=(nrow, ncol)).astype(np.float32)
        height = smooth_heightmap(height, smooth_passes)
        height -= float(height.min())
        peak = float(height.max())
        if peak > 1e-8:
            height /= peak

        if height_scale is not None and z_scale > 1e-8:
            height *= float(height_scale) / z_scale
            height = np.clip(height, 0.0, 1.0)

        flat_height_norm = float(flat_start_height) / z_scale if z_scale > 1e-8 else 0.0
        height = apply_flat_start(
            height,
            x_half=x_half,
            flat_start_length=flat_start_length,
            transition_length=flat_transition_length,
            flat_height_norm=flat_height_norm,
        )
    else:
        raise ValueError(f"Unsupported terrain generation mode: {mode}")

    height = np.clip(height, 0.0, 1.0).astype(np.float32, copy=False)
    model.hfield_data[adr : adr + nrow * ncol] = height.reshape(-1)
    return height


def sample_hfield_height_m(
    model,
    data,
    x: float,
    y: float,
    hfield_name: str = "random_terrain",
    geom_name: str = "floor",
) -> float:
    """Sample hfield surface height in world meters at world x/y."""
    hid = _hfield_id(model, hfield_name)
    gid = _geom_id(model, geom_name)

    adr = int(model.hfield_adr[hid])
    nrow = int(model.hfield_nrow[hid])
    ncol = int(model.hfield_ncol[hid])
    x_half, y_half, z_scale, _ = model.hfield_size[hid]
    cx, cy, cz = data.geom_xpos[gid]

    height = model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol)

    u = (float(x) - float(cx) + float(x_half)) / (2.0 * float(x_half))
    v = (float(y) - float(cy) + float(y_half)) / (2.0 * float(y_half))
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))

    col = u * (ncol - 1)
    row = v * (nrow - 1)
    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, nrow - 1)
    c1 = min(c0 + 1, ncol - 1)
    dr = row - r0
    dc = col - c0

    h_norm = (
        height[r0, c0] * (1.0 - dr) * (1.0 - dc)
        + height[r0, c1] * (1.0 - dr) * dc
        + height[r1, c0] * dr * (1.0 - dc)
        + height[r1, c1] * dr * dc
    )
    return float(cz + h_norm * float(z_scale))


def get_local_height_patch_m(
    env,
    grid_x: Iterable[float] = (-0.5, 0.0, 0.5, 1.0, 1.5),
    grid_y: Iterable[float] = (-0.4, -0.2, 0.0, 0.2, 0.4),
    hfield_name: str = "random_terrain",
    geom_name: str = "floor",
    relative_to_center: bool = True,
) -> np.ndarray:
    """Sample a pelvis-heading-aligned patch flattened as lateral rows by forward columns."""
    pelvis_id = env.model.body("pelvis").id
    origin = np.asarray(env.data.xpos[pelvis_id], dtype=np.float64)

    forward = np.asarray(env.pelvis_heading[:2], dtype=np.float64)
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        forward = np.array([1.0, 0.0], dtype=np.float64)
    else:
        forward = forward / norm
    left = np.array([-forward[1], forward[0]], dtype=np.float64)

    center_h = 0.0
    if relative_to_center:
        center_h = sample_hfield_height_m(
            env.model,
            env.data,
            float(origin[0]),
            float(origin[1]),
            hfield_name=hfield_name,
            geom_name=geom_name,
        )

    values = []
    for dy in grid_y:
        for dx in grid_x:
            xy = origin[:2] + float(dx) * forward + float(dy) * left
            h = sample_hfield_height_m(
                env.model,
                env.data,
                float(xy[0]),
                float(xy[1]),
                hfield_name=hfield_name,
                geom_name=geom_name,
            )
            values.append(h - center_h if relative_to_center else h)

    return np.asarray(values, dtype=np.float32)
