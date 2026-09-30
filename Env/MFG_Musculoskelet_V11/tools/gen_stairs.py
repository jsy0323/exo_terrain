"""
Generate Env/MFG_Musculoskelet_V11/stair.xml from a small parameter set,
without reading any existing XML. Parameters can be given via CLI or fall
back to reasonable defaults.

Geometry model:
- Stair i (1-based) has rise h_i = h0 + (i-1)*delta.
- Its top surface height equals cumulative sum of rises up to i.
- The box half-height is h_i/2, centered at z = cumulative - h_i/2.
- Step centers along +x are spaced by `run`; first center at `start_x`.
- Box size uses MuJoCo half-sizes: (width, depth, h_i/2).

CLI examples:
    python gen_stairs.py                       # use defaults
    python gen_stairs.py --count 12 --height 0.18 --delta 0.01
    python gen_stairs.py --width 0.14 --depth 0.40 --run 0.30 --start-x 0.12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
STAIR_PATH = REPO_ROOT / "stair.xml"


def default_params() -> Dict:
    return {
        "count": 24,
        "height": 0.2,       # first rise
        "delta": 0.0,        # arithmetic increment per step
        "width": 0.135,      # half-size in x
        "depth": 0.435,      # half-size in y
        "run": None,         # center spacing; default set to 2*width
        "start_x": None,     # first center; default set to width
        "body_pos": [0.3, 0.0, 0.0],
        "mass": 0.001,
        "diaginertia": [1e-6, 1e-6, 1e-6],
    }


def build_heights(count: int, h0: float, delta: float) -> List[float]:
    return [h0 + (i * delta) for i in range(count)]


def generate_body(params: Dict) -> str:
    count = params["count"]
    h0 = params["height"]
    delta = params["delta"]
    widths = params["width"]
    depth = params["depth"]
    run = params["run"] if params["run"] is not None else 2 * widths
    start_x = params["start_x"] if params["start_x"] is not None else widths
    body_pos = params["body_pos"]
    mass = params["mass"]
    diaginertia = params["diaginertia"]

    heights = build_heights(count, h0, delta)

    lines = []
    lines.append(
        f'    <body name="stairs" pos="{body_pos[0]} {body_pos[1]} {body_pos[2]}">'
    )
    lines.append(
        f'        <inertial pos="0 0 0" mass="{mass}" diaginertia="{diaginertia[0]:.6f} {diaginertia[1]:.6f} {diaginertia[2]:.6f}"/>'
    )

    cumulative = 0.0
    for idx, h in enumerate(heights, start=1):
        cumulative += h
        hz = h / 2.0
        z_center = cumulative - hz
        x = start_x + (idx - 1) * run
        lines.append(
            f'        <geom name="stair_{idx:02d}" type="box" class="collision_stair" '
            f'pos="{x:.3f} 0 {z_center:.4f}" size="{widths} {depth} {hz:.4f}" density="0"/>'
        )

    lines.append("    </body>")
    return "\n".join(lines), heights


def write_file(params: Dict) -> None:
    body_block, heights = generate_body(params)
    # store generated heights for transparency/debug
    params_to_store = dict(params)
    params_to_store["heights"] = heights
    if params_to_store["run"] is None:
        params_to_store["run"] = 2 * params_to_store["width"]
    if params_to_store["start_x"] is None:
        params_to_store["start_x"] = params_to_store["width"]

    param_block = json.dumps(params_to_store, indent=4)
    content = (
        "<!--\nPARAMS\n"
        f"{param_block}\n"
        "ENDPARAMS\n"
        "修改上面的参数或用命令行参数，然后运行 tools\\gen_stairs.py 生成下面的几何体；不要手改下面的 geom 列表。\n"
        "-->\n"
        "<mujocoinclude>\n"
        f"{body_block}\n"
        "</mujocoinclude>\n"
    )
    STAIR_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, help="number of steps")
    parser.add_argument("--height", type=float, help="height of first step (rise)")
    parser.add_argument("--delta", type=float, help="arithmetic increment per step")
    parser.add_argument("--width", type=float, help="half-size along x (tread depth/2)")
    parser.add_argument("--depth", type=float, help="half-size along y (width/2)")
    parser.add_argument("--run", type=float, help="center spacing along x")
    parser.add_argument("--start-x", type=float, dest="start_x", help="center of first step")
    parser.add_argument("--body-x", type=float, help="stair body position x")
    parser.add_argument("--body-y", type=float, help="stair body position y")
    parser.add_argument("--body-z", type=float, help="stair body position z")
    args = parser.parse_args()

    params = default_params()

    # apply CLI overrides
    if args.count is not None:
        params["count"] = args.count
    if args.height is not None:
        params["height"] = args.height
    if args.delta is not None:
        params["delta"] = args.delta
    if args.width is not None:
        params["width"] = args.width
    if args.depth is not None:
        params["depth"] = args.depth
    if args.run is not None:
        params["run"] = args.run
    if args.start_x is not None:
        params["start_x"] = args.start_x

    # body position overrides
    if any(v is not None for v in (args.body_x, args.body_y, args.body_z)):
        bx, by, bz = params["body_pos"]
        params["body_pos"] = [
            args.body_x if args.body_x is not None else bx,
            args.body_y if args.body_y is not None else by,
            args.body_z if args.body_z is not None else bz,
        ]

    write_file(params)


if __name__ == "__main__":
    main()
