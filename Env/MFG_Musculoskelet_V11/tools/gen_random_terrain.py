"""
Generate a MuJoCo hfield terrain and a matching XML model variant.

Outputs by default:
    Env/MFG_Musculoskelet_V11/terrain/random_terrain.png
    Env/MFG_Musculoskelet_V11/terrain/random_terrain.npy
    Env/MFG_Musculoskelet_V11/terrain/random_terrain_meta.json
    Env/MFG_Musculoskelet_V11/MFG_Raj2015_V11_randomterrain.xml

Use ``--mode random`` for the original smooth random height field, or
``--mode structured`` to bake the same flat/slope/stairs/rough generator used
by the runtime terrain randomizer into a fixed hfield file.

The PNG is normalized for MuJoCo's hfield loader. The NPY stores the same height
map in meters so downstream code can extract terrain information without
reverse-engineering image values.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np


ENV_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ENV_DIR.parent.parent
TERRAIN_DIR = ENV_DIR / "terrain"
BASE_XML = ENV_DIR / "MFG_Raj2015_V11.xml"
OUT_XML = ENV_DIR / "MFG_Raj2015_V11_randomterrain.xml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Env.MFG_Musculoskelet_V11.mfgenv.terrain import (  # noqa: E402
    build_structured_heightmap_m,
    normalize_structured_kinds,
)


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


def flatten_spawn_area(
    height: np.ndarray,
    x_half: float,
    y_half: float,
    radius: float,
) -> np.ndarray:
    if radius <= 0:
        return height

    nrow, ncol = height.shape
    x = np.linspace(-x_half, x_half, ncol, dtype=np.float32)
    y = np.linspace(-y_half, y_half, nrow, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx * xx + yy * yy)

    flat_h = float(height[nrow // 2, ncol // 2])
    blend_width = max(radius * 0.5, 1e-6)
    alpha = np.clip((dist - radius) / blend_width, 0.0, 1.0)
    return flat_h * (1.0 - alpha) + height * alpha


def flatten_start_region(
    height_m: np.ndarray,
    x_half: float,
    flat_until_x: float,
    transition_length: float,
    flat_height: float,
) -> np.ndarray:
    """Keep the initial walking region flat, then blend into random terrain."""
    if flat_until_x <= -x_half:
        return height_m

    nrow, ncol = height_m.shape
    x = np.linspace(-x_half, x_half, ncol, dtype=np.float32)
    transition = max(float(transition_length), 1e-6)
    alpha = np.clip((x - float(flat_until_x)) / transition, 0.0, 1.0)
    alpha = alpha.reshape(1, ncol)
    return float(flat_height) * (1.0 - alpha) + height_m * alpha


def build_heightmap(
    nrow: int,
    ncol: int,
    height_scale: float,
    smooth_passes: int,
    seed: int | None,
    x_half: float,
    y_half: float,
    flat_radius: float,
    flat_start_length: float,
    flat_transition_length: float,
    flat_start_height: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height = rng.uniform(0.0, 1.0, size=(nrow, ncol)).astype(np.float32)
    height = smooth_heightmap(height, smooth_passes)
    height = flatten_spawn_area(height, x_half, y_half, flat_radius)

    height -= float(height.min())
    peak = float(height.max())
    if peak > 1e-8:
        height /= peak
    height_m = height * float(height_scale)
    height_m = flatten_start_region(
        height_m,
        x_half=x_half,
        flat_until_x=float(flat_start_length),
        transition_length=flat_transition_length,
        flat_height=flat_start_height,
    )
    return height_m.astype(np.float32, copy=False)


def _structured_kinds_arg(values: list[str] | None) -> list[str] | str | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def build_structured_heightmap(args: argparse.Namespace) -> tuple[np.ndarray, tuple[str, ...]]:
    structured_kinds = normalize_structured_kinds(_structured_kinds_arg(args.structured_kinds))
    height_m = build_structured_heightmap_m(
        nrow=args.nrow,
        ncol=args.ncol,
        x_half=args.x_half,
        height_scale=args.height_scale,
        rng=np.random.default_rng(args.seed),
        flat_start_length=args.flat_start_length,
        flat_start_height=args.flat_start_height,
        segment_length_range=tuple(args.structured_segment_length_range),
        slope_range_deg=tuple(args.structured_slope_range_deg),
        step_height_range=tuple(args.structured_step_height_range),
        step_run_range=tuple(args.structured_step_run_range),
        roughness_range=tuple(args.structured_roughness_range),
        structured_kinds=structured_kinds,
    )
    return height_m.astype(np.float32, copy=False), structured_kinds


def write_height_assets(height_m: np.ndarray, name: str, meta: dict) -> tuple[Path, Path, Path]:
    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)

    png_path = TERRAIN_DIR / f"{name}.png"
    npy_path = TERRAIN_DIR / f"{name}.npy"
    meta_path = TERRAIN_DIR / f"{name}_meta.json"

    h_norm = height_m / max(float(meta["height_scale"]), 1e-8)
    h_u16 = np.clip(np.round(h_norm * 65535.0), 0, 65535).astype(np.uint16)

    write_gray16_png(png_path, h_u16)
    np.save(npy_path, height_m.astype(np.float32))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return png_path, npy_path, meta_path


def write_gray16_png(path: Path, image: np.ndarray) -> None:
    """Write a 16-bit grayscale PNG without optional image dependencies."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    img = np.asarray(image, dtype=np.uint16)
    height, width = img.shape
    img_be = img.astype(">u2", copy=False)
    rows = b"".join(b"\x00" + img_be[r].tobytes() for r in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag)
        crc = zlib.crc32(data, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    content = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(content)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "\t"
        for child in elem:
            indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def write_xml_variant(
    base_xml: Path,
    out_xml: Path,
    terrain_png: Path,
    nrow: int,
    ncol: int,
    x_half: float,
    y_half: float,
    height_scale: float,
    base_thickness: float,
) -> None:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(base_xml, parser=parser)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    for hfield in list(asset.findall("hfield")):
        if hfield.get("name") == "random_terrain":
            asset.remove(hfield)

    rel_png = terrain_png.relative_to(ENV_DIR).as_posix()
    hfield = ET.Element(
        "hfield",
        {
            "name": "random_terrain",
            "file": rel_png,
            "nrow": str(nrow),
            "ncol": str(ncol),
            "size": f"{x_half:g} {y_half:g} {height_scale:g} {base_thickness:g}",
        },
    )
    asset.append(hfield)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"{base_xml} has no <worldbody> section.")

    floor = None
    for geom in worldbody.findall("geom"):
        if geom.get("name") == "floor":
            floor = geom
            break
    if floor is None:
        floor = ET.Element("geom")
        worldbody.insert(0, floor)

    floor.attrib.clear()
    floor.attrib.update(
        {
            "name": "floor",
            "type": "hfield",
            "hfield": "random_terrain",
            "pos": "0 0 0",
            "material": "MatPlane",
            "condim": "3",
            "friction": "1 0.05 0.05",
        }
    )

    root.set("model", "MFG_ZXY_V11_randomterrain")
    indent_xml(root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("random", "structured"), default="random")
    parser.add_argument("--name", default="random_terrain")
    parser.add_argument("--nrow", type=int, default=101)
    parser.add_argument("--ncol", type=int, default=1001)
    parser.add_argument("--x-half", type=float, default=50.0)
    parser.add_argument("--y-half", type=float, default=5.0)
    parser.add_argument("--height-scale", type=float, default=1)
    parser.add_argument("--base-thickness", type=float, default=0.02)
    parser.add_argument("--smooth-passes", type=int, default=8)
    parser.add_argument("--flat-radius", type=float, default=0.0)
    parser.add_argument("--flat-start-length", type=float, default=2.0)
    parser.add_argument("--flat-transition-length", type=float, default=1.0)
    parser.add_argument("--flat-start-height", type=float, default=0.0)
    parser.add_argument(
        "--structured-kinds",
        nargs="*",
        default=None,
        help="Structured terrain kinds: flat slope stairs rough. Empty/all/mixed uses all.",
    )
    parser.add_argument("--structured-segment-length-range", nargs=2, type=float, default=(1.0, 3.0))
    parser.add_argument("--structured-slope-range-deg", nargs=2, type=float, default=(5.0, 7.0))
    parser.add_argument("--structured-step-height-range", nargs=2, type=float, default=(0.05, 0.1))
    parser.add_argument("--structured-step-run-range", nargs=2, type=float, default=(1.0, 2.0))
    parser.add_argument("--structured-roughness-range", nargs=2, type=float, default=(0.01, 0.04))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-xml", type=Path, default=BASE_XML)
    parser.add_argument("--out-xml", type=Path, default=OUT_XML)
    parser.add_argument("--backup-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structured_kinds = None
    if args.mode == "structured":
        height_m, structured_kinds = build_structured_heightmap(args)
    else:
        height_m = build_heightmap(
            nrow=args.nrow,
            ncol=args.ncol,
            height_scale=args.height_scale,
            smooth_passes=args.smooth_passes,
            seed=args.seed,
            x_half=args.x_half,
            y_half=args.y_half,
            flat_radius=args.flat_radius,
            flat_start_length=args.flat_start_length,
            flat_transition_length=args.flat_transition_length,
            flat_start_height=args.flat_start_height,
        )

    meta = {
        "name": args.name,
        "mode": args.mode,
        "seed": args.seed,
        "nrow": args.nrow,
        "ncol": args.ncol,
        "x_half": args.x_half,
        "y_half": args.y_half,
        "height_scale": args.height_scale,
        "base_thickness": args.base_thickness,
        "smooth_passes": args.smooth_passes,
        "flat_radius": args.flat_radius,
        "flat_start_length": args.flat_start_length,
        "flat_transition_length": args.flat_transition_length,
        "flat_start_height": args.flat_start_height,
        "height_min_m": float(height_m.min()),
        "height_max_m": float(height_m.max()),
        "height_mean_m": float(height_m.mean()),
    }
    if args.mode == "structured":
        meta.update(
            {
                "structured_kinds": list(structured_kinds),
                "structured_segment_length_range": list(args.structured_segment_length_range),
                "structured_slope_range_deg": list(args.structured_slope_range_deg),
                "structured_step_height_range": list(args.structured_step_height_range),
                "structured_step_run_range": list(args.structured_step_run_range),
                "structured_roughness_range": list(args.structured_roughness_range),
            }
        )
    png_path, npy_path, meta_path = write_height_assets(height_m, args.name, meta)

    if args.backup_existing and args.out_xml.exists():
        shutil.copy2(args.out_xml, args.out_xml.with_suffix(args.out_xml.suffix + ".bak"))

    write_xml_variant(
        base_xml=args.base_xml,
        out_xml=args.out_xml,
        terrain_png=png_path,
        nrow=args.nrow,
        ncol=args.ncol,
        x_half=args.x_half,
        y_half=args.y_half,
        height_scale=args.height_scale,
        base_thickness=args.base_thickness,
    )

    print(f"Wrote {png_path}")
    print(f"Wrote {npy_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {args.out_xml}")


if __name__ == "__main__":
    main()
