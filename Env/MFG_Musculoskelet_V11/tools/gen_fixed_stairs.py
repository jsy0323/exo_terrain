"""Generate deterministic ascending stairs for the exoskeleton model."""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_random_terrain import ENV_DIR, write_height_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step-height', type=float, default=0.06)
    parser.add_argument('--step-run', type=float, default=0.8)
    parser.add_argument('--num-steps', type=int, default=40)
    parser.add_argument('--flat-start-length', type=float, default=0.5)
    args = parser.parse_args()
    if args.step_height <= 0 or args.step_run <= 0 or args.num_steps <= 0:
        parser.error('Step height, run, and count must be positive.')
    if args.flat_start_length < 0 or args.flat_start_length + (args.num_steps - 1) * args.step_run >= 50:
        parser.error('Stairs must start at nonnegative x and fit within x < 50 m.')

    # 2 cm horizontal sampling keeps hfield riser interpolation short.
    nrow, ncol = 101, 5001
    x = np.linspace(-50.0, 50.0, ncol)
    count = np.where(x >= args.flat_start_length,
                     np.floor((x - args.flat_start_length) / args.step_run + 1e-9) + 1, 0)
    profile = np.clip(count, 0, args.num_steps) * args.step_height
    height = np.repeat(profile[None, :], nrow, axis=0).astype(np.float32)
    max_height = args.step_height * args.num_steps
    meta = dict(height_scale=max_height, step_height=args.step_height,
                step_run=args.step_run, num_steps=args.num_steps,
                first_riser_x=args.flat_start_length, grid_spacing_m=0.02,
                plateau_start_x=args.flat_start_length + (args.num_steps - 1) * args.step_run)
    png, _, _ = write_height_assets(height, 'fixed_stairs', meta)
    tree = ET.parse(ENV_DIR / 'MFG_Raj2015_V11_randomterrainexo.xml',
                    parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    hfield = tree.getroot().find("asset/hfield[@name='random_terrain']")
    hfield.set('file', png.relative_to(ENV_DIR).as_posix())
    hfield.set('nrow', str(nrow))
    hfield.set('ncol', str(ncol))
    hfield.set('size', f'50 5 {max_height:g} 0.02')
    target = ENV_DIR / 'MFG_Raj2015_V11_fixed_stairs_exo.xml'
    tree.write(target, encoding='utf-8', xml_declaration=True)
    print(target)


if __name__ == '__main__':
    main()
