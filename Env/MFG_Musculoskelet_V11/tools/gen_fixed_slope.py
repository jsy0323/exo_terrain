"""Generate a fixed 6-degree uphill followed by a 2 m plateau."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_random_terrain import ENV_DIR, write_height_assets


def main():
    x = np.linspace(-50.0, 50.0, 1001)
    profile = np.clip((x - 0.5) * np.tan(np.deg2rad(6.0)), 0.0, 2.0)
    height = np.repeat(profile[None, :], 101, axis=0).astype(np.float32)
    meta = dict(height_scale=2.0, slope_deg=6.0, flat_until_x=0.5,
                plateau_start_x=float(0.5 + 2.0 / np.tan(np.deg2rad(6.0))))
    png, _, _ = write_height_assets(height, 'fixed_slope_6deg', meta)
    source = ENV_DIR / 'MFG_Raj2015_V11_randomterrainexo.xml'
    tree = ET.parse(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    hfield = tree.getroot().find("asset/hfield[@name='random_terrain']")
    hfield.set('file', png.relative_to(ENV_DIR).as_posix())
    hfield.set('size', '50 5 2 0.02')
    target = ENV_DIR / 'MFG_Raj2015_V11_fixed_slope_6deg_exo.xml'
    tree.write(target, encoding='utf-8', xml_declaration=True)
    print(target)


if __name__ == '__main__':
    main()
