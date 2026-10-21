"""Install sparse human/cuff contact interfaces in the three terrain exo models.

The generator deliberately preserves all existing human MJCF text and actuators.
See ../SOFT_CONTACT.md for the approximation, parameters and load paths.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ENV_DIR = Path(__file__).resolve().parents[1]
MODELS = (
    'MFG_Raj2015_V11_randomterrainexo.xml',
    'MFG_Raj2015_V11_fixed_slope_6deg_exo.xml',
    'MFG_Raj2015_V11_fixed_stairs_exo.xml',
)
PRECOMPRESSION = 0.002  # m; interference fit representing strap tightening
CONTACT_TIME = 0.01    # s; equivalent soft layer, NOT a measured Young's modulus
FRICTION = 0.7
HIP_OFFSET = (0.005, -0.003, 0.0)  # m, exo hip centre relative to human hip


def vec(values):
    return ' '.join(f'{x:.9g}' for x in values)


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def fragment(node):
    ET.indent(node, space='\t')
    return ET.tostring(node, encoding='unicode')


def floating_joints(body, name):
    # No centering spring or damping: normal operation transfers load by contact.
    # Limits are emergency travel stops, which must be monitored during training.
    for kind, limit in (('slide', .03), ('hinge', .25)):
        for axis, direction in zip('xyz', ('1 0 0', '0 1 0', '0 0 1')):
            element(body, 'joint', name=f'interface_{name}_{kind}_{axis}',
                    type=kind, axis=direction, limited='true',
                    range=f'{-limit} {limit}', stiffness='0', damping='0',
                    frictionloss='0', armature='0', solreflimit='.02 1')


def tissue_body(name, pos):
    body = ET.Element('body', name=name, pos=pos)
    # MuJoCo 3.3.2 requires a nonzero COM mass even on fixed geom containers.
    element(body, 'inertial', pos='0 0 0', mass='.001', diaginertia='.001 .001 .001')
    return body


def pads(region, radii, halfwidth, cuff, pairs):
    """Eight spherical tissue patches against eight flat cuff patches.

    Explicit pairs allow parent/child contacts and prevent tissue-ground,
    opposite-leg, muscle-wrap and visual-mesh collisions.
    """
    tissue = ET.Element('mujoco')
    for row, y in enumerate((-halfwidth, halfwidth)):
        for axis, radius in ((0, radii[0]), (2, radii[1])):
            for sign in (-1, 1):
                tag = f'{region}_{row}_{axis}_{sign:+d}'.replace('+', 'p').replace('-', 'm')
                tpos = [0., y, 0.]
                tpos[axis] = sign * radius
                tname, cname = f'interface_tissue_{tag}', f'interface_pad_{tag}'
                element(tissue, 'geom', name=tname, type='sphere',
                        pos=vec(tpos), size='.018', mass='0', group='0',
                        contype='0', conaffinity='0', rgba='.85 .4 .35 .45')
                cpos = tpos.copy()
                cpos[axis] += sign * (.018 + .006 - PRECOMPRESSION)
                size = [.026, .025, .026]
                size[axis] = .006
                element(cuff, 'geom', name=cname, type='box',
                        pos=vec(cpos), size=vec(size), mass='0', group='0',
                        contype='0', conaffinity='0', rgba='.15 .25 .35 .75')
                element(pairs, 'pair', name=f'interface_contact_{tag}',
                        geom1=tname, geom2=cname, condim='3',
                        friction=f'{FRICTION} {FRICTION} .001 .0001 .0001',
                        solref=f'{CONTACT_TIME} 1', solimp='.95 .99 .004 .5 2',
                        margin='0', gap='0')
    return list(tissue)


def build_interfaces():
    pairs = ET.Element('contact')
    inserts = {}
    # Local axes follow the MyoSuite skeleton: x anterior, y proximal, z lateral.
    waist = ET.Element('body', name='interface_pelvis_cuff', pos='-.07 .015 0')
    element(waist, 'inertial', pos='0 0 0', mass='.4', diaginertia='.008 .012 .008')
    floating_joints(waist, 'pelvis')
    tissue = tissue_body('interface_pelvis_tissue', '-.07 .015 0')
    tissue.extend(pads('pelvis', (.105, .14), .025, waist, pairs))
    inserts['pelvis'] = [tissue, waist]
    for side, sign in (('r', 1), ('l', -1)):
        hip = (-.056276, -.07849, sign * .07726)
        # Robot branch is connected to the floating waist, never to human femur.
        exohip = element(waist, 'body', name=f'interface_exo_hip_{side}',
                         pos=vec((hip[0] + HIP_OFFSET[0] + .07,
                                  hip[1] + HIP_OFFSET[1] - .015, hip[2])))
        for name, axis, limits in (
            ('flexion', '0 0 1', '-.7 2.3'),
            ('adduction', f'{sign} 0 0', '-1 1'),
            ('rotation', f'0 {sign} 0', '-.9 .9'),
        ):
            element(exohip, 'joint', name=f'interface_hip_{name}_{side}',
                    type='hinge', axis=axis, limited='true', range=limits,
                    stiffness='0', damping='.05', armature='.001')
        element(exohip, 'inertial', pos='0 -.12 0', mass='.2',
                diaginertia='.003 .001 .003')
        element(exohip, 'geom', type='capsule', fromto='0 0 0 0 -.22 0',
                size='.008', mass='0', contype='0', conaffinity='0',
                rgba='.15 .25 .35 1', group='0')
        thigh = element(exohip, 'body', name=f'interface_thigh_cuff_{side}',
                        pos=vec((-HIP_OFFSET[0], -.22 - HIP_OFFSET[1], 0)))
        element(thigh, 'inertial', pos='0 0 0', mass='.13', diaginertia='.001 .001 .001')
        thigh_tissue = tissue_body(f'interface_thigh_tissue_{side}', '0 -.22 0')
        thigh_tissue.extend(pads(f'thigh_{side}', (.06, .055), .035, thigh, pairs))
        inserts[f'femur_{side}'] = [thigh_tissue]
        shank = ET.Element('body', name=f'exo_shank_{side}', pos='0 -.2 0')
        element(shank, 'inertial', pos='0 0 0', mass='.13',
                diaginertia='.001 .001 .001')
        floating_joints(shank, f'shank_{side}')
        element(shank, 'site', name=f'exo_cable_guide_{side}',
                pos=f'-.09 .05 {0 if side == "r" else -.003}', size='.005',
                rgba='1 0 0 1')
        shank_tissue = tissue_body(f'interface_shank_tissue_{side}', '0 -.2 0')
        shank_tissue.extend(pads(f'shank_{side}', (.035, .032), .035, shank, pairs))
        inserts[f'tibia_{side}'] = [shank_tissue, shank]
    return inserts, pairs


def update_model(path):
    text = path.read_text(encoding='utf-8')
    # Remove only this generator's regions, making re-running idempotent.
    text = re.sub(r'\n\s*<!-- BEGIN SOFT CONTACT .*?<!-- END SOFT CONTACT -->',
                  '', text, flags=re.S)
    # Replace the original rigid shank attachments, without touching distal shoes.
    text = re.sub(r'\s*<body name="exo_shank_[rl]" pos="0 -0.2 0">.*?</body>',
                  '', text, flags=re.S)
    inserts, pairs = build_interfaces()
    for parent, nodes in inserts.items():
        pattern = rf'(?m)^([\t ]*)(<body name="{parent}"[^>]*>)'
        content = '<!-- BEGIN SOFT CONTACT ' + parent + ' -->\n'
        content += '\n'.join(fragment(node) for node in nodes)
        content += '\n<!-- END SOFT CONTACT -->'
        def insert(match):
            indent = match[1] + '\t'
            return match[0] + '\n' + '\n'.join(indent + line for line in content.splitlines())
        text, count = re.subn(pattern, insert, text, count=1)
        if count != 1:
            raise ValueError(f'{path}: missing body {parent}')
    pair_text = '\n\t\t<!-- BEGIN SOFT CONTACT pairs -->\n'
    pair_text += '\n'.join('\t\t' + fragment(pair) for pair in pairs)
    pair_text += '\n\t\t<!-- END SOFT CONTACT -->'
    text = text.replace('<contact>', '<contact>' + pair_text, 1)
    # Check well-formedness before writing; MJCF compile is a separate check.
    ET.fromstring(text)
    path.write_text(text, encoding='utf-8', newline='\n')
    print(path.name)


if __name__ == '__main__':
    for filename in MODELS:
        path = ENV_DIR / filename
        if path.exists():
            update_model(path)
