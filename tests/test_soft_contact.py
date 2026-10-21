"""Physical and RL integration checks for the sparse cuff interfaces."""
import copy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from Env.MFG_Musculoskelet_V11.tools.add_soft_contact import (
    ENV_DIR, MODELS, build_interfaces, update_model,
)
from Env.MFG_Musculoskelet_V11.mfgenv.human_state import HumanStateMap
from Env.MFG_Musculoskelet_V11.mfgenv.mujoco_utils import load_mujoco_model


def shank_rig(contact=True, cable=False):
    """Fixed tibia, freely moving cuff; no stops or centering forces to mask slip."""
    inserts, pairs = build_interfaces()
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', boundmass='.001', boundinertia='.001')
    ET.SubElement(root, 'option', timestep='.001', gravity='0 0 0',
                  integrator='implicitfast', cone='elliptic', impratio='5')
    world = ET.SubElement(root, 'worldbody')
    for body in inserts['tibia_r']:
        world.append(copy.deepcopy(body))
    for joint in world.iter('joint'):
        joint.set('limited', 'false')
    if cable:
        ET.SubElement(world, 'site', name='distal_shoe', pos='-.03 -.45 0', size='.005')
        tendon = ET.SubElement(root, 'tendon')
        spatial = ET.SubElement(tendon, 'spatial', name='test_cable')
        ET.SubElement(spatial, 'site', site='exo_cable_guide_r')
        ET.SubElement(spatial, 'site', site='distal_shoe')
        actuator = ET.SubElement(root, 'actuator')
        ET.SubElement(actuator, 'motor', name='test_motor', tendon='test_cable', gear='-200')
    if contact:
        contacts = ET.SubElement(root, 'contact')
        contacts.extend(copy.deepcopy(p) for p in pairs if 'shank_r_' in p.get('name'))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def settled_force(model, force=(0, 0, 0), steps=1500):
    data = mujoco.MjData(model)
    if model.nu:
        data.ctrl[:] = 1.
    data.xfrc_applied[model.body('exo_shank_r').id, :3] = force
    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    net = np.zeros(3)
    normal = 0.
    for i, contact in enumerate(data.contact):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        normal += wrench[0]
        net += contact.frame.reshape(3, 3).T @ wrench[:3]
    return data, net, normal


class SoftContactTest(unittest.TestCase):
    def test_generator_is_repeatable_after_terrain_xml_round_trip(self):
        # Terrain generators parse/write the MJCF, retaining the region markers.
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'terrain_exo.xml'
            tree = ET.parse(ENV_DIR / MODELS[0],
                            parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
            tree.write(path, encoding='utf-8', xml_declaration=True)
            with redirect_stdout(StringIO()):
                update_model(path)
                first = path.read_bytes()
                update_model(path)
            self.assertEqual(first, path.read_bytes())
            names = [node.get('name') for node in ET.parse(path).getroot().iter('joint')
                     if node.get('name', '').startswith('interface_')]
            self.assertEqual(len(names), 24)
            self.assertEqual(len(set(names)), 24)

    def test_all_variants_compile_in_training_and_visual_modes(self):
        for filename in MODELS:
            for muscle in (True, False):
                for mesh in (False, True):
                    with self.subTest(filename=filename, muscle=muscle, mesh=mesh):
                        model, data = load_mujoco_model(str(ENV_DIR / filename), muscle, mesh)
                        self.assertEqual(model.npair, 40)
                        self.assertEqual(model.nq, 77)
                        self.assertEqual(model.nv, 76)
                        # Preserve existing files that deliberately disable motors.
                        source = ET.parse(ENV_DIR / filename).getroot()
                        motors = [a.get('name') for a in source.findall('actuator/motor')]
                        self.assertEqual([model.actuator(i).name for i in range(model.nu)
                                          if model.actuator(i).name.startswith('ankle_motor')], motors)
                        mujoco.mj_forward(model, data)
                        self.assertTrue(np.isfinite(data.qacc).all())

    def test_human_mapping_round_trip_and_independent_cuff_state(self):
        model = mujoco.MjModel.from_xml_path(str(ENV_DIR / MODELS[0]))
        mapping = HumanStateMap(model)
        self.assertEqual(mapping.qpos_indices.size, 53)
        self.assertEqual(mapping.qvel_indices.size, 52)
        q = model.qpos0[mapping.qpos_indices].copy()
        v = np.linspace(-.2, .2, 52)
        q[7] = .35
        full_q, full_v = mapping.expand(q, v)
        np.testing.assert_array_equal(full_q[mapping.qpos_indices], q)
        np.testing.assert_array_equal(full_v[mapping.qvel_indices], v)
        for eq, hq, ev, hv in mapping.hip_copies:
            self.assertEqual(full_q[eq], full_q[hq])
            self.assertEqual(full_v[ev], full_v[hv])
        cuff = model.joint('interface_shank_r_slide_x')
        full_q[cuff.qposadr[0]] = .004
        self.assertAlmostEqual(mapping.travel_fraction(full_q), .004 / .03)
        np.testing.assert_array_equal(full_q[mapping.qpos_indices], q)

    def test_preload_and_contact_carry_force_without_travel_stops(self):
        model = shank_rig()
        initial, _, preload = settled_force(model)
        self.assertGreater(preload, 0)
        force = np.array([15., 200., 0.])
        data, net, normal = settled_force(model, force)
        # Contact frame direction depends on pair order; compare magnitudes.
        np.testing.assert_allclose(np.abs(net), np.abs(force), atol=1.)
        self.assertLess(np.max(np.abs(data.qpos[:3])), .02)
        self.assertLess(np.max(np.abs(data.qpos[3:])), .2)
        self.assertLess(np.linalg.norm(data.qvel), .1)
        self.assertTrue(np.isfinite(data.qpos).all())
        # Without contacts the free cuff cannot transmit the applied force.
        free, net_free, _ = settled_force(shank_rig(False), force, steps=50)
        self.assertEqual(np.linalg.norm(net_free), 0.)
        self.assertGreater(np.linalg.norm(free.qvel), 1.)
        print(f'\nCuff rig: preload={preload:.2f} N, loaded normal={normal:.2f} N, '
              f'slip={np.max(np.abs(data.qpos[:3])):.5f} m')

    def test_offcentre_cable_load_is_carried_by_contacts(self):
        model = shank_rig(cable=True)
        data, _, normal = settled_force(model)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertLess(np.max(np.abs(data.qpos[:3])), .024)
        self.assertLess(np.max(np.abs(data.qpos[3:])), .2)
        self.assertLess(np.linalg.norm(data.qvel), .1)
        self.assertGreater(normal, 0.)
        self.assertLess(np.linalg.norm(data.qfrc_constraint + data.qfrc_actuator), 1.)

    def test_environment_reference_reset_rewards_and_steps(self):
        from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11
        from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager
        from Env.MFG_Musculoskelet_V11.mfgenv.reward import compute_joint_imitation

        config = yaml.safe_load((ENV_DIR / 'cfg/mujocoenv_default.yaml').read_text(encoding='utf-8'))
        tm = config['traj_manager']
        config['traj_manager'] = TrajectoryManager(**tm)
        config.update(render_mode=None, auto_render=False, load_mesh=False,
                      terrain_randomize_on_reset=False)
        env = MFG_Musculoskeletal_V11(config, render_mode=None)
        try:
            obs, reset_info = env.reset(seed=7)
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertEqual(obs.shape, (1069,))
            self.assertEqual(env.num_privileged_obs, 1117)
            self.assertEqual(reset_info['critic_extra_obs'].shape, (48,))
            # The original 92 human joint features retain their old positions,
            # including the exo actor's original first-256 proprioception block.
            mapping = env.human_state_map
            expected_joints = np.concatenate((env.data.qpos[mapping.qpos_indices[7:]],
                                              env.data.qvel[mapping.qvel_indices[6:]]))
            np.testing.assert_allclose(obs[164:256], expected_joints, atol=1e-6)
            self.assertTrue(np.isfinite(obs).all())
            rpos, rvel, info = compute_joint_imitation(env)
            self.assertNotIn('error', info)
            self.assertAlmostEqual(rpos, 1., places=6)
            self.assertAlmostEqual(rvel, 1., places=6)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[-2:] = .1
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
                self.assertIn('interface_travel_fraction', info)
                np.testing.assert_array_equal(info['critic_extra_obs'],
                                              mapping.interface_state(env.data))
                if terminated or truncated:
                    obs, _ = env.reset()
            env._compute_site_positions(env.ref_traj.get_qpos())
        finally:
            env.close()

    def test_supported_gait_poses_stay_inside_cuff_travel(self):
        from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11
        from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager

        config = yaml.safe_load((ENV_DIR / 'cfg/mujocoenv_default.yaml').read_text(encoding='utf-8'))
        config['traj_manager'] = TrajectoryManager(**config['traj_manager'])
        config.update(render_mode=None, auto_render=False, load_mesh=False,
                      terrain_randomize_on_reset=False)
        env = MFG_Musculoskeletal_V11(config, render_mode=None)
        try:
            mapping = env.human_state_map
            # Support the skeleton without prescribing the cuff state each step.
            env.model.dof_armature[mapping.qvel_indices] = 1e9
            env.reset(seed=7)
            frames = (0, int(env.ref_traj.traj_frames) // 4,
                      int(env.ref_traj.traj_frames) // 2)
            for frame in frames:
                env.reset(traj_id=0, frame=frame)
                env.data.qvel[:] = 0.
                peak_travel = 0.
                env.data.ctrl[:] = 0.
                env.data.ctrl[-2:] = 1.
                for _ in range(1500):
                    mujoco.mj_step(env.model, env.data)
                    peak_travel = max(peak_travel, mapping.travel_fraction(env.data.qpos))
                self.assertLess(peak_travel, .8, f'frame={frame}, peak travel={peak_travel}')
                self.assertTrue(np.isfinite(env.data.qpos).all())
                self.assertFalse(np.any(env.data.warning.number))
            print(f'\nSupported gait poses: nq={env.model.nq}, nv={env.model.nv}, '
                  f'observation={env.observation_space.shape}, action={env.action_space.shape}')
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
