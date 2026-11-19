"""Physics, temporal alignment, missing sensors, causality and uncertainty checks."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import mujoco
import numpy as np
import torch
import yaml

from force_estimation.features import EXTRA_NAMES, HUMAN_EXTRA_NAMES, EncoderFilter, FeatureBuilder, contact_phase
from force_estimation.model import MomentEstimator, RealtimeEstimator, estimator_loss
from force_estimation.runtime import OnlineMomentEstimator
from force_estimation.sensors import WearableSimulator, fsr_model, inverse_dynamics_moment
from force_estimation.train import selection_score, train
from force_estimation.data import split_episodes
from force_estimation.preprocessing import CausalLowPass, GravityAHRS
from force_estimation.experiments import SimulationConditions
from visualize import PolicyVisualizer


class EstimatorTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        with open("configs/force_estimation.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["window_s"] = .04
        config["training"].update(device="cpu", batch_size=4, stride_samples=2, peak_weight=1)
        rng = np.random.default_rng(12)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            manifest = {"sample_rate_hz": 100, "extra_names": list(HUMAN_EXTRA_NAMES),
                        "policy_sha256": "test-policy", "gravity_source": "test",
                        "assistance": "unassisted", "episodes": []}
            for episode in range(6):
                filename = f"episode_{episode}.npz"
                count = 10
                torque = rng.normal(size=(count, 6)).astype(np.float32)
                phase = np.zeros((count, 2, 2), dtype=np.float32)
                phase[..., 1] = 1
                np.savez(data / filename, time=np.arange(count) * .01,
                         imu=rng.normal(size=(count, 7, 9)).astype(np.float32),
                         imu_mask=np.ones((count, 7), dtype=bool),
                         extra=rng.normal(size=(count, len(HUMAN_EXTRA_NAMES))).astype(np.float32),
                         tau_bio=torque, grf_v=np.ones((count, 2), dtype=np.float32),
                         contact=np.ones((count, 2), dtype=np.float32), phase=phase,
                         phase_mask=np.ones((count, 2), dtype=bool))
                manifest["episodes"].append({"file": filename})
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with redirect_stdout(StringIO()):
                train(config, data_dir=data, output_dir=root / "full", epochs=2)
                train(config, data_dir=data, output_dir=root / "resumed", epochs=1)
                config["training"]["seed"] = 999
                train(config, epochs=1, resume=root / "resumed" / "model_last.pt")
            full = torch.load(root / "full" / "model_last.pt", weights_only=False)
            resumed = torch.load(root / "resumed" / "model_last.pt", weights_only=False)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(full["partitions"], resumed["partitions"])
            self.assertEqual([entry["epoch"] for entry in resumed["history"]], [1, 2])
            for name in full["model_state_dict"]:
                torch.testing.assert_close(full["model_state_dict"][name], resumed["model_state_dict"][name], rtol=0, atol=0)
            # A fine-tuning ablation must not open the test recordings.
            hidden = []
            for path in full["partitions"]["test"]:
                source_path = Path(path)
                destination = source_path.with_suffix(".hidden")
                source_path.rename(destination)
                hidden.append((source_path, destination))
            config["training"].update(selection_metric="hip_mae", loss={"hip_type": "smooth_l1", "hip_beta_nm": 2.})
            try:
                with redirect_stdout(StringIO()):
                    _, deferred = train(config, output_dir=root / "ablation", epochs=1,
                                        resume=root / "full" / "model_last.pt", reset_optimizer=True, evaluate_test=False)
                self.assertIsNone(deferred)
                self.assertFalse((root / "ablation" / "metrics_test.json").exists())
                candidate = torch.load(root / "ablation" / "model_best.pt", weights_only=False)
                self.assertEqual(candidate["config"]["training"]["loss"], config["training"]["loss"])
                self.assertEqual(candidate["partitions"], full["partitions"])
            finally:
                for source_path, destination in hidden:
                    destination.rename(source_path)

    def test_mae_loss_units_weights_and_selection(self):
        hip = torch.tensor([[-1., 3.]], requires_grad=True)
        prediction = {"tau_hip": hip, "tau_aux": torch.ones(1, 4), "grf_v": torch.ones(1, 2),
                      "contact_logit": torch.ones(1, 2), "phase": torch.ones(1, 2, 2), "sigma_nm": torch.ones(1, 2) * 4}
        target = {"torque": torch.zeros(1, 6), "grf_v": torch.zeros(1, 2), "contact": torch.zeros(1, 2),
                  "phase": torch.zeros(1, 2, 2), "phase_mask": torch.zeros(1, 2, dtype=torch.bool)}
        weights = {"aux": 0., "grf": 0., "contact": 0., "phase": 0., "nll": 0.}
        loss, _ = estimator_loss(prediction, target, loss_config={"hip_type": "smooth_l1", "hip_beta_nm": 2., "weights": weights})
        self.assertAlmostEqual(float(loss.detach()), .0225, places=6)
        loss.backward()
        torch.testing.assert_close(hip.grad, torch.tensor([[-.005, .01]]))
        default, terms = estimator_loss(prediction, target)
        torch.testing.assert_close(default, terms["hip"] + .3 * terms["aux"] + .2 * terms["grf"] +
                                   .2 * terms["contact"] + .2 * terms["phase"] + .1 * terms["nll"])
        self.assertEqual(selection_score({"loss": -10., "hip_mae_nm": [2., 4.]}, "hip_mae"), 3.)
        with self.assertRaises(ValueError):
            estimator_loss(prediction, target, loss_config={"weights": {"nll": -1}})

    def test_human_only_and_legacy_checkpoint_inputs(self):
        with open("configs/force_estimation.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        subject = [75, 1.75, .42, .43, .26]
        imu = np.zeros((7, 9), dtype=np.float32)
        imu[:, 8] = -1
        mask = np.ones(7, dtype=bool)
        with TemporaryDirectory() as directory:
            for names in (HUMAN_EXTRA_NAMES, EXTRA_NAMES):
                model_args = {"window_s": .02, "extra_dim": len(names), "dropout": 0}
                model = MomentEstimator(**model_args)
                path = Path(directory) / f"model_{len(names)}.pt"
                torch.save({"model_args": model_args, "model_state_dict": model.state_dict(),
                            "extra_names": list(names), "target": "bio", "config": config}, path)
                online = OnlineMomentEstimator(path, subject)
                self.assertEqual(len(online.features.extra_names), len(names))
                self.assertIsNone(online.step(.01, imu, mask, [0, 0], np.zeros(4)))
                output = online.step(.02, imu, mask, [0, 0], np.zeros(4))
                self.assertEqual(output["tau_hip"].shape, (2,))
                self.assertTrue(np.isfinite(output["tau_hip"]).all())
                if names == HUMAN_EXTRA_NAMES:
                    with self.assertRaises(ValueError):
                        online.step(.03, imu, mask, [0, 0], np.zeros(4), exo=np.ones(8))

    def test_causality_and_absent_imus(self):
        torch.manual_seed(4)
        model = MomentEstimator(dropout=0).eval()
        imu, extra = torch.randn(2, 20, 7, 9), torch.randn(2, 20, len(EXTRA_NAMES))
        mask = torch.ones(2, 20, 7, dtype=torch.bool)
        first = model.forward_sequence(imu, mask, extra)
        changed = imu.clone()
        changed[:, 10:] += 100
        changed_extra = extra.clone()
        changed_extra[:, 10:] -= 100
        second = model.forward_sequence(changed, mask, changed_extra)
        for name in first:
            torch.testing.assert_close(first[name][:, :10], second[name][:, :10])
        mask[:] = False
        absent = model(imu, mask, extra)
        nan = model(torch.full_like(imu, float("nan")), mask, extra)
        for name in absent:
            self.assertTrue(torch.isfinite(absent[name]).all())
            torch.testing.assert_close(absent[name], nan[name])
        with self.assertRaises(ValueError):
            MomentEstimator(sample_rate_hz=200, window_s=1)

    def test_streaming_matches_window_inference_and_loss(self):
        model = MomentEstimator(window_s=.1, dropout=0).eval()
        stream = RealtimeEstimator(model)
        imu = np.zeros((10, 7, 9), dtype=np.float32)
        mask = np.ones((10, 7), dtype=bool)
        extra = np.zeros((10, len(EXTRA_NAMES)), dtype=np.float32)
        for index in range(9):
            self.assertIsNone(stream.step(imu[index], mask[index], extra[index]))
        prediction = stream.step(imu[-1], mask[-1], extra[-1])
        offline = model(torch.from_numpy(imu[None]), torch.from_numpy(mask[None]), torch.from_numpy(extra[None]))
        for name in prediction:
            np.testing.assert_allclose(prediction[name], offline[name][0].detach().numpy(), rtol=1e-5)
        target = {"torque": torch.zeros(1, 6), "grf_v": torch.zeros(1, 2),
                  "contact": torch.zeros(1, 2), "phase": torch.zeros(1, 2, 2),
                  "phase_mask": torch.zeros(1, 2, dtype=torch.bool)}
        loss, terms = estimator_loss(offline, target)
        loss.backward()
        self.assertEqual(float(terms["phase"].detach()), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((offline["sigma_nm"] > 0).all())

    def test_fsr_units_and_phase_edges(self):
        parameters = {"deadzone": np.ones(4) * 20, "gain": np.ones(4), "alpha": np.ones(4),
                      "f_sat": np.ones(4) * 800, "lag_coef": np.ones(4), "creep": np.zeros(4),
                      "bias": np.ones(4) * .02, "sigma": 0.0}
        state = {"y": np.zeros(4), "drift": np.zeros(4)}
        value = fsr_model(np.ones(4) * 20, state, parameters, .01, 500, .02, np.random.default_rng(0))
        np.testing.assert_allclose(value, .02)
        value = fsr_model(np.ones(4) * 520, state, parameters, .01, 500, .02, np.random.default_rng(0))
        np.testing.assert_allclose(value, (500 / (1 + 500 / 800)) / 500 + .02)
        contact = np.zeros((25, 2))
        contact[3:7] = 1
        contact[13:17] = 1
        phase, valid = contact_phase(contact)
        self.assertFalse(valid[:3].any())
        self.assertFalse(valid[13:].any())
        np.testing.assert_allclose(phase[3, 0], [0, 1])
        np.testing.assert_allclose(phase[8, 0], [0, -1], atol=1e-6)


class SensorPhysicsTests(unittest.TestCase):
    def test_simulation_ahrs_filtering_and_known_torque_balance(self):
        with open("configs/force_estimation_sim.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with redirect_stdout(StringIO()):
            visualizer = PolicyVisualizer(config["collection"]["visualization_config"], overrides={
                "env": {"render_mode": None, "config": {"wearable_sensor_sites": True}},
                "policy": {"checkpoint": config["collection"]["checkpoint"]}})
        env = visualizer.base_env
        conditions = SimulationConditions(env, config)
        original_mass = env.model.body_mass.copy()
        try:
            metadata = conditions.select(6)
            self.assertEqual(metadata["motion_id"], "hip_perturbation")
            np.testing.assert_allclose(env.model.body_mass, original_mass * .94)
            visualizer.env.reset(seed=4)
            simulator = WearableSimulator(env, config)
            simulator.reset(4)
            env.physics_step_callback = simulator.on_physics_step
            env.physics_pre_step_callback = conditions.apply_torque
            for _ in range(15):
                obs, _ = visualizer.env.get_observations()
                with torch.inference_mode():
                    action = visualizer.policy.act_inference(visualizer.obs_normalizer(obs))
                visualizer.env.step(action)
            data = simulator.arrays()
            np.testing.assert_allclose(data["tau_total"] - data["tau_bio"], data["tau_applied"], atol=2e-5)
            np.testing.assert_allclose(data["tau_total_raw"] - data["tau_bio_raw"], data["tau_applied_raw"], atol=2e-5)
            self.assertGreater(float(np.max(abs(data["tau_applied_raw"]))), .1)
            self.assertFalse(data["sample_valid"].any())
            self.assertEqual(simulator.calibration_curves.shape, (4, 8, 2))
            self.assertTrue(np.isfinite(data["extra"]).all())
        finally:
            env.physics_step_callback = None
            env.physics_pre_step_callback = None
            conditions.restore()
            np.testing.assert_array_equal(env.model.body_mass, original_mass)
            visualizer.env.close()

    def test_actual_policy_sensor_sites_timing_and_inverse_dynamics(self):
        with open("configs/force_estimation.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with redirect_stdout(StringIO()):
            visualizer = PolicyVisualizer(config["collection"]["visualization_config"], overrides={
                "env": {"render_mode": None, "config": {"wearable_sensor_sites": True}},
                "policy": {"checkpoint": config["collection"]["checkpoint"]},
            })
        env = visualizer.base_env
        try:
            self.assertEqual(env.observation_dim, 471)
            simulator = WearableSimulator(env, config)
            visualizer.env.reset(seed=4)
            simulator.reset(4)
            original_delays = simulator.delays.copy()
            simulator.reset(4)
            np.testing.assert_array_equal(simulator.delays, original_delays)
            self.assertTrue(((simulator.delays >= 0) & (simulator.delays <= .005)).all())
            env.physics_step_callback = simulator.on_physics_step
            for _ in range(10):
                obs, _ = visualizer.env.get_observations()
                with torch.inference_mode():
                    action = visualizer.policy.act_inference(visualizer.obs_normalizer(obs))
                visualizer.env.step(action)
            arrays = simulator.arrays()
            self.assertEqual(arrays["imu"].shape, (20, 7, 9))
            self.assertEqual(arrays["extra"].shape, (20, len(HUMAN_EXTRA_NAMES)))
            np.testing.assert_allclose(np.diff(arrays["time"]), .01, atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(arrays["imu"][:, :, 6:9], axis=-1), 1, atol=1e-6)
            total = inverse_dynamics_moment(env.model, env.data, simulator.dof_indices)
            balance = (env.data.qfrc_actuator + env.data.qfrc_passive + env.data.qfrc_applied)[simulator.dof_indices]
            np.testing.assert_allclose(total, balance, atol=1e-5, rtol=1e-5)
            self.assertTrue(np.all(arrays["grf_v"] >= 0))
            self.assertGreater(float(arrays["grf_v"].max()), 100)
            # Replaying raw encoder angles through the hardware filter reproduces features.
            encoder = EncoderFilter(config["sample_rate_hz"], config["encoder"]["cutoff_hz"])
            builder = FeatureBuilder(config["sample_rate_hz"], config["fsr"]["calibration_value"],
                                     simulator.subject, include_exo_features=False)
            for index in range(len(arrays["time"])):
                extra = builder.step(arrays["imu"][index], arrays["imu_mask"][index],
                                     encoder.step(arrays["encoder"][index, :2]), arrays["fsr"][index], np.zeros(8))
                np.testing.assert_allclose(extra, arrays["extra"][index], rtol=2e-4, atol=2e-4)
        finally:
            env.physics_step_callback = None
            visualizer.env.close()


class PreprocessingTests(unittest.TestCase):
    def test_ahrs_rotates_gravity_using_sensor_stream(self):
        ahrs = GravityAHRS()
        ahrs.calibrate(np.tile([0., 0., 9.81], (7, 1)))
        for index in range(100):
            angle = (index + 1) * .01
            gravity = np.tile([np.sin(angle), 0., -np.cos(angle)], (7, 1))
            predicted = ahrs.step(np.tile([0., 1., 0.], (7, 1)), -9.81 * gravity, np.ones(7, dtype=bool))
            np.testing.assert_allclose(predicted, gravity, atol=1e-6)

    def test_antialias_filter_and_group_holdout(self):
        lowpass = CausalLowPass(1000, 30)
        output = np.array([lowpass.step(np.array([np.sin(2 * np.pi * 150 * i / 1000)]))[0] for i in range(2000)])
        self.assertLess(float(np.sqrt(np.mean(output[1000:] ** 2))), .01)
        paths = [f"subject_{subject}_motion_{motion}.npz" for subject in range(6) for motion in range(3)]
        groups = [f"subject_{subject}" for subject in range(6) for _ in range(3)]
        partitions = split_episodes(paths, .2, .2, 42, groups)
        mapping = dict(zip(paths, groups))
        sets = {name: {mapping[path] for path in entries} for name, entries in partitions.items()}
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["train"] & sets["validation"])


if __name__ == "__main__":
    unittest.main()
