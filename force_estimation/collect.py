"""Collect randomized wearable episodes from an existing locomotion policy."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import yaml

from visualize import PolicyVisualizer
from .features import IMU_NAMES, JOINT_NAMES, feature_names
from .sensors import WearableSimulator
from .experiments import SimulationConditions, EstimatorFeedback


def collect(config, episodes=None, output_dir=None, duration_s=None, resume=False,
            estimator_checkpoint=None, assistance_gain=0.0):
    torch.set_num_threads(1)
    collection = config["collection"]
    episode_count = int(episodes if episodes is not None else collection["episodes"])
    duration = float(duration_s if duration_s is not None else collection["duration_s"])
    if episode_count < 1 or duration <= 0:
        raise ValueError("episodes and duration_s must be positive")
    output = Path(output_dir or collection["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    existing = None
    if resume:
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing["config"] != config or not np.isclose(existing.get("requested_duration_s", duration), duration):
            raise ValueError("Collection resume requires the same sensor and experiment configuration")
    if existing is None and (any(output.glob("episode_*.npz")) or (output / "manifest.json").exists()):
        raise FileExistsError(f"Dataset already exists in {output}. Choose a new output directory.")
    visualizer = PolicyVisualizer(collection["visualization_config"], overrides={
        "policy": {"checkpoint": collection["checkpoint"]},
        "env": {"render_mode": None, "config": {"wearable_sensor_sites": True}},
    })
    env = visualizer.base_env
    conditions = SimulationConditions(env, config)
    manifest = {"schema_version": 2, "sample_rate_hz": config["sample_rate_hz"],
                "imu_names": list(IMU_NAMES), "imu_fields": ["gyro_xyz", "specific_force_xyz", "gravity_direction_xyz"],
                "joint_names": list(JOINT_NAMES), "extra_names": list(feature_names(
                    bool(config["training"].get("include_exo_features", False)))),
                "torque_units": "N m", "grf_units": "N", "joint_sign": "positive along MJCF joint axis",
                "torque_formula": "M(q) @ qacc + qfrc_bias - qfrc_constraint",
                "bio_formula": "tau_total - known_applied_joint_moment",
                "assistance": "ideal_joint_torque_source" if config.get("experiments") else "unassisted",
                "gravity_source": config.get("preprocessing", {}).get("gravity_source", "simulated_orientation"),
                "processing": config.get("preprocessing", {}),
                "subject_count": len(conditions.subjects), "geometry_variants": 1,
                "mujoco_version": __import__("mujoco").__version__,
                "requested_duration_s": duration, "config": config, "episodes": []}
    if estimator_checkpoint:
        manifest["estimator_feedback"] = {
            "checkpoint": str(estimator_checkpoint), "gain_relative_to_biological_torque": assistance_gain,
            "sha256": hashlib.sha256(Path(estimator_checkpoint).read_bytes()).hexdigest(),
        }
    try:
        if env.human_state_map.interface_state_dim or any("ankle_motor" in name for name in env.actuator_names):
            raise ValueError("This collector measures unassisted humanoid data. An exo delivery adapter is needed for powered collection.")
        simulator = WearableSimulator(env, config)
        checkpoint = Path(collection["checkpoint"])
        manifest["policy_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if existing is not None:
            if existing["policy_sha256"] != manifest["policy_sha256"]:
                raise ValueError("The locomotion checkpoint changed since collection started")
            manifest = existing
        control_dt = env.opt_time * env.frame_skip
        steps = int(np.ceil(duration / control_dt))
        env.max_episode_steps = max(env.max_episode_steps, steps)
        for episode in range(len(manifest["episodes"]), episode_count):
            seed = int(collection["seed"]) + episode
            condition_metadata = conditions.select(episode)
            obs, info = visualizer.env.reset(seed=seed)
            simulator.reset(seed)
            if estimator_checkpoint:
                feedback = EstimatorFeedback(estimator_checkpoint, simulator, conditions, assistance_gain)
                simulator.sample_callback = feedback.on_sample
            env.physics_step_callback = simulator.on_physics_step
            env.physics_pre_step_callback = conditions.apply_torque
            # Mounting randomization must not alter the policy's legacy proprioception.
            obs = visualizer.obs_normalizer(visualizer._compose_policy_obs(obs, info))
            done = False
            for _ in range(steps):
                with torch.inference_mode():
                    actions = visualizer._align_actions_to_env(visualizer.policy.act_inference(obs))
                raw, _, done, info = visualizer.env.step(actions)
                obs = visualizer.obs_normalizer(visualizer._compose_policy_obs(raw, info))
                if bool(done):
                    break
            arrays = simulator.arrays()
            for name, values in arrays.items():
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {name} in episode {episode}")
            achieved_duration = float(arrays["time"][-1])
            terminated = any(bool(value) for value in info.get("terminated_info", {}).values())
            accepted = achieved_duration >= duration * float(collection.get("minimum_duration_fraction", .9)) and not terminated
            tracking_error = arrays["forward_speed"] - arrays["command_speed"]
            coverage = {"speed_min_mps": float(np.min(arrays["forward_speed"])),
                        "speed_mean_mps": float(np.mean(arrays["forward_speed"])),
                        "speed_max_mps": float(np.max(arrays["forward_speed"])),
                        "tracking_rmse_mps": float(np.sqrt(np.mean(tracking_error ** 2))),
                        "both_feet_below_contact_threshold_fraction": float(np.mean(~arrays["contact"].astype(bool).any(axis=1))),
                        "imu_available_fraction": float(np.mean(arrays["imu_mask"]))}
            metadata = {"episode": episode, "seed": seed, "traj_id": env.ref_traj.traj_id,
                        **condition_metadata, "accepted": accepted, "coverage": coverage,
                        "ramp_acceleration": None if env.speed_command is None else env.speed_command.ramp_acceleration,
                        "termination": info.get("terminated_info", {}), "subject": simulator.subject.tolist(),
                        "randomization": simulator.randomization_metadata()}
            filename = f"episode_{episode:05d}.npz"
            temporary = output / f".{filename}.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays, metadata=np.array(json.dumps(metadata)))
            temporary.replace(output / filename)
            manifest["episodes"].append({"file": filename, "seed": seed, "samples": len(arrays["time"]),
                                         "duration_s": achieved_duration, **condition_metadata,
                                         "accepted": accepted, "coverage": coverage})
            temporary = output / "manifest.json.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(output / "manifest.json")
            print(f"Saved {filename}, {len(arrays['time'])} samples, {arrays['time'][-1]:.2f} s, "
                  f"{condition_metadata['subject_id']}/{condition_metadata['motion_id']}, accepted={accepted}", flush=True)
    finally:
        env.physics_step_callback = None
        env.physics_pre_step_callback = None
        conditions.restore()
        visualizer.env.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/force_estimation.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted collection")
    parser.add_argument("--estimator-checkpoint", help="Enable ideal hip-torque feedback in simulation")
    parser.add_argument("--assistance-gain", type=float, default=0.0)
    arguments = parser.parse_args()
    with open(arguments.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    collect(config, arguments.episodes, arguments.output_dir, arguments.duration_s, arguments.resume,
            arguments.estimator_checkpoint, arguments.assistance_gain)


if __name__ == "__main__":
    main()
