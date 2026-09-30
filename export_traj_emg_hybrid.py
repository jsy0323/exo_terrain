#!/usr/bin/env python3
"""Export EMG/control rollouts from the hybrid terrain-humanoid + flat-exo policy.

The generated ``traj_<id>.npz`` files use the same schema as
``export_traj_emg.py`` and can therefore be read directly by
``plot_emg_evaluate.py``.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from export_traj_emg import (
    _compute_applied_ctrl,
    _get_motor_metadata,
    _terrain_metadata,
    _terrain_seed_for_traj,
)
from visualize_hybrid_terrain_exo import HybridTerrainExoVisualizer


class HeadlessHybridTerrainExoVisualizer(HybridTerrainExoVisualizer):
    """Load the hybrid policy through the visualizer without opening a viewer."""

    def load_config(self, cfg_path):
        super().load_config(cfg_path)
        self.cfg.setdefault("env", {})["render_mode"] = None
        self.cfg.setdefault("visualization", {})["wandb"] = False


def _get_exo_metadata(base_env, exo_action_dim: int = 2):
    actuator_names = list(getattr(base_env, "actuator_names", ()))
    num_actuators = len(actuator_names)
    if exo_action_dim <= 0 or exo_action_dim > num_actuators:
        raise ValueError(
            f"Invalid exo action dimension {exo_action_dim} for {num_actuators} actuators."
        )

    exo_indices = np.arange(
        num_actuators - exo_action_dim,
        num_actuators,
        dtype=np.int64,
    )
    exo_names = np.asarray(
        [str(actuator_names[int(index)]) for index in exo_indices],
        dtype=str,
    )
    if not all(base_env_name_is_exo(name) for name in exo_names):
        raise ValueError(
            "The final environment actuators are not recognized as exoskeleton "
            f"actuators: {exo_names.tolist()}"
        )
    return exo_indices, exo_names


def base_env_name_is_exo(name: str) -> bool:
    name = str(name).lower()
    return name.startswith("exo_") or name.startswith("ankle_motor_")


def rollout_hybrid_emg(
    visualizer,
    traj_id: int,
    terrain_seed: int | None = None,
    save_terrain_heightmap: bool = False,
    max_steps: int | None = None,
):
    """Roll out one trajectory and return an export_traj_emg-compatible dict."""

    obs_raw, info = visualizer.env.reset(
        id=int(traj_id),
        frame=0,
        terrain_seed=terrain_seed,
    )

    base_env = visualizer.env.envs[0].unwrapped
    terrain_metadata, terrain_heightmap_m = _terrain_metadata(base_env, terrain_seed)
    motor_indices, motor_names = _get_motor_metadata(base_env)
    exo_indices, exo_names = _get_exo_metadata(base_env, exo_action_dim=2)

    obs = visualizer.obs_normalizer(
        visualizer._compose_policy_obs(obs_raw, info)
    )
    phases = []
    emgs = []
    motor_ctrls = []
    exo_ctrls = []

    done = False
    step_count = 0
    while not done and (max_steps is None or step_count < max_steps):
        phases.append(float(base_env.ref_traj.phase))
        emgs.append(np.asarray(base_env.data.act, dtype=np.float32).copy())

        with torch.no_grad():
            actions = visualizer.policy.act_inference(obs)
            actions = visualizer._align_actions_to_env(actions)

        applied_ctrl = _compute_applied_ctrl(actions[0], visualizer.env, base_env)
        motor_ctrls.append(applied_ctrl[motor_indices].copy())
        exo_ctrls.append(applied_ctrl[exo_indices].copy())

        obs_raw, _, done_value, info = visualizer.env.step(actions)
        if torch.is_tensor(done_value):
            done = bool(done_value.item())
        else:
            done = bool(np.asarray(done_value).reshape(-1)[0])
        obs = visualizer.obs_normalizer(
            visualizer._compose_policy_obs(obs_raw, info)
        )
        step_count += 1

    if not phases:
        raise RuntimeError(f"Trajectory {traj_id} produced no rollout samples.")

    result = {
        "phase": np.asarray(phases, dtype=np.float32),
        "emg": np.stack(emgs).astype(np.float32),
        "motor_ctrl": np.stack(motor_ctrls).astype(np.float32),
        "motor_indices": motor_indices,
        "motor_names": motor_names,
        "exo_ctrl": np.stack(exo_ctrls).astype(np.float32),
        "exo_indices": exo_indices,
        "exo_names": exo_names,
        "traj_id": np.int64(traj_id),
        "speed": np.float32(base_env.ref_traj.speed),
        **terrain_metadata,
    }
    if save_terrain_heightmap:
        result["terrain_heightmap_m"] = terrain_heightmap_m
    return result


def _parse_args():
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Export EMG and applied controls using a terrain humanoid policy "
            "combined with a flat-trained exoskeleton actor."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_dir / "configs" / "visualize.yaml"),
        help="Visualization YAML containing the terrain policy and terrain settings.",
    )
    parser.add_argument(
        "--exo-checkpoint",
        type=str,
        default=str(repo_dir / "models" / "exo_flat_terrain" / "model_best.pt"),
        help="Flat-trained checkpoint supplying the two exoskeleton actions.",
    )
    parser.add_argument(
        "--terrain-checkpoint",
        "--checkpoint",
        dest="terrain_checkpoint",
        type=str,
        default=None,
        help="Optional override for the terrain humanoid checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to <terrain_checkpoint_dir>/traj_emg_hybrid."
        ),
    )
    parser.add_argument(
        "--traj-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional trajectory IDs to export. By default all trajectories are exported.",
    )
    parser.add_argument(
        "--terrain_seed",
        type=int,
        default=None,
        help="Override export.terrain_seed for reproducible terrain generation.",
    )
    parser.add_argument(
        "--terrain_seed_mode",
        choices=("shared", "per_trajectory"),
        default=None,
        help="Reuse one seed or derive a stable seed for every trajectory.",
    )
    parser.add_argument(
        "--save-terrain-heightmap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store the exact terrain heightmap in every NPZ file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional rollout step limit, mainly useful for quick validation.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    visualizer = HeadlessHybridTerrainExoVisualizer(
        cfg_path=args.config,
        exo_checkpoint=args.exo_checkpoint,
        terrain_checkpoint=args.terrain_checkpoint,
    )

    try:
        export_cfg = dict(visualizer.cfg.get("export", {}) or {})
        if args.terrain_seed is not None:
            export_cfg["terrain_seed"] = int(args.terrain_seed)
        if args.terrain_seed_mode is not None:
            export_cfg["terrain_seed_mode"] = args.terrain_seed_mode

        save_heightmap = (
            bool(export_cfg.get("save_terrain_heightmap", False))
            if args.save_terrain_heightmap is None
            else bool(args.save_terrain_heightmap)
        )
        terrain_checkpoint = visualizer.cfg["policy"]["checkpoint"]
        checkpoint_dir = os.path.dirname(terrain_checkpoint)
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else Path(checkpoint_dir) / "traj_emg_hybrid"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        traj_total = int(visualizer.num_traj)
        traj_ids = list(range(traj_total)) if args.traj_ids is None else args.traj_ids
        invalid_ids = [traj_id for traj_id in traj_ids if not 0 <= traj_id < traj_total]
        if invalid_ids:
            raise ValueError(
                f"Trajectory IDs out of range [0, {traj_total - 1}]: {invalid_ids}"
            )

        print(
            f"Exporting {len(traj_ids)} hybrid rollout(s) to {out_dir} "
            f"from {traj_total} available trajectories."
        )
        for traj_id in traj_ids:
            terrain_seed = _terrain_seed_for_traj(export_cfg, traj_id)
            data = rollout_hybrid_emg(
                visualizer,
                traj_id=traj_id,
                terrain_seed=terrain_seed,
                save_terrain_heightmap=save_heightmap,
                max_steps=args.max_steps,
            )
            out_path = out_dir / f"traj_{traj_id}.npz"
            np.savez_compressed(out_path, **data)
            print(
                f"Saved {out_path} (steps={data['phase'].shape[0]}, "
                f"speed={float(data['speed']):.2f}, "
                f"terrain_seed={int(data['terrain_seed'])}, "
                f"terrain_hash={str(data['terrain_hash'])[:12]})"
            )
    finally:
        visualizer.env.close()


if __name__ == "__main__":
    main()
