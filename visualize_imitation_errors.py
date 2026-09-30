#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path

import torch

from Env.MFG_Musculoskelet_V11.mfgenv.reward import get_imitation_reward
from visualize import PolicyVisualizer


class ImitationErrorVisualizer(PolicyVisualizer):
    ERROR_KEYS = ("site_pos_err", "site_vel_err", "joint_angle_err", "l2_sq")

    def __init__(
        self,
        cfg_path,
        episodes=None,
        max_steps=None,
        output=None,
        per_step=False,
        sleep=False,
        no_render=False,
    ):
        self.requested_episodes = episodes
        self.max_steps = max_steps
        self.output_path = output
        self.print_per_step = per_step
        self.sleep_during_rollout = sleep
        self.no_render = no_render
        super().__init__(cfg_path)

    def load_config(self, cfg_path):
        super().load_config(cfg_path)
        if self.no_render:
            self.cfg.setdefault("env", {})["render_mode"] = None

    @staticmethod
    def _done_to_bool(done):
        if torch.is_tensor(done):
            return bool(done.item())
        return bool(done)

    def _policy_action(self, obs):
        if hasattr(self.policy, "act_inference"):
            return self.policy.act_inference(obs)
        return self.policy.act(obs)

    def _extract_errors_from_info(self, info):
        reward_info = info.get("reward_info", {}) if isinstance(info, dict) else {}
        imitation = reward_info.get("imitation", {}) if isinstance(reward_info, dict) else {}

        site_info = imitation.get("site_info", {}) if isinstance(imitation, dict) else {}
        joint_info = imitation.get("joint_info", {}) if isinstance(imitation, dict) else {}
        l2_sq = self._extract_l2_sq_from_reward_info(reward_info)

        metrics = self._metrics_from_breakdown(site_info, joint_info, l2_sq)
        if metrics is not None:
            return metrics

        # Fallback for configs where reward_info is disabled but obs_components is present.
        comps = info.get("obs_components", None) if isinstance(info, dict) else None
        if comps is None:
            return None
        try:
            _, breakdown = get_imitation_reward(self.base_env, comps)
        except Exception:
            return None

        site_info = breakdown.get("site_info", {}) if isinstance(breakdown, dict) else {}
        joint_info = breakdown.get("joint_info", {}) if isinstance(breakdown, dict) else {}
        if l2_sq is None:
            l2_sq = self._compute_current_act_l2_sq()
        return self._metrics_from_breakdown(site_info, joint_info, l2_sq)

    def _extract_l2_sq_from_reward_info(self, reward_info):
        if not isinstance(reward_info, dict):
            return None
        smooth = reward_info.get("smooth", {})
        if not isinstance(smooth, dict):
            return None
        energy = smooth.get("energy", {})
        if not isinstance(energy, dict):
            return None
        metrics = energy.get("metrics", {})
        if not isinstance(metrics, dict):
            return None
        return self._to_scalar(metrics.get("l2_sq"))

    def _compute_current_act_l2_sq(self):
        if not bool(getattr(self.base_env, "_has_muscle_actuators", False)):
            return 0.0
        data = getattr(self.base_env, "data", None)
        act = getattr(data, "act", None)
        if act is None:
            return None
        if getattr(act, "size", 0) == 0:
            return 0.0
        return float((act * act).sum())

    def _metrics_from_breakdown(self, site_info, joint_info, l2_sq):
        if not isinstance(site_info, dict) or not isinstance(joint_info, dict):
            return None

        values = {
            "site_pos_err": self._to_scalar(site_info.get("pos_err")),
            "site_vel_err": self._to_scalar(site_info.get("vel_err")),
            "joint_angle_err": self._to_scalar(joint_info.get("ang_err")),
            "l2_sq": self._to_scalar(l2_sq),
        }
        if any(values[k] is None for k in self.ERROR_KEYS):
            return None
        return values

    def _resolve_output_path(self):
        if self.output_path is not None:
            text = str(self.output_path).strip()
            if text and text.lower() not in {"none", "none.txt"}:
                return text
            return None
        return os.path.join(self.base_dir, "imitation_errors.csv")

    def run(self):
        num_episodes = self.num_traj
        if self.requested_episodes is not None:
            num_episodes = min(num_episodes, int(self.requested_episodes))

        print(f"Evaluating imitation errors for {num_episodes} episode(s)...")
        rows = []
        overall_sums = {k: 0.0 for k in self.ERROR_KEYS}
        overall_count = 0

        try:
            for episode in range(num_episodes):
                obs_raw, info = self.env.reset(id=episode, frame=0)
                obs = self.obs_normalizer(self._compose_policy_obs(obs_raw, info))
                done = False
                step_count = 0
                metric_count = 0
                sums = {k: 0.0 for k in self.ERROR_KEYS}
                speed, traj_id = self.env.speed()

                while not done:
                    step_count += 1
                    with torch.no_grad():
                        actions = self._policy_action(obs)
                        actions = self._align_actions_to_env(actions)

                    obs_raw, _, done_raw, info = self.env.step(actions)
                    done = self._done_to_bool(done_raw)
                    obs = self.obs_normalizer(self._compose_policy_obs(obs_raw, info))

                    metrics = self._extract_errors_from_info(info)
                    if metrics is not None:
                        metric_count += 1
                        overall_count += 1
                        for key in self.ERROR_KEYS:
                            sums[key] += metrics[key]
                            overall_sums[key] += metrics[key]

                        if self.print_per_step:
                            print(
                                f"traj={traj_id} episode={episode + 1} step={step_count} | "
                                f"site_pos_err={metrics['site_pos_err']:.8f} | "
                                f"site_vel_err={metrics['site_vel_err']:.8f} | "
                                f"joint_angle_err={metrics['joint_angle_err']:.8f} | "
                                f"l2_sq={metrics['l2_sq']:.8f}"
                            )

                    if self.max_steps is not None and step_count >= int(self.max_steps):
                        break

                    if self.sleep_during_rollout:
                        time.sleep(
                            1.0
                            / (self.env.max_episode_length * self.cfg["visualization"]["speedup"])
                        )

                denom = max(metric_count, 1)
                row = {
                    "episode": episode + 1,
                    "traj_id": traj_id,
                    "speed": float(speed),
                    "steps": step_count,
                    "metric_samples": metric_count,
                    **{key: sums[key] / denom for key in self.ERROR_KEYS},
                }
                rows.append(row)

                print(
                    f"episode={episode + 1} traj_id={traj_id} speed={float(speed):.3f} "
                    f"steps={step_count} samples={metric_count} | "
                    f"site_pos_err={row['site_pos_err']:.8f} | "
                    f"site_vel_err={row['site_vel_err']:.8f} | "
                    f"joint_angle_err={row['joint_angle_err']:.8f} | "
                    f"l2_sq={row['l2_sq']:.8f}"
                )

            if overall_count > 0:
                overall = {key: overall_sums[key] / overall_count for key in self.ERROR_KEYS}
                print(
                    "overall mean | "
                    f"site_pos_err={overall['site_pos_err']:.8f} | "
                    f"site_vel_err={overall['site_vel_err']:.8f} | "
                    f"joint_angle_err={overall['joint_angle_err']:.8f} | "
                    f"l2_sq={overall['l2_sq']:.8f}"
                )

            save_path = self._resolve_output_path()
            if save_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                with open(save_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "episode",
                            "traj_id",
                            "speed",
                            "steps",
                            "metric_samples",
                            *self.ERROR_KEYS,
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Saved imitation error summary to {save_path}")
        finally:
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Output imitation tracking errors.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Path to visualization config.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of trajectories/episodes to evaluate. Default: all trajectories.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max rollout steps per episode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Default: checkpoint directory/imitation_errors.csv. Use 'none' to disable.",
    )
    parser.add_argument(
        "--per-step",
        action="store_true",
        help="Print every step's errors instead of only episode means.",
    )
    parser.add_argument(
        "--sleep",
        action="store_true",
        help="Match visualize.py playback delay during rollout.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Override config and run without the MuJoCo human viewer.",
    )
    args = parser.parse_args()

    visualizer = ImitationErrorVisualizer(
        args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        output=args.output,
        per_step=args.per_step,
        sleep=args.sleep,
        no_render=args.no_render,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
