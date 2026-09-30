#!/usr/bin/env python3
import os
import time
import pandas as pd
import wandb
import yaml
import torch
import argparse
from pathlib import Path
# from RL.modules.actor_critic_latent_test import ActorCritic
from RL.modules.actor_critic_latent_exo import ActorCritic
# from RL.modules import ActorCritic
from RL.modules import EmpiricalNormalization
from env_wrapper_eval import GymMujocoWrapper
from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager
from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11
from dataclasses import dataclass
from RL.utils.utils import AMPNormalizer
torch.serialization.add_safe_globals([AMPNormalizer])

class PolicyVisualizer:
    def __init__(self, cfg_path):
        self.load_config(cfg_path)
        self.setup_environment()
        self.num_obs = self.env.num_obs
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs], until=1.0e8).to(self.device)
        self.base_dir = os.path.dirname(self.cfg["policy"]["checkpoint"])
        self.txt_path = os.path.join(self.base_dir, self.cfg["visualization"]["save_name"])
        self.load_policy()

    def load_config(self, cfg_path):
        """Load visualization configuration"""
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self):
        with open("Env/MFG_Musculoskelet_V11/cfg/mujocoenv_default.yaml", "r",encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
            
        tm_args = base_cfg["traj_manager"]    
        shared_tm = TrajectoryManager(
            data_path=tm_args["data_path"],
            sample_frequency=float(tm_args["sample_frequency"]),
            speed_range=(float(tm_args["speed_range"][0]), float(tm_args["speed_range"][1])),
            verbose=bool(tm_args["verbose"])
        )
        self.num_traj = shared_tm.length()

        base_cfg = dict(base_cfg)
        base_cfg["traj_manager"] = shared_tm
        
        self.env = MFG_Musculoskeletal_V11(
            config=base_cfg,
            render_mode="human" if self.cfg["env"]["render_mode"] == "human" else None
        )
        # import ipdb;ipdb.set_trace()
        self.env = GymMujocoWrapper(
            self.env,
            device=self.device,
            is_finite_horizon=False
        )

    def load_policy(self):
        """Load trained policy network"""
        self.policy = ActorCritic(
            num_actions=self.env.num_actions,
            num_actor_obs=self.env.num_obs,
            num_critic_obs=self.env.num_obs,
            actor_hidden_dims=self.cfg["policy"]["hidden_dims"],
            critic_hidden_dims=self.cfg["policy"]["hidden_dims"],
            muscle_action_dim=self.cfg["policy"].get("muscle_action_dim"),
            latent_cov_scale=self.cfg["policy"].get("latent_cov_scale", 1.0)
        ).to(self.device)
        
        if torch.cuda.is_available():
            checkpoint = torch.load(self.cfg["policy"]["checkpoint"], weights_only=False)
        else:
            checkpoint = torch.load(self.cfg["policy"]["checkpoint"],map_location=torch.device('cpu'))
        
        self.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        self.obs_normalizer.eval()
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        print(f"Loaded policy from {self.cfg['policy']['checkpoint']}")

    def _get_metric(self, info, path):
        cur = info
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                raise KeyError(f"Missing key '{key}' in info path: {'/'.join(path)}")
            cur = cur[key]
        return float(cur)

    def run(self):
        """Run evaluation loop over all trajectories and multiple trials each."""
        num_trajectories = self.num_traj
        num_trials = int(self.cfg["visualization"].get("trials_per_traj", 50))
        print(f"Evaluating {num_trajectories} trajectories, {num_trials} trials each...")
        if self.cfg['visualization']["wandb"]:
            wandb.init(project="Test_Env",name="baseline")

        metric_paths = {
            "pos_err": ("reward_info", "imitation", "site_info", "pos_err"),
            "ang_err": ("reward_info", "imitation", "joint_info", "ang_err"),
            "l1": ("reward_info", "smooth", "energy", "metrics", "l1"),
            "l2_sq": ("reward_info", "smooth", "energy", "metrics", "l2_sq"),
            "metab_cost": ("reward_info", "smooth", "energy", "metrics", "metab_cost"),
        }

        results = []
        for traj_id in range(num_trajectories):
            success_count = 0
            traj_metric_sums = {k: 0.0 for k in metric_paths}
            traj_step_sum = 0

            for trial in range(num_trials):
                obs, _ = self.env.reset(id=traj_id)
                obs = self.obs_normalizer(obs)
                done = False
                steps = 0
                trial_metric_sums = {k: 0.0 for k in metric_paths}
                last_info = None
                import ipdb; ipdb.set_trace()
                while not done:
                    steps += 1
                    with torch.no_grad():
                        actions = self.policy.act(obs)
                    obs, reward, done, info = self.env.step(actions)
                    done = bool(done.item()) if hasattr(done, "item") else bool(done)
                    obs = self.obs_normalizer(obs)
                    last_info = info

                    for k, path in metric_paths.items():
                        trial_metric_sums[k] += self._get_metric(info, path)

                import ipdb; ipdb.set_trace()
                terminated_info = (last_info or {}).get("terminated_info", {}) or {}
                terminated_flag = any(bool(v) for v in terminated_info.values())
                success = (not terminated_flag)
                if success:
                    success_count += 1

                steps_div = max(steps, 1)
                for k in metric_paths:
                    traj_metric_sums[k] += trial_metric_sums[k] / steps_div
                traj_step_sum += steps

            trials_div = max(num_trials, 1)
            traj_metrics_avg = {k: v / trials_div for k, v in traj_metric_sums.items()}
            success_rate = success_count / trials_div
            avg_steps = traj_step_sum / trials_div

            results.append(
                {
                    "traj_id": traj_id,
                    "success_rate": success_rate,
                    "avg_steps": avg_steps,
                    **traj_metrics_avg,
                }
            )

            print(
                f"traj {traj_id} | success_rate={success_rate:.3f} | "
                f"pos_err={traj_metrics_avg['pos_err']:.6f} | "
                f"ang_err={traj_metrics_avg['ang_err']:.6f} | "
                f"l1={traj_metrics_avg['l1']:.6f} | "
                f"l2_sq={traj_metrics_avg['l2_sq']:.6f} | "
                f"metab_cost={traj_metrics_avg['metab_cost']:.6f}"
            )

        self.env.close()

        df = pd.DataFrame(results)
        save_name = str(self.cfg["visualization"].get("save_name", "")).strip()
        if (not save_name) or (save_name.lower() in {"none", "none.txt"}):
            save_name = "traj_eval.csv"
        save_path = os.path.join(self.base_dir, save_name)
        df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="RSL-RL Policy Visualizer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    visualizer = PolicyVisualizer(args.config)
    visualizer.run()
    

if __name__ == "__main__":
    main()
