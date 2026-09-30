#!/usr/bin/env python3
import os
import time
import pandas as pd
import wandb
import yaml
import torch
import argparse
from pathlib import Path
from RL.modules.actor_critic_latent_test import ActorCritic
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

    def run(self):
        """Run visualization loop"""
        num_episodes = self.num_traj
        print(f"Starting visualization for {num_episodes} episodes (all trajectories)...")
        if self.cfg['visualization']["wandb"]:
            wandb.init(project="Test_Env",name="baseline")
            
        for episode in range(num_episodes):
            obs, _ = self.env.reset(id=episode)
            obs = self.obs_normalizer(obs)
            episode_reward = 0
            done = False
            speed,id = self.env.speed()
            print(f"Episode {episode + 1} (traj_id={id}) starting with speed: {speed:.2f}")

            count = 0
            import ipdb;ipdb.set_trace()
            while not done:
                count +=1
                with torch.no_grad():
                    actions = self.policy.act(obs)
                    # actions[:,-2:]=actions[:,-2:]*0.0  #disable exo
                obs, reward, done, info = self.env.step(actions)
                # import ipdb;ipdb.set_trace()
                # print(reward,info['reward_info']["imitation"],info['reward_info']["smooth"])
                obs = self.obs_normalizer(obs)
                episode_reward += reward.item()
                # Control playback speed
                time.sleep(1.0 / (self.env.max_episode_length * self.cfg["visualization"]["speedup"]))
            
            print("episode reward =",episode_reward, ", steps =", count) 
            print(info["terminated_info"])
        self.env.close()

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
