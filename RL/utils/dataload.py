from pathlib import Path
import pickle
from typing import List, Union, Tuple, Generator
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class MotionData:

    joint_positions: Union[torch.Tensor, np.ndarray]
    device: torch.device = torch.device("cpu")
    stride: int = 2  # ?? 100 Hz ?? -> 50 Hz ??
    target_dt: float = 0.02  # 50 Hz ????

    def __post_init__(self) -> None:
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        else:
            self.joint_positions = self.joint_positions.to(
                self.device, dtype=torch.float32
            )

        self._prepare_amp_buffers()

    def _prepare_amp_buffers(self) -> None:
        data = self.joint_positions
        stride = self.stride
        dt = self.target_dt
        total_frames = data.shape[0]
        feature_dim = 165  # 3 + 12 + 46 + 46 + 60

        if total_frames <= stride:
            self.amp_obs = torch.empty((0, feature_dim), device=self.device)
            self.amp_next_obs = torch.empty_like(self.amp_obs)
            return

        pelvis = data[:, 0:1]
        endpoints = data[:, 1:13]
        joint_angles = data[:, 13:59]
        limb_pos = data[:, 59:119]

        pelvis_vel = torch.zeros_like(pelvis)
        pelvis_vel[stride:] = (pelvis[stride:] - pelvis[:-stride]) / dt

        joint_vel = torch.zeros_like(joint_angles)
        joint_vel[stride:] = (joint_angles[stride:] - joint_angles[:-stride]) / dt

        limb_vel = torch.zeros_like(limb_pos)
        limb_vel[stride:] = (limb_pos[stride:] - limb_pos[:-stride]) / dt

        features = torch.cat(
            (pelvis_vel, endpoints, joint_angles, joint_vel, limb_vel), dim=1
        )

        obs_idx = torch.arange(0, total_frames - stride, device=self.device)
        next_idx = obs_idx + stride

        self.amp_obs = features[obs_idx]
        self.amp_next_obs = features[next_idx]

    def __len__(self) -> int:
        return self.amp_obs.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        return self.amp_obs[indices]

    def get_amp_dataset_next_obs(self, indices: torch.Tensor) -> torch.Tensor:
        return self.amp_next_obs[indices]


class AMPLoader:

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str] = None,
        dataset_weights: List[float] = None,
    ) -> None:

        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)
        if dataset_names is None:
            dataset_names = [p.stem for p in dataset_path_root.glob("*.npz")]
        if dataset_weights is None:
            dataset_weights = [1.0] * len(dataset_names)

        self.motion_data: List[MotionData] = []
        dataset_weight_list: List[float] = []
        for dataset_name, weight in zip(dataset_names, dataset_weights):
            dataset_path = dataset_path_root / f"{dataset_name}.npz"
            md = self.load_data(
                dataset_path,
            )
            if len(md) == 0:
                print(f"[AMPLoader] '{dataset_name}' skipped (insufficient frames for 50 Hz processing).")
                continue
            self.motion_data.append(md)
            dataset_weight_list.append(weight)

        if not self.motion_data:
            raise ValueError("No valid motion datasets were loaded for AMP.")

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weight_list, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list = [], []
        for data in self.motion_data:
            obs_list.append(data.amp_obs)
            next_obs_list.append(data.amp_next_obs)

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        print("REFERENCE_LENGTH: ", self.all_next_obs.shape[0])
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    def load_data(
        self,
        dataset_path: Path,
    ) -> MotionData:

        data = np.load(str(dataset_path), allow_pickle=True)

        return MotionData(
            joint_positions=data["expert_obs"],
            device=self.device,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:

        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]


if __name__ == "__main__":
    loader = AMPLoader(
        device="cuda",
        dataset_path_root="Env/data_test",
        dataset_names=None,
        dataset_weights=None,
    )
    print("=== Testing feed_forward_generator ===")
    num_batches = 10
    batch_size = 40000
    gen = loader.feed_forward_generator(num_mini_batch=num_batches, mini_batch_size=batch_size)
    for i, (obs, next_obs) in enumerate(gen):
        # obs, next_obs -> shape [batch_size, feature_dim]
        print(f"Batch {i}: obs.shape = {obs.shape}, next_obs.shape = {next_obs.shape}")
    print()
