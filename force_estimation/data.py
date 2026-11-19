"""Episode-level partitions and causal windows without cross-episode leakage."""

import bisect
import numpy as np
import torch
from torch.utils.data import Dataset


def split_episodes(paths, validation_fraction, test_fraction, seed, groups=None):
    group_for_path = None if groups is None else dict(zip(map(str, paths), groups))
    paths = sorted(str(path) for path in paths)
    if len(paths) < 5 or validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("At least five episodes and positive validation/test fractions summing to <1 are required")
    rng = np.random.default_rng(seed)
    units = paths if group_for_path is None else sorted(set(group_for_path.values()))
    if len(units) < 5:
        raise ValueError("Grouped validation requires at least five distinct groups")
    indices = rng.permutation(len(units))
    n_val = max(1, int(round(len(units) * validation_fraction)))
    n_test = max(1, int(round(len(units) * test_fraction)))
    if n_val + n_test >= len(units):
        raise ValueError("No training episodes remain after partitioning")
    assignment = {"validation": [units[i] for i in indices[:n_val]],
                  "test": [units[i] for i in indices[n_val:n_val + n_test]],
                  "train": [units[i] for i in indices[n_val + n_test:]]}
    if group_for_path is None:
        return assignment
    return {name: [path for path in paths if group_for_path[path] in selected]
            for name, selected in assignment.items()}


class EpisodeWindows(Dataset):
    def __init__(self, paths, window_steps, stride, target, extra_indices=None):
        if target not in ("total", "bio") or stride < 1 or window_steps < 1:
            raise ValueError("Invalid dataset target, window or stride")
        self.window_steps, self.stride, self.target = int(window_steps), int(stride), target
        self.episodes, self.paths, self.ends, self.window_ends = [], [], [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                episode = {name: data[name].copy() for name in
                           ("time", "imu", "imu_mask", "extra", f"tau_{target}", "grf_v", "contact", "phase", "phase_mask")}
                valid = data["sample_valid"].astype(bool) if "sample_valid" in data else np.ones(len(data["time"]), dtype=bool)
            if extra_indices is not None:
                episode["extra"] = episode["extra"][:, extra_indices]
            endpoints = np.arange(self.window_steps - 1, len(episode["imu"]), self.stride)
            invalid_prefix = np.concatenate(([0], np.cumsum(~valid)))
            endpoints = endpoints[(invalid_prefix[endpoints + 1] - invalid_prefix[endpoints + 1 - self.window_steps]) == 0]
            count = len(endpoints)
            if count:
                self.episodes.append(episode)
                self.paths.append(str(path))
                self.window_ends.append(endpoints)
                self.ends.append(count + (self.ends[-1] if self.ends else 0))
        if not self.ends:
            raise ValueError("No complete input windows in this partition")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode_index = bisect.bisect_right(self.ends, index)
        previous = self.ends[episode_index - 1] if episode_index else 0
        last = int(self.window_ends[episode_index][index - previous])
        episode = self.episodes[episode_index]
        window = slice(last + 1 - self.window_steps, last + 1)
        inputs = tuple(torch.from_numpy(episode[name][window]) for name in ("imu", "imu_mask", "extra"))
        labels = {"torque": torch.from_numpy(episode[f"tau_{self.target}"][last]),
                  **{name: torch.from_numpy(episode[name][last]) for name in ("grf_v", "contact", "phase", "phase_mask")}}
        return inputs, labels

    def extra_statistics(self):
        features = np.concatenate([episode["extra"] for episode in self.episodes]).astype(np.float64)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-3, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    def sampling_weights(self, peak_weight):
        measures = []
        for episode, endpoints in zip(self.episodes, self.window_ends):
            torque = episode[f"tau_{self.target}"][:, :2]
            amplitude = np.max(np.abs(torque[endpoints]), axis=1)
            change = np.max(np.abs(torque[endpoints] - torque[np.maximum(endpoints - 1, 0)]), axis=1)
            measures.append(np.stack((amplitude, change), axis=1))
        measures = np.concatenate(measures)
        thresholds = np.quantile(measures, .9, axis=0)
        peaks = np.any(measures >= thresholds, axis=1)
        return torch.as_tensor(1 + float(peak_weight) * peaks, dtype=torch.double)
