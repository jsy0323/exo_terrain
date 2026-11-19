"""Compare estimator pipelines on the same held-out simulation conditions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EpisodeWindows
from .features import EncoderFilter, FeatureBuilder
from .model import MomentEstimator
from .train import evaluate


def compare(checkpoint_path, baseline_path, data_dir, output_dir=None):
    torch.set_num_threads(1)
    checkpoint_path = Path(checkpoint_path)
    output = Path(output_dir or checkpoint_path.parent / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    paths = checkpoint["partitions"]["test"]
    results, predictions = {}, {}
    for name, path in (("baseline", baseline_path), ("improved", checkpoint_path)):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = MomentEstimator(**saved["model_args"]).eval()
        model.load_state_dict(saved["model_state_dict"])
        model.uncertainty_scale.copy_(torch.as_tensor(saved.get("uncertainty_scale", [1., 1.])))
        indices = [manifest["extra_names"].index(feature) for feature in saved["extra_names"]]
        dataset = EpisodeWindows(paths, model.window_steps, saved["config"]["training"]["stride_samples"], saved["target"], indices)
        if name == "baseline":
            for episode, episode_path in zip(dataset.episodes, dataset.paths):
                with np.load(episode_path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata"]))
                    builder = FeatureBuilder(saved["config"]["sample_rate_hz"], saved["config"]["fsr"]["calibration_value"],
                                             metadata["subject"], include_exo_features=False)
                    encoder = EncoderFilter(saved["config"]["sample_rate_hz"], saved["config"]["encoder"]["cutoff_hz"])
                    episode["extra"] = np.stack([builder.step(imu, mask, encoder.step(angle[:2]), fsr)
                                                  for imu, mask, angle, fsr in zip(data["imu"], data["imu_mask"], data["encoder"], data["fsr"])])
        metrics, arrays = evaluate(model, DataLoader(dataset, batch_size=64), "cpu", return_arrays=True,
                                   loss_config=saved["config"]["training"].get("loss", {}))
        applied, true_bio, raw_bio = [], [], []
        for episode_path, endpoints in zip(dataset.paths, dataset.window_ends):
            with np.load(episode_path, allow_pickle=False) as data:
                applied.append(data["tau_applied"][endpoints, :2])
                true_bio.append(data["tau_bio"][endpoints, :2])
                raw_bio.append(data["tau_bio_raw"][endpoints, :2])
        estimate_bio = arrays["estimate"].copy()
        if saved["target"] == "total":
            estimate_bio -= np.concatenate(applied)
        metrics["instantaneous_bio_mae_nm"] = np.mean(np.abs(estimate_bio - np.concatenate(raw_bio)), axis=0).tolist()
        results[name] = metrics
        predictions[name] = estimate_bio
        if name == "improved":
            first_count = len(dataset.window_ends[0])
            times = dataset.episodes[0]["time"][dataset.window_ends[0]]
            label = np.concatenate(true_bio)[:first_count]
    summary = {"comparison": "same held-out mass-profile episodes with AHRS inputs; each model uses its own feature processing",
               "label_bandwidth": manifest.get("processing", {}).get("anti_alias_hz"),
               "models": results, "new_best_epoch": checkpoint["epoch"]}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    history = json.loads((checkpoint_path.parent / "history.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    epochs = [entry["epoch"] for entry in history]
    axes[0, 0].plot(epochs, [entry["train_loss"] for entry in history], label="Train")
    axes[0, 0].plot(epochs, [entry["validation"]["loss"] for entry in history], label="Held-out mass validation")
    axes[0, 0].set(title="Simulation pipeline training", xlabel="Epoch", ylabel="Weighted objective")
    for side, title in enumerate(("Left hip", "Right hip")):
        axes[0, 1].plot(epochs, [entry["validation"]["hip_mae_nm"][side] for entry in history], label=title)
    axes[0, 1].set(title="Validation net hip moment error", xlabel="Epoch", ylabel="MAE (N m)")
    visible = times <= times[0] + 6
    for side, title in enumerate(("Left hip", "Right hip")):
        axis = axes[1, side]
        axis.plot(times[visible], label[visible, side], color="black", label="Filtered simulation label")
        axis.plot(times[visible], predictions["baseline"][:first_count, side][visible], label="Earlier estimator", alpha=.8)
        axis.plot(times[visible], predictions["improved"][:first_count, side][visible], label="Improved simulation pipeline")
        axis.set(title=f"{title}, unseen mass profile", xlabel="Simulation time (s)", ylabel="Biological moment (N m)")
    for axis in axes.ravel():
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/force_estimator_sim_v2/model_best.pt")
    parser.add_argument("--baseline", default="models/force_estimator_human_only/model_best.pt")
    parser.add_argument("--data-dir", default="data/force_estimation_sim_v2_calibrated")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    compare(arguments.checkpoint, arguments.baseline, arguments.data_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
