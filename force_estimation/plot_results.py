"""Plot the current estimator's held-out results without a baseline model."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EpisodeWindows
from .model import MomentEstimator


BLUE, ORANGE, INK = "#2166ac", "#d97724", "#252b33"
MOTIONS = {"steady": "Steady walking", "acceleration": "Acceleration",
           "deceleration": "Deceleration", "hip_perturbation": "Hip torque perturbation"}


def infer(model, dataset):
    predictions, labels = {}, {}
    with torch.inference_mode():
        for inputs, target in DataLoader(dataset, batch_size=64, shuffle=False):
            for name, value in model(*inputs).items():
                predictions.setdefault(name, []).append(value.numpy())
            for name, value in target.items():
                labels.setdefault(name, []).append(value.numpy())
    return ({name: np.concatenate(values) for name, values in predictions.items()},
            {name: np.concatenate(values) for name, values in labels.items()})


def statistics(prediction, target, select):
    error = prediction["tau_hip"][select] - target["torque"][select, :2]
    truth = target["torque"][select, :2]
    sigma = prediction["sigma_nm"][select]
    return {
        "windows": int(np.sum(select)),
        "hip_mae_nm": np.abs(error).mean(axis=0).tolist(),
        "hip_rmse_nm": np.sqrt((error ** 2).mean(axis=0)).tolist(),
        "hip_abs_error_p95_nm": np.percentile(np.abs(error), 95, axis=0).tolist(),
        "hip_abs_error_p99_nm": np.percentile(np.abs(error), 99, axis=0).tolist(),
        "hip_bias_nm": error.mean(axis=0).tolist(),
        "hip_r2": (1 - (error ** 2).sum(axis=0) /
                   np.maximum(((truth - truth.mean(axis=0)) ** 2).sum(axis=0), 1e-9)).tolist(),
        "one_sigma_coverage": (np.abs(error) <= sigma).mean(axis=0).tolist(),
        "two_sigma_coverage": (np.abs(error) <= 2 * sigma).mean(axis=0).tolist(),
        "aux_mae_nm": np.abs(prediction["tau_aux"][select] - target["torque"][select, 2:]).mean(axis=0).tolist(),
        "grf_mae_n": np.abs(prediction["grf_v"][select] - target["grf_v"][select]).mean(axis=0).tolist(),
        "contact_accuracy": float(((prediction["contact_logit"][select] > 0) == target["contact"][select]).mean()),
    }


def style(axis):
    axis.grid(alpha=.16, linewidth=.7)
    axis.spines[["top", "right"]].set_visible(False)


def save(figure, name, output, pdf):
    for axis in figure.axes:
        style(axis)
    figure.savefig(output / f"{name}.png", dpi=180)
    figure.savefig(output / f"{name}.svg")
    pdf.savefig(figure)
    plt.close(figure)


def plot(checkpoint_path, data_dir, output_dir=None, seconds=6.):
    if seconds <= 0:
        raise ValueError("Plot duration must be positive")
    torch.set_num_threads(1)
    checkpoint_path, data_dir = Path(checkpoint_path), Path(data_dir)
    output = Path(output_dir or checkpoint_path.parent / "results")
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    model = MomentEstimator(**saved["model_args"]).eval()
    model.load_state_dict(saved["model_state_dict"])
    model.uncertainty_scale.copy_(torch.as_tensor(saved.get("uncertainty_scale", [1., 1.])))
    paths = [data_dir / Path(path).name for path in saved["partitions"]["test"]]
    indices = [manifest["extra_names"].index(name) for name in saved["extra_names"]]
    dataset = EpisodeWindows(paths, model.window_steps, 1, saved["target"], indices)
    prediction, target = infer(model, dataset)
    lookup = {entry["file"]: entry for entry in manifest["episodes"]}
    episode_ids = np.concatenate([np.full(len(ends), index) for index, ends in enumerate(dataset.window_ends)])
    ends = np.concatenate(dataset.window_ends)
    time = np.concatenate([episode["time"][last] for episode, last in zip(dataset.episodes, dataset.window_ends)])
    motions = np.array([lookup[Path(dataset.paths[index]).name]["motion_id"] for index in episode_ids])
    stride = saved["config"]["training"]["stride_samples"]
    metric_rows = (ends - (model.window_steps - 1)) % stride == 0
    metrics = statistics(prediction, target, metric_rows)
    motion_names = [name for name in MOTIONS if name in motions]
    motion_names += sorted(set(motions) - set(motion_names))
    metrics["by_motion"] = {name: statistics(prediction, target, metric_rows & (motions == name)) for name in motion_names}
    cutoff = manifest.get("processing", {}).get("anti_alias_hz")
    demand = "Total net joint moment" if saved["target"] == "total" else "Biological joint moment"
    note = f"Held-out test: {len(paths)} episodes | {metrics['windows']} evaluation windows | {cutoff:g} Hz causal labels" if cutoff else f"Held-out test: {len(paths)} episodes | {metrics['windows']} evaluation windows"
    cases = {}
    for name in motion_names:
        first = int(episode_ids[np.flatnonzero(motions == name)[0]])
        selected = np.flatnonzero(episode_ids == first)
        cases[name] = selected[time[selected] <= time[selected[0]] + seconds]
    steady = cases.get("steady", next(iter(cases.values())))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9,
                         "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#b1b7be",
                         "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
    with PdfPages(output / "estimator_results.pdf") as pdf:
        history_path = checkpoint_path.parent / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else saved.get("history", [])
        if history:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
            epoch = [entry["epoch"] for entry in history]
            axes[0].plot(epoch, [entry["train_loss"] for entry in history], color=BLUE, label="Training")
            axes[0].plot(epoch, [entry["validation"]["loss"] for entry in history], color=ORANGE, label="Validation")
            axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Weighted loss (includes Gaussian NLL)")
            for side, color in enumerate((BLUE, ORANGE)):
                axes[1].plot(epoch, [entry["validation"]["hip_mae_nm"][side] for entry in history],
                             color=color, label=("Left hip", "Right hip")[side])
            axes[1].set(title="Validation hip moment error", xlabel="Epoch", ylabel="MAE (N m)")
            for axis in axes:
                axis.axvline(saved["epoch"], color=INK, ls=":", lw=1, label=f"Selected epoch {saved['epoch']}")
                axis.legend()
            fig.suptitle("Current estimator | Training and validation", fontsize=15)
            save(fig, "01_training", output, pdf)

        fig, axes = plt.subplots(len(cases), 2, figsize=(12, 2.5 * len(cases) + .6), squeeze=False, layout="constrained")
        for row, (motion, selected) in enumerate(cases.items()):
            for side in range(2):
                axis = axes[row, side]
                estimate, sigma = prediction["tau_hip"][selected, side], prediction["sigma_nm"][selected, side]
                axis.fill_between(time[selected], estimate - 2 * sigma, estimate + 2 * sigma,
                                  color=BLUE, alpha=.15, label="Prediction +/-2 sigma")
                axis.plot(time[selected], target["torque"][selected, side], color=INK, lw=1.5, label="Simulation label")
                axis.plot(time[selected], estimate, color=BLUE, lw=1.3, label="Estimator")
                axis.set(title=f"{MOTIONS.get(motion, motion)} | {('Left', 'Right')[side]} hip",
                         ylabel="Moment (N m)", xlabel="Simulation time (s)")
        handles, legends = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legends, loc="outside lower center", ncol=3)
        fig.suptitle(f"{demand} | {cutoff:g} Hz causal labels\nFirst test episode per condition; 100 Hz predictions; +/-2 sigma is a model interval", fontsize=13)
        save(fig, "02_hip_waveforms", output, pdf)

        error = prediction["tau_hip"][metric_rows] - target["torque"][metric_rows, :2]
        sigma = prediction["sigma_nm"][metric_rows]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for side, color in enumerate((BLUE, ORANGE)):
            truth, estimate = target["torque"][metric_rows, side], prediction["tau_hip"][metric_rows, side]
            low, high = min(truth.min(), estimate.min()), max(truth.max(), estimate.max())
            axis = axes[0, side]
            axis.scatter(truth, estimate, s=10, alpha=.4, color=color, linewidths=0, rasterized=True)
            axis.plot([low, high], [low, high], color=INK, ls="--", lw=1)
            axis.set(title=f"{('Left', 'Right')[side]} hip | MAE {metrics['hip_mae_nm'][side]:.2f} N m | R2 {metrics['hip_r2'][side]:.3f}",
                     xlabel="Simulation label (N m)", ylabel="Estimated moment (N m)", aspect="equal")
            absolute = np.sort(np.abs(error[:, side]))
            axes[1, 0].plot(absolute, np.arange(1, len(absolute) + 1) / len(absolute), color=color, label=("Left", "Right")[side])
            multiples = np.linspace(0, 3, 61)
            coverage = np.array([(np.abs(error[:, side]) <= k * sigma[:, side]).mean() for k in multiples])
            axes[1, 1].plot(multiples, coverage, color=color, label=("Left", "Right")[side])
        axis = axes[0, 2]
        positions = np.arange(len(motion_names))
        for side, color in enumerate((BLUE, ORANGE)):
            axis.bar(positions + (side - .5) * .36, [metrics["by_motion"][name]["hip_mae_nm"][side] for name in motion_names],
                     .36, color=color, label=("Left", "Right")[side])
        axis.set(xticks=positions, xticklabels=[MOTIONS.get(name, name).replace(" ", "\n", 1) for name in motion_names],
                 ylabel="MAE (N m)", title="Error by test condition")
        axis.legend()
        axes[1, 0].axhline(.95, color=INK, ls="--", lw=1)
        axes[1, 0].set(title="Absolute error distribution", xlabel="Absolute error (N m)", ylabel="Cumulative fraction", ylim=(0, 1.02))
        axes[1, 0].legend()
        axes[1, 1].plot(multiples, [math.erf(k / math.sqrt(2)) for k in multiples], color=INK, ls="--", label="Gaussian reference")
        axes[1, 1].set(title="Uncertainty coverage", xlabel="Predicted sigma multiplier", ylabel="Observed coverage", ylim=(0, 1.02))
        axes[1, 1].legend()
        axes[1, 2].axis("off")
        rows = [("MAE (N m)", "hip_mae_nm"), ("RMSE (N m)", "hip_rmse_nm"), ("95th error (N m)", "hip_abs_error_p95_nm"),
                ("99th error (N m)", "hip_abs_error_p99_nm"), ("Bias (N m)", "hip_bias_nm"), ("R2", "hip_r2"), ("+/-2 sigma (%)", "two_sigma_coverage")]
        cells = [[label, *[format(value * (100 if key == "two_sigma_coverage" else 1),
                                 ".3f" if key == "hip_r2" else ".2f") for value in metrics[key]]] for label, key in rows]
        table = axes[1, 2].table(cellText=cells, colLabels=["Test metric", "Left", "Right"], loc="center", cellLoc="center", colWidths=[.52, .24, .24])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#e8eef5" if row == 0 else ("#f4f6f8" if row % 2 else "white"))
        fig.suptitle(f"Current estimator | Test accuracy\n{note}", fontsize=14)
        save(fig, "03_test_accuracy", output, pdf)

        fig, axes = plt.subplots(3, 2, figsize=(12, 8.6), layout="constrained")
        for row, joint in enumerate(("Knee", "Ankle", "Vertical GRF")):
            for side in range(2):
                axis = axes[row, side]
                if row < 2:
                    channel = 2 * row + side
                    truth, estimate = target["torque"][steady, channel + 2], prediction["tau_aux"][steady, channel]
                    metric, unit = metrics["aux_mae_nm"][channel], "N m"
                else:
                    truth, estimate = target["grf_v"][steady, side], prediction["grf_v"][steady, side]
                    metric, unit = metrics["grf_mae_n"][side], "N"
                axis.plot(time[steady], truth, color=INK, label="Simulation label")
                axis.plot(time[steady], estimate, color=BLUE, label="Estimator")
                axis.set(title=f"{('Left', 'Right')[side]} {joint.lower()} | all-test MAE {metric:.2f} {unit}",
                         xlabel="Simulation time (s)", ylabel=f"{'Moment' if row < 2 else 'Force'} ({unit})")
        handles, legends = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legends, loc="outside lower center", ncol=2)
        fig.suptitle(f"Auxiliary outputs | First steady test episode\n{cutoff:g} Hz causal labels; titles report metrics across the full test set", fontsize=13)
        save(fig, "04_auxiliary_outputs", output, pdf)

        fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), layout="constrained")
        for side in range(2):
            probability = 1 / (1 + np.exp(-np.clip(prediction["contact_logit"][steady, side], -50, 50)))
            axes[0, side].step(time[steady], target["contact"][steady, side], color=INK, where="post", label="Contact label")
            axes[0, side].plot(time[steady], probability, color=BLUE, label="Predicted probability")
            axes[0, side].axhline(.5, color="gray", ls=":", lw=1)
            axes[0, side].set(title=f"{('Left', 'Right')[side]} foot contact", xlabel="Simulation time (s)", ylabel="Probability / contact", ylim=(-.05, 1.05))
            valid = target["phase_mask"][steady, side].astype(bool)
            for values, color, label in ((target["phase"], INK, "Contact-derived phase label"), (prediction["phase"], BLUE, "Predicted phase")):
                phase = (np.arctan2(values[steady, side, 0], values[steady, side, 1]) / (2 * np.pi)) % 1
                wraps = np.r_[False, np.abs(np.diff(phase)) > .5]
                phase[~valid | wraps] = np.nan
                axes[1, side].plot(time[steady], phase, color=color, label=label)
            axes[1, side].set(title=f"{('Left', 'Right')[side]} gait phase", xlabel="Simulation time (s)", ylabel="Gait cycle (0-1)", ylim=(-.05, 1.05))
        handles, legends = [], []
        for row in range(2):
            row_handles, row_legends = axes[row, 0].get_legend_handles_labels()
            handles.extend(row_handles)
            legends.extend(row_legends)
        fig.legend(handles, legends, loc="outside lower center", ncol=2)
        fig.suptitle(f"Contact and phase | First steady test episode\nFull-test contact accuracy: {100 * metrics['contact_accuracy']:.2f}% | Phase labels valid between detected contact onsets", fontsize=13)
        save(fig, "05_contact_phase", output, pdf)

    summary = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
               "epoch": saved["epoch"], "target": saved["target"], "label_lowpass_hz": cutoff,
               "evaluation_stride_samples": stride, "waveform_stride_samples": 1,
               "waveform_selection": "First test episode in each condition, starting at its first valid window; no time shift or extra smoothing",
               "test_episodes": dataset.paths, "metrics": metrics}
    (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(output / "test_predictions.npz", time=time, episode_id=episode_ids, window_end=ends,
                        evaluation_mask=metric_rows, **{f"pred_{key}": value for key, value in prediction.items()},
                        **{f"label_{key}": value for key, value in target.items()})
    print(json.dumps({"output_dir": str(output), "test_windows": metrics["windows"], "hip_mae_nm": metrics["hip_mae_nm"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/force_estimator_sim_v2/model_best.pt")
    parser.add_argument("--data-dir", default="data/force_estimation_sim_v2_calibrated")
    parser.add_argument("--output-dir")
    parser.add_argument("--seconds", type=float, default=6.)
    arguments = parser.parse_args()
    plot(arguments.checkpoint, arguments.data_dir, arguments.output_dir, arguments.seconds)


if __name__ == "__main__":
    main()
