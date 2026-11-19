"""Controlled hip-MAE fine-tuning; lock validation selection before test evaluation."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data import EpisodeWindows
from .model import MomentEstimator
from .train import evaluate, selection_score, train


VARIANTS = {
    "huber_control": {},
    "smooth_l1": {"hip_type": "smooth_l1", "hip_beta_nm": 2.},
    "smooth_l1_low_aux": {"hip_type": "smooth_l1", "hip_beta_nm": 2.,
                          "weights": {"aux": .1, "grf": .05, "contact": .05, "phase": .05, "nll": .02}},
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate_checkpoint(path, data_dir):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = MomentEstimator(**saved["model_args"]).eval()
    model.load_state_dict(saved["model_state_dict"])
    model.uncertainty_scale.copy_(torch.as_tensor(saved.get("uncertainty_scale", [1., 1.])))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    indices = [manifest["extra_names"].index(name) for name in saved["extra_names"]]
    paths = [data_dir / Path(item).name for item in saved["partitions"]["test"]]
    options = saved["config"]["training"]
    dataset = EpisodeWindows(paths, model.window_steps, options["stride_samples"], saved["target"], indices)
    metrics, arrays = evaluate(model, DataLoader(dataset, batch_size=64), "cpu", return_arrays=True,
                               loss_config=options.get("loss", {}))
    raw, applied = [], []
    masks = []
    lookup = {entry["file"]: entry for entry in manifest["episodes"]}
    for episode_path, ends in zip(dataset.paths, dataset.window_ends):
        with np.load(episode_path, allow_pickle=False) as data:
            raw.append(data["tau_bio_raw"][ends, :2])
            applied.append(data["tau_applied"][ends, :2])
        masks.extend([lookup[Path(episode_path).name]["motion_id"]] * len(ends))
    biological = arrays["estimate"] - np.concatenate(applied) if saved["target"] == "total" else arrays["estimate"]
    metrics["instantaneous_bio_mae_nm"] = np.abs(biological - np.concatenate(raw)).mean(axis=0).tolist()
    masks = np.array(masks)
    metrics["by_motion"] = {}
    for motion in sorted(set(masks)):
        selected = arrays["error"][masks == motion]
        metrics["by_motion"][motion] = {"windows": len(selected), "hip_mae_nm": np.abs(selected).mean(axis=0).tolist(),
                                        "hip_rmse_nm": np.sqrt((selected ** 2).mean(axis=0)).tolist()}
    return metrics


def run(config_path, checkpoint_path, output_dir, epochs=25, learning_rate=1e-4):
    torch.set_num_threads(1)
    output, checkpoint_path = Path(output_dir), Path(checkpoint_path)
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive")
    with open(config_path, encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_dir = Path(base_config["collection"]["output_dir"])
    plan = {"source_checkpoint": str(checkpoint_path), "source_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "epochs_per_variant": epochs, "learning_rate": learning_rate, "variants": VARIANTS,
            "selection_metric": "mean validation hip MAE", "optimizer": "fresh Adam for all variants",
            "config": base_config, "test_selection": "Test set excluded from training and variant selection"}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        if json.loads((output / "plan.json").read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing experiment has a different plan; choose another output directory")
    else:
        write_json(output / "plan.json", plan)
    selections = {}
    for name, loss in VARIANTS.items():
        directory = output / name
        directory.mkdir(exist_ok=True)
        config = copy.deepcopy(base_config)
        config["training"].update(output_dir=str(directory), learning_rate=learning_rate, epochs=epochs,
                                  selection_metric="hip_mae", loss=loss)
        (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if not (directory / "completed.json").exists():
            last = directory / "model_last.pt"
            if last.exists():
                previous = torch.load(last, map_location="cpu", weights_only=False)
                remaining = int(source["epoch"]) + epochs - int(previous["epoch"])
                if remaining < 1:
                    raise RuntimeError(f"Training finished without completion metadata in {directory}; verify calibration before resuming")
                train(config, output_dir=directory, epochs=remaining, resume=last, evaluate_test=False)
            else:
                train(config, output_dir=directory, epochs=epochs, resume=checkpoint_path, reset_optimizer=True, evaluate_test=False)
            saved = torch.load(directory / "model_best.pt", map_location="cpu", weights_only=False)
            write_json(directory / "completed.json", {"epoch": saved["epoch"], "validation": saved["validation"]})
        selected = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
        selections[name] = {"checkpoint": str(directory / "model_best.pt"), "epoch": selected["epoch"],
                            "validation_hip_mae_nm": selected["validation"]["hip_mae_nm"],
                            "selection_score": selection_score(selected["validation"], "hip_mae")}
        print(json.dumps({"variant": name, **selections[name]}), flush=True)
    winner = min(selections, key=lambda name: selections[name]["selection_score"])
    selection = {"selected_variant": winner, "selected_checkpoint": selections[winner]["checkpoint"],
                 "criterion": "Mean left/right hip MAE on mass_097 validation episodes only",
                 "variants": selections, "selection_locked_before_test": True}
    # This file is committed before any held-out test data are opened.
    if (output / "selection.json").exists() and json.loads((output / "selection.json").read_text(encoding="utf-8")) != selection:
        raise ValueError("Selection differs from the already locked validation decision")
    write_json(output / "selection.json", selection)
    print("Validation selection locked. Starting final held-out test evaluation.", flush=True)
    test_results = {}
    for name, path in [("initial_checkpoint", checkpoint_path)] + [(name, Path(row["checkpoint"])) for name, row in selections.items()]:
        metrics = evaluate_checkpoint(path, data_dir)
        test_results[name] = metrics
        if name != "initial_checkpoint":
            write_json(Path(path).parent / "metrics_test.json", metrics)
        print(json.dumps({"variant": name, "test_hip_mae_nm": metrics["hip_mae_nm"],
                          "test_hip_p95_nm": metrics["hip_abs_error_p95_nm"],
                          "instantaneous_bio_mae_nm": metrics["instantaneous_bio_mae_nm"]}), flush=True)
    summary = {"plan": plan, "selection": selection, "test_results": test_results,
               "caveat": "One paired optimization seed; same simulation geometry and policy family; no new real-world validation"}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/force_estimation_sim.yaml")
    parser.add_argument("--checkpoint", default="models/force_estimator_sim_v2/model_best.pt")
    parser.add_argument("--output-dir", default="models/force_estimator_mae_ablation")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
