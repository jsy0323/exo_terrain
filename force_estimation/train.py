"""Train the wearable estimator and evaluate an untouched episode-level test set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from .data import EpisodeWindows, split_episodes
from .features import IMU_NAMES, JOINT_NAMES, feature_names
from .model import MomentEstimator, estimator_loss


def selection_score(validation, metric="loss"):
    if metric == "loss":
        score = float(validation["loss"])
    elif metric == "hip_mae":
        score = float(np.mean(validation["hip_mae_nm"]))
    else:
        raise ValueError(f"Unknown checkpoint selection metric: {metric}")
    if not np.isfinite(score):
        raise FloatingPointError("Non-finite checkpoint selection score")
    return score


def evaluate(model, loader, device, return_arrays=False, loss_config=None):
    model.eval()
    total_loss, count = 0.0, 0
    errors, auxiliary, ground_force, sigmas, truths = [], [], [], [], []
    contact_correct, contact_count = 0, 0
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = [value.to(device) for value in inputs]
            target = {name: value.to(device) for name, value in target.items()}
            prediction = model(*inputs)
            loss, _ = estimator_loss(prediction, target, model.torque_scale_nm, model.grf_scale_n, loss_config)
            size = len(inputs[0])
            total_loss += float(loss) * size
            count += size
            errors.append((prediction["tau_hip"] - target["torque"][:, :2]).cpu().numpy())
            truths.append(target["torque"][:, :2].cpu().numpy())
            auxiliary.append((prediction["tau_aux"] - target["torque"][:, 2:]).cpu().numpy())
            ground_force.append((prediction["grf_v"] - target["grf_v"]).cpu().numpy())
            sigmas.append(prediction["sigma_nm"].cpu().numpy())
            contact_correct += int(((prediction["contact_logit"] > 0) == target["contact"].bool()).sum())
            contact_count += target["contact"].numel()
    error, sigma = np.concatenate(errors), np.concatenate(sigmas)
    truth = np.concatenate(truths)
    result = {"loss": total_loss / count, "windows": count,
            "hip_mae_nm": np.mean(np.abs(error), axis=0).tolist(),
            "hip_rmse_nm": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "aux_mae_nm": np.mean(np.abs(np.concatenate(auxiliary)), axis=0).tolist(),
            "grf_mae_n": np.mean(np.abs(np.concatenate(ground_force)), axis=0).tolist(),
            "contact_accuracy": contact_correct / contact_count,
            "sigma_mean_nm": sigma.mean(axis=0).tolist(),
            "one_sigma_coverage": (np.abs(error) <= sigma).mean(axis=0).tolist(),
            "two_sigma_coverage": (np.abs(error) <= 2 * sigma).mean(axis=0).tolist(),
            "hip_abs_error_p95_nm": np.percentile(np.abs(error), 95, axis=0).tolist(),
            "hip_abs_error_p99_nm": np.percentile(np.abs(error), 99, axis=0).tolist(),
            "hip_bias_nm": np.mean(error, axis=0).tolist(),
            "hip_r2": (1 - np.sum(error ** 2, axis=0) / np.maximum(np.sum((truth - truth.mean(axis=0)) ** 2, axis=0), 1e-9)).tolist()}
    if return_arrays:
        return result, {"error": error, "sigma": sigma, "truth": truth, "estimate": truth + error}
    return result


def train(config, data_dir=None, output_dir=None, epochs=None, resume=None, reset_optimizer=False, evaluate_test=True):
    torch.set_num_threads(1)
    options = config["training"]
    loss_config = options.get("loss", {})
    selection_metric = options.get("selection_metric", "loss")
    if selection_metric not in ("loss", "hip_mae"):
        raise ValueError("selection_metric must be loss or hip_mae")
    seed = int(options["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(options["device"])
    resumed = None
    saved_partitions = None
    if resume:
        resume = Path(resume)
        resumed = torch.load(resume, map_location=device, weights_only=False)
        saved_partitions = resumed.get("partitions")
        if saved_partitions is None:
            saved_partitions = json.loads((resume.parent / "split.json").read_text(encoding="utf-8"))
    default_output = resume.parent if resume else options["output_dir"]
    output = Path(output_dir or default_output)
    output.mkdir(parents=True, exist_ok=True)
    continuing_same_run = resume is not None and output.resolve() == resume.parent.resolve()
    if continuing_same_run and (reset_optimizer or resumed["config"]["training"].get("loss", {}) != loss_config):
        raise ValueError("Changing the objective or resetting the optimizer requires a new output directory")
    if (output / "model_best.pt").exists() and not continuing_same_run:
        raise FileExistsError(f"A trained model already exists in {output}. Choose another output directory.")
    default_data = Path(saved_partitions["train"][0]).parent if saved_partitions else config["collection"]["output_dir"]
    data_path = Path(data_dir or default_data)
    manifest = json.loads((data_path / "manifest.json").read_text(encoding="utf-8"))
    selected_features = feature_names(bool(options.get("include_exo_features", False)))
    stored_features = manifest["extra_names"]
    if (manifest["sample_rate_hz"] != config["sample_rate_hz"]
            or len(stored_features) != len(set(stored_features))
            or any(name not in stored_features for name in selected_features)):
        raise ValueError("Dataset sample rate or feature schema differs from the training configuration")
    extra_indices = [stored_features.index(name) for name in selected_features]
    entries = [entry for entry in manifest["episodes"] if entry.get("accepted", True)]
    if saved_partitions is None:
        group_key = options.get("group_by")
        groups = None if group_key is None else [entry[group_key] for entry in entries]
        if group_key and options.get("test_groups"):
            validation_groups = set(options["validation_groups"])
            test_groups = set(options["test_groups"])
            if validation_groups & test_groups or not (validation_groups | test_groups).issubset(set(groups)):
                raise ValueError("Configured holdout groups must exist and be disjoint")
            partitions = {"train": [], "validation": [], "test": []}
            for entry, group in zip(entries, groups):
                partition = "test" if group in test_groups else "validation" if group in validation_groups else "train"
                partitions[partition].append(str(data_path / entry["file"]))
        else:
            partitions = split_episodes([data_path / entry["file"] for entry in entries],
                                        options["validation_fraction"], options["test_fraction"], seed, groups)
    else:
        partitions = {name: [str(data_path / Path(path).name) for path in paths]
                      for name, paths in saved_partitions.items()}
        partition_files = [Path(path).name for paths in partitions.values() for path in paths]
        manifest_files = [entry["file"] for entry in entries]
        if (len(partition_files) != len(set(partition_files)) or set(partition_files) != set(manifest_files)
                or resumed["collection_policy_sha256"] != manifest["policy_sha256"]
                or resumed.get("processing", {}) != manifest.get("processing", {})):
            raise ValueError("Resume requires the saved episode partition and the original collection policy")
    model_args = {"sample_rate_hz": config["sample_rate_hz"], "window_s": config["window_s"],
                  "extra_dim": len(selected_features), "dropout": options["dropout"],
                  "torque_scale_nm": options["torque_scale_nm"], "grf_scale_n": options["grf_scale_n"]}
    model = MomentEstimator(**model_args).to(device)
    if resumed is not None:
        if (resumed["model_args"] != model_args or resumed["extra_names"] != list(selected_features)
                or resumed["target"] != options["target"]):
            raise ValueError("Resume checkpoint architecture, features or target differs from the configuration")
        model.load_state_dict(resumed["model_state_dict"])
    datasets = {name: EpisodeWindows(paths, model.window_steps, options["stride_samples"], options["target"], extra_indices)
                for name, paths in partitions.items() if evaluate_test or name != "test"}
    if resumed is None:
        mean, std = datasets["train"].extra_statistics()
        model.extra_mean.copy_(torch.as_tensor(mean, device=device))
        model.extra_std.copy_(torch.as_tensor(std, device=device))
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if float(options.get("peak_weight", 0)) > 0:
        sampler = WeightedRandomSampler(datasets["train"].sampling_weights(options["peak_weight"]),
                                        len(datasets["train"]), replacement=True, generator=generator)
    loaders = {name: DataLoader(dataset, batch_size=int(options["batch_size"]), shuffle=name == "train",
                               num_workers=0, generator=generator if name == "train" else None)
               for name, dataset in datasets.items()}
    if sampler is not None:
        loaders["train"] = DataLoader(datasets["train"], batch_size=int(options["batch_size"]),
                                      sampler=sampler, num_workers=0, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options["learning_rate"]))
    if resumed is not None:
        if "optimizer_state_dict" in resumed and not reset_optimizer:
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = float(options["learning_rate"])
            print(f"Restored model and Adam optimizer from {resume}", flush=True)
        else:
            print(f"Restored model from {resume}; optimizer reset or optimizer state unavailable. "
                  "Continuing with a freshly initialized Adam optimizer.", flush=True)
        if "torch_rng_state" in resumed:
            torch.set_rng_state(resumed["torch_rng_state"].cpu())
            generator.set_state(resumed["loader_rng_state"].cpu())
            np.random.set_state(resumed["numpy_rng_state"])
            if device.type == "cuda" and resumed.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in resumed["cuda_rng_state"]])
    (output / "split.json").write_text(json.dumps(partitions, indent=2), encoding="utf-8")
    start_epoch = int(resumed["epoch"]) if resumed is not None else 0
    history = resumed.get("history", []) if resumed is not None and not reset_optimizer else []
    if resumed is not None and not reset_optimizer and not history and (resume.parent / "history.json").exists():
        history = [entry for entry in json.loads((resume.parent / "history.json").read_text(encoding="utf-8"))
                   if entry["epoch"] <= start_epoch]
    best_validation = float("inf")
    best_score = float("inf")
    if resumed is not None:
        baseline = resumed
        if continuing_same_run and (output / "model_best.pt").exists():
            baseline = torch.load(output / "model_best.pt", map_location=device, weights_only=False)
        changed_objective = baseline["config"]["training"].get("loss", {}) != loss_config
        if changed_objective or reset_optimizer:
            # Evaluating the initialization must not advance the training RNG.
            state = torch.get_rng_state()
            validation = evaluate(model, loaders["validation"], device, loss_config=loss_config)
            torch.set_rng_state(state)
            baseline = dict(baseline, config=config, validation=validation, history=history,
                            optimizer_state_dict=optimizer.state_dict(),
                            selection_metric=selection_metric, origin_checkpoint=str(resume))
            baseline.pop("uncertainty_scale", None)
            baseline.pop("uncertainty_calibration", None)
        best_validation = float(baseline["validation"]["loss"])
        best_score = selection_score(baseline["validation"], selection_metric)
        if not (output / "model_best.pt").exists():
            temporary = output / "model_best.pt.tmp"
            torch.save(baseline, temporary)
            temporary.replace(output / "model_best.pt")
    epoch_count = int(epochs if epochs is not None else options["epochs"])
    if epoch_count < 1:
        raise ValueError("epochs must be positive")
    for epoch in range(start_epoch, start_epoch + epoch_count):
        model.train()
        train_loss, samples = 0.0, 0
        for inputs, target in loaders["train"]:
            inputs = [value.to(device) for value in inputs]
            target = {name: value.to(device) for name, value in target.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*inputs)
            loss, _ = estimator_loss(prediction, target, model.torque_scale_nm, model.grf_scale_n, loss_config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Estimator loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(inputs[0])
            samples += len(inputs[0])
        validation = evaluate(model, loaders["validation"], device, loss_config=loss_config)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / samples, "validation": validation})
        print(f"Epoch {epoch + 1}/{start_epoch + epoch_count}: train={train_loss / samples:.4f}, "
              f"val={validation['loss']:.4f}, hip MAE={validation['hip_mae_nm']} N m", flush=True)
        score = selection_score(validation, selection_metric)
        improved = score < best_score
        best_score = min(best_score, score)
        best_validation = min(best_validation, validation["loss"])
        checkpoint = {"schema_version": 1, "model_state_dict": model.state_dict(), "model_args": model_args,
                      "optimizer_state_dict": optimizer.state_dict(),
                      "torch_rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state(),
                      "numpy_rng_state": np.random.get_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                      "config": config, "target": options["target"], "imu_names": list(IMU_NAMES),
                      "joint_names": list(JOINT_NAMES), "extra_names": list(selected_features),
                      "epoch": epoch + 1, "validation": validation, "history": history,
                      "partitions": partitions, "best_validation_loss": best_validation,
                      "selection_metric": selection_metric, "best_selection_score": best_score,
                      "collection_policy_sha256": manifest["policy_sha256"],
                      "processing": manifest.get("processing", {}),
                      "gravity_source": manifest["gravity_source"], "assistance": manifest["assistance"]}
        temporary = output / "model_last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "model_last.pt")
        if improved:
            temporary = output / "model_best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "model_best.pt")
        temporary = output / "history.json.tmp"
        temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
        temporary.replace(output / "history.json")
    checkpoint = torch.load(output / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if options.get("calibrate_uncertainty", False):
        _, calibration = evaluate(model, loaders["validation"], device, return_arrays=True, loss_config=loss_config)
        scale = np.quantile(np.abs(calibration["error"]) / calibration["sigma"], .95, axis=0) / 1.96
        model.uncertainty_scale.copy_(torch.as_tensor(scale, device=device))
        checkpoint["uncertainty_scale"] = scale.tolist()
        checkpoint["uncertainty_calibration"] = {"partition": "validation", "nominal_coverage": .95}
        temporary = output / "model_best.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "model_best.pt")
    if not evaluate_test:
        print("Test evaluation deferred; model selected and uncertainty calibrated using validation only.", flush=True)
        return model, None
    test = evaluate(model, loaders["test"], device, loss_config=loss_config)
    if options.get("group_by"):
        lookup = {entry["file"]: entry for entry in entries}
        test["by_motion"] = {}
        for motion in sorted({lookup[Path(path).name]["motion_id"] for path in partitions["test"]}):
            paths = [path for path in partitions["test"] if lookup[Path(path).name]["motion_id"] == motion]
            dataset = EpisodeWindows(paths, model.window_steps, options["stride_samples"], options["target"], extra_indices)
            test["by_motion"][motion] = evaluate(model, DataLoader(dataset, batch_size=options["batch_size"]), device, loss_config=loss_config)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "metrics_test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print("Held-out test:", json.dumps(test), flush=True)
    return model, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/force_estimation.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", help="Checkpoint to resume. --epochs specifies additional epochs.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Start a fresh optimizer for a fine-tuning experiment.")
    parser.add_argument("--defer-test", action="store_true", help="Use training and validation only; leave test evaluation to a later experiment.")
    parser.add_argument("--learning-rate", type=float, help="Override the configured Adam learning rate.")
    arguments = parser.parse_args()
    with open(arguments.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if arguments.learning_rate is not None:
        if not np.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
            parser.error("--learning-rate must be positive and finite")
        config["training"]["learning_rate"] = arguments.learning_rate
    train(config, arguments.data_dir, arguments.output_dir, arguments.epochs, arguments.resume,
          arguments.reset_optimizer, not arguments.defer_test)


if __name__ == "__main__":
    main()
